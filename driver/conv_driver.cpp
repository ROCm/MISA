/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "args.h"
#include "config_parser.h"
#include <chrono>
#include <functional>
#include <hip/hip_runtime.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <vector>

#ifdef USE_XDNN
#include "xdnn_conv.h"
#define conv_fwd_nchw xdnn_conv_fwd_nchw
#define conv_bwd_d_nchw xdnn_conv_bwd_d_nchw
#define conv_bwd_f_nchw xdnn_conv_bwd_f_nchw
#else
#define NAIVE_CONV_THREADED
#include "naive_conv.h"
#define conv_fwd_nchw naive_conv_fwd_nchw
#define conv_bwd_d_nchw naive_conv_bwd_d_nchw
#define conv_bwd_f_nchw naive_conv_bwd_f_nchw
#endif

static inline size_t conv_out_size(size_t in_size, size_t pad, size_t dilation,
                                   size_t ksize, size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}
class gpu_timer_t {
  public:
    gpu_timer_t(hipStream_t stream_) {
        stream = stream_;
        hipEventCreate(&evt_0);
        hipEventCreate(&evt_1);
    }
    ~gpu_timer_t() {
        hipEventDestroy(evt_0);
        hipEventDestroy(evt_1);
    }
    void start() {
        // hipDeviceSynchronize();
        hipEventRecord(evt_0, stream);
    }
    void stop() {
        hipEventRecord(evt_1, stream);
        hipEventSynchronize(evt_1);
        // hipDeviceSynchronize();
    }
    float duration() {
        float ms;
        hipEventElapsedTime(&ms, evt_0, evt_1);
        return ms;
    }

  private:
    hipEvent_t evt_0, evt_1;
    hipStream_t stream;
};
static int next_pow2(int n) {
    if (n == 0)
        return 1;
    if ((n & (n - 1)) == 0)
        return n;
    while ((n & (n - 1)) > 0)
        n &= (n - 1);
    return n << 1;
}
typedef struct {
    int return_code;
    float duration_ms;
    std::string kernel_name;
} result_t;

#define HIP_CALL(call)                                                         \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            printf("[hiperror](%d) fail to call %s,(%s)", (int)err, #call,     \
                   hipGetErrorString(err));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static inline double theoritical_fp32_gflops(double sclk_ghz, size_t cu,
                                             size_t simd) {
    return 2 * sclk_ghz * cu * simd;
}
static inline double
theoritical_fp32_conv_flop(size_t n, size_t c, size_t hi, size_t wi, size_t k,
                           size_t y, size_t x, size_t stride_h, size_t stride_w,
                           size_t dilation_h, size_t dilation_w, size_t pad_h,
                           size_t pad_w) {
    size_t ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    size_t wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

    double flop = (double)n * c * ho * wo * k * y * x * 2;
    return flop;
}
static inline double
measured_fp32_conv_gflops(double time_ms, size_t n, size_t c, size_t hi,
                          size_t wi, size_t k, size_t y, size_t x,
                          size_t stride_h, size_t stride_w, size_t dilation_h,
                          size_t dilation_w, size_t pad_h, size_t pad_w) {
    double flop =
        theoritical_fp32_conv_flop(n, c, hi, wi, k, y, x, stride_h, stride_w,
                                   dilation_h, dilation_w, pad_h, pad_w);
    return flop / (time_ms * 1e6);
}

#include "igemm_gtc_base.h"
#include "igemm_bwd_gtc_driver.h"

#ifndef ABS
#define ABS(x) ((x) > 0 ? (x) : -1 * (x))
#endif

#ifndef IGEMM_HSACO
#define IGEMM_HSACO "igemm_gtc.hsaco"
#endif

#ifndef IGEMM_CONFIG_FILE
#define IGEMM_CONFIG_FILE "igemm_gtc.config"
#endif

#define WARMUP 3
#define REPEAT 8
#define SCLK_MHZ 1283

static inline int env_get_int(const char *var_name, int default_int) {
    char *v = getenv(var_name);
    int r = default_int;
    if (v)
        r = atoi(v);
    return r;
}

static inline char *env_get_str(const char *var_name, char *default_str) {
    char *v = getenv(var_name);
    if (v)
        return v;
    return default_str;
}

template <typename T> T gen_rand(T fmin, T fmax) {
    static int init = 0;
    if (!init) {
        srand(time(NULL));
        init = 1;
    }
    double d = static_cast<double>(rand() / (static_cast<double>(RAND_MAX)));
    return (static_cast<T>(d) * (fmax - fmin)) + fmin;
}

template <typename T>
struct distribution_t{
};

template <>
struct distribution_t<int>{
    distribution_t(int min, int max) : distribution(min, max) {}
    template<class URNG>
    int operator()(URNG & rng){ return distribution(rng);}
    std::uniform_int_distribution<int> distribution;
};
template <>
struct distribution_t<float>{
    distribution_t(float min, float max) : distribution(min, max) {}
    template<class URNG>
    float operator()(URNG & rng){ return distribution(rng);}
    std::uniform_real_distribution<float> distribution;
};

template <typename Dst_T, typename Src_T>
void block_wise_rand_generator(Dst_T *p, int tid, int block_size, int total_size, Src_T min, Src_T max, Src_T scale)
{
    std::mt19937 rng(std::chrono::system_clock::now()
                        .time_since_epoch()
                        .count() +
                    std::hash<std::thread::id>()(std::this_thread::get_id()));
    distribution_t<Src_T> distribution(min,max);
    for (int i = tid; i < total_size; i += block_size) {
        p[i] = static_cast<Dst_T>(scale * distribution(rng));
    }
}

template <typename Dst_T, typename Src_T>
void gen_rand_vector(Dst_T *vec, size_t vec_size, Src_T fmin, Src_T fmax, Src_T scale = 1) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads < 4)
        num_threads = 4;
    // printf("total threads:%d\n",num_threads);
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.push_back(std::thread(block_wise_rand_generator<Dst_T, Src_T>,
            vec, t, num_threads, vec_size, fmin, fmax, scale));
    }
    for (auto &th : threads)
        th.join();
}

static inline bool valid_vector(const float *ref, const float *pred, int n,
                                double nrms = 1e-6) {
    double s0 = 0.0;
    double s1 = 0.0;
    int igemm_per_pixel_check = env_get_int("PER_PIXEL_CHECK", 0);
    int igemm_per_pixel_check_print = env_get_int("PER_PIXEL_CHECK_PRINT", 1);
    int pp_err = 0;

    for (int i = 0; i < n; ++i) {
        double ri = (double)ref[i];
        double pi = (double)pred[i];
        double d = ri - pi;
        double dd = d * d;
        double rr = 2.0 * ri * ri;
        s0 += dd;
        s1 += rr;
        if(igemm_per_pixel_check){
            double delta = ABS((ri - pi) / ri);
            printf("[%d] ref:%lf, pred:%lf(0x%08x) [%s]\n", i, ri, pi, ((uint32_t *)pred)[i], delta > 3e-5? "N":"Y");
            if (delta > 3e-5) {
                if(igemm_per_pixel_check_print){
                    if (pp_err < 100)
                        printf("diff at %4d, ref:%lf, pred:%lf(0x%08x), d:%lf\n", i, ri,
                            pi, ((uint32_t *)pred)[i], delta);
                }
                pp_err++;
            }

        }
    }
    // printf("nrms:%lf, s0:%lf, s1:%lf\n",sqrt(s0/s1),s0,s1);
    return (sqrt(s0 / s1) < nrms)
#ifdef PER_PIXEL_CHECK
           && (pp_err == 0)
#endif
        ;
}

static inline double get_fwd_nrms()
{
    return 1e-6;
}
static inline double get_bwd_nrms()
{
#ifdef USE_XDNN
    return 5e-5;
#else
    return 1e-6;
#endif
}
static inline double get_wrw_nrms()
{
#ifdef USE_XDNN
    return 1e-4;
#else
    return 1e-6;
#endif
}

void dump_arg(const args_t *arg) {
    int hi = arg->get_int("in_h");
    int wi = arg->get_int("in_w");
    int n = arg->get_int("batchsize");
    int k = arg->get_int("out_channels");
    int c = arg->get_int("in_channels");

    int stride_h = arg->get_int("conv_stride_h");
    int stride_w = arg->get_int("conv_stride_w");
    int dilation_h = arg->get_int("dilation_h");
    int dilation_w = arg->get_int("dilation_w");
    int pad_h = arg->get_int("pad_h");
    int pad_w = arg->get_int("pad_w");
    int y = arg->get_int("fil_h");
    int x = arg->get_int("fil_w");
    int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

    printf("conv_driver.exe conv: n:%d, c:%d, h:%d, w:%d, k:%d, y:%d, x:%d, sy:%d, sx:%d, dy:%d, "
           "dx:%d, py:%d, px:%d, ho:%d, wo:%d\n",
           n, c, hi, wi, k, y, x, stride_h, stride_w, dilation_h, dilation_w,
           pad_h, pad_w, ho, wo);
}

int main(int argc, char **argv) {
    
    char *hsaco = env_get_str("IGEMM_HSACO", IGEMM_HSACO);
    char *config_file = env_get_str("IGEMM_CONFIG_FILE", IGEMM_CONFIG_FILE);
    int warmup = env_get_int("IGEMM_WARMUP", WARMUP);
    int repeat = env_get_int("IGEMM_REPEAT", REPEAT);
    int sclk_mhz = env_get_int("IGEMM_SCLK_MHZ", SCLK_MHZ);
    int skip_cpu_conv = env_get_int("IGEMM_SKIP_CPU_CONV", 0);
    config_parser_t config_parser(config_file);
    auto content = config_parser.parse();
    //content.dump();

    auto tunables = igemm_gtc_tunable_from_config(content);
    if(tunables.size() == 0){
        printf("no tunable specified, may not work\n");
        return 0;
    }
    // printf("tunables:%d\n", tunables.size());

    hipModule_t module;
    HIP_CALL(hipModuleLoad(&module, hsaco));

    args_t conv_args = create_conv_args(argc, argv);
    // dump_arg(&conv_args);

    int hi = conv_args.get_int("in_h");
    int wi = conv_args.get_int("in_w");
    int n = conv_args.get_int("batchsize");
    int k = conv_args.get_int("out_channels");
    int c = conv_args.get_int("in_channels");

    int stride_h = conv_args.get_int("conv_stride_h");
    int stride_w = conv_args.get_int("conv_stride_w");
    int dilation_h = conv_args.get_int("dilation_h");
    int dilation_w = conv_args.get_int("dilation_w");
    int pad_h = conv_args.get_int("pad_h");
    int pad_w = conv_args.get_int("pad_w");
    int y = conv_args.get_int("fil_h");
    int x = conv_args.get_int("fil_w");
    int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);
    int forw = conv_args.get_int("forw");

    int need_fwd = (forw == 0 ? 1 : (forw & 1 ? 1 : 0));
    int need_bwd = (forw == 0 ? 1 : (forw & 2 ? 1 : 0));
    int need_wrw = (forw == 0 ? 1 : (forw & 4 ? 1 : 0));

    // init host side
    float *host_input = (float *)malloc(n * c * hi * wi * sizeof(float));
    float *host_weight = (float *)malloc(k * c * y * x * sizeof(float));
    float *host_output = (float *)malloc(n * k * ho * wo * sizeof(float));

    float *device_input;
    float *device_weight;
    float *device_output;

    HIP_CALL(hipMalloc(&device_input, n * c * hi * wi * sizeof(float)));
    HIP_CALL(hipMalloc(&device_weight, k * c * y * x * sizeof(float)));
    HIP_CALL(hipMalloc(&device_output, n * k * ho * wo * sizeof(float)));

    int need_verify = conv_args.get_int("verify");

    // printf("fwd:%d, bwd:%d, wrw:%d, verify:%d\n",need_fwd, need_bwd, need_wrw, need_verify);

    int num_cu;
    int num_simd = 64; // hard coded
    int gcn_arch = 0;
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        num_cu = dev_prop.multiProcessorCount;
        gcn_arch = dev_prop.gcnArch;
#if 0
#define P_DEVICE_PROP_INT(prop) \
        printf(#prop":%d\n", dev_prop.prop)


        P_DEVICE_PROP_INT(clockRate);
        P_DEVICE_PROP_INT(memoryClockRate);
        P_DEVICE_PROP_INT(memoryBusWidth);
        P_DEVICE_PROP_INT(major);
        P_DEVICE_PROP_INT(minor);
        P_DEVICE_PROP_INT(gcnArch);
#endif
    }
    if(gcn_arch == 908){
        num_simd = 4 * 32 ; // 4x miSIMD, 32x mac unit
    }
    double fp32_gflops =
        theoritical_fp32_gflops(((double)sclk_mhz) / 1000.0, num_cu, num_simd);

    if (need_fwd){
        float *device_output_to_host = NULL;
        if (need_verify) {
            // gen rand
            gen_rand_vector<float, float>(host_input, n * c * hi * wi, 0.0, 1.0);
            gen_rand_vector<float, float>(host_weight, k * c * y * x, -0.5, 0.5);

            if(!skip_cpu_conv)
                conv_fwd_nchw(host_input, host_weight, host_output, n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h);
            device_output_to_host = (float *)malloc(n * k * ho * wo * sizeof(float));
        }

        HIP_CALL(hipMemcpy(device_input, host_input,
                       n * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(device_weight, host_weight,
                       k * c * y * x * sizeof(float), hipMemcpyHostToDevice));

        if (need_verify)
            free(device_output_to_host);
    }
    if (need_bwd){
        float *device_input_to_host = NULL;
        if (need_verify) {
            // gen rand
            gen_rand_vector<float, float>(host_output, n * k * ho * wo, 0.0, 1.0);
            // for(int i_n = 0; i_n < n; i_n++){
            //     for(int i_k = 0; i_k < k; i_k++){
            //         for(int i_ho = 0; i_ho < ho; i_ho++){
            //             for(int i_wo = 0; i_wo < wo; i_wo++){
            //                 int data = ((i_n  & 0xff) << 24) |
            //                             ((i_k  & 0xff) << 16) |
            //                             ((i_ho & 0xff) << 8) |
            //                             (i_wo & 0xff);
            //                 int index = i_n * k * ho * wo + i_k * ho * wo + i_ho * wo + i_wo;
            //                 memcpy(&host_output[index],&data,4 );
            //                 // host_output[index] = *(float*)(&data);
            //             }
            //         }
            //     }
            // }


            gen_rand_vector<float, float>(host_weight, k * c * y * x, -0.5, 0.5);
            // gen_rand_vector<float, int>(host_output, n * k * ho * wo,1, 1);
            // gen_rand_vector<float, int>(host_weight, k * c * y * x, 1, 1);

            if(!skip_cpu_conv)
                conv_bwd_d_nchw(host_input, host_weight, host_output, n,
                                         wi, hi, c, k, x, y, pad_w,
                                         pad_h, stride_w, stride_h, dilation_w, dilation_h);
            device_input_to_host = (float *)malloc(n * c * hi * wi * sizeof(float));
            // printf("len:%d\n", n * c * hi * wi * sizeof(float) );
        }

        HIP_CALL(hipMemcpy(device_output, host_output,
                       n * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(device_weight, host_weight,
                       k * c * y * x * sizeof(float), hipMemcpyHostToDevice));
        

        igemm_bwd_gtc_t conv_bwd_driver;
        double nrms = get_bwd_nrms();
        for (int i = 0; i < tunables.size(); i++) {
            igemm_gtc_tunable_t *tunable = &tunables[i];

            printf("[%2d] %s, ", i, conv_bwd_driver.get_kernel_name(tunable).c_str());

            if (need_verify)
                HIP_CALL(hipMemset(device_input, 0,
                                   n * c * hi * wi * sizeof(float)));
            result_t result =
                conv_bwd_driver.run(&conv_args, tunable, module, device_input,
                                device_weight, device_output, warmup, repeat);
            if (result.return_code != 0){
                printf("not applicatble\n");
                continue;
            }

            double gflops = measured_fp32_conv_gflops(
                result.duration_ms, n, c, hi, wi, k, y, x, stride_h, stride_w,
                dilation_h, dilation_w, pad_h, pad_w);
            printf("cost:%.3fms, tflops:%.3f(%.2f%%)", result.duration_ms,
                   gflops / 1000 , (gflops / fp32_gflops) * 100);
            if (need_verify) {
                HIP_CALL(hipMemcpy(device_input_to_host, device_input,
                                   n * c * hi * wi * sizeof(float),
                                   hipMemcpyDeviceToHost));
                bool is_valid = valid_vector(host_input, device_input_to_host,
                                            n * c * hi * wi, nrms);
                printf(", valid:%s", is_valid ? "y" : "n");
                // if (!is_valid) {
                //     printf("\n");
                //     break;
                // }
            }
            printf("\n");
        }
        if (need_verify) 
            free(device_input_to_host);
    }
    if (need_wrw){
        // un implemented
    }

    free(host_input);
    free(host_weight);
    free(host_output);

    hipFree(device_input);
    hipFree(device_weight);
    hipFree(device_output);
}