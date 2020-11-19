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
#include <chrono>
#include <functional>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <vector>
#include <float.h>

typedef struct {
    int return_code;
    float duration_ms;
    float gflops;
    float efficiency;
    std::string kernel_name;
} result_t;


#ifndef USE_EXT_MODULE_LAUNCH
#define USE_EXT_MODULE_LAUNCH 1
#endif

#ifdef USE_XDNN
#include "xdnn_conv.h"
#define conv_fwd_nchw xdnn_conv_fwd_nchw
#define conv_bwd_nchw xdnn_conv_bwd_nchw
#define conv_wrw_nchw xdnn_conv_wrw_nchw
#else
#define NAIVE_CONV_THREADED
#include "naive_conv.h"
#define conv_fwd_nchw naive_conv_fwd_nchw
#define conv_bwd_nchw naive_conv_bwd_nchw
#define conv_wrw_nchw naive_conv_wrw_nchw
#endif

static inline size_t conv_out_size(size_t in_size, size_t pad, size_t dilation,
                                   size_t ksize, size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

#define HIP_CALL(call)                                                         \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            printf("[hiperror](%d) fail to call %s,(%s)", (int)err, #call,     \
                   hipGetErrorString(err));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)



#include "reference_naive_conv_driver.h"

static inline double theoritical_fp32_gflops(double sclk_ghz, size_t cu,
                                             size_t simd) {
    return 2 * sclk_ghz * cu * simd;
}
static inline double
theoritical_fp32_conv_flop(size_t n, size_t c, size_t hi, size_t wi, size_t k,
                           size_t y, size_t x, size_t stride_h, size_t stride_w,
                           size_t dilation_h, size_t dilation_w, size_t pad_h,
                           size_t pad_w, size_t ngroups) {
    size_t ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    size_t wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

    double flop = (double)n * c * ho * wo * k * y * x * 2 / ngroups;
    return flop;
}
static inline double
measured_fp32_conv_gflops(double time_ms, size_t n, size_t c, size_t hi,
                          size_t wi, size_t k, size_t y, size_t x,
                          size_t stride_h, size_t stride_w, size_t dilation_h,
                          size_t dilation_w, size_t pad_h, size_t pad_w, size_t ngroups) {
    double flop =
        theoritical_fp32_conv_flop(n, c, hi, wi, k, y, x, stride_h, stride_w,
                                   dilation_h, dilation_w, pad_h, pad_w, ngroups);
    return flop / (time_ms * 1e6);
}

#ifndef ABS
#define ABS(x) ((x) > 0 ? (x) : -1 * (x))
#endif

#ifndef IGEMM_HSACO
#define IGEMM_HSACO "naive_conv.hsaco"
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
            double delta = ABS(ABS(ri - pi) / ri);
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
#ifdef USE_XDNN
    return 5e-5;
#else
    return 1e-6;
#endif
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

    printf("n:%d, c:%d, h:%d, w:%d, k:%d, y:%d, x:%d, sy:%d, sx:%d, dy:%d, "
           "dx:%d, py:%d, px:%d, ho:%d, wo:%d\n",
           n, c, hi, wi, k, y, x, stride_h, stride_w, dilation_h, dilation_w,
           pad_h, pad_w, ho, wo);
}

int main(int argc, char **argv) {
    char *hsaco = env_get_str("IGEMM_HSACO", IGEMM_HSACO);
    int warmup = env_get_int("IGEMM_WARMUP", WARMUP);
    int repeat = env_get_int("IGEMM_REPEAT", REPEAT);
    int sclk_mhz = env_get_int("IGEMM_SCLK_MHZ", SCLK_MHZ);
    int skip_cpu_conv = env_get_int("IGEMM_SKIP_CPU_CONV", 0);
    int log_fastest_config = env_get_int("IGEMM_LOG_FASTEST_CONFIG", 0);


    hipModule_t module;
    HIP_CALL(hipModuleLoad(&module, hsaco));

    args_t conv_args = create_conv_args(argc, argv);
    // dump_arg(&conv_args);

    std::string precision;
    std::string base(argv[1]);
    if(base == "conv")
        precision = "fp32";
    else if(base == "convfp16")
        precision = "fp16";
    else if(base == "convbfp16")
        precision = "bf16";
    else
        assert(0);

    std::string tensor_layout = "nchw";     // TODO: other layout
    if(conv_args.get_int("spatial_dim") == 3)
        tensor_layout = "ncdhw";

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
    int ngroups = conv_args.get_int("group_count");
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
        reference_naive_conv_ctrl_t ctrl{"fwd", tensor_layout, precision};
        result_t fastest_result_fwd;
        fastest_result_fwd.duration_ms = FLT_MAX;
        int fastest_id = -1;
        float *device_output_to_host = NULL;
        if (need_verify) {
            // gen rand
            gen_rand_vector<float, float>(host_input, n * c * hi * wi, 0.0, 1.0);
            gen_rand_vector<float, float>(host_weight, k * c * y * x, -0.5, 0.5);

            //gen_rand_vector<float, int>(host_input, n * c * hi * wi, 1, 1);
            //gen_rand_vector<float, int>(host_weight, k * c * y * x, 1, 1);

            if(!skip_cpu_conv)
                conv_fwd_nchw(host_input, host_weight, host_output, n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            device_output_to_host = (float *)malloc(n * k * ho * wo * sizeof(float));
        }

        HIP_CALL(hipMemcpy(device_input, host_input,
                       n * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(device_weight, host_weight,
                       k * c * y * x * sizeof(float), hipMemcpyHostToDevice));

        reference_naive_conv_t conv_fwd_driver;
        double nrms = get_fwd_nrms();
        std::string kernel_name = reference_naive_conv_get_kernel_name(&ctrl);
        printf("[fwd] %s, ", kernel_name.c_str());

        //if (need_verify)
        //    HIP_CALL(hipMemset(device_output, 0,
        //                       n * c * ho * wo * sizeof(float)));
        result_t result =
            conv_fwd_driver.run(&conv_args, &ctrl, module, device_input,
                            device_weight, device_output, warmup, repeat);

        double gflops = measured_fp32_conv_gflops(
            result.duration_ms, n, c, hi, wi, k, y, x, stride_h, stride_w,
            dilation_h, dilation_w, pad_h, pad_w, ngroups);
        printf("cost:%.3fms, tflops:%.3f(%.2f%%)", result.duration_ms,
                gflops / 1000 , (gflops / fp32_gflops) * 100);
        if (need_verify) {
            HIP_CALL(hipMemcpy(device_output_to_host, device_output,
                                n * k * ho * wo * sizeof(float),
                                hipMemcpyDeviceToHost));
            bool is_valid = valid_vector(host_output, device_output_to_host,
                                        n * k * ho * wo, nrms);
            printf(", valid:%s", is_valid ? "y" : "n");
        }
        printf("\n");

        if (need_verify)
            free(device_output_to_host);
    }

    if (need_bwd){
        reference_naive_conv_ctrl_t ctrl{"bwd", tensor_layout, precision};
        float *device_input_to_host = NULL;
        result_t fastest_result_bwd;
        fastest_result_bwd.duration_ms = FLT_MAX;
        int fastest_id = -1;
        if (need_verify) {
            // gen rand
            gen_rand_vector<float, float>(host_output, n * k * ho * wo, 0.0, 1.0);
            gen_rand_vector<float, float>(host_weight, k * c * y * x, -0.5, 0.5);
            gen_rand_vector<float, float>(host_input, n * c * hi * wi, 999999., 9999999.);  // manually input value to a very large number
            // gen_rand_vector<float, int>(host_output, n * k * ho * wo,1, 1);
            // gen_rand_vector<float, int>(host_weight, k * c * y * x, 1, 1);

            if(!skip_cpu_conv)
                conv_bwd_nchw(host_input, host_weight, host_output, n,
                                         wi, hi, c, k, x, y, pad_w,
                                         pad_h, stride_w, stride_h, dilation_w, dilation_h, ngroups);
            device_input_to_host = (float *)malloc(n * c * hi * wi * sizeof(float));
            // printf("len:%d\n", n * c * hi * wi * sizeof(float) );
        }

        HIP_CALL(hipMemcpy(device_output, host_output,
                       n * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(device_weight, host_weight,
                       k * c * y * x * sizeof(float), hipMemcpyHostToDevice));


        reference_naive_conv_t conv_bwd_driver;
        double nrms = get_bwd_nrms();

        std::string kernel_name = reference_naive_conv_get_kernel_name(&ctrl);
        printf("[bwd] %s, ", kernel_name.c_str());

        if (need_verify)
            HIP_CALL(hipMemset(device_input, 0x7f,
                                n * c * hi * wi * sizeof(float)));   // 0x7f7f7f7f ~= 7.41e+28, a very large number
        result_t result =
            conv_bwd_driver.run(&conv_args, &ctrl, module, device_input,
                            device_weight, device_output, warmup, repeat);

        double gflops = measured_fp32_conv_gflops(
            result.duration_ms, n, c, hi, wi, k, y, x, stride_h, stride_w,
            dilation_h, dilation_w, pad_h, pad_w, ngroups);
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

        if (need_verify) 
            free(device_input_to_host);
    }
    if (need_wrw){
        reference_naive_conv_ctrl_t ctrl{"wrw", tensor_layout, precision};
        float *device_weight_to_host = NULL;
        if (need_verify) {
            // gen rand
            gen_rand_vector<float, float>(host_input, n * c * hi * wi, 0.0, 1.0);
            gen_rand_vector<float, float>(host_output, n * k * ho * wo, -0.5, 0.5);
            //gen_rand_vector<float, int>(host_input, n * k * hi * wi, -5, 5);
            //gen_rand_vector<float, int>(host_output, n * k * ho * wo, 1, 1);

            conv_wrw_nchw(host_input, host_weight, host_output, n,
                                         wi, hi, c, k, x, y, pad_w,
                                         pad_h, stride_w, stride_h, dilation_w, dilation_h, ngroups);
            device_weight_to_host = (float *)malloc(k * c * y * x * sizeof(float));
            // printf("len:%d\n", k * c * y * x * sizeof(float));
        }

        HIP_CALL(hipMemcpy(device_input, host_input,
                       n * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(device_output, host_output,
                       n * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));



        reference_naive_conv_t conv_wrw_driver;
        float min_duration = 10000000.0f;
        float selected_duration = 10000000.0f;
        double nrms = get_wrw_nrms();

        std::string kernel_name = reference_naive_conv_get_kernel_name(&ctrl);
        printf("[wrw] %s, ", kernel_name.c_str());

        if (need_verify)
            HIP_CALL(hipMemset(device_weight, 0,
                                k * c * y * x * sizeof(float)));
        result_t result =
            conv_wrw_driver.run(&conv_args, &ctrl, module, device_input,
                            device_weight, device_output, warmup, repeat);


        double gflops = measured_fp32_conv_gflops(
            result.duration_ms, n, c, hi, wi, k, y, x, stride_h, stride_w,
            dilation_h, dilation_w, pad_h, pad_w, ngroups);
        printf("cost:%.3fms, tflops:%.3f(%.2f%%)", result.duration_ms,
                gflops / 1000 , (gflops / fp32_gflops) * 100);

        if (need_verify) {
            HIP_CALL(hipMemcpy(device_weight_to_host, device_weight,
                                k * c * y * x * sizeof(float),
                                hipMemcpyDeviceToHost));
            bool is_valid = valid_vector(host_weight, device_weight_to_host,
                                        k * c * y * x, nrms);
            printf(", valid:%s", is_valid ? "y" : "n");
            // if (!is_valid) {
            //     printf("\n");
            //     break;
            // }
        }
        printf("\n");

        if (need_verify) 
            free(device_weight_to_host);
    }

    free(host_input);
    free(host_weight);
    free(host_output);

    hipFree(device_input);
    hipFree(device_weight);
    hipFree(device_output);
}