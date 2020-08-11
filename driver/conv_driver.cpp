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
#include "naive_conv.h"
#include <chrono>
#include <functional>
#include <hip/hip_runtime.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <vector>
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
        hipDeviceSynchronize();
        hipEventRecord(evt_0, stream);
    }
    void stop() {
        hipEventRecord(evt_1, NULL);
        hipEventSynchronize(evt_1);
        hipDeviceSynchronize();
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

#include "igemm_v4r1_dynamic_driver.h"

#ifndef ABS
#define ABS(x) ((x) > 0 ? (x) : -1 * (x))
#endif

#ifndef IGEMM_HSACO
#define IGEMM_HSACO "igemm_v4r1_dynamic.hsaco"
#endif

#ifndef REDUCTION_HSACO
#define REDUCTION_HSACO "wrw_reduction_hip.hip.cc.o.hsaco"
#endif

#ifndef IGEMM_CONFIG_FILE
#define IGEMM_CONFIG_FILE "igemm_v4r1_dynamic.config"
#endif

#define WARMUP 3
#define REPEAT 5
#define SCLK_MHZ 1800

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
void gen_rand_vector(T *vec, size_t vec_size, T fmin, T fmax) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads < 4)
        num_threads = 4;
    // printf("total threads:%d\n",num_threads);
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.push_back(std::thread(
            // thread function
            [](float *p, int tid, int block_size, int total_size, float fmin,
               float fmax) {
                std::mt19937 rng(
                    std::chrono::system_clock::now()
                        .time_since_epoch()
                        .count() +
                    std::hash<std::thread::id>()(std::this_thread::get_id()));
                std::uniform_real_distribution<float> distribution(fmin, fmax);
                for (int i = tid; i < total_size; i += block_size) {
                    p[i] = floor(distribution(rng) * 10);
                    //p[i] = distribution(rng);
                }
            },
            vec, t, num_threads, vec_size, fmin, fmax));
    }
    for (auto &th : threads)
        th.join();
}

#define PER_PIXEL_CHECK
#define PER_PIXEL_CHECK_PRINT 1

static inline bool valid_vector(const float *ref, const float *pred, int n,
                                double nrms = 1e-6) {
    double s0 = 0.0;
    double s1 = 0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    for (int i = 0; i < n; ++i) {
        double ri = (double)ref[i];
        double pi = (double)pred[i];
        double d = ri - pi;
        double dd = d * d;
        double rr = 2.0 * ri * ri;
        s0 += dd;
        s1 += rr;
#ifdef PER_PIXEL_CHECK
        double delta = ABS(ABS(ri - pi) / ri);
        if (delta > 3e-5) 
        //if (i > 255)
        {
#if PER_PIXEL_CHECK_PRINT
            if (pp_err < 128)
                printf("diff at %4d, ref:%lf, pred:%lf(0x%08x), d:%lf\n", i, ri,
                       pi, ((uint32_t *)pred)[i], delta);
#endif
            pp_err++;
        }
#endif
    }
    printf("nrms:%lf, s0:%lf, s1:%lf\n",sqrt(s0/s1),s0,s1);
    return (sqrt(s0 / s1) < nrms)
#ifdef PER_PIXEL_CHECK
           && (pp_err == 0)
#endif
        ;
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
    char *hsaco = env_get_str("IGEMM_HSACO", (char*)IGEMM_HSACO);
    char *hsaco_reduction = env_get_str("REDUCTION_HSACO", (char*)REDUCTION_HSACO);
    char *config_file = env_get_str("IGEMM_CONFIG_FILE", (char*)IGEMM_CONFIG_FILE);
    int warmup = env_get_int("IGEMM_WARMUP", WARMUP);
    int repeat = env_get_int("IGEMM_REPEAT", REPEAT);
    int sclk_mhz = env_get_int("IGEMM_SCLK_MHZ", SCLK_MHZ);
    config_parser_t config_parser(config_file);
    auto content = config_parser.parse();
    auto tunables = igemm_v4r1_dynamic_tunable_from_config(content);

    hipModule_t module;
    HIP_CALL(hipModuleLoad(&module, hsaco));

    hipModule_t module_reduction;
    HIP_CALL(hipModuleLoad(&module_reduction, hsaco_reduction));

    args_t conv_args = create_conv_args(argc, argv);
    dump_arg(&conv_args);
    int hi = conv_args.get_int("in_h");
    int wi = conv_args.get_int("in_w");
    int n = conv_args.get_int("batchsize");
    int k = conv_args.get_int("out_channels");
    int c = conv_args.get_int("in_channels");
    int conv_dir = conv_args.get_int("forw");

    printf("conv_dir is %d\r\n", conv_dir);

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

    
    // init host side
    float *host_input = (float *)malloc(n * c * hi * wi * sizeof(float));
    float *host_weight = (float *)malloc(k * c * y * x * sizeof(float));
    float *host_output = (float *)malloc(n * k * ho * wo * sizeof(float));

    int need_verify = conv_args.get_int("verify");

    int num_cu;
    int num_simd = 64; // hard coded
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        num_cu = dev_prop.multiProcessorCount;
    }
    double fp32_gflops =
        theoritical_fp32_gflops(((double)sclk_mhz) / 1000.0, num_cu, num_simd);

    if (need_verify) {
        // forward 1, 0
        if ((1 == conv_dir) || (0 == conv_dir))
        {
            // gen rand
            gen_rand_vector<float>(host_input, n * c * hi * wi, 0.0, 1.0);
            gen_rand_vector<float>(host_weight, k * c * y * x, -0.5, 0.5);
            // gen_rand_vector<float>(host_input, n*c*hi*wi, 1.0, 1.0);
            // gen_rand_vector<float>(host_weight, k*c*y*x, 0.5, 0.5);

            // TODO: other direction
            // TODO: This is slow. try mkl_conv.h
            naive_conv_fwd_nchw(host_input, host_weight, host_output, n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_h, dilation_w);
        }
        // data gradient(backward data) 2, 0
        if ((2 == conv_dir) || (0 == conv_dir))
        {
            // gen rand
            gen_rand_vector<float>(host_output, n * c * ho * wo, 0.0, 1.0);
            gen_rand_vector<float>(host_weight, k * c * y * x, -0.5, 0.5);
            // gen_rand_vector<float>(host_input, n*c*hi*wi, 1.0, 1.0);
            // gen_rand_vector<float>(host_weight, k*c*y*x, 0.5, 0.5);

            // TODO: other direction
            // TODO: This is slow. try mkl_conv.h
            naive_conv_bwd_d_nchw(host_input, host_weight, host_output, n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_h, dilation_w);
        }
        // weight gradient(backward weights) 4, 0
        if ((4 == conv_dir) || (0 == conv_dir))
        {
            // gen rand
            gen_rand_vector<float>(host_input, n * c * hi * wi, 0.0, 1.0);
            gen_rand_vector<float>(host_output, n * k * ho * wo, -0.5, 0.5);

            // TODO: other direction
            // TODO: This is slow. try mkl_conv.h
            naive_conv_bwd_f_nchw(host_input, host_weight, host_output, n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_h, dilation_w);
        }
    }

    float *device_input;
    float *device_weight;
    float *device_output;
    float *device_output_to_host =
        (float *)malloc(n * k * ho * wo * sizeof(float));
    float *device_weight_to_host =
        (float *)malloc(k * c * y * x * sizeof(float));

    HIP_CALL(hipSetDevice(0));
    HIP_CALL(hipMalloc(&device_input, n * c * hi * wi * sizeof(float)));
    HIP_CALL(hipMalloc(&device_weight, k * c * y * x * sizeof(float)));
    HIP_CALL(hipMalloc(&device_output, n * k * ho * wo * sizeof(float)));

    if ((1 == conv_dir) || (0 == conv_dir))
    {
        HIP_CALL(hipMemcpy(device_input, host_input,
                           n * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(device_weight, host_weight,
                           k * c * y * x * sizeof(float), hipMemcpyHostToDevice));

        igemm_v4r1_dynamic_driver_t conv_driver;
        for (size_t i = 0; i < tunables.size(); i++) {
            bool kernel_1x1 = false;
            if ((y == 1) && (x == 1))
            {
                kernel_1x1 = true;
            }
        
            igemm_v4r1_dynamic_tunable_t *tunable = &tunables[i];
            // if(std::string("igemm_v4r1_dynamic_64x64x8_8x8_4x4x2x4x2x4_8x1x8x1_4x16")
            // != conv_driver.get_kernel_name(tunable))
            //    continue;
            if (tunable->OPT_1x1){
                if (!kernel_1x1)
                {
                    continue;
                }
            }
            printf("  %s, ", conv_driver.get_kernel_name(tunable).c_str());
            if (need_verify)
                HIP_CALL(hipMemset(device_output, 0,
                                   n * k * ho * wo * sizeof(float)));
            result_t result =
                conv_driver.run(&conv_args, tunable, module, module_reduction, device_input,
                                device_weight, device_output, warmup, repeat, CONV_FWD);
            if (result.return_code != 0)
                continue;
            double gflops = measured_fp32_conv_gflops(
                result.duration_ms, n, c, hi, wi, k, y, x, stride_h, stride_w,
                dilation_h, dilation_w, pad_h, pad_w);
            printf("cost:%.3fms, gflops:%.1f(%.2f%%)", result.duration_ms,
                   gflops, (gflops / fp32_gflops) * 100);
            if (need_verify) {
                HIP_CALL(hipMemcpy(device_output_to_host, device_output,
                                   n * k * ho * wo * sizeof(float),
                                   hipMemcpyDeviceToHost));
                bool is_valid = valid_vector(host_output, device_output_to_host,
                                             n * k * ho * wo);
                printf(", valid:%s", is_valid ? "y" : "n");
                if (!is_valid) {
                    printf("\n");
                    break;
                }
            }
            printf("\n");
        }
    }

    if ((4 == conv_dir) || (0 == conv_dir))
    {
        HIP_CALL(hipMemcpy(device_input, host_input,
                           n * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(device_output, host_output,
                           n * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));

        igemm_v4r1_dynamic_driver_t conv_driver;
        for (size_t i = 0; i < tunables.size(); i++) {
        
            igemm_v4r1_dynamic_tunable_t *tunable = &tunables[i];
            printf("  %s, ", conv_driver.get_kernel_name(tunable).c_str());
            if (need_verify)
                HIP_CALL(hipMemset(device_weight, 0,
                                   k * c * y * x * sizeof(float)));
            result_t result =
                conv_driver.run(&conv_args, tunable, module, module_reduction, device_input,
                                device_weight, device_output, warmup, repeat, CONV_WRW);
            if (result.return_code != 0)
                continue;
            double gflops = measured_fp32_conv_gflops(
                result.duration_ms, n, c, hi, wi, k, y, x, stride_h, stride_w,
                dilation_h, dilation_w, pad_h, pad_w);
            printf("cost:%.3fms, gflops:%.1f(%.2f%%)", result.duration_ms,
                   gflops, (gflops / fp32_gflops) * 100);
            if (need_verify) {
                HIP_CALL(hipMemcpy(device_weight_to_host, device_weight,
                                   k * c * y * x * sizeof(float),
                                   hipMemcpyDeviceToHost));

                //printf("var to monitor:[%f, %f, %f, %f]\r\n", device_weight_to_host[0], device_weight_to_host[1], device_weight_to_host[2], device_weight_to_host[3]);
                //printf("var to monitor:[%d, %d, %d, %d]\r\n", ((int *)device_weight_to_host)[0], ((int *)device_weight_to_host)[1], 
                //                                ((int *)device_weight_to_host)[2], ((int *)device_weight_to_host)[3]);
                printf("\r\n");
                for (int i_check = 0; i_check < (0+8); i_check++)
                {
                    //printf("[%d]th var to monitor:[%f, %d]\r\n", i_check, device_weight_to_host[i_check], ((int *)device_weight_to_host)[i_check]);
                }

                bool is_valid = valid_vector(host_weight, device_weight_to_host,
                                             k * c * y * x);
                printf(", valid:%s", is_valid ? "y" : "n");
                if (!is_valid) {
                    printf("\n");
                    //break;
                }
            }
            printf("\n");
        }
    }

    free(host_input);
    free(host_weight);
    free(host_output);
    free(device_output_to_host);

    hipFree(device_input);
    hipFree(device_weight);
    hipFree(device_output);
}