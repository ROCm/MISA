/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <vector>
#include <float.h>
#include <cmath>

#ifndef USE_EXT_MODULE_LAUNCH
#define USE_EXT_MODULE_LAUNCH 1
#endif

#ifndef USE_MAGIC_DIV
#define USE_MAGIC_DIV 0
#endif

#ifndef USE_SOURCE_ACCESS_ENCODING_KERNEL_NAME
#define USE_SOURCE_ACCESS_ENCODING_KERNEL_NAME 0
#endif

#ifdef USE_GPU_NAIVE_CONV
#   include "gpu_naive_conv.h"
#   ifndef IGEMM_GPU_NAIVE_CONV_HSACO
#       define  IGEMM_GPU_NAIVE_CONV_HSACO "naive_conv.hsaco"
#   endif
#else
#   ifdef USE_XDNN
#       include "xdnn_conv.h"
#       define conv_fwd_nchw xdnn_conv_fwd_nchw
#       define conv_bwd_nchw xdnn_conv_bwd_nchw
#       define conv_wrw_nchw xdnn_conv_wrw_nchw
#   else
#       define NAIVE_CONV_THREADED
#       include "naive_conv.h"
#       define conv_fwd_nchw naive_conv_fwd_nchw
#       define conv_bwd_nchw naive_conv_bwd_nchw
#       define conv_wrw_nchw naive_conv_wrw_nchw
#   endif
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
    float gflops;
    float efficiency;
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

#include "igemm_gtc_base.h"
#include "igemm_fwd_gtc_driver.h"
#include "igemm_bwd_gtc_driver.h"
#include "igemm_wrw_gtc_driver.h"

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

template <typename Dst_T, typename Src_T>
void block_wise_tensor_copy(Dst_T *p_dst, Src_T *p_src, int tid, int block_size, int total_size)
{
    for (int i = tid; i < total_size; i += block_size) {
        p_dst[i] = static_cast<Dst_T>(p_src[i]);
    }
}

template <typename Dst_T, typename Src_T>
void tensor_copy(Dst_T *p_dst, Src_T *p_src, size_t tensor_size) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads < 4)
        num_threads = 4;
    // printf("total threads:%d\n",num_threads);
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.push_back(std::thread(block_wise_tensor_copy<Dst_T, Src_T>,
            p_dst, p_src, t, num_threads, tensor_size));
    }
    for (auto &th : threads)
        th.join();
}

static inline bool valid_float(float p)
{
    return !(std::isnan(p) || std::isinf(p));
}

template<typename T>
static inline bool valid_vector(const float *ref, const T *pred, int n,
                                double nrms = 1e-6) {
    double s0 = 0.0;
    double s1 = 0.0;
    int igemm_per_pixel_check = env_get_int("PER_PIXEL_CHECK", 0);
    int igemm_per_pixel_check_print = env_get_int("PER_PIXEL_CHECK_PRINT", 1);
    size_t pp_err = 0;

    for (size_t i = 0; i < n; ++i) {
        if(!(valid_float(ref[i]) && valid_float(pred[i]))){
            printf(" invalid float at %4d, ref:%f, pred:%f\n", i, ref[i], pred[i]);
            return false;
        }
        double ri = (double)ref[i];
        double pi = (double)pred[i];
        double d = ri - pi;
        double dd = d * d;
        double rr = 2.0 * ri * ri;
        s0 += dd;
        s1 += rr;
        if(igemm_per_pixel_check){
            double delta = ABS(ABS(ri - pi) / ri);
            printf("[%zu] ref:%lf, pred:%lf(0x%08x) [%s]\n", i, ri, pi, ((uint32_t *)pred)[i], delta > 3e-5? "N":"Y");
            if (delta > 3e-5) {
                if(igemm_per_pixel_check_print){
                    if (pp_err < 100)
                        printf("diff at %zu, ref:%lf, pred:%lf(0x%08x), d:%lf\n", i, ri,
                            pi, ((uint32_t *)pred)[i], delta);
                }
                pp_err++;
            }

        }
    }
    //printf("\nnrms:%lf, s0:%lf, s1:%lf, expected_nrms is %1f\n",sqrt(s0/s1),s0,s1,nrms);
    return (sqrt(s0 / s1) < nrms)
#ifdef PER_PIXEL_CHECK
           && (pp_err == 0)
#endif
        ;
}

static inline double get_nrms(int forw, driverDataType_t driver_data_type){
    auto basic_tolerance = [=]() -> double{
        if (driver_data_type == driverFloat){
#ifdef USE_XDNN
            return 5e-5;
#else
            return 1.5e-6;
#endif
        }
        else if (driver_data_type == driverHalf){
#ifdef USE_XDNN
            return 5*8.2e-3;
#else
            return 8.2e-3;
#endif
        }
    };
    double nrms = basic_tolerance();
    // wrw has a high tolerance
    if (forw == 4){
        nrms *= 2;
        if(driver_data_type == driverFloat){
            nrms = 0.01;
        }
        else if(driver_data_type == driverHalf){
            nrms *= 5;
        }
    }
    return nrms;
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
    char *config_file = env_get_str("IGEMM_CONFIG_FILE", IGEMM_CONFIG_FILE);
    int warmup = env_get_int("IGEMM_WARMUP", WARMUP);
    int repeat = env_get_int("IGEMM_REPEAT", REPEAT);
    int sclk_mhz = env_get_int("IGEMM_SCLK_MHZ", SCLK_MHZ);
    int log_fastest_config = env_get_int("IGEMM_LOG_FASTEST_CONFIG", 0);
    int wrw_kernel_selection = env_get_int("IGEMM_LOG_SELECTED_CONFIG", 0);
    int run_first_applicable = env_get_int("IGEMM_RUN_FIRST_APPLICABLE_CONFIG", 0); 
    int assert_when_invalid = env_get_int("IGEMM_ASSERT_WHEN_INVALID", 0);
    config_parser_t config_parser(config_file);
    auto content = config_parser.parse();
    //content.dump();

#ifdef USE_GPU_NAIVE_CONV
    char *gpu_naive_conv_hsaco = env_get_str("IGEMM_GPU_NAIVE_CONV_HSACO", IGEMM_GPU_NAIVE_CONV_HSACO);
    gpu_naive_conv_init(gpu_naive_conv_hsaco);
#endif

    auto tunables = igemm_gtc_tunable_from_config(content);
    if(tunables.size() == 0){
        printf("no tunable specified, may not work\n");
        return 0;
    }
    // printf("tunables:%d\n", tunables.size());

    hipModule_t module;
    HIP_CALL(hipModuleLoad(&module, hsaco));

    // base arg might be "conv" or "convfp16" now;
    std::string base_arg = ParseBaseArg(argc, argv);
    if(base_arg == "--version")
    {
        size_t major, minor, patch;
        major = minor = patch = 0;
        std::cout << "conv_driver version: " << major << "." << minor << "." << patch << ")"
                  << std::endl;
        std::cout << "All of above is a fake version. We are sorry that driver does not have a version yet, lol" << std::endl;
        exit(0);
    }

    // determine data type of the running config
    driverDataType_t driver_data_type;
    if(base_arg == "conv")
        driver_data_type = driverFloat;
    else if(base_arg == "convfp16")
        driver_data_type = driverHalf;
    else if(base_arg == "convbf16") {
        driver_data_type = driverBFloat16;
        exit(0);
    }
    else
        exit(0);

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
    int ngroups = conv_args.get_int("group_count");
    int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);
    int forw = conv_args.get_int("forw");

    int need_fwd = (forw == 0 ? 1 : (forw & 1 ? 1 : 0));
    int need_bwd = (forw == 0 ? 1 : (forw & 2 ? 1 : 0));
    int need_wrw = (forw == 0 ? 1 : (forw & 4 ? 1 : 0));

    // init host side
    float *host_input = (float *)malloc(static_cast<size_t>(n) * c * hi * wi * sizeof(float));
    float *host_weight = (float *)malloc(static_cast<size_t>(k) * c * y * x * sizeof(float));
    float *host_output = (float *)malloc(static_cast<size_t>(n) * k * ho * wo * sizeof(float));

    float *device_input;
    float *device_weight;
    float *device_output;

    HIP_CALL(hipMalloc(&device_input, static_cast<size_t>(n) * c * hi * wi * sizeof(float)));
    HIP_CALL(hipMalloc(&device_weight, static_cast<size_t>(k) * c * y * x * sizeof(float)));
    HIP_CALL(hipMalloc(&device_output, static_cast<size_t>(n) * k * ho * wo * sizeof(float)));

    // fp16 type
    float16 *host_input_f16  = (float16 *)malloc(n * c * hi * wi * sizeof(float16));
    float16 *host_weight_f16 = (float16 *)malloc(k * c * y * x * sizeof(float16));
    float16 *host_output_f16 = (float16 *)malloc(n * k * ho * wo * sizeof(float16));

    float16 *device_input_f16;
    float16 *device_weight_f16;
    float16 *device_output_f16;

    HIP_CALL(hipMalloc(&device_input_f16, n * c * hi * wi * sizeof(float16)));
    HIP_CALL(hipMalloc(&device_weight_f16, k * c * y * x * sizeof(float16)));
    HIP_CALL(hipMalloc(&device_output_f16, n * k * ho * wo * sizeof(float16)));

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
        if(driver_data_type == driverFloat)
            num_simd = 4 * 32 ; // 4x miSIMD, 32x mac unit
        else if(driver_data_type == driverHalf)
            num_simd = 4 * 128; // 4x miSIMD, 128x mac unit for fp16
        else if(driver_data_type == driverBFloat16)
            num_simd = 4 * 64 ; // 4x miSIMD, 64x mac unit for bf16
    }
    double fp32_gflops =
        theoritical_fp32_gflops(((double)sclk_mhz) / 1000.0, num_cu, num_simd);

    double nrms = get_nrms(forw, driver_data_type);

    if (need_fwd){
        result_t fastest_result_fwd;
        fastest_result_fwd.duration_ms = FLT_MAX;
        int fastest_id = -1;
        float *device_output_to_host = NULL;
        //float16 *device_output_to_host_f16 = NULL;
        if (need_verify) {
            // gen rand
            //gen_rand_vector<float, float>(host_input, n * c * hi * wi, 0.0, 1.0);
            //gen_rand_vector<float, float>(host_weight, k * c * y * x, -0.5, 0.5);
            gen_rand_vector<float, int>(host_input, n * c * hi * wi, -5, 5);
            gen_rand_vector<float, int>(host_weight, k * c * y * x, -2, 2);
            //gen_rand_vector<float, int>(host_input, n * c * hi * wi, 1, 1);
            //gen_rand_vector<float, int>(host_weight, k * c * y * x, 1, 1);
            if(driver_data_type == driverHalf){
                // move to different data type
                tensor_copy<float16, float>(host_input_f16, host_input, n * c * hi * wi);
                tensor_copy<float16, float>(host_weight_f16, host_weight, k * c * y * x);
                tensor_copy<float, float16>(host_input, host_input_f16, n * c * hi * wi);
                tensor_copy<float, float16>(host_weight, host_weight_f16, k * c * y * x);
            }

#ifdef USE_GPU_NAIVE_CONV
            HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                       static_cast<size_t>(k) * c * y * x * sizeof(float), hipMemcpyHostToDevice));
            
            gpu_naive_conv_fwd_nchw_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipMemcpy(host_output, device_output,
                                   static_cast<size_t>(n) * k * ho * wo * sizeof(float),
                                   hipMemcpyDeviceToHost));
#else
            conv_fwd_nchw(host_input, host_weight, host_output, n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
#endif
            if(driver_data_type == driverHalf)
                device_output_to_host = (float *)malloc((static_cast<size_t>(n) * k * ho * wo * sizeof(float16) + 3) / 4 * 4);
            else
                device_output_to_host = (float *)malloc(static_cast<size_t>(n) * k * ho * wo * sizeof(float));
            
        }
        if(driver_data_type == driverFloat){
            HIP_CALL(hipMemcpy(device_input, host_input,
                        static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                        static_cast<size_t>(k0 * c * y * x * sizeof(float), hipMemcpyHostToDevice));
        }
        else if(driver_data_type == driverHalf){
            HIP_CALL(hipMemcpy(device_input_f16, host_input_f16,
                        static_cast<size_t>(n) * c * hi * wi * sizeof(float16), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight_f16, host_weight_f16,
                        static_cast<size_t>(k) * c * y * x * sizeof(float16), hipMemcpyHostToDevice));
        }
        igemm_fwd_gtc_t conv_fwd_driver;
        for (int i = 0; i < tunables.size(); i++) {
            igemm_gtc_tunable_t *tunable = &tunables[i];

            if(!run_first_applicable) 
                 printf("[fwd:%2d] %s, ", i, conv_fwd_driver.get_kernel_name(tunable).c_str());

            result_t result;
            if(driver_data_type == driverFloat)
                result = conv_fwd_driver.run(&conv_args, tunable, module, device_input,
                                              device_weight, device_output, warmup, repeat, driver_data_type);
            else
                result = conv_fwd_driver.run(&conv_args, tunable, module, device_input_f16,
                                              device_weight_f16, device_output_f16, warmup, repeat, driver_data_type);
            
            if (result.return_code != 0){
                if ( ! run_first_applicable ) 
                     printf("not applicatble\n");
                continue;
            }

#if 0
            printf("input\r\n");
            for (int i_check = 0; i_check < (0+32); i_check++)
            {
                printf("[%d]th var to monitor:[%f, %d]\r\n", i_check*hi*wi, host_input[i_check*hi*wi], ((int *)host_input)[i_check*hi*wi]);
            }
            printf("input fp16\r\n");
            for (int i_check = 0; i_check < (0+32); i_check++)
            {
                printf("[%d]th var to monitor:[%f, %d]\r\n", i_check*hi*wi, (float)(host_input_f16[i_check*hi*wi]), ((unsigned short *)host_input_f16)[i_check*hi*wi]);
            }
#endif

            double gflops = measured_fp32_conv_gflops(
                result.duration_ms, n, c, hi, wi, k, y, x, stride_h, stride_w,
                dilation_h, dilation_w, pad_h, pad_w, ngroups);
            printf("cost:%.3fms, tflops:%.3f(%.2f%%)", result.duration_ms,
                   gflops / 1000 , (gflops / fp32_gflops) * 100);
            if (need_verify) {
                bool is_valid;
                if(driver_data_type == driverFloat) {
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output,
                                   static_cast<size_t>(n) * k * ho * wo * sizeof(float),
                                   hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_output, device_output_to_host,
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                }
                else if(driver_data_type == driverHalf) {
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output_f16,
                                   static_cast<size_t>(n) * k * ho * wo * sizeof(float16),
                                   hipMemcpyDeviceToHost));
                    float16 *device_output_to_host_fp16 = (float16 *)device_output_to_host;
                    is_valid = valid_vector<float16>(host_output, device_output_to_host_fp16,
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                }
                
                
                printf(", valid:%s", is_valid ? "y" : "n");
                if(assert_when_invalid) assert(is_valid);
            }
            printf("\n");

            if ( run_first_applicable ) {
                printf("\n"); 
		        break; 
            }; 
            if(result.duration_ms < fastest_result_fwd.duration_ms){
                fastest_result_fwd = result;
                fastest_result_fwd.gflops = (float)gflops;
                fastest_result_fwd.efficiency = (gflops / fp32_gflops) * 100;
                fastest_id = i;
            }
        }
        if(log_fastest_config && !run_first_applicable){
            dump_arg(&conv_args);
            if(fastest_id == -1)
                printf("  fastest: no suitable kernel\n");
            else
                printf("  fastest: [%d]%s, cost:%.3fms, tflops:%.3f(%.2f%%)\n",
                    fastest_id,
                    fastest_result_fwd.kernel_name.c_str(),
                    fastest_result_fwd.duration_ms,
                    fastest_result_fwd.gflops / 1000,
                    fastest_result_fwd.efficiency);
        }
        if (need_verify){
            free(device_output_to_host);
        }
    }

    if (need_bwd){
        float *device_input_to_host = NULL;
        result_t fastest_result_bwd;
        fastest_result_bwd.duration_ms = FLT_MAX;
        int fastest_id = -1;
        if (need_verify) {
            // gen rand
            gen_rand_vector<float, float>(host_output, static_cast<size_t>(n) * k * ho * wo, 0.0, 1.0);
            gen_rand_vector<float, float>(host_weight, static_cast<size_t>(k) * c * y * x, -0.5, 0.5);
            gen_rand_vector<float, float>(host_input, static_cast<size_t>(n) * c * hi * wi, 999999., 9999999.);  // manually input value to a very large number
            // gen_rand_vector<float, int>(host_output, n * k * ho * wo,1, 1);
            // gen_rand_vector<float, int>(host_weight, k * c * y * x, 1, 1);
#ifdef USE_GPU_NAIVE_CONV
            HIP_CALL(hipMemcpy(device_output, host_output,
                       static_cast<size_t>(n) * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                       static_cast<size_t>(k) * c * y * x * sizeof(float), hipMemcpyHostToDevice));
            gpu_naive_conv_bwd_nchw_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipMemcpy(host_input, device_input,
                                   static_cast<size_t>(n) * c * hi * wi * sizeof(float),
                                   hipMemcpyDeviceToHost));
#else
            conv_bwd_nchw(host_input, host_weight, host_output, n,
                                         wi, hi, c, k, x, y, pad_w,
                                         pad_h, stride_w, stride_h, dilation_w, dilation_h, ngroups);
#endif
            device_input_to_host = (float *)malloc(static_cast<size_t>(n) * c * hi * wi * sizeof(float));
            // printf("len:%d\n", n * c * hi * wi * sizeof(float) );
        }

        HIP_CALL(hipMemcpy(device_output, host_output,
                       static_cast<size_t>(n) * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(device_weight, host_weight,
                       static_cast<size_t>(k) * c * y * x * sizeof(float), hipMemcpyHostToDevice));


        igemm_bwd_gtc_t conv_bwd_driver;
        //double nrms = get_bwd_nrms();
        for (int i = 0; i < tunables.size(); i++) {
            igemm_gtc_tunable_t *tunable = &tunables[i];

            printf("[bwd:%2d] %s, ", i, conv_bwd_driver.get_kernel_name(tunable).c_str());
            fflush(stdout);

            if (need_verify)
                HIP_CALL(hipMemset(device_input, 0x7f,
                                   static_cast<size_t>(n) * c * hi * wi * sizeof(float)));   // 0x7f7f7f7f ~= 7.41e+28, a very large number
            result_t result =
                conv_bwd_driver.run(&conv_args, tunable, module, device_input,
                                device_weight, device_output, warmup, repeat);
            if (result.return_code != 0){
                printf("not applicatble\n");
                continue;
            }

            double gflops = measured_fp32_conv_gflops(
                result.duration_ms, n, c, hi, wi, k, y, x, stride_h, stride_w,
                dilation_h, dilation_w, pad_h, pad_w, ngroups);
            printf("cost:%.3fms, tflops:%.3f(%.2f%%)", result.duration_ms,
                   gflops / 1000 , (gflops / fp32_gflops) * 100);
            if (need_verify) {
                HIP_CALL(hipMemcpy(device_input_to_host, device_input,
                                   static_cast<size_t>(n) * c * hi * wi * sizeof(float),
                                   hipMemcpyDeviceToHost));
                bool is_valid = valid_vector<float>(host_input, device_input_to_host,
                                            static_cast<size_t>(n) * c * hi * wi, nrms);
                printf(", valid:%s", is_valid ? "y" : "n");
                if(assert_when_invalid) assert(is_valid);
                // if (!is_valid) {
                //     printf("\n");
                //     break;
                // }
            }
            printf("\n");
            if(result.duration_ms < fastest_result_bwd.duration_ms){
                fastest_result_bwd = result;
                fastest_result_bwd.gflops = (float)gflops;
                fastest_result_bwd.efficiency = (gflops / fp32_gflops) * 100;
                fastest_id = i;
            }
        }
        if(log_fastest_config){
            dump_arg(&conv_args);
            if(fastest_id == -1)
                printf("  fastest: no suitable kernel\n");
            else
                printf("  fastest: [%d]%s, cost:%.3fms, tflops:%.3f(%.2f%%)\n",
                    fastest_id,
                    fastest_result_bwd.kernel_name.c_str(),
                    fastest_result_bwd.duration_ms,
                    fastest_result_bwd.gflops / 1000,
                    fastest_result_bwd.efficiency);
        }
        if (need_verify) 
            free(device_input_to_host);
    }
    if (need_wrw){
        float *device_weight_to_host = NULL;
        if (need_verify) {
            // gen rand
            gen_rand_vector<float, float>(host_input, static_cast<size_t>(n) * c * hi * wi, 0.0, 1.0);
            gen_rand_vector<float, float>(host_output, static_cast<size_t>(n) * k * ho * wo, -0.5, 0.5);
            //gen_rand_vector<float, int>(host_input, n * k * hi * wi, -5, 5);
            //gen_rand_vector<float, int>(host_output, n * k * ho * wo, 1, 1);
#ifdef USE_GPU_NAIVE_CONV
            HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_output, host_output,
                       static_cast<size_t>(n) * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));
            gpu_naive_conv_wrw_nchw_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipMemcpy(host_weight, device_weight,
                                   static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x * sizeof(float),
                                   hipMemcpyDeviceToHost));
#else
            conv_wrw_nchw(host_input, host_weight, host_output, n,
                                         wi, hi, c, k, x, y, pad_w,
                                         pad_h, stride_w, stride_h, dilation_w, dilation_h, ngroups);
#endif
            device_weight_to_host = (float *)malloc(static_cast<size_t>(k) * c * y * x * sizeof(float));
            // printf("len:%d\n", k * c * y * x * sizeof(float));
        }

        HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(device_output, host_output,
                       static_cast<size_t>(n) * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));

#if 0
        printf("input\r\n");
        for (int i_check = 0; i_check < (0+32); i_check++)
        {
            printf("[%d]th var to monitor:[%f, %d]\r\n", i_check*hi*wi, host_input[i_check*hi*wi], ((int *)host_input)[i_check*hi*wi]);
        }
        printf("output\r\n");
        for (int i_check = 0; i_check < (0+32); i_check++)
        {
            printf("[%d]th var to monitor:[%f, %d]\r\n", i_check*ho*wo, host_output[i_check*ho*wo], ((int *)host_output)[i_check*ho*wo]);
        }
        printf("input\r\n");
        for (int i_check = 0; i_check < (0+32); i_check++)
        {
            printf("[%d]th var to monitor:[%f, %d]\r\n", i_check, host_input[i_check], ((int *)host_input)[i_check]);
        }
        printf("output\r\n");
        for (int i_check = 0; i_check < (0+32); i_check++)
        {
            printf("[%d]th var to monitor:[%f, %d]\r\n", i_check, host_output[i_check], ((int *)host_output)[i_check]);
        }
        printf("workspace debug end \r\n");
#endif   

        igemm_wrw_gtc_t conv_wrw_driver;
        float min_duration = 10000000.0f;
        float selected_duration = 10000000.0f;
        //double nrms = get_wrw_nrms();
        std::string kernel_name;

        std::string selected_kernel;

        selected_kernel = conv_wrw_driver.select_kernel(&conv_args, tunables);

        int min_grid = 0;
        int sel_grid = 0;

        for (int i = 0; i < tunables.size(); i++) {
            igemm_gtc_tunable_t *tunable = &tunables[i];

            printf("[wrw:%2d] %s, ", i, conv_wrw_driver.get_kernel_name(tunable).c_str());
            fflush(stdout);

            if (need_verify)
                HIP_CALL(hipMemset(device_weight, 0,
                                   k * c * y * x * sizeof(float)));
            result_t result =
                conv_wrw_driver.run(&conv_args, tunable, module, device_input,
                                device_weight, device_output, warmup, repeat);

            if (result.return_code != 0)
                continue;
            int grid_size = conv_wrw_driver.get_grid_size(&conv_args, tunable); 
            double gflops = measured_fp32_conv_gflops(
                result.duration_ms, n, c, hi, wi, k, y, x, stride_h, stride_w,
                dilation_h, dilation_w, pad_h, pad_w, ngroups);
            printf("cost:%.3fms, tflops:%.3f(%.2f%%)", result.duration_ms,
                   gflops / 1000 , (gflops / fp32_gflops) * 100);
            if (result.duration_ms < min_duration)
            {
                min_duration = result.duration_ms;
                kernel_name = conv_wrw_driver.get_kernel_name(tunable).c_str();
                min_grid = grid_size;
            }
            if (selected_kernel == conv_wrw_driver.get_kernel_name(tunable).c_str())
            {
                selected_duration = result.duration_ms;
                sel_grid = grid_size;
            }
            if (need_verify) {
                HIP_CALL(hipMemcpy(device_weight_to_host, device_weight,
                                   static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x * sizeof(float),
                                   hipMemcpyDeviceToHost));
                bool is_valid = valid_vector<float>(host_weight, device_weight_to_host,
                                            static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x, nrms);
                printf(", valid:%s", is_valid ? "y" : "n");
                if(assert_when_invalid) assert(is_valid);
                // if (!is_valid) {
                //     printf("\n");
                //     break;
                // }
            }
            printf("\n");
        }
        double gflops = measured_fp32_conv_gflops(
                min_duration, n, c, hi, wi, k, y, x, stride_h, stride_w,
                dilation_h, dilation_w, pad_h, pad_w, ngroups);
        printf("min cost:%.3fms, tflops:%.3f(%.2f%%),  min grid:%d\r\n", min_duration,
                   gflops / 1000 , (gflops / fp32_gflops) * 100, min_grid);
        std::cout << "min name:" << kernel_name << std::endl;
        double selected_gflops = measured_fp32_conv_gflops(
                selected_duration, n, c, hi, wi, k, y, x, stride_h, stride_w,
                dilation_h, dilation_w, pad_h, pad_w, ngroups);
        printf("sel cost:%.3fms, tflops:%.3f(%.2f%%), sel grid:%d\r\n", selected_duration,
                   selected_gflops / 1000 , (selected_gflops / fp32_gflops) * 100, sel_grid);
        std::cout << "sel name:" << selected_kernel << std::endl;

        // write out log file to see if selected one is good enough.
        if (wrw_kernel_selection == 1)
        {
            FILE *debug_log = fopen("./wrw_select_kernel.log", "a+");
            if (debug_log != nullptr){
                fprintf(debug_log, "conv n=%d, c=%d, hi=%d, wi=%d, k=%d, y=%d, x=%d, stride_h=%d, stride_w=%d, ho=%d, wo=%d \r\n", n, c, hi, wi, k, y, x, stride_h, stride_w, ho, wo);
                fprintf(debug_log, "min_kernel: %s, min cost:%.3fms, min grid:%d\r\n", kernel_name.data(), min_duration, min_grid);
                fprintf(debug_log, "sel_kernel: %s, sel cost:%.3fms, sel grid:%d\r\n", selected_kernel.data(), selected_duration, sel_grid);
            }
            fclose(debug_log);
        }
        if (need_verify) 
            free(device_weight_to_host);
    }

    free(host_input);
    free(host_weight);
    free(host_output);

    hipFree(device_input);
    hipFree(device_weight);
    hipFree(device_output);

    free(host_input_f16);
    free(host_weight_f16);
    free(host_output_f16);

    hipFree(device_input_f16);
    hipFree(device_weight_f16);
    hipFree(device_output_f16);
}
