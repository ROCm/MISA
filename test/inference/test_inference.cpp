
#include <vector>
#include <string>
#include <assert.h>
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
#include <iostream>
#include "args.h"


#include "half.hpp"
using float16 = half_float::half;

std::string parse_base_arg(int argc, char* argv[])
{
    if(argc < 2)
    {
        printf("Invalid Number of Input Arguments\n");
        exit(0);
    }

    std::string arg = argv[1];

    if(arg != "conv" && arg != "convfp16" && arg != "convint8" && arg != "--version")
    {
        printf("Invalid Base Input Argument\n");
        exit(0);
    }
    else if(arg == "-h" || arg == "--help" || arg == "-?")
        exit(0);
    else
        return arg;
}

static inline size_t conv_out_size(size_t in_size, size_t pad, size_t dilation,
                                   size_t ksize, size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}
typedef struct {
    uint32_t magic;
    uint8_t shift;
} magic_div_u32_t;
static inline magic_div_u32_t magic_div_u32_gen(uint32_t d) {
    assert(d >= 1 && d <= INT32_MAX);
    uint8_t shift;
    for (shift = 0; shift < 32; shift++)
        if ((1U << shift) >= d)
            break;

    uint64_t one = 1;
    uint64_t magic = ((one << 32) * ((one << shift) - d)) / d + 1;
    assert(magic <= 0xffffffffUL);

    magic_div_u32_t result;
    result.magic = magic;
    result.shift = shift;
    return result;
}
static inline uint32_t magic_div_u32_pack_shift(uint8_t s0, uint8_t s1, uint8_t s2, uint8_t s3)
{
    uint32_t shift_0 = static_cast<uint32_t>(s0);
    uint32_t shift_1 = static_cast<uint32_t>(s1);
    uint32_t shift_2 = static_cast<uint32_t>(s2);
    uint32_t shift_3 = static_cast<uint32_t>(s3);
    return (shift_3 << 24) | (shift_2 << 16) | (shift_1 << 8) | shift_0;
}
typedef struct {
    int return_code;
    float duration_ms;
    float gflops;
    float efficiency;
    std::string kernel_name;
} result_t;


typedef enum {
    driverHalf  = 0, /*!< 16-bit floating point (Fully supported) */
    driverFloat = 1, /*!< 32-bit floating point (Fully supported) */
    driverInt8  = 3,
    driverBFloat16 = 5, /*!< 16-bit binary floating point (8-bit exponent, 7-bit fraction)
                           (Partially supported) */
} driverDataType_t;

static inline size_t get_data_byte(driverDataType_t dtype)
{
    if(dtype == driverHalf)
        return 2;
    if(dtype == driverFloat)
        return 4;
    if(dtype == driverInt8)
        return 1;
    if(dtype == driverBFloat16)
        return 2;
    assert(0);
    return 0;
}

static inline int env_get_int(const char *var_name, int default_int) {
    char *v = getenv(var_name);
    int r = default_int;
    if (v)
        r = atoi(v);
    return r;
}

#define NAIVE_CONV_THREADED
#include "naive_conv.h"
#include "gpu_naive_conv.h"
#include "igemm_fwd_btm_nhwc.h"


#define HIP_CALL(call)                                                         \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            printf("[hiperror](%d) fail to call %s,(%s)", (int)err, #call,     \
                   hipGetErrorString(err));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static int gen_rand_integer()
{
    static int inited = 0;
    if(inited == 0)
    {
        std::srand(std::time(nullptr));
        inited = 1;
    }
    return std::rand();
}


static inline char *env_get_str(const char *var_name, char *default_str) {
    char *v = getenv(var_name);
    if (v)
        return v;
    return default_str;
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

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.push_back(std::thread(block_wise_tensor_copy<Dst_T, Src_T>,
            p_dst, p_src, t, num_threads, tensor_size));
    }
    for (auto &th : threads)
        th.join();
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

template<typename T>
bool valid_float(T p)
{
    return !(std::isnan(p) || std::isinf(p));
}

template<>
bool valid_float<int8_t>(int8_t p)
{
    // there is no meaning to valid integer number
    return true;
}

#ifndef ABS
#define ABS(b) ((b) > 0 ? (b) : -1 * (b))
#endif
template<typename T>
bool valid_vector(const float *ref, const T *pred, size_t n,
                                double nrms = 1e-6) {
    double s0 = 0.0;
    double s1 = 0.0;
    int igemm_per_pixel_check = env_get_int("PER_PIXEL_CHECK", 0);
    int igemm_per_pixel_check_print = env_get_int("PER_PIXEL_CHECK_PRINT", 1);
    size_t pp_err = 0;

    for (size_t i = 0; i < n; ++i) {
        double ri = (double)ref[i];
        double pi = (double)pred[i];
        if(!(valid_float<float>(ref[i]) && valid_float<T>(pred[i]))){
            printf(" invalid float at %4zu, ref:%f, pred:%f\n", i, ri, pi);
            return false;
        }
        double d = ri - pi;
        double dd = d * d;
        double rr = 2.0 * ri * ri;
        s0 += dd;
        s1 += rr;
        if(igemm_per_pixel_check){
            double delta = ABS(ABS(ri - pi) / ri);
            printf("[%zu] ref:%lf, pred:%lf(0x%08x) [%s]\n", i, ri, pi,  *(uint32_t*)(&pred[i]), delta > 3e-5? "N":"Y");
            if (delta > 3e-5) {
                if(igemm_per_pixel_check_print){
                    if (pp_err < 100)
                        printf("diff at %zu, ref:%lf, pred:%lf(0x%08x), d:%lf\n", i, ri,
                            pi, *(uint32_t*)(&pred[i]), delta);
                }
                pp_err++;
            }
        }
    }
    // printf("\nnrms:%lf, s0:%lf, s1:%lf, expected_nrms is %1f\n",sqrt(s0/s1),s0,s1,nrms);
    return (sqrt(s0 / s1) < nrms)
#ifdef PER_PIXEL_CHECK
           && (pp_err == 0)
#endif
        ;
}

template<>
bool valid_vector<int8_t>(const float *ref, const int8_t *pred, size_t n,
                                double nrms) {
    // int8 valid, we prefer a per pixel match
    int igemm_per_pixel_check = env_get_int("PER_PIXEL_CHECK", 0);
    int igemm_per_pixel_check_print = env_get_int("PER_PIXEL_CHECK_PRINT", 1);
    size_t pp_err = 0;

    for (size_t i = 0; i < n; ++i) {
        if(!(valid_float<float>(ref[i]) ) ){
            printf(" invalid float at %4zu, ref:%f\n", i, ref[i]);
            return false;
        }
        int8_t pi = pred[i];
        int32_t ri = static_cast<int32_t>(ref[i]);
        int8_t ri_clamp;
        memcpy(&ri_clamp, &ri, 1);

        if(igemm_per_pixel_check){
            printf("[%zu] ref:%d(%d), pred:%d(0x%08x) [%s]\n", i, ri, ri_clamp, pi,
                        *(uint32_t*)(&pred[i]), pi != ri_clamp ? "N":"Y");
        }

        if(pi != ri_clamp){
            pp_err++;
        }
    }
    return pp_err == 0;
}

static inline void dump_output_dword(const float *out, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        double pi = (double)out[i];
        printf("[%zu] pred:%lf(0x%08x)\n", i, pi, ((uint32_t *)out)[i]);
    }
}

static inline double theoritical_gflops(double sclk_ghz, size_t cu,
                                             size_t simd) {
    return 2 * sclk_ghz * cu * simd;
}

static inline double
theoritical_conv_flop(size_t n, size_t c, size_t hi, size_t wi, size_t k,
                           size_t y, size_t x, size_t stride_h, size_t stride_w,
                           size_t dilation_h, size_t dilation_w, size_t pad_h,
                           size_t pad_w, size_t ngroups) {
    size_t ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    size_t wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

    double flop = (double)n * c * ho * wo * k * y * x * 2 / ngroups;
    return flop;
}
static inline double
measured_conv_gflops(double time_ms, size_t n, size_t c, size_t hi,
                          size_t wi, size_t k, size_t y, size_t x,
                          size_t stride_h, size_t stride_w, size_t dilation_h,
                          size_t dilation_w, size_t pad_h, size_t pad_w, size_t ngroups) {
    double flop =
        theoritical_conv_flop(n, c, hi, wi, k, y, x, stride_h, stride_w,
                                   dilation_h, dilation_w, pad_h, pad_w, ngroups);
    return flop / (time_ms * 1e6);
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

#define     GPU_NAIVE_CONV_HSACO    "naive_conv.hsaco"
#define     SCLK_MHZ    2200
#define WARMUP 3
#define REPEAT 8

#ifndef HSACO
#define HSACO "igemm_fwd_btm_nhwc_fp16.hsaco"
#endif
int main(int argc, char **argv){
    int warmup = env_get_int("WARMUP", WARMUP);
    int repeat = env_get_int("REPEAT", REPEAT);
    int sclk_mhz = env_get_int("SCLK_MHZ", SCLK_MHZ);
    int dump_out = env_get_int("DUMP_OUT", 0);
    int log_fastest_config = env_get_int("IGEMM_LOG_FASTEST_CONFIG", 0);
    
    char *gpu_naive_conv_hsaco = env_get_str("GPU_NAIVE_CONV_HSACO", GPU_NAIVE_CONV_HSACO);
    gpu_naive_conv_init(gpu_naive_conv_hsaco);

    std::string base_arg = parse_base_arg(argc, argv);
    std::string default_hsaco = "igemm_fwd_btm_nhwc_";

    driverDataType_t driver_data_type;
    int fp_factor = 1;
    if(base_arg == "conv"){
        driver_data_type = driverFloat;
        default_hsaco += "fp32.hsaco";
    }
    else if(base_arg == "convfp16"){
        driver_data_type = driverHalf;
        default_hsaco += "fp16.hsaco";
        fp_factor = 2;
    }
    else if(base_arg == "convbf16") {
        driver_data_type = driverBFloat16;
        exit(0);
    }
    else if(base_arg == "convint8") {
        driver_data_type = driverInt8;
        default_hsaco += "int8.hsaco";
        fp_factor = 4;
    }
    else
        exit(0);

    size_t data_byte = get_data_byte(driver_data_type);
    char *hsaco = env_get_str("HSACO", const_cast<char*>(default_hsaco.c_str()));

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


    void *host_input_dtype  = malloc(n * c * hi * wi * data_byte);
    void *host_weight_dtype = malloc(k * c * y * x * data_byte);
    void *host_output_dtype = malloc(n * k * ho * wo * data_byte);

    void *device_input_dtype;
    void *device_weight_dtype;
    void *device_output_dtype;

    HIP_CALL(hipMalloc(&device_input_dtype, n * c * hi * wi * data_byte));
    HIP_CALL(hipMalloc(&device_weight_dtype, k * c * y * x * data_byte));
    HIP_CALL(hipMalloc(&device_output_dtype, n * k * ho * wo * data_byte));

    int need_verify = conv_args.get_int("verify");

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
        if(gcn_arch >= 1000)
            num_cu *= 2;
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

    double theo_gflops = theoritical_gflops(((double)sclk_mhz) / 1000.0, num_cu, num_simd * fp_factor);
    double nrms = get_nrms(forw, driver_data_type);

    printf("num_cu:%d, gcn_arch:%d, theo_gflops:%f\n", num_cu, gcn_arch, theo_gflops);

    if (need_fwd){
        void *device_output_to_host = NULL;
        if (need_verify) {
            // gen rand
            //gen_rand_vector<float, float>(host_input, static_cast<size_t>(n) * c * hi * wi, 0.0, 1.0);
            //gen_rand_vector<float, float>(host_weight, static_cast<size_t>(k) * c * y * x, -0.5, 0.5);
            gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * c * hi * wi, -5, 5);
            gen_rand_vector<float, int>(host_weight, static_cast<size_t>(k) * c * y * x, -5, 5);
            //gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * c * hi * wi, 1, 1);
            //gen_rand_vector<float, int>(host_weight, static_cast<size_t>(k) * c * y * x, 1, 1);

            if(driver_data_type == driverHalf){
                // move to different data type
                tensor_copy<float16, float>(static_cast<float16*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * hi * wi);
                tensor_copy<float16, float>(static_cast<float16*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * y * x);
            }
            else if(driver_data_type == driverInt8){
                // move to different data type
                tensor_copy<int8_t, float>(static_cast<int8_t*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * hi * wi);
                tensor_copy<int8_t, float>(static_cast<int8_t*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * y * x);
            }

            HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                       static_cast<size_t>(k) * c * y * x * sizeof(float), hipMemcpyHostToDevice));
            
            gpu_naive_conv_fwd_nhwc_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipMemcpy(host_output, device_output,
                                   static_cast<size_t>(n) * k * ho * wo * sizeof(float),
                                   hipMemcpyDeviceToHost));

            if(driver_data_type == driverHalf || driver_data_type == driverInt8){
                device_output_to_host = malloc((static_cast<size_t>(n) * k * ho * wo * data_byte + 3) / 4 * 4);
            }
            else{
                device_output_to_host = malloc(static_cast<size_t>(n) * k * ho * wo * sizeof(float));
            }
            
        }

        if(driver_data_type == driverFloat){
            HIP_CALL(hipMemcpy(device_input, host_input,
                        static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                        static_cast<size_t>(k) * c * y * x * sizeof(float), hipMemcpyHostToDevice));
        }else{
            HIP_CALL(hipMemcpy(device_input_dtype, host_input_dtype,
                        static_cast<size_t>(n) * c * hi * wi * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight_dtype, host_weight_dtype,
                        static_cast<size_t>(k) * c * y * x * data_byte, hipMemcpyHostToDevice));
        }

        igemm_fwd_btm_t conv_fwd_driver;
        int valid_index = 0;
        result_t fastest_result;
        fastest_result.duration_ms = FLT_MAX;
        int fastest_id = -1;
        for (int i = 0; i < sizeof(igemm_fwd_btm_kernel_list)/sizeof(igemm_fwd_btm_kernel_list[0]); i++) {
            igemm_fwd_btm_kernel_info_t *kinfo = &igemm_fwd_btm_kernel_list[i];
            if(driver_data_type == driverHalf){
                if(kinfo->data_type != "fp16")
                    continue;
            }
            else if(driver_data_type == driverInt8){
                if(kinfo->data_type != "int8")
                    continue;
            }

            printf("[fwd:%2d] %s, ", valid_index, conv_fwd_driver.get_kernel_name(kinfo).c_str());
            fflush(stdout);

            result_t result;

            result = conv_fwd_driver.run(&conv_args, module, kinfo, device_input_dtype,
                                               device_weight_dtype, device_output_dtype, warmup, repeat, driver_data_type);

            valid_index++;

            if (result.return_code != 0){
                printf("not applicatble\n");
                continue;
            }

            double gflops = measured_conv_gflops(
                result.duration_ms, n, c, hi, wi, k, y, x, stride_h, stride_w,
                dilation_h, dilation_w, pad_h, pad_w, ngroups);
            result.gflops =  gflops;
            result.efficiency = (gflops / theo_gflops) * 100;
            printf("cost:%.3fms, tflops:%.3f(%.2f%%)", result.duration_ms,
                   gflops / 1000 , (gflops / theo_gflops) * 100);

            if(result.duration_ms < fastest_result.duration_ms){
                fastest_result = result;
                fastest_id = valid_index - 1;
            }
            if (need_verify) {
                bool is_valid;
                if(driver_data_type == driverFloat) {
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output,
                                   static_cast<size_t>(n) * k * ho * wo * sizeof(float),
                                   hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_output, static_cast<float*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                }
                else if(driver_data_type == driverHalf || driver_data_type == driverInt8) {
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output_dtype,
                                   static_cast<size_t>(n) * k * ho * wo * data_byte,
                                   hipMemcpyDeviceToHost));
                    if(dump_out)
                        dump_output_dword(static_cast<float*>(device_output_to_host), static_cast<size_t>(n) * k * ho * wo / fp_factor);
                    if(driver_data_type == driverHalf)
                        is_valid = valid_vector<float16>(host_output, static_cast<float16*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                    else if(driver_data_type == driverInt8)
                        is_valid = valid_vector<int8_t>(host_output, static_cast<int8_t*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                }
                printf(", valid:%s", is_valid ? "y" : "n");
            }
            printf("\n");
        }

        if(log_fastest_config){
            // dump_arg(conv_args);
            if(fastest_id == -1)
                printf("  fastest: no suitable kernel\n");
            else
                printf("  fastest: [%d]%s, cost:%.3fms, tflops:%.3f(%.2f%%)\n",
                    fastest_id,
                    fastest_result.kernel_name.c_str(),
                    fastest_result.duration_ms,
                    fastest_result.gflops / 1000,
                    fastest_result.efficiency);
        }

        if (need_verify){
            free(device_output_to_host);
        }
    }
}