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
#include <chrono>
#include <functional>

#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <vector>
#include <float.h>
#include <cmath>
#include <algorithm>
#include <limits>

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
#   define NAIVE_CONV_THREADED
#   include "naive_conv.h"
#endif

#ifndef USE_MIOPEN_NRMS
#define USE_MIOPEN_NRMS 1
#endif

#include "common.h"
#include "args.h"
#include "config_parser.h"
#include "perf.h"
#include "igemm_gtc_base.h"
#include "igemm_fwd_gtc_driver.h"
#include "igemm_bwd_gtc_driver.h"
#include "igemm_wrw_gtc_driver.h"

static inline double theoritical_gflops(double sclk_ghz, size_t cu,
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


static inline double get_theoritical_conv_flop(const args_t * conv_args)
{
    int hi = conv_args->get_int("in_h");
    int wi = conv_args->get_int("in_w");
    int n = conv_args->get_int("batchsize");
    int k = conv_args->get_int("out_channels");
    int c = conv_args->get_int("in_channels");

    int stride_h = conv_args->get_int("conv_stride_h");
    int stride_w = conv_args->get_int("conv_stride_w");
    int dilation_h = conv_args->get_int("dilation_h");
    int dilation_w = conv_args->get_int("dilation_w");
    int pad_h = conv_args->get_int("pad_h");
    int pad_w = conv_args->get_int("pad_w");
    int y = conv_args->get_int("fil_h");
    int x = conv_args->get_int("fil_w");
    int ngroups = conv_args->get_int("group_count");
    int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

    return theoritical_fp32_conv_flop(n, c, hi, wi, k, y, x, stride_h, stride_w,
                                   dilation_h, dilation_w, pad_h, pad_w, ngroups);
}

static inline double get_theoritical_gpu_gflops(int sclk_mhz, driverDataType_t data_type)
{
    int num_cu;
    int gcn_arch = 0;
    int num_simd = 4 * 16;
    hipDeviceProp_t dev_prop;
    hipDevice_t dev;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    num_cu = dev_prop.multiProcessorCount;
    gcn_arch = dev_prop.gcnArch;
    if(gcn_arch >= 1000)
        num_cu *= 2;

    int fp_factor = 1;
    if(data_type == driverHalf){
        if(gcn_arch == 908)
            fp_factor = 4;  // xdlops
        else
            fp_factor = 2;  // dlops
    }
    if(data_type == driverInt8){
        if(gcn_arch == 908)
            fp_factor = 4;  // xdlops
        else
            fp_factor = 4;  // dlops
    }
    // else if(data_type == driverInt8){
    //     if(gcn_arch == 908)
    //     fp_factor = 4;
    // }

    if(gcn_arch == 908){
        num_simd = 4 * 32 ; // 4x miSIMD, 32x mac unit
    }

    return theoritical_gflops(((double)sclk_mhz) / 1000.0, num_cu, num_simd * fp_factor);
}

#ifndef ABS
#define ABS(x) ((x) > 0 ? (x) : -1 * (x))
#endif

#ifndef IGEMM_HSACO
#define IGEMM_HSACO "igemm_gtc.hsaco"
#endif

#ifndef IGEMM_CONFIG_FILE
#define IGEMM_CONFIG_FILE "igemm_gtc.config"
#endif

#define IGEMM_RUN_ONLY_KERNEL_DEFAULT "off"

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
void block_wise_rand_generator(Dst_T *p, int tid, int block_size, size_t total_size, Src_T min, Src_T max, Src_T scale)
{
    std::mt19937 rng(std::chrono::system_clock::now()
                        .time_since_epoch()
                        .count() +
                    std::hash<std::thread::id>()(std::this_thread::get_id()));
    distribution_t<Src_T> distribution(min,max);
    for (size_t i = tid; i < total_size; i += block_size) {
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

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.push_back(std::thread(block_wise_tensor_copy<Dst_T, Src_T>,
            p_dst, p_src, t, num_threads, tensor_size));
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

template<typename T>
static inline bool valid_vector(const float *ref, const T *pred, size_t n,
                                double nrms = 1.5e-6) {
    double s0 = 0.0;
    double s1 = 0.0;
    int igemm_per_pixel_check = env_get_int("PER_PIXEL_CHECK", 0);
    int igemm_per_pixel_check_print = env_get_int("PER_PIXEL_CHECK_PRINT", 1);
    int igemm_valid_float = env_get_int("VALID_FLOAT", 1);
    int dump_pred_dword = env_get_int("DUMP_PRED", 0);
    size_t pp_err = 0;

    if(dump_pred_dword){
        // dump as dword, weather the type of pred
        size_t total_safe_size = n / ( sizeof(float) / sizeof(T) );
        for(size_t i=0; i<total_safe_size;i++ ){
            printf("[%zu] ref:%lf, pred:0x%08x\n", i, ref[i], ((uint32_t*)pred)[i]);
        }
    }
#if USE_MIOPEN_NRMS
    double square_difference = .0;
    double mag1 = .0;
    double mag2 = .0;
    for (size_t i = 0; i < n; ++i) {
        if(igemm_valid_float)
            if(!(valid_float<float>(ref[i]) && valid_float<T>(pred[i]))){
                printf(" invalid float at %zu, ref:%f, pred:%f\n", i, ref[i], pred[i]);
                return false;
            }
        

        double ri = (double)ref[i];
        double pi = (double)pred[i];
        double d = ri - pi;

        if(igemm_per_pixel_check){
            double delta = ABS(ABS(ri - pi) / ri);      // TODO: this is just a reference compare
            printf("[%zu] ref:%lf, pred:%lf(0x%08x) [%s]\n", i, ri, pi, *(uint32_t*)(&pred[i]), delta > 3e-5? "N":"Y");
            if (delta > 3e-5) {
                if(igemm_per_pixel_check_print){
                    if (pp_err < 100)
                        printf("diff at %zu, ref:%lf, pred:%lf(0x%08x), d:%lf\n", i, ri,
                            pi, *(uint32_t*)(&pred[i]), delta);
                }
                pp_err++;
            }
        }

        square_difference += d * d;
        if(ABS(mag1) < ABS(ri)) mag1 = ri;
        if(ABS(mag2) < ABS(pi)) mag2 = pi;
    }
    double mag = std::max({std::fabs(mag1), std::fabs(mag2), std::numeric_limits<double>::min()});
    double computed_nrms = std::sqrt(square_difference) / (std::sqrt(n) * mag);
    return (computed_nrms < nrms)
#ifdef PER_PIXEL_CHECK
           && (pp_err == 0)
#endif
        ;
#else
    for (size_t i = 0; i < n; ++i) {
        if(igemm_valid_float)
            if(!(valid_float<float>(ref[i]) && valid_float<T>(pred[i]))){
                printf(" invalid float at %zu, ref:%f, pred:%f\n", i, ref[i], pred[i]);
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
            printf("[%zu] ref:%lf, pred:%lf(0x%08x) [%s]\n", i, ri, pi, *(uint32_t*)(&pred[i]), delta > 3e-5? "N":"Y");
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
#endif
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

static inline double get_nrms(std::string direction, driverDataType_t driver_data_type){
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
    if (direction == "bwd"){
        // nrms *= 10;
    }
    // wrw has a high tolerance
    if (direction == "wrw"){
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

int string_to_dir(std::string direction)
{
    if(direction == "fwd")
        return 1;
    if(direction == "bwd")
        return 2;
    if(direction == "wrw")
        return 4;
    assert(0);
}

std::string log_cmd(const args_t *arg, driverDataType_t driver_data_type, std::string direction)
{
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
    int ngroups = arg->get_int("group_count");
    std::string in_layout = arg->get_str("in_layout");
    std::string out_layout = arg->get_str("out_layout");
    std::string fil_layout = arg->get_str("fil_layout");

    std::stringstream ss;
    if(driver_data_type == driverHalf)
    {
        ss << "convfp16";
    }
    else if(driver_data_type == driverFloat)
    {
        ss << "conv";
    }

    ss << " -n " << n
        << " -c " << c
        << " -H " << hi
        << " -W " << wi
        << " -k " << k
        << " -y " << y
        << " -x " << x
        << " -p " << pad_h
        << " -q " << pad_w
        << " -u " << stride_h
        << " -v " << stride_w
        << " -l " << stride_h
        << " -j " << stride_w;

    if(in_layout != "NCHW")
        ss << " --in_layout " << in_layout;
    if(fil_layout != "NCHW")
        ss << " --fil_layout " << fil_layout;
    if(out_layout != "NCHW")
        ss << " --out_layout " << out_layout;

    ss << " -g " << ngroups
            << " -F " << std::to_string(string_to_dir(direction))
            << " -t 1";

    return ss.str();
}


template<typename driver_t, typename pre_func_t, typename post_func_t>
void launch_conv_driver(driver_t * driver, const args_t *conv_args, const std::vector<igemm_gtc_tunable_t> & tunables, std::string direction,
                    driverDataType_t driver_data_type, FILE * p_bcsv,
                    void* device_input, void* device_weight, void* device_output,
                    pre_func_t && pre_func, post_func_t && post_func)
{
    int sclk_mhz = env_get_int("IGEMM_SCLK_MHZ", SCLK_MHZ);
    std::string run_only_kernel = env_get_str("IGEMM_RUN_ONLY_KERNEL", IGEMM_RUN_ONLY_KERNEL_DEFAULT);
    int log_fastest_config = env_get_int("IGEMM_LOG_FASTEST_CONFIG", 0);
    int sleep_ms = env_get_int("IGEMM_SLEEP_MS", 0);
    int dump_gmap = env_get_int("IGEMM_DUMP_GMAP", 0);

    double theo_conv_flop  = get_theoritical_conv_flop(conv_args);
    double theo_gpu_gflops = get_theoritical_gpu_gflops(sclk_mhz, driver->data_type);

    auto launch = [&](const igemm_gtc_tunable_t * tunable, int index) -> result_t {
        if(run_only_kernel != IGEMM_RUN_ONLY_KERNEL_DEFAULT){
            if(run_only_kernel != driver->get_kernel_name(tunable)){
                return result_t{};
            }
        }
        
        printf("[%s:%2d] %s", direction.c_str(), index, driver->get_kernel_name(tunable).c_str());
        fflush(stdout);

        pre_func();

        result_t result = driver->run(conv_args, tunable, device_input, device_weight, device_output);

        std::string gks_string = "";
        if(tunable->gemm_k_global_split){
            gks_string = "[" + std::to_string(result.gks) + "]";
        }
        printf("%s, ", gks_string.c_str());

        if (result.return_code != 0){
            printf("not applicatble\n");
            return result_t{};
        }

        double gflops = theo_conv_flop / (result.duration_ms * 1e6);
        printf("cost:%.3fms, tflops:%.3f(%.2f%%)", result.duration_ms,
                gflops / 1000 , (gflops / theo_gpu_gflops) * 100);

        post_func();

        printf("\n");
        result.gflops = gflops;
        result.efficiency = (gflops / theo_gpu_gflops) * 100;

        if(dump_gmap)
            gmap_dump(conv_args, tunable);
        return result;
    };

    result_t fastest_result;
    fastest_result.duration_ms = FLT_MAX;
    int fastest_id = -1;
    if(driver->driver_mode == driver_mode_normal){
        for(int i=0; i<tunables.size(); i++){
            result_t result = launch(&tunables[i], i);

            if(result.duration_ms < fastest_result.duration_ms){
                fastest_result = result;
                fastest_id = i;
            }
        }

        if(log_fastest_config){
            dump_arg(conv_args);
            if(fastest_id == -1)
                printf("  fastest: no suitable kernel\n");
            else{
                std::string kernel_name_mock = fastest_result.kernel_name;
                std::string gks_kernel_ending = "_gkgs";
                if(fastest_result.kernel_name.compare(fastest_result.kernel_name.length() - gks_kernel_ending.length(),
                                            gks_kernel_ending.length(), gks_kernel_ending) == 0){
                    kernel_name_mock += "[" + std::to_string(fastest_result.gks) + "]";
                }
                printf("  fastest: [%d]%s, cost:%.3fms, tflops:%.3f(%.2f%%)\n",
                    fastest_id,
                    kernel_name_mock.c_str(),
                    fastest_result.duration_ms,
                    fastest_result.gflops / 1000,
                    fastest_result.efficiency);
            }
        }
    }else if(driver->driver_mode == driver_mode_heuristic){
        igemm_gtc_tunable_t selected_tunable = driver->heuristic_select_kernel(conv_args);
        if(run_only_kernel != IGEMM_RUN_ONLY_KERNEL_DEFAULT)
            if(run_only_kernel != driver->get_kernel_name(&selected_tunable)){
                printf("heuristic selected tunable not match your request\n");
                return;
            }

        result_t result = launch(&selected_tunable, 0);
        fastest_result = result;
        fastest_id = 0;
    }else{
        assert(0);
    }

    if(p_bcsv){
        //          time tflops efficiency kernel
        fprintf(p_bcsv, "%.3f,%.3f,%.2f%%,%s,",
            fastest_result.duration_ms, fastest_result.gflops/1000, fastest_result.efficiency, fastest_result.kernel_name.c_str());
        std::string conv_cmd = log_cmd(conv_args, driver_data_type, direction);
        fprintf(p_bcsv, "%s\n", conv_cmd.c_str());
        fflush(p_bcsv);
    }

    if(sleep_ms != 0)
        usleep(1000 * sleep_ms);
}

int main(int argc, char **argv) {
    char *hsaco = env_get_str("IGEMM_HSACO", IGEMM_HSACO);
    char *config_file = env_get_str("IGEMM_CONFIG_FILE", IGEMM_CONFIG_FILE);
    std::string run_only_kernel = env_get_str("IGEMM_RUN_ONLY_KERNEL", IGEMM_RUN_ONLY_KERNEL_DEFAULT);
    int warmup = env_get_int("IGEMM_WARMUP", WARMUP);
    int repeat = env_get_int("IGEMM_REPEAT", REPEAT);
    int assert_when_invalid = env_get_int("IGEMM_ASSERT_WHEN_INVALID", 0);
    int verbose     = env_get_int("IGEMM_VERBOSE", 0);
    int igemm_rand_int = env_get_int("IGEMM_RAND_INT", 0);
    int igemm_bench_csv = env_get_int("IGEMM_BENCH_CSV", 0);
    driver_mode_t driver_mode = static_cast<driver_mode_t>(env_get_int("IGEMM_MODE", 0));
    config_parser_t config_parser(config_file);
    auto content = config_parser.parse();
    //content.dump();
    FILE * p_bcsv = nullptr;
    if(igemm_bench_csv){
        p_bcsv = fopen ("bench_model.csv", "a");
        assert(p_bcsv);
    }

#ifdef USE_GPU_NAIVE_CONV
    char *gpu_naive_conv_hsaco = env_get_str("IGEMM_GPU_NAIVE_CONV_HSACO", IGEMM_GPU_NAIVE_CONV_HSACO);
    gpu_naive_conv_init(gpu_naive_conv_hsaco);
#endif

    auto tunables = igemm_gtc_tunable_from_config(content);
    if(tunables.size() == 0){
        printf("no tunable specified, may not work\n");
        return 0;
    }
    // printf("tunables:%d, hsaco:%s\n", tunables.size(), hsaco);

    hipModule_t module;
    HIP_CALL(hipModuleLoad(&module, hsaco));

    std::string base_arg = create_base_args(argc, argv);
    args_t conv_args = create_conv_args(argc, argv);
    // dump_arg(&conv_args);
    driverDataType_t driver_data_type;

    if(base_arg == "conv"){
        driver_data_type = driverFloat;
    }
    else if(base_arg == "convfp16"){
        driver_data_type = driverHalf;
    }
    else if(base_arg == "convbf16") {
        driver_data_type = driverBFloat16;
        exit(0);
    }
    else if(base_arg == "convint8") {
        driver_data_type = driverInt8;
    }
    else
        exit(0);

    size_t data_byte = get_data_byte(driver_data_type);

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
    std::string in_layout = conv_args.get_str("in_layout");
    std::string out_layout = conv_args.get_str("out_layout");
    std::string fil_layout = conv_args.get_str("fil_layout");

    int need_fwd = (forw == 0 ? 1 : (forw & 1 ? 1 : 0));
    int need_bwd = (forw == 0 ? 1 : (forw & 2 ? 1 : 0));
    int need_wrw = (forw == 0 ? 1 : (forw & 4 ? 1 : 0));

    assert(in_layout == out_layout && in_layout == fil_layout); // currently only support all layout is the same
    assert(in_layout == "NCHW" || in_layout == "NHWC"); // currently only support these layout
    assert((in_layout == "NCHW" && tunables[0].tensor_layout == "nchw") || 
            (in_layout == "NHWC" && tunables[0].tensor_layout == "nhwc"));  // check pairs

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


    void *host_input_dtype;
    void *host_weight_dtype;
    void *host_output_dtype;

    void *device_input_dtype;
    void *device_weight_dtype;
    void *device_output_dtype;
#if defined(USE_HALF) || defined(USE_INT8)
    host_input_dtype  = malloc(n * c * hi * wi * data_byte);
    host_weight_dtype = malloc(k * c * y * x * data_byte);
    host_output_dtype = malloc(n * k * ho * wo * data_byte);

    HIP_CALL(hipMalloc(&device_input_dtype, n * c * hi * wi * data_byte));
    HIP_CALL(hipMalloc(&device_weight_dtype, k * c * y * x * data_byte));
    HIP_CALL(hipMalloc(&device_output_dtype, n * k * ho * wo * data_byte));
#endif


    int need_verify = conv_args.get_int("verify");
    if(p_bcsv){
        //               N   C   H   W   K   Y   X   P   Q   U   V   L   J   G
        fprintf(p_bcsv, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,",
                         n,  c, hi, wi,  k,  y,  x, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, ngroups);
        //               GFLOP
        fprintf(p_bcsv, "%.2f,", get_theoritical_conv_flop(&conv_args)/1e9);
        fflush(p_bcsv);
    }

    if(driver_data_type == driverInt8)
        igemm_rand_int = 1;

    if (need_fwd){
        int fastest_id = -1;
        void *device_output_to_host = NULL;
        if (need_verify) {
            // gen rand
            if(!igemm_rand_int){
                gen_rand_vector<float, float>(host_input, static_cast<size_t>(n) * c * hi * wi, 0.0, 1.0);
                gen_rand_vector<float, float>(host_weight, static_cast<size_t>(k) * c * y * x, -0.5, 0.5);
            }else{
                gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * c * hi * wi, -5, 5);
                gen_rand_vector<float, int>(host_weight, static_cast<size_t>(k) * c * y * x, -5, 5);
            }

            //gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * c * hi * wi, 1, 1);
            //gen_rand_vector<float, int>(host_weight, static_cast<size_t>(k) * c * y * x, 1, 1);
            if(driver_data_type == driverHalf){
                tensor_copy<float16, float>(static_cast<float16*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * hi * wi);
                tensor_copy<float16, float>(static_cast<float16*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * y * x);
            }
            else if(driver_data_type == driverInt8){
                tensor_copy<int8_t, float>(static_cast<int8_t*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * hi * wi);
                tensor_copy<int8_t, float>(static_cast<int8_t*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * y * x);
            }

#ifdef USE_GPU_NAIVE_CONV
            HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                       static_cast<size_t>(k) * c * y * x * sizeof(float), hipMemcpyHostToDevice));

            if(in_layout == "NCHW")
                gpu_naive_conv_fwd_nchw_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else if(in_layout == "NHWC")
                gpu_naive_conv_fwd_nhwc_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else
                assert(0);
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipMemcpy(host_output, device_output,
                                   static_cast<size_t>(n) * k * ho * wo * sizeof(float),
                                   hipMemcpyDeviceToHost));
#else
            if(in_layout == "NCHW")
                naive_conv_fwd_nchw(host_input, host_weight, host_output, n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else if(in_layout == "NHWC")
                naive_conv_fwd_nhwc(host_input, host_weight, host_output, n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else
                assert(0);
#endif
            if(driver_data_type == driverHalf || driver_data_type == driverInt8){
                device_output_to_host = malloc((static_cast<size_t>(n) * k * ho * wo * data_byte + 3) / 4 * 4);
            }
            else{
                device_output_to_host = malloc(static_cast<size_t>(n) * k * ho * wo * sizeof(float));
            }
        }

        if(driver_data_type == driverFloat){
            HIP_CALL(hipMemcpy(device_input, host_input,
                        static_cast<size_t>(n) * c * hi * wi * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                        static_cast<size_t>(k) * c * y * x * data_byte, hipMemcpyHostToDevice));
        }else{
            HIP_CALL(hipMemcpy(device_input_dtype, host_input_dtype,
                        static_cast<size_t>(n) * c * hi * wi * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight_dtype, host_weight_dtype,
                        static_cast<size_t>(k) * c * y * x * data_byte, hipMemcpyHostToDevice));
        }

        igemm_fwd_gtc_t conv_fwd_driver(module, driver_mode, driver_data_type, warmup, repeat, verbose);

        auto fwd_pre = [&](){
            if (need_verify)
                HIP_CALL(hipMemset(device_output, 0, static_cast<size_t>(n) * k * ho * wo * data_byte));
        };

        auto fwd_post = [&](){
            if (need_verify) {
                double nrms = get_nrms("fwd", driver_data_type);
                bool is_valid = false;
                if(driver_data_type == driverFloat){
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output,
                                   static_cast<size_t>(n) * k * ho * wo * data_byte,
                                   hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_output, static_cast<float*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                }else{
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output_dtype,
                                   static_cast<size_t>(n) * k * ho * wo * data_byte,
                                   hipMemcpyDeviceToHost));
                    if(driver_data_type == driverHalf)
                        is_valid = valid_vector<float16>(host_output, static_cast<float16*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                    else if (driver_data_type == driverInt8)
                        is_valid = valid_vector<int8_t>(host_output, static_cast<int8_t*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                }
                printf(", valid:%s", is_valid ? "y" : "n");
                if(assert_when_invalid) assert(is_valid);
            }
        };

        if(driver_data_type == driverFloat)
            launch_conv_driver(&conv_fwd_driver, &conv_args, tunables, "fwd", driver_data_type, p_bcsv, device_input, device_weight, device_output, fwd_pre, fwd_post);
        else
            launch_conv_driver(&conv_fwd_driver, &conv_args, tunables, "fwd", driver_data_type, p_bcsv, device_input_dtype, device_weight_dtype, device_output_dtype, fwd_pre, fwd_post);

        if (need_verify)
            free(device_output_to_host);
    }

    if (need_bwd){
        void *device_input_to_host = NULL;
        result_t fastest_result_bwd;
        fastest_result_bwd.duration_ms = FLT_MAX;
        int fastest_id = -1;
        if (need_verify) {
            // gen rand
            if(!igemm_rand_int){
                gen_rand_vector<float, float>(host_output, static_cast<size_t>(n) * k * ho * wo, 0.0, 1.0);
                gen_rand_vector<float, float>(host_weight, static_cast<size_t>(k) * c * y * x, -0.5, 0.5);
            }
            else{
                gen_rand_vector<float, int>(host_output, static_cast<size_t>(n) * k * ho * wo, -5, 5);
                gen_rand_vector<float, int>(host_weight, static_cast<size_t>(k) * c * y * x, -5, 5);
            }
            gen_rand_vector<float, float>(host_input, static_cast<size_t>(n) * c * hi * wi, 999999., 9999999.);  // manually input value to a very large number
            // gen_rand_vector<float, int>(host_output, static_cast<size_t>(n) * k * ho * wo,1, 1);
            // gen_rand_vector<float, int>(host_weight, static_cast<size_t>(k) * c * y * x, 1, 1);

            if(driver_data_type == driverHalf){
                tensor_copy<float16, float>(static_cast<float16*>(host_output_dtype), host_output, static_cast<size_t>(n) * k * ho * wo);
                tensor_copy<float16, float>(static_cast<float16*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * y * x);
            }
            else if(driver_data_type == driverInt8){
                tensor_copy<int8_t, float>(static_cast<int8_t*>(host_output_dtype), host_output, static_cast<size_t>(n) * k * ho * wo);
                tensor_copy<int8_t, float>(static_cast<int8_t*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * y * x);
            }

#ifdef USE_GPU_NAIVE_CONV
            HIP_CALL(hipMemcpy(device_output, host_output,
                       static_cast<size_t>(n) * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                       static_cast<size_t>(k) * c * y * x * sizeof(float), hipMemcpyHostToDevice));
            if(in_layout == "NCHW")
                gpu_naive_conv_bwd_nchw_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else if(in_layout == "NHWC")
                gpu_naive_conv_bwd_nhwc_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else
                assert(0);
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipMemcpy(host_input, device_input,
                                   static_cast<size_t>(n) * c * hi * wi * sizeof(float),
                                   hipMemcpyDeviceToHost));
#else
            if(in_layout == "NCHW")
                naive_conv_bwd_nchw(host_input, host_weight, host_output, n,
                                         wi, hi, c, k, x, y, pad_w,
                                         pad_h, stride_w, stride_h, dilation_w, dilation_h, ngroups);
            else if(in_layout == "NHWC")
                naive_conv_bwd_nhwc(host_input, host_weight, host_output, n,
                                         wi, hi, c, k, x, y, pad_w,
                                         pad_h, stride_w, stride_h, dilation_w, dilation_h, ngroups);
            else
                assert(0);
#endif
            if(driver_data_type == driverHalf || driver_data_type == driverInt8){
                device_input_to_host = malloc((static_cast<size_t>(n) * c * hi * wi * data_byte + 3) / 4 * 4 );
            }
            else{
                device_input_to_host = malloc(static_cast<size_t>(n) * c * hi * wi * sizeof(float));
            }
            // printf("len:%d\n", n * c * hi * wi * sizeof(float) );
        }

        if(driver_data_type == driverFloat){
            HIP_CALL(hipMemcpy(device_output, host_output,
                        static_cast<size_t>(n) * k * ho * wo * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                        static_cast<size_t>(k) * c * y * x * data_byte, hipMemcpyHostToDevice));
        }else{
            HIP_CALL(hipMemcpy(device_output_dtype, host_output_dtype,
                        static_cast<size_t>(n) * k * ho * wo * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight_dtype, host_weight_dtype,
                        static_cast<size_t>(k) * c * y * x * data_byte, hipMemcpyHostToDevice));
        }

        igemm_bwd_gtc_t conv_bwd_driver(module, driver_mode, driver_data_type, warmup, repeat, verbose);

        auto bwd_pre = [&](){
            if (need_verify)
                HIP_CALL(hipMemset(device_input, 0x7f, static_cast<size_t>(n) * c * hi * wi * data_byte)); // 0x7f7f7f7f ~= 7.41e+28, a very large number
        };

        auto bwd_post = [&](){
            if (need_verify) {
                double nrms = get_nrms("bwd", driver_data_type);
                bool is_valid = false;
                if(driver_data_type == driverFloat){
                    HIP_CALL(hipMemcpy(device_input_to_host, device_input,
                                    static_cast<size_t>(n) * c * hi * wi * data_byte,
                                    hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_input, static_cast<float*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * hi * wi, nrms);
                } else {
                    HIP_CALL(hipMemcpy(device_input_to_host, device_input_dtype,
                                    static_cast<size_t>(n) * c * hi * wi * data_byte,
                                    hipMemcpyDeviceToHost));
                    if(driver_data_type == driverHalf)
                        is_valid = valid_vector<float16>(host_input, static_cast<float16*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * hi * wi, nrms);
                    else if (driver_data_type == driverInt8)
                        is_valid = valid_vector<int8_t>(host_input, static_cast<int8_t*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * hi * wi, nrms);
                }
                printf(", valid:%s", is_valid ? "y" : "n");
                if(assert_when_invalid) assert(is_valid);
            }
        };

        if(driver_data_type == driverFloat)
            launch_conv_driver(&conv_bwd_driver, &conv_args, tunables, "bwd",  driver_data_type, p_bcsv, device_input, device_weight, device_output, bwd_pre, bwd_post);
        else
            launch_conv_driver(&conv_bwd_driver, &conv_args, tunables, "bwd",  driver_data_type, p_bcsv, device_input_dtype, device_weight_dtype, device_output_dtype, bwd_pre, bwd_post);

        if (need_verify) 
            free(device_input_to_host);
    }

    if (need_wrw){
        float *device_weight_to_host = NULL;
        if (need_verify) {
            // gen rand
            if(!igemm_rand_int){
                gen_rand_vector<float, float>(host_input, static_cast<size_t>(n) * c * hi * wi, 0.0, 1.0);
                gen_rand_vector<float, float>(host_output, static_cast<size_t>(n) * k * ho * wo, -0.5, 0.5);
            }else{
                gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * c * hi * wi, -5, 5);
                gen_rand_vector<float, int>(host_output, static_cast<size_t>(n) * k * ho * wo, -5, 5);
            }
            //gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * k * hi * wi, -5, 5);
            //gen_rand_vector<float, int>(host_output, static_cast<size_t>(n) * k * ho * wo, 1, 1);
#ifdef USE_GPU_NAIVE_CONV
            HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_output, host_output,
                       static_cast<size_t>(n) * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));
            if(in_layout == "NCHW")
                gpu_naive_conv_wrw_nchw_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else if(in_layout == "NHWC")
                gpu_naive_conv_wrw_nhwc_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else
                assert(0);
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipMemcpy(host_weight, device_weight,
                                   static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x * sizeof(float),
                                   hipMemcpyDeviceToHost));
#else
            if(in_layout == "NCHW")
                naive_conv_wrw_nchw(host_input, host_weight, host_output, n,
                                         wi, hi, c, k, x, y, pad_w,
                                         pad_h, stride_w, stride_h, dilation_w, dilation_h, ngroups);
            else if(in_layout == "NHWC")
                naive_conv_wrw_nhwc(host_input, host_weight, host_output, n,
                                         wi, hi, c, k, x, y, pad_w,
                                         pad_h, stride_w, stride_h, dilation_w, dilation_h, ngroups);
            else
                assert(0);
#endif
            device_weight_to_host = (float *)malloc(static_cast<size_t>(k) * c * y * x * sizeof(float));
            // printf("len:%d\n", k * c * y * x * sizeof(float));
        }

        HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(device_output, host_output,
                       static_cast<size_t>(n) * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));



        igemm_wrw_gtc_t conv_wrw_driver(module, driver_mode, driver_data_type, warmup, repeat, verbose);
        
        auto wrw_pre = [&](){
            if (need_verify)
                HIP_CALL(hipMemset(device_weight, 0, static_cast<size_t>(k) * c * y * x * sizeof(float)));
        };

        auto wrw_post = [&](){
            if (need_verify) {
                double nrms = get_nrms("wrw", driver_data_type);
                HIP_CALL(hipMemcpy(device_weight_to_host, device_weight,
                                   static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x * sizeof(float),
                                   hipMemcpyDeviceToHost));
                bool is_valid = valid_vector(host_weight, device_weight_to_host,
                                            static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x, nrms);
                printf(", valid:%s", is_valid ? "y" : "n");
                if(assert_when_invalid) assert(is_valid);
            }
        };

        launch_conv_driver(&conv_wrw_driver, &conv_args, tunables, "wrw",  driver_data_type, p_bcsv, device_input, device_weight, device_output, wrw_pre, wrw_post);

        if (need_verify) 
            free(device_weight_to_host);
    }

    if(p_bcsv)
        fclose(p_bcsv);

    free(host_input);
    free(host_weight);
    free(host_output);

    hipFree(device_input);
    hipFree(device_weight);
    hipFree(device_output);
}
