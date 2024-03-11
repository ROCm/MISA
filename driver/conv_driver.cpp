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
#include "tensor_transpose.h"
#include "tensor_copy_cpu.h"
#include "tensor_validation_cpu.h"
#include "igemm_gtc_base.h"
#include "igemm_fwd_gtc_driver.h"
#include "igemm_bwd_gtc_driver.h"
#include "igemm_wrw_gtc_driver.h"

static inline double theoritical_gflops(double sclk_ghz, size_t cu,
                                             size_t simd) {
    return 2 * sclk_ghz * cu * simd;
}
static inline double
theoritical_fp32_conv_flop(size_t n, size_t c, size_t di, size_t hi, size_t wi, size_t k,
                           size_t z, size_t y, size_t x, size_t stride_d, size_t stride_h, size_t stride_w,
                           size_t dilation_d, size_t dilation_h, size_t dilation_w, size_t pad_d,size_t pad_h,
                           size_t pad_w, size_t ngroups) {
    size_t do_ = conv_out_size(di, pad_d, dilation_d, z, stride_d);
    size_t ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    size_t wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

    double flop = (double)n * c * do_ * ho * wo * k * z * y * x * 2 / ngroups;
    return flop;
}
static inline double
measured_fp32_conv_gflops(double time_ms, size_t n, size_t c, size_t di, size_t hi,
                          size_t wi, size_t k, size_t z, size_t y, size_t x,
                          size_t stride_d, size_t stride_h, size_t stride_w, size_t dilation_d, size_t dilation_h,
                          size_t dilation_w, size_t pad_d, size_t pad_h, size_t pad_w, size_t ngroups) {
    double flop =
        theoritical_fp32_conv_flop(n, c, di, hi, wi, k, z, y, x, stride_d, stride_h, stride_w,
                                   dilation_d, dilation_h, dilation_w, pad_d, pad_h, pad_w, ngroups);
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
    std::string in_layout = conv_args->get_str("in_layout");

    int di = 1;
    int stride_d = 1;
    int dilation_d = 1;
    int pad_d = 0;
    int z = 1;
    int do_ = 1;
    if(in_layout == "NDHWC"){
        di = conv_args->get_int("in_d");
        stride_d = conv_args->get_int("conv_stride_d");
        dilation_d = conv_args->get_int("dilation_d");
        pad_d = conv_args->get_int("pad_d");
        z = conv_args->get_int("fil_d");
        do_ = conv_out_size(di, pad_d, dilation_d, z, stride_d);
    }

    return (double)n * c * do_ * ho * wo * k * z * y * x * 2 / ngroups;
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
        if(gcn_arch == 908 || gcn_arch == 910)
            fp_factor = 4;  // xdlops
        else
            fp_factor = 2;  // dlops
        if(gcn_arch >= 1000)
            fp_factor = 2;
    }
    if(data_type == driverInt8){
        if(gcn_arch == 908 || gcn_arch == 910)
            fp_factor = 4;  // xdlops
        else
            fp_factor = 4;  // dlops
    }
    if(data_type == driverInt4){
        if(gcn_arch >= 1000)
            fp_factor = 8;  // xdlops
    }
    // else if(data_type == driverInt8){
    //     if(gcn_arch == 908)
    //     fp_factor = 4;
    // }

    if(gcn_arch == 908 || gcn_arch == 910){
        num_simd = 4 * 32 ; // 4x miSIMD, 32x mac unit
    }

    return theoritical_gflops(((double)sclk_mhz) / 1000.0, num_cu, num_simd * fp_factor);
}

#ifndef IGEMM_HSACO
#define IGEMM_HSACO "igemm_gtc.hsaco"
#endif

#ifndef IGEMM_TENSOR_CAST_HSACO
#define IGEMM_TENSOR_CAST_HSACO "igemm_gtc_tensor_cast.hsaco"
#endif

#ifndef IGEMM_CONFIG_FILE
#define IGEMM_CONFIG_FILE "igemm_gtc.config"
#endif

#define IGEMM_RUN_ONLY_KERNEL_DEFAULT "off"

#define WARMUP 3
#define REPEAT 8
#define SCLK_MHZ 1283

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

#if DEBUG_ASM_PRINT
void dump_asm_print(float* host_print){
    // dump values
    return;
}
#endif

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
    int ngroups = arg->get_int("group_count");
    std::string in_layout = arg->get_str("in_layout");

    int di = 1;
    int stride_d = 1;
    int dilation_d = 1;
    int pad_d = 0;
    int z = 1;
    int do_ = 1;
    if(in_layout == "NDHWC"){
        di = arg->get_int("in_d");
        stride_d = arg->get_int("conv_stride_d");
        dilation_d = arg->get_int("dilation_d");
        pad_d = arg->get_int("pad_d");
        z = arg->get_int("fil_d");
        do_ = conv_out_size(di, pad_d, dilation_d, z, stride_d);
    }

    printf("n:%d, c:%d, d:%d, h:%d, w:%d, k:%d, z:%d, y:%d, x:%d, sz:%d, sy:%d, sx:%d, dz:%d, dy:%d, "
           "dx:%d, pz:%d, py:%d, px:%d, do:%d, ho:%d, wo:%d, group:%d\n",
           n, c, di, hi, wi, k, z, y, x, stride_d, stride_h, stride_w, dilation_d, dilation_h, dilation_w,
           pad_d, pad_h, pad_w, do_, ho, wo, ngroups);
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

    int di = 1;
    int stride_d = 1;
    int dilation_d = 1;
    int pad_d = 0;
    int z = 1;
    int do_ = 1;
    if(in_layout == "NDHWC"){
        di = arg->get_int("in_d");
        stride_d = arg->get_int("conv_stride_d");
        dilation_d = arg->get_int("dilation_d");
        pad_d = arg->get_int("pad_d");
        z = arg->get_int("fil_d");
        do_ = conv_out_size(di, pad_d, dilation_d, z, stride_d);
    }

    std::stringstream ss;
    if(driver_data_type == driverHalf)
    {
        ss << "convfp16";
    }
    else if(driver_data_type == driverFloat)
    {
        ss << "conv";
    }

    //  n  c  !  H  W  k  @  y  x  $  p  q  #  u  v  ^  l  j  g
    ss << " -n " << n
        << " -c " << c
        << " -! " << di
        << " -H " << hi
        << " -W " << wi
        << " -k " << k
        << " -@ " << z
        << " -y " << y
        << " -x " << x
        << " -$ " << pad_d
        << " -p " << pad_h
        << " -q " << pad_w
        << " -# " << stride_d
        << " -u " << stride_h
        << " -v " << stride_w
        << " -^ " << dilation_d
        << " -l " << dilation_h
        << " -j " << dilation_w;

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

template<typename driver_t>
std::string get_tiling_string(driver_t * driver, const args_t *conv_args)
{
    int hi = conv_args->get_int("in_h");
    int wi = conv_args->get_int("in_w");

    int stride_h = conv_args->get_int("conv_stride_h");
    int stride_w = conv_args->get_int("conv_stride_w");
    int dilation_h = conv_args->get_int("dilation_h");
    int dilation_w = conv_args->get_int("dilation_w");
    int pad_h = conv_args->get_int("pad_h");
    int pad_w = conv_args->get_int("pad_w");
    int y = conv_args->get_int("fil_h");
    int x = conv_args->get_int("fil_w");

    int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

    igemm_spatial_tiling_t tiling = driver->get_spatial_tiling(conv_args);
    if((tiling.tile_h == 0 && tiling.tile_w == 0) || 
        (tiling.tile_h == ho && tiling.tile_w == wo))
        return "";
    else{
        return std::string("[") + std::to_string(tiling.tile_w) + "x" + std::to_string(tiling.tile_h) + "]";
    }
}

template<typename driver_t, typename pre_func_t, typename post_func_t>
void launch_conv_driver(driver_t * driver, const args_t *conv_args, const std::vector<igemm_gtc_tunable_t> & tunables, std::string direction,
                    driverDataType_t driver_data_type, FILE * p_bcsv,
                    void* device_input, void* device_weight, void* device_output,
                    pre_func_t && pre_func, post_func_t && post_func, void* device_print=nullptr)
{
    int sclk_mhz = env_get_int("IGEMM_SCLK_MHZ", SCLK_MHZ);
    std::string run_only_kernel = env_get_str("IGEMM_RUN_ONLY_KERNEL", IGEMM_RUN_ONLY_KERNEL_DEFAULT);
    int log_fastest_config = env_get_int("IGEMM_LOG_FASTEST_CONFIG", 0);
    int sleep_ms = env_get_int("IGEMM_SLEEP_MS", 0);
    int dump_gmap = env_get_int("IGEMM_DUMP_GMAP", 0);
    int gks_iterative = env_get_int("IGEMM_GKS_ITERATIVE", 0);
    int max_mpb = env_get_int("IGEMM_MAX_MPB", -1);
    int max_npb = env_get_int("IGEMM_MAX_NPB", -1);
    int max_kpb = env_get_int("IGEMM_MAX_KPB", -1);
    int max_gks = env_get_int("IGEMM_MAX_GKS", -1);
    int silent_not_applicable_level0 = env_get_int("IGEMM_SILENT_NA_L0", 1);  // ignore kernel that has different direction & layout
    std::string in_layout = conv_args->get_str("in_layout");
    std::string fil_layout = conv_args->get_str("fil_layout");

    double theo_conv_flop  = get_theoritical_conv_flop(conv_args);
    double theo_gpu_gflops = get_theoritical_gpu_gflops(sclk_mhz, driver->data_type);

    auto launch = [&](const igemm_gtc_tunable_t * tunable, int index, int current_gks, bool is_tunable_predicted = false) -> result_t {
        igemm_gtc_tunable_t predicted_tunable;
        const igemm_gtc_tunable_t * current_tunable = tunable;
        if(is_tunable_predicted){
            predicted_tunable = *tunable;
            // in prediction, the gks will be 0, 1, 2... if tunable support gks, other wise it is -1.
            // here we restore the gemm_k_global_split inside the tunable
            predicted_tunable.gemm_k_global_split = current_gks >= 0 ? 1 : 0;
            current_tunable = &predicted_tunable;
        }
        if(run_only_kernel != IGEMM_RUN_ONLY_KERNEL_DEFAULT){
            if(run_only_kernel != driver->get_kernel_name(current_tunable))
                {result_t result; result.return_code = -2; return result;}
        }
        if(silent_not_applicable_level0){
            // direction
            if(direction != current_tunable->direction)
                {result_t result; result.return_code = -2; return result;}

            // layout
            if(in_layout == "NCHW"){
                if(current_tunable->tensor_layout != "nchw")
                    {result_t result; result.return_code = -2; return result;}
            }else if(in_layout == "NHWC"){
                if(current_tunable->tensor_layout != "nhwc")
                    {result_t result; result.return_code = -2; return result;}
            }else if(in_layout == "NDHWC"){
                if(current_tunable->tensor_layout != "ndhwc")
                    {result_t result; result.return_code = -2; return result;}
            }else if(in_layout == "NCHWC"){
                if(current_tunable->tensor_layout.compare(0, 5, "nchwc") != 0)
                    {result_t result; result.return_code = -2; return result;}
                auto wei_layout_config = current_tunable->tensor_layout.substr(6);
                if((fil_layout == "NCHWC" && wei_layout_config != "kcyxc") || 
                    (fil_layout == "CHWNC" && wei_layout_config != "cyxkc"))
                    {result_t result; result.return_code = -2; return result;}
            }
        }

        printf("[%s:%2d] %s", direction.c_str(), index, driver->get_kernel_name(current_tunable).c_str());
        fflush(stdout);

        pre_func();

        result_t result = driver->run(conv_args, current_tunable, device_input, device_weight, device_output, current_gks, device_print);

        std::string gks_string = "";
        if(current_tunable->gemm_k_global_split){
            gks_string = "[" + std::to_string(result.gks) + "]";
        }
        printf("%s", gks_string.c_str());
        std::string tiling_string = get_tiling_string(driver, conv_args);
        printf("%s", tiling_string.c_str());

        printf(", ");
        fflush(stdout);

        if (result.return_code != 0){
            printf("not applicable\n");
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
            gmap_dump(conv_args, current_tunable, result.gks);
        return result;
    };

    auto need_skip_due_to_macro_tile_boundary = [&](const igemm_gtc_tunable_t * tunable){
        if(max_mpb != -1 && tunable->gemm_m_per_block > max_mpb)
            return true;
        if(max_npb != -1 && tunable->gemm_n_per_block > max_npb)
            return true;
        if(max_kpb != -1 && tunable->gemm_k_per_block > max_kpb)
            return true;
        return false;
    };

    driver->set_block_tile_boundary(max_mpb, max_npb, max_kpb, max_gks);
    result_t fastest_result;
    fastest_result.duration_ms = FLT_MAX;
    int fastest_id = -1;
    if(driver->driver_mode == driver_mode_normal){
        int unique_index = 0;
        std::vector<igemm_gtc_tunable_t> unique_tunables;
        for(int i=0; i<tunables.size(); i++){
            if(need_skip_due_to_macro_tile_boundary(&tunables[i]))
                continue;
            if(gks_iterative){
                if(tunables[i].gemm_k_global_split != 0){
                    std::vector<int> gks_list = driver->get_gks_list(conv_args, &tunables[i]);
                    for(int gks : gks_list){
                        result_t result = launch(&tunables[i], unique_index, gks);
                        if(result.return_code == -2) continue;
                        unique_tunables.push_back(tunables[i]);
                        unique_tunables.back().gemm_k_global_split = gks;
                        if(result.duration_ms < fastest_result.duration_ms){
                            fastest_result = result;
                            fastest_id = unique_index;
                        }
                        unique_index++;
                    }
                }else{
                    result_t result = launch(&tunables[i], unique_index, 0);
                    if(result.return_code == -2) continue;
                    unique_tunables.push_back(tunables[i]);
                    unique_tunables.back().gemm_k_global_split = 0;
                    if(result.duration_ms < fastest_result.duration_ms){
                        fastest_result = result;
                        fastest_id = unique_index;
                    }
                    unique_index++;
                }
            }
            else{
                result_t result = launch(&tunables[i], unique_index, -1);
                if(result.return_code == -2) continue;
                unique_tunables.push_back(tunables[i]);
                unique_tunables.back().gemm_k_global_split = result.gks;
                if(result.duration_ms < fastest_result.duration_ms){
                    fastest_result = result;
                    fastest_id = unique_index;
                }
                unique_index++;
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

        result_t result = launch(&selected_tunable, 0, -1);
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

#define COMPARE_GPU_NAIVE_NDHWC_NHWC 0

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
    auto unexpanded_content = config_parser.parse();
    auto content = igemm_try_expand_tunable_content(unexpanded_content);
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
#ifndef IGEMM_SPLIT_KERNEL
    HIP_CALL(hipModuleLoad(&module, hsaco));
#endif

    std::string base_arg = create_base_args(argc, argv);
    args_t conv_args = create_conv_args(argc, argv);
    // dump_arg(&conv_args);
    driverDataType_t driver_data_type;
    auto vec_found = base_arg.find("x");
    std::string base_type = base_arg.substr(0, vec_found);
    int vector_c = find_vector_c_from_base_arg(base_arg);
    vector_c = env_get_int("VECTOR_C", vector_c);

    if(base_type == "conv"){
        driver_data_type = driverFloat;
    }
    else if(base_type == "convfp16"){
        driver_data_type = driverHalf;
    }
    else if(base_type == "convbfp16") {
        driver_data_type = driverBFloat16;
    }
    else if(base_type == "convint8") {
        driver_data_type = driverInt8;
    }
    else if(base_type == "convint4") {
        driver_data_type = driverInt4;
    }
    else{
        printf("invalid base type:%s\n", base_type.c_str());
        exit(0);
    }

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

    int di = 1;
    int stride_d = 1;
    int dilation_d = 1;
    int pad_d = 0;
    int z = 1;
    int do_ = 1;
    if (in_layout == "NDHWC"){
        di = conv_args.get_int("in_d");
        stride_d = conv_args.get_int("conv_stride_d");
        dilation_d = conv_args.get_int("dilation_d");
        pad_d = conv_args.get_int("pad_d");
        z = conv_args.get_int("fil_d");
        do_ = conv_out_size(di, pad_d, dilation_d, z, stride_d);
    }

    int need_fwd = (forw == 0 ? 1 : (forw & 1 ? 1 : 0));
    int need_bwd = (forw == 0 ? 1 : (forw & 2 ? 1 : 0));
    int need_wrw = (forw == 0 ? 1 : (forw & 4 ? 1 : 0));

    //assert(in_layout == out_layout && in_layout == fil_layout); // currently only support all layout is the same
    assert(in_layout == out_layout); 
    assert(in_layout == "NCHW" || in_layout == "NHWC" || in_layout == "NDHWC" || in_layout == "NCHWC"); // currently only support these layout
    assert((in_layout == "NCHW" && tunables[0].tensor_layout == "nchw") || 
           (in_layout == "NHWC" && tunables[0].tensor_layout == "nhwc") ||
           (in_layout == "NDHWC" && tunables[0].tensor_layout == "ndhwc") ||
           (in_layout == "NCHWC" && tunables[0].tensor_layout.compare(0, 5, "nchwc") == 0));  // check pairs

    // init host side
    float *host_input = (float *)malloc(static_cast<size_t>(n) * c * di * hi * wi * sizeof(float));
    float *host_weight = (float *)malloc(static_cast<size_t>(k) * c * z * y * x * sizeof(float));
    float *host_output = (float *)malloc(static_cast<size_t>(n) * k * do_ * ho * wo * sizeof(float));

    float *device_input;
    float *device_weight;
    float *device_output;

    HIP_CALL(hipMalloc(&device_input, static_cast<size_t>(n) * c * di * hi * wi * sizeof(float)));
    HIP_CALL(hipMalloc(&device_weight, static_cast<size_t>(k) * c * z * y * x * sizeof(float)));
    HIP_CALL(hipMalloc(&device_output, static_cast<size_t>(n) * k * do_ * ho * wo * sizeof(float)));

#if COMPARE_GPU_NAIVE_NDHWC_NHWC    // when depth = 1
    float *host_output2 = (float *)malloc(static_cast<size_t>(n) * k * do_ * ho * wo * sizeof(float));
    float *device_output2;
    HIP_CALL(hipMalloc(&device_output2, static_cast<size_t>(n) * k * do_ * ho * wo * sizeof(float)));
#endif

    float *device_print = nullptr;
#if DEBUG_ASM_PRINT
    float *host_print = nullptr;
    size_t print_size = 256*32*sizeof(float);
    host_print = (float *)malloc(print_size);
    HIP_CALL(hipMalloc(&device_print, print_size));
    HIP_CALL(hipMemset(device_print, 0, print_size));
#endif

    void *host_input_dtype;
    void *host_weight_dtype;
    void *host_output_dtype;

    void *device_input_dtype;
    void *device_weight_dtype;
    void *device_output_dtype;
#if defined(USE_HALF) || defined(USE_INT8) || defined(USE_BF16) || defined(USE_INT4)
    host_input_dtype  = malloc(n * c * di * hi * wi * data_byte);
    host_weight_dtype = malloc(k * c * z * y * x * data_byte);
    host_output_dtype = malloc(n * k * do_ * ho * wo * data_byte);

    HIP_CALL(hipMalloc(&device_input_dtype, n * c * di * hi * wi * data_byte));
    HIP_CALL(hipMalloc(&device_weight_dtype, k * c * z * y * x * data_byte));
    HIP_CALL(hipMalloc(&device_output_dtype, n * k * do_ * ho * wo * data_byte));
#endif


    int need_verify = conv_args.get_int("verify");
    if(p_bcsv){
        //               n  c  !  H  W  k  @  y  x  $  p  q  #  u  v  ^  l  j  g
        fprintf(p_bcsv, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,",
                         n, c, di,hi,wi,k, z, y, x, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, dilation_d, dilation_h, dilation_w, ngroups);
        //               GFLOP
        fprintf(p_bcsv, "%.2f,", get_theoritical_conv_flop(&conv_args)/1e9);
        fflush(p_bcsv);
    }

    if(driver_data_type == driverInt8 || driver_data_type == driverInt4)
        igemm_rand_int = 1;

    // launch tensor cast module
    hipModule_t module_tensor_cast;
    char *hsaco_tensor_cast = env_get_str("IGEMM_TENSOR_CAST_HSACO", IGEMM_TENSOR_CAST_HSACO);
    HIP_CALL(hipModuleLoad(&module_tensor_cast, hsaco_tensor_cast));

    if (need_fwd){
        int fastest_id = -1;
        void *device_output_to_host = NULL;
        if (need_verify) {
            // gen rand
            if(!igemm_rand_int){
                gen_rand_vector<float, float>(host_input, static_cast<size_t>(n) * c * di * hi * wi, 0.0, 1.0);
                gen_rand_vector<float, float>(host_weight, static_cast<size_t>(k) * c * z * y * x, -0.5, 0.5);
            }else{
                gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * c * di * hi * wi, -5, 5);
                gen_rand_vector<float, int>(host_weight, static_cast<size_t>(k) * c * z * y * x, -5, 5);
            }

            //gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * c * di * hi * wi, -5, 5);
            //gen_rand_vector<float, int>(host_weight, static_cast<size_t>(k) * c * z * y * x, -5, 5);
            //gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * c * di * hi * wi, 1, 1);
            //gen_rand_vector<float, int>(host_weight, static_cast<size_t>(k) * c * z * y * x, 1, 1);
            if(driver_data_type == driverHalf){
                tensor_copy<float16, float>(static_cast<float16*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * di * hi * wi);
                tensor_copy<float16, float>(static_cast<float16*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * z * y * x);
            }
            else if(driver_data_type == driverBFloat16){
                tensor_copy<bfloat16, float>(static_cast<bfloat16*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * di * hi * wi);
                tensor_copy<bfloat16, float>(static_cast<bfloat16*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * z * y * x);
            }
            else if(driver_data_type == driverInt8){
                tensor_copy<int8_t, float>(static_cast<int8_t*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * di * hi * wi);
                tensor_copy<int8_t, float>(static_cast<int8_t*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * z * y * x);
            }
            else if(driver_data_type == driverInt4)
            {
                tensor_copy<int4x2_t, float>(static_cast<int4x2_t*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * di * hi * wi);
                tensor_copy<int4x2_t, float>(static_cast<int4x2_t*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * z * y * x);
            }

#ifdef USE_GPU_NAIVE_CONV
            HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * di * hi * wi * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                       static_cast<size_t>(k) * c * z * y * x * sizeof(float), hipMemcpyHostToDevice));

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
            else if(in_layout == "NDHWC"){
#if COMPARE_GPU_NAIVE_NDHWC_NHWC    // when depth = 1
                gpu_naive_conv_fwd_nhwc_fp32(device_input, device_weight, device_output2,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
                HIP_CALL(hipDeviceSynchronize());
                HIP_CALL(hipMemcpy(host_output2, device_output2,
                                    static_cast<size_t>(n) * k * do_ * ho * wo * sizeof(float),
                                    hipMemcpyDeviceToHost));
#endif
                gpu_naive_conv_fwd_ndhwc_fp32(device_input, device_weight, device_output,
                                n, wi, hi, di, c,
                                k, x, y, z, pad_w, pad_h, pad_d, stride_w, stride_h, stride_d,
                                dilation_w, dilation_h, dilation_d, ngroups);
            }
            else if(in_layout == "NCHWC"){
                if(((c / ngroups) % vector_c != 0) || ((k / ngroups) % vector_c != 0)){
                    dump_arg(&conv_args);
                    printf("can't support c:%d k:%d with vec_c:%d\n", c, k, vector_c);
                    if(p_bcsv)
                    {
                        fprintf(p_bcsv, "\n");
                        fflush(p_bcsv);
                    }
                    exit(-1);
                }
                float* aux_in = (float*)malloc(static_cast<size_t>(n) * c * di * hi * wi * sizeof(float));
                float* aux_wei = (float*)malloc(static_cast<size_t>(k) * c * z * y * x * sizeof(float));
                float* aux_out = (float*)malloc(static_cast<size_t>(n) * k * do_ * ho * wo * sizeof(float));

                tensor_transpose_nchwc_2_nchw<float*>(aux_in, host_input, n, c, hi, wi, vector_c);
                for(int i_groups = 0; i_groups < ngroups; i_groups++){
                    int group_offset = i_groups * (k / ngroups) * (c / ngroups) * y * x;
                    if(fil_layout == "CHWNC")
                        tensor_transpose_chwnc_2_nchw<float*>(aux_wei + group_offset, host_weight + group_offset, k / ngroups, c / ngroups, y, x, vector_c);
                    else if(fil_layout == "NCHWC")
                        tensor_transpose_nchwc_2_nchw<float*>(aux_wei + group_offset, host_weight + group_offset, k / ngroups, c / ngroups, y, x, vector_c);
                }

                if(env_get_int("IGEMM_CHECK_TRNASPOSE", 0)){
                    float* aux_wei_check = (float*)malloc(static_cast<size_t>(k) * c * z * y * x * sizeof(float));
                    float* aux_in_check = (float*)malloc(static_cast<size_t>(n) * c * di * hi * wi * sizeof(float));

                    tensor_transpose_nchw_2_nchwc<float*>(aux_in_check, aux_in, n, c, hi, wi, vector_c);
                    if(fil_layout == "CHWNC")
                        tensor_transpose_nchw_2_chwnc<float*>(aux_wei_check, aux_wei, k, c, y, x, vector_c);
                    else if(fil_layout == "NCHWC")
                        tensor_transpose_nchw_2_nchwc<float*>(aux_wei_check, aux_wei, k, c, y, x, vector_c);

                    double transpose_nrms = get_nrms("fwd", driver_data_type);
                    valid_vector<float>(host_input, aux_in_check, static_cast<size_t>(n) * c * di * hi * wi, transpose_nrms);
                    valid_vector<float>(host_weight, aux_wei_check, static_cast<size_t>(k) * c * z * y * x, transpose_nrms);

                    free(aux_in_check);
                    free(aux_wei_check);
                }
                
                HIP_CALL(hipMemcpy(device_input, aux_in,
                       static_cast<size_t>(n) * c * di * hi * wi * sizeof(float), hipMemcpyHostToDevice));
                HIP_CALL(hipMemcpy(device_weight, aux_wei,
                       static_cast<size_t>(k) * c * z * y * x * sizeof(float), hipMemcpyHostToDevice));

                gpu_naive_conv_fwd_nchw_fp32(device_input, device_weight, device_output,
                        n, wi, hi, c,
                        k, x, y, pad_w, pad_h, stride_w, stride_h,
                        dilation_w, dilation_h, ngroups);

                HIP_CALL(hipMemcpy(host_output, device_output,
                                   static_cast<size_t>(n) * k * do_ * ho * wo * sizeof(float),
                                   hipMemcpyDeviceToHost));

                tensor_transpose_nchw_2_nchwc<float*>(aux_out, host_output, n, k, ho, wo, vector_c);

                HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * di * hi * wi * sizeof(float), hipMemcpyHostToDevice));
                HIP_CALL(hipMemcpy(device_weight, host_weight,
                       static_cast<size_t>(k) * c * z * y * x * sizeof(float), hipMemcpyHostToDevice));

                HIP_CALL(hipMemcpy(device_output, aux_out,
                       static_cast<size_t>(n) * k * do_ * ho * wo * sizeof(float), hipMemcpyHostToDevice));

                free(aux_in);
                free(aux_wei);
                free(aux_out);
                // exit(1);
            }
            else
                assert(0);
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipMemcpy(host_output, device_output,
                                   static_cast<size_t>(n) * k * do_ * ho * wo * sizeof(float),
                                   hipMemcpyDeviceToHost));
#if COMPARE_GPU_NAIVE_NDHWC_NHWC    // when depth = 1
            double nrms = get_nrms("fwd", driver_data_type);
            bool is_valid = valid_vector<float>(host_output, static_cast<float*>(host_output2),
                                    static_cast<size_t>(n) * k * do_ * ho * wo, nrms);
            printf("COMPARE_GPU_NAIVE_NDHWC_NHWC valid:%s /n", is_valid ? "y" : "n");
            free(host_output2);
            hipFree(device_output2);
#endif
#else
            if(in_layout == "NCHW")
                naive_conv_fwd_nchw(host_input, host_weight, host_output, n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else if(in_layout == "NHWC")
                naive_conv_fwd_nhwc(host_input, host_weight, host_output, n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else if(in_layout == "NDHWC")
                naive_conv_fwd_ndhwc(host_input, host_weight, host_output, n, wi, hi, di, c,
                                k, x, y, z, pad_w, pad_h, pad_d, stride_w, stride_h, stride_d,
                                dilation_w, dilation_h, dilation_d, ngroups);
            else
                assert(0);
#endif
            if(driver_data_type != driverHalf){
                device_output_to_host = malloc((static_cast<size_t>(n) * k * do_ * ho * wo * data_byte + 3) / 4 * 4);
            }
            else{
                device_output_to_host = malloc(static_cast<size_t>(n) * k * do_ * ho * wo * sizeof(float));
            }
        }

        if(driver_data_type == driverFloat)
        {
            HIP_CALL(hipMemcpy(device_input, host_input,
                        static_cast<size_t>(n) * c * di * hi * wi * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                        static_cast<size_t>(k) * c * z * y * x * data_byte, hipMemcpyHostToDevice));
        }
        else if(driver_data_type == driverInt4)
        {
            HIP_CALL(hipMemcpy(device_input_dtype, host_input_dtype,
                        static_cast<size_t>(n) * c * di * hi * wi / 2, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight_dtype, host_weight_dtype,
                        static_cast<size_t>(k) * c * z * y * x / 2, hipMemcpyHostToDevice));
        }
        else
        {
            HIP_CALL(hipMemcpy(device_input_dtype, host_input_dtype,
                        static_cast<size_t>(n) * c * di * hi * wi * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight_dtype, host_weight_dtype,
                        static_cast<size_t>(k) * c * z * y * x * data_byte, hipMemcpyHostToDevice));
        }

        igemm_fwd_gtc_t conv_fwd_driver(module_tensor_cast, module, driver_mode, driver_data_type, warmup, repeat, verbose);
        conv_fwd_driver.set_vector_c(vector_c);

        auto fwd_pre = [&](){
            if (need_verify)
                HIP_CALL(hipMemset(driver_data_type == driverFloat ? device_output : device_output_dtype,
                    0, static_cast<size_t>(n) * k * do_ * ho * wo * data_byte));
        };

        auto fwd_post = [&](){
#if DEBUG_ASM_PRINT
            HIP_CALL(hipMemcpy(host_print, device_print, print_size, hipMemcpyDeviceToHost));
            dump_asm_print(host_print);
#endif
            if (need_verify) {
                double nrms = get_nrms("fwd", driver_data_type);
                bool is_valid = false;
                if(driver_data_type == driverFloat){
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output,
                                   static_cast<size_t>(n) * k * do_ * ho * wo * data_byte,
                                   hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_output, static_cast<float*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * do_ * ho * wo, nrms);
                }else{
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output_dtype,
                                   static_cast<size_t>(n) * k * do_ * ho * wo * data_byte,
                                   hipMemcpyDeviceToHost));
                    if(driver_data_type == driverHalf)
                        is_valid = valid_vector<float16>(host_output, static_cast<float16*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * do_ * ho * wo, nrms);
                    else if(driver_data_type == driverBFloat16)
                        is_valid = valid_vector<bfloat16>(host_output, static_cast<bfloat16*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * do_ * ho * wo, nrms);
                    else if (driver_data_type == driverInt8)
                        is_valid = valid_vector<int8_t>(host_output, static_cast<int8_t*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * do_ * ho * wo, nrms);
                    else if (driver_data_type == driverInt4)
                        is_valid = valid_vector<int4x2_t>(host_output, static_cast<int4x2_t*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * do_ * ho * wo, nrms);
                }
                printf(", valid:%s", is_valid ? "y" : "n");
                if(assert_when_invalid) assert(is_valid);
            }
        };

        if(driver_data_type == driverFloat)
            launch_conv_driver(&conv_fwd_driver, &conv_args, tunables, "fwd", driver_data_type, p_bcsv, device_input, device_weight, device_output, fwd_pre, fwd_post);
        else
            launch_conv_driver(&conv_fwd_driver, &conv_args, tunables, "fwd", driver_data_type, p_bcsv,
                device_input_dtype, device_weight_dtype, device_output_dtype, fwd_pre, fwd_post, device_print);

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
                gen_rand_vector<float, float>(host_output, static_cast<size_t>(n) * k * do_ * ho * wo, 0.0, 1.0);
                gen_rand_vector<float, float>(host_weight, static_cast<size_t>(k) * c * z * y * x, -0.5, 0.5);
            }
            else{
                gen_rand_vector<float, int>(host_output, static_cast<size_t>(n) * k * do_ * ho * wo, -5, 5);
                gen_rand_vector<float, int>(host_weight, static_cast<size_t>(k) * c * z * y * x, -5, 5);
            }
            gen_rand_vector<float, float>(host_input, static_cast<size_t>(n) * c * di * hi * wi, 999999., 9999999.);  // manually input value to a very large number
            // gen_rand_vector<float, int>(host_output, static_cast<size_t>(n) * k * do_ * ho * wo,1, 1);
            // gen_rand_vector<float, int>(host_weight, static_cast<size_t>(k) * c * z * y * x, 1, 1);

            if(driver_data_type == driverHalf){
                tensor_copy<float16, float>(static_cast<float16*>(host_output_dtype), host_output, static_cast<size_t>(n) * k * do_ * ho * wo);
                tensor_copy<float16, float>(static_cast<float16*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * z * y * x);
            }
            else if(driver_data_type == driverBFloat16){
                tensor_copy<bfloat16, float>(static_cast<bfloat16*>(host_output_dtype), host_output, static_cast<size_t>(n) * k * do_ * ho * wo);
                tensor_copy<bfloat16, float>(static_cast<bfloat16*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * z * y * x);
            }
            else if(driver_data_type == driverInt8){
                tensor_copy<int8_t, float>(static_cast<int8_t*>(host_output_dtype), host_output, static_cast<size_t>(n) * k * do_ * ho * wo);
                tensor_copy<int8_t, float>(static_cast<int8_t*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * z * y * x);
            }

#ifdef USE_GPU_NAIVE_CONV
            HIP_CALL(hipMemcpy(device_output, host_output,
                       static_cast<size_t>(n) * k * do_ * ho * wo * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                       static_cast<size_t>(k) * c * z * y * x * sizeof(float), hipMemcpyHostToDevice));
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
                                   static_cast<size_t>(n) * c * di * hi * wi * sizeof(float),
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
            if(driver_data_type != driverFloat){
                device_input_to_host = malloc((static_cast<size_t>(n) * c * di * hi * wi * data_byte + 3) / 4 * 4 );
            }
            else{
                device_input_to_host = malloc(static_cast<size_t>(n) * c * di * hi * wi * sizeof(float));
            }
            // printf("len:%d\n", n * c * di * hi * wi * sizeof(float) );
        }

        if(driver_data_type == driverFloat){
            HIP_CALL(hipMemcpy(device_output, host_output,
                        static_cast<size_t>(n) * k * do_ * ho * wo * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                        static_cast<size_t>(k) * c * z * y * x * data_byte, hipMemcpyHostToDevice));
        }else{
            HIP_CALL(hipMemcpy(device_output_dtype, host_output_dtype,
                        static_cast<size_t>(n) * k * do_ * ho * wo * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight_dtype, host_weight_dtype,
                        static_cast<size_t>(k) * c * z * y * x * data_byte, hipMemcpyHostToDevice));
        }

        igemm_bwd_gtc_t conv_bwd_driver(module_tensor_cast, module, driver_mode, driver_data_type, warmup, repeat, verbose);
        conv_bwd_driver.set_vector_c(vector_c);

        auto bwd_pre = [&](){
            if (need_verify)
                HIP_CALL(hipMemset(driver_data_type == driverFloat ? device_input : device_input_dtype,
                    0x7f, static_cast<size_t>(n) * c * di * hi * wi * data_byte)); // 0x7f7f7f7f ~= 7.41e+28, a very large number
        };

        auto bwd_post = [&](){
            if (need_verify) {
                double nrms = get_nrms("bwd", driver_data_type);
                bool is_valid = false;
                if(driver_data_type == driverFloat){
                    HIP_CALL(hipMemcpy(device_input_to_host, device_input,
                                    static_cast<size_t>(n) * c * di * hi * wi * data_byte,
                                    hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_input, static_cast<float*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * di * hi * wi, nrms);
                } else {
                    HIP_CALL(hipMemcpy(device_input_to_host, device_input_dtype,
                                    static_cast<size_t>(n) * c * di * hi * wi * data_byte,
                                    hipMemcpyDeviceToHost));
                    if(driver_data_type == driverHalf)
                        is_valid = valid_vector<float16>(host_input, static_cast<float16*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * di * hi * wi, nrms);
                    else if (driver_data_type == driverBFloat16)
                        is_valid = valid_vector<bfloat16>(host_input, static_cast<bfloat16*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * di * hi * wi, nrms);
                    else if (driver_data_type == driverInt8)
                        is_valid = valid_vector<int8_t>(host_input, static_cast<int8_t*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * di * hi * wi, nrms);
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
        void *device_weight_to_host = NULL;

        // begin wrw
        if (need_verify) {
            // gen rand
            if(!igemm_rand_int){
                gen_rand_vector<float, float>(host_input, static_cast<size_t>(n) * c * di * hi * wi, 0.0, 1.0);
                gen_rand_vector<float, float>(host_output, static_cast<size_t>(n) * k * do_ * ho * wo, -0.5, 0.5);
            }else{
                gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * c * di * hi * wi, -5, 5);
                gen_rand_vector<float, int>(host_output, static_cast<size_t>(n) * k * do_ * ho * wo, -5, 5);
            }
            //gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * c * di * hi * wi, 1, 1);
            //gen_rand_vector<float, int>(host_output, static_cast<size_t>(n) * k * do_ * ho * wo, 1, 1);
            //gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * c * di * hi * wi, -1, 1);
            //gen_rand_vector<float, int>(host_output, static_cast<size_t>(n) * k * do_ * ho * wo, -1, 1);
            if(driver_data_type == driverHalf){
                tensor_copy<float16, float>(static_cast<float16*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * di * hi * wi);
                tensor_copy<float16, float>(static_cast<float16*>(host_output_dtype), host_output, static_cast<size_t>(n) * k * do_ * ho * wo);
            }
            else if(driver_data_type == driverBFloat16){
                tensor_copy<bfloat16, float>(static_cast<bfloat16*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * di * hi * wi);
                tensor_copy<bfloat16, float>(static_cast<bfloat16*>(host_output_dtype), host_output, static_cast<size_t>(n) * k * do_ * ho * wo);
            }
#ifdef USE_GPU_NAIVE_CONV
            HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * di * hi * wi * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_output, host_output,
                       static_cast<size_t>(n) * k * do_ * ho * wo * sizeof(float), hipMemcpyHostToDevice));
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
            if(driver_data_type == driverHalf){
                device_weight_to_host = malloc((static_cast<size_t>(k) * c * z * y * x * data_byte + 3) / 4 * 4);
            }
            else{
                device_weight_to_host = malloc(static_cast<size_t>(k) * c * z * y * x * sizeof(float));
            }
        }

        if(driver_data_type == driverFloat){
            HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * di * hi * wi * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_output, host_output,
                       static_cast<size_t>(n) * k * do_ * ho * wo * sizeof(float), hipMemcpyHostToDevice));
        }else{
            HIP_CALL(hipMemcpy(device_input_dtype, host_input_dtype,
                        static_cast<size_t>(n) * c * di * hi * wi * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_output_dtype, host_output_dtype,
                        static_cast<size_t>(n) * k * do_ * ho * wo * data_byte, hipMemcpyHostToDevice));
        }

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


        igemm_wrw_gtc_t conv_wrw_driver(module_tensor_cast, module, driver_mode, driver_data_type, warmup, repeat, verbose);
        conv_wrw_driver.set_vector_c(vector_c);
        
        auto wrw_pre = [&](){
            if (need_verify)
                HIP_CALL(hipMemset(driver_data_type == driverFloat ? device_weight : device_weight_dtype,
                    0, static_cast<size_t>(k) * c * z * y * x * data_byte));
        };

        auto wrw_post = [&](){
            if (need_verify) {
                double nrms = get_nrms("wrw", driver_data_type);
                bool is_valid;
                if(driver_data_type == driverFloat){
                    HIP_CALL(hipMemcpy(device_weight_to_host, device_weight,
                                   static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x * sizeof(float),
                                   hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_weight, static_cast<float*>(device_weight_to_host),
                                    static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x, nrms);
                }else{
                    HIP_CALL(hipMemcpy(device_weight_to_host, device_weight_dtype,
                                   static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x * data_byte,
                                   hipMemcpyDeviceToHost));
                    if(driver_data_type == driverHalf)
                        is_valid = valid_vector<float16>(host_weight, static_cast<float16*>(device_weight_to_host),
                                    static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x, nrms);
                    else if(driver_data_type == driverBFloat16)
                        is_valid = valid_vector<bfloat16>(host_weight, static_cast<bfloat16*>(device_weight_to_host),
                                    static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x, nrms);
                }
                printf(", valid:%s", is_valid ? "y" : "n");
                if(assert_when_invalid) assert(is_valid);
            }
        };

        if(driver_data_type == driverFloat)
            launch_conv_driver(&conv_wrw_driver, &conv_args, tunables, "wrw", driver_data_type, p_bcsv, device_input, device_weight, device_output, wrw_pre, wrw_post);
        else
            launch_conv_driver(&conv_wrw_driver, &conv_args, tunables, "wrw", driver_data_type, p_bcsv, device_input_dtype, device_weight_dtype, device_output_dtype, wrw_pre, wrw_post);

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

#if DEBUG_ASM_PRINT
    free(host_print);
    hipFree(device_print);
#endif

#if defined(USE_HALF) || defined(USE_INT8) || defined(USE_BF16) || defined(USE_INT4)
    free(host_input_dtype);
    free(host_weight_dtype);
    free(host_output_dtype);

    hipFree(device_input_dtype);
    hipFree(device_weight_dtype);
    hipFree(device_output_dtype);
#endif
}
