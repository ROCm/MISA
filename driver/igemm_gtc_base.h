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

#ifndef __IGEMM_GTC_BASE_H
#define __IGEMM_GTC_BASE_H

#ifdef USE_HALF
#include "half.hpp"
using float16 = half_float::half;
#else
using float16 = int16_t;
#endif
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include "config_parser.h"
#include "utility.h"
#include <string>
#include <unistd.h>
#include <vector>
#include <assert.h>
#include <math.h>
#include <functional>
#include <stdint.h>
#include <numeric>
#include "magic_div.h"

#define IGEMM_GTC_TUNABLE_FMA_TYPE_MAC              "mac"
#define IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS            "dlops"
#define IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS           "xdlops"
#define IGEMM_GTC_TUNABLE_FMA_TYPE_NA               "fma_na"
#define AMDGPU_WAVE_SIZE        64

typedef enum {
    driverHalf  = 0, /*!< 16-bit floating point (Fully supported) */
    driverFloat = 1, /*!< 32-bit floating point (Fully supported) */
    driverInt8  = 3,
    driverBFloat16 = 5, /*!< 16-bit binary floating point (8-bit exponent, 7-bit fraction)
                           (Partially supported) */
} driverDataType_t;

typedef struct {
    void* output;
    void* input;
    int total_length;
} __attribute__((packed)) tensor_cast_karg_t;

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

typedef enum {
    driver_mode_normal      = 0,    // bench all solutions
    driver_mode_heuristic   = 1,    // find suitable heuristic
} driver_mode_t;

typedef struct {
    std::string tensor_layout;
    int gemm_m_per_block;
    int gemm_n_per_block;
    int gemm_k_per_block;
    std::string fma_type;
    union{
        struct{
            int lanegroup_tile_m;
            int lanegroup_wave_m;
            int lanegroup_repeat_m;
            int lanegroup_tile_n;
            int lanegroup_wave_n;
            int lanegroup_repeat_n;
            int dummy_0;
        };
        struct{
            int gemm_m_per_thread;
            int gemm_m_level0_cluster;
            int gemm_m_level1_cluster;
            int gemm_n_per_thread;
            int gemm_n_level0_cluster;
            int gemm_n_level1_cluster;
            int dummy;
        };
        struct{
            int wave_tile_m;
            int wave_step_m;
            int wave_repeat_m;
            int wave_tile_n;
            int wave_step_n;
            int wave_repeat_n;
            int wave_tile_k;
        };
    };
    int tensor_a_pass_through;
    int tensor_b_pass_through;
    std::vector<int> tensor_a_thread_lengths;
    std::vector<int> tensor_a_cluster_lengths;
    std::vector<int> tensor_b_thread_lengths;
    std::vector<int> tensor_b_cluster_lengths;
    std::string direction;
    std::string precision;
    int nxb;
    int nxe;
    int gemm_m_unmerge_cluster;
    int gemm_n_unmerge_cluster;
    int gemm_k_unmerge_cluster;
    int multihead;
    int source_access_order;
    int vector_store;
    int gemm_k_global_split;
    int merge_e;
} igemm_gtc_tunable_t;

static inline std::string get_igemm_gtc_fma_type(std::string arch_string, const config_section_t &sec){
    if(sec.count("lanegroup_tile_m") > 0 && sec.count("lanegroup_tile_n") > 0){
        if(arch_string == "gfx900")
            return IGEMM_GTC_TUNABLE_FMA_TYPE_MAC;
        if(arch_string == "gfx906" || arch_string == "gfx1030")
            return IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS;
        if(arch_string == "gfx908" || arch_string == "gfx90a")
            return IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS;
    }else if(sec.count("wave_tile_m") > 0 && sec.count("wave_tile_n") > 0){
        assert(arch_string == "gfx908" || arch_string == "gfx90a");
        return IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS;
    }
    return IGEMM_GTC_TUNABLE_FMA_TYPE_NA;
}

static inline std::vector<igemm_gtc_tunable_t>
igemm_gtc_tunable_from_config(const config_content_t &content) {
    std::vector<igemm_gtc_tunable_t> tunables;
    config_section_t codegen_sec = content.get_section("codegen");
    assert(codegen_sec.get_name() == "codegen");
    for (const auto &sec : content) {
        if (sec.get_name() == "igemm_fwd_gtc" ||
            sec.get_name() == "igemm_bwd_gtc" || 
            sec.get_name() == "igemm_wrw_gtc")
        {
            igemm_gtc_tunable_t tunable;
            tunable.tensor_layout            = sec.count("tensor_layout") > 0 ? sec.at("tensor_layout").get_string() : "nchw";
            tunable.gemm_m_per_block         = sec.at("gemm_m_per_block").get_int();
            tunable.gemm_n_per_block         = sec.at("gemm_n_per_block").get_int();
            tunable.gemm_k_per_block         = sec.at("gemm_k_per_block").get_int();
            tunable.fma_type                 = get_igemm_gtc_fma_type(codegen_sec.at("arch").get_string(), sec);
            assert(tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_NA);
            if(tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC){
                tunable.gemm_m_per_thread        = sec.at("gemm_m_per_thread").get_int();
                tunable.gemm_m_level0_cluster    = sec.at("gemm_m_level0_cluster").get_int();
                tunable.gemm_m_level1_cluster    = sec.at("gemm_m_level1_cluster").get_int();
                tunable.gemm_n_per_thread        = sec.at("gemm_n_per_thread").get_int();
                tunable.gemm_n_level0_cluster    = sec.at("gemm_n_level0_cluster").get_int();
                tunable.gemm_n_level1_cluster    = sec.at("gemm_n_level1_cluster").get_int();
            }else if(tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS){
                tunable.lanegroup_tile_m        = sec.at("lanegroup_tile_m").get_int();
                tunable.lanegroup_wave_m    = sec.at("lanegroup_wave_m").get_int();
                tunable.lanegroup_repeat_m    = sec.at("lanegroup_repeat_m").get_int();
                tunable.lanegroup_tile_n        = sec.at("lanegroup_tile_n").get_int();
                tunable.lanegroup_wave_n    = sec.at("lanegroup_wave_n").get_int();
                tunable.lanegroup_repeat_n    = sec.at("lanegroup_repeat_n").get_int();
            }
            else{
                tunable.wave_tile_m              = sec.at("wave_tile_m").get_int();
                tunable.wave_step_m              = sec.at("wave_step_m").get_int();
                tunable.wave_repeat_m            = sec.at("wave_repeat_m").get_int();
                tunable.wave_tile_n              = sec.at("wave_tile_n").get_int();
                tunable.wave_step_n              = sec.at("wave_step_n").get_int();
                tunable.wave_repeat_n            = sec.at("wave_repeat_n").get_int();
                tunable.wave_tile_k              = sec.count("wave_tile_k") > 0 ? sec.at("wave_tile_k").get_int() : 1;
            }
            tunable.tensor_a_pass_through    = sec.count("tensor_a_pass_through") > 0 ? sec.at("tensor_a_pass_through").get_int() : 0;
            tunable.tensor_b_pass_through    = sec.count("tensor_b_pass_through") > 0 ? sec.at("tensor_b_pass_through").get_int() : 0;
            tunable.tensor_a_thread_lengths  = sec.at("tensor_a_thread_lengths").get_list_int();
            tunable.tensor_a_cluster_lengths = sec.at("tensor_a_cluster_lengths").get_list_int();
            tunable.tensor_b_thread_lengths  = sec.at("tensor_b_thread_lengths").get_list_int();
            tunable.tensor_b_cluster_lengths = sec.at("tensor_b_cluster_lengths").get_list_int();
            tunable.direction                = sec.at("direction").get_string();
            tunable.precision                = sec.at("precision").get_string();
            tunable.nxb                      = sec.at("nxb").get_int();
            tunable.nxe                      = sec.at("nxe").get_int();
            tunable.gemm_m_unmerge_cluster   = sec.count("gemm_m_unmerge_cluster") > 0 ? sec.at("gemm_m_unmerge_cluster").get_int() : 0;
            tunable.gemm_n_unmerge_cluster   = sec.count("gemm_n_unmerge_cluster") > 0 ? sec.at("gemm_n_unmerge_cluster").get_int() : 0;
            tunable.gemm_k_unmerge_cluster   = sec.count("gemm_k_unmerge_cluster") > 0 ? sec.at("gemm_k_unmerge_cluster").get_int() : 0;
            int default_mh                   = tunable.direction == "bwd" && tunable.tensor_layout == "nhwc" && tunable.nxe != 0 ? 1 : 0;
            tunable.multihead                = sec.count("multihead") > 0 ? sec.at("multihead").get_int() : default_mh;
            int default_source_access_order  = tunable.direction == "fwd" ? 1 : 0;
            tunable.source_access_order      = sec.count("source_access_order") > 0 ? sec.at("source_access_order").get_int() : default_source_access_order;
            tunable.vector_store             = sec.count("vector_store") > 0 ? sec.at("vector_store").get_int() : 0;
            tunable.gemm_k_global_split      = sec.count("gemm_k_global_split") > 0 ? sec.at("gemm_k_global_split").get_int() : 0;
            tunable.merge_e                  = sec.count("merge_e") > 0 ? sec.at("merge_e").get_int() : 0;
            tunables.push_back(tunable);
        }
    }
    return tunables;
}

static inline std::string
igemm_gtc_encode_kernel_name(const igemm_gtc_tunable_t *tunable) {
    auto tensor_layout            = tunable->tensor_layout;
    auto gemm_m_per_block         = tunable->gemm_m_per_block;
    auto gemm_n_per_block         = tunable->gemm_n_per_block;
    auto gemm_k_per_block         = tunable->gemm_k_per_block;
    auto fma_type                 = tunable->fma_type;
    // auto gemm_m_per_thread        = tunable->gemm_m_per_thread;
    // auto gemm_m_level0_cluster    = tunable->gemm_m_level0_cluster;
    // auto gemm_m_level1_cluster    = tunable->gemm_m_level1_cluster;
    // auto gemm_n_per_thread        = tunable->gemm_n_per_thread;
    // auto gemm_n_level0_cluster    = tunable->gemm_n_level0_cluster;
    // auto gemm_n_level1_cluster    = tunable->gemm_n_level1_cluster;
    auto tensor_a_pass_through    = tunable->tensor_a_pass_through;
    auto tensor_b_pass_through    = tunable->tensor_b_pass_through;
    auto tensor_a_thread_lengths  = tunable->tensor_a_thread_lengths;
    auto tensor_a_cluster_lengths = tunable->tensor_a_cluster_lengths;
    auto tensor_b_thread_lengths  = tunable->tensor_b_thread_lengths;
    auto tensor_b_cluster_lengths = tunable->tensor_b_cluster_lengths;
    auto direction                = tunable->direction;
    auto precision                = tunable->precision;
    auto nxb                      = tunable->nxb;
    auto nxe                      = tunable->nxe;
    auto gemm_m_unmerge_cluster   = tunable->gemm_m_unmerge_cluster;
    auto gemm_n_unmerge_cluster   = tunable->gemm_n_unmerge_cluster;
    auto gemm_k_unmerge_cluster   = tunable->gemm_k_unmerge_cluster;
    auto source_access_order      = tunable->source_access_order;
    auto multihead                = tunable->multihead;
    auto vector_store             = tunable->vector_store;
    auto gemm_k_global_split      = tunable->gemm_k_global_split;
    auto merge_e                  = tunable->merge_e;

    static int gcn_arch = -1;
    if(gcn_arch == -1){
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        gcn_arch = dev_prop.gcnArch;
    }

    std::string kernel_name = std::string("igemm_") + direction + "_";
    if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC)
        kernel_name += "gtcm_";
    else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS)
        if(gcn_arch == 1030)
            kernel_name += "gtcn2_";
        else
            kernel_name += "gtc_";
    else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS){
        if(gcn_arch == 908)
            kernel_name += "gtcx_";
        else if(gcn_arch == 910)
            kernel_name += "gtcx2_";
    }

    kernel_name += tensor_layout + std::string("_") + precision +
        std::string("_bx") + std::to_string(nxb) + 
        std::string("_ex") + std::to_string(nxe) +
#if USE_SOURCE_ACCESS_ENCODING_KERNEL_NAME
        std::string("_sa") + std::to_string(source_access_order) + "_";
#else
        "_";
#endif

    kernel_name += std::string("bt") +
            std::to_string(gemm_m_per_block) + "x" +
            std::to_string(gemm_n_per_block) + "x" +
            std::to_string(gemm_k_per_block) + "_";

    if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC){
        auto gemm_m_per_thread        = tunable->gemm_m_per_thread;
        auto gemm_m_level0_cluster    = tunable->gemm_m_level0_cluster;
        auto gemm_m_level1_cluster    = tunable->gemm_m_level1_cluster;
        auto gemm_n_per_thread        = tunable->gemm_n_per_thread;
        auto gemm_n_level0_cluster    = tunable->gemm_n_level0_cluster;
        auto gemm_n_level1_cluster    = tunable->gemm_n_level1_cluster;
        assert(gemm_m_per_block % (gemm_m_per_thread * gemm_m_level0_cluster * gemm_m_level1_cluster) == 0);
        assert(gemm_n_per_block % (gemm_n_per_thread * gemm_n_level0_cluster * gemm_n_level1_cluster) == 0);
        int gemm_m_repeat = gemm_m_per_block / (gemm_m_per_thread * gemm_m_level0_cluster * gemm_m_level1_cluster);
        int gemm_n_repeat = gemm_n_per_block / (gemm_n_per_thread * gemm_n_level0_cluster * gemm_n_level1_cluster);

        int thread_tile_m = gemm_m_repeat * gemm_m_per_thread;
        int thread_tile_n = gemm_n_repeat * gemm_n_per_thread;
        kernel_name += std::string("tt") +
            std::to_string(thread_tile_m) + "x" +
            std::to_string(thread_tile_n) + "_" +
            "gm" + 
            std::to_string(gemm_m_repeat) + "x" +
            std::to_string(gemm_m_level0_cluster) + "x" +
            std::to_string(gemm_m_level1_cluster) + "_" +
            "gn" + 
            std::to_string(gemm_n_repeat) + "x" +
            std::to_string(gemm_n_level0_cluster) + "x" +
            std::to_string(gemm_n_level1_cluster) + "_";
    }else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS){
        kernel_name +=   std::string("lt") + std::to_string(tunable->lanegroup_tile_m) + "x" + std::to_string(tunable->lanegroup_tile_n) + "_" + 
                         "lw" + std::to_string(tunable->lanegroup_wave_m) + "x" + std::to_string(tunable->lanegroup_wave_n) + "_" +
                         "ws" + std::to_string(tunable->lanegroup_repeat_m) + "x" + std::to_string(tunable->lanegroup_repeat_n) + "_";
    }else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS){
        kernel_name +=   std::string("wt") + std::to_string(tunable->wave_tile_m) + "x" + std::to_string(tunable->wave_tile_n) + "x" + std::to_string(tunable->wave_tile_k) + "_" + 
                         "ws" + std::to_string(tunable->wave_step_m) + "x" + std::to_string(tunable->wave_step_n) + "_" +
                         "wr" + std::to_string(tunable->wave_repeat_m) + "x" + std::to_string(tunable->wave_repeat_n) + "_";
    }

    kernel_name +=
            "ta" + utility_int_list_to_string(tensor_a_thread_lengths) + "_" + 
                    utility_int_list_to_string(tensor_a_cluster_lengths)+ "_" + 
            "tb" + utility_int_list_to_string(tensor_b_thread_lengths) + "_" + 
                    utility_int_list_to_string(tensor_b_cluster_lengths);
    // printf("[%s]\n",kernel_name.c_str());
    if(tensor_a_pass_through)
        kernel_name += std::string("_pta");
    if(tensor_b_pass_through)
        kernel_name += std::string("_ptb");
    if(gemm_m_unmerge_cluster)
        kernel_name += std::string("_mc");
    if(gemm_n_unmerge_cluster)
        kernel_name += std::string("_nc");
    if(gemm_k_unmerge_cluster)
        kernel_name += std::string("_kc");
    if(multihead)
        kernel_name += std::string("_mh");
    if(merge_e)
        kernel_name += std::string("_me");
    // when split in gemmk, we need call atomic add function
    if(vector_store)
        kernel_name += std::string("_vs") + std::to_string(vector_store);
    if(gemm_k_global_split > 0)
        kernel_name += std::string("_gkgs");
    return kernel_name;
}

static inline float igemm_launch_kernel_single(hipFunction_t kernel_func, void* args, size_t arg_size, std::vector<size_t> grid_size, std::vector<size_t> block_size)
{
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
                        HIP_LAUNCH_PARAM_END};
    float ms = .0;

    hipEvent_t start;
    hipEvent_t stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // for hipHccModuleLaunchKernel/hipExtModuleLaunchKernel, the grid_size is in unit of workitem
    HIP_CALL(hipHccModuleLaunchKernel(kernel_func, grid_size[0], grid_size[1], grid_size[2],
                                        block_size[0], block_size[1], block_size[2], 0, 0, NULL,
                                        (void **)&config, start, stop));


    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    return ms;
}

static inline float igemm_launch_kernel(hipFunction_t kernel_func, void* args, size_t arg_size, std::vector<size_t> grid_size, std::vector<size_t> block_size, int warmup, int repeat)
{
    assert(repeat > 2);
    std::vector<float> duration_list;
    for (int i = 0; i < warmup; i++) {
        igemm_launch_kernel_single(kernel_func, args, arg_size, grid_size, block_size);
    }

    for (int i = 0; i < repeat; i++) {
        float d = igemm_launch_kernel_single(kernel_func, args, arg_size, grid_size, block_size);
        duration_list.push_back(d);
    }
    // remove min and max from list, then do average
    auto imin = std::min_element(begin(duration_list), end(duration_list));
    duration_list.erase(imin);
    auto imax = std::max_element(begin(duration_list), end(duration_list));
    duration_list.erase(imax);

    assert(duration_list.size() == (repeat - 2));
    float avg_duration = std::accumulate(duration_list.begin(), duration_list.end(), (float).0) / duration_list.size();
    return avg_duration;
}

typedef struct{
    hipFunction_t           kernel_func;
    void *                  args;
    size_t                  arg_size;
    std::vector<size_t>     grid_size;
    std::vector<size_t>     block_size;
}igemm_launch_kernel_t;

template<typename prolog_kernel_t, typename postlog_kernel_t>
static inline float igemm_launch_kernels(const std::vector<igemm_launch_kernel_t> & kernels, prolog_kernel_t prolog_kernel, postlog_kernel_t postlog_kernel, int warmup, int repeat)
{
    auto launch_kernels = [&]() -> float{
        float ms = .0;
        ms += prolog_kernel();
        for(const auto & ker :  kernels){
            float t = igemm_launch_kernel_single(ker.kernel_func, ker.args, ker.arg_size, ker.grid_size, ker.block_size);
            //std::cout << ker.kernel_func << ": " << t << std::endl;
            ms += t;
        }
        ms += postlog_kernel();
        return ms;
    };

    assert(repeat > 2);
    std::vector<float> duration_list;
    for (int i = 0; i < warmup; i++) {
        launch_kernels();
    }

    for (int i = 0; i < repeat; i++) {
        float d = launch_kernels();
        duration_list.push_back(d);
    }
    // remove min and max from list, then do average
    auto imin = std::min_element(begin(duration_list), end(duration_list));
    duration_list.erase(imin);
    auto imax = std::max_element(begin(duration_list), end(duration_list));
    duration_list.erase(imax);

    assert(duration_list.size() == (repeat - 2));
    float avg_duration = std::accumulate(duration_list.begin(), duration_list.end(), (float).0) / duration_list.size();
    return avg_duration;
}

static inline int igemm_get_max_gks(int gemm_k, int gemm_k_per_block, int max_log2_splits)
{
    if(gemm_k % gemm_k_per_block != 0)
        return 0;
    int rem = gemm_k / gemm_k_per_block;
    // to find the highest power of 2 value that can divide rem
    // https://www.geeksforgeeks.org/highest-power-of-two-that-divides-a-given-number/
    int rem_pow2 = rem & (~(rem - 1));
    int gks = (int)log2(rem_pow2);
    if(gks > max_log2_splits)
        gks = max_log2_splits;
    return gks;
}

// this is to support big tensor > 4G. need to decide how many splits needed
// return the number of splits
static inline size_t igemm_split_batch_size(const args_t *arg, int data_byte)
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

    // int data_byte = utility_string_to_data_byte(tunable->precision);
    size_t image_size_input = static_cast<size_t>(c) * hi * wi * data_byte;
    size_t image_size_output = static_cast<size_t>(k) * ho * wo * data_byte;
    size_t size_4g = 0xffffffffUL;
    if(image_size_input >= size_4g || image_size_output >= size_4g)
        return 0;

    size_t image_size = image_size_input >= image_size_output ? image_size_input : image_size_output;
    size_t splited_n = size_4g / image_size;

    // round up splits, we must match
    // 1. splited_n * image_size < size_4g
    // 2. n % splited_n == 0
    // if(splited_n >= n)
    //     return 1;
    assert(splited_n != 0);
    while(splited_n >= 1){
        // printf("n:%d, splited_n:%d\n", n, splited_n);
        if(n % splited_n == 0 && splited_n * image_size < size_4g)
            break;
        splited_n--;
    }
    assert(splited_n * image_size < size_4g && n % splited_n == 0);
    return static_cast<size_t>(n) / splited_n;
}

class igemm_driver_base_t{
public:
    igemm_driver_base_t(hipModule_t module_tensor_cast_, hipModule_t module_, driver_mode_t driver_mode_, driverDataType_t data_type_, int warmup_, int repeat_, bool verbose_) : 
        module_tensor_cast(module_tensor_cast_), module(module_), driver_mode(driver_mode_), data_type(data_type_), warmup(warmup_), repeat(repeat_), verbose(verbose_)
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        this->num_cu = dev_prop.multiProcessorCount;
        this->gcn_arch = dev_prop.gcnArch;
        if(this->gcn_arch >= 1000)
            this->num_cu *= 2;
        max_mpb = -1;
        max_npb = -1;
        max_kpb = -1;
        max_gks = -1;
    }
    std::string get_kernel_name(const igemm_gtc_tunable_t *tunable) {
        return igemm_gtc_encode_kernel_name(tunable);
    }

    size_t get_workspace_size(const args_t *arg, const igemm_gtc_tunable_t *tunable){
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
        int group = arg->get_int("group_count");
        int forw = arg->get_int("forw");

        size_t workspace_size = 0;
        if(forw & 1) // forward ws size
        {
            if(tunable->precision == "fp16" && tunable->gemm_k_global_split == 1 && tunable->vector_store == 1)
                workspace_size = static_cast<size_t>(n) * k * ho * wo;
            else if(tunable->precision == "bf16" && tunable->gemm_k_global_split == 1)
                workspace_size = static_cast<size_t>(n) * k * ho * wo;
        }
        else if(forw & 2) // backward data ws size
        {
            if(tunable->precision == "fp16" && tunable->gemm_k_global_split == 1 && tunable->vector_store == 1)
                workspace_size = static_cast<size_t>(n) * c * hi * wi;
            else if(tunable->precision == "bf16" && tunable->gemm_k_global_split == 1)
                workspace_size = static_cast<size_t>(n) * c * hi * wi;
        }
        else if(forw & 4) // backward weights ws size
        {
            if(tunable->precision == "fp16" && tunable->gemm_k_global_split == 1 && (tunable->tensor_b_thread_lengths[3] == 1 || tunable->vector_store == 1))
                workspace_size = static_cast<size_t>(group) * (k / group) * (c / group) * y * x;
            else if(tunable->precision == "bf16" && tunable->gemm_k_global_split == 1)
                workspace_size = static_cast<size_t>(group) * (k / group) * (c / group) * y * x;
        }
        else if(forw == 0) // all dirs
        {
            std::cout << "not support direction" << std::endl;
            assert(false);
        }
        else
        {
            std::cout << "wrong direction" << std::endl;
            assert(false);
        }
        return workspace_size * sizeof(float);
    }

    void set_block_tile_boundary(int max_mpb_, int max_npb_, int max_kpb_, int max_gks_){
        // CAUSTION! when setting this value to none -1, you need to understand what will happen
        this->max_mpb = max_mpb_;
        this->max_npb = max_npb_;
        this->max_kpb = max_kpb_;
        this->max_gks = max_gks_;
    }

    virtual size_t get_block_size(const igemm_gtc_tunable_t *tunable) = 0;
    virtual size_t get_grid_size(const args_t *arg, const igemm_gtc_tunable_t *tunable) = 0;
    virtual bool tunable_is_valid(const args_t *arg, const igemm_gtc_tunable_t *tunable) = 0;
    virtual result_t run(const args_t *arg, const igemm_gtc_tunable_t *tunable, void *p_in, void *p_wei, void *p_out, int current_gks) = 0;
    virtual std::vector<int> get_gks_list(const args_t *arg, const igemm_gtc_tunable_t *tunable) = 0;

    virtual igemm_gtc_tunable_t heuristic_select_kernel(const args_t *arg) {return igemm_gtc_tunable_t{}; }
    virtual int heuristic_select_gks(const args_t *arg, const igemm_gtc_tunable_t *tunable) {return 0; }

    hipModule_t         module_tensor_cast;
    hipModule_t         module;         // not used in IGEMM_SPLIT_KERNEL case
    driver_mode_t       driver_mode;
    driverDataType_t    data_type;
    int                 warmup;
    int                 repeat;
    bool                verbose;

    int                 num_cu;
    int                 gcn_arch;

    int                 max_mpb;
    int                 max_npb;
    int                 max_kpb;
    int                 max_gks;
};

#endif