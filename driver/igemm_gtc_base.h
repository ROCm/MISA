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

#include "config_parser.h"
#include "utility.h"
#include <string>
#include <unistd.h>
#include <vector>
#include <assert.h>

#define IGEMM_GTC_TUNABLE_FMA_TYPE_MAC              "mac"
#define IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS            "dlops"
#define IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS           "xdlops"
#define IGEMM_GTC_TUNABLE_FMA_TYPE_NA               "fma_na"
#define AMDGPU_WAVE_SIZE        64

#if USE_MAGIC_DIV
typedef struct {
    uint32_t magic;
    uint8_t shift;
} magic_div_u32_t;

/*
*
* numer / denom = quotient, reminder
*
* use magic number to do integer division of uint32 (acctually INT32_MAX, the 31 bit divisoin)
* most algorithm to compute uint32 need branching if cover all 32 bit of uint32.
* since we compute the magic number on host side, implement the division in gpu side, it is better not use branching
* hence add more restriction to numer and denom, to be 1 bit less. hence need less-or-equal than INT32_MAX 
*
* magic_div_u32_gen() compute from input arg d, to get a magic and a shift.
* to use the value, below is a example host-side code to do this
*
* // host side version
* static inline uint32_t magic_div_mulhi_u32(uint32_t x, uint32_t y) {
*     uint64_t xl = x, yl = y;
*     uint64_t rl = xl * yl;
*     return (uint32_t)(rl >> 32);
* }
* uint32_t magic_div_u32_do(uint32_t numer, const struct magic_div_u32_t *denom) {
*     uint32_t tmp = magic_div_mulhi_u32(denom->magic, numer);
*     return (tmp + numer) >> denom->shift;
* }
*
*/
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
#endif

typedef struct {
    std::string tensor_layout;
    int gemm_m_per_block;
    int gemm_n_per_block;
    int gemm_k_per_block;
    std::string fma_type;
    union{
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
    int gemm_k_global_split;
} igemm_gtc_tunable_t;

static inline std::string get_igemm_gtc_fma_type(std::string arch_string, const config_section_t &sec){
    if(sec.count("gemm_m_per_thread") > 0 && sec.count("gemm_n_per_thread") > 0){
        if(arch_string == "gfx900")
            return IGEMM_GTC_TUNABLE_FMA_TYPE_MAC;
        if(arch_string == "gfx906")
            return IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS;
        if(arch_string == "gfx908")
            return IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS;
    }else if(sec.count("wave_tile_m") > 0 && sec.count("wave_tile_n") > 0){
        assert(arch_string == "gfx908");
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
            if(tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC || tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS){
                tunable.gemm_m_per_thread        = sec.at("gemm_m_per_thread").get_int();
                tunable.gemm_m_level0_cluster    = sec.at("gemm_m_level0_cluster").get_int();
                tunable.gemm_m_level1_cluster    = sec.at("gemm_m_level1_cluster").get_int();
                tunable.gemm_n_per_thread        = sec.at("gemm_n_per_thread").get_int();
                tunable.gemm_n_level0_cluster    = sec.at("gemm_n_level0_cluster").get_int();
                tunable.gemm_n_level1_cluster    = sec.at("gemm_n_level1_cluster").get_int();
            }else{
                tunable.wave_tile_m              = sec.at("wave_tile_m").get_int();
                tunable.wave_step_m              = sec.at("wave_step_m").get_int();
                tunable.wave_repeat_m            = sec.at("wave_repeat_m").get_int();
                tunable.wave_tile_n              = sec.at("wave_tile_n").get_int();
                tunable.wave_step_n              = sec.at("wave_step_n").get_int();
                tunable.wave_repeat_n            = sec.at("wave_repeat_n").get_int();
                tunable.wave_tile_k              = sec.count("wave_tile_k") > 0 ? sec.at("wave_tile_k").get_int() : 1;
            }
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
            tunable.multihead                = sec.count("multihead") > 0 ? sec.at("multihead").get_int() : 0;
            int default_source_access_order  = tunable.direction == "fwd" ? 1 : 0;
            tunable.source_access_order      = sec.count("source_access_order") > 0 ? sec.at("source_access_order").get_int() : default_source_access_order;
            tunable.gemm_k_global_split      = sec.count("gemm_k_global_split") > 0 ? sec.at("gemm_k_global_split").get_int() : 0;

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
    auto gemm_k_global_split      = tunable->gemm_k_global_split;

    std::string kernel_name = std::string("igemm_") + direction + "_";
    if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC)
        kernel_name += "gtcm_";
    else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS)
        kernel_name += "gtc_";
    else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS)
        kernel_name += "gtcx_";

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

    if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC || tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS){
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
    if(gemm_m_unmerge_cluster)
        kernel_name += std::string("_mc");
    if(gemm_n_unmerge_cluster)
        kernel_name += std::string("_nc");
    if(gemm_k_unmerge_cluster)
        kernel_name += std::string("_kc");
    if(multihead)
        kernel_name += std::string("_mh");
    // when split in gemmk, we need call atomic add function
    if(gemm_k_global_split > 0)
        kernel_name += std::string("_gkgs");
    return kernel_name;
}

// this is to support big tensor > 4G. need to decide how many splits needed
// return the number of splits, valid for nchw, nhwc, 2d/3d conv
int igemm_split_batch_size(int n, int wi, int hi, int di, int c, int k, int wo, int ho, int do_, int data_byte)
{
    size_t image_size_input = static_cast<size_t>(c) * di * hi * wi * data_byte;
    size_t image_size_output = static_cast<size_t>(k) * do_ * ho * wo * data_byte;
    size_t size_4g = 0xffffffffUL;
    if(image_size_input >= size_4g || image_size_output >= size_4g)
        return 0;

    size_t image_size = image_size_input >= image_size_output ? image_size_input : image_size_output;
    size_t splited_n = size_4g / image_size;

    // round up splits, we must match
    // 1. splited_n * image_size < size_4g
    // 2. n % splited_n == 0
    assert(splited_n != 0);

    if(splited_n >= n)
        return 1;       // speed up following while loop

    while(splited_n >= 1){
        if(n % splited_n == 0)
            break;
        splited_n--;
    }

    assert(splited_n * image_size < size_4g && n % splited_n == 0);
    return n / splited_n;
}

#endif