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
        };
        struct{
            int wave_tile_m;
            int wave_step_m;
            int wave_repeat_m;
            int wave_tile_n;
            int wave_step_n;
            int wave_repeat_n;
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
    int gemmk_groups;
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
            tunable.gemmk_groups             = codegen_sec.at("arch").get_string() != "gfx908" ? 0 : sec.count("gemmk_groups") > 0 ? sec.at("gemmk_groups").get_int() : 0;

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
    auto multihead                = tunable->multihead;

    std::string kernel_name = std::string("igemm_") + direction + "_";
    if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC)
        kernel_name += "gtcm_";
    else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS)
        kernel_name += "gtc_";
    else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS)
        kernel_name += "gtcx_";
    
    kernel_name += tensor_layout + std::string("_") + precision +
        std::string("_bx") + std::to_string(nxb) + 
        std::string("_ex") + std::to_string(nxe) + "_";
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
        kernel_name +=   std::string("wt") + std::to_string(tunable->wave_tile_m) + "x" + std::to_string(tunable->wave_tile_n) + "_" + 
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
    return kernel_name;
}

#endif