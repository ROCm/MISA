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
#include <string>
#include <unistd.h>
#include <vector>


typedef struct {
    std::string tensor_layout;
    int gemm_m_per_block;
    int gemm_n_per_block;
    int gemm_k_per_block;
    int gemm_m_per_thread;
    int gemm_m_level0_cluster;
    int gemm_m_level1_cluster;
    int gemm_n_per_thread;
    int gemm_n_level0_cluster;
    int gemm_n_level1_cluster;
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
} igemm_gtc_tunable_t;

static inline std::vector<igemm_gtc_tunable_t>
igemm_gtc_tunable_from_config(const config_content_t &content) {
    std::vector<igemm_gtc_tunable_t> tunables;
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
            tunable.gemm_m_per_thread        = sec.at("gemm_m_per_thread").get_int();
            tunable.gemm_m_level0_cluster    = sec.at("gemm_m_level0_cluster").get_int();
            tunable.gemm_m_level1_cluster    = sec.at("gemm_m_level1_cluster").get_int();
            tunable.gemm_n_per_thread        = sec.at("gemm_n_per_thread").get_int();
            tunable.gemm_n_level0_cluster    = sec.at("gemm_n_level0_cluster").get_int();
            tunable.gemm_n_level1_cluster    = sec.at("gemm_n_level1_cluster").get_int();
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

            tunables.push_back(tunable);
        }
    }
    return tunables;
}

#endif