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

#ifndef __IGEMM_WRW_GTC_DRIVER_H
#define __IGEMM_WRW_GTC_DRIVER_H

#include "igemm_gtc_base.h"
#include "config_parser.h"
#include "utility.h"
#include <string>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <numeric>

typedef struct {
    void *p_in;
    void *p_wei;
    void *p_out;
    int hi;
    int wi;
    int n;
    int k;                      // this is indeed k_per_group
    int c;                      // this is indeed c_per_group
    int ho;
    int wo;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int pad_h;
    int pad_w;
    int y;
    int x;
    int gemm_k_global_split;
    int group;
    int __pack_0;
} __attribute__((packed)) igemm_wrw_gtc_karg_t;

static void dump_wrw_karg(igemm_wrw_gtc_karg_t * karg){
    std::cout<<"p_in:"         <<karg->p_in<<",";
    std::cout<<"p_wei:"        <<karg->p_wei<<",";
    std::cout<<"p_out:"        <<karg->p_out<<",";
    std::cout<<"hi:"           <<karg->hi<<",";
    std::cout<<"wi:"           <<karg->wi<<",";
    std::cout<<"n:"            <<karg->n<<",";
    std::cout<<"k:"            <<karg->k<<",";
    std::cout<<"c:"            <<karg->c<<",";
    std::cout<<"ho:"           <<karg->ho<<",";
    std::cout<<"wo:"           <<karg->wo<<",";
    std::cout<<"stride_h:"     <<karg->stride_h<<",";
    std::cout<<"stride_w:"     <<karg->stride_w<<",";
    std::cout<<"dilation_h:"   <<karg->dilation_h<<",";
    std::cout<<"dilation_w:"   <<karg->dilation_w<<",";
    std::cout<<"pad_h:"        <<karg->pad_h<<",";
    std::cout<<"pad_w:"        <<karg->pad_w<<",";
    std::cout<<"y:"            <<karg->y<<",";
    std::cout<<"x:"            <<karg->x<<",";
    std::cout<<"gemm_k_global_split:" <<karg->gemm_k_global_split<<",";
    std::cout<<"group:"        <<karg->group;
    std::cout<<std::endl;
}

class igemm_wrw_gtc_t : public igemm_driver_base_t {
public:
    igemm_wrw_gtc_t(hipModule_t module_, driver_mode_t driver_mode_, driverDataType_t data_type_, int warmup_, int repeat_, bool verbose_)
        : igemm_driver_base_t(module_, driver_mode_, data_type_, warmup_, repeat_, verbose_) {}
    ~igemm_wrw_gtc_t(){}

    size_t get_block_size(const igemm_gtc_tunable_t *tunable) override {
        if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC || tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS){
            return tunable->gemm_m_level0_cluster * tunable->gemm_n_level0_cluster *
               tunable->gemm_m_level1_cluster * tunable->gemm_n_level1_cluster;
        }
        else if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS){
            int waves_per_m = tunable->gemm_m_per_block / (tunable->wave_tile_m * tunable->wave_step_m * tunable->wave_repeat_m);
            int waves_per_n = tunable->gemm_n_per_block / (tunable->wave_tile_n * tunable->wave_step_n * tunable->wave_repeat_n);
            return waves_per_m * waves_per_n * AMDGPU_WAVE_SIZE;
        }
        else{
            std::cout << "not valid fma_type: " << tunable->fma_type << std::endl;
            assert(false);
            return 0;
        }
    }
    size_t get_grid_size(const args_t *arg,
                      const igemm_gtc_tunable_t *tunable) override {
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

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;
        int gemm_k_global_split      = tunable->gemm_k_global_split;

        gemm_k_global_split = update_gemm_k_global_split(arg, tunable);

        int gemm_m = k / group ;
        int gemm_n = (c / group) * y * x;

        size_t grid_size = static_cast<size_t>(group) * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
        int num_of_gemm = 1 << gemm_k_global_split;
        grid_size *= num_of_gemm;

        assert(grid_size <= 0xffffffffUL);
        return grid_size;
    }

    int get_lds_size(const igemm_gtc_tunable_t *tunable) {
        // TODO: fp16/bf16, xdlops
        int lds_a = utility_string_to_data_byte(tunable->precision) * tunable->gemm_k_per_block * tunable->gemm_m_per_block;
        int lds_b = utility_string_to_data_byte(tunable->precision) * tunable->gemm_k_per_block * tunable->gemm_n_per_block;
        return 2 * utility_next_pow2(utility_next_pow2(lds_a) + utility_next_pow2(lds_b));
    }

    bool tunable_is_valid(const args_t *arg,
                          const igemm_gtc_tunable_t *tunable) override
    {
        // TODO:
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

        int need_wrw = (forw == 0 ? 1 : (forw & 4 ? 1 : 0));
        if(need_wrw == 0)
            return false;

        int b  = tunable->nxe == 0 ? (ho * wo) : ((ho * wo + tunable->nxb - 1) / tunable->nxb) * tunable->nxb;   // pad to nxb modulo when nxe != 0
        assert(c % group == 0 && k % group == 0);

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int gemm_k_global_split      = tunable->gemm_k_global_split;
        int gemmk_blocks             = 1 << gemm_k_global_split;
        
        if (n % gemmk_blocks != 0){
            return false;
        }

        int n_per_block = n >> gemm_k_global_split;

        int gemm_n = (c / group) * y * x;
        int gemm_k = n * b;

        int nxe = tunable->nxe == 0 ? 1 : tunable->nxe;
        bool unit_conv = (x==1)&&(y==1)&&(stride_h==1)&&(stride_w==1)&&(dilation_h==1)&&(dilation_w==1)&&(pad_h==0)&&(pad_w==0);

        if(((c / group) % (gemm_n_per_block / nxe) != 0) || (((x * y) % nxe) != 0))
        {
            return false;
        }
        if (gemm_k % gemm_k_per_block != 0){
            //std::cout << __func__ << " false: gemm_n is " << gemm_n << ", gemm_n_per_block is " << gemm_n_per_block << ", gemm_m is " << gemm_m << ", gemm_m_per_block is " << gemm_m_per_block << std::endl;
            return false;
        }

        if (gemm_k_per_block % tunable->nxb != 0){
            //std::cout << __func__ << " false: gemm_n_per_block is " << gemm_n_per_block << ", tunable->nxb is " << tunable->nxb << std::endl;
            return false;
        }

        if ((x * y * stride_h * stride_w != 1) && (tunable->nxe == 0))
            return false;

        if (b % tunable->nxb != 0){
            //std::cout << __func__ << " false: (ho * wo) is " << (ho * wo) << ", tunable->nxb is " << tunable->nxb << std::endl;
            return false;
        }

        int n_n0 = tunable->tensor_a_cluster_lengths[0] * tunable->tensor_a_thread_lengths[0];
        
        if (n_n0 > 1){
            if (n_per_block % (tunable->tensor_a_thread_lengths[1] * tunable->tensor_a_cluster_lengths[1] * n_n0) != 0){
                return false;
            }
        }
        else {
            if (n_per_block * b % gemm_k_per_block !=0){
                return false;
            }
        }

        // input vector load limitation, n1b
        if(tunable->tensor_b_thread_lengths[1] > 1 && (
            !unit_conv ||
            unit_conv && (hi * wi) % tunable->tensor_b_thread_lengths[1] != 0)) {
            return false;
        }

        // output vector load limitation, n1b
        if(tunable->tensor_a_thread_lengths[1] > 1 && (
            !unit_conv ||
            unit_conv && (ho * wo) % tunable->tensor_a_thread_lengths[1] != 0)) {
            return false;
        }

        return true;
    }

    int update_gemm_k_global_split(const args_t *arg,
                                   const igemm_gtc_tunable_t *tunable)
    {
        // choose a largest gemmk splits
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
        assert(c % group == 0 && k % group == 0);

        int b  = tunable->nxe == 0 ? (ho * wo) : ((ho * wo + tunable->nxb - 1) / tunable->nxb) * tunable->nxb;

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int gemm_k_global_split      = tunable->gemm_k_global_split;

        int gemm_m = k / group;
        int gemm_n = (c / group) * y * x;
        

        int max_grid_size = 1200;

        int grid_size = group * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
        int n_n0 = tunable->tensor_a_cluster_lengths[0] * tunable->tensor_a_thread_lengths[0];
        
        if (gemm_k_global_split > 0){
            int update_gemm_k_global_split = 1;
            for (int i = 1; i < 8; i++){
                if ((grid_size << i) > max_grid_size){
                    break;
                }
                
                int n_per_block = n >> i;
                if (n_per_block == 0){
                    break;
                }
                if (n_n0 > 1){
                    if (n_per_block % (tunable->tensor_a_thread_lengths[1] * tunable->tensor_a_cluster_lengths[1] * n_n0) != 0){
                        break;
                    }
                }
                else {
                    if (n_per_block * b % gemm_k_per_block !=0){
                        break;
                    }
                }
                update_gemm_k_global_split = i;
            }
            return update_gemm_k_global_split;
        }
        else{
            return 0;
        }
    }

    static int if_gemm_k_global_split(const args_t *arg,
                                  const int gemm_m_per_block,
                                  const int gemm_n_per_block,
                                  const int gemm_k_per_block)
    {
        int gemm_k_global_split = 0;
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
        assert(c % group == 0 && k % group == 0);

        // int b      = tunable->nxe == 0 ? (ho * wo) : ((ho * wo + tunable->nxb - 1) / tunable->nxb) * tunable->nxb;
        int gemm_m = k / group;
        int gemm_n = (c / group) * y * x;
        int gemm_k = n * ho * wo;

        int grid_size;
        grid_size = group * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
        if ((n % 2 == 0) && (grid_size < 512) && ((n >> 1) * ho * wo % gemm_k_per_block == 0)){
            gemm_k_global_split = 1;
        }
        else {
            gemm_k_global_split = 0;
        }
        return gemm_k_global_split;
    }

    static inline int find_tunable(const std::vector<igemm_gtc_tunable_t> tunables, 
                                    const int gemm_m_per_block,
                                    const int gemm_n_per_block,
                                    const int gemm_k_per_block,
                                    const int gemm_k_global_split,
                                    const int nxb,
                                    const int nxe)
    {
        int i;
        for (i = 0; i < tunables.size(); i++) {
            if ((tunables[i].gemm_m_per_block == gemm_m_per_block) &&
                (tunables[i].gemm_n_per_block == gemm_n_per_block) &&
                (tunables[i].gemm_k_per_block == gemm_k_per_block) &&
                (tunables[i].gemm_k_global_split == gemm_k_global_split) &&
                (tunables[i].nxb == nxb) &&
                (tunables[i].nxe == nxe)){
                break;
            }
        }
        return i;
    }

    std::string select_kernel(const args_t *arg, const std::vector<igemm_gtc_tunable_t> tunables)
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
        int group = arg->get_int("group_count");
        assert(c % group == 0 && k % group == 0);

        int gemm_m_per_block = 0;
        int gemm_n_per_block = 0;
        int gemm_k_per_block = 0;

        int gemm_k_global_split = 0;

        int gemm_m = k / group;
        int gemm_n = (c / group) * y * x;
        int gemm_k = n * ho * wo;

        int grid_size;
        int block_size;
        int nxb = 1;
        int nxe = 1;

        int sel_index = - 1;

        std::string selected_kernel = std::string("NONE");

        igemm_gtc_tunable_t selected_tunable;

        /* applicable table (except 128x128 case):
        gemm_m/gemmn        256 64  32  16  4
                    --------------------------
                    256 |   0  |1  |0  |0  |0
                    64  |   1  |1  |0  |0  |1
                    32  |   1  |1  |1  |1  |0
                    16  |   0  |1  |0  |0  |0
        
        */
        int gemm_m_per_block_table[5] = {256, 64, 32, 16, 4};
        int gemm_n_per_block_table[4] = {256, 64, 32, 16};
        int applicable_table[4 * 5] = {
            0, 1, 0, 0, 0,
            1, 1, 0, 0, 1,
            1, 1, 1, 1, 0,
            0, 1, 0, 0, 0
        };
        int i, j, r, l;
        int max_grid_size = 0;
        int cur_grid_size = 0;
        int num_cu = 120;
        int max_block_size = 0;
        for (i = 15; i > 7; i--){
            r = (i + 1) >> 1;
            l = i - r;
            while (l > 1 && r < 9){
                for (int swap = 0; swap < 2; swap++){

                    if (swap == 0){
                        gemm_m_per_block = 1 << r;
                        gemm_n_per_block = 1 << l;
                    }
                    else{
                        gemm_m_per_block = 1 << l;
                        gemm_n_per_block = 1 << r;
                    }
                    
                    if (gemm_n % gemm_n_per_block != 0)
                        continue;
                    for (j = 5; j > 1; j--){
                        gemm_k_per_block = 1 << j;
                        if (gemm_k % gemm_k_per_block != 0)
                            continue;
                        gemm_k_global_split = if_gemm_k_global_split(arg, 
                                                     gemm_m_per_block, 
                                                     gemm_n_per_block,
                                                     gemm_k_per_block);

                        nxb = 1;
                        nxe = 1;
                        int tunable_index = -1;
                        
                        if ((x * y * stride_h * stride_w == 1) && (ho * wo % 4 == 0)){
                            nxb = 4;
                            nxe = 0;
                            tunable_index = find_tunable(tunables, gemm_m_per_block, gemm_n_per_block, gemm_k_per_block, gemm_k_global_split, nxb, nxe);
                            if (tunable_index < 0 || tunable_index >= tunables.size()){
                                nxb = 1;
                                nxe = 1;

                                // std::cout << gemm_m_per_block << ", " << gemm_n_per_block << ", " << gemm_k_per_block << std::endl;
                        
                                tunable_index = find_tunable(tunables, gemm_m_per_block, gemm_n_per_block, gemm_k_per_block, gemm_k_global_split, nxb, nxe);

                            }
                        }
                        else{
                            tunable_index = find_tunable(tunables, gemm_m_per_block, gemm_n_per_block, gemm_k_per_block, gemm_k_global_split, nxb, nxe);
                        }

                        
                        if (tunable_index < 0 || tunable_index >= tunables.size())
                            continue;

                        int log2_gemm_k_splits = 0;
                        int grid_size = group * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) * gemm_n / gemm_n_per_block;
                        for (int gs = 0; gs < 8; gs++){
                            if ((grid_size << gs) > 1200)
                                break;
                            
                            if ((n % (1 << gs)) != 0){
                                break;
                            }
                
                            if ((n >> gs) * ho * wo % gemm_k_per_block !=0){
                                break;
                            }
                            log2_gemm_k_splits = gs;
                        }

                        if (!gemm_k_global_split)
                            log2_gemm_k_splits = 0;

                        //std::cout << tunable_index << std::endl;

                        int block_size = get_block_size(&tunables[tunable_index]);

                        cur_grid_size = grid_size << log2_gemm_k_splits;

                        if (block_size >= max_block_size && cur_grid_size > max_grid_size)
                        {
                            max_block_size = block_size;
                            max_grid_size = cur_grid_size;
                            sel_index = tunable_index;
                        }
                        
                        if (max_grid_size > num_cu * 2)
                            break;
                    
                    }
                    if (max_grid_size > num_cu * 2)
                        break;
                }
                if (max_grid_size > num_cu * 2)
                    break;
                
                r++;
                l--;
            }
            if (max_grid_size > num_cu)
                break;
        }
        //std::cout << "sel_index:" << sel_index << std::endl;

        if (sel_index < 0 || sel_index >= tunables.size())
        {
            return std::string("NONE");
        }
        else
        {
            const igemm_gtc_tunable_t *tunable_return = &tunables[sel_index];
            std::cout << get_kernel_name(tunable_return) <<std::endl;

            return get_kernel_name(tunable_return);
        }
        
    }

    result_t run(const args_t *arg, const igemm_gtc_tunable_t *tunable,
                 void *p_in, void *p_wei, void *p_out) override {
        if (!tunable_is_valid(arg, tunable)) {
            result_t result;
            result.return_code = -1;
            // std::cout << "not valid tunable config." << std::endl;
            return result;
        }
        
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
        assert(c % group == 0 && k % group == 0);

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int gemm_k_global_split      = tunable->gemm_k_global_split;

        gemm_k_global_split = update_gemm_k_global_split(arg, tunable);
        
        int num_of_gemm = 1 << gemm_k_global_split;

        igemm_wrw_gtc_karg_t karg;
        size_t karg_size = sizeof(karg);
        karg.p_in          = p_in;
        karg.p_wei         = p_wei;
        karg.p_out         = p_out;
        karg.hi            = hi;
        karg.wi            = wi;
        karg.n             = n;
        karg.k             = k / group;
        karg.c             = c / group;
        karg.ho            = ho;
        karg.wo            = wo;

        karg.stride_h      = stride_h;
        karg.stride_w      = stride_w;
        karg.dilation_h    = dilation_h;
        karg.dilation_w    = dilation_w;
        karg.pad_h         = pad_h;
        karg.pad_w         = pad_w;
        karg.y             = y;
        karg.x             = x;
        karg.gemm_k_global_split  = gemm_k_global_split;
        karg.group         = group;

        //printf("gemmk split is %d\r\n", 1 << gemm_k_global_split);

        size_t block_size = get_block_size(tunable);
        size_t grid_size = get_grid_size(arg, tunable);

        hipFunction_t kernel_func;
        std::string kernel_name = get_kernel_name(tunable);
        //dump_wrw_karg(&karg);
        //printf("kernel:%s\n, block:%d, grid:%d, gemm_k_global_split:%d\n", kernel_name.c_str(), block_size, grid_size, gemm_k_global_split);
#ifdef IGEMM_SPLIT_KERNEL
        hipModule_t cur_kernel_module;
        std::string cur_kernel_hsaco = kernel_name + ".hsaco";
        HIP_CALL(hipModuleLoad(&cur_kernel_module, cur_kernel_hsaco.c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, cur_kernel_module, kernel_name.c_str()));
#else
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));
#endif
        // hipMemset(p_wei, 0x0, group * (k / group) * (c / group) * y * x * sizeof(float));

        auto wrw_prolog = gemm_k_global_split ? 
            std::function<float()>{[&]() -> float{
                hipMemset(p_wei, 0x0, group * (k / group) * (c / group) * y * x * utility_string_to_data_byte(tunable->precision));
                return .0;
            }} : 
            std::function<float()>{[&]() -> float{
                return .0;
            }};

        result_t result;
        float duration = igemm_launch_kernels_with_prolog({
                        {kernel_func, &karg, karg_size, {grid_size * block_size, 1, 1}, {block_size, 1, 1}}
                    }, wrw_prolog, this->warmup, this->repeat);

        result.return_code = 0;
        result.duration_ms = duration;
        result.gks         = gemm_k_global_split;
        result.kernel_name = kernel_name;

#ifdef IGEMM_SPLIT_KERNEL
        HIP_CALL(hipModuleUnload(cur_kernel_module));
#endif
        return result;
    }
};

#endif