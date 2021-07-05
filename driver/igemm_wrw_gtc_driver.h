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

#define USE_HOST_REDUCTION_TO_CHECK 0

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

typedef struct {
    void* output;
    void* input;
    int out_length;
    int in_stride;
    int n_groups;
    int __pack_0;
} __attribute__((packed)) reduction_karg_t;

typedef struct {
    void* output;
    void* input;
    int thread_length;
    int total_length;
} __attribute__((packed)) tensor_cast_karg_t;

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

class igemm_wrw_gtc_t {
public:
    igemm_wrw_gtc_t(){}
    ~igemm_wrw_gtc_t(){}
    std::string get_kernel_name(const igemm_gtc_tunable_t *tunable) {
        return igemm_gtc_encode_kernel_name(tunable);
    }
    int get_block_size(const igemm_gtc_tunable_t *tunable) {
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
    int get_grid_size(const args_t *arg,
                      const igemm_gtc_tunable_t *tunable) {
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
        int e_x                      = tunable->nxe == 0 ? 1 : tunable->nxe;

        gemm_k_global_split = update_gemm_k_global_split(arg, tunable);

        int gemm_m = k / group ;
        int gemm_n = (c / group) * y * (((x + e_x - 1) / e_x) * e_x);

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
                          const igemm_gtc_tunable_t *tunable,
                          driverDataType_t driver_data_type)
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
        int b  = tunable->nxe == 0 ? (ho * wo) : ((ho * wo + tunable->nxb - 1) / tunable->nxb) * tunable->nxb;   // pad to nxb modulo when nxe != 0
        assert(c % group == 0 && k % group == 0);

        std::string precision = tunable->precision;
        if(precision == "fp16"){
            if(driver_data_type != driverHalf)
                return false;
        }
        else if(precision == "fp32"){
            if(driver_data_type != driverFloat)
                return false;
        }
        else if(precision == "bf16"){
            return false;
        }
        else{
            std::cout << std::endl;
            std::cout << precision << std::endl;
            return false;
        }

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int gemm_k_global_split      = tunable->gemm_k_global_split;
        int gemmk_blocks             = 1 << gemm_k_global_split;
        
        if (n % gemmk_blocks != 0){
            return false;
        }

        int n_per_block = n >> gemm_k_global_split;

        int gemm_m = k / group;
        int gemm_n = (c / group) * y * x;
        int gemm_k = n * b;
        int nxe = tunable->nxe == 0 ? 1 : tunable->nxe;
        bool unit_conv = (x==1)&&(y==1)&&(stride_h==1)&&(stride_w==1)&&(dilation_h==1)&&(dilation_w==1)&&(pad_h==0)&&(pad_w==0);

        // do not need to check gemm_n when n_c0 == 1; n_c0 = t_c0 * c_c0
        if((tunable->tensor_b_thread_lengths[2] * tunable->tensor_b_cluster_lengths[2] > 1) || (tunable->nxe == 0)){
            if(((c / group) % (gemm_n_per_block / nxe) != 0) || (((x * y) % nxe) != 0))
            {
                return false;
            }
        }

        if((tunable->tensor_a_thread_lengths[2] * tunable->tensor_a_thread_lengths[3] > 1) || (tunable->nxe == 0)){
            if(gemm_m % gemm_m_per_block !=0 ){
                return false;
            }
        }

        if(gemm_k % gemm_k_per_block != 0){
            return false;
        }

        if (gemm_k_per_block % tunable->nxb != 0){
            //std::cout << __func__ << " false: gemm_n_per_block is " << gemm_n_per_block << ", tunable->nxb is " << tunable->nxb << std::endl;
            return false;
        }

        // 1x1 case pad and stride condition
        if(tunable->nxe == 0){
            if(x * y != 1)
                return false;
            if(pad_h + pad_w != 0)
                return false;
            if(stride_h * stride_w != 1 && tunable->tensor_a_thread_lengths[1] > 1)
                return false;
        }
        //if ((x * y * stride_h * stride_w != 1) && (tunable->nxe == 0))
        //    return false;

        if (b % tunable->nxb != 0){
            //std::cout << __func__ << " false: (ho * wo) is " << (ho * wo) << ", tunable->nxb is " << tunable->nxb << std::endl;
            return false;
        }

        if(tunable->nxe == 0 && (ho * wo) % tunable->tensor_a_thread_lengths[1] != 0){
            return false;
        }

        if(tunable->nxe != 0 && wo % tunable->tensor_a_thread_lengths[1] != 0){
            return false;
        }

        if(tunable->nxe != 0 && tunable->tensor_a_thread_lengths[1] != 1 && (pad_h + pad_w) != 0){
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

        if (precision == "fp32"){
            // these two cases are only valid for fp32 data type
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
        int e_x = tunable->nxe == 0 ? 1 : tunable->nxe;

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int gemm_k_global_split      = tunable->gemm_k_global_split;

        int gemm_m = k / group;
        int gemm_n = (c / group) * y * (((x + e_x - 1) / e_x) * e_x);
        

        int max_grid_size = 1200;

        int grid_size = group * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
        int n_n0 = tunable->tensor_a_cluster_lengths[0] * tunable->tensor_a_thread_lengths[0];
        
        if (gemm_k_global_split > 0){
            int update_gemm_k_global_split = 1;
            for (int i = 1; i < 9; i++){
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
        // int e_x = tunable->nxe == 0 ? 1 : tunable->nxe;
        int gemm_m = k / group;
        int gemm_n = (c / group) * y * x;//(((x + e_x - 1) / e_x) * e_x);
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

#ifdef USE_HALF_HPP
    void host_wrw_reduction(float16* out, float16* input, int length, int n_groups){
        int i_len, i_group;
        float16 val_out; 
        std::cout << "vec_length: " << length << std::endl;
        for (i_len = 0; i_len < length; i_len++){
            val_out = (float16)(0.f);
            for (i_group = 0; i_group < n_groups; i_group++){
                val_out += input[i_len + i_group * length];
            }
            out[i_len] = val_out;
        }
    }
#endif

    result_t run(const args_t *arg, const igemm_gtc_tunable_t *tunable,
                 hipModule_t module, hipModule_t module_reduction, hipModule_t module_tensor_cast, void *p_in, void *p_wei, void *p_out,
                 int warmup, int repeat, driverDataType_t driver_data_type) {
        if (!tunable_is_valid(arg, tunable, driver_data_type)) {
            result_t result;
            result.return_code = -1;
            std::cout << "not valid tunable config." << std::endl;
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
        void *p_wei_workspace = nullptr;
        HIP_CALL(hipMalloc(&p_wei_workspace, 2 * num_of_gemm * k * c * y * x * sizeof(short)));

#if USE_HOST_REDUCTION_TO_CHECK
        float16 *p_wei_host_workspace = (float16 *)malloc(num_of_gemm * k * c * y * x * sizeof(float16));
        float16 *p_wei_host = (float16 *)malloc(k * c * y * x * sizeof(float16));
#endif

        igemm_wrw_gtc_karg_t karg;
        size_t karg_size = sizeof(karg);
        karg.p_in          = p_in;

        if(driver_data_type == driverFloat){
            karg.p_wei     = p_wei;
        }
        else{
            if(gemm_k_global_split > 0)
                karg.p_wei = p_wei_workspace;
            else
                karg.p_wei = p_wei;
        }
        
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

        int dim_b          = (ho * wo + tunable->nxb - 1) / tunable->nxb * tunable->nxb;
        karg.__pack_0      = dim_b == tunable->nxb ? utility_integer_divide_ceil(dim_b, wo) : ho;

        // printf("gemmk split is %d\r\n", 1 << gemm_k_global_split);

        // reduction kernel args
        size_t reduction_per_thread = 8;
        reduction_karg_t karg_reduction;
        karg_reduction.output = p_wei;
        karg_reduction.input = p_wei_workspace; 
        karg_reduction.in_stride = group * (k / group) * (c / group) * y * x;
        karg_reduction.out_length = reduction_per_thread;
        karg_reduction.n_groups = num_of_gemm;

        size_t karg_reduction_size = sizeof(karg_reduction);

        // tensor cast kernel args
        size_t cast_per_thread = 8;
        tensor_cast_karg_t karg_tensor_cast;
        karg_tensor_cast.output = p_wei;
        karg_tensor_cast.input = p_wei_workspace; 
        karg_tensor_cast.thread_length = cast_per_thread;
        karg_tensor_cast.total_length = group * (k / group) * (c / group) * y * x;

        size_t karg_tensor_cast_size = sizeof(karg_tensor_cast);


        int block_size = get_block_size(tunable);
        int grid_size = get_grid_size(arg, tunable);

        hipFunction_t kernel_func;
        std::string kernel_name = get_kernel_name(tunable);
        //dump_wrw_karg(&karg);
        printf("kernel:%s\n, block:%d, grid:%d, gemm_k_global_split:%d\n", kernel_name.c_str(), block_size, grid_size, gemm_k_global_split);
        HIP_CALL(
            hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));

        hipFunction_t reduction_func;
        HIP_CALL(
                hipModuleGetFunction(&reduction_func, module_reduction, "wrw_reduction_fp16"));

        hipFunction_t tensor_cast_func;
        HIP_CALL(
                hipModuleGetFunction(&tensor_cast_func, module_tensor_cast, "tensor_cast_fp16_fp32_1d"));

        // hipMemset(p_wei, 0x0, group * (k / group) * (c / group) * y * x * sizeof(float));

        auto launch_wrw_driver = [&](){
            void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                              HIP_LAUNCH_PARAM_END};

            void *config_reduction[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg_reduction,
                                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_reduction_size,
                                        HIP_LAUNCH_PARAM_END};

            void *config_tensor_cast[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg_tensor_cast,
                                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_tensor_cast_size,
                                        HIP_LAUNCH_PARAM_END};

            float ms = .0;
            float ms_reduction = .0;

            if(gemm_k_global_split){
                // TODO: current implementation of global split K need pre-clear the wei tensor
                // This may not be true in the future!
                hipMemset(p_wei, 0x0, group * (k / group) * (c / group) * y * x * sizeof(float));
                hipMemset(p_wei_workspace, 0x0, group * (k / group) * (c / group) * y * x * sizeof(float));
            }

#if USE_EXT_MODULE_LAUNCH
            hipEvent_t start;
            hipEvent_t stop;
            hipEventCreate(&start);
            hipEventCreate(&stop);
            hipEvent_t start_reduction;
            hipEvent_t stop_reduction;
            hipEventCreate(&start_reduction);
            hipEventCreate(&stop_reduction);

            // for hipHccModuleLaunchKernel/hipExtModuleLaunchKernel, the grid_size is in unit of workitem
            HIP_CALL(hipHccModuleLaunchKernel(kernel_func, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, start, stop));

            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms, start, stop);
            hipEventDestroy(start);
            hipEventDestroy(stop);

#if IGEMM_WRW_USE_ATOMIC_ADD
            // fp16 reduction
            if(driver_data_type == driverHalf && gemm_k_global_split > 0){
                HIP_CALL(hipHccModuleLaunchKernel(tensor_cast_func, group * (k / group) * (c / group) * y * x / cast_per_thread + 1, 1, 1,
                                            256, 1, 1, 0, 0, NULL,
                                            (void **)&config_tensor_cast, start_reduction, stop_reduction));
                hipEventSynchronize(stop_reduction);
                hipEventElapsedTime(&ms_reduction, start_reduction, stop_reduction);
                hipEventDestroy(start_reduction);
                hipEventDestroy(stop_reduction);
            }
#else
            // fp16 reduction
            if(driver_data_type == driverHalf && gemm_k_global_split > 0){
                HIP_CALL(hipHccModuleLaunchKernel(reduction_func, group * (k / group) * (c / group) * y * x / cast_per_thread, 1, 1,
                                            256, 1, 1, 0, 0, NULL,
                                            (void **)&config_reduction, start_reduction, stop_reduction));
                hipEventSynchronize(stop_reduction);
                hipEventElapsedTime(&ms_reduction, start_reduction, stop_reduction);
                hipEventDestroy(start_reduction);
                hipEventDestroy(stop_reduction);
            }
#endif
#else
            gpu_timer_t timer(NULL);
            timer.start();

            HIP_CALL(hipModuleLaunchKernel(kernel_func, grid_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config));

            timer.stop();
            ms = timer.duration();
#endif
            return ms + ms_reduction;
        };

        for (int i = 0; i < warmup; i++) {
            launch_wrw_driver();
        }
        std::vector<float> duration_list;
        for (int i = 0; i < repeat; i++) {
            float d = launch_wrw_driver();
            duration_list.push_back(d);
        }

#if USE_HOST_REDUCTION_TO_CHECK
        printf("host reduction now\n");
        HIP_CALL(hipMemcpy(p_wei_host_workspace, p_wei_workspace,
                           num_of_gemm * k * c * y * x * sizeof(float16),
                           hipMemcpyDeviceToHost));

        host_wrw_reduction(p_wei_host, p_wei_host_workspace, k * c * y * x, num_of_gemm);
        HIP_CALL(hipMemcpy(p_wei, p_wei_host,
                           k * c * y * x * sizeof(float16), hipMemcpyHostToDevice));
#endif

        // for (int i = 0; i < warmup; i++) {
        //     hipMemset(p_wei, 0x0, k * c * y * x * sizeof(float));
        //     launch_wrw_driver();
        // }

        // remove min and max from list, then do average
        auto imin = std::min_element(begin(duration_list), end(duration_list));
        duration_list.erase(imin);
        auto imax = std::max_element(begin(duration_list), end(duration_list));
        duration_list.erase(imax);
        assert(duration_list.size() == (repeat - 2));
        float avg_duration = std::accumulate(duration_list.begin(), duration_list.end(), (float).0) / duration_list.size();

        usleep(1000 * 1);

        // debug section of code
#if 0
        printf("workspace debug \r\n");
        float* gemmc_host_check = (float* )malloc((1 << gemm_k_global_split) * k * c * y * x * sizeof(float));
        if(gemm_k_global_split > 0){
            printf("copy workspace\n");
            hipMemcpy(gemmc_host_check, p_wei_workspace, (1 << gemm_k_global_split) * k * c * y * x * sizeof(float), hipMemcpyDeviceToHost);
        }
        else{
            printf("copy weight\n");
            hipMemcpy(gemmc_host_check, p_wei, group * (k / group) * (c / group) * y * x * sizeof(float), hipMemcpyDeviceToHost);
        }
        for (int i_check = 0; i_check < (0+block_size); i_check++)
        {
            float16 *gemmc_host_check_fp16 = (float16 *)gemmc_host_check;
            float16 check_num0 = gemmc_host_check_fp16[i_check*2];
            float16 check_num1 = gemmc_host_check_fp16[i_check*2+1];
            float check_num0_fp32 = (float)check_num0;
            float check_num1_fp32 = (float)check_num1;
            printf("[%d]th var to monitor:[%f, %d, fp16(%f, %f)]\r\n", i_check, gemmc_host_check[i_check], ((int *)gemmc_host_check)[i_check], check_num0_fp32, check_num1_fp32);
        }
        printf("workspace debug end \r\n");
        free(gemmc_host_check);
#endif
        hipFree(p_wei_workspace);
#if USE_HOST_REDUCTION_TO_CHECK
        free(p_wei_host_workspace);
        free(p_wei_host);
#endif
        result_t result;
        result.return_code = 0;
        result.duration_ms = avg_duration;
        result.kernel_name = kernel_name;
        return result;
    }
};


#endif