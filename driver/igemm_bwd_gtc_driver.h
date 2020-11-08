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

#ifndef __IGEMM_BWD_GTC_DRIVER_H
#define __IGEMM_BWD_GTC_DRIVER_H

#include "igemm_gtc_base.h"
#include "config_parser.h"
#include "utility.h"
#include <string>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <numeric>

typedef struct {
    float *p_in;
    float *p_wei;
    float *p_out;
    int hi;
    int wi;
    int n;
    int k;
    int c;
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
    int dtile_iy;
    int dtile_ix;
    int dtile_dy;
    int dtile_dx;
    int dtile_y;
    int dtile_x;
    int dtile_h;
    int dtile_w;
    int dslice_y;
    int dslice_x;
    int dslice_h;
    int dslice_w;
    int dslice_h_left;
    int dslice_w_left;
    int __pack0;
} __attribute__((packed)) igemm_bwd_gtc_karg_t;

typedef struct {
    float *p_in;
    int hi;
    int wi;
    int n;
    int k;
    int c;
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
    int __pack_0;
#if USE_MAGIC_DIV
    uint32_t magic_0;            // denom of wi
    uint32_t magic_1;            // denom of stride_h
    uint32_t magic_2;            // denom of stride_w
    uint32_t shift_pack_0;
#endif
} __attribute__((packed)) igemm_upsampling_clear_karg_t;

static void dump_bwd_karg(igemm_bwd_gtc_karg_t * karg){
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
    std::cout<<"dtile_iy:"     <<karg->dtile_iy<<",";
    std::cout<<"dtile_ix:"     <<karg->dtile_ix<<",";
    std::cout<<"dtile_dy:"     <<karg->dtile_dy<<",";
    std::cout<<"dtile_dx:"     <<karg->dtile_dx<<",";
    std::cout<<"dtile_y:"      <<karg->dtile_y<<",";
    std::cout<<"dtile_x:"      <<karg->dtile_x<<",";
    std::cout<<"dtile_h:"      <<karg->dtile_h<<",";
    std::cout<<"dtile_w:"      <<karg->dtile_w<<",";
    std::cout<<"dslice_y:"     <<karg->dslice_y<<",";
    std::cout<<"dslice_x:"     <<karg->dslice_x<<",";
    std::cout<<"dslice_h:"     <<karg->dslice_h<<",";
    std::cout<<"dslice_w:"     <<karg->dslice_w<<",";
    std::cout<<"dslice_h_left:"<<karg->dslice_h_left<<",";
    std::cout<<"dslice_w_left:"<<karg->dslice_w_left<<",";
    std::cout<<std::endl;
}

class igemm_bwd_gtc_t {
public:
    igemm_bwd_gtc_t(){}
    ~igemm_bwd_gtc_t(){}
    std::string get_kernel_name(const igemm_gtc_tunable_t *tunable) {
#if 0
        auto tensor_layout            = tunable->tensor_layout;
        auto gemm_m_per_block         = tunable->gemm_m_per_block;
        auto gemm_n_per_block         = tunable->gemm_n_per_block;
        auto gemm_k_per_block         = tunable->gemm_k_per_block;
        auto gemm_m_per_thread        = tunable->gemm_m_per_thread;
        auto gemm_m_level0_cluster    = tunable->gemm_m_level0_cluster;
        auto gemm_m_level1_cluster    = tunable->gemm_m_level1_cluster;
        auto gemm_n_per_thread        = tunable->gemm_n_per_thread;
        auto gemm_n_level0_cluster    = tunable->gemm_n_level0_cluster;
        auto gemm_n_level1_cluster    = tunable->gemm_n_level1_cluster;
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

        assert(gemm_m_per_block % (gemm_m_per_thread * gemm_m_level0_cluster * gemm_m_level1_cluster) == 0);
        assert(gemm_n_per_block % (gemm_n_per_thread * gemm_n_level0_cluster * gemm_n_level1_cluster) == 0);
        int gemm_m_repeat = gemm_m_per_block / (gemm_m_per_thread * gemm_m_level0_cluster * gemm_m_level1_cluster);
        int gemm_n_repeat = gemm_n_per_block / (gemm_n_per_thread * gemm_n_level0_cluster * gemm_n_level1_cluster);

        int thread_tile_m = gemm_m_repeat * gemm_m_per_thread;
        int thread_tile_n = gemm_n_repeat * gemm_n_per_thread;

        assert(direction == "bwd");

        std::string kernel_prefix = std::string("igemm_") + direction + std::string("_gtc_") + 
                tensor_layout + std::string("_") + precision +
                std::string("_bx") + std::to_string(nxb) + 
                std::string("_ex") + std::to_string(nxe) + "_";

        std::string kernel_name =
            kernel_prefix +
               "bt" +
               std::to_string(gemm_m_per_block) + "x" +
               std::to_string(gemm_n_per_block) + "x" +
               std::to_string(gemm_k_per_block) + "_" +
               "tt" +
               std::to_string(thread_tile_m) + "x" +
               std::to_string(thread_tile_n) + "_" +
               "gm" + 
               std::to_string(gemm_m_repeat) + "x" +
               std::to_string(gemm_m_level0_cluster) + "x" +
               std::to_string(gemm_m_level1_cluster) + "_" +
               "gn" + 
               std::to_string(gemm_n_repeat) + "x" +
               std::to_string(gemm_n_level0_cluster) + "x" +
               std::to_string(gemm_n_level1_cluster) + "_" +
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
#endif
        return igemm_gtc_encode_kernel_name(tunable);
    }
    int get_block_size(const igemm_gtc_tunable_t *tunable) {
        if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC || tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS){
            return tunable->gemm_m_level0_cluster * tunable->gemm_n_level0_cluster *
               tunable->gemm_m_level1_cluster * tunable->gemm_n_level1_cluster;
        }else if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS){
            int waves_per_m = tunable->gemm_m_per_block / (tunable->wave_tile_m * tunable->wave_step_m * tunable->wave_repeat_m);
            int waves_per_n = tunable->gemm_n_per_block / (tunable->wave_tile_n * tunable->wave_step_n * tunable->wave_repeat_n);
            return waves_per_m * waves_per_n * AMDGPU_WAVE_SIZE;
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

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int gcd_stride_dilation_h = utility_gcd(stride_h, dilation_h);
        int gcd_stride_dilation_w = utility_gcd(stride_w, dilation_w);

        int y_tilda = stride_h / gcd_stride_dilation_h;
        int x_tilda = stride_w / gcd_stride_dilation_w;

        int y_dot = utility_integer_divide_ceil(y, y_tilda);
        int x_dot = utility_integer_divide_ceil(x, x_tilda);

        int h_tilda = ho + utility_integer_divide_ceil(dilation_h * (y - 1), stride_h);
        int w_tilda = wo + utility_integer_divide_ceil(dilation_w * (x - 1), stride_w);

        int h_tilda_left = utility_integer_divide_floor(
            utility_max(0, pad_h - dilation_h * (y_tilda - 1)), stride_h);
        int w_tilda_left = utility_integer_divide_floor(
            utility_max(0, pad_w - dilation_w * (x_tilda - 1)), stride_w);

        int h_tilda_right = utility_min(
            h_tilda, utility_integer_divide_ceil(pad_h + hi - 1, stride_h) + 1);
        int w_tilda_right = utility_min(
            w_tilda, utility_integer_divide_ceil(pad_w + wi - 1, stride_w) + 1);

        int h_tilda_slice = h_tilda_right - h_tilda_left;
        int w_tilda_slice = w_tilda_right - w_tilda_left;

        int gemm_m = c;
        int nxe = tunable->nxe;
        int nxb = tunable->nxb;
        int b = h_tilda_slice * w_tilda_slice;
        b = (nxe == 0) ? (b) : ((b + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0
        int gemm_n = n * b;

        int grid_size = utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
        int num_of_gemm = y_tilda * x_tilda;
        if(tunable->multihead)
            grid_size *= num_of_gemm;
        return grid_size;
    }

    int get_lds_size(const igemm_gtc_tunable_t *tunable) {
        // TODO: fp16/bf16, xdlops
        int lds_a = utility_string_to_data_byte(tunable->precision) * tunable->gemm_k_per_block * tunable->gemm_m_per_block;
        int lds_b = utility_string_to_data_byte(tunable->precision) * tunable->gemm_k_per_block * tunable->gemm_n_per_block;
        return 2 * utility_next_pow2(utility_next_pow2(lds_a) + utility_next_pow2(lds_b));
    }

    bool tunable_is_valid(const args_t *arg,
                          const igemm_gtc_tunable_t *tunable)
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

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int gcd_stride_dilation_h = utility_gcd(stride_h, dilation_h);
        int gcd_stride_dilation_w = utility_gcd(stride_w, dilation_w);

        int y_tilda = stride_h / gcd_stride_dilation_h;
        int x_tilda = stride_w / gcd_stride_dilation_w;

        int y_dot = utility_integer_divide_ceil(y, y_tilda);
        int x_dot = utility_integer_divide_ceil(x, x_tilda);

        int h_tilda = ho + utility_integer_divide_ceil(dilation_h * (y - 1), stride_h);
        int w_tilda = wo + utility_integer_divide_ceil(dilation_w * (x - 1), stride_w);

        int h_tilda_left = utility_integer_divide_floor(
            utility_max(0, pad_h - dilation_h * (y_tilda - 1)), stride_h);
        int w_tilda_left = utility_integer_divide_floor(
            utility_max(0, pad_w - dilation_w * (x_tilda - 1)), stride_w);

        int h_tilda_right = utility_min(
            h_tilda, utility_integer_divide_ceil(pad_h + hi - 1, stride_h) + 1);
        int w_tilda_right = utility_min(
            w_tilda, utility_integer_divide_ceil(pad_w + wi - 1, stride_w) + 1);

        int h_tilda_slice = h_tilda_right - h_tilda_left;
        int w_tilda_slice = w_tilda_right - w_tilda_left;
        int num_of_gemm = y_tilda * x_tilda;

        int gemm_m = c;
        int nxe = tunable->nxe;
        int nxb = tunable->nxb;
        int b = h_tilda_slice * w_tilda_slice;
        b = (nxe == 0) ? (b) : ((b + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0
        int gemm_n = n * b;

        if((gemm_n%gemm_n_per_block!=0)||(gemm_m%gemm_m_per_block!=0)){
            // printf("tunable_is_valid false:: gemm_n is %d, gemm_n_per_block is %d, gemm_m is %d, gemm_m_per_block is %d\n", gemm_n,gemm_n_per_block,gemm_m,gemm_m_per_block);
            return false;
        }

        if(gemm_n_per_block%tunable->nxb!=0){
            // printf("tunable_is_valid false: gemm_n_per_block%tunable->nxb!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            return false;
        }
        //# ho * wo is 4x, gemm_n is 256, hence need batch size 256/4=64x
        if(n%(gemm_n_per_block/tunable->nxb)!=0){
            // printf("tunable_is_valid false: n%(gemm_n_per_block/tunable->nxb)!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            return false;
        }
        if( (tunable->nxe == 0)&& ((h_tilda_slice * w_tilda_slice) % tunable->nxb != 0) ){
            return false;
        }
        bool gemm_k_valid = true;
        for(int gemm_id = 0; gemm_id < num_of_gemm; gemm_id++){
            int i_y_tilda = gemm_id / x_tilda;
            int i_x_tilda = gemm_id % x_tilda;
            int y_dot_slice = (i_y_tilda + 1) * y_dot <= y ? y_dot : y % y_dot;
            int x_dot_slice = (i_x_tilda + 1) * x_dot <= x ? x_dot : x % x_dot;
            int gemm_k = k * y_dot_slice * x_dot_slice;
            bool is_gemm_not_empty = gemm_k > 0;
            if(is_gemm_not_empty){
                if(gemm_k % gemm_k_per_block != 0)
                    gemm_k_valid = false;
            }
        }
        if(!gemm_k_valid)
            return false;

        if(tunable->nxe == 0){
            if((x!=1)||(y!=1)||(stride_h!=1)||(stride_w!=1)||(dilation_h!=1)||(dilation_w!=1)||(pad_h!=0)||(pad_w!=0)){
                return false;
            }
        }

        return true;
    }

    result_t run(const args_t *arg, const igemm_gtc_tunable_t *tunable,
                 hipModule_t module, float *p_in, float *p_wei, float *p_out,
                 int warmup, int repeat) {
        if (!tunable_is_valid(arg, tunable)) {
            result_t result;
            result.return_code = -1;
            //printf("this kernel can not support this config\n");
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

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int gcd_stride_dilation_h = utility_gcd(stride_h, dilation_h);
        int gcd_stride_dilation_w = utility_gcd(stride_w, dilation_w);

        int y_tilda = stride_h / gcd_stride_dilation_h;
        int x_tilda = stride_w / gcd_stride_dilation_w;

        int y_dot = utility_integer_divide_ceil(y, y_tilda);
        int x_dot = utility_integer_divide_ceil(x, x_tilda);

        int h_tilda = ho + utility_integer_divide_ceil(dilation_h * (y - 1), stride_h);
        int w_tilda = wo + utility_integer_divide_ceil(dilation_w * (x - 1), stride_w);

        int h_tilda_left = utility_integer_divide_floor(
            utility_max(0, pad_h - dilation_h * (y_tilda - 1)), stride_h);
        int w_tilda_left = utility_integer_divide_floor(
            utility_max(0, pad_w - dilation_w * (x_tilda - 1)), stride_w);

        int h_tilda_right = utility_min(
            h_tilda, utility_integer_divide_ceil(pad_h + hi - 1, stride_h) + 1);
        int w_tilda_right = utility_min(
            w_tilda, utility_integer_divide_ceil(pad_w + wi - 1, stride_w) + 1);

        int h_tilda_slice = h_tilda_right - h_tilda_left;
        int w_tilda_slice = w_tilda_right - w_tilda_left;
        int num_of_gemm = y_tilda * x_tilda;

        igemm_bwd_gtc_karg_t karg;
        size_t karg_size = sizeof(karg);
        karg.p_in          = p_in;
        karg.p_wei         = p_wei;
        karg.p_out         = p_out;
        karg.hi            = hi;
        karg.wi            = wi;
        karg.n             = n;
        karg.k             = k;
        karg.c             = c;
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

        karg.dtile_iy      = 0;
        karg.dtile_ix      = 0;
        karg.dtile_dy      = dilation_h / gcd_stride_dilation_h;
        karg.dtile_dx      = dilation_w / gcd_stride_dilation_w;
        karg.dtile_y       = y_tilda;
        karg.dtile_x       = x_tilda;
        karg.dtile_h       = h_tilda;
        karg.dtile_w       = w_tilda;
        karg.dslice_y      = 0;
        karg.dslice_x      = 0;
        karg.dslice_h      = h_tilda_slice;
        karg.dslice_w      = w_tilda_slice;
        karg.dslice_h_left = h_tilda_left;
        karg.dslice_w_left = w_tilda_left;

        bool need_set_zero = false;
        if(y < stride_h || x < stride_w || dilation_h != 1 || dilation_w != 1)
            need_set_zero = true;

        int block_size = get_block_size(tunable);
        int grid_size = get_grid_size(arg, tunable);

        igemm_upsampling_clear_karg_t ukarg;
        ukarg.p_in          = p_in;
        ukarg.hi            = hi;
        ukarg.wi            = wi;
        ukarg.n             = n;
        ukarg.k             = k;
        ukarg.c             = c;
        ukarg.ho            = ho;
        ukarg.wo            = wo;
        ukarg.stride_h      = stride_h;
        ukarg.stride_w      = stride_w;
        ukarg.dilation_h    = dilation_h;
        ukarg.dilation_w    = dilation_w;
        ukarg.pad_h         = pad_h;
        ukarg.pad_w         = pad_w;
        ukarg.y             = y;
        ukarg.x             = x;
#if USE_MAGIC_DIV
        magic_div_u32_t mdiv_0 = magic_div_u32_gen(wi);
        magic_div_u32_t mdiv_1 = magic_div_u32_gen(stride_h);
        magic_div_u32_t mdiv_2 = magic_div_u32_gen(stride_w);
        ukarg.magic_0       = mdiv_0.magic;
        ukarg.magic_1       = mdiv_1.magic;
        ukarg.magic_2       = mdiv_2.magic;
        ukarg.shift_pack_0  = magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, 0);
#endif
        size_t ukarg_size = sizeof(ukarg);

        int u_block_size = 256;
        int u_grid_size = n * c;

        hipFunction_t kernel_func;
        std::string kernel_name = get_kernel_name(tunable);
        //printf("kernel:%s\n, block:%d, grid:%d\n", kernel_name.c_str(), block_size, grid_size);
        HIP_CALL(
            hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));
        
        hipFunction_t upsampling_clear_kernel_func;
        std::string upsampling_clear_kernel_name = std::string("igemm_upsampling_clear_") + tunable->tensor_layout + "_" + tunable->precision;
        HIP_CALL(
            hipModuleGetFunction(&upsampling_clear_kernel_func, module, upsampling_clear_kernel_name.c_str()));

        auto launch_bwd = [&]() -> float{
            float ms_total = .0;
            if(need_set_zero){
                void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ukarg,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, &ukarg_size,
                          HIP_LAUNCH_PARAM_END};
                float ms = .0;
                hipEvent_t start;
                hipEvent_t stop;
                hipEventCreate(&start);
                hipEventCreate(&stop);
                HIP_CALL(hipHccModuleLaunchKernel(upsampling_clear_kernel_func, u_grid_size * u_block_size, 1, 1,
                                            u_block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, start, stop));
                hipEventSynchronize(stop);
                hipEventElapsedTime(&ms, start, stop);
                hipEventDestroy(start);
                hipEventDestroy(stop);

                ms_total += ms;
            }

            for(int gemm_id = 0; gemm_id < num_of_gemm; gemm_id++){
                int i_y_tilda = gemm_id / x_tilda;
                int i_x_tilda = gemm_id % x_tilda;
                int y_dot_slice = (i_y_tilda + 1) * y_dot <= y ? y_dot : y % y_dot;
                int x_dot_slice = (i_x_tilda + 1) * x_dot <= x ? x_dot : x % x_dot;
                int gemm_k = k * y_dot_slice * x_dot_slice;
                bool is_gemm_not_empty = gemm_k > 0;

                karg.dtile_iy = i_y_tilda;
                karg.dtile_ix = i_x_tilda;
                karg.dslice_y = y_dot_slice;
                karg.dslice_x = x_dot_slice;
                // printf("start launch id:%d(%d), block:%d, grid:%d\n", gemm_id, is_gemm_not_empty?1:0, block_size, grid_size);
                // dump_bwd_karg(&karg);

                void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                          HIP_LAUNCH_PARAM_END};
                float ms = .0;

                if(is_gemm_not_empty){
#if USE_EXT_MODULE_LAUNCH
                    hipEvent_t start;
                    hipEvent_t stop;
                    hipEventCreate(&start);
                    hipEventCreate(&stop);
                    // for hipHccModuleLaunchKernel/hipExtModuleLaunchKernel, the grid_size is in unit of workitem
                    HIP_CALL(hipHccModuleLaunchKernel(kernel_func, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, start, stop));
                    hipEventSynchronize(stop);
                    hipEventElapsedTime(&ms, start, stop);
                    hipEventDestroy(start);
                    hipEventDestroy(stop);
#else
                    gpu_timer_t timer(NULL);
                    timer.start();
                    HIP_CALL(hipModuleLaunchKernel(kernel_func, grid_size, 1, 1,
                                             block_size, 1, 1, 0, 0, NULL,
                                             (void **)&config));
                    timer.stop();
                    ms = timer.duration();
#endif
                }
                ms_total += ms;
            }
            return ms_total;
        };

        auto launch_bwd_multihead = [&]() -> float{
            float ms_total = .0;
            if(need_set_zero){
                void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ukarg,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, &ukarg_size,
                          HIP_LAUNCH_PARAM_END};
                float ms = .0;
                hipEvent_t start;
                hipEvent_t stop;
                hipEventCreate(&start);
                hipEventCreate(&stop);
                HIP_CALL(hipHccModuleLaunchKernel(upsampling_clear_kernel_func, u_grid_size * u_block_size, 1, 1,
                                            u_block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, start, stop));
                hipEventSynchronize(stop);
                hipEventElapsedTime(&ms, start, stop);
                hipEventDestroy(start);
                hipEventDestroy(stop);

                ms_total += ms;
            }
            // if 1x1 and stride/dilation > 1, will have empty gemms which will waste launch grid. better ignore that case at runtime
            int origin_grid_size = grid_size/num_of_gemm;
            karg.dtile_iy = origin_grid_size;
            karg.dtile_ix = x_dot | (y_dot<<16);
            karg.dslice_y = y % y_dot;
            karg.dslice_x = x % x_dot;
            // printf("start launch id:%d(%d), block:%d, grid:%d\n", gemm_id, is_gemm_not_empty?1:0, block_size, grid_size);
            // dump_bwd_karg(&karg);

            void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};
            float ms = .0;
#if USE_EXT_MODULE_LAUNCH
            hipEvent_t start;
            hipEvent_t stop;
            hipEventCreate(&start);
            hipEventCreate(&stop);
            // for hipHccModuleLaunchKernel/hipExtModuleLaunchKernel, the grid_size is in unit of workitem
            HIP_CALL(hipHccModuleLaunchKernel(kernel_func, grid_size * block_size, 1, 1,
                                    block_size, 1, 1, 0, 0, NULL,
                                    (void **)&config, start, stop));
            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms, start, stop);
            hipEventDestroy(start);
            hipEventDestroy(stop);
#else
            gpu_timer_t timer(NULL);
            timer.start();
            HIP_CALL(hipModuleLaunchKernel(kernel_func, grid_size, 1, 1,
                                        block_size, 1, 1, 0, 0, NULL,
                                        (void **)&config));
            timer.stop();
            ms = timer.duration();
#endif
            ms_total += ms;
            return ms_total;
        };

        auto launch_bwd_driver = [&](){
            if(tunable->multihead)
                return launch_bwd_multihead();
            else
                return launch_bwd();
        };

        for (int i = 0; i < warmup; i++) {
            launch_bwd_driver();
        }
        std::vector<float> duration_list;
        for (int i = 0; i < repeat; i++) {
            float d = launch_bwd_driver();
            duration_list.push_back(d);
        }

        // remove min and max from list, then do average
        auto imin = std::min_element(begin(duration_list), end(duration_list));
        duration_list.erase(imin);
        auto imax = std::max_element(begin(duration_list), end(duration_list));
        duration_list.erase(imax);
        assert(duration_list.size() == (repeat - 2));
        float avg_duration = std::accumulate(duration_list.begin(), duration_list.end(), (float).0) / duration_list.size();

        usleep(1000 * 5);

        result_t result;
        result.return_code = 0;
        result.duration_ms = avg_duration;
        result.kernel_name = kernel_name;
        return result;
    }
};


#endif