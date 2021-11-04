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

// #define IGEMM_BWD_UPSAMPLING_USE_CUSTOM_KERNEL 1
#define MAX_GEMM_K_SPLITS_BWD 8

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
    int group;
#if USE_MAGIC_DIV
    uint32_t magic_0;                       // denom: dslice_y * dslice_x
    uint32_t magic_1;                       // denom: dslice_x
    uint32_t magic_2;                       // denom: (c/group)*n*b / (gemm_m_per_block * gemm_n_per_block)
    uint32_t magic_3;                       // denom: n * b / gemm_n_per_block
    uint32_t magic_4;                       // denom: gemm_n_unmerge_cluster==0? b * unmerge_sub_n1 / n1b : (n/nb_n0 * b) / nb_n1b
    uint32_t magic_5;                       // denom: b
    uint32_t magic_6;                       // denom: nxb == 0? wi : s_dslice_w
    uint32_t shift_pack_0;
    uint32_t shift_pack_1;
    uint32_t __pack_0;
#endif
} __attribute__((packed)) igemm_bwd_gtc_karg_t;


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
    int group;
#if USE_MAGIC_DIV
    uint32_t magic_0;                       // denom: (np / gemm_n_per_block) if SOURCE_ACCESS_ORDER_GEMM_M_GEMM_N else (mp / gemm_m_per_block)
    uint32_t magic_1;                       // denom: (mp / gemm_m_per_block) * (np / gemm_n_per_block)
    uint32_t magic_2;                       // denom: dslice_w if nxe != 0 else wi
    uint32_t magic_3;                       // denom: br
    uint32_t shift_pack_0;
    uint32_t ks;
#endif
} __attribute__((packed)) igemm_bwd_gtc_nhwc_karg_t;

#ifdef IGEMM_BWD_UPSAMPLING_USE_CUSTOM_KERNEL
typedef struct {
    float *p_in;
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
    int group;
#if USE_MAGIC_DIV
    uint32_t magic_0;            // denom of wi
    uint32_t magic_1;            // denom of stride_h
    uint32_t magic_2;            // denom of stride_w
    uint32_t shift_pack_0;
#endif
} __attribute__((packed)) igemm_upsampling_clear_karg_t;
#endif

#define IGEMM_BWD_GTC_MAX_KARG_SIZE     180

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
    std::cout<<"group:"        <<karg->group<<",";
#if USE_MAGIC_DIV
    std::cout<<"magic_0:"      <<karg->magic_0<<",";
    std::cout<<"magic_1:"      <<karg->magic_1<<",";
    std::cout<<"magic_2:"      <<karg->magic_2<<",";
    std::cout<<"magic_3:"      <<karg->magic_3<<",";
    std::cout<<"magic_4:"      <<karg->magic_4<<",";
    std::cout<<"magic_5:"      <<karg->magic_5<<",";
    std::cout<<"magic_6:"      <<karg->magic_6<<",";
    std::cout<<"shift_pack_0:" <<karg->shift_pack_0<<",";
    std::cout<<"shift_pack_1:" <<karg->shift_pack_1<<",";
#endif
    std::cout<<std::endl;
}

static void dump_bwd_karg(igemm_bwd_gtc_nhwc_karg_t * karg){
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
    std::cout<<"group:"        <<karg->group<<",";
#if USE_MAGIC_DIV
    std::cout<<"magic_0:"      <<karg->magic_0<<",";
    std::cout<<"magic_1:"      <<karg->magic_1<<",";
    std::cout<<"magic_2:"      <<karg->magic_2<<",";
    std::cout<<"magic_3:"      <<karg->magic_3<<",";
    std::cout<<"shift_pack_0:" <<karg->shift_pack_0<<",";
#endif
    std::cout<<std::endl;
}


class igemm_bwd_gtc_t : public igemm_driver_base_t{
public:
    igemm_bwd_gtc_t(hipModule_t module_tensor_cast_, hipModule_t module_, driver_mode_t driver_mode_, driverDataType_t data_type_, int warmup_, int repeat_, bool verbose_)
        : igemm_driver_base_t(module_tensor_cast_, module_, driver_mode_, data_type_, warmup_, repeat_, verbose_) {}
    ~igemm_bwd_gtc_t(){}

    size_t get_block_size(const igemm_gtc_tunable_t *tunable) override {
        if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC || tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS){
            return tunable->gemm_m_level0_cluster * tunable->gemm_n_level0_cluster *
               tunable->gemm_m_level1_cluster * tunable->gemm_n_level1_cluster;
        }else if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS){
            int waves_per_m = tunable->gemm_m_per_block / (tunable->wave_tile_m * tunable->wave_step_m * tunable->wave_repeat_m);
            int waves_per_n = tunable->gemm_n_per_block / (tunable->wave_tile_n * tunable->wave_step_n * tunable->wave_repeat_n);
            return waves_per_m * waves_per_n * AMDGPU_WAVE_SIZE;
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

        size_t splits = igemm_split_batch_size(arg, utility_string_to_data_byte(tunable->precision));
        n = n / splits;   // split batch size here

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

        size_t grid_size = 0;
        if(tunable->tensor_layout == "nchw"){
            int nxe = tunable->nxe;
            int nxb = tunable->nxb;
            int b = h_tilda_slice * w_tilda_slice;
            b = (nxe == 0) ? (b) : ((b + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0
            int gemm_m = c / group;
            int gemm_n = n * b;
            grid_size = static_cast<size_t>(group) * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
        }else if (tunable->tensor_layout == "nhwc"){
            int gemm_m = n * h_tilda_slice * w_tilda_slice;
            int gemm_n = c / group;
            grid_size = static_cast<size_t>(group) * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
        }
        
        int num_of_gemm = y_tilda * x_tilda;
        if(tunable->multihead)
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

        int need_bwd = (forw == 0 ? 1 : (forw & 2 ? 1 : 0));
        if(need_bwd == 0)
            return false;

        assert(c % group == 0 && k % group == 0);

        size_t splits = igemm_split_batch_size(arg, utility_string_to_data_byte(tunable->precision));
        if(splits == 0){
            printf("image size (c*h*w) is bigger than 4g, which is not supported now\n");
            return false;
        }
        n = n/splits;   // split batch size here

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

        bool unit_conv = (x==1)&&(y==1)&&(stride_h==1)&&(stride_w==1)&&(dilation_h==1)&&(dilation_w==1)&&(pad_h==0)&&(pad_w==0);

        if(tunable->tensor_layout == "nchw"){
            int nxe = tunable->nxe;
            int nxb = tunable->nxb;
            int b = h_tilda_slice * w_tilda_slice;
            b = (nxe == 0) ? (b) : ((b + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0
            int gemm_n = n * b;
            if(gemm_n%gemm_n_per_block!=0){
                // printf("tunable_is_valid false:: gemm_n is %d, gemm_n_per_block is %d, gemm_m is %d, gemm_m_per_block is %d\n", gemm_n,gemm_n_per_block,gemm_m,gemm_m_per_block);
                return false;
            }
            if((tunable->tensor_a_thread_lengths[0] != 1 || tunable->tensor_a_thread_lengths[1] != 1 ||
                tunable->tensor_b_thread_lengths[0] != 1 || tunable->tensor_b_thread_lengths[1] != 1) && (k / group) % gemm_k_per_block != 0)
                return false;

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
                int y_dot_slice = utility_integer_divide_ceil(y - i_y_tilda, y_tilda);
                int x_dot_slice = utility_integer_divide_ceil(x - i_x_tilda, x_tilda);

                int gemm_k = (k / group) * y_dot_slice * x_dot_slice;
                bool is_gemm_not_empty = gemm_k > 0 && y_dot_slice > 0 && x_dot_slice > 0;
                if(is_gemm_not_empty){
                    if(gemm_k % gemm_k_per_block != 0)
                        gemm_k_valid = false;
                }
            }
            if(!gemm_k_valid)
                return false;

            if(tunable->nxe == 0 && !unit_conv){
                return false;
            }

            // output vector load limitation, n1b
            if(tunable->tensor_b_thread_lengths[3] > 1 && (
                !unit_conv ||
                unit_conv && (ho * wo) % tunable->tensor_b_thread_lengths[3] != 0)) {
                return false;
            }
        } else if (tunable->tensor_layout == "nhwc"){
            if(tunable->tensor_a_thread_lengths[1] == 1){
                ;   // if output k 1, indicate padded k support
            }
            else{
                if(k >> tunable->gemm_k_global_split == 0 || ((k >> tunable->gemm_k_global_split) / group) % gemm_k_per_block != 0)
                    return false;
            }
            if((tunable->nxe == 0) && !unit_conv){
                return false;
            }

            if(tunable->precision == "fp16"){
                // fp16 support vector writeout by default. check get_vector_write_out()
                if(tunable->tensor_a_thread_lengths[1] == 1 && tunable->tensor_b_thread_lengths[3] == 1 && tunable->merge_e && !tunable->gemm_k_global_split){
                    ;   // only case that support every config
                        // thread_k, thread_c is one, merge_e, and not gks
                }
                else{
                    if(tunable->gemm_k_global_split){
                        if((c / group) % 2 != 0)
                            return false;
                    }
                    else{
                        if((c / group) % utility_gcd(tunable->gemm_n_per_block, tunable->vector_store == 0 ? 8 : tunable->vector_store) != 0)
                            return false;
                    }
                }
            }

            if(tunable->precision == "int8"){
                // fp16 support vector writeout by default. check get_vector_write_out()
                if(tunable->tensor_a_thread_lengths[1] == 1){
                    ;   // if both 1, c is also write out one by one
                }
                else{
                    if(tunable->gemm_k_global_split){
                        assert(false);
                    }
                    else{
                        if((c / group) % utility_gcd(tunable->gemm_n_per_block, tunable->vector_store == 0 ? 16 : tunable->vector_store) != 0)
                            return false;
                    }
                }
            }
        }

        return true;
    }

    result_t run(const args_t *arg, const igemm_gtc_tunable_t *tunable,
                 void *p_in, void *p_wei, void *p_out, int current_gks) override {
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
        int group = arg->get_int("group_count");
        int data_byte = utility_string_to_data_byte(tunable->precision);

        assert(c % group == 0 && k % group == 0);

        size_t splits = igemm_split_batch_size(arg, utility_string_to_data_byte(tunable->precision));
        n = n/splits;   // split batch size here

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

        int use_workspace = 0;

        if(tunable->gemm_k_global_split == 1 && tunable->precision == "fp16" && tunable->vector_store == 1)
            use_workspace = 1;
        else if(tunable->gemm_k_global_split == 1 && tunable->precision == "bf16")
            use_workspace = 1;
        else
            use_workspace = 0;

        size_t workspace_size = get_workspace_size(arg, tunable);
        void *p_in_workspace;
        if(workspace_size == 0)
            p_in_workspace = nullptr;
        else
            HIP_CALL(hipMalloc(&p_in_workspace, workspace_size));

        size_t karg_size = 0;
        uint8_t karg_buffer[IGEMM_BWD_GTC_MAX_KARG_SIZE];
#if USE_MAGIC_DIV
        magic_div_u32_t mdiv_0, mdiv_1, mdiv_2, mdiv_3, mdiv_4, mdiv_5, mdiv_6;
#endif
        if(tunable->tensor_layout == "nchw"){
            int nxe = tunable->nxe;
            int nxb = tunable->nxb;
            int b = h_tilda_slice * w_tilda_slice;
            b = (nxe == 0) ? (b) : ((b + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0
            int gemm_m = c / group;
            int gemm_n = n * b;

            igemm_bwd_gtc_karg_t karg;
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
            karg.group         = group;
#if USE_MAGIC_DIV
            // init magic division parameters
            uint32_t nb_n0          = tunable->tensor_b_cluster_lengths[2] * tunable->tensor_b_thread_lengths[2];
            uint32_t nb_n1b         = tunable->tensor_b_cluster_lengths[3] * tunable->tensor_b_thread_lengths[3];
            uint32_t unmerge_sub_n  = gemm_n_per_block / nxb;
            uint32_t unmerge_sub_n1 = tunable->gemm_n_unmerge_cluster == 0 ? unmerge_sub_n / nb_n0 : unmerge_sub_n;

            mdiv_2  = magic_div_u32_gen(utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                        utility_integer_divide_ceil(gemm_n, gemm_n_per_block));
            mdiv_3  = magic_div_u32_gen((n * b) / gemm_n_per_block);
            mdiv_4  = magic_div_u32_gen(tunable->gemm_n_unmerge_cluster == 0 ?
                                                                    b * unmerge_sub_n1 / nb_n1b :
                                                                    (n / nb_n0 * b) / nb_n1b);
            mdiv_5  = magic_div_u32_gen(b);
            mdiv_6  = magic_div_u32_gen(w_tilda_slice);


            // karg.magic_0        = mdiv_0.magic;
            // karg.magic_1        = mdiv_1.magic;
            karg.magic_2        = mdiv_2.magic;
            karg.magic_3        = mdiv_3.magic;
            karg.magic_4        = mdiv_4.magic;
            karg.magic_5        = mdiv_5.magic;
            karg.magic_6        = mdiv_6.magic;
            // karg.shift_pack_0   = magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, mdiv_3.shift);
            // karg.shift_pack_1   = magic_div_u32_pack_shift(mdiv_4.shift, mdiv_5.shift, mdiv_6.shift, 0);
#endif
            karg_size = sizeof(karg);
            memcpy(static_cast<void*>(&karg_buffer[0]), static_cast<void*>(&karg), karg_size);
        }else if(tunable->tensor_layout == "nhwc"){
            int gemm_m = n * h_tilda_slice * w_tilda_slice;
            int gemm_n = c / group;

            igemm_bwd_gtc_nhwc_karg_t karg;
            if(use_workspace == 1)
                karg.p_in      = p_in_workspace;
            else
                karg.p_in      = p_in;
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
            karg.group         = group;
#if USE_MAGIC_DIV
            mdiv_0  = magic_div_u32_gen(tunable->source_access_order == 0? utility_integer_divide_ceil(gemm_n, gemm_n_per_block): 
                                                                utility_integer_divide_ceil(gemm_m, gemm_m_per_block));
            mdiv_1  = magic_div_u32_gen(utility_integer_divide_ceil(gemm_n, gemm_n_per_block) * utility_integer_divide_ceil(gemm_m, gemm_m_per_block));
            mdiv_2  = magic_div_u32_gen(tunable->nxe != 0? w_tilda_slice : wi);
            mdiv_3  = magic_div_u32_gen(h_tilda_slice * w_tilda_slice);

            karg.magic_0        = mdiv_0.magic;
            karg.magic_1        = mdiv_1.magic;
            karg.magic_2        = mdiv_2.magic;
            karg.magic_3        = mdiv_3.magic;

            karg.shift_pack_0  = magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, mdiv_3.shift);
#endif
            karg_size = sizeof(karg);
            memcpy(static_cast<void*>(&karg_buffer[0]), static_cast<void*>(&karg), karg_size);
        }
        bool need_set_zero = false;
        if(y < stride_h || x < stride_w || dilation_h != 1 || dilation_w != 1)
            need_set_zero = true;

        size_t block_size = get_block_size(tunable);

#ifdef IGEMM_BWD_UPSAMPLING_USE_CUSTOM_KERNEL
        igemm_upsampling_clear_karg_t ukarg;
        ukarg.p_in          = p_in;
        ukarg.hi            = hi;
        ukarg.wi            = wi;
        ukarg.n             = n;
        ukarg.k             = k / group;
        ukarg.c             = c / group;
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
        ukarg.group         = group;
#if USE_MAGIC_DIV
        magic_div_u32_t umdiv_0 = magic_div_u32_gen(wi);
        magic_div_u32_t umdiv_1 = magic_div_u32_gen(stride_h);
        magic_div_u32_t umdiv_2 = magic_div_u32_gen(stride_w);
        ukarg.magic_0       = umdiv_0.magic;
        ukarg.magic_1       = umdiv_1.magic;
        ukarg.magic_2       = umdiv_2.magic;
        ukarg.shift_pack_0  = magic_div_u32_pack_shift(umdiv_0.shift, umdiv_1.shift, umdiv_2.shift, 0);
#endif
        size_t ukarg_size = sizeof(ukarg);

        int u_block_size = 256;
        int u_grid_size = n * (c / group) * group;
#endif

        hipFunction_t kernel_func;
        std::string kernel_name = get_kernel_name(tunable);
        // printf("kernel:%s\n, block:%d, grid:%d\n", kernel_name.c_str(), block_size, grid_size);
#ifdef IGEMM_SPLIT_KERNEL
        hipModule_t cur_kernel_module;
        std::string cur_kernel_hsaco = kernel_name + ".hsaco";
        HIP_CALL(hipModuleLoad(&cur_kernel_module, cur_kernel_hsaco.c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, cur_kernel_module, kernel_name.c_str()));
#else
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));
#endif

#ifdef IGEMM_BWD_UPSAMPLING_USE_CUSTOM_KERNEL
        hipFunction_t upsampling_clear_kernel_func;
        std::string upsampling_clear_kernel_name = std::string("igemm_upsampling_clear_") + tunable->tensor_layout + "_" + tunable->precision;
        HIP_CALL(
            hipModuleGetFunction(&upsampling_clear_kernel_func, module, upsampling_clear_kernel_name.c_str()));
#endif

        // tensor cast kernel args
        tensor_cast_karg_t karg_tensor_cast;
        int length_per_thread = 8;
        karg_tensor_cast.output = p_in;
        karg_tensor_cast.input = p_in_workspace; 
        karg_tensor_cast.total_length = n * c * hi * wi;

        size_t karg_tensor_cast_size = sizeof(karg_tensor_cast);

        hipFunction_t tensor_cast_func;
        if(use_workspace == 1){
            std::string tensor_cast_kernel_name = tunable->precision == "fp16" ? "tensor_cast_fp16_fp32_1d" : "tensor_cast_bf16_fp32_1d";
            HIP_CALL(hipModuleGetFunction(&tensor_cast_func, module_tensor_cast, tensor_cast_kernel_name.c_str()));
        }

        auto bwd_prolog = (need_set_zero || tunable->gemm_k_global_split)? 
            std::function<float()>{[&]() -> float{
                if(use_workspace == 1)
                    hipMemset(p_in_workspace, 0, static_cast<size_t>(splits)*n*c*hi*wi*sizeof(float));
                else
                    hipMemset(p_in, 0, static_cast<size_t>(splits)*n*c*hi*wi*utility_string_to_data_byte(tunable->precision));
                return .0;
            }} : 
            std::function<float()>{[&]() -> float{
                return .0;
            }};
        auto bwd_postlog = use_workspace == 1 ?
            std::function<float()>{[&]() -> float{
                size_t thread_length_cast = (static_cast<size_t>(n) * c * hi * wi + 8 * 256) / (8 * 256) * (8 * 256) / 8;
                igemm_launch_kernel_single(tensor_cast_func, &karg_tensor_cast, karg_tensor_cast_size, {thread_length_cast, 1, 1}, {256, 1, 1});
                return .0;
            }} :
            std::function<float()>{[&]() -> float{
                return .0;
            }};

        result_t result;
        result.kernel_name = kernel_name;

        if(this->driver_mode == driver_mode_normal){
            float min_duration = FLT_MAX;
            float duration = .0;
            int selected_gks = 0;
            auto run_with_gks = [&](int _gks){
                size_t grid_size = get_grid_size(arg, tunable) * (1 << _gks);
                if(tunable->multihead){
                    std::vector<igemm_launch_kernel_t> kernels;

                    if(tunable->tensor_layout == "nhwc"){
                        int gemm_m = n * h_tilda_slice * w_tilda_slice;
                        int gemm_n = c / group;

                        igemm_bwd_gtc_nhwc_karg_t * karg = reinterpret_cast<igemm_bwd_gtc_nhwc_karg_t*>(&karg_buffer[0]);
                        magic_div_u32_t mdiv_x_tilda = magic_div_u32_gen(x_tilda);
                        magic_div_u32_t mdiv_y_tilda = magic_div_u32_gen(y_tilda);
                        magic_div_u32_t mdiv_group_mn = magic_div_u32_gen(group * utility_integer_divide_ceil(gemm_n, gemm_n_per_block) * utility_integer_divide_ceil(gemm_m, gemm_m_per_block));
                        karg->dtile_iy = num_of_gemm > 1 ? mdiv_x_tilda.magic : 0;
                        karg->dtile_ix = num_of_gemm > 1 ? mdiv_x_tilda.shift : 0;
                        karg->dslice_y = num_of_gemm > 1 ? mdiv_y_tilda.magic : y;
                        karg->dslice_x = num_of_gemm > 1 ? mdiv_y_tilda.shift : x;
                        karg->dtile_h  = num_of_gemm > 1 ? mdiv_group_mn.magic : h_tilda;
                        karg->dtile_w  = num_of_gemm > 1 ? mdiv_group_mn.shift : w_tilda;
                        karg->ks       = _gks;

                        kernels.push_back({kernel_func, karg_buffer, karg_size, std::vector<size_t>{grid_size * block_size, splits, 1}, std::vector<size_t>{block_size, 1, 1}});
                        // if(use_workspace == 1){
                        //     size_t thread_length_cast = (static_cast<size_t>(n) * c * hi * wi + 8 * 256) / (8 * 256) * (8 * 256) / 8;
                        //     kernels.push_back({tensor_cast_func, &karg_tensor_cast, karg_tensor_cast_size, {thread_length_cast, 1, 1}, {256, 1, 1}});
                        // }
                    }else{
                        assert(0);
                    }

                    // dump_bwd_karg(reinterpret_cast<igemm_bwd_gtc_nhwc_karg_t*>(&karg_buffer[0]));
                    duration = igemm_launch_kernels(kernels, bwd_prolog, bwd_postlog, warmup, repeat);

                    if(min_duration > duration){
                        min_duration = duration;
                        selected_gks = _gks;
                    }
                }else{
                    std::vector<igemm_launch_kernel_t> kernels;
                    uint8_t * kargs = num_of_gemm != 0 ? (uint8_t*)malloc(num_of_gemm * karg_size) : NULL;
                    int valid_kernel_index = 0;
                    for(int gemm_id = 0; gemm_id < num_of_gemm; gemm_id++){
                        int i_y_tilda = gemm_id / x_tilda;
                        int i_x_tilda = gemm_id % x_tilda;
                        int y_dot_slice = utility_integer_divide_ceil(y - i_y_tilda,  y_tilda);
                        int x_dot_slice = utility_integer_divide_ceil(x - i_x_tilda,  x_tilda);

                        int gemm_k = (k / group) * y_dot_slice * x_dot_slice;
                        bool is_gemm_not_empty = gemm_k > 0 && y_dot_slice > 0 && x_dot_slice > 0;
                        if(is_gemm_not_empty){
                            if(tunable->tensor_layout == "nchw"){
                                igemm_bwd_gtc_karg_t * karg = reinterpret_cast<igemm_bwd_gtc_karg_t*>(&kargs[valid_kernel_index * karg_size]);
                                memcpy(karg, karg_buffer, karg_size);
                                karg->dtile_iy = i_y_tilda;
                                karg->dtile_ix = i_x_tilda;
                                karg->dslice_y = y_dot_slice;
                                karg->dslice_x = x_dot_slice;
#if USE_MAGIC_DIV
                                mdiv_0  = is_gemm_not_empty ? magic_div_u32_gen(y_dot_slice * x_dot_slice) : magic_div_u32_t({0, 0});
                                mdiv_1  = is_gemm_not_empty ? magic_div_u32_gen(x_dot_slice) : magic_div_u32_t({0, 0});
                                karg->magic_0        = mdiv_0.magic;
                                karg->magic_1        = mdiv_1.magic;

                                karg->shift_pack_0   = magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, mdiv_3.shift);
                                karg->shift_pack_1   = magic_div_u32_pack_shift(mdiv_4.shift, mdiv_5.shift, mdiv_6.shift, 0);
#endif
                            }else if(tunable->tensor_layout == "nhwc"){
                                igemm_bwd_gtc_nhwc_karg_t * karg = reinterpret_cast<igemm_bwd_gtc_nhwc_karg_t*>(&kargs[valid_kernel_index * karg_size]);
                                memcpy(karg, karg_buffer, karg_size);
                                karg->dtile_iy = i_y_tilda;
                                karg->dtile_ix = i_x_tilda;
                                karg->dslice_y = y_dot_slice;
                                karg->dslice_x = x_dot_slice;
                                karg->ks       = _gks;
                            }

                            kernels.push_back({kernel_func, (void*)&kargs[valid_kernel_index * karg_size], karg_size, std::vector<size_t>{grid_size * block_size, splits, 1}, std::vector<size_t>{block_size, 1, 1}});

                            valid_kernel_index++;
                        }
                    }
                    //if(use_workspace == 1){
                    //    size_t thread_length_cast = (static_cast<size_t>(n) * c * hi * wi + 8 * 256) / (8 * 256) * (8 * 256) / 8;
                    //        kernels.push_back({tensor_cast_func, &karg_tensor_cast, karg_tensor_cast_size, {thread_length_cast, 1, 1}, {256, 1, 1}});
                    //    valid_kernel_index++;
                    //}
                    // dump_bwd_karg(reinterpret_cast<igemm_bwd_gtc_nhwc_karg_t*>(&kargs[0]));

                    assert(kernels.size() == valid_kernel_index);
                    
                    duration = igemm_launch_kernels(kernels, bwd_prolog, bwd_postlog, warmup, repeat);

                    if(min_duration > duration){
                        min_duration = duration;
                        selected_gks = _gks;
                    }
                    if(kargs)
                        free(kargs);
                }
            };
            if(current_gks != -1){
                run_with_gks(current_gks);
            }else{
                std::vector<int> all_gks = get_gks_list(arg, tunable);
                for(int gks : all_gks){
                    run_with_gks(gks);
                }
            }
            result.return_code = 0;
            result.duration_ms = min_duration;
            result.gks         = selected_gks;
        }else if(this->driver_mode == driver_mode_heuristic){
            assert(0);
        }

#ifdef IGEMM_SPLIT_KERNEL
        HIP_CALL(hipModuleUnload(cur_kernel_module));
#endif
        hipFree(p_in_workspace);
        usleep(1000 * 5);
        return result;
    }
    std::vector<int> get_gks_list(const args_t *arg, const igemm_gtc_tunable_t *tunable) override
    {
        if (!tunable_is_valid(arg, tunable)) {
            return std::vector<int>{0};
        }

        if(tunable->gemm_k_global_split == 0)
            return std::vector<int>{0};
        else{
            int k = arg->get_int("out_channels");
            int group = arg->get_int("group_count");

            int max_split_num = tunable->gemm_k_global_split == 0 ?
                0 : igemm_get_max_gks(k / group, tunable->gemm_k_per_block, this->max_gks == -1? MAX_GEMM_K_SPLITS_BWD : this->max_gks);
            if(tunable->gemm_k_global_split == 1 && tunable->merge_e == 1){
                // this is merge_e, which indicate support padding k
                int padded_k_num = ((k / group) + tunable->gemm_k_per_block - 1) / tunable->gemm_k_per_block;
                int k_pow2 = (int)log2(utility_prev_pow2(padded_k_num));
                max_split_num = k_pow2 <= (this->max_gks == -1? MAX_GEMM_K_SPLITS_BWD : this->max_gks) ? k_pow2 : (this->max_gks == -1? MAX_GEMM_K_SPLITS_BWD : this->max_gks);
            }
            int start_gks = (tunable->gemm_k_global_split == 0 || max_split_num == 0)? 0 : 1;

            std::vector<int> gks_list;
            for(int gks = start_gks; gks <= max_split_num; gks++)
                gks_list.push_back(gks);
            assert(gks_list.size() != 0);
            return gks_list;
        }
    }
};


#endif