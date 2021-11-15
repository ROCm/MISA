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

#ifndef __IGEMM_FWD_GTC_DRIVER_H
#define __IGEMM_FWD_GTC_DRIVER_H

#include "igemm_gtc_base.h"
#include "config_parser.h"
#include "utility.h"
#include <string>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <math.h>

//#define GEMM_K_GLOBAL_SPLIT 3
#define MAX_GEMM_K_SPLITS 8

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
    int group;
#if USE_MAGIC_DIV
    uint32_t magic_0;                       // denom: n*b / n_per_block
    uint32_t magic_1;                       // denom: ((n / nb_n0) * b) / nb_n1b
    uint32_t magic_2;                       // denom: y*x, if nxe==0 not used
    uint32_t magic_3;                       // denom: x, if nxe==0 not used
    uint32_t magic_4;                       // denom: b
    uint32_t magic_5;                       // denom: wo
    uint32_t magic_6;                       // denom: n*b*k / (m_per_block*n_per_block)
    uint32_t shift_pack_0;
    uint32_t shift_pack_1;
    uint32_t ks;
#endif
} __attribute__((packed)) igemm_fwd_gtc_karg_t;

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
    int group;
#if USE_MAGIC_DIV
    uint32_t magic_0;                       // denom: (gemm_n + n_per_block - 1) / n_per_block
    uint32_t magic_1;                       // denom: ho*wo
    uint32_t magic_2;                       // denom: wo
    uint32_t magic_3;                       // denom: (gemm_m/m_per_block) * (gemm_n/n_per_block)
    uint32_t magic_4;                       // denom: x*c
    uint32_t magic_5;                       // denom: c
    uint32_t shift_pack_0;
    uint32_t shift_pack_1;
    uint32_t ks;
    uint32_t __pack_0;
#endif
} __attribute__((packed)) igemm_fwd_gtc_nhwc_karg_t;

#define IGEMM_FWD_GTC_MAX_KARG_SIZE     160

static void dump_fwd_karg(igemm_fwd_gtc_karg_t * karg){
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

static void dump_fwd_karg(igemm_fwd_gtc_nhwc_karg_t * karg){
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
    std::cout<<"group:"        <<karg->group<<",";
#if USE_MAGIC_DIV
    std::cout<<"magic_0:"      <<karg->magic_0<<",";
    std::cout<<"magic_1:"      <<karg->magic_1<<",";
    std::cout<<"magic_2:"      <<karg->magic_2<<",";
    std::cout<<"magic_3:"      <<karg->magic_3<<",";
    std::cout<<"magic_4:"      <<karg->magic_4<<",";
    std::cout<<"magic_5:"      <<karg->magic_5<<",";
    std::cout<<"shift_pack_0:" <<karg->shift_pack_0<<",";
    std::cout<<"shift_pack_1:" <<karg->shift_pack_1<<",";
#endif
    std::cout<<"ks:"           <<karg->ks;
    std::cout<<std::endl;
}

class igemm_fwd_gtc_t : public igemm_driver_base_t {
public:
    igemm_fwd_gtc_t(hipModule_t module_tensor_cast_, hipModule_t module_, driver_mode_t driver_mode_, driverDataType_t data_type_, int warmup_, int repeat_, bool verbose_)
        : igemm_driver_base_t(module_tensor_cast_, module_, driver_mode_, data_type_, warmup_, repeat_, verbose_) {}
    ~igemm_fwd_gtc_t(){}

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
    // return grid size without consideration of split k
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
        n = n/splits;   // split batch size here

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int nxe                      = tunable->nxe;
        int nxb                      = tunable->nxb;
        int b                        = ho * wo;

        if(tunable->tensor_layout == "nchw")
            b = nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0

        int gemm_m = 0;
        int gemm_n = 0;

        if(tunable->tensor_layout == "nchw"){
            gemm_m = ((k/group + gemm_m_per_block -1)/gemm_m_per_block) * gemm_m_per_block;
            gemm_n = n * b;
        }else if (tunable->tensor_layout == "nhwc"){
            gemm_m = n * b;
            // gemm_n = ((k/group + gemm_n_per_block -1)/gemm_n_per_block) * gemm_n_per_block;
            gemm_n = k / group;
        }else{
            assert(false);
        }
        size_t grid_size = static_cast<size_t>(group) * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                        utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
        assert(grid_size <= 0xffffffffUL);
        return grid_size;
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

        int need_fwd = (forw == 0 ? 1 : (forw & 1 ? 1 : 0));
        if(need_fwd == 0)
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

        int nxe                      = tunable->nxe;
        int nxb                      = tunable->nxb;
        int b                        = ho * wo;
        if(tunable->tensor_layout == "nchw")
            b = nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0

        bool unit_conv = (x==1)&&(y==1)&&(stride_h==1)&&(stride_w==1)&&(dilation_h==1)&&(dilation_w==1)&&(pad_h==0)&&(pad_w==0);

        if(tunable->tensor_layout == "nchw"){
            int gemm_m = ((k/group + gemm_m_per_block -1)/gemm_m_per_block) * gemm_m_per_block;
            int gemm_n                   = n * b;
            int gemm_k                   = (c / group) * y * x;

            // support pad to modulo, hence only check when nxe is 0
            if((gemm_n % gemm_n_per_block != 0) || (gemm_m % gemm_m_per_block != 0))
            {
                return false;
            }

            if(gemm_n_per_block % tunable->nxb != 0){
                //printf("tunable_is_valid false: gemm_n_per_block%tunable->nxb!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
                return false;
            }

            if(n % (gemm_n_per_block / tunable->nxb) != 0){
                //printf("tunable_is_valid false: n%(gemm_n_per_block/tunable->nxb)!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
                return false;
            }

            if((nxe == 0) && ((b % tunable->nxb != 0) || (gemm_k % gemm_k_per_block != 0))){
                return false;
            }

            if((nxe == 0) && !unit_conv){
                return false;
            }

            // input vector load limitation, n1b
            if(tunable->tensor_b_thread_lengths[3] > 1 && (
                !unit_conv ||
                unit_conv && (hi * wi) % tunable->tensor_b_thread_lengths[3] != 0)) {
                return false;
            }

            // weight vector load limitation, c1e
            if(tunable->tensor_a_thread_lengths[1] > 1 &&
                    gemm_k % tunable->tensor_a_thread_lengths[1] != 0){
                return false;
            }

            // if tb_c1e > 1, only 1x1 case is runable, it can not check gemm_k_padding either.
            if(tunable->tensor_b_thread_lengths[1] > 1 && (( x !=1 || y != 1)||(gemm_k % gemm_k_per_block != 0))){
                return false;
            }

            // if t_c0 > 1, need to check gemmk per block
            if(tunable->tensor_b_thread_lengths[0] > 1 && (gemm_k % gemm_k_per_block != 0)){
                return false;
            }
        }else if(tunable->tensor_layout == "nhwc"){
            //int gemm_m = n * b;
            // int gemm_n = ((k/group + gemm_n_per_block -1)/gemm_n_per_block) * gemm_n_per_block;
            //int gemm_n = k / group;
            //int gemm_k = (c / group) * y * x;

            // support pad to modulo, hence only check when nxe is 0
            //if((gemm_n % gemm_n_per_block != 0) || (gemm_m % gemm_m_per_block != 0))
            //{
            //    return false;
            //}
            if(tunable->merge_e){
                uint32_t s_move_slice_k_y = (tunable->gemm_k_per_block / ( x * (c / group))) % y;
                uint32_t s_move_slice_k_x = (tunable->gemm_k_per_block /  (c / group)) % x;
                uint32_t s_move_slice_k_c = tunable->gemm_k_per_block %  (c / group);
                if((c / group) >= 0xffffff || y >= 0xffffff || x >= 0xffffff)   // 24 bit
                    return false;
                if(s_move_slice_k_y >= 256 || s_move_slice_k_x >= 256 || s_move_slice_k_c >= 256)   // 8 bit
                    return false;
            }

            if(tunable->tensor_a_thread_lengths[1] == 1 && tunable->tensor_b_thread_lengths[1] == 1){
                ;   // if both 1, indicate padded c support
            }
            else{
                if(c >> tunable->gemm_k_global_split == 0  || ((c >> tunable->gemm_k_global_split) / group) % gemm_k_per_block != 0)
                    return false;
            }

            // if(gemm_m_per_block % tunable->nxb != 0){
            //     //printf("tunable_is_valid false: gemm_n_per_block%tunable->nxb!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            //     return false;
            // }

            // if(n % (gemm_m_per_block / tunable->nxb) != 0){
            //     //printf("tunable_is_valid false: n%(gemm_n_per_block/tunable->nxb)!=0, gemm_n_per_block is %d, tunable->nxb is %d\n", gemm_n_per_block, tunable->nxb);
            //     return false;
            // }

            // if((nxe == 0) && ((b % tunable->nxb != 0) || (gemm_k % gemm_k_per_block != 0))){
            //     return false;
            // }

            if((nxe == 0) && !unit_conv){
                return false;
            }
            if(tunable->precision == "fp16"){
                // fp16 support vector writeout by default. check get_vector_write_out()
                if(tunable->tensor_a_thread_lengths[1] == 1 && tunable->tensor_b_thread_lengths[1] == 1){
                    ;   // if both 1, k is also write out one by one
                }
                else{
                    if(tunable->gemm_k_global_split){
                        if((k / group) % 2 != 0)
                            return false;
                    }
                    else{
                        if((k / group) % utility_gcd(tunable->gemm_n_per_block, tunable->vector_store == 0 ? 8 : tunable->vector_store) != 0)
                            return false;
                    }
                }
            }

            if(tunable->precision == "int8"){
                // fp16 support vector writeout by default. check get_vector_write_out()
                if(tunable->tensor_a_thread_lengths[1] == 1 && tunable->tensor_b_thread_lengths[1] == 1){
                    ;   // if both 1, k is also write out one by one
                }
                else{
                    if(tunable->gemm_k_global_split){
                        assert(false);
                    }
                    else{
                        if((k / group) % utility_gcd(tunable->gemm_n_per_block, tunable->vector_store == 0 ? 16 : tunable->vector_store) != 0)
                            return false;
                    }
                }
            }

            // input vector load limitation, n1b
            //if(tunable->tensor_a_thread_lengths[3] > 1 && (
            //    !unit_conv ||
            //    unit_conv && (hi * wi) % tunable->tensor_a_thread_lengths[3] != 0)) {
            //    return false;
            //}

            // // weight vector load limitation, c1e
            // if(tunable->tensor_a_thread_lengths[1] > 1 &&
            //         gemm_k % tunable->tensor_a_thread_lengths[1] != 0){
            //     return false;
            // }

            // // if tb_c1e > 1, only 1x1 case is runable, it can not check gemm_k_padding either.
            // if(tunable->tensor_b_thread_lengths[1] > 1 && (( x !=1 || y != 1)||(gemm_k % gemm_k_per_block != 0))){
            //     return false;
            // }

            // // if t_c0 > 1, need to check gemmk per block
            // if(tunable->tensor_b_thread_lengths[0] > 1 && (gemm_k % gemm_k_per_block != 0)){
            //     return false;
            // }
        }else{
            assert(0);
        }
        return true;
    }

    result_t run(const args_t *arg, const igemm_gtc_tunable_t *tunable, void *p_in, void *p_wei, void *p_out, int current_gks) override {
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
        int nxe                      = tunable->nxe;
        int nxb                      = tunable->nxb;
        int b                        = ho * wo;
        if(tunable->tensor_layout == "nchw")
            b = nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0

        size_t karg_size = 0;
        uint8_t karg_buffer[IGEMM_FWD_GTC_MAX_KARG_SIZE];

        int use_workspace = 0;

        if(tunable->gemm_k_global_split == 1 && tunable->precision == "fp16" && tunable->vector_store == 1)
            use_workspace = 1;
        else if(tunable->gemm_k_global_split == 1 && tunable->precision == "bf16")
            use_workspace = 1;
        else
            use_workspace = 0;

        size_t workspace_size = get_workspace_size(arg, tunable);
        void *p_out_workspace;
        if(workspace_size == 0)
            p_out_workspace = nullptr;
        else
            HIP_CALL(hipMalloc(&p_out_workspace, workspace_size));

        if(tunable->tensor_layout == "nchw"){
            igemm_fwd_gtc_karg_t karg;
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
            karg.group         = group;

#if USE_MAGIC_DIV
            int gemm_m = ((k/group + gemm_m_per_block -1)/gemm_m_per_block) * gemm_m_per_block;
            int gemm_n = n * b;
            {
                // init magic division parameters
                uint32_t nb_n0          = tunable->tensor_b_cluster_lengths[2] * tunable->tensor_b_thread_lengths[2];
                uint32_t nb_n1b         = tunable->tensor_b_cluster_lengths[3] * tunable->tensor_b_thread_lengths[3];
                uint32_t unmerge_sub_n  = gemm_n_per_block / nxb;
                uint32_t unmerge_sub_n1 = tunable->gemm_n_unmerge_cluster == 0 ? unmerge_sub_n / nb_n0 : unmerge_sub_n;

                magic_div_u32_t mdiv_0 = magic_div_u32_gen(tunable->source_access_order == 0 ? ((n * b) / gemm_n_per_block) : ((gemm_m) / gemm_m_per_block));
                magic_div_u32_t mdiv_1 = magic_div_u32_gen(tunable->gemm_n_unmerge_cluster == 0 ? 
                                                                                b * unmerge_sub_n1 / nb_n1b :
                                                                                (n / nb_n0) * b / nb_n1b   );
                magic_div_u32_t mdiv_2 = magic_div_u32_gen(y * x);
                magic_div_u32_t mdiv_3 = magic_div_u32_gen(x);
                magic_div_u32_t mdiv_4 = magic_div_u32_gen(b);
                magic_div_u32_t mdiv_5 = magic_div_u32_gen(wo);
                magic_div_u32_t mdiv_6 = magic_div_u32_gen(utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                            utility_integer_divide_ceil(gemm_n, gemm_n_per_block));

                karg.magic_0        = mdiv_0.magic;
                karg.magic_1        = mdiv_1.magic;
                karg.magic_2        = mdiv_2.magic;
                karg.magic_3        = mdiv_3.magic;
                karg.magic_4        = mdiv_4.magic;
                karg.magic_5        = mdiv_5.magic;
                karg.magic_6        = mdiv_6.magic;
                karg.shift_pack_0   = magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, mdiv_3.shift);
                karg.shift_pack_1   = magic_div_u32_pack_shift(mdiv_4.shift, mdiv_5.shift, mdiv_6.shift, 0);
            }
#endif
            karg_size = sizeof(karg);
            memcpy(static_cast<void*>(&karg_buffer[0]), static_cast<void*>(&karg), karg_size);
        }else if(tunable->tensor_layout == "nhwc"){
            igemm_fwd_gtc_nhwc_karg_t karg;
            karg.p_in          = p_in;
            karg.p_wei         = p_wei;
            if(use_workspace == 1)
                karg.p_out     = p_out_workspace;
            else
                karg.p_out     = p_out;
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
            karg.group         = group;
            if(tunable->merge_e){
                uint32_t s_move_slice_k_y = (tunable->gemm_k_per_block / ( x * (c / group))) % y;
                uint32_t s_move_slice_k_x = (tunable->gemm_k_per_block /  (c / group)) % x;
                uint32_t s_move_slice_k_c = tunable->gemm_k_per_block %  (c / group);
                karg.y = (s_move_slice_k_y << 24) | karg.y;
                karg.x = (s_move_slice_k_x << 24) | karg.x;
                karg.c = (s_move_slice_k_c << 24) | karg.c;
            }
#if USE_MAGIC_DIV
            int gemm_m = n * ho * wo;
            int gemm_n = k / group;

            magic_div_u32_t mdiv_0 = magic_div_u32_gen(utility_integer_divide_ceil(gemm_n, gemm_n_per_block));
            magic_div_u32_t mdiv_1 = magic_div_u32_gen(ho*wo);
            magic_div_u32_t mdiv_2 = magic_div_u32_gen(wo);
            magic_div_u32_t mdiv_3 = magic_div_u32_gen(utility_integer_divide_ceil(gemm_m, gemm_m_per_block) * utility_integer_divide_ceil(gemm_n, gemm_n_per_block));
            karg.magic_0        = mdiv_0.magic;
            karg.magic_1        = mdiv_1.magic;
            karg.magic_2        = mdiv_2.magic;
            karg.magic_3        = mdiv_3.magic;
            karg.shift_pack_0   = magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, mdiv_3.shift);
            if(tunable->merge_e){
                magic_div_u32_t mdiv_4 = magic_div_u32_gen(x*(c / group));
                magic_div_u32_t mdiv_5 = magic_div_u32_gen(c / group);
                karg.magic_4           = mdiv_4.magic;
                karg.magic_5           = mdiv_5.magic;
                karg.shift_pack_1      = magic_div_u32_pack_shift(mdiv_4.shift, mdiv_5.shift, 0, 0);
            }
#endif
            karg_size = sizeof(karg);
            memcpy(static_cast<void*>(&karg_buffer[0]), static_cast<void*>(&karg), karg_size);
        } else {
            assert(0);
        }

        size_t block_size = get_block_size(tunable);

        hipFunction_t kernel_func;
        std::string kernel_name = get_kernel_name(tunable);

#ifdef IGEMM_SPLIT_KERNEL
        hipModule_t cur_kernel_module;
        std::string cur_kernel_hsaco = kernel_name + ".hsaco";
        HIP_CALL(hipModuleLoad(&cur_kernel_module, cur_kernel_hsaco.c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, cur_kernel_module, kernel_name.c_str()));
#else
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));
#endif

        // tensor cast kernel args
        tensor_cast_karg_t karg_tensor_cast;
        karg_tensor_cast.output = p_out;
        karg_tensor_cast.input = p_out_workspace; 
        karg_tensor_cast.total_length = n * k * ho * wo;

        size_t karg_tensor_cast_size = sizeof(karg_tensor_cast);

        hipFunction_t tensor_cast_func;
        if(use_workspace == 1){
            std::string tensor_cast_kernel_name = tunable->precision == "fp16" ? "tensor_cast_fp16_fp32_1d" : "tensor_cast_bf16_fp32_1d";
            HIP_CALL(hipModuleGetFunction(&tensor_cast_func, module_tensor_cast, tensor_cast_kernel_name.c_str()));
        }

        // TODO: use kernel to pre-clear when atomic
        auto fwd_prolog = tunable->gemm_k_global_split ? 
            std::function<float()>{[&]() -> float{
                if(use_workspace == 1)
                    hipMemset(p_out_workspace, 0, static_cast<size_t>(n) * splits * k * ho * wo * sizeof(float));
                else
                    hipMemset(p_out, 0, static_cast<size_t>(n) * splits * k * ho * wo * utility_string_to_data_byte(tunable->precision));
                return .0;
            }} : 
            std::function<float()>{[&]() -> float{
                return .0;
            }};

        auto fwd_postlog = use_workspace == 1 ?
            std::function<float()>{[&]() -> float{
                size_t thread_length_cast = (static_cast<size_t>(n) * k * ho * wo + 8 * 256) / (8 * 256) * (8 * 256) / 8;
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
            int selected_gks = 0;
            auto run_with_gks = [&](int _gks){
                size_t grid_size = get_grid_size(arg, tunable) * (1 << _gks);
                if(tunable->tensor_layout == "nhwc"){
                    // This is hacky, but in MIOpen we prefer a heuristic way to set gks, so ok now. 
                    igemm_fwd_gtc_nhwc_karg_t *karg_revalue = (igemm_fwd_gtc_nhwc_karg_t *)(karg_buffer);
                    karg_revalue->ks = _gks;
                    // dump_fwd_karg(karg_revalue);
                    // printf("block:%d, grid:%d\n", block_size, grid_size);
                    // fflush(stdout);
                }
                std::vector<igemm_launch_kernel_t> kernel_launchers;
                kernel_launchers.push_back({kernel_func, karg_buffer, karg_size, {grid_size * block_size, splits, 1}, {block_size, 1, 1}});
                // if(use_workspace == 1){
                //     size_t thread_length_cast = (static_cast<size_t>(n) * k * ho * wo + 8 * 256) / (8 * 256) * (8 * 256) / 8;
                //     kernel_launchers.push_back({tensor_cast_func, &karg_tensor_cast, karg_tensor_cast_size, {thread_length_cast, 1, 1}, {256, 1, 1}});
                // }
                float duration = igemm_launch_kernels(kernel_launchers, fwd_prolog, fwd_postlog, this->warmup, this->repeat);

                if(min_duration > duration){
                    min_duration = duration;
                    selected_gks = _gks;
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
            int gks   = heuristic_select_gks(arg, tunable);
            size_t grid_size = get_grid_size(arg, tunable) * (1 << gks);

            float duration = igemm_launch_kernels({
                    {kernel_func, karg_buffer, karg_size, {grid_size * block_size, splits, 1}, {block_size, 1, 1}}
                }, fwd_prolog, fwd_postlog, this->warmup, this->repeat);

            result.return_code = 0;
            result.duration_ms = duration;
            result.gks         = gks;
        }else{
            assert(0);
        }

#ifdef IGEMM_SPLIT_KERNEL
        HIP_CALL(hipModuleUnload(cur_kernel_module));
#endif
        hipFree(p_out_workspace);
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
            int c = arg->get_int("in_channels");
            int group = arg->get_int("group_count");

            int max_split_num = tunable->gemm_k_global_split == 0 ?
                0 : igemm_get_max_gks(c / group, tunable->gemm_k_per_block, this->max_gks < 0 ? MAX_GEMM_K_SPLITS : this->max_gks);
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
