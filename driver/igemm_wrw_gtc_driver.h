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

#define WRW_MAX_GEMM_K_SPLITS 10

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
    int gemm_k_per_wg;
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
    igemm_wrw_gtc_t(hipModule_t module_tensor_cast_, hipModule_t module_, driver_mode_t driver_mode_, driverDataType_t data_type_, int warmup_, int repeat_, bool verbose_)
        : igemm_driver_base_t(module_tensor_cast_, module_, driver_mode_, data_type_, warmup_, repeat_, verbose_) {}
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
		
        int block_size               = get_block_size(tunable);
        int c_vec_min                = tunable->tensor_layout == "nchw" ? 1 : (tunable->tensor_b_thread_lengths[3]);
        int max_grid_size            = 1200;

        int gemm_m = k / group;
        int c_padded = ((c / group) + c_vec_min - 1) / c_vec_min * c_vec_min;
        int gemm_n = (c_padded * y * x  + gemm_n_per_block - 1) / gemm_n_per_block * gemm_n_per_block;
        size_t grid_size = static_cast<size_t>(group) * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);

        int splits = igemm_split_batch_size(arg, utility_string_to_data_byte(tunable->precision));
        if(splits == 0){
            printf("image size (c*h*w or k*h*w) is bigger than 4g, which is not supported now\n");
            return false;
        }
        n = n/splits;   // split batch size here

        int min_n_per_block = 1;
        if(tunable->tensor_layout == "nhwc" && tunable->nxe == 1)
            min_n_per_block = tunable->tensor_a_thread_lengths[1];

        int b = ho * wo;
        if(tunable->tensor_layout == "nchw")
            b = tunable->nxe == 0 ? (ho * wo) : ((ho * wo + tunable->nxb - 1) / tunable->nxb) * tunable->nxb;

        if(tunable->tensor_layout == "nchw"){
            int gemm_k_global_splits = gemm_k_global_split == 1 ? compute_log2_gemmk_global_splits(grid_size, max_grid_size, n / min_n_per_block, b, gemm_k_per_block)
                                                                 : 0;

            int num_of_gemm = 1 << gemm_k_global_splits;
            grid_size *= num_of_gemm;
        }

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

        int nxb = tunable->nxb == 0 ? 1 : tunable->nxb;
        int b  = tunable->nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0
        int data_byte = utility_string_to_data_byte(tunable->precision);
        assert(c % group == 0 && k % group == 0);

        int splits = igemm_split_batch_size(arg, utility_string_to_data_byte(tunable->precision));
        if(splits == 0){
            printf("image size (c*h*w or k*h*w) is bigger than 4g, which is not supported now\n");
            return false;
        }
        n = n/splits;   // split batch size here

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int gemm_k_global_split      = tunable->gemm_k_global_split;
        int gemmk_blocks             = 1 << gemm_k_global_split;

        int n_per_block = n >> gemm_k_global_split;

        int gemm_n = (c / group) * y * x;
        int gemm_k = n * b;

        int nxe = tunable->nxe == 0 ? 1 : tunable->nxe;
        bool unit_conv = (x==1)&&(y==1)&&(stride_h==1)&&(stride_w==1)&&(dilation_h==1)&&(dilation_w==1)&&(pad_h==0)&&(pad_w==0);

        if(splits > 1 && gemm_k_global_split == 0)
        {
            // large tensor can only used for gkgs kernel
            return false;
        }

        if(tunable->tensor_layout == "nchw"){
            if (n % gemmk_blocks != 0){
                return false;
            }
            if(((c / group) % (gemm_n_per_block / nxe) != 0) || (((x * y) % nxe) != 0))
            {
                return false;
            }
            if (gemm_k % gemm_k_per_block != 0){
                //std::cout << __func__ << " false: gemm_n is " << gemm_n << ", gemm_n_per_block is " << gemm_n_per_block << ", gemm_m is " << gemm_m << ", gemm_m_per_block is " << gemm_m_per_block << std::endl;
                return false;
            }

            if (gemm_k_per_block % nxb != 0){
                //std::cout << __func__ << " false: gemm_n_per_block is " << gemm_n_per_block << ", nxb is " << nxb << std::endl;
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
            if (b % nxb != 0){
                //std::cout << __func__ << " false: (ho * wo) is " << (ho * wo) << ", nxb is " << nxb << std::endl;
                return false;
            }
        }
        else{
            if(data_byte == 2){
                if(c % tunable->tensor_b_thread_lengths[3] != 0){
                    return false;
                }
            }
        }

        if ((x * y * stride_h * stride_w != 1) && (tunable->nxe == 0))
            return false;

        return true;
    }

    // calculate log2_gemm_k_global_splits
    static inline int compute_gemmk_global_splits(const int& grid_size,
                                                  const int& potential_occupancy)
    {
        int num_cu;
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        num_cu = dev_prop.multiProcessorCount;
        int gemm_k_global_splits = num_cu * potential_occupancy / grid_size;
        
        return gemm_k_global_splits;
    }

    // calculate log2_gemm_k_global_splits
    static inline int compute_log2_gemmk_global_splits(const int& grid_size,
                                                       const int& max_grid_size,
                                                       const int& n,
                                                       const int& b,
                                                       const int& gemm_k_per_block)
    {
        int log2_gemm_k_global_splits = 0;
        for(int gs = 0; gs < 9; gs++)
        {
            if((grid_size << gs) > max_grid_size)
                break;

            if((n % (1 << gs)) != 0)
                break;

            //if((n >> gs) * b % gemm_k_per_block != 0)
            //    break;
            log2_gemm_k_global_splits = gs;
        }
        return log2_gemm_k_global_splits;
    }

    static int if_gemm_k_global_split(const args_t *arg,
                                  const int gemm_m_per_block,
                                  const int gemm_n_per_block,
                                  const int gemm_k_per_block,
                                  const int data_byte,
                                  const std::string tensor_layout)
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

        int splits = igemm_split_batch_size(arg, data_byte);
        assert(splits != 0);
        n = n/splits;   // split batch size here

        int gemm_m = k / group;
        int block_size = 256;
        int c_vec_min = tensor_layout == "nchw" ? 1 : (gemm_n_per_block * gemm_k_per_block / block_size);
        int c_padded = ((c / group) + c_vec_min - 1) / c_vec_min * c_vec_min;
        int gemm_n = (c_padded * y * x  + gemm_n_per_block - 1) / gemm_n_per_block * gemm_n_per_block;
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
        int data_byte = utility_string_to_data_byte(tunables[0].precision);
        std::string data_layout = tunables[0].tensor_layout;
        if(data_layout == "nhwc")
            return std::string("NONE");
        assert(c % group == 0 && k % group == 0);

        int gemm_m_per_block = 0;
        int gemm_n_per_block = 0;
        int gemm_k_per_block = 0;

        int gemm_k_global_split = 0;

        int gemm_m = k / group;
        int gemm_n = (c / group) * y * x;

        int grid_size;
        int block_size;
        int max_grid_size                 = 1200;
        int sel_index                     = -1;
        int sel_block_size                = 0;
        int sel_grid_size                 = 0;
        int sel_log2_gemm_k_global_splits = 0;
        int num_cu                        = 120;
        std::vector<int> nxb_list         = {16, 8, 4, 1};
        std::vector<int> nxe_list         = {0, 1};

        // i=log2(gemm_m_per_block*gemm_n_per_block)  to find largest kernel
        // when pack=0, means no need to search with pack image size. when pack=1, we need pack
        for(int pack = 0; pack < 2; pack++)
        {
            for (int i = 15; i > 7; i--){
                int r, l;
                r = (i + 1) >> 1;
                l = i - r;
                while (l > 1 && r < 9){
                    for (int swap = 0; swap < 2; swap++){

                        const auto gemm_m_per_block = swap == 0 ? 1 << r : 1 << l;
                        const auto gemm_n_per_block = swap == 0 ? 1 << l : 1 << r;
                    
                        if (gemm_n % gemm_n_per_block != 0)
                            continue;

                        for (int j = 5; j > 1; j--){
                            gemm_k_per_block = 1 << j;
                            for(const auto& nxe : nxe_list)
                            {
                                for(const auto& nxb : nxb_list)
                                {
                                    const auto b = pack == 0
                                        ? ho * wo
                                        : (nxe == 0 ? ho * wo : ((ho * wo + nxb - 1) / nxb) * nxb);
                                    const auto gemm_k = n * b;
                                    if(c % (gemm_n_per_block / (nxe == 0 ? 1 : nxe)) != 0)
                                        continue;
                                    if(gemm_k % gemm_k_per_block != 0)
                                        continue;

                                    if(nxe == 0)
                                    {
                                        if((x != 1) || (y != 1) || (dilation_h != 1) ||
                                            (dilation_w != 1) || (pad_h != 0) || (pad_w != 0))
                                            continue;
                                        if(stride_h != 1 || stride_w != 1)
                                        {
                                            if(nxb != 1)
                                                continue;
                                        }
                                        else
                                        {
                                            // nxe==0 case, need vector check(in nxe==0 case, nxb means
                                            // vector length)
                                            if(ho * wo % nxb != 0)
                                                continue;
                                        }
                                    }

                                    gemm_k_global_split = if_gemm_k_global_split(arg, 
                                        gemm_m_per_block, 
                                        gemm_n_per_block,
                                        gemm_k_per_block,
                                        data_byte,
                                        data_layout);

                                    int tunable_index = find_tunable(tunables, gemm_m_per_block, gemm_n_per_block, gemm_k_per_block, gemm_k_global_split, nxb, nxe);
                                    if (tunable_index < 0 || tunable_index >= tunables.size())
                                        continue;

                                    int log2_gemm_k_global_splits = 0;
                                    int grid_size = group * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) * utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
                                    int block_size = get_block_size(&tunables[tunable_index]);
                                    log2_gemm_k_global_splits = compute_log2_gemmk_global_splits(grid_size, max_grid_size, n, b, gemm_k_per_block);
                                    if (gemm_k_global_split == 0)
                                        log2_gemm_k_global_splits = 0;

                                    // in nxe==1 cases, wo%tb[1] need to be 0; when tb[1] > 1, need (pad_h+pad_w)==0
                                    if(nxe != 0)
                                    {
                                        if(wo % tunables[tunable_index].tensor_b_thread_lengths[1] != 0)
                                            continue;
                                        if(tunables[tunable_index].tensor_b_thread_lengths[1] > 1 &&
                                            (pad_h != 0 || pad_w != 0))
                                            continue;
                                    }

                                    grid_size = grid_size << log2_gemm_k_global_splits;

                                    if(block_size >= sel_block_size && grid_size > sel_grid_size)
                                    {
                                        sel_block_size                = block_size;
                                        sel_grid_size                 = grid_size;
                                        sel_index                     = tunable_index;
                                        sel_log2_gemm_k_global_splits = log2_gemm_k_global_splits;
                                        break;
                                    }
                                }
                            }
                            if (sel_grid_size > num_cu * 2)
                                break;
                        }
                        if (sel_grid_size > num_cu * 2)
                            break;
                    }
                    if (sel_grid_size > num_cu * 2)
                        break;
                    r++;
                    l--;
                }
                if (sel_grid_size > num_cu)
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
                // std::cout << get_kernel_name(tunable_return) <<std::endl;
                return get_kernel_name(tunable_return);
            }
        }
    }

    // get grid size without gks
    size_t get_cur_grid_size(const args_t *arg, const igemm_gtc_tunable_t *tunable){
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

        size_t block_size            = get_block_size(tunable);
        int c_vec_min                = tunable->tensor_layout == "nchw" ? 1 : (tunable->tensor_b_thread_lengths[3]);

        int gemm_m = k / group ;
        int c_padded = ((c / group) + c_vec_min - 1) / c_vec_min * c_vec_min;
        int gemm_n = (c_padded * y * x  + gemm_n_per_block - 1) / gemm_n_per_block * gemm_n_per_block;
        size_t grid_size = static_cast<size_t>(group) * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);

        return grid_size;
    }

    result_t run(const args_t *arg, const igemm_gtc_tunable_t *tunable,
                 void *p_in, void *p_wei, void *p_out, int current_gks) override {
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
        int data_byte = utility_string_to_data_byte(tunable->precision);
        assert(c % group == 0 && k % group == 0);

        size_t splits = igemm_split_batch_size(arg, utility_string_to_data_byte(tunable->precision));
        assert(splits != 0);
        n = n/splits;   // split batch size here

        int gemm_k_per_block         = tunable->gemm_k_per_block;

        size_t block_size            = get_block_size(tunable);
        int gemm_k_global_split      = tunable->gemm_k_global_split;

        size_t cur_grid_size = get_cur_grid_size(arg, tunable);

        int b                        = ho * wo;
        if(tunable->tensor_layout == "nchw")
            b  = tunable->nxe == 0 ? (ho * wo) : ((ho * wo + tunable->nxb - 1) / tunable->nxb) * tunable->nxb;
        int max_grid_size = 1200;
        int min_n_per_block = 1;
        if(tunable->tensor_layout == "nhwc" && tunable->nxe == 1)
            min_n_per_block = tunable->tensor_a_thread_lengths[1];

        int nb_per_block = tunable->gemm_k_per_block;
        if(tunable->tensor_layout == "nhwc" && tunable->nxe == 1)
            nb_per_block = tunable->tensor_a_cluster_lengths[1];

        size_t gemm_k_global_splits;
        if(tunable->tensor_layout == "nchw"){
            gemm_k_global_splits = gemm_k_global_split == 1 ? compute_log2_gemmk_global_splits(cur_grid_size, max_grid_size, n / min_n_per_block, b, gemm_k_per_block)
                                                                 : 0;
        }else{        
            gemm_k_global_splits = gemm_k_global_split == 1 ? compute_gemmk_global_splits(cur_grid_size, 3)
                                                                 : 0;
        }

        int use_workspace = 0;

        if(gemm_k_global_split == 1 && tunable->precision == "fp16" && (tunable->tensor_b_thread_lengths[3] == 1 || tunable->vector_store == 1))
            use_workspace = 1;
        else if(gemm_k_global_split == 1 && tunable->precision == "bf16")
            use_workspace = 1;
        else
            use_workspace = 0;

        size_t workspace_size = get_workspace_size(arg, tunable);
        void *p_wei_workspace;
        if(workspace_size == 0)
            p_wei_workspace = nullptr;
        else
            HIP_CALL(hipMalloc(&p_wei_workspace, workspace_size));

        igemm_wrw_gtc_karg_t karg;
        size_t karg_size = sizeof(karg);
        karg.p_in          = p_in;
        if(use_workspace == 1){
            karg.p_wei     = p_wei_workspace;
        } else{
            karg.p_wei     = p_wei;
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
        karg.gemm_k_global_split = gemm_k_global_splits;
        karg.group         = group;
        //karg.gemm_k_per_wg = (int)(ceil((n / min_n_per_block) * b / (float)gemm_k_global_splits));
        //karg.gemm_k_per_wg = (karg.gemm_k_per_wg + nb_per_block - 1) / nb_per_block * nb_per_block;

        //gemm_k_global_splits = (int)(ceil((n / min_n_per_block) * b / (float)(karg.gemm_k_per_wg)));

        // tensor cast kernel args
        tensor_cast_karg_t karg_tensor_cast;
        karg_tensor_cast.output = p_wei;
        karg_tensor_cast.input = p_wei_workspace; 
        karg_tensor_cast.total_length = group * (k / group) * (c / group) * y * x;

        size_t karg_tensor_cast_size = sizeof(karg_tensor_cast);

        //int block_size = get_block_size(tunable);
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

        hipFunction_t tensor_cast_func;
        if(use_workspace == 1){
            std::string tensor_cast_kernel_name = tunable->precision == "fp16" ? "tensor_cast_fp16_fp32_1d" : "tensor_cast_bf16_fp32_1d";
            HIP_CALL(hipModuleGetFunction(&tensor_cast_func, module_tensor_cast, tensor_cast_kernel_name.c_str()));
        }

        auto wrw_prolog = gemm_k_global_split ? 
            std::function<float()>{[&]() -> float{
                if(use_workspace == 1)
                    hipMemset(p_wei_workspace, 0x0, group * (k / group) * (c / group) * y * x * sizeof(float));
                else
                    hipMemset(p_wei, 0x0, group * (k / group) * (c / group) * y * x * data_byte);
                return .0;
            }} : 
            std::function<float()>{[&]() -> float{
                return .0;
            }};
        auto wrw_postlog = use_workspace == 1 ?
            std::function<float()>{[&]() -> float{
                size_t thread_length_cast = (static_cast<size_t>(group) * (k / group) * (c / group) * y * x + 8 * 256) / (8 * 256) * (8 * 256) / 8;
                igemm_launch_kernel_single(tensor_cast_func, &karg_tensor_cast, karg_tensor_cast_size, {thread_length_cast, 1, 1}, {256, 1, 1});
                return .0;
            }} :
            std::function<float()>{[&]() -> float{
                return .0;
            }};

        result_t result;

        int max_split_num = tunable->gemm_k_global_split == 0 ? 1 : (this->max_gks == -1 ? WRW_MAX_GEMM_K_SPLITS : this->max_gks);
        float min_duration = FLT_MAX;
        int selected_gkgs = 0;
        int selected_grid_size = 0;
        //max_split_num = 1;
        auto run_with_gks = [&](int _gks){
            if(tunable->tensor_layout == "nhwc"){
                //for(int gkgs = 0; gkgs < max_split_num; gkgs++){
                std::vector<igemm_launch_kernel_t> kernel_launchers;

                // This is hacky, but in MIOpen we prefer a heuristic way to set gks, so ok now. 
                gemm_k_global_splits = _gks == 0 ? 1 : compute_gemmk_global_splits(cur_grid_size, _gks);
                if(gemm_k_global_splits == 0){
                    gemm_k_global_splits = 1;
                }
                int tmp_gemm_k_per_wg = (int)(ceil(ceil(n / (float)min_n_per_block) * b / (float)gemm_k_global_splits));
                tmp_gemm_k_per_wg = (tmp_gemm_k_per_wg + nb_per_block - 1) / nb_per_block * nb_per_block;
                gemm_k_global_splits = (int)(ceil(ceil(n / (float)min_n_per_block) * b / (float)(tmp_gemm_k_per_wg)));
                karg.gemm_k_global_split = gemm_k_global_splits;
                karg.gemm_k_per_wg = tmp_gemm_k_per_wg;
                // printf("gemm_k_global_splits=%d, tmp_gemm_k_per_wg=%d\n", gemm_k_global_splits, tmp_gemm_k_per_wg);
                // fflush(stdout);

                kernel_launchers.push_back({kernel_func, &karg, karg_size, {grid_size * block_size, splits, gemm_k_global_splits}, {block_size, 1, 1}});
                // if(use_workspace == 1){
                //     size_t thread_length_cast = (static_cast<size_t>(group) * (k / group) * (c / group) * y * x + 8 * 256) / (8 * 256) * (8 * 256) / 8;
                //     kernel_launchers.push_back({tensor_cast_func, &karg_tensor_cast, karg_tensor_cast_size, {thread_length_cast, 1, 1}, {256, 1, 1}});
                // }
                float duration = igemm_launch_kernels({
                        kernel_launchers
                    }, wrw_prolog, wrw_postlog, this->warmup, this->repeat);
                if(min_duration > duration){
                    min_duration = duration;
                    selected_gkgs = _gks;
                    selected_grid_size = grid_size * gemm_k_global_splits;
                }
                // printf("block:%d, grid:%d, split:%d, duration:%f\n", block_size, grid_size, gemm_k_global_splits, duration);
                // fflush(stdout);

            }else{
                // nchw do not search for gemmksplit
                float duration = igemm_launch_kernels({
                            {kernel_func, &karg, karg_size, {grid_size * block_size, 1, 1}, {block_size, 1, 1}}
                        }, wrw_prolog, wrw_postlog, this->warmup, this->repeat);
                min_duration = duration;
                selected_gkgs = gemm_k_global_splits;
                selected_grid_size = grid_size;

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
        result.gks         = selected_gkgs;
        result.kernel_name = kernel_name;
        result.grid_size   = selected_grid_size;
		// debug section of code
#if 0
        printf("workspace debug \r\n");
        float* gemmc_host_check = (float* )malloc((1 << gemm_k_global_split) * k * c * y * x * sizeof(float));
        printf("gemmc_host_check size=%d\n", (1 << gemm_k_global_split) * k * c * y * x * sizeof(float));
        if(gemm_k_global_split > 0){
            printf("copy workspace\n");
            //hipMemcpy(gemmc_host_check, p_wei_workspace, (1 << gemm_k_global_split) * k * c * y * x * sizeof(float), hipMemcpyDeviceToHost);
            hipMemcpy(gemmc_host_check, p_wei, group * (k / group) * (c / group) * y * x * sizeof(float16), hipMemcpyDeviceToHost);
        }
        else{
            printf("copy weight\n");
            hipMemcpy(gemmc_host_check, p_wei, group * (k / group) * (c / group) * y * x * sizeof(float16), hipMemcpyDeviceToHost);
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
        printf("s_p_in=%x\n", p_in);
        printf("workspace debug end \r\n");
        free(gemmc_host_check);
#endif
#ifdef IGEMM_SPLIT_KERNEL
        HIP_CALL(hipModuleUnload(cur_kernel_module));
#endif
        if(workspace_size > 0)
            hipFree(p_wei_workspace);
        return result;
    }
    std::vector<int> get_gks_list(const args_t *arg, const igemm_gtc_tunable_t *tunable) override
    {
        if (!tunable_is_valid(arg, tunable)) {
            return std::vector<int>{0};
        }
        size_t cur_grid_size_t = get_cur_grid_size(arg, tunable);
        int hi = arg->get_int("in_h");
        int wi = arg->get_int("in_w");
        int n = arg->get_int("batchsize");

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

        int min_n_per_block = 1;
        if(tunable->tensor_layout == "nhwc" && tunable->nxe == 1)
            min_n_per_block = tunable->tensor_a_thread_lengths[1];

        int nb_per_block = tunable->gemm_k_per_block;
        if(tunable->tensor_layout == "nhwc" && tunable->nxe == 1)
            nb_per_block = tunable->tensor_a_cluster_lengths[1];

        int b = ho * wo;
        if(tunable->tensor_layout == "nchw")
            b = tunable->nxe == 0 ? (ho * wo) : ((ho * wo + tunable->nxb - 1) / tunable->nxb) * tunable->nxb;

        if(tunable->gemm_k_global_split == 0)
            return std::vector<int>{0};
        else{
            int max_split_num = tunable->gemm_k_global_split == 0 ? 1 : (this->max_gks == -1 ? WRW_MAX_GEMM_K_SPLITS : this->max_gks);

            std::vector<int> gks_list;
            std::vector<int> real_gks_list;
            for(int gks = 0; gks <= max_split_num; gks++){
                auto real_gks = compute_gemmk_global_splits(cur_grid_size_t, gks);
                if(real_gks == 0){
                    real_gks = 1;
                }
                int tmp_gemm_k_per_wg = (int)(ceil(ceil(n / (float)min_n_per_block) * b / (float)real_gks));
                tmp_gemm_k_per_wg = (tmp_gemm_k_per_wg + nb_per_block - 1) / nb_per_block * nb_per_block;
                real_gks = (int)(ceil(ceil(n / (float)min_n_per_block) * b / (float)(tmp_gemm_k_per_wg)));
                if(std::find(real_gks_list.begin(), real_gks_list.end(), real_gks) != real_gks_list.end()){
                    continue;
                }
                else{
                    real_gks_list.push_back(real_gks);
                    gks_list.push_back(gks);
                }
            }
            assert(gks_list.size() != 0);
            return gks_list;
        }
    }
};

#endif