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
#if USE_MAGIC_DIV
    uint32_t magic_0;           // denom: n*ho*wo / n_per_block
    uint32_t magic_1;           // denom: ((n / nb_n0) * ho*wo) / nb_n1b
    uint32_t magic_2;           // denom: y*x, if nxe==0 not used
    uint32_t magic_3;           // denom: x, if nxe==0 not used
    uint32_t magic_4;           // denom: ho*wo
    uint32_t magic_5;           // denom: wo
    uint32_t shift_pack_0;
    uint32_t shift_pack_1;
#endif
    int __pack0;
} __attribute__((packed)) igemm_fwd_gtc_karg_t;

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
    std::cout<<std::endl;
}

class igemm_fwd_gtc_t {
public:
    igemm_fwd_gtc_t(){}
    ~igemm_fwd_gtc_t(){}
    std::string get_kernel_name(const igemm_gtc_tunable_t *tunable) {
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
        int nxe                      = tunable->nxe;
        int nxb                      = tunable->nxb;
        int b                        = nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0

        int gemm_m = k;
        int gemm_n = n * b;

        int grid_size = utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
        return grid_size;
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

        int nxe                      = tunable->nxe;
        int nxb                      = tunable->nxb;
        int b                        = nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0

        int gemm_m                   = k;
        int gemm_n                   = n * b;
        int gemm_k                   = c * y * x;

        // support pad to modulo, hence only check when nxe is 0
        if((nxe == 0) && (gemm_n % gemm_n_per_block != 0)){
            return false;
        }

        if(gemm_m % gemm_m_per_block != 0){
            return false;
        }

        if(gemm_k % gemm_k_per_block != 0){
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

        if((nxe == 0) && (b % tunable->nxb != 0)){
            return false;
        }

        if(nxe == 0){
            if((x!=1)||(y!=1)||(stride_h!=1)||(stride_w!=1)||(dilation_h!=1)||(dilation_w!=1)||(pad_h!=0)||(pad_w!=0)){
                return false;
            }
        }
        if(tunable->tensor_b_thread_lengths[1] > 1 && ( x !=1 || y != 1)){
            return false;
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


        igemm_fwd_gtc_karg_t karg;
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
#if USE_MAGIC_DIV
        {
            // init magic division parameters
            uint32_t nb_n0 = tunable->tensor_b_cluster_lengths[2] * tunable->tensor_b_thread_lengths[2];
            uint32_t nb_n1b = tunable->tensor_b_cluster_lengths[3] * tunable->tensor_b_thread_lengths[3];
            uint32_t denom_0 = (tunable->source_access_order == 0) ? ((n * ho * wo) / gemm_n_per_block) : (k / gemm_m_per_block);
            uint32_t denom_1 = ((n / nb_n0) * ho * wo) / nb_n1b;
            uint32_t denom_2 = y * x;
            uint32_t denom_3 = x;
            uint32_t denom_4 = ho * wo;
            uint32_t denom_5 = wo;

            magic_div_u32_t mdiv_0 = magic_div_u32_gen(denom_0);
            magic_div_u32_t mdiv_1 = magic_div_u32_gen(denom_1);
            magic_div_u32_t mdiv_2 = magic_div_u32_gen(denom_2);
            magic_div_u32_t mdiv_3 = magic_div_u32_gen(denom_3);
            magic_div_u32_t mdiv_4 = magic_div_u32_gen(denom_4);
            magic_div_u32_t mdiv_5 = magic_div_u32_gen(denom_5);

            karg.magic_0        = mdiv_0.magic;
            karg.magic_1        = mdiv_1.magic;
            karg.magic_2        = mdiv_2.magic;
            karg.magic_3        = mdiv_3.magic;
            karg.magic_4        = mdiv_4.magic;
            karg.magic_5        = mdiv_5.magic;
            karg.shift_pack_0   = magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, mdiv_3.shift);
            karg.shift_pack_1   = magic_div_u32_pack_shift(mdiv_4.shift, mdiv_5.shift, 0, 0);
        }
#endif

        int block_size = get_block_size(tunable);
        int grid_size = get_grid_size(arg, tunable);

        hipFunction_t kernel_func;
        std::string kernel_name = get_kernel_name(tunable);
        // printf("kernel:%s\n, block:%d, grid:%d\n", kernel_name.c_str(), block_size, grid_size);
        HIP_CALL(
            hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));

        auto launch_fwd = [&]() -> float {
            // printf("launch fwd block:%d, grid:%d\n", block_size, grid_size);
            // dump_fwd_karg(&karg);
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
            return ms;
        };

        for (int i = 0; i < warmup; i++) {
            launch_fwd();
        }

        std::vector<float> duration_list;
        for (int i = 0; i < repeat; i++) {
            float d = launch_fwd();
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