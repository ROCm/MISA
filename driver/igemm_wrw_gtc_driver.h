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
    int gemmk_groups;
    int __pack0;
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
    std::cout<<"gemmk_groups:" <<karg->gemmk_groups;
    std::cout<<std::endl;
}

class igemm_wrw_gtc_t {
public:
    igemm_wrw_gtc_t(){}
    ~igemm_wrw_gtc_t(){}
    std::string get_kernel_name(const igemm_gtc_tunable_t *tunable) {
#if 0
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

        assert(direction == "wrw");

        std::string kernel_prefix = std::string("igemm_") + direction + std::string("_gtc_") + precision +
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
#else
        return igemm_gtc_encode_kernel_name(tunable);
#endif
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

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;
        int gemmk_groups             = tunable->gemmk_groups;

        int gemm_m = k;
        int gemm_n = c * y * x;

        int grid_size = utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
        int num_of_gemm = 1 << gemmk_groups;
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

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int gemmk_groups             = tunable->gemmk_groups;
        int gemmk_blocks             = 1 << gemmk_groups;
        
        if (n % gemmk_blocks != 0){
            return false;
        }

        int n_per_block = n >> gemmk_groups;

        int gemm_m = k;
        int gemm_n = c * y * x;
        int gemm_k = n * ho * wo;

        if ((gemm_n % gemm_n_per_block != 0) || (gemm_m % gemm_m_per_block !=0 )){
            std::cout << __func__ << " false: gemm_n is " << gemm_n << ", gemm_n_per_block is " << gemm_n_per_block << ", gemm_m is " << gemm_m << ", gemm_m_per_block is " << gemm_m_per_block << std::endl;
            return false;
        }

        if (gemm_k_per_block % tunable->nxb != 0){
            std::cout << __func__ << " false: gemm_n_per_block is " << gemm_n_per_block << ", tunable->nxb is " << tunable->nxb << std::endl;
            return false;
        }

        if ((x * y * stride_h * stride_w != 1) && (tunable->nxe == 0))
            return false;

        if ((tunable->nxb != 1) && (tunable->nxe != 0)){
            std::cout << __func__ << " false: tunable->nxb is " << tunable->nxb << ", tunable->nxe is " << tunable->nxe << std::endl;
            return false;
        }

        if ((ho * wo) % tunable->nxb != 0){
            std::cout << __func__ << " false: (ho * wo) is " << (ho * wo) << ", tunable->nxb is " << tunable->nxb << std::endl;
            return false;
        }

        int n_n0 = tunable->tensor_a_cluster_lengths[0] * tunable->tensor_a_thread_lengths[0];
        
        if (n_n0 > 1){
            if (n_per_block % (tunable->tensor_a_thread_lengths[1] * tunable->tensor_a_cluster_lengths[1] * n_n0) != 0){
                return false;
            }
        }
        else {
            if (n_per_block * ho * wo % gemm_k_per_block !=0){
                return false;
            }
        }

        return true;
    }

    int update_gemmk_groups(const args_t *arg,
                            const igemm_gtc_tunable_t *tunable)
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

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int gemmk_groups             = tunable->gemmk_groups;
        int gemmk_blocks             = 1 << gemmk_groups;

        int gemm_m = k;
        int gemm_n = c * y * x;

        int max_grid_size = 1200;

        int grid_size = utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
        
        if (gemmk_groups > 0){
            for (int i = 1; i < 6; i++){
                if ((grid_size << i) > max_grid_size){
                    gemmk_groups = i;
                    break;
                }
                int n_per_block = n >> i;
                if (n_n0 > 1){
                    if (n_per_block % (tunable->tensor_a_thread_lengths[1] * tunable->tensor_a_cluster_lengths[1] * n_n0) != 0){
                        return false;
                    }
                }
                else {
                    if (n_per_block * ho * wo % gemm_k_per_block !=0){
                        return false;
                    }
                }
            }
        }
        else{
            return 0;
        }
        
        
    }

    result_t run(const args_t *arg, const igemm_gtc_tunable_t *tunable,
                 hipModule_t module, float *p_in, float *p_wei, float *p_out,
                 int warmup, int repeat) {
        if (!tunable_is_valid(arg, tunable)) {
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

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int gemmk_groups             = tunable->gemmk_groups;

        gemmk_groups = update_gemmk_groups(arg, tunable);
        
        int num_of_gemm = 1 << gemmk_groups;

        igemm_wrw_gtc_karg_t karg;
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
        karg.gemmk_groups  = gemmk_groups;

        int block_size = get_block_size(tunable);
        int grid_size = get_grid_size(arg, tunable);

        hipFunction_t kernel_func;
        std::string kernel_name = get_kernel_name(tunable);
        printf("kernel:%s\n, block:%d, grid:%d\n", kernel_name.c_str(), block_size, grid_size);
        HIP_CALL(
            hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));

        hipMemset(p_wei, 0x0, k * c * y * x * sizeof(float));

        auto launch_wrw_driver = [&](){
            void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                              HIP_LAUNCH_PARAM_END};
            HIP_CALL(hipModuleLaunchKernel(kernel_func, grid_size, 1, 1,
                                           block_size, 1, 1, 0, 0, NULL,
                                           (void **)&config));
        };

        for (int i = 0; i < warmup; i++) {
            launch_wrw_driver();
        }
        std::vector<float> duration_list;
        for (int i = 0; i < repeat; i++) {
            gpu_timer_t timer(NULL);
            timer.start();
            launch_wrw_driver();
            timer.stop();
            float d = timer.duration();
            duration_list.push_back(d);
        }

        for (int i = 0; i < warmup; i++) {
            hipMemset(p_wei, 0x0, k * c * y * x * sizeof(float));
            launch_wrw_driver();
        }

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
        float* gemmc_host_check = (float* )malloc((1 << gemmk_groups) * k * c * y * x * sizeof(float));
        hipMemcpy(gemmc_host_check, p_wei, k * c * y * x * sizeof(float), hipMemcpyDeviceToHost);
        for (int i_check = 0; i_check < (0+block_size); i_check++)
        {
            printf("[%d]th var to monitor:[%f, %d]\r\n", i_check, gemmc_host_check[i_check], ((int *)gemmc_host_check)[i_check]);
        }
        printf("workspace debug end \r\n");
#endif
        result_t result;
        result.return_code = 0;
        result.duration_ms = avg_duration;
        result.kernel_name = kernel_name;
        return result;
    }
};


#endif