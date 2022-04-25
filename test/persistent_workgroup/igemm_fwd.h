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

#ifndef __DIRECT_CONV_DRIVER_H
#define __DIRECT_CONV_DRIVER_H


#include <string>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include "utility.h"

#ifndef HIP_CALL
#define HIP_CALL(call)                                                         \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            printf("[hiperror](%d) fail to call %s,(%s)", (int)err, #call,     \
                   hipGetErrorString(err));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)
#endif

static inline size_t gpu_conv_out_size(size_t in_size, size_t pad,
                                         size_t dilation, size_t ksize,
                                         size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

typedef struct {
    void    *p_in;
    void    *p_wei;
    void    *p_out;

    uint32_t tile_hw;
    uint32_t ntile_hw;
    uint32_t hi;
    uint32_t wi;
    uint32_t n;
    uint32_t k;                      // this is indeed k_per_group
    uint32_t c;                      // this is indeed c_per_group
    uint32_t group;
    uint32_t ks;

    uint32_t ho;
    uint32_t wo;
    uint32_t stride_hw;
    uint32_t dilation_hw;
    uint32_t pad_hw;
    uint32_t wei_hw;
    uint32_t move_slice_k;

#if USE_MAGIC_DIV
    uint32_t magic_0;
    uint32_t magic_1;
    uint32_t magic_2;
    uint32_t magic_3;
    uint32_t magic_4;
    uint32_t magic_5;
    uint32_t magic_6;
    uint32_t magic_7;
    uint32_t shift_pack_0;
    uint32_t shift_pack_1;
#endif

#if USE_PERSISTENT_WORKGROUP
    uint32_t workgroup_stride;
    uint32_t total_workgroups;
                                // uint32_t    se_per_soc  // 
                                // uint32_t    wgp_per_se  // new_gid = gid % wgp_per_se * se_per_soc + gid / wgp_per_se
    uint32_t magic_se_per_soc;
    uint32_t shift_se_per_soc;
    uint32_t magic_wgp_per_se;
    uint32_t shift_wgp_per_se;
    uint32_t se_per_soc;
    uint32_t wgp_per_se;
#endif
} __attribute__((packed)) igemm_fwd_gtc_nchwc_karg_t;

static void dump_fwd_karg(igemm_fwd_gtc_nchwc_karg_t * karg){
    std::cout<<"p_in:"         <<karg->p_in<<",";
    std::cout<<"p_wei:"        <<karg->p_wei<<",";
    std::cout<<"p_out:"        <<karg->p_out<<",";
    std::cout<<"tile_hw:"      <<std::hex<<karg->tile_hw<<std::dec<<",";
    std::cout<<"ntile_hw:"     <<std::hex<<karg->ntile_hw<<std::dec<<",";
    std::cout<<"hi:"           <<karg->hi<<",";
    std::cout<<"wi:"           <<karg->wi<<",";
    std::cout<<"n:"            <<karg->n<<",";
    std::cout<<"k:"            <<karg->k<<",";
    std::cout<<"c:"            <<karg->c<<",";
    std::cout<<"group:"        <<karg->group<<",";
    std::cout<<"ks:"           <<karg->ks<<",";
    std::cout<<"ho:"           <<karg->ho<<",";
    std::cout<<"wo:"           <<karg->wo<<",";
    std::cout<<"stride_hw:"    <<std::hex<<karg->stride_hw<<std::dec<<",";
    std::cout<<"dilation_hw:"  <<std::hex<<karg->dilation_hw<<std::dec<<",";
    std::cout<<"pad_hw:"       <<std::hex<<karg->pad_hw<<std::dec<<",";
    std::cout<<"wei_hw:"       <<std::hex<<karg->wei_hw<<std::dec<<",";
    
#if USE_MAGIC_DIV
    std::cout<<"magic_0:"      <<karg->magic_0<<",";
    std::cout<<"magic_1:"      <<karg->magic_1<<",";
    std::cout<<"magic_2:"      <<karg->magic_2<<",";
    std::cout<<"magic_3:"      <<karg->magic_3<<",";
    std::cout<<"magic_4:"      <<karg->magic_4<<",";
    std::cout<<"magic_5:"      <<karg->magic_5<<",";
    std::cout<<"magic_6:"      <<karg->magic_6<<",";
    std::cout<<"magic_7:"      <<karg->magic_7<<",";
    std::cout<<"shift_pack_0:" <<karg->shift_pack_0<<",";
    std::cout<<"shift_pack_1:" <<karg->shift_pack_1<<",";
#endif
#if USE_PERSISTENT_WORKGROUP
    std::cout<<"workgroup_stride:"      <<karg->workgroup_stride<<",";
    std::cout<<"total_workgroups:"      <<karg->total_workgroups<<",";
    std::cout<<"magic_se_per_soc:"      <<karg->magic_se_per_soc<<",";
    std::cout<<"shift_se_per_soc:"      <<karg->shift_se_per_soc<<",";
    std::cout<<"magic_wgp_per_se:"      <<karg->magic_wgp_per_se<<",";
    std::cout<<"shift_wgp_per_se:"      <<karg->shift_wgp_per_se<<",";
    std::cout<<"se_per_soc:"            <<karg->se_per_soc<<",";
    std::cout<<"wgp_per_se:"            <<karg->wgp_per_se<<",";
#endif
    std::cout<<std::endl;
}
typedef struct {
    uint32_t tile_w {0};
    uint32_t tile_h {0};
} igemm_spatial_tiling_t;

typedef struct {
    std::string kernel_name;
    std::string data_type;
    uint32_t m_per_block;
    uint32_t n_per_block;
    uint32_t k_per_block;
    uint32_t block_size;
    uint32_t vector_c;
} igemm_fwd_kernel_info_t;

igemm_fwd_kernel_info_t igemm_fwd_kernel_list [] = 
{
    {"igemm_fwd_gtcn2_nchwc_kcyxc_fp16x8_bx0_ex1_bt128x128x32_lt8x8_lw2x4_lr2x4_ta1x1x1x16_1x4x1x64_tb1x1x2x8_1x4x1x64"  , "fp16",  128,  128, 32,  256, 8},
};

class igemm_fwd_t {
public:
    int num_cu;
    int gcn_arch = 0;
    igemm_fwd_t(){
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        num_cu = dev_prop.multiProcessorCount;
        gcn_arch = dev_prop.gcnArch;
        if(gcn_arch >= 1000)
            num_cu *= 2;
    }
    ~igemm_fwd_t(){}
    std::string get_kernel_name(const igemm_fwd_kernel_info_t *kernel_info) {
        return kernel_info->kernel_name;
    }

    bool is_valid(const args_t *arg, igemm_fwd_kernel_info_t * kernel_info)
    {
        size_t hi = arg->get_int("in_h");
        size_t wi = arg->get_int("in_w");
        size_t n = arg->get_int("batchsize");
        size_t k = arg->get_int("out_channels");
        size_t c = arg->get_int("in_channels");

        size_t sy = arg->get_int("conv_stride_h");
        size_t sx = arg->get_int("conv_stride_w");
        size_t dy = arg->get_int("dilation_h");
        size_t dx = arg->get_int("dilation_w");
        size_t py = arg->get_int("pad_h");
        size_t px = arg->get_int("pad_w");
        size_t fy = arg->get_int("fil_h");
        size_t fx = arg->get_int("fil_w");
        size_t ho = gpu_conv_out_size(hi, py, dy, fy, sy);
        size_t wo = gpu_conv_out_size(wi, px, dx, fx, sx);
        size_t group = arg->get_int("group_count");

        assert(c % group == 0 && k % group == 0);

        assert(group != 0 && c % group == 0 && k % group == 0);

        size_t k_per_group  = k / group;
        size_t c_per_group  = c / group;

        //if(c_per_group != kernel_info->k_per_block)
        //    return false;

        //if(k_per_group % kernel_info->n_per_block != 0)
        //    return false;
        
        return true;
    }

    result_t run(const args_t *arg,  hipModule_t module, igemm_fwd_kernel_info_t * kernel_info,
                 void *p_in, void *p_wei, void *p_out,
                 int warmup, int repeat, const driverDataType_t& data_type) {
        if(!is_valid(arg, kernel_info)){
            result_t result;
            result.return_code = -1;
            return result;
        }
        size_t hi = arg->get_int("in_h");
        size_t wi = arg->get_int("in_w");
        size_t n = arg->get_int("batchsize");
        size_t k = arg->get_int("out_channels");
        size_t c = arg->get_int("in_channels");

        size_t sy = arg->get_int("conv_stride_h");
        size_t sx = arg->get_int("conv_stride_w");
        size_t dy = arg->get_int("dilation_h");
        size_t dx = arg->get_int("dilation_w");
        size_t py = arg->get_int("pad_h");
        size_t px = arg->get_int("pad_w");
        size_t fy = arg->get_int("fil_h");
        size_t fx = arg->get_int("fil_w");
        size_t ho = gpu_conv_out_size(hi, py, dy, fy, sy);
        size_t wo = gpu_conv_out_size(wi, px, dx, fx, sx);
        size_t group = arg->get_int("group_count");

        assert(c % group == 0 && k % group == 0);

        assert(group != 0 && c % group == 0 && k % group == 0);

        size_t k_per_group  = k / group;
        size_t c_per_group  = c / group;
        igemm_fwd_gtc_nchwc_karg_t karg;
        igemm_spatial_tiling_t tiling;
        tiling.tile_h = ho;
        tiling.tile_w = wo;
        uint32_t ntile_h   = (ho + tiling.tile_h - 1) / tiling.tile_h;
        uint32_t ntile_w   = (wo + tiling.tile_w - 1) / tiling.tile_w;
        karg.p_in          = p_in;
        karg.p_wei         = p_wei;
        karg.p_out         = p_out;
        karg.tile_hw       = (tiling.tile_h << 16) | tiling.tile_w;
        karg.ntile_hw      = (ntile_h << 16) | ntile_w;
        karg.hi            = hi;
        karg.wi            = wi;
        karg.n             = n;
        karg.k             = k / group;
        karg.c             = c / group / kernel_info->vector_c;
        karg.group         = group;
        karg.ks            = 1;

        karg.ho            = ho;
        karg.wo            = wo;
        karg.stride_hw     = (sy << 16) | sx;
        karg.dilation_hw   = (dy << 16) | dx;
        karg.pad_hw        = (py << 16 )| px;
        karg.wei_hw        = (fy << 16) | fx;
        
        uint32_t s_move_slice_k_y = (kernel_info->k_per_block / kernel_info->vector_c / fx) % fy;
        uint32_t s_move_slice_k_x = kernel_info->k_per_block / kernel_info->vector_c % fx;
        uint32_t s_move_slice_k_c = (kernel_info->k_per_block / kernel_info->vector_c / (fx * fy)) % (c / group);
        karg.move_slice_k  = (s_move_slice_k_y << 16) | (s_move_slice_k_x << 8) | s_move_slice_k_c;

#if USE_MAGIC_DIV
        uint32_t gemm_n = n * tiling.tile_h * tiling.tile_w;
        uint32_t gemm_m = k / group;

        magic_div_u32_t mdiv_0 = magic_div_u32_gen(utility_integer_divide_ceil(gemm_n, kernel_info->n_per_block));
        magic_div_u32_t mdiv_1 = magic_div_u32_gen(utility_integer_divide_ceil(gemm_m, kernel_info->m_per_block));
        magic_div_u32_t mdiv_2 = magic_div_u32_gen(tiling.tile_h);
        magic_div_u32_t mdiv_3 = magic_div_u32_gen(tiling.tile_w);
        magic_div_u32_t mdiv_4 = magic_div_u32_gen(fy);
        magic_div_u32_t mdiv_5 = magic_div_u32_gen(fx);
        magic_div_u32_t mdiv_6 = magic_div_u32_gen(ntile_h);
        magic_div_u32_t mdiv_7 = magic_div_u32_gen(ntile_w);

        karg.magic_0        = mdiv_0.magic;
        karg.magic_1        = mdiv_1.magic;
        karg.magic_2        = mdiv_2.magic;
        karg.magic_3        = mdiv_3.magic;
        karg.shift_pack_0   = magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, mdiv_3.shift);

        karg.magic_4        = mdiv_4.magic;
        karg.magic_5        = mdiv_5.magic;
        karg.magic_6        = mdiv_6.magic;
        karg.magic_7        = mdiv_7.magic;
        karg.shift_pack_1   = magic_div_u32_pack_shift(mdiv_4.shift, mdiv_5.shift, mdiv_6.shift, mdiv_7.shift);
#endif
        size_t karg_size = sizeof(karg);

        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                            HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                            HIP_LAUNCH_PARAM_END};

        hipFunction_t kernel_func;
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_info->kernel_name.c_str()));

        int block_size  = kernel_info->block_size;

        int num_gemm_m  = (gemm_m + kernel_info->m_per_block - 1) / kernel_info->m_per_block;
        int num_gemm_n  = (gemm_n + kernel_info->n_per_block - 1) / kernel_info->n_per_block;

        int grid_size = num_gemm_m * num_gemm_n;
#if USE_PERSISTENT_WORKGROUP
        {
            hipDeviceProp_t dev_prop;
            hipDevice_t dev;
            HIP_CALL(hipGetDevice(&dev));
            HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
            int num_cu = dev_prop.multiProcessorCount;

            // num_cu *= 2;

            karg.workgroup_stride = num_cu;
            karg.total_workgroups = grid_size;

            uint32_t se_per_soc = env_get_int("SE_PER_SOC" ,4);    // TODO: hardcode
            uint32_t wgp_per_se = env_get_int("WGP_PER_SE", num_cu / se_per_soc);

            printf("se_per_soc:%d, wgp_per_se:%d ", se_per_soc, wgp_per_se );

            magic_div_u32_t mdiv_se_per_soc = magic_div_u32_gen(se_per_soc);
            magic_div_u32_t mdiv_wgp_per_se = magic_div_u32_gen(wgp_per_se);

            karg.magic_se_per_soc = mdiv_se_per_soc.magic;
            karg.shift_se_per_soc = mdiv_se_per_soc.shift;
            karg.magic_wgp_per_se = mdiv_wgp_per_se.magic;
            karg.shift_wgp_per_se = mdiv_wgp_per_se.shift;

            karg.se_per_soc = se_per_soc;
            karg.wgp_per_se = wgp_per_se;


            grid_size = num_cu;
        }
#endif

        //printf("launch fwd block:%d, grid:%d\n", block_size, grid_size);
        //dump_fwd_karg(&karg);

        auto launch_fwd = [&]() -> float {
            void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};
            float ms = .0;

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

        usleep(1000 * 1);

        result_t result;
        result.return_code = 0;
        result.duration_ms = avg_duration;
        result.kernel_name = kernel_info->kernel_name;
        return result;
    }
};

#endif
