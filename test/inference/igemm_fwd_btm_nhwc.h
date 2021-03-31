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
    void *   p_in;
    void *   p_wei;
    void *   p_out;
    uint32_t hi;
    uint32_t wi;
    uint32_t n;
    uint32_t k_per_group;
    uint32_t c_per_group;
    uint32_t ho;
    uint32_t wo;
    uint32_t sy;
    uint32_t sx;
    uint32_t dy;
    uint32_t dx;
    uint32_t py;
    uint32_t px;
    uint32_t fy;
    uint32_t fx;
    uint32_t group;
    uint32_t batch_m;
    uint32_t stride_m;
    float    alpha;
    float    beta;
    float    gamma;
    uint32_t magic_0;
    uint32_t magic_1;
    uint32_t magic_2;
    uint32_t shift_pack_0;
} __attribute__((packed)) igemm_fwd_btm_2d_karg_t;
static inline void dump_igemm_fwd_btm_2d_karg(igemm_fwd_btm_2d_karg_t * karg)
{
    std::cout<<"p_in:"<<karg->p_in<<", ";
    std::cout<<"p_wei:"<<karg->p_wei<<", ";
    std::cout<<"p_out:"<<karg->p_out<<", ";
    std::cout<<"hi:"<<karg->hi<<", ";
    std::cout<<"wi:"<<karg->wi<<", ";
    std::cout<<"n:"<<karg->n<<", ";
    std::cout<<"k_per_group:"<<karg->k_per_group<<", ";
    std::cout<<"c_per_group:"<<karg->c_per_group<<", ";
    std::cout<<"ho:"<<karg->ho<<", ";
    std::cout<<"wo:"<<karg->wo<<", ";
    std::cout<<"sy:"<<karg->sy<<", ";
    std::cout<<"sx:"<<karg->sx<<", ";
    std::cout<<"dy:"<<karg->dy<<", ";
    std::cout<<"dx:"<<karg->dx<<", ";
    std::cout<<"py:"<<karg->py<<", ";
    std::cout<<"px:"<<karg->px<<", ";
    std::cout<<"fy:"<<karg->fy<<", ";
    std::cout<<"fx:"<<karg->fx<<", ";
    std::cout<<"group:"<<karg->group<<", ";
    std::cout<<"batch_m:"<<karg->batch_m<<", ";
    std::cout<<"stride_m:"<<karg->stride_m<<", ";
    std::cout<<"alpha:"<<karg->alpha<<", ";
    std::cout<<"beta:"<<karg->beta<<", ";
    std::cout<<"gamma:"<<karg->gamma<<", ";
    std::cout<<"magic_0:"<<karg->magic_0<<", ";
    std::cout<<"magic_1:"<<karg->magic_1<<", ";
    std::cout<<"magic_2:"<<karg->magic_2<<", ";
    std::cout<<"shift_pack_0:"<<karg->shift_pack_0<<std::endl;
}

typedef struct {
    std::string kernel_name;
    std::string data_type;
    uint32_t m_per_block;
    uint32_t n_per_block;
    uint32_t k_per_block;
    uint32_t block_size;
    uint32_t r;
    uint32_t occupancy;
} igemm_fwd_btm_kernel_info_t;

igemm_fwd_btm_kernel_info_t igemm_fwd_btm_kernel_list [] = 
{
    {"igemm_fwd_btm_nhwc_fp16_128x4x16_r2"  , "fp16",  128,  4, 16,  64, 2, 4},
    {"igemm_fwd_btm_nhwc_fp16_128x16x16_r3" , "fp16",  128, 16, 16, 128, 3, 4},
    {"igemm_fwd_btm_nhwc_fp16_256x16x16_r3" , "fp16",  256, 16, 16, 128, 3, 4},
    {"igemm_fwd_btm_nhwc_fp16_256x4x16_r1"  , "fp16",  256,  4, 16, 128, 1, 4},
    {"igemm_fwd_btm_nhwc_fp16_256x8x8_r2"   , "fp16",  256,  8,  8,  64, 2, 4},
    {"igemm_fwd_btm_nhwc_fp16_256x8x16_r2"  , "fp16",  256,  8, 16, 128, 2, 4},
    {"igemm_fwd_btm_nhwc_fp16_384x4x16_r1"  , "fp16",  384,  4, 16, 128, 1, 4},
    {"igemm_fwd_btm_nhwc_fp16_512x4x16_r1"  , "fp16",  512,  4, 16, 128, 1, 3},
    {"igemm_fwd_btm_nhwc_fp16_512x8x16_r2"  , "fp16",  512,  8, 16, 128, 2, 2},
    {"igemm_fwd_btm_nhwc_fp16_512x8x8_r1"   , "fp16",  512,  8,  8, 128, 1, 4},
    {"igemm_fwd_btm_nhwc_fp16_1024x8x8_r1"  , "fp16", 1024,  8,  8, 128, 1, 2},

    {"igemm_fwd_btm_nhwc_int8_256x4x16_r1"  , "int8",  256,  4, 16,  64, 1, 4},
    {"igemm_fwd_btm_nhwc_int8_256x8x16_r1"  , "int8",  256,  8, 16, 128, 1, 4},
    {"igemm_fwd_btm_nhwc_int8_512x8x16_r1"  , "int8",  512,  8, 16, 128, 1, 4},
    {"igemm_fwd_btm_nhwc_int8_512x16x8_r2"  , "int8",  512, 16,  8, 128, 2, 3},
    {"igemm_fwd_btm_nhwc_int8_512x16x16_r2" , "int8",  512, 16, 16, 128, 2, 2},
    {"igemm_fwd_btm_nhwc_int8_1024x16x8_r2" , "int8", 1024, 16,  8, 128, 2, 2},
};

class igemm_fwd_btm_t {
public:
    int num_cu;
    int gcn_arch = 0;
    igemm_fwd_btm_t(){
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        num_cu = dev_prop.multiProcessorCount;
        gcn_arch = dev_prop.gcnArch;
        if(gcn_arch >= 1000)
            num_cu *= 2;
    }
    ~igemm_fwd_btm_t(){}
    std::string get_kernel_name(const igemm_fwd_btm_kernel_info_t *kernel_info) {
        return kernel_info->kernel_name;
    }

    bool is_valid(const args_t *arg, igemm_fwd_btm_kernel_info_t * kernel_info)
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

        if(c_per_group != kernel_info->k_per_block)
            return false;

        if(k_per_group % kernel_info->n_per_block != 0)
            return false;
        
        return true;
    }

    result_t run(const args_t *arg,  hipModule_t module, igemm_fwd_btm_kernel_info_t * kernel_info,
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
        igemm_fwd_btm_2d_karg_t karg;
        karg.p_in           = p_in;
        karg.p_wei          = p_wei;
        karg.p_out          = p_out;
        karg.hi             = static_cast<int>(hi);
        karg.wi             = static_cast<int>(wi);
        karg.n              = static_cast<int>(n);
        karg.k_per_group    = static_cast<int>(k_per_group);
        karg.c_per_group    = static_cast<int>(c_per_group);
        karg.ho             = static_cast<int>(ho);
        karg.wo             = static_cast<int>(wo);
        karg.sy             = static_cast<int>(sy);
        karg.sx             = static_cast<int>(sx);
        karg.dy             = static_cast<int>(dy);
        karg.dx             = static_cast<int>(dx);
        karg.py             = static_cast<int>(py);
        karg.px             = static_cast<int>(px);
        karg.fy             = static_cast<int>(fy);
        karg.fx             = static_cast<int>(fx);
        karg.group          = static_cast<int>(group);
        size_t karg_size    = sizeof(karg);

        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                            HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                            HIP_LAUNCH_PARAM_END};

        hipFunction_t kernel_func;
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_info->kernel_name.c_str()));

        int block_size  = kernel_info->block_size;
        int num_gemm_m  = (ho * wo + kernel_info->m_per_block - 1) / kernel_info->m_per_block;
        int num_gemm_n  = (k_per_group + kernel_info->n_per_block - 1) / kernel_info->n_per_block;

        int grid_size = kernel_info->occupancy * num_cu;
        grid_size = env_get_int("GRID_SIZE", grid_size);
        if(grid_size % num_gemm_n == 0){
            int grids_for_m = grid_size / num_gemm_n;
            karg.batch_m    = (num_gemm_m + grids_for_m - 1) / grids_for_m;
            karg.stride_m   = kernel_info->m_per_block * grids_for_m;

        }else{
            grid_size = num_gemm_m * num_gemm_n;
            karg.batch_m    = 1;
            karg.stride_m   = 0;
        }

        // TODO: proper set alpha/beta/gamma
        karg.alpha          = 1.0f;
        karg.beta           = 1.0f;
        karg.gamma          = 1.0f;

        magic_div_u32_t mdiv_0 = magic_div_u32_gen(fx);
        magic_div_u32_t mdiv_1 = magic_div_u32_gen(wo);
        magic_div_u32_t mdiv_2 = magic_div_u32_gen(num_gemm_n);
        karg.magic_0        = mdiv_0.magic;
        karg.magic_1        = mdiv_1.magic;
        karg.magic_2        = mdiv_2.magic;
        karg.shift_pack_0   = magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, 0);

        // printf("launch fwd block:%d, grid:%d\n", block_size, grid_size);
        // dump_igemm_fwd_btm_2d_karg(&karg);

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
            HIP_CALL(hipHccModuleLaunchKernel(kernel_func, grid_size * block_size, n, group,
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
