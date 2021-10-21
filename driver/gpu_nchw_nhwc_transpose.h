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
#ifndef __GPU_NAIVE_CONV_H
#define __GPU_NAIVE_CONV_H

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <assert.h>
#include "magic_div.h"

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


static struct {
    hipModule_t     module;
    hipFunction_t   kernel_gpu_batched_transpose_16x16_dword;
    hipFunction_t   kernel_gpu_batched_transpose_16x16_half;
    hipFunction_t   kernel_gpu_batched_transpose_16x16_byte;

    hipFunction_t   kernel_gpu_batched_transpose_32x32_pack_2x2_half;
    hipFunction_t   kernel_gpu_batched_transpose_64x32_pack_4x2_half;
} the_transpose_gpu_handle;

static inline void gpu_nhwc_nchw_transpose_init(const char * hsaco){
    static int inited = 0;
    if(!inited){
        HIP_CALL(hipModuleLoad(&the_transpose_gpu_handle.module, hsaco));
        HIP_CALL(hipModuleGetFunction(&the_transpose_gpu_handle.kernel_gpu_batched_transpose_16x16_dword,  the_transpose_gpu_handle.module, "gpu_batched_transpose_16x16_dword"));
        HIP_CALL(hipModuleGetFunction(&the_transpose_gpu_handle.kernel_gpu_batched_transpose_16x16_half,   the_transpose_gpu_handle.module, "gpu_batched_transpose_16x16_half"));
        HIP_CALL(hipModuleGetFunction(&the_transpose_gpu_handle.kernel_gpu_batched_transpose_16x16_byte,   the_transpose_gpu_handle.module, "gpu_batched_transpose_16x16_byte"));

        HIP_CALL(hipModuleGetFunction(&the_transpose_gpu_handle.kernel_gpu_batched_transpose_32x32_pack_2x2_half,   the_transpose_gpu_handle.module, "gpu_batched_transpose_32x32_pack_2x2_half"));
        HIP_CALL(hipModuleGetFunction(&the_transpose_gpu_handle.kernel_gpu_batched_transpose_64x32_pack_4x2_half,   the_transpose_gpu_handle.module, "gpu_batched_transpose_64x32_pack_4x2_half"));

        inited = 1;
    }
}

typedef struct {
    void * p_dst;
    void * p_src;
    uint32_t height;
    uint32_t width;
    uint32_t dim_stride;
    uint32_t dim_total;
    uint32_t magic_h;
    uint32_t shift_h;
    uint32_t magic_w;
    uint32_t shift_w;
} __attribute__((packed)) transpose_kernel_t;

static inline void dump_transpose_kernel_arg(transpose_kernel_t * karg)
{
    printf("dst:%p, src:%p, h:%u, w:%u, dim_stride:%u, dim_total:%u, mh:%u, sh:%u, mw:%u, sw:%u\n",
        karg->p_dst,
        karg->p_src,
        karg->height,
        karg->width,
        karg->dim_stride,
        karg->dim_total,
        karg->magic_h,
        karg->shift_h,
        karg->magic_w,
        karg->shift_w);
    fflush(stdout);
}

typedef struct{
    int tile_x;
    int tile_y;
    int pack_x;
    int pack_y;
}transpose_kernel_param_t;

template<size_t type_size>
struct transpose_kernel_get_all_param_t{
};

template<>
struct transpose_kernel_get_all_param_t<4>{
    static std::vector<transpose_kernel_param_t> get(){
        std::vector<transpose_kernel_param_t> the_list {
            {16, 16, 1, 1},
        };
        return the_list;
    }
};

template<>
struct transpose_kernel_get_all_param_t<2>{
    static std::vector<transpose_kernel_param_t> get(){
        std::vector<transpose_kernel_param_t> the_list {
            {16, 16, 1, 1},
            {32, 32, 2, 2},
            {64, 32, 4, 2},
        };
        return the_list;
    }
};

template<>
struct transpose_kernel_get_all_param_t<1>{
    static std::vector<transpose_kernel_param_t> get(){
        std::vector<transpose_kernel_param_t> the_list {
            {16, 16, 1, 1},
        };
        return the_list;
    }
};

template<size_t type_size>
struct transpose_kernel_select_t{
};

template<>
struct transpose_kernel_select_t<4>{
    static hipFunction_t get(const transpose_kernel_param_t * kparam){return the_transpose_gpu_handle.kernel_gpu_batched_transpose_16x16_dword;}
};

template<>
struct transpose_kernel_select_t<2>{
    static hipFunction_t get(const transpose_kernel_param_t * kparam){
        if(kparam->tile_x == 16 && kparam->tile_y == 16){
            return the_transpose_gpu_handle.kernel_gpu_batched_transpose_16x16_half;
        }
        else if(kparam->tile_x == 32 && kparam->tile_y == 32){
            if(kparam->pack_x == 2 && kparam->pack_y == 2)
                return the_transpose_gpu_handle.kernel_gpu_batched_transpose_32x32_pack_2x2_half;
        }
        // else if(tile_x == 32 && tile_y == 64){
        //     if(pack_x == 2 && pack_y == 4)
        //         return the_transpose_gpu_handle.kernel_gpu_batched_transpose_32x64_pack_2x4_half;
        // }
        else if(kparam->tile_x == 64 && kparam->tile_y == 32){
            if(kparam->pack_x == 4 && kparam->pack_y == 2)
                return the_transpose_gpu_handle.kernel_gpu_batched_transpose_64x32_pack_4x2_half;
        }
        assert(false);
    }
};

template<>
struct transpose_kernel_select_t<1>{
    static hipFunction_t get(const transpose_kernel_param_t * kparam){return the_transpose_gpu_handle.kernel_gpu_batched_transpose_16x16_byte;}
};

template<typename T>
void gpu_batched_transpose(T * dst, T * src, uint32_t batch, uint32_t height, uint32_t width, const transpose_kernel_param_t * kparam)
{
    hipDeviceProp_t dev_prop;
    hipDevice_t dev;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    int num_cu = dev_prop.multiProcessorCount;

    // TODO: need find better way to decide transpose tile size

    const int occupancy = 4;
    size_t block_size = 256;
    size_t grid_size = num_cu * occupancy;

    uint32_t dim_h = (height + kparam->tile_y - 1) / kparam->tile_y;
    uint32_t dim_w = (width + kparam->tile_x - 1) / kparam->tile_x;
    uint32_t dim_total = batch * dim_h * dim_w;

    magic_div_u32_t magic_h = magic_div_u32_gen(dim_h);
    magic_div_u32_t magic_w = magic_div_u32_gen(dim_w);

    hipFunction_t kernel = transpose_kernel_select_t<sizeof(T)>::get(kparam);

    transpose_kernel_t karg;
    karg.p_dst          = reinterpret_cast<void*>(dst);
    karg.p_src          = reinterpret_cast<void*>(src);
    karg.height         = height;
    karg.width          = width;
    karg.dim_stride     = grid_size;
    karg.dim_total      = dim_total;
    karg.magic_h        = magic_h.magic;
    karg.shift_h        = magic_h.shift;
    karg.magic_w        = magic_w.magic;
    karg.shift_w        = magic_w.shift;
    size_t karg_size    = sizeof(karg);

    // dump_transpose_kernel_arg(&karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    HIP_CALL(hipHccModuleLaunchKernel(kernel, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));
}

template<typename T>
void gpu_nchw2nhwc(T * dst, T * src, uint32_t n, uint32_t c, uint32_t h, uint32_t w, const transpose_kernel_param_t * kparam)
{
    gpu_batched_transpose(dst, src, n, c, h * w, kparam);
}

template<typename T>
void gpu_nhwc2nchw(T * dst, T * src, uint32_t n, uint32_t c, uint32_t h, uint32_t w, const transpose_kernel_param_t * kparam)
{
    gpu_batched_transpose(dst, src, n, h * w, c, kparam);
}
#endif
