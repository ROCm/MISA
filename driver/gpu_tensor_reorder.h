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

#include "gpu_batched_transpose.h"
#include "gpu_general_tensor_reorder.h"


static inline bool gpu_nchw2nhwc_is_kernel_valid(uint32_t n, uint32_t c, uint32_t h, uint32_t w, const transpose_kernel_param_t * kparam)
{
    return transpose_kernel_is_valid(n, c, h * w, kparam);
}

static inline bool gpu_nhwc2nchw_is_kernel_valid(uint32_t n, uint32_t c, uint32_t h, uint32_t w, const transpose_kernel_param_t * kparam)
{
    return transpose_kernel_is_valid(n, h * w, c, kparam);
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


static inline bool gpu_tensor_reorder_is_kernel_valid(uint32_t n, uint32_t c, uint32_t h, uint32_t w, const transpose_kernel_param_t * kparam)
{
    return transpose_kernel_is_valid(n, c, h * w, kparam);
}

template<typename T, typename dst_order>
void gpu_tensor_reorder(T * dst, T * src, uint32_t n, uint32_t c, uint32_t h, uint32_t w, const transpose_kernel_param_t * kparam)
{
    if(dst_order::at(0)==0 && dst_order::at(1)==2 && dst_order::at(2)==3 && dst_order::at(3)==1){
        gpu_batched_transpose(dst, src, n, c, h * w, kparam);
    }
    else if(dst_order::at(0)==0 && dst_order::at(1)==3 && dst_order::at(2)==1 && dst_order::at(3)==2){
        gpu_batched_transpose(dst, src, n, c*h , w, kparam);
    }
    else{
        //printf("GPU choose general kernel\n");
        gpu_general_tensor_reorder<T, dst_order>(dst, src, n, c, h, w, kparam);
    }
}

#endif
