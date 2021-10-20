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
 * The above copyright notice and this permission notice shall be included in all
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
#include <hip/hip_runtime.h>

inline __device__ uint32_t
magic_div_u32(uint32_t numer, uint32_t magic, uint32_t shift)
{
    uint32_t tmp = __umulhi(numer, magic);
    return (tmp + numer) >> shift;
}

template <typename T>
inline __device__ void
gpu_batched_transpose_16x16(T * dst, T * src, uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,
    uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)
{
    /*
    * assume src is batch * height * width, dst is batch * width * height
    */
    constexpr auto element_byte = sizeof(T);
    constexpr auto padding_element = 4 / element_byte;
    constexpr auto smem_stride = 16 + padding_element;
    __shared__ T smem[16 * smem_stride];

    uint32_t h_dim = (height + 15) >> 4;
    uint32_t w_dim = (width + 15) >> 4;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride){
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 4) + i_src_h;

        __syncthreads();
        if(g_src_h < height && g_src_w < width){
            size_t src_index = static_cast<size_t>(dim_in) * height * width + static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
            smem[i_src_h * smem_stride + i_src_w] = src[src_index];
        }
        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 4) + i_dst_w;

        if(g_dst_h < height && g_dst_w < width){
            size_t dst_index = static_cast<size_t>(dim_in) * width * height + static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);
            dst[dst_index] = smem[i_dst_h * smem_stride + i_dst_w];
        }
    }
}


#define DEFINE_BATCHED_TRANSPOSE_KERNEL(tile_trait, accept_data_type, cast_data_type, lb_threads_per_block, lb_blocks_per_cu)   \
    extern "C" __global__ void __launch_bounds__(lb_threads_per_block, lb_blocks_per_cu)                    \
    gpu_batched_transpose_ ## tile_trait ## _ ## accept_data_type(void * dst, void * src,                   \
                uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,                   \
                uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)                     \
    {                                                                                                       \
        gpu_batched_transpose_ ## tile_trait<cast_data_type>(                                               \
                reinterpret_cast<cast_data_type*>(dst),                                                     \
                reinterpret_cast<cast_data_type*>(src),                                                     \
                height, width, dim_stride, dim_total, magic_h, shift_h, magic_w, shift_w);                  \
    }

DEFINE_BATCHED_TRANSPOSE_KERNEL(16x16,  dword,   float,  256,    4)
DEFINE_BATCHED_TRANSPOSE_KERNEL(16x16,   half,  ushort,  256,    4)
DEFINE_BATCHED_TRANSPOSE_KERNEL(16x16,   byte, uint8_t,  256,    4)
