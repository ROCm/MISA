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
#include <hip/hip_fp16.h>

inline __device__ uint32_t
magic_div_u32(const uint32_t & numer, const uint32_t & magic, const uint32_t & shift)
{
    uint32_t tmp = __umulhi(numer, magic);
    return (tmp + numer) >> shift;
}

inline __device__ void
v_pack_b32_f16_00(float & c, const float & a, const float & b)
{
#if 0
    asm volatile("v_pack_b32_f16 %0, %1, %2\n"
                 : "=v"(c)
                 : "v"(a), "v"(b));
#else
    const int x = *reinterpret_cast<const int*>(&a);
    const int y = *reinterpret_cast<const int*>(&b);
    int z = (x & 0xffff) | ((y & 0xffff) << 16);
    c = *reinterpret_cast<float*>(&z);
#endif
}

inline __device__ void
v_pack_b32_f16_11(float & c, const float & a, const float & b)
{
#if 0
    asm volatile("v_pack_b32_f16 %0, %1, %2 op_sel:[1, 1]\n"
                 : "=v"(c)
                 : "v"(a), "v"(b));
#else
    const int x = *reinterpret_cast<const int*>(&a);
    const int y = *reinterpret_cast<const int*>(&b);
    int z = ((x & 0xffff0000) >> 16) | (y & 0xffff0000);
    c = *reinterpret_cast<float*>(&z);
#endif
}

inline __device__ void
v_pack_b32_f16_2x2(float & y0, float & y1, const float & x0, const float & x1)
{
#if 0
    asm volatile("\n \
                    v_pack_b32_f16 %0, %2, %3\n \
                    v_pack_b32_f16 %1, %2, %3 op_sel:[1, 1]\n"
                 : "=v"(y0), "=v"(y1)
                 : "v"(x0), "v"(x1), "0"(y0), "1"(y1));
#else
    const int a0 = *reinterpret_cast<const int*>(&x0);
    const int a1 = *reinterpret_cast<const int*>(&x1);
    int b0 = (a0 & 0xffff) | ((a1 & 0xffff) << 16);
    int b1 = ((a0 & 0xffff0000) >> 16) | (a1 & 0xffff0000);
    y0 = *reinterpret_cast<float*>(&b0);
    y1 = *reinterpret_cast<float*>(&b1);
#endif
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

template <typename T>
inline __device__ void
gpu_batched_transpose_32x32_pack_2x2_smod_2x2(T * dst, T * src, uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,
    uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)
{
}

template <>
inline __device__ void
gpu_batched_transpose_32x32_pack_2x2_smod_2x2<ushort>(ushort * dst, ushort * src, uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,
    uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    __shared__ float smem[32 * smem_stride];

    float * p_dst = reinterpret_cast<float*>(dst);
    float * p_src = reinterpret_cast<float*>(src);

    uint32_t height_2 = height >> 1;
    uint32_t width_2 = width >> 1;

    uint32_t h_dim = (height + 31) >> 5;
    uint32_t w_dim = (width + 31) >> 5;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride){
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 5) + (i_src_h << 1);

        __syncthreads();
        if(g_src_h < height && g_src_w < width_2)
        {
            float v_a, v_b, v_a_pack, v_b_pack;
            size_t src_index = static_cast<size_t>(dim_in) * height * width_2 + static_cast<size_t>(g_src_h) * width_2 + static_cast<size_t>(g_src_w);
            v_a = p_src[src_index];
            v_b = p_src[src_index + width_2];
            v_pack_b32_f16_2x2(v_a_pack, v_b_pack, v_a, v_b);

            smem[i_src_w * smem_stride + i_src_h] = v_a_pack;
            smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] = v_b_pack;
        }
        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 5) + (i_dst_w << 1);

        if(g_dst_h < height_2 && g_dst_w < width)
        {
            float v_a, v_b;
            v_a = smem[i_dst_w * smem_stride + i_dst_h];
            v_b = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];
            size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 + static_cast<size_t>(g_dst_w) * height_2 + static_cast<size_t>(g_dst_h);
            p_dst[dst_index] = v_a;
            p_dst[dst_index + height_2] = v_b;
        }
    }
}

template <typename T>
inline __device__ void
gpu_batched_transpose_64x32_pack_4x2_smod_4x2(T * dst, T * src, uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,
    uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)
{
}

template <>
inline __device__ void
gpu_batched_transpose_64x32_pack_4x2_smod_4x2<ushort>(ushort * dst, ushort * src, uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,
    uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    __shared__ float smem[64 * smem_stride];
    //__shared__ float4 smem[16 * smem_stride];

    float * p_dst = reinterpret_cast<float*>(dst);
    float2 * p_src = reinterpret_cast<float2*>(src);

    uint32_t height_2 = height >> 1;
    uint32_t width_4 = width >> 2;

    uint32_t h_dim = (height + 31) >> 5;
    uint32_t w_dim = (width + 63) >> 6;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride){
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 5) + (i_src_h << 1);

        __syncthreads();
        if(g_src_h < height && g_src_w < width_4)
        {
#if 1
            float2 v_a, v_b;
            float v_pack[4];
            size_t src_index = static_cast<size_t>(dim_in) * height * width_4 + static_cast<size_t>(g_src_h) * width_4 + static_cast<size_t>(g_src_w);
            v_a = p_src[src_index];
            v_b = p_src[src_index + width_4];
            v_pack_b32_f16_2x2(v_pack[0], v_pack[1], v_a.x, v_b.x);
            v_pack_b32_f16_2x2(v_pack[2], v_pack[3], v_a.y, v_b.y);

            smem[i_src_w * smem_stride + i_src_h] = v_pack[0];
            smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] = v_pack[1];
            smem[i_src_w * smem_stride + i_src_h + 32 * smem_stride] = v_pack[2];
            smem[i_src_w * smem_stride + i_src_h + 48 * smem_stride] = v_pack[3];
#else
            float2 v_a, v_b;
            float4 v_pack;
            size_t src_index = static_cast<size_t>(dim_in) * height * width_4 + static_cast<size_t>(g_src_h) * width_4 + static_cast<size_t>(g_src_w);
            v_a = p_src[src_index];
            v_b = p_src[src_index + width_4];
            v_pack_b32_f16_2x2(v_pack.x, v_pack.y, v_a.x, v_b.x);
            v_pack_b32_f16_2x2(v_pack.z, v_pack.w, v_a.y, v_b.y);

            smem[i_src_w * smem_stride + i_src_h] = v_pack;
#endif
        }
        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 6) + (i_dst_w << 2);

        if(g_dst_h < height_2 && g_dst_w < width)
        {
#if 1
            float v[4];
            v[0] = smem[i_dst_w * smem_stride + i_dst_h];
            v[1] = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];
            v[2] = smem[i_dst_w * smem_stride + i_dst_h + 32 * smem_stride];
            v[3] = smem[i_dst_w * smem_stride + i_dst_h + 48 * smem_stride];
            size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 + static_cast<size_t>(g_dst_w) * height_2 + static_cast<size_t>(g_dst_h);
            p_dst[dst_index] = v[0];
            p_dst[dst_index + height_2] = v[1];
            p_dst[dst_index + 2 * height_2] = v[2];
            p_dst[dst_index + 3 * height_2] = v[3];
#else
            float4 v;
            v = smem[i_dst_w * smem_stride + i_dst_h];
            size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 + static_cast<size_t>(g_dst_w) * height_2 + static_cast<size_t>(g_dst_h);
            p_dst[dst_index] = v.x;
            p_dst[dst_index + height_2] = v.y;
            p_dst[dst_index + 2 * height_2] = v.z;
            p_dst[dst_index + 3 * height_2] = v.w;
#endif
        }
    }
}

template <typename T>
inline __device__ void
gpu_batched_transpose_32x64_pack_2x4_smod_2x4(T * dst, T * src, uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,
    uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)
{
}

template <>
inline __device__ void
gpu_batched_transpose_32x64_pack_2x4_smod_2x4<ushort>(ushort * dst, ushort * src, uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,
    uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    //__shared__ float smem[64 * smem_stride];
    __shared__ float4 smem[16 * smem_stride];

    float2 * p_dst = reinterpret_cast<float2*>(dst);
    float * p_src = reinterpret_cast<float*>(src);

    uint32_t height_4 = height >> 2;
    uint32_t width_2 = width >> 1;

    uint32_t h_dim = (height + 63) >> 6;
    uint32_t w_dim = (width + 31) >> 5;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride){
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 6) + (i_src_h << 2);

        __syncthreads();
        /*
        * 4x2 -> 2x4 transpose: (0, 1, 2, 3 is in float)
        *        lo hi
        *      0 |_|_|      lo |0|2| 
        *      1 |_|_|  ->  hi |_|_| 
        *      2 |_|_|      lo |1|3| 
        *      3 |_|_|      hi |_|_|
        */
        if(g_src_h < height && g_src_w < width_2)
        {
#if 0
            float v[4];
            float v_pack[4];
            size_t src_index = static_cast<size_t>(dim_in) * height * width_2 + static_cast<size_t>(g_src_h) * width_2 + static_cast<size_t>(g_src_w);
            v[0] = p_src[src_index];
            v[1] = p_src[src_index + width_2];
            v[2] = p_src[src_index + 2 * width_2];
            v[3] = p_src[src_index + 3 * width_2];
            v_pack_b32_f16_2x2(v_pack[0], v_pack[2], v[0], v[1]);
            v_pack_b32_f16_2x2(v_pack[1], v_pack[3], v[2], v[3]);

            smem[i_src_w * smem_stride + i_src_h] = v_pack[0];
            smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] = v_pack[1];
            smem[i_src_w * smem_stride + i_src_h + 32 * smem_stride] = v_pack[2];
            smem[i_src_w * smem_stride + i_src_h + 48 * smem_stride] = v_pack[3];
#else
            float v[4];
            float4 v_pack;
            size_t src_index = static_cast<size_t>(dim_in) * height * width_2 + static_cast<size_t>(g_src_h) * width_2 + static_cast<size_t>(g_src_w);
            v[0] = p_src[src_index];
            v[1] = p_src[src_index + width_2];
            v[2] = p_src[src_index + 2 * width_2];
            v[3] = p_src[src_index + 3 * width_2];
            v_pack_b32_f16_2x2(v_pack.x, v_pack.z, v[0], v[1]);
            v_pack_b32_f16_2x2(v_pack.y, v_pack.w, v[2], v[3]);

            smem[i_src_w * smem_stride + i_src_h] = v_pack;
#endif
        }
        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 5) + (i_dst_w << 1);

        if(g_dst_h < height_4 && g_dst_w < width)
        {
#if 0
            float v[4];
            v[0] = smem[i_dst_w * smem_stride + i_dst_h];
            v[1] = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];
            v[2] = smem[i_dst_w * smem_stride + i_dst_h + 32 * smem_stride];
            v[3] = smem[i_dst_w * smem_stride + i_dst_h + 48 * smem_stride];
            size_t dst_index = static_cast<size_t>(dim_in) * width * height_4 + static_cast<size_t>(g_dst_w) * height_4 + static_cast<size_t>(g_dst_h);
            p_dst[dst_index] = v[0];
            p_dst[dst_index + height_4] = v[1];
            p_dst[dst_index + 2 * height_4] = v[2];
            p_dst[dst_index + 3 * height_4] = v[3];
#else
            float4 v;
            v = smem[i_dst_w * smem_stride + i_dst_h];
            size_t dst_index = static_cast<size_t>(dim_in) * width * height_4 + static_cast<size_t>(g_dst_w) * height_4 + static_cast<size_t>(g_dst_h);
            p_dst[dst_index] = make_float2(v.x, v.y);
            p_dst[dst_index + height_4] = make_float2(v.z, v.w);
#endif
        }
    }
}




template <typename T>
inline __device__ void
gpu_batched_transpose_32x64_pack_2x4_smod_2x2(T * dst, T * src, uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,
    uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)
{
}

template <>
inline __device__ void
gpu_batched_transpose_32x64_pack_2x4_smod_2x2<ushort>(ushort * dst, ushort * src, uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,
    uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    //__shared__ float smem[64 * smem_stride];
    __shared__ float4 smem[16 * smem_stride];

    float * p_dst = reinterpret_cast<float*>(dst);
    float * p_src = reinterpret_cast<float*>(src);

    uint32_t height_2 = height >> 1;
    uint32_t width_2 = width >> 1;

    uint32_t h_dim = (height + 63) >> 6;
    uint32_t w_dim = (width + 31) >> 5;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride){
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 6) + (i_src_h << 1);

        __syncthreads();

        float v_src[4];
        size_t src_index = static_cast<size_t>(dim_in) * height * width_2 + static_cast<size_t>(g_src_h) * width_2 + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width_2)
        {
            v_src[0] = p_src[src_index];
            v_src[1] = p_src[src_index + width_2];
        }
        if((g_src_h + 32) < height && g_src_w < width_2)
        {
            v_src[2] = p_src[src_index + 32 * width_2];
            v_src[3] = p_src[src_index + 33 * width_2];
        }

        float4 v_pack;
        v_pack_b32_f16_2x2(v_pack.x, v_pack.z, v_src[0], v_src[1]);
        v_pack_b32_f16_2x2(v_pack.y, v_pack.w, v_src[2], v_src[3]);

        smem[i_src_w * smem_stride + i_src_h] = v_pack;

        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 5) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 5) + (i_dst_w << 1);

        size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 + static_cast<size_t>(g_dst_w) * height_2 + static_cast<size_t>(g_dst_h);

        float4 v_dst = smem[i_dst_w * smem_stride + i_dst_h];
        if(g_dst_h < height_2 && g_dst_w < width){
            p_dst[dst_index] = v_dst.x;
            p_dst[dst_index + height_2] = v_dst.z;
        }
        if((g_dst_h + 16) < height_2 && g_dst_w < width){
            p_dst[dst_index + 16] = v_dst.y;
            p_dst[dst_index + height_2 + 16] = v_dst.w;
        }
    }
}


template <typename T>
inline __device__ void
gpu_batched_transpose_64x64_pack_4x4_smod_4x4(T * dst, T * src, uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,
    uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)
{
}

template <>
inline __device__ void
gpu_batched_transpose_64x64_pack_4x4_smod_4x4<ushort>(ushort * dst, ushort * src, uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,
    uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    //__shared__ float smem[64 * smem_stride];
    __shared__ float4 smem[32 * smem_stride];

    float2 * p_dst = reinterpret_cast<float2*>(dst);
    float2 * p_src = reinterpret_cast<float2*>(src);

    uint32_t height_4 = height >> 2;
    uint32_t width_4 = width >> 2;

    uint32_t h_dim = (height + 63) >> 6;
    uint32_t w_dim = (width + 63) >> 6;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride){
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 6) + (i_src_h << 2);

        __syncthreads();
        /*
        * 4x2 -> 2x4 transpose: (0, 1, 2, 3 is in float2)
        *        lo hi
        *      0 |_|_|_|_|      lo |0|1|2|3| 
        *      1 |_|_|_|_|  ->  hi |_|_|_|_| 
        *      2 |_|_|_|_|      lo | | | | | 
        *      3 |_|_|_|_|      hi |_|_|_|_|
        */

        float2 v_src[4];
        size_t src_index = static_cast<size_t>(dim_in) * height * width_4 + static_cast<size_t>(g_src_h) * width_4 + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width_4)
        {
            v_src[0] = p_src[src_index];
            v_src[1] = p_src[src_index + width_4];
            v_src[2] = p_src[src_index + 2 * width_4];
            v_src[3] = p_src[src_index + 3 * width_4];
        }

        float2 v_pack[4];
        v_pack_b32_f16_2x2(v_pack[0].x, v_pack[1].x, v_src[0].x, v_src[1].x);
        v_pack_b32_f16_2x2(v_pack[2].x, v_pack[3].x, v_src[0].y, v_src[1].y);
        v_pack_b32_f16_2x2(v_pack[0].y, v_pack[1].y, v_src[2].x, v_src[3].x);
        v_pack_b32_f16_2x2(v_pack[2].y, v_pack[3].y, v_src[2].y, v_src[3].y);

        smem[i_src_w * smem_stride + i_src_h] =
                make_float4(v_pack[0].x, v_pack[0].y, v_pack[1].x, v_pack[1].y);
        smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] =
                make_float4(v_pack[2].x, v_pack[2].y, v_pack[3].x, v_pack[3].y);

        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 6) + (i_dst_w << 2);

        size_t dst_index = static_cast<size_t>(dim_in) * width * height_4 + static_cast<size_t>(g_dst_w) * height_4 + static_cast<size_t>(g_dst_h);

        float4 v_dst[2];
        v_dst[0] = smem[i_dst_w * smem_stride + i_dst_h];
        v_dst[1] = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];
        if(g_dst_h < height_4 && g_dst_w < width){
            p_dst[dst_index] = make_float2(v_dst[0].x, v_dst[0].y);
            p_dst[dst_index + height_4] = make_float2(v_dst[0].z, v_dst[0].w);
            p_dst[dst_index + 2 * height_4] = make_float2(v_dst[1].x, v_dst[1].y);
            p_dst[dst_index + 3 * height_4] = make_float2(v_dst[1].z, v_dst[1].w);
        }
    }
}


template <typename T>
inline __device__ void
gpu_batched_transpose_64x64_pack_4x4_smod_2x2(T * dst, T * src, uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,
    uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)
{
}

template <>
inline __device__ void
gpu_batched_transpose_64x64_pack_4x4_smod_2x2<ushort>(ushort * dst, ushort * src, uint32_t height, uint32_t width, uint32_t dim_stride, uint32_t dim_total,
    uint32_t magic_h, uint32_t shift_h, uint32_t magic_w, uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    __shared__ float4 smem[32 * smem_stride];

    float * p_dst = reinterpret_cast<float*>(dst);
    float * p_src = reinterpret_cast<float*>(src);

    uint32_t height_2 = height >> 1;
    uint32_t width_2 = width >> 1;

    uint32_t h_dim = (height + 63) >> 6;
    uint32_t w_dim = (width + 63) >> 6;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride){
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 5) + i_src_w;
        uint32_t g_src_h = (dim_ih << 6) + (i_src_h << 1);

        __syncthreads();
        /*
        * 4x4 -> 4x4 transpose: (0, 1, 2, 3 is in float, a, b, c, d is in float2)
        *        lo hi
        *       |0|_|1|_|      lo |a|b|c|d| 
        *       |2|_|3|_|  ->  hi |_|_|_|_| 
        *       |4|_|5|_|      lo | | | | | 
        *       |6|_|7|_|      hi |_|_|_|_|
        */

        float v_src[8];
        size_t src_index = static_cast<size_t>(dim_in) * height * width_2 + static_cast<size_t>(g_src_h) * width_2 + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width_2)
        {
            v_src[0] = p_src[src_index];
            v_src[2] = p_src[src_index + width_2];
        }
        if(g_src_h < height && (g_src_w + 16) < width_2)
        {
            v_src[1] = p_src[src_index + 16];
            v_src[3] = p_src[src_index + width_2 + 16];
        }
        if((g_src_h + 32) < height && g_src_w < width_2)
        {
            v_src[4] = p_src[src_index + 32 * width_2];
            v_src[6] = p_src[src_index + 33 * width_2];
        }
        if((g_src_h + 32) < height && (g_src_w + 16) < width_2)
        {
            v_src[5] = p_src[src_index + 32 * width_2 + 16];
            v_src[7] = p_src[src_index + 33 * width_2 + 16];
        }

        float2 v_pack[4];
        v_pack_b32_f16_2x2(v_pack[0].x, v_pack[1].x, v_src[0], v_src[2]);
        v_pack_b32_f16_2x2(v_pack[2].x, v_pack[3].x, v_src[1], v_src[3]);
        v_pack_b32_f16_2x2(v_pack[0].y, v_pack[1].y, v_src[4], v_src[6]);
        v_pack_b32_f16_2x2(v_pack[2].y, v_pack[3].y, v_src[5], v_src[7]);

        smem[i_src_w * smem_stride + i_src_h] =
                make_float4(v_pack[0].x, v_pack[0].y, v_pack[1].x, v_pack[1].y);
        smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] =
                make_float4(v_pack[2].x, v_pack[2].y, v_pack[3].x, v_pack[3].y);

        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 5) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 6) + (i_dst_w << 1);

        size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 + static_cast<size_t>(g_dst_w) * height_2 + static_cast<size_t>(g_dst_h);

        float4 v_dst[2];
        v_dst[0] = smem[i_dst_w * smem_stride + i_dst_h];
        v_dst[1] = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];
        if(g_dst_h < height_2 && g_dst_w < width){
            p_dst[dst_index] = v_dst[0].x;
            p_dst[dst_index + height_2] = v_dst[0].z;
        }
        if((g_dst_h + 16) < height_2 && g_dst_w < width){
            p_dst[dst_index + 16] = v_dst[0].y;
            p_dst[dst_index + height_2 + 16] = v_dst[0].w;
        }
        if(g_dst_h < height_2 && (g_dst_w + 32) < width){
            p_dst[dst_index + 32 * height_2] = v_dst[1].x;
            p_dst[dst_index + 33 * height_2] = v_dst[1].z;
        }
        if((g_dst_h + 16) < height_2 && (g_dst_w + 32) < width){
            p_dst[dst_index + 32 * height_2 + 16] = v_dst[1].y;
            p_dst[dst_index + 33 * height_2 + 16] = v_dst[1].w;
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

DEFINE_BATCHED_TRANSPOSE_KERNEL(32x32_pack_2x2_smod_2x2,   half, ushort,  256,    4)
DEFINE_BATCHED_TRANSPOSE_KERNEL(64x32_pack_4x2_smod_4x2,   half, ushort,  256,    4)
DEFINE_BATCHED_TRANSPOSE_KERNEL(32x64_pack_2x4_smod_2x4,   half, ushort,  256,    4)
DEFINE_BATCHED_TRANSPOSE_KERNEL(32x64_pack_2x4_smod_2x2,   half, ushort,  256,    4)
DEFINE_BATCHED_TRANSPOSE_KERNEL(64x64_pack_4x4_smod_4x4,   half, ushort,  256,    4)
DEFINE_BATCHED_TRANSPOSE_KERNEL(64x64_pack_4x4_smod_2x2,   half, ushort,  256,    4)
