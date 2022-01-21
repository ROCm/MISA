/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
#include "sequence.hpp"

#ifndef BATCHED_TRANSPOSE_OCCUPANCY
#define BATCHED_TRANSPOSE_OCCUPANCY 4
#endif

inline __device__ uint32_t magic_div_u32(const uint32_t& numer,
                                         const uint32_t& magic,
                                         const uint32_t& shift)
{
    uint32_t tmp = __umulhi(numer, magic);
    return (tmp + numer) >> shift;
}

inline __device__ void v_pack_b32_f16_00(float& c, const float& a, const float& b)
{
#if 0
    asm volatile("v_pack_b32_f16 %0, %1, %2\n"
                 : "=v"(c)
                 : "v"(a), "v"(b));
#else
    // cppcheck-suppress invalidPointerCast
    const uint32_t x = *reinterpret_cast<const uint32_t*>(&a);
    // cppcheck-suppress invalidPointerCast
    const uint32_t y = *reinterpret_cast<const uint32_t*>(&b);
    uint32_t z       = (x & 0xffff) | ((y & 0xffff) << 16);
    // cppcheck-suppress invalidPointerCast
    c = *reinterpret_cast<float*>(&z);
#endif
}

inline __device__ void v_pack_b32_f16_11(float& c, const float& a, const float& b)
{
#if 0
    asm volatile("v_pack_b32_f16 %0, %1, %2 op_sel:[1, 1]\n"
                 : "=v"(c)
                 : "v"(a), "v"(b));
#else
    // cppcheck-suppress invalidPointerCast
    const uint32_t x = *reinterpret_cast<const uint32_t*>(&a);
    // cppcheck-suppress invalidPointerCast
    const uint32_t y = *reinterpret_cast<const uint32_t*>(&b);
    uint32_t z       = ((x & 0xffff0000) >> 16) | (y & 0xffff0000);
    // cppcheck-suppress invalidPointerCast
    c = *reinterpret_cast<float*>(&z);
#endif
}

inline __device__ void v_pack_b32_f16_2x2(float& y0, float& y1, const float& x0, const float& x1)
{
#if 0
    asm volatile("\n \
                    v_pack_b32_f16 %0, %2, %3\n \
                    v_pack_b32_f16 %1, %2, %3 op_sel:[1, 1]\n"
                 : "=v"(y0), "=v"(y1)
                 : "v"(x0), "v"(x1), "0"(y0), "1"(y1));
#else
    // cppcheck-suppress invalidPointerCast
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&x0);
    // cppcheck-suppress invalidPointerCast
    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&x1);
    uint32_t b0       = (a0 & 0xffff) | ((a1 & 0xffff) << 16);
    uint32_t b1       = ((a0 & 0xffff0000) >> 16) | (a1 & 0xffff0000);
    // cppcheck-suppress invalidPointerCast
    y0 = *reinterpret_cast<float*>(&b0);
    // cppcheck-suppress invalidPointerCast
    y1 = *reinterpret_cast<float*>(&b1);
#endif
}

inline __device__ void v_pack_b32_f16_2x2_half_x0(
    float& y0, float& y1, const ushort& x0_lo, const ushort& x0_hi, const float& x1)
{
    // cppcheck-suppress invalidPointerCast
    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&x1);
    uint32_t b0       = x0_lo | ((a1 & 0xffff) << 16);
    uint32_t b1       = x0_hi | (a1 & 0xffff0000);
    // cppcheck-suppress invalidPointerCast
    y0 = *reinterpret_cast<float*>(&b0);
    // cppcheck-suppress invalidPointerCast
    y1 = *reinterpret_cast<float*>(&b1);
}

inline __device__ void v_pack_b32_f16_2x2_half_x1(
    float& y0, float& y1, const float& x0, const ushort& x1_lo, const ushort& x1_hi)
{
    // cppcheck-suppress invalidPointerCast
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&x0);
    uint32_t b0       = (a0 & 0xffff) | (x1_lo << 16);
    uint32_t b1       = ((a0 & 0xffff0000) >> 16) | (x1_hi << 16);
    // cppcheck-suppress invalidPointerCast
    y0 = *reinterpret_cast<float*>(&b0);
    // cppcheck-suppress invalidPointerCast
    y1 = *reinterpret_cast<float*>(&b1);
}

inline __device__ void v_pack_b32_f16_2x2_half_x0_half_x1(float& y0,
                                                          float& y1,
                                                          const ushort& x0_lo,
                                                          const ushort& x0_hi,
                                                          const ushort& x1_lo,
                                                          const ushort& x1_hi)
{
    uint32_t b0 = x0_lo | (x1_lo << 16);
    uint32_t b1 = x0_hi | (x1_hi << 16);
    // cppcheck-suppress invalidPointerCast
    y0 = *reinterpret_cast<float*>(&b0);
    // cppcheck-suppress invalidPointerCast
    y1 = *reinterpret_cast<float*>(&b1);
}

template <typename T, int N>
struct mapped_vector_type
{
};

template <>
struct mapped_vector_type<float, 4>
{
    using type = float4;
};

template <>
struct mapped_vector_type<float, 2>
{
    using type = float2;
};

template <>
struct mapped_vector_type<float, 1>
{
    using type = float;
};

template <>
struct mapped_vector_type<ushort, 4>
{
    using type = ushort4;
};

template <>
struct mapped_vector_type<ushort, 2>
{
    using type = ushort2;
};

template <>
struct mapped_vector_type<ushort, 1>
{
    using type = ushort;
};

template <>
struct mapped_vector_type<uchar, 4>
{
    using type = uchar4;
};

template <>
struct mapped_vector_type<uchar, 2>
{
    using type = uchar2;
};

template <>
struct mapped_vector_type<uchar, 1>
{
    using type = uchar;
};

template <typename T, typename dst_order>
inline __device__ void general_4d_reorder_1p(T* dst,
                                               T* src,
                                               uint32_t dim_0,
                                               uint32_t dim_1,
                                               uint32_t dim_2,
                                               uint32_t dim_3,
                                               uint32_t dim_stride,
                                               uint32_t dim_total,
                                               uint32_t magic_stride0,
                                               uint32_t shift_stride0,
                                               uint32_t magic_stride1,
                                               uint32_t shift_stride1,
                                               uint32_t magic_stride2,
                                               uint32_t shift_stride2)
{
    constexpr auto dorder = dst_order{};
    uint32_t pixel_total = dim_0 * dim_1 * dim_2 * dim_3; 
    uint32_t src_index =0, dst_index=0;
    const uint64_t src_dim[4]  = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {src_dim[dorder.at(0)], src_dim[dorder.at(1)], src_dim[dorder.at(2)], src_dim[dorder.at(3)]};
    const uint64_t src_stride[4]  =  {src_dim[1] * src_dim[2] * src_dim[3], 
                                   src_dim[2] * src_dim[3], 
                                   src_dim[3],
                                   1 };
    const uint64_t dst_stride[4]  =  {dst_dim[1] * dst_dim[2] * dst_dim[3], 
                                   dst_dim[2] * dst_dim[3], 
                                   dst_dim[3],
                                   1 };

     uint32_t i_src[4] = {0, 0, 0, 0};
     uint32_t i_dst[4] = {0, 0, 0, 0};

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        for (uint32_t k = 0; k < 1; k++)
        {
                        //unroll k         block          thread
            src_index = k*dim_total*256 + dim_id * 256 + threadIdx.x;
            if(src_index < pixel_total){
                i_src[0] = magic_div_u32(src_index,                                                   magic_stride0, shift_stride0);
                i_src[1] = magic_div_u32(src_index -  i_src[0] * src_stride[0],                          magic_stride1, shift_stride1);
                i_src[2] = magic_div_u32(src_index -  i_src[0] * src_stride[0] -  i_src[1] * src_stride[1], magic_stride2, shift_stride2);
                i_src[3] = src_index -  i_src[0] * src_stride[0] -  i_src[1] * src_stride[1] -  i_src[2] * src_stride[2];
    
                i_dst[0] = i_src[dorder.at(0)];
                i_dst[1] = i_src[dorder.at(1)];
                i_dst[2] = i_src[dorder.at(2)];
                i_dst[3] = i_src[dorder.at(3)];
    
                dst_index = i_dst[0] * dst_stride[0] + i_dst[1] * dst_stride[1] + i_dst[2] * dst_stride[2] + i_dst[3] * dst_stride[3];
                dst[dst_index] = src[src_index];
            }
        }
    }
}

template <typename T, typename dst_order>
inline __device__ void general_4d_reorder_2p(T* dst,
                                               T* src,
                                               uint32_t dim_0,
                                               uint32_t dim_1,
                                               uint32_t dim_2,
                                               uint32_t dim_3,
                                               uint32_t dim_stride,
                                               uint32_t dim_total,
                                               uint32_t magic_stride0,
                                               uint32_t shift_stride0,
                                               uint32_t magic_stride1,
                                               uint32_t shift_stride1,
                                               uint32_t magic_stride2,
                                               uint32_t shift_stride2)
{
    constexpr auto dorder = dst_order{};
    uint32_t pixel_total = dim_0 * dim_1 * dim_2 * dim_3; 
    uint32_t src_index =0, dst_index=0;
    const uint64_t src_dim[4]  = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {src_dim[dorder.at(0)], src_dim[dorder.at(1)], src_dim[dorder.at(2)], src_dim[dorder.at(3)]};
    const uint64_t src_stride[4]  =  {src_dim[1] * src_dim[2] * src_dim[3], 
                                   src_dim[2] * src_dim[3], 
                                   src_dim[3],
                                   1 };
    const uint64_t dst_stride[4]  =  {dst_dim[1] * dst_dim[2] * dst_dim[3], 
                                   dst_dim[2] * dst_dim[3], 
                                   dst_dim[3],
                                   1 };

     uint32_t i_src[4] = {0, 0, 0, 0};
     uint32_t i_dst[4] = {0, 0, 0, 0};

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        for (uint32_t k = 0; k < 2; k++)
        {
                        //unroll k         block          thread
            src_index = k*dim_total*256 + dim_id * 256 + threadIdx.x;
            if(src_index < pixel_total){
                i_src[0] = magic_div_u32(src_index,                                                   magic_stride0, shift_stride0);
                i_src[1] = magic_div_u32(src_index -  i_src[0] * src_stride[0],                          magic_stride1, shift_stride1);
                i_src[2] = magic_div_u32(src_index -  i_src[0] * src_stride[0] -  i_src[1] * src_stride[1], magic_stride2, shift_stride2);
                i_src[3] = src_index -  i_src[0] * src_stride[0] -  i_src[1] * src_stride[1] -  i_src[2] * src_stride[2];
    
                i_dst[0] = i_src[dorder.at(0)];
                i_dst[1] = i_src[dorder.at(1)];
                i_dst[2] = i_src[dorder.at(2)];
                i_dst[3] = i_src[dorder.at(3)];
    
                dst_index = i_dst[0] * dst_stride[0] + i_dst[1] * dst_stride[1] + i_dst[2] * dst_stride[2] + i_dst[3] * dst_stride[3];
                dst[dst_index] = src[src_index];
            }
        }
    }
}

template <typename T, typename dst_order>
inline __device__ void general_4d_reorder_4p(T* dst,
                                               T* src,
                                               uint32_t dim_0,
                                               uint32_t dim_1,
                                               uint32_t dim_2,
                                               uint32_t dim_3,
                                               uint32_t dim_stride,
                                               uint32_t dim_total,
                                               uint32_t magic_stride0,
                                               uint32_t shift_stride0,
                                               uint32_t magic_stride1,
                                               uint32_t shift_stride1,
                                               uint32_t magic_stride2,
                                               uint32_t shift_stride2)
{
    constexpr auto dorder = dst_order{};
    uint32_t pixel_total = dim_0 * dim_1 * dim_2 * dim_3; 
    uint32_t src_index =0, dst_index=0;
    const uint64_t src_dim[4]  = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {src_dim[dorder.at(0)], src_dim[dorder.at(1)], src_dim[dorder.at(2)], src_dim[dorder.at(3)]};
    const uint64_t src_stride[4]  =  {src_dim[1] * src_dim[2] * src_dim[3], 
                                   src_dim[2] * src_dim[3], 
                                   src_dim[3],
                                   1 };
    const uint64_t dst_stride[4]  =  {dst_dim[1] * dst_dim[2] * dst_dim[3], 
                                   dst_dim[2] * dst_dim[3], 
                                   dst_dim[3],
                                   1 };

     uint32_t i_src[4] = {0, 0, 0, 0};
     uint32_t i_dst[4] = {0, 0, 0, 0};

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        for (uint32_t k = 0; k < 4; k++)
        {
                        //unroll k         block          thread
            src_index = k*dim_total*256 + dim_id * 256 + threadIdx.x;
            if(src_index < pixel_total){
                i_src[0] = magic_div_u32(src_index,                                                   magic_stride0, shift_stride0);
                i_src[1] = magic_div_u32(src_index -  i_src[0] * src_stride[0],                          magic_stride1, shift_stride1);
                i_src[2] = magic_div_u32(src_index -  i_src[0] * src_stride[0] -  i_src[1] * src_stride[1], magic_stride2, shift_stride2);
                i_src[3] = src_index -  i_src[0] * src_stride[0] -  i_src[1] * src_stride[1] -  i_src[2] * src_stride[2];
    
                i_dst[0] = i_src[dorder.at(0)];
                i_dst[1] = i_src[dorder.at(1)];
                i_dst[2] = i_src[dorder.at(2)];
                i_dst[3] = i_src[dorder.at(3)];
    
                dst_index = i_dst[0] * dst_stride[0] + i_dst[1] * dst_stride[1] + i_dst[2] * dst_stride[2] + i_dst[3] * dst_stride[3];
                dst[dst_index] = src[src_index];
            }
        }
    }
}

template <typename T, typename dst_order>
inline __device__ void general_4d_reorder_8p(T* dst,
                                               T* src,
                                               uint32_t dim_0,
                                               uint32_t dim_1,
                                               uint32_t dim_2,
                                               uint32_t dim_3,
                                               uint32_t dim_stride,
                                               uint32_t dim_total,
                                               uint32_t magic_stride0,
                                               uint32_t shift_stride0,
                                               uint32_t magic_stride1,
                                               uint32_t shift_stride1,
                                               uint32_t magic_stride2,
                                               uint32_t shift_stride2)
{
    constexpr auto dorder = dst_order{};
    uint32_t pixel_total = dim_0 * dim_1 * dim_2 * dim_3; 
    uint32_t src_index =0, dst_index=0;
    const uint64_t src_dim[4]  = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {src_dim[dorder.at(0)], src_dim[dorder.at(1)], src_dim[dorder.at(2)], src_dim[dorder.at(3)]};
    const uint64_t src_stride[4]  =  {src_dim[1] * src_dim[2] * src_dim[3], 
                                   src_dim[2] * src_dim[3], 
                                   src_dim[3],
                                   1 };
    const uint64_t dst_stride[4]  =  {dst_dim[1] * dst_dim[2] * dst_dim[3], 
                                   dst_dim[2] * dst_dim[3], 
                                   dst_dim[3],
                                   1 };

     uint32_t i_src[4] = {0, 0, 0, 0};
     uint32_t i_dst[4] = {0, 0, 0, 0};

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        for (uint32_t k = 0; k < 8; k++)
        {
                        //unroll k         block          thread
            src_index = k*dim_total*256 + dim_id * 256 + threadIdx.x;
            if(src_index < pixel_total){
                i_src[0] = magic_div_u32(src_index,                                                   magic_stride0, shift_stride0);
                i_src[1] = magic_div_u32(src_index -  i_src[0] * src_stride[0],                          magic_stride1, shift_stride1);
                i_src[2] = magic_div_u32(src_index -  i_src[0] * src_stride[0] -  i_src[1] * src_stride[1], magic_stride2, shift_stride2);
                i_src[3] = src_index -  i_src[0] * src_stride[0] -  i_src[1] * src_stride[1] -  i_src[2] * src_stride[2];
    
                i_dst[0] = i_src[dorder.at(0)];
                i_dst[1] = i_src[dorder.at(1)];
                i_dst[2] = i_src[dorder.at(2)];
                i_dst[3] = i_src[dorder.at(3)];
    
                dst_index = i_dst[0] * dst_stride[0] + i_dst[1] * dst_stride[1] + i_dst[2] * dst_stride[2] + i_dst[3] * dst_stride[3];
                dst[dst_index] = src[src_index];
            }
        }
    }
}

template <typename T, typename dst_order>
inline __device__ void general_4d_reorder_16p(T* dst,
                                               T* src,
                                               uint32_t dim_0,
                                               uint32_t dim_1,
                                               uint32_t dim_2,
                                               uint32_t dim_3,
                                               uint32_t dim_stride,
                                               uint32_t dim_total,
                                               uint32_t magic_stride0,
                                               uint32_t shift_stride0,
                                               uint32_t magic_stride1,
                                               uint32_t shift_stride1,
                                               uint32_t magic_stride2,
                                               uint32_t shift_stride2)
{
    constexpr auto dorder = dst_order{};
    uint32_t pixel_total = dim_0 * dim_1 * dim_2 * dim_3; 
    uint32_t src_index =0, dst_index=0;
    const uint64_t src_dim[4]  = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {src_dim[dorder.at(0)], src_dim[dorder.at(1)], src_dim[dorder.at(2)], src_dim[dorder.at(3)]};
    const uint64_t src_stride[4]  =  {src_dim[1] * src_dim[2] * src_dim[3], 
                                   src_dim[2] * src_dim[3], 
                                   src_dim[3],
                                   1 };
    const uint64_t dst_stride[4]  =  {dst_dim[1] * dst_dim[2] * dst_dim[3], 
                                   dst_dim[2] * dst_dim[3], 
                                   dst_dim[3],
                                   1 };

     uint32_t i_src[4] = {0, 0, 0, 0};
     uint32_t i_dst[4] = {0, 0, 0, 0};

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        for (uint32_t k = 0; k < 16; k++)
        {
                        //unroll k         block          thread
            src_index = k*dim_total*256 + dim_id * 256 + threadIdx.x;
            if(src_index < pixel_total){
                i_src[0] = magic_div_u32(src_index,                                                   magic_stride0, shift_stride0);
                i_src[1] = magic_div_u32(src_index -  i_src[0] * src_stride[0],                          magic_stride1, shift_stride1);
                i_src[2] = magic_div_u32(src_index -  i_src[0] * src_stride[0] -  i_src[1] * src_stride[1], magic_stride2, shift_stride2);
                i_src[3] = src_index -  i_src[0] * src_stride[0] -  i_src[1] * src_stride[1] -  i_src[2] * src_stride[2];
    
                i_dst[0] = i_src[dorder.at(0)];
                i_dst[1] = i_src[dorder.at(1)];
                i_dst[2] = i_src[dorder.at(2)];
                i_dst[3] = i_src[dorder.at(3)];
    
                dst_index = i_dst[0] * dst_stride[0] + i_dst[1] * dst_stride[1] + i_dst[2] * dst_stride[2] + i_dst[3] * dst_stride[3];
                dst[dst_index] = src[src_index];
            }
        }
    }
}

#define DEFINE_GENERAL_4D_REORDER_KERNEL(                                                                  \
    tile_trait, dst_order, accept_data_type, cast_data_type, lb_threads_per_block, lb_blocks_per_cu)       \
    extern "C" __global__ void __launch_bounds__(lb_threads_per_block, lb_blocks_per_cu)                   \
        general_4d_reorder_##tile_trait##_##accept_data_type##_##dst_order(void* dst,                      \
                                                                         void* src,                        \
                                                                         uint32_t dim_0,                   \
                                                                         uint32_t dim_1,                   \
                                                                         uint32_t dim_2,                   \
                                                                         uint32_t dim_3,                   \
                                                                         uint32_t dim_stride,              \
                                                                         uint32_t dim_total,               \
                                                                         uint32_t magic_stride0,           \
                                                                         uint32_t shift_stride0,           \
                                                                         uint32_t magic_stride1,           \
                                                                         uint32_t shift_stride1,           \
                                                                         uint32_t magic_stride2,           \
                                                                         uint32_t shift_stride2)          \
    {                                                                                                      \
        general_4d_reorder_##tile_trait<cast_data_type, dst_order>(reinterpret_cast<cast_data_type*>(dst), \
                                                                   reinterpret_cast<cast_data_type*>(src), \
                                                                   dim_0,                                  \
                                                                   dim_1,                                  \
                                                                   dim_2,                                  \
                                                                   dim_3,                                  \
                                                                   dim_stride,                             \
                                                                   dim_total,                              \
                                                                   magic_stride0,                          \
                                                                   shift_stride0,                          \
                                                                   magic_stride1,                          \
                                                                   shift_stride1,                          \
                                                                   magic_stride2,                          \
                                                                   shift_stride2);                         \
    }
//default order is 0 1 2 3
using r0132   = sequence<0, 1, 3, 2>;
using r0213   = sequence<0, 2, 1, 3>;//nhwc2nchwc
using r0231   = sequence<0, 2, 3, 1>;//nchw2nchwc
using r0312   = sequence<0, 3, 1, 2>;//nhwc2nchw
using r0321   = sequence<0, 3, 2, 1>;
using r1023   = sequence<1, 0, 2, 3>;
using r1032   = sequence<1, 0, 3, 2>;
using r1203   = sequence<1, 2, 0, 3>;
using r1230   = sequence<1, 2, 3, 0>;
using r1302   = sequence<1, 3, 0, 2>;//nchw2chwnc
using r1320   = sequence<1, 3, 2, 0>;
using r2013   = sequence<2, 0, 1, 3>;
using r2031   = sequence<2, 0, 3, 1>;
using r2103   = sequence<2, 1, 0, 3>;//nhwc2chwnc
using r2130   = sequence<2, 1, 3, 0>;
using r2301   = sequence<2, 3, 0, 1>;
using r2310   = sequence<2, 3, 1, 0>;
using r3012   = sequence<3, 0, 1, 2>;
using r3021   = sequence<3, 0, 2, 1>;
using r3102   = sequence<3, 1, 0, 2>;
using r3120   = sequence<3, 1, 2, 0>;
using r3201   = sequence<3, 2, 0, 1>;
using r3210   = sequence<3, 2, 1, 0>;

DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0132, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0213, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0231, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0312, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0321, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1023, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1032, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1203, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1230, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1302, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1320, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2013, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2031, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2103, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2130, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2301, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2310, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3012, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3021, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3102, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3120, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3201, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3210, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0132, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0213, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0231, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0312, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0321, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1023, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1032, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1203, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1230, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1302, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1320, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2013, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2031, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2103, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2130, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2301, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2310, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3012, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3021, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3102, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3120, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3201, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3210, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0132, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0213, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0231, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0312, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0321, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1023, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1032, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1203, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1230, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1302, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1320, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2013, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2031, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2103, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2130, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2301, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2310, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3012, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3021, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3102, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3120, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3201, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3210, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0132, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0213, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0231, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0312, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0321, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1023, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1032, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1203, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1230, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1302, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1320, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2013, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2031, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2103, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2130, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2301, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2310, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3012, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3021, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3102, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3120, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3201, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3210, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0132, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0213, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0231, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0312, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0321, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1023, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1032, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1203, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1230, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1302, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1320, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2013, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2031, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2103, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2130, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2301, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2310, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3012, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3021, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3102, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3120, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3201, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3210, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)


DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0132, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0213, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0231, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0312, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0321, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1023, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1032, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1203, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1230, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1302, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1320, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2013, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2031, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2103, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2130, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2301, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2310, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3012, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3021, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3102, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3120, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3201, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3210, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0132, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0213, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0231, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0312, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0321, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1023, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1032, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1203, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1230, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1302, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1320, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2013, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2031, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2103, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2130, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2301, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2310, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3012, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3021, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3102, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3120, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3201, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3210, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0132, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0213, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0231, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0312, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0321, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1023, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1032, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1203, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1230, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1302, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1320, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2013, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2031, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2103, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2130, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2301, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2310, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3012, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3021, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3102, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3120, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3201, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3210, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0132, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0213, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0231, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0312, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0321, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1023, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1032, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1203, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1230, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1302, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1320, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2013, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2031, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2103, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2130, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2301, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2310, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3012, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3021, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3102, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3120, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3201, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3210, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0132, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0213, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0231, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0312, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0321, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1023, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1032, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1203, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1230, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1302, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1320, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2013, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2031, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2103, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2130, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2301, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2310, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3012, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3021, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3102, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3120, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3201, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3210, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)


DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0132, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0213, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0231, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0312, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r0321, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1023, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1032, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1203, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1230, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1302, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r1320, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2013, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2031, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2103, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2130, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2301, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r2310, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3012, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3021, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3102, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3120, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3201, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, r3210, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0132, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0213, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0231, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0312, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r0321, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1023, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1032, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1203, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1230, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1302, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r1320, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2013, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2031, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2103, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2130, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2301, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r2310, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3012, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3021, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3102, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3120, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3201, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, r3210, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0132, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0213, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0231, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0312, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r0321, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1023, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1032, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1203, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1230, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1302, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r1320, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2013, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2031, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2103, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2130, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2301, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r2310, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3012, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3021, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3102, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3120, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3201, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, r3210, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0132, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0213, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0231, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0312, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r0321, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1023, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1032, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1203, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1230, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1302, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r1320, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2013, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2031, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2103, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2130, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2301, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r2310, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3012, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3021, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3102, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3120, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3201, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, r3210, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0132, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0213, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0231, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0312, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r0321, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1023, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1032, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1203, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1230, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1302, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r1320, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2013, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2031, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2103, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2130, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2301, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r2310, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3012, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3021, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3102, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3120, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3201, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, r3210, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)