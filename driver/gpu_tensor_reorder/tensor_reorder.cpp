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
#include "../sequence.hpp"

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
    /*
     * assume input is 0, 1, 2, 3, 
     */
    //DIM_3%vectorc==0
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
    /*
     * assume input is 0, 1, 2, 3, 
     */
    //DIM_3%vectorc==0
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
    /*
     * assume input is 0, 1, 2, 3, 
     */
    //DIM_3%vectorc==0
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
    /*
     * assume input is 0, 1, 2, 3, 
     */
    //DIM_3%vectorc==0
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
    /*
     * assume input is 0, 1, 2, 3, 
     */
    //DIM_3%vectorc==0
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
using nchw2ncwh   = sequence<0, 1, 3, 2>;
using nchw2nhcw   = sequence<0, 2, 1, 3>;
using nchw2nhwc   = sequence<0, 2, 3, 1>;
using nchw2nwch   = sequence<0, 3, 1, 2>;//nhwc2nchw
using nchw2nwhc   = sequence<0, 3, 2, 1>;
using nchw2cnhw   = sequence<1, 0, 2, 3>;
using nchw2cnwh   = sequence<1, 0, 3, 2>;
using nchw2chnw   = sequence<1, 2, 0, 3>;
using nchw2chwn   = sequence<1, 2, 3, 0>;
using nchw2cwnh   = sequence<1, 3, 0, 2>;//nchw2chwnc
using nchw2cwhn   = sequence<1, 3, 2, 0>;
using nchw2hncw   = sequence<2, 0, 1, 3>;
using nchw2hnwc   = sequence<2, 0, 3, 1>;
using nchw2hcnw   = sequence<2, 1, 0, 3>;//nhwc2chwnc
using nchw2hcwn   = sequence<2, 1, 3, 0>;
using nchw2hwnc   = sequence<2, 3, 0, 1>;
using nchw2hwcn   = sequence<2, 3, 1, 0>;
using nchw2wnch   = sequence<3, 0, 1, 2>;
using nchw2wnhc   = sequence<3, 0, 2, 1>;
using nchw2wcnh   = sequence<3, 1, 0, 2>;
using nchw2wchn   = sequence<3, 1, 2, 0>;
using nchw2whnc   = sequence<3, 2, 0, 1>;
using nchw2whcn   = sequence<3, 2, 1, 0>;

DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2ncwh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2nhcw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2nhwc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2nwch, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2nwhc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2cnhw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2cnwh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2chnw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2chwn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2cwnh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2cwhn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hncw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hnwc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hcnw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hcwn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hwnc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hwcn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2wnch, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2wnhc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2wcnh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2wchn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2whnc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2whcn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2ncwh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2nhcw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2nhwc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2nwch, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2nwhc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2cnhw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2cnwh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2chnw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2chwn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2cwnh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2cwhn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hncw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hnwc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hcnw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hcwn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hwnc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hwcn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2wnch, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2wnhc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2wcnh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2wchn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2whnc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2whcn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2ncwh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2nhcw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2nhwc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2nwch, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2nwhc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2cnhw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2cnwh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2chnw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2chwn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2cwnh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2cwhn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hncw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hnwc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hcnw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hcwn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hwnc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hwcn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2wnch, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2wnhc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2wcnh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2wchn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2whnc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2whcn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2ncwh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2nhcw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2nhwc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2nwch, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2nwhc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2cnhw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2cnwh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2chnw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2chwn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2cwnh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2cwhn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hncw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hnwc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hcnw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hcwn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hwnc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hwcn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2wnch, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2wnhc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2wcnh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2wchn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2whnc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2whcn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2ncwh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2nhcw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2nhwc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2nwch, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2nwhc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2cnhw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2cnwh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2chnw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2chwn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2cwnh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2cwhn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hncw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hnwc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hcnw, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hcwn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hwnc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hwcn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2wnch, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2wnhc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2wcnh, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2wchn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2whnc, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2whcn, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)


DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2ncwh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2nhcw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2nhwc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2nwch, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2nwhc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2cnhw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2cnwh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2chnw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2chwn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2cwnh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2cwhn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hncw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hnwc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hcnw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hcwn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hwnc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hwcn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2wnch, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2wnhc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2wcnh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2wchn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2whnc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2whcn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2ncwh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2nhcw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2nhwc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2nwch, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2nwhc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2cnhw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2cnwh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2chnw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2chwn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2cwnh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2cwhn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hncw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hnwc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hcnw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hcwn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hwnc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hwcn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2wnch, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2wnhc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2wcnh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2wchn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2whnc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2whcn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2ncwh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2nhcw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2nhwc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2nwch, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2nwhc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2cnhw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2cnwh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2chnw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2chwn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2cwnh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2cwhn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hncw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hnwc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hcnw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hcwn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hwnc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hwcn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2wnch, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2wnhc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2wcnh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2wchn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2whnc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2whcn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2ncwh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2nhcw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2nhwc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2nwch, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2nwhc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2cnhw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2cnwh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2chnw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2chwn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2cwnh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2cwhn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hncw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hnwc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hcnw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hcwn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hwnc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hwcn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2wnch, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2wnhc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2wcnh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2wchn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2whnc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2whcn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2ncwh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2nhcw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2nhwc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2nwch, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2nwhc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2cnhw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2cnwh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2chnw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2chwn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2cwnh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2cwhn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hncw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hnwc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hcnw, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hcwn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hwnc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hwcn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2wnch, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2wnhc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2wcnh, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2wchn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2whnc, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2whcn, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)


DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2ncwh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2nhcw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2nhwc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2nwch, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2nwhc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2cnhw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2cnwh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2chnw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2chwn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2cwnh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2cwhn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hncw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hnwc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hcnw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hcwn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hwnc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2hwcn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2wnch, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2wnhc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2wcnh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2wchn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2whnc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(1p, nchw2whcn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2ncwh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2nhcw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2nhwc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2nwch, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2nwhc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2cnhw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2cnwh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2chnw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2chwn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2cwnh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2cwhn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hncw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hnwc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hcnw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hcwn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hwnc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2hwcn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2wnch, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2wnhc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2wcnh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2wchn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2whnc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(2p, nchw2whcn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2ncwh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2nhcw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2nhwc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2nwch, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2nwhc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2cnhw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2cnwh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2chnw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2chwn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2cwnh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2cwhn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hncw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hnwc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hcnw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hcwn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hwnc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2hwcn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2wnch, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2wnhc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2wcnh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2wchn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2whnc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(4p, nchw2whcn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2ncwh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2nhcw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2nhwc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2nwch, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2nwhc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2cnhw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2cnwh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2chnw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2chwn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2cwnh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2cwhn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hncw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hnwc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hcnw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hcwn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hwnc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2hwcn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2wnch, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2wnhc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2wcnh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2wchn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2whnc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(8p, nchw2whcn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2ncwh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2nhcw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2nhwc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2nwch, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2nwhc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2cnhw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2cnwh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2chnw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2chwn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2cwnh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2cwhn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hncw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hnwc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hcnw, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hcwn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hwnc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2hwcn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2wnch, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2wnhc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2wcnh, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2wchn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2whnc, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_GENERAL_4D_REORDER_KERNEL(16p, nchw2whcn, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)