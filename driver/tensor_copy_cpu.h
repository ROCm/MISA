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
#ifndef _TENSOR_COPY_CPU_H
#define _TENSOR_COPY_CPU_H

#include <stddef.h>
#include <thread>

typedef struct
{
    union
    {
        int8_t v;
        struct
        {
            int lo : 4;
            int hi : 4;
        };
    };
}int4x2_t;

template <typename Dst_T, typename Src_T>
void block_wise_tensor_copy(Dst_T *p_dst, Src_T *p_src, int tid, size_t block_size, size_t total_size)
{
    for (int i = tid; i < total_size; i += block_size) {
        p_dst[i] = static_cast<Dst_T>(p_src[i]);
    }
}

template <>
void block_wise_tensor_copy<int4x2_t, float>(int4x2_t *p_dst, float *p_src, int tid, size_t block_size, size_t total_size)
{
    // sizeof(int4x2_t) is 4. So need to find a way to avoid seg fault
    int8_t *tmp_dst = (int8_t*)(p_dst);
    for (int i = tid; i < (total_size / 2); i += block_size) {
        int8_t lo = static_cast<int8_t>(p_src[2 * i]);
        int8_t hi = static_cast<int8_t>(p_src[2 * i + 1]);

        lo = lo & 0xf;
        hi = hi & 0xf;

        int8_t composed = (hi << 4) + lo;
        tmp_dst[i] = composed;
    }
}

template <typename Dst_T, typename Src_T>
void tensor_copy(Dst_T *p_dst, Src_T *p_src, size_t tensor_size) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads < 4)
        num_threads = 4;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.push_back(std::thread(block_wise_tensor_copy<Dst_T, Src_T>,
            p_dst, p_src, t, num_threads, tensor_size));
    }
    for (auto &th : threads)
        th.join();
}


#endif