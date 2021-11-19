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
#ifndef _TENSOR_TRANSPOSE_H
#define _TENSOR_TRANSPOSE_H
#include <assert.h>
#include <thread>
#include <vector>
#include <functional>

template<typename dtype>
void tensor_transpose_nchw_2_nchwc(void* dst_ptr, void* src_ptr, size_t n, size_t c, size_t h, size_t w, size_t vector_c){
    dtype tmp_src = static_cast<dtype>(src_ptr);
    dtype tmp_dst = static_cast<dtype>(dst_ptr);

    assert(c % vector_c == 0);

    for(size_t i_n = 0; i_n < n; i_n++){
        for(size_t i_c = 0; i_c < c; i_c += vector_c){
            for(size_t i_hw = 0; i_hw < h * w; i_hw++){
                for(size_t i_vecc = 0; i_vecc < vector_c; i_vecc++){
                    size_t dst_index = i_n * c * h * w + i_c * h * w + i_hw * vector_c + i_vecc;
                    size_t src_index = i_n * c * h * w + (i_c + i_vecc) * h * w + i_hw;
                    tmp_dst[dst_index] = tmp_src[src_index];
                }
            }
        }
    }
}

// for output it will be nkhwk to nkhw
template<typename dtype>
void tensor_transpose_nchwc_2_nchw(void* dst_ptr, void* src_ptr, size_t n, size_t c, size_t h, size_t w, size_t vector_c){
    dtype tmp_src = static_cast<dtype>(src_ptr);
    dtype tmp_dst = static_cast<dtype>(dst_ptr);

    assert(c % vector_c == 0);
    for(size_t i_n = 0; i_n < n; i_n++){
        for(size_t i_c = 0; i_c < c; i_c += vector_c){
            for(size_t i_hw = 0; i_hw < h * w; i_hw++){
                for(size_t i_vecc = 0; i_vecc < vector_c; i_vecc++){
                    size_t src_index = i_n * c * h * w + i_c * h * w + i_hw * vector_c + i_vecc;
                    size_t dst_index = i_n * c * h * w + (i_c + i_vecc) * h * w + i_hw;
                    tmp_dst[dst_index] = tmp_src[src_index];
                }
            }
        }
    }
}

// for weight, it will be kcyx to cyxkc
template<typename dtype>
void tensor_transpose_nchw_2_chwnc(void* dst_ptr, void* src_ptr, size_t n, size_t c, size_t h, size_t w, size_t vector_c){
    dtype tmp_src = static_cast<dtype>(src_ptr);
    dtype tmp_dst = static_cast<dtype>(dst_ptr);

    assert(c % vector_c == 0);
    for(size_t i_n = 0; i_n < n; i_n++){
        for(size_t i_c = 0; i_c < c; i_c += vector_c){
            for(size_t i_hw = 0; i_hw < h * w; i_hw++){
                for(size_t i_vecc = 0; i_vecc < vector_c; i_vecc++){
                    size_t src_index = i_n * c * h * w + (i_c + i_vecc) * h * w + i_hw;
                    size_t dst_index = i_c * h * w * n + i_hw * n * vector_c + i_n * vector_c + i_vecc;
                    tmp_dst[dst_index] = tmp_src[src_index];
                }
            }
        }
    }
}

template<typename dtype>
void tensor_transpose_chwnc_2_nchw(void* dst_ptr, void* src_ptr, size_t n, size_t c, size_t h, size_t w, size_t vector_c){
    dtype tmp_src = static_cast<dtype>(src_ptr);
    dtype tmp_dst = static_cast<dtype>(dst_ptr);

    assert(c % vector_c == 0);
    for(size_t i_n = 0; i_n < n; i_n++){
        for(size_t i_c = 0; i_c < c; i_c += vector_c){
            for(size_t i_hw = 0; i_hw < h * w; i_hw++){
                for(size_t i_vecc = 0; i_vecc < vector_c; i_vecc++){
                    size_t dst_index = i_n * c * h * w + (i_c + i_vecc) * h * w + i_hw;
                    size_t src_index = i_c * h * w * n + i_hw * n * vector_c + i_n * vector_c + i_vecc;
                    tmp_dst[dst_index] = tmp_src[src_index];
                }
            }
        }
    }
}

#endif