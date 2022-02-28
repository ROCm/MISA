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
#ifndef __COMMON_H
#define __COMMON_H

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <float.h>
#include <functional>
#include <cstdint>
#include <stdlib.h>

// return_code : -1, not applicable
//             : -2, need skip, unique_index not accumulate
//             :  0, success
typedef struct {
    int return_code     {-1};
    int gks             {0};  // this is to store the gks value after benchmarked.
    int grid_size       {0};
    float duration_ms   {FLT_MAX};
    float gflops        {0};
    float efficiency    {0};
    std::string kernel_name;
} result_t;

static inline size_t conv_out_size(size_t in_size, size_t pad, size_t dilation,
                                   size_t ksize, size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

#define HIP_CALL(call)                                                         \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            printf("[hiperror](%d) fail to call %s,(%s)", (int)err, #call,     \
                   hipGetErrorString(err));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define MIOPEN_USE_RNE_BFLOAT16 1
class bfloat16
{
    public:
    bfloat16() : data_{0} {}
    explicit bfloat16(float rhs)
    {
        union
        {
            float float_st;
            std::uint32_t bf16_st;
        } bits_st = {rhs};

        // BF16 round and NaN preservation code matches
        // https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/include/rocblas_bfloat16.h
        if((~bits_st.bf16_st & 0x7f800000) == 0) // Inf or NaN
        {
            // When all of the exponent bits are 1, the value is Inf or NaN.
            // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
            // mantissa bit. Quiet NaN is indicated by the most significant mantissa
            // bit being 1. Signaling NaN is indicated by the most significant
            // mantissa bit being 0 but some other bit(s) being 1. If any of the
            // lower 16 bits of the mantissa are 1, we set the least significant bit
            // of the bfloat16 mantissa, in order to preserve signaling NaN in case
            // the bloat16's mantissa bits are all 0.
            if((bits_st.bf16_st & 0xffff) != 0)
            {
                bits_st.bf16_st |= 0x10000; // Preserve signaling NaN
            }
        }
        else
        {
#if MIOPEN_USE_RNE_BFLOAT16 == 1
            // When the exponent bits are not all 1s, then the value is zero, normal,
            // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
            // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
            // This causes the bfloat16's mantissa to be incremented by 1 if the 16
            // least significant bits of the float mantissa are greater than 0x8000,
            // or if they are equal to 0x8000 and the least significant bit of the
            // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
            // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
            // has the value 0x7f, then incrementing it causes it to become 0x00 and
            // the exponent is incremented by one, which is the next higher FP value
            // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
            // with an exponent of 0x00 and a mantissa of 0x7F, it may be rounded up
            // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
            // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
            // incrementing it causes it to become an exponent of 0xFF and a mantissa
            // of 0x00, which is Inf, the next higher value to the unrounded value.
            bits_st.bf16_st +=
                (0x7fff + ((bits_st.bf16_st >> 16) & 1)); // Round to nearest, round to even
#else                                                     // truncation
// do nothing
#endif
        }
        data_ = bits_st.bf16_st >> 16;
    }
    operator float() const
    {
        union
        {
            std::uint32_t bf16_st;
            float float_st;
        } bits_st = {data_};

        bits_st.bf16_st = bits_st.bf16_st << 16;
        return bits_st.float_st;
    }

    bfloat16 operator-() const { return bfloat16(-static_cast<float>(*this)); }
    bfloat16 operator+() const { return *this; }

    bfloat16& operator=(const float rhs)
    {
        *this = bfloat16(rhs);
        return *this;
    }
    bfloat16& operator+=(bfloat16 rhs)
    {
        *this = bfloat16(static_cast<float>(*this) + static_cast<float>(rhs));
        return *this;
    }

    bfloat16& operator+=(float rhs)
    {
        *this = bfloat16(static_cast<float>(*this) + rhs);
        return *this;
    }

    bfloat16& operator-=(bfloat16 rhs)
    {
        *this += -rhs;
        return *this;
    }
    bfloat16& operator*=(bfloat16 rhs)
    {
        *this = bfloat16(static_cast<float>(*this) * static_cast<float>(rhs));
        return *this;
    }
    bfloat16& operator*=(float rhs)
    {
        *this = bfloat16(static_cast<float>(*this) * rhs);
        return *this;
    }

    bfloat16& operator/=(bfloat16 rhs)
    {
        *this = bfloat16(static_cast<float>(*this) / static_cast<float>(rhs));
        return *this;
    }
    bool operator<(bfloat16 rhs) const
    {
        return static_cast<float>(*this) < static_cast<float>(rhs);
    }
    bool operator==(bfloat16 rhs) const { return std::equal_to<float>()(*this, rhs); }

    static constexpr bfloat16 generate(uint16_t val) { return bfloat16{val, true}; }

    private:
    constexpr bfloat16(std::uint16_t val, bool) : data_{val} {}

    std::uint16_t data_;
};

static inline int env_get_int(const char *var_name, int default_int) {
    char *v = getenv(var_name);
    int r = default_int;
    if (v)
        r = atoi(v);
    return r;
}

static inline char *env_get_str(const char *var_name, char *default_str) {
    char *v = getenv(var_name);
    if (v)
        return v;
    return default_str;
}

// return 1 for case like "conv", "convfp16"
// return vec_c for case ike "convfp16x8", "convint8x16"...
static inline int find_vector_c_from_base_arg(const std::string& base_arg)
{
    if(base_arg.compare(0, 4, "conv") == 0){
        auto found_vec = base_arg.find("x");       // we are looking for the "x" character
        if(found_vec != std::string::npos){
            std::string vec_str = base_arg.substr(found_vec + 1);
            int vector_c = std::stoi( vec_str );    // TODO: try...catch...
            return vector_c;
        }else{
            return 1;
        }
    }
    return 0;
}

#endif
