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

#ifndef __UTILITY_H
#define __UTILITY_H

#include <string>
#include <vector>
#include <assert.h>

template <typename T>
T utility_gcd(T x, T y)
{
    if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return utility_gcd(x - y, y);
    }
    else
    {
        return utility_gcd(x, y - x);
    }
}

template <typename T>
T utility_integer_divide_floor(T x, T y)
{
    return x / y;
}

template <typename T>
T utility_integer_divide_ceil(T x, T y)
{
    return (x + y - 1) / y;
}

template <typename T>
T utility_max(T x, T y)
{
    return x > y ? x : y;
}

template <typename T>
T utility_min(T x, T y)
{
    return x < y ? x : y;
}

static inline std::string
utility_int_list_to_string(const std::vector<int> list){
    std::string enc;
    for(int i=0;i<list.size();i++){
        enc.append(std::to_string(list[i]));
        if(i != (list.size() - 1))
            enc.append("x");
    }
    return enc;
}

static inline uint32_t utility_next_pow2(uint32_t n) {
    if (n == 0)
        return 1;
    if ((n & (n - 1)) == 0)
        return n;
    while ((n & (n - 1)) > 0)
        n &= (n - 1);
    return n << 1;
}

static inline uint32_t utility_prev_pow2(uint32_t n) {
    n = n | (n >> 1);
    n = n | (n >> 2);
    n = n | (n >> 4);
    n = n | (n >> 8);
    n = n | (n >> 16);
    return n - (n >> 1);
}


static inline int utility_string_to_data_byte(std::string precision)
{
    if(precision == "fp32")
        return 4;
    if(precision == "fp16" || precision == "bf16")
        return 2;
    if(precision == "int8")
        return 1;
    assert(false);
    return 1;
}


#endif