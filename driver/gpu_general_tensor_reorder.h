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
#ifndef __GPU_GENERAL_TENSOR_REORDER_H
#define __GPU_GENERAL_TENSOR_REORDER_H

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <assert.h>
#include "magic_div.h"

#ifndef HIP_CALL
#define HIP_CALL(call)                                                         \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            printf("[hiperror](%d) fail to call %s,(%s)\n", (int)err, #call,     \
                   hipGetErrorString(err));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)
#endif

static struct {
    hipModule_t     module;

    hipFunction_t   kernel_general_4d_reorder_1p_dword_r0132;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r0213;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r0231;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r0312;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r0321;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r1023;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r1032;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r1203;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r1230;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r1302;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r1320;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r2013;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r2031;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r2103;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r2130;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r2301;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r2310;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r3012;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r3021;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r3102;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r3120;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r3201;
    hipFunction_t   kernel_general_4d_reorder_1p_dword_r3210;

    hipFunction_t   kernel_general_4d_reorder_2p_dword_r0132;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r0213;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r0231;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r0312;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r0321;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r1023;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r1032;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r1203;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r1230;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r1302;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r1320;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r2013;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r2031;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r2103;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r2130;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r2301;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r2310;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r3012;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r3021;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r3102;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r3120;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r3201;
    hipFunction_t   kernel_general_4d_reorder_2p_dword_r3210;

    hipFunction_t   kernel_general_4d_reorder_4p_dword_r0132;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r0213;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r0231;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r0312;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r0321;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r1023;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r1032;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r1203;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r1230;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r1302;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r1320;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r2013;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r2031;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r2103;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r2130;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r2301;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r2310;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r3012;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r3021;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r3102;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r3120;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r3201;
    hipFunction_t   kernel_general_4d_reorder_4p_dword_r3210;

    hipFunction_t   kernel_general_4d_reorder_8p_dword_r0132;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r0213;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r0231;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r0312;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r0321;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r1023;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r1032;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r1203;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r1230;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r1302;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r1320;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r2013;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r2031;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r2103;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r2130;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r2301;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r2310;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r3012;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r3021;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r3102;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r3120;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r3201;
    hipFunction_t   kernel_general_4d_reorder_8p_dword_r3210;

    hipFunction_t   kernel_general_4d_reorder_16p_dword_r0132;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r0213;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r0231;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r0312;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r0321;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r1023;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r1032;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r1203;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r1230;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r1302;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r1320;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r2013;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r2031;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r2103;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r2130;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r2301;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r2310;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r3012;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r3021;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r3102;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r3120;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r3201;
    hipFunction_t   kernel_general_4d_reorder_16p_dword_r3210;

    hipFunction_t   kernel_general_4d_reorder_1p_half_r0132;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r0213;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r0231;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r0312;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r0321;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r1023;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r1032;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r1203;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r1230;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r1302;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r1320;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r2013;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r2031;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r2103;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r2130;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r2301;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r2310;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r3012;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r3021;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r3102;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r3120;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r3201;
    hipFunction_t   kernel_general_4d_reorder_1p_half_r3210;

    hipFunction_t   kernel_general_4d_reorder_2p_half_r0132;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r0213;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r0231;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r0312;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r0321;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r1023;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r1032;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r1203;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r1230;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r1302;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r1320;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r2013;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r2031;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r2103;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r2130;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r2301;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r2310;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r3012;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r3021;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r3102;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r3120;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r3201;
    hipFunction_t   kernel_general_4d_reorder_2p_half_r3210;

    hipFunction_t   kernel_general_4d_reorder_4p_half_r0132;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r0213;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r0231;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r0312;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r0321;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r1023;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r1032;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r1203;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r1230;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r1302;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r1320;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r2013;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r2031;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r2103;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r2130;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r2301;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r2310;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r3012;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r3021;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r3102;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r3120;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r3201;
    hipFunction_t   kernel_general_4d_reorder_4p_half_r3210;

    hipFunction_t   kernel_general_4d_reorder_8p_half_r0132;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r0213;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r0231;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r0312;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r0321;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r1023;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r1032;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r1203;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r1230;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r1302;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r1320;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r2013;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r2031;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r2103;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r2130;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r2301;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r2310;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r3012;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r3021;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r3102;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r3120;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r3201;
    hipFunction_t   kernel_general_4d_reorder_8p_half_r3210;

    hipFunction_t   kernel_general_4d_reorder_16p_half_r0132;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r0213;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r0231;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r0312;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r0321;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r1023;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r1032;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r1203;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r1230;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r1302;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r1320;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r2013;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r2031;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r2103;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r2130;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r2301;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r2310;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r3012;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r3021;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r3102;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r3120;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r3201;
    hipFunction_t   kernel_general_4d_reorder_16p_half_r3210;

    hipFunction_t   kernel_general_4d_reorder_1p_byte_r0132;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r0213;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r0231;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r0312;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r0321;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r1023;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r1032;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r1203;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r1230;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r1302;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r1320;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r2013;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r2031;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r2103;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r2130;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r2301;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r2310;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r3012;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r3021;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r3102;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r3120;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r3201;
    hipFunction_t   kernel_general_4d_reorder_1p_byte_r3210;

    hipFunction_t   kernel_general_4d_reorder_2p_byte_r0132;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r0213;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r0231;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r0312;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r0321;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r1023;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r1032;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r1203;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r1230;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r1302;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r1320;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r2013;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r2031;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r2103;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r2130;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r2301;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r2310;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r3012;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r3021;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r3102;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r3120;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r3201;
    hipFunction_t   kernel_general_4d_reorder_2p_byte_r3210;

    hipFunction_t   kernel_general_4d_reorder_4p_byte_r0132;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r0213;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r0231;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r0312;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r0321;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r1023;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r1032;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r1203;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r1230;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r1302;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r1320;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r2013;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r2031;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r2103;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r2130;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r2301;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r2310;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r3012;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r3021;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r3102;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r3120;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r3201;
    hipFunction_t   kernel_general_4d_reorder_4p_byte_r3210;

    hipFunction_t   kernel_general_4d_reorder_8p_byte_r0132;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r0213;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r0231;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r0312;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r0321;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r1023;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r1032;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r1203;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r1230;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r1302;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r1320;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r2013;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r2031;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r2103;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r2130;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r2301;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r2310;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r3012;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r3021;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r3102;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r3120;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r3201;
    hipFunction_t   kernel_general_4d_reorder_8p_byte_r3210;

    hipFunction_t   kernel_general_4d_reorder_16p_byte_r0132;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r0213;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r0231;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r0312;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r0321;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r1023;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r1032;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r1203;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r1230;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r1302;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r1320;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r2013;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r2031;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r2103;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r2130;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r2301;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r2310;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r3012;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r3021;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r3102;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r3120;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r3201;
    hipFunction_t   kernel_general_4d_reorder_16p_byte_r3210;

} the_reorder_gpu_handle;

//reorder kernel
static inline void gpu_tensor_reorder_init(const char * hsaco){
    static int inited = 0;
    if(!inited){
        HIP_CALL(hipModuleLoad(&the_reorder_gpu_handle.module, hsaco));

        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_dword_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_dword_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_dword_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_dword_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_dword_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_dword_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_dword_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_dword_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_dword_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_dword_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_dword_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_dword_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_dword_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_dword_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_dword_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_dword_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_dword_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_dword_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_dword_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_dword_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_dword_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_dword_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_dword_r3210"));

        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_dword_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_dword_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_dword_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_dword_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_dword_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_dword_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_dword_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_dword_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_dword_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_dword_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_dword_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_dword_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_dword_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_dword_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_dword_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_dword_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_dword_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_dword_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_dword_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_dword_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_dword_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_dword_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_dword_r3210"));

        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_dword_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_dword_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_dword_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_dword_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_dword_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_dword_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_dword_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_dword_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_dword_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_dword_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_dword_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_dword_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_dword_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_dword_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_dword_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_dword_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_dword_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_dword_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_dword_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_dword_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_dword_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_dword_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_dword_r3210"));
        
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_dword_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_dword_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_dword_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_dword_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_dword_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_dword_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_dword_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_dword_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_dword_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_dword_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_dword_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_dword_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_dword_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_dword_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_dword_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_dword_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_dword_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_dword_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_dword_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_dword_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_dword_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_dword_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_dword_r3210"));

        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_dword_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_dword_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_dword_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_dword_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_dword_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_dword_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_dword_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_dword_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_dword_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_dword_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_dword_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_dword_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_dword_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_dword_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_dword_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_dword_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_dword_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_dword_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_dword_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_dword_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_dword_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_dword_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_dword_r3210"));
        
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_half_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_half_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_half_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_half_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_half_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_half_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_half_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_half_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_half_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_half_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_half_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_half_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_half_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_half_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_half_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_half_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_half_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_half_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_half_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_half_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_half_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_half_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_half_r3210"));

        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_half_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_half_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_half_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_half_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_half_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_half_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_half_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_half_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_half_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_half_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_half_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_half_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_half_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_half_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_half_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_half_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_half_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_half_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_half_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_half_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_half_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_half_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_half_r3210"));

        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_half_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_half_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_half_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_half_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_half_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_half_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_half_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_half_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_half_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_half_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_half_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_half_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_half_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_half_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_half_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_half_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_half_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_half_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_half_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_half_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_half_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_half_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_half_r3210"));
        
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_half_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_half_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_half_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_half_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_half_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_half_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_half_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_half_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_half_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_half_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_half_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_half_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_half_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_half_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_half_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_half_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_half_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_half_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_half_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_half_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_half_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_half_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_half_r3210"));

        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_half_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_half_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_half_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_half_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_half_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_half_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_half_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_half_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_half_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_half_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_half_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_half_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_half_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_half_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_half_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_half_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_half_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_half_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_half_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_half_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_half_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_half_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_half_r3210"));

        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_byte_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_byte_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_byte_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_byte_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_byte_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_byte_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_byte_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_byte_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_byte_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_byte_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_byte_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_byte_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_byte_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_byte_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_byte_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_byte_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_byte_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_byte_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_byte_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_byte_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_byte_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_1p_byte_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_1p_byte_r3210"));

        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_byte_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_byte_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_byte_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_byte_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_byte_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_byte_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_byte_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_byte_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_byte_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_byte_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_byte_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_byte_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_byte_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_byte_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_byte_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_byte_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_byte_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_byte_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_byte_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_byte_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_byte_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_2p_byte_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_2p_byte_r3210"));

        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_byte_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_byte_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_byte_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_byte_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_byte_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_byte_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_byte_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_byte_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_byte_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_byte_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_byte_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_byte_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_byte_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_byte_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_byte_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_byte_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_byte_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_byte_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_byte_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_byte_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_byte_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_4p_byte_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_4p_byte_r3210"));
        
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_byte_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_byte_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_byte_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_byte_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_byte_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_byte_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_byte_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_byte_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_byte_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_byte_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_byte_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_byte_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_byte_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_byte_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_byte_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_byte_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_byte_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_byte_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_byte_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_byte_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_byte_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_8p_byte_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_8p_byte_r3210"));

        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r0132, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_byte_r0132"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r0213, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_byte_r0213"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r0231,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_byte_r0231"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r0312,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_byte_r0312"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r0321, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_byte_r0321"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r1023, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_byte_r1023"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r1032,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_byte_r1032"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r1203,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_byte_r1203"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r1230, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_byte_r1230"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r1302, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_byte_r1302"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r1320,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_byte_r1320"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r2013,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_byte_r2013"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r2031, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_byte_r2031"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r2103, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_byte_r2103"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r2130,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_byte_r2130"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r2301,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_byte_r2301"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r2310, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_byte_r2310"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r3012, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_byte_r3012"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r3021,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_byte_r3021"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r3102,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_byte_r3102"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r3120, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_byte_r3120"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r3201, the_reorder_gpu_handle.module,  "general_4d_reorder_16p_byte_r3201"));
        HIP_CALL(hipModuleGetFunction(&the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r3210,  the_reorder_gpu_handle.module, "general_4d_reorder_16p_byte_r3210"));

        inited = 1;
    }
}

typedef struct {
    void * p_dst;
    void * p_src;
    uint32_t dim_0;
    uint32_t dim_1;
    uint32_t dim_2;
    uint32_t dim_3;
    uint32_t dim_stride;
    uint32_t dim_total;
    uint32_t magic_stride0;
    uint32_t shift_stride0;
    uint32_t magic_stride1;
    uint32_t shift_stride1;
    uint32_t magic_stride2;
    uint32_t shift_stride2;
} __attribute__((packed)) reorder_kernel_t;

template<size_t type_size>
struct tensor_reorder_kernel_get_all_param_t{
};

template<>
struct tensor_reorder_kernel_get_all_param_t<4>{
    static std::vector<transpose_kernel_param_t> get(){
        std::vector<transpose_kernel_param_t> the_list {
            {1, 1, 1, 1, 1, 1},
            {2, 1, 1, 1, 1, 1},
            {4, 1, 1, 1, 1, 1},
            {8, 1, 1, 1, 1, 1},
            {16, 1, 1, 1, 1, 1},
        };
        return the_list;
    }
};

template<>
struct tensor_reorder_kernel_get_all_param_t<2>{
    static std::vector<transpose_kernel_param_t> get(){
        std::vector<transpose_kernel_param_t> the_list {
            {1, 1, 1, 1, 1, 1},
            {2, 1, 1, 1, 1, 1},
            {4, 1, 1, 1, 1, 1},
            {8, 1, 1, 1, 1, 1},
            {16, 1, 1, 1, 1, 1},
        };
        return the_list;
    }
};
template<>
struct tensor_reorder_kernel_get_all_param_t<1>{
    static std::vector<transpose_kernel_param_t> get(){
        std::vector<transpose_kernel_param_t> the_list {
            {1, 1, 1, 1, 1, 1},
            {2, 1, 1, 1, 1, 1},
            {4, 1, 1, 1, 1, 1},
            {8, 1, 1, 1, 1, 1},
            {16, 1, 1, 1, 1, 1},
        };
        return the_list;
    }
};

template<size_t type_size, typename dst_order>
struct reorder_kernel_select_t{
};

template<typename dst_order>
struct reorder_kernel_select_t<4, dst_order>{
    static hipFunction_t get(const transpose_kernel_param_t * kparam){
        //(0, ...)
        if(dst_order::at(0)==0 && dst_order::at(1)==1 && dst_order::at(2)==3 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r0132;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r0132;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r0132;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r0132;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r0132;
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==2 && dst_order::at(2)==1 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r0213;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r0213;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r0213;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r0213;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r0213;
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==2 && dst_order::at(2)==3 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r0231;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r0231;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r0231;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r0231;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r0231;      
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==3 && dst_order::at(2)==1 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r0312;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r0312;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r0312;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r0312;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r0312;          
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==3 && dst_order::at(2)==2 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r0321;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r0321;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r0321;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r0321;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r0321; 
        }
        //(1,...)
        else if(dst_order::at(0)==1 && dst_order::at(1)==0 && dst_order::at(2)==2 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r1023;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r1023;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r1023;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r1023;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r1023;      
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==0 && dst_order::at(2)==3 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r1032;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r1032;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r1032;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r1032;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r1032;           
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==2 && dst_order::at(2)==0 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r1203;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r1203;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r1203;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r1203;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r1203;  
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==2 && dst_order::at(2)==3 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r1230;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r1230;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r1230;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r1230;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r1230;        
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==3 && dst_order::at(2)==0 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r1302;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r1302;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r1302;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r1302;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r1302;          
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==3 && dst_order::at(2)==2 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r1320;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r1320;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r1320;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r1320;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r1320; 
        }
        //(2,...)
        else if(dst_order::at(0)==2 && dst_order::at(1)==0 && dst_order::at(2)==1 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r2013;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r2013;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r2013;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r2013;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r2013;       
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==0 && dst_order::at(2)==3 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r2031;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r2031;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r2031;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r2031;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r2031;          
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==1 && dst_order::at(2)==0 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r2103;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r2103;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r2103;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r2103;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r2103;
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==1 && dst_order::at(2)==3 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r2130;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r2130;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r2130;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r2130;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r2130;      
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==3 && dst_order::at(2)==0 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r2301;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r2301;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r2301;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r2301;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r2301;          
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==3 && dst_order::at(2)==1 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r2310;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r2310;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r2310;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r2310;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r2310;  
        }
        //(3,...)
        else if(dst_order::at(0)==3 && dst_order::at(1)==0 && dst_order::at(2)==1 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r3012;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r3012;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r3012;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r3012;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r3012;      
        }
        else if(dst_order::at(0)==3 && dst_order::at(1)==0 && dst_order::at(2)==2 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r3021;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r3021;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r3021;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r3021;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r3021;           
        }
        else if(dst_order::at(0)==3 && dst_order::at(1)==1 && dst_order::at(2)==0 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r3102;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r3102;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r3102;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r3102;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r3102; 
        }
        else if(dst_order::at(0)==3 && dst_order::at(1)==1 && dst_order::at(2)==2 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r3120;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r3120;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r3120;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r3120;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r3120;       
        }
        else if(dst_order::at(0)==3 && dst_order::at(1)==2 && dst_order::at(2)==0 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r3201;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r3201;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r3201;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r3201;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r3201;           
        }
        //else if(dst_order::at(0)==3 && dst_order::at(1)==2 && dst_order::at(2)==1 && dst_order::at(3)==0){
        else{
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_dword_r3210;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_dword_r3210;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_dword_r3210;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_dword_r3210;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_dword_r3210;          
        } 
    }
};

template<typename dst_order>
struct reorder_kernel_select_t<2, dst_order>{
    static hipFunction_t get(const transpose_kernel_param_t * kparam){
        //(0, ...)
        if(dst_order::at(0)==0 && dst_order::at(1)==1 && dst_order::at(2)==3 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r0132;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r0132;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r0132;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r0132;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r0132;
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==2 && dst_order::at(2)==1 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r0213;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r0213;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r0213;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r0213;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r0213;
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==2 && dst_order::at(2)==3 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r0231;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r0231;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r0231;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r0231;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r0231;      
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==3 && dst_order::at(2)==1 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r0312;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r0312;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r0312;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r0312;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r0312;          
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==3 && dst_order::at(2)==2 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r0321;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r0321;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r0321;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r0321;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r0321; 
        }
        //(1,...)
        else if(dst_order::at(0)==1 && dst_order::at(1)==0 && dst_order::at(2)==2 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r1023;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r1023;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r1023;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r1023;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r1023;      
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==0 && dst_order::at(2)==3 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r1032;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r1032;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r1032;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r1032;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r1032;           
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==2 && dst_order::at(2)==0 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r1203;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r1203;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r1203;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r1203;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r1203;  
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==2 && dst_order::at(2)==3 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r1230;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r1230;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r1230;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r1230;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r1230;        
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==3 && dst_order::at(2)==0 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r1302;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r1302;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r1302;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r1302;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r1302;          
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==3 && dst_order::at(2)==2 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r1320;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r1320;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r1320;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r1320;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r1320; 
        }
        //(2,...)
        else if(dst_order::at(0)==2 && dst_order::at(1)==0 && dst_order::at(2)==1 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r2013;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r2013;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r2013;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r2013;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r2013;       
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==0 && dst_order::at(2)==3 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r2031;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r2031;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r2031;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r2031;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r2031;          
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==1 && dst_order::at(2)==0 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r2103;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r2103;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r2103;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r2103;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r2103;
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==1 && dst_order::at(2)==3 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r2130;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r2130;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r2130;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r2130;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r2130;      
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==3 && dst_order::at(2)==0 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r2301;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r2301;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r2301;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r2301;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r2301;          
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==3 && dst_order::at(2)==1 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r2310;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r2310;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r2310;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r2310;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r2310;  
        }
        //(3,...)
        else if(dst_order::at(0)==3 && dst_order::at(1)==0 && dst_order::at(2)==1 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r3012;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r3012;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r3012;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r3012;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r3012;      
        }
        else if(dst_order::at(0)==3 && dst_order::at(1)==0 && dst_order::at(2)==2 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r3021;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r3021;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r3021;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r3021;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r3021;           
        }
        else if(dst_order::at(0)==3 && dst_order::at(1)==1 && dst_order::at(2)==0 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r3102;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r3102;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r3102;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r3102;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r3102; 
        }
        else if(dst_order::at(0)==3 && dst_order::at(1)==1 && dst_order::at(2)==2 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r3120;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r3120;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r3120;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r3120;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r3120;       
        }
        else if(dst_order::at(0)==3 && dst_order::at(1)==2 && dst_order::at(2)==0 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r3201;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r3201;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r3201;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r3201;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r3201;           
        }
        //else if(dst_order::at(0)==3 && dst_order::at(1)==2 && dst_order::at(2)==1 && dst_order::at(3)==0){
        else{
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_half_r3210;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_half_r3210;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_half_r3210;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_half_r3210;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_half_r3210;          
        } 
    }
};

template<typename dst_order>
struct reorder_kernel_select_t<1, dst_order>{
    static hipFunction_t get(const transpose_kernel_param_t * kparam){
        //(0, ...)
        if(dst_order::at(0)==0 && dst_order::at(1)==1 && dst_order::at(2)==3 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r0132;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r0132;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r0132;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r0132;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r0132;
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==2 && dst_order::at(2)==1 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r0213;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r0213;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r0213;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r0213;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r0213;
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==2 && dst_order::at(2)==3 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r0231;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r0231;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r0231;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r0231;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r0231;      
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==3 && dst_order::at(2)==1 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r0312;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r0312;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r0312;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r0312;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r0312;          
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==3 && dst_order::at(2)==2 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r0321;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r0321;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r0321;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r0321;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r0321; 
        }
        //(1,...)
        else if(dst_order::at(0)==1 && dst_order::at(1)==0 && dst_order::at(2)==2 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r1023;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r1023;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r1023;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r1023;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r1023;      
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==0 && dst_order::at(2)==3 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r1032;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r1032;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r1032;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r1032;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r1032;           
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==2 && dst_order::at(2)==0 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r1203;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r1203;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r1203;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r1203;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r1203;  
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==2 && dst_order::at(2)==3 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r1230;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r1230;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r1230;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r1230;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r1230;        
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==3 && dst_order::at(2)==0 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r1302;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r1302;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r1302;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r1302;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r1302;          
        }
        else if(dst_order::at(0)==1 && dst_order::at(1)==3 && dst_order::at(2)==2 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r1320;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r1320;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r1320;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r1320;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r1320; 
        }
        //(2,...)
        else if(dst_order::at(0)==2 && dst_order::at(1)==0 && dst_order::at(2)==1 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r2013;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r2013;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r2013;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r2013;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r2013;       
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==0 && dst_order::at(2)==3 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r2031;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r2031;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r2031;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r2031;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r2031;          
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==1 && dst_order::at(2)==0 && dst_order::at(3)==3){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r2103;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r2103;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r2103;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r2103;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r2103;
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==1 && dst_order::at(2)==3 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r2130;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r2130;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r2130;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r2130;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r2130;      
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==3 && dst_order::at(2)==0 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r2301;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r2301;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r2301;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r2301;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r2301;          
        }
        else if(dst_order::at(0)==2 && dst_order::at(1)==3 && dst_order::at(2)==1 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r2310;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r2310;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r2310;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r2310;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r2310;  
        }
        //(3,...)
        else if(dst_order::at(0)==3 && dst_order::at(1)==0 && dst_order::at(2)==1 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r3012;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r3012;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r3012;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r3012;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r3012;      
        }
        else if(dst_order::at(0)==3 && dst_order::at(1)==0 && dst_order::at(2)==2 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r3021;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r3021;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r3021;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r3021;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r3021;           
        }
        else if(dst_order::at(0)==3 && dst_order::at(1)==1 && dst_order::at(2)==0 && dst_order::at(3)==2){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r3102;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r3102;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r3102;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r3102;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r3102; 
        }
        else if(dst_order::at(0)==3 && dst_order::at(1)==1 && dst_order::at(2)==2 && dst_order::at(3)==0){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r3120;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r3120;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r3120;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r3120;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r3120;       
        }
        else if(dst_order::at(0)==3 && dst_order::at(1)==2 && dst_order::at(2)==0 && dst_order::at(3)==1){
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r3201;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r3201;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r3201;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r3201;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r3201;           
        }
        //else if(dst_order::at(0)==3 && dst_order::at(1)==2 && dst_order::at(2)==1 && dst_order::at(3)==0){
        else{
            if(kparam->tile_x == 1)        return the_reorder_gpu_handle.kernel_general_4d_reorder_1p_byte_r3210;
            else if(kparam->tile_x == 2)   return the_reorder_gpu_handle.kernel_general_4d_reorder_2p_byte_r3210;
            else if(kparam->tile_x == 4)   return the_reorder_gpu_handle.kernel_general_4d_reorder_4p_byte_r3210;
            else if(kparam->tile_x == 8)   return the_reorder_gpu_handle.kernel_general_4d_reorder_8p_byte_r3210;
            else if(kparam->tile_x ==16)  return the_reorder_gpu_handle.kernel_general_4d_reorder_16p_byte_r3210;          
        } 
    }
};

template<typename T, typename dst_order>
void gpu_general_tensor_reorder(T * dst, T * src, uint32_t batch, uint32_t channel, uint32_t height, uint32_t width, const transpose_kernel_param_t * kparam)
{
    hipDeviceProp_t dev_prop;
    hipDevice_t dev;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    // TODO: need find better way to decide transpose tile size
    uint32_t pixel_total =  batch * channel * height * width;
    size_t block_size = 256;
    uint32_t dim_total = (pixel_total+ block_size * kparam->tile_x -1) / (block_size * kparam->tile_x);

#if BATCHED_TRANSPOSE_PERSISTENT
    int num_cu = dev_prop.multiProcessorCount;
    const int occupancy = 4;
    size_t grid_size = num_cu * occupancy;
#else
    size_t grid_size = dim_total;
#endif

    magic_div_u32_t magic_stride0 = magic_div_u32_gen(channel * height * width);
    magic_div_u32_t magic_stride1 = magic_div_u32_gen(height * width);
    magic_div_u32_t magic_stride2 = magic_div_u32_gen(width);
    //loop over
    hipFunction_t kernel = reorder_kernel_select_t<sizeof(T), dst_order>::get(kparam);

    reorder_kernel_t karg;
    karg.p_dst          = reinterpret_cast<void*>(dst);
    karg.p_src          = reinterpret_cast<void*>(src);
    karg.dim_0          = batch;
    karg.dim_1          = channel;
    karg.dim_2          = height;
    karg.dim_3          = width;
    karg.dim_stride     = grid_size;
    karg.dim_total      = dim_total;
    karg.magic_stride0  = magic_stride0.magic;
    karg.shift_stride0  = magic_stride0.shift;
    karg.magic_stride1  = magic_stride1.magic;
    karg.shift_stride1  = magic_stride1.shift;
    karg.magic_stride2  = magic_stride2.magic;
    karg.shift_stride2  = magic_stride2.shift;
    size_t karg_size    = sizeof(karg);


    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    HIP_CALL(hipExtModuleLaunchKernel(kernel, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));
}
#endif