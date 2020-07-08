/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef __IGEMM_V4R1_DYNAMIC_DRIVER_H
#define __IGEMM_V4R1_DYNAMIC_DRIVER_H

#include "config_parser.h"
#include <string>
#include <unistd.h>
#include <vector>
#include <math.h>

#define CONV_FWD 1
#define CONV_WRW 4

typedef struct {
    int b_per_block;
    int k_per_block;
    int e_per_block;

    int gemm_n_repeat;

    int gemm_m_per_thread_subc;
    int gemm_n_per_thread_subc;

    int gemm_m_level0_cluster;
    int gemm_n_level0_cluster;
    int gemm_m_level1_cluster;
    int gemm_n_level1_cluster;

    int in_block_copy_cluster_lengths_e;
    int in_block_copy_cluster_lengths_n1;
    int in_block_copy_cluster_lengths_b;
    int in_block_copy_cluster_lengths_n2;

    int wei_block_copy_cluster_lengths_e;
    int wei_block_copy_cluster_lengths_k;

    int OPT_1x1;
} igemm_v4r1_dynamic_tunable_t;

static inline std::vector<igemm_v4r1_dynamic_tunable_t>
igemm_v4r1_dynamic_tunable_from_config(const config_content_t &content) {
    std::vector<igemm_v4r1_dynamic_tunable_t> tunables;
    for (const auto &sec : content) {
        if (sec.get_name() == "v4r1_dynamic_kernel") {
            igemm_v4r1_dynamic_tunable_t tunable;
            tunable.b_per_block = sec.at("b_per_block").get_int();
            tunable.k_per_block = sec.at("k_per_block").get_int();
            tunable.e_per_block = sec.at("e_per_block").get_int();
            tunable.gemm_n_repeat = sec.at("gemm_n_repeat").get_int();
            tunable.gemm_m_per_thread_subc =
                sec.at("gemm_m_per_thread_subc").get_int();
            tunable.gemm_n_per_thread_subc =
                sec.at("gemm_n_per_thread_subc").get_int();
            tunable.gemm_m_level0_cluster =
                sec.at("gemm_m_level0_cluster").get_int();
            tunable.gemm_n_level0_cluster =
                sec.at("gemm_n_level0_cluster").get_int();
            tunable.gemm_m_level1_cluster =
                sec.at("gemm_m_level1_cluster").get_int();
            tunable.gemm_n_level1_cluster =
                sec.at("gemm_n_level1_cluster").get_int();
            tunable.in_block_copy_cluster_lengths_e =
                sec.at("in_block_copy_cluster_lengths_e").get_int();
            tunable.in_block_copy_cluster_lengths_n1 =
                sec.at("in_block_copy_cluster_lengths_n1").get_int();
            tunable.in_block_copy_cluster_lengths_b =
                sec.at("in_block_copy_cluster_lengths_b").get_int();
            tunable.in_block_copy_cluster_lengths_n2 =
                sec.at("in_block_copy_cluster_lengths_n2").get_int();
            tunable.wei_block_copy_cluster_lengths_e =
                sec.at("wei_block_copy_cluster_lengths_e").get_int();
            tunable.wei_block_copy_cluster_lengths_k =
                sec.at("wei_block_copy_cluster_lengths_k").get_int();
            tunable.OPT_1x1 = 0;
            tunables.push_back(tunable);
        }
        else if (sec.get_name() == "v4r1_1x1_dynamic_kernel") {
            igemm_v4r1_dynamic_tunable_t tunable;
            tunable.b_per_block = sec.at("b_per_block").get_int();
            tunable.k_per_block = sec.at("k_per_block").get_int();
            tunable.e_per_block = sec.at("e_per_block").get_int();
            tunable.gemm_n_repeat = sec.at("gemm_n_repeat").get_int();
            tunable.gemm_m_per_thread_subc =
                sec.at("gemm_m_per_thread_subc").get_int();
            tunable.gemm_n_per_thread_subc =
                sec.at("gemm_n_per_thread_subc").get_int();
            tunable.gemm_m_level0_cluster =
                sec.at("gemm_m_level0_cluster").get_int();
            tunable.gemm_n_level0_cluster =
                sec.at("gemm_n_level0_cluster").get_int();
            tunable.gemm_m_level1_cluster =
                sec.at("gemm_m_level1_cluster").get_int();
            tunable.gemm_n_level1_cluster =
                sec.at("gemm_n_level1_cluster").get_int();
            tunable.in_block_copy_cluster_lengths_e =
                sec.at("in_block_copy_cluster_lengths_e").get_int();
            tunable.in_block_copy_cluster_lengths_n1 =
                sec.at("in_block_copy_cluster_lengths_n1").get_int();
            tunable.in_block_copy_cluster_lengths_b =
                sec.at("in_block_copy_cluster_lengths_b").get_int();
            tunable.in_block_copy_cluster_lengths_n2 =
                sec.at("in_block_copy_cluster_lengths_n2").get_int();
            tunable.wei_block_copy_cluster_lengths_e =
                sec.at("wei_block_copy_cluster_lengths_e").get_int();
            tunable.wei_block_copy_cluster_lengths_k =
                sec.at("wei_block_copy_cluster_lengths_k").get_int();
            tunable.OPT_1x1 = 1;
            tunables.push_back(tunable);
        }
    }
    return tunables;
}

typedef struct {
    float *p_in;
    float *p_wei;
    float *p_out;
    int hi;
    int wi;
    int n;
    int k;
    int c;
    int ho;
    int wo;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int pad_h;
    int pad_w;
    int y;
    int x;
    int gemmk_groups;
} __attribute__((packed)) igemm_v4r1_dynamic_karg_t;

typedef struct {
    float* output;
    float* input;
    int out_length;
    int in_stride;
    int n_groups;
    int __pack_0;
} __attribute__((packed)) reduction_karg_t;

#define VALID_COND_RTN_FALSE(cond)                                             \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("Valid Faild! %s\n", #cond);                                \
            return false;                                                      \
        }                                                                      \
    } while (0)

class igemm_v4r1_dynamic_driver_t {
  public:
    igemm_v4r1_dynamic_driver_t() {}
    ~igemm_v4r1_dynamic_driver_t() {}
    std::string get_kernel_name(const igemm_v4r1_dynamic_tunable_t *tunable) {
        int b_per_block = tunable->b_per_block;
        int k_per_block = tunable->k_per_block;
        int e_per_block = tunable->e_per_block;
        int gemm_n_repeat = tunable->gemm_n_repeat;
        int gemm_m_per_thread_subc = tunable->gemm_m_per_thread_subc;
        int gemm_n_per_thread_subc = tunable->gemm_n_per_thread_subc;
        int gemm_m_level1_cluster = tunable->gemm_m_level1_cluster;
        int gemm_n_level1_cluster = tunable->gemm_n_level1_cluster;
        int gemm_m_level0_cluster = tunable->gemm_m_level0_cluster;
        int gemm_n_level0_cluster = tunable->gemm_n_level0_cluster;
        int in_block_copy_cluster_lengths_e =
            tunable->in_block_copy_cluster_lengths_e;
        int in_block_copy_cluster_lengths_n1 =
            tunable->in_block_copy_cluster_lengths_n1;
        int in_block_copy_cluster_lengths_b =
            tunable->in_block_copy_cluster_lengths_b;
        int in_block_copy_cluster_lengths_n2 =
            tunable->in_block_copy_cluster_lengths_n2;
        int wei_block_copy_cluster_lengths_e =
            tunable->wei_block_copy_cluster_lengths_e;
        int wei_block_copy_cluster_lengths_k =
            tunable->wei_block_copy_cluster_lengths_k;

        assert(k_per_block % (gemm_m_per_thread_subc * gemm_m_level0_cluster *
                              gemm_m_level1_cluster) ==
               0);
        int gemm_m_repeat =
            k_per_block / (gemm_m_per_thread_subc * gemm_m_level0_cluster *
                           gemm_m_level1_cluster);
        int thread_tile_m = gemm_m_repeat * gemm_m_per_thread_subc;
        int thread_tile_n = gemm_n_repeat * gemm_n_per_thread_subc;

        // std::string kernel_prefix = tunable->OPT_1x1 ? std::string("igemm_v4r1_1x1_dynamic_") : std::string("igemm_v4r1_dynamic_");
        std::string kernel_prefix = std::string("igemm_v4r1_dynamic_wrw_");

        return kernel_prefix +
               std::to_string(k_per_block) + "x" +
               std::to_string(b_per_block * gemm_n_repeat *
                              gemm_n_per_thread_subc) +
               "x" + std::to_string(e_per_block) + "_" +
               std::to_string(thread_tile_m) + "x" +
               std::to_string(thread_tile_n) + "_" +
               std::to_string(gemm_m_per_thread_subc) + "x" +
               std::to_string(gemm_m_level0_cluster) + "x" +
               std::to_string(gemm_m_level1_cluster) + "x" +
               std::to_string(gemm_n_per_thread_subc) + "x" +
               std::to_string(gemm_n_level0_cluster) + "x" +
               std::to_string(gemm_n_level1_cluster) + "_" +
               std::to_string(in_block_copy_cluster_lengths_e) + "x" +
               std::to_string(in_block_copy_cluster_lengths_n1) + "x" +
               std::to_string(in_block_copy_cluster_lengths_b) + "x" +
               std::to_string(in_block_copy_cluster_lengths_n2) + "_" +
               std::to_string(wei_block_copy_cluster_lengths_e) + "x" +
               std::to_string(wei_block_copy_cluster_lengths_k);
    }
    int get_block_size(const igemm_v4r1_dynamic_tunable_t *tunable) {
        return tunable->gemm_m_level0_cluster * tunable->gemm_n_level0_cluster *
               tunable->gemm_m_level1_cluster * tunable->gemm_n_level1_cluster;
    }
    int get_grid_size(const args_t *arg,
                      const igemm_v4r1_dynamic_tunable_t *tunable) {
        int hi = arg->get_int("in_h");
        int wi = arg->get_int("in_w");
        int n = arg->get_int("batchsize");
        int k = arg->get_int("out_channels");
        // int c          = arg->get_int("in_channels");

        int stride_h = arg->get_int("conv_stride_h");
        int stride_w = arg->get_int("conv_stride_w");
        int dilation_h = arg->get_int("dilation_h");
        int dilation_w = arg->get_int("dilation_w");
        int pad_h = arg->get_int("pad_h");
        int pad_w = arg->get_int("pad_w");
        int y = arg->get_int("fil_h");
        int x = arg->get_int("fil_w");
        int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
        int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

        int b_per_block = tunable->b_per_block;
        int k_per_block = tunable->k_per_block;

        int gemm_n_repeat = tunable->gemm_n_repeat;
        int gemm_n_per_thread_subc = tunable->gemm_n_per_thread_subc;

        int n1 = gemm_n_repeat;
        int n2 = gemm_n_per_thread_subc;

        int n0 = n / (n1 * n2);

        int b = n0 * ho * wo;

        int grid_size = (b / b_per_block) * (k / k_per_block);
        return grid_size;
    }
    int get_grid_size_wrw(const igemm_v4r1_dynamic_karg_t *karg,
                          const igemm_v4r1_dynamic_tunable_t *tunable) {
        int b_per_block = tunable->b_per_block;
        int k_per_block = tunable->k_per_block;

        int gemm_n_repeat = tunable->gemm_n_repeat;
        int gemm_n_per_thread_subc = tunable->gemm_n_per_thread_subc;

        int n1 = gemm_n_repeat;
        int n2 = gemm_n_per_thread_subc;

        int n0 = karg->n / (n1 * n2);

        int b = n0 * karg->ho * karg->wo;

        int grid_size = (b / b_per_block) * (karg->k / k_per_block);
        return grid_size;
    }
    int get_lds_size(const igemm_v4r1_dynamic_tunable_t *tunable) {
        // TODO: fp16/bf16, xdlops
        int lds_a = 4 * tunable->e_per_block * tunable->k_per_block;
        int lds_b = 4 * tunable->e_per_block * tunable->gemm_n_repeat *
                    tunable->b_per_block * tunable->gemm_n_per_thread_subc;
        return 2 * next_pow2(next_pow2(lds_a) + next_pow2(lds_b));
    }
    bool tunable_is_valid(const args_t *arg,
                          const igemm_v4r1_dynamic_tunable_t *tunable) {
        // PerformanceImplicitGemmV4R1::IsValid
        int hi = arg->get_int("in_h");
        int wi = arg->get_int("in_w");
        int n = arg->get_int("batchsize");
        int k = arg->get_int("out_channels");
        int c = arg->get_int("in_channels");

        int stride_h = arg->get_int("conv_stride_h");
        int stride_w = arg->get_int("conv_stride_w");
        int dilation_h = arg->get_int("dilation_h");
        int dilation_w = arg->get_int("dilation_w");
        int pad_h = arg->get_int("pad_h");
        int pad_w = arg->get_int("pad_w");
        int y = arg->get_int("fil_h");
        int x = arg->get_int("fil_w");
        int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
        int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

        int b_per_block = tunable->b_per_block;
        int k_per_block = tunable->k_per_block;
        int e_per_block = tunable->e_per_block;
        int gemm_n_repeat = tunable->gemm_n_repeat;
        int gemm_m_per_thread_subc = tunable->gemm_m_per_thread_subc;
        int gemm_n_per_thread_subc = tunable->gemm_n_per_thread_subc;
        int gemm_m_level1_cluster = tunable->gemm_m_level1_cluster;
        int gemm_n_level1_cluster = tunable->gemm_n_level1_cluster;
        int gemm_m_level0_cluster = tunable->gemm_m_level0_cluster;
        int gemm_n_level0_cluster = tunable->gemm_n_level0_cluster;
        int in_block_copy_cluster_lengths_e =
            tunable->in_block_copy_cluster_lengths_e;
        int in_block_copy_cluster_lengths_n1 =
            tunable->in_block_copy_cluster_lengths_n1;
        int in_block_copy_cluster_lengths_b =
            tunable->in_block_copy_cluster_lengths_b;
        int in_block_copy_cluster_lengths_n2 =
            tunable->in_block_copy_cluster_lengths_n2;
        int wei_block_copy_cluster_lengths_e =
            tunable->wei_block_copy_cluster_lengths_e;
        int wei_block_copy_cluster_lengths_k =
            tunable->wei_block_copy_cluster_lengths_k;

        int gemm_m_repeat =
            k_per_block / (gemm_m_per_thread_subc * gemm_m_level0_cluster *
                           gemm_m_level1_cluster);

        int n1 = gemm_n_repeat;
        int n2 = gemm_n_per_thread_subc;
        VALID_COND_RTN_FALSE((n % (n1 * n2) == 0));

        int n0 = n / (n1 * n2);

        int b = n0 * ho * wo;

        // TODO: non vector size
        int e = c * y * x;

        // printf("e_per_block:%d, b_per_block:%d, k_per_block:%d,
        // in_e_n1_b_n2:{%d,%d,%d,%d}, wei_e_k:{%d,%d}, n1:%d, n2:%d\n",
        //    e_per_block, b_per_block, k_per_block,
        //    in_block_copy_cluster_lengths_e, in_block_copy_cluster_lengths_n1,
        //    in_block_copy_cluster_lengths_b, in_block_copy_cluster_lengths_n2,
        //    wei_block_copy_cluster_lengths_e,
        //    wei_block_copy_cluster_lengths_k,
        //    n1, n2);

        VALID_COND_RTN_FALSE(
            (e_per_block % in_block_copy_cluster_lengths_e == 0 &&
             e_per_block % wei_block_copy_cluster_lengths_e == 0 &&
             b_per_block % in_block_copy_cluster_lengths_b == 0 &&
             k_per_block % wei_block_copy_cluster_lengths_k == 0 &&
             n1 % in_block_copy_cluster_lengths_n1 == 0 &&
             n2 % in_block_copy_cluster_lengths_n2 == 0));

        // divide block work by [K, B]
        VALID_COND_RTN_FALSE(k % k_per_block == 0 && b % b_per_block == 0 &&
                             e % e_per_block == 0);

        // const auto KBlockWork = K / k_per_block;
        // if(KBlockWork % ctx.group_counts != 0)
        //    return false;

        VALID_COND_RTN_FALSE((n1 * n2 * b_per_block) %
                                 (gemm_n_per_thread_subc *
                                  gemm_n_level0_cluster *
                                  gemm_n_level1_cluster) ==
                             0);

        // fp16/bf16 check
        VALID_COND_RTN_FALSE(
            (k_per_block % (gemm_m_per_thread_subc * gemm_m_level0_cluster *
                            gemm_m_level1_cluster)) == 0);

        VALID_COND_RTN_FALSE(gemm_n_repeat ==
                             (n1 * n2 * b_per_block) / (gemm_n_per_thread_subc *
                                                        gemm_n_level0_cluster *
                                                        gemm_n_level1_cluster));

        int block_size = get_block_size(tunable);
        VALID_COND_RTN_FALSE(block_size >= 64 && block_size <= 512);

        VALID_COND_RTN_FALSE(block_size ==
                             in_block_copy_cluster_lengths_e *
                                 in_block_copy_cluster_lengths_n1 *
                                 in_block_copy_cluster_lengths_b *
                                 in_block_copy_cluster_lengths_n2);

        VALID_COND_RTN_FALSE(block_size ==
                             wei_block_copy_cluster_lengths_e *
                                 wei_block_copy_cluster_lengths_k);

        VALID_COND_RTN_FALSE(gemm_m_repeat == 2 && gemm_n_repeat == 2);

        int lds_size = get_lds_size(tunable);
        VALID_COND_RTN_FALSE(lds_size <= 65536);
        int in_block_copy_sub_lengths_e =
            e_per_block / in_block_copy_cluster_lengths_e;
        int in_block_copy_sub_lengths_b =
            b_per_block / in_block_copy_cluster_lengths_b;
        VALID_COND_RTN_FALSE(in_block_copy_sub_lengths_e == 1 &&
                             in_block_copy_sub_lengths_b == 1);
        return true;
    }

    void host_wrw_reduction(float* out, float* input, int length, int n_groups){
        int i_len, i_group;
        float val_out = 0;
        std::cout << "vec_length: " << length << std::endl;
        for (i_len = 0; i_len < length; i_len++){
            val_out = 0;
            for (i_group = 0; i_group < n_groups; i_group++){
                val_out += input[i_len + i_group * length];
            }
            out[i_len] = val_out;
        }
    }

    result_t run(const args_t *arg, const igemm_v4r1_dynamic_tunable_t *tunable,
                 hipModule_t module, hipModule_t module_reduction, float *p_in, float *p_wei, float *p_out,
                 int warmup, int repeat, int dir) {

        if (!tunable_is_valid(arg, tunable)) {
            result_t result;
            result.return_code = -1;
            return result;
        }
        
        if (CONV_FWD == dir) {
            igemm_v4r1_dynamic_karg_t karg;
            size_t karg_size = sizeof(karg);
            karg.p_in = p_in;
            karg.p_wei = p_wei;
            karg.p_out = p_out;
            karg.hi = arg->get_int("in_h");
            karg.wi = arg->get_int("in_w");
            karg.n = arg->get_int("batchsize");
            karg.k = arg->get_int("out_channels");
            karg.c = arg->get_int("in_channels");

            karg.stride_h = arg->get_int("conv_stride_h");
            karg.stride_w = arg->get_int("conv_stride_w");
            karg.dilation_h = arg->get_int("dilation_h");
            karg.dilation_w = arg->get_int("dilation_w");
            karg.pad_h = arg->get_int("pad_h");
            karg.pad_w = arg->get_int("pad_w");
            karg.y = arg->get_int("fil_h");
            karg.x = arg->get_int("fil_w");

            karg.ho = conv_out_size(karg.hi, karg.pad_h, karg.dilation_h, karg.y,
                                    karg.stride_h);
            karg.wo = conv_out_size(karg.wi, karg.pad_w, karg.dilation_w, karg.x,
                                    karg.stride_w);

            void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                            HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                            HIP_LAUNCH_PARAM_END};

            int block_size = get_block_size(tunable);
            int grid_size = get_grid_size(arg, tunable);

            hipFunction_t kernel_func;
            std::string kernel_name = get_kernel_name(tunable);
            //printf("kernel:%s\n", kernel_name.c_str());
            HIP_CALL(
                hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));
            gpu_timer_t timer(NULL);
            for (int i = 0; i < warmup; i++) {
                HIP_CALL(hipModuleLaunchKernel(kernel_func, grid_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config));
            }
            timer.start();
            for (int i = 0; i < repeat; i++) {
                HIP_CALL(hipModuleLaunchKernel(kernel_func, grid_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config));
            }
            timer.stop();
            float duration_ms = timer.duration();

            usleep(1000 * 10);

            result_t result;
            result.return_code = 0;
            result.duration_ms = duration_ms / repeat;
            result.kernel_name = kernel_name;
            return result;
        }

        else if (CONV_WRW == dir) {
            igemm_v4r1_dynamic_karg_t karg;
            size_t karg_size = sizeof(karg);
            karg.p_in = p_in;
            karg.p_wei = p_out;
            karg.p_out = p_wei;
            karg.hi = arg->get_int("in_h");
            karg.wi = arg->get_int("in_w");
            karg.n = arg->get_int("in_channels");
            karg.k = arg->get_int("out_channels");
            karg.c = arg->get_int("batchsize");

            int stride_h = arg->get_int("conv_stride_h");
            int stride_w = arg->get_int("conv_stride_w");
            int dilation_h = arg->get_int("dilation_h");
            int dilation_w = arg->get_int("dilation_w");
            karg.pad_h = arg->get_int("pad_h");
            karg.pad_w = arg->get_int("pad_w");
            int y = arg->get_int("fil_h");
            int x = arg->get_int("fil_w");

            int ho = conv_out_size(karg.hi, karg.pad_h, dilation_h, y,
                                   stride_h);
            int wo = conv_out_size(karg.wi, karg.pad_w, dilation_w, x,
                                   stride_w);

            karg.y = ho;
            karg.x = wo;

            karg.ho = y;
            karg.wo = x;

            karg.dilation_h = stride_h;
            karg.dilation_w = stride_w;
            karg.stride_h = dilation_h;
            karg.stride_w = dilation_w;


            void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                            HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                            HIP_LAUNCH_PARAM_END};

            int block_size = get_block_size(tunable);
            int grid_size = get_grid_size_wrw(&karg, tunable);

            // groups to reduction
            int gemmk_groups = 32;
            grid_size *= gemmk_groups;

            karg.gemmk_groups = (int)(log2f(gemmk_groups));

            // extra workspace
            float* gemmc_workspace = NULL;
            float* gemmc_host_check = (float* )malloc(gemmk_groups * karg.n * karg.k * y * x * sizeof(float));
            float* gemmc_host_reduction = (float* )malloc(gemmk_groups * karg.n * karg.k * y * x * sizeof(float));
            hipError_t err = hipMalloc(&gemmc_workspace, gemmk_groups * karg.n * karg.k * y * x * sizeof(float));
            if (err != hipSuccess) {                                               
                printf("[hiperror](%d) fail to malloc workspace,(%s)", (int)err,     
                       hipGetErrorString(err));                                    
                exit(1);                                                           
            }

            karg.p_out = gemmc_workspace;

            // reduction kernel args
            size_t reduction_per_thread = 8;
            reduction_karg_t karg_reduction;
            karg_reduction.output = p_wei;
            karg_reduction.input = gemmc_workspace; 
            karg_reduction.in_stride = karg.n * karg.k * y * x;
            karg_reduction.out_length = reduction_per_thread;
            karg_reduction.n_groups = gemmk_groups;

            size_t karg_reduction_size = sizeof(karg_reduction);

            void *config_reduction[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg_reduction,
                            HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_reduction_size,
                            HIP_LAUNCH_PARAM_END};

            printf("\r\ngrid_size is: %d\r\n", grid_size);
            printf("block_size is: %d\r\n", block_size);

            printf("kernel args cyx=[%d, %d, %d]\r\n", karg.c, karg.y, karg.x);
            printf("kernel args nkhowo=[%d, %d, %d, %d]\r\n", karg.n, karg.k, karg.ho, karg.wo);

            hipFunction_t kernel_func;
            hipFunction_t reduction_func;

            std::string kernel_name = get_kernel_name(tunable);
            //printf("kernel:%s\n", kernel_name.c_str());
            HIP_CALL(
                hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));
            HIP_CALL(
                hipModuleGetFunction(&reduction_func, module_reduction, "wrw_reduction"));
            gpu_timer_t timer(NULL);
            for (int i = 0; i < warmup; i++) {
                HIP_CALL(hipModuleLaunchKernel(kernel_func, grid_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config));
                HIP_CALL(hipModuleLaunchKernel(reduction_func, karg.n * karg.k * y * x / (reduction_per_thread * 256), 1, 1,
                                            256, 1, 1, 0, 0, NULL,
                                            (void **)&config_reduction));
            }
            timer.start();
            for (int i = 0; i < repeat; i++) {
                HIP_CALL(hipModuleLaunchKernel(kernel_func, grid_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config));
                //HIP_CALL(hipModuleLaunchKernel(reduction_func, karg.n * karg.k * y * x / (reduction_per_thread * 256), 1, 1,
                //                            256, 1, 1, 0, 0, NULL,
                //                            (void **)&config_reduction));
            }
            timer.stop();
            float duration_ms = timer.duration();

            // gridwise reduction

            usleep(1000 * 1);

            // debug section of code
            printf("workspace debug \r\n");
            hipMemcpy(gemmc_host_check, gemmc_workspace, gemmk_groups * karg.n * karg.k * y * x * sizeof(float), hipMemcpyDeviceToHost);
            for (int i_check = 0; i_check < (0+8); i_check++)
            {
                printf("[%d]th var to monitor:[%f, %d]\r\n", i_check, gemmc_host_check[i_check], ((int *)gemmc_host_check)[i_check]);
            }
            printf("workspace debug end \r\n");

            // host reduction to check group conv's correctness
            //host_wrw_reduction(gemmc_host_reduction, gemmc_host_check, karg.n * karg.k * y * x, gemmk_groups);
            //hipMemcpy(p_wei, gemmc_host_reduction, karg.n * karg.k * y * x * sizeof(float), hipMemcpyHostToDevice);

            //hipMemcpy(p_wei, gemmc_workspace, gemmk_groups * karg.n * karg.k * y * x * sizeof(float), hipMemcpyDeviceToDevice);

            hipFree(gemmc_workspace);
            free(gemmc_host_check);
            free(gemmc_host_reduction);

            result_t result;
            result.return_code = 0;
            result.duration_ms = duration_ms / repeat;
            result.kernel_name = kernel_name;
            return result;
        }
        else {
            printf("not supported direction\r\n");
            result_t result;
            result.return_code = -1;
            return result;
        }
    }
};

#endif