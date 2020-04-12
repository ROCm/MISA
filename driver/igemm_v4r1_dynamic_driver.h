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
#ifndef __IGEMM_V4R1_DYNAMIC_DRIVER_H
#define __IGEMM_V4R1_DYNAMIC_DRIVER_H

#include <string>
#include <unistd.h>

typedef struct {
    int BPerBlock;
    int KPerBlock;
    int EPerBlock;

    int GemmNRepeat;

    int GemmMPerThreadSubC;
    int GemmNPerThreadSubC;

    int GemmMLevel0Cluster;
    int GemmNLevel0Cluster;
    int GemmMLevel1Cluster;
    int GemmNLevel1Cluster;

    int InBlockCopyClusterLengths_E;
    int InBlockCopyClusterLengths_N1;
    int InBlockCopyClusterLengths_B;
    int InBlockCopyClusterLengths_N2;

    int WeiBlockCopyClusterLengths_E;
    int WeiBlockCopyClusterLengths_K;

    int OPT_1x1;
} igemm_v4r1_dynamic_tunable_t;

// clang-format off
static igemm_v4r1_dynamic_tunable_t igemm_v4r1_dynamic_tunables[] = {
    //bpb  kpb epb nrep mptc nptc ml0c nl0c ml1c nl1c in_e  n1   b  n2 weie   k  1x1
    { 16, 128, 16,   2,   4,   4,   4,   4,   4,   4,  16,  1,  16,  1,  4,  64,  0},
    //{ 16, 128, 16,   2,   4,   4,   4,   4,   4,   4,  16,  1,  16,  1,  4,  64,  1},
    { 16, 128,  8,   2,   4,   4,   4,   4,   4,   4,   8,  2,  16,  1,  2, 128,  0},
    //{ 16, 128,  8,   2,   4,   4,   4,   4,   4,   4,   8,  2,  16,  1,  2, 128,  1},
    {  8, 128,  8,   2,   4,   4,   4,   4,   4,   2,   8,  1,   8,  2,  2,  64,  0},
    //{  8, 128,  8,   2,   4,   4,   4,   4,   4,   2,   8,  1,   8,  2,  2,  64,  1},
    {  8,  64,  8,   2,   4,   4,   4,   2,   2,   4,   8,  1,   8,  1,  4,  16,  0},
    //{  8,  64,  8,   2,   4,   4,   4,   2,   2,   4,   8,  1,   8,  1,  4,  16,  1},
    { 16,  32,  4,   2,   4,   4,   1,   4,   4,   4,   4,  1,  16,  1,  4,  16,  0},
    //{ 16,  32,  4,   2,   4,   4,   1,   4,   4,   4,   4,  1,  16,  1,  4,  16,  1},
    { 16,  16,  4,   2,   2,   2,   2,   4,   2,   4,   4,  1,  16,  1,  4,  16,  0},
    //{ 16,  16,  4,   2,   2,   2,   2,   4,   2,   4,   4,  1,  16,  1,  4,  16,  1},
    {  8,  32,  4,   2,   2,   2,   2,   4,   4,   2,   4,  2,   8,  1,  4,  16,  0},
    //{  8,  32,  4,   2,   2,   2,   2,   4,   4,   2,   4,  2,   8,  1,  4,  16,  1},
};
// clang-format on

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
    int __pack0;
} __attribute__((packed)) igemm_v4r1_dynamic_karg_t;

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
        int b_per_block = tunable->BPerBlock;
        int k_per_block = tunable->KPerBlock;
        int e_per_block = tunable->EPerBlock;
        int gemm_n_repeat = tunable->GemmNRepeat;
        int gemm_m_per_thread_subc = tunable->GemmMPerThreadSubC;
        int gemm_n_per_thread_subc = tunable->GemmNPerThreadSubC;
        int gemm_m_level1_cluster = tunable->GemmMLevel1Cluster;
        int gemm_n_level1_cluster = tunable->GemmNLevel1Cluster;
        int gemm_m_level0_cluster = tunable->GemmMLevel0Cluster;
        int gemm_n_level0_cluster = tunable->GemmNLevel0Cluster;
        int in_block_copy_cluster_lengths_e =
            tunable->InBlockCopyClusterLengths_E;
        int in_block_copy_cluster_lengths_n1 =
            tunable->InBlockCopyClusterLengths_N1;
        int in_block_copy_cluster_lengths_b =
            tunable->InBlockCopyClusterLengths_B;
        int in_block_copy_cluster_lengths_n2 =
            tunable->InBlockCopyClusterLengths_N2;
        int wei_block_copy_cluster_lengths_e =
            tunable->WeiBlockCopyClusterLengths_E;
        int wei_block_copy_cluster_lengths_k =
            tunable->WeiBlockCopyClusterLengths_K;

        assert(k_per_block % (gemm_m_per_thread_subc * gemm_m_level0_cluster *
                              gemm_m_level1_cluster) ==
               0);
        int gemm_m_repeat =
            k_per_block / (gemm_m_per_thread_subc * gemm_m_level0_cluster *
                           gemm_m_level1_cluster);
        int thread_tile_m = gemm_m_repeat * gemm_m_per_thread_subc;
        int thread_tile_n = gemm_n_repeat * gemm_n_per_thread_subc;

        return std::string("igemm_v4r1_dynamic_") +
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
        return tunable->GemmMLevel0Cluster * tunable->GemmNLevel0Cluster *
               tunable->GemmMLevel1Cluster * tunable->GemmNLevel1Cluster;
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

        int b_per_block = tunable->BPerBlock;
        int k_per_block = tunable->KPerBlock;

        int gemm_n_repeat = tunable->GemmNRepeat;
        int gemm_n_per_thread_subc = tunable->GemmNPerThreadSubC;

        int n1 = gemm_n_repeat;
        int n2 = gemm_n_per_thread_subc;

        int n0 = n / (n1 * n2);

        int b = n0 * ho * wo;

        int grid_size = (b / b_per_block) * (k / k_per_block);
        return grid_size;
    }
    int get_lds_size(const igemm_v4r1_dynamic_tunable_t *tunable) {
        // TODO: fp16/bf16, xdlops
        int lds_a = 4 * tunable->EPerBlock * tunable->KPerBlock;
        int lds_b = 4 * tunable->EPerBlock * tunable->GemmNRepeat *
                    tunable->BPerBlock * tunable->GemmNPerThreadSubC;
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

        int b_per_block = tunable->BPerBlock;
        int k_per_block = tunable->KPerBlock;
        int e_per_block = tunable->EPerBlock;
        int gemm_n_repeat = tunable->GemmNRepeat;
        int gemm_m_per_thread_subc = tunable->GemmMPerThreadSubC;
        int gemm_n_per_thread_subc = tunable->GemmNPerThreadSubC;
        int gemm_m_level1_cluster = tunable->GemmMLevel1Cluster;
        int gemm_n_level1_cluster = tunable->GemmNLevel1Cluster;
        int gemm_m_level0_cluster = tunable->GemmMLevel0Cluster;
        int gemm_n_level0_cluster = tunable->GemmNLevel0Cluster;
        int in_block_copy_cluster_lengths_e =
            tunable->InBlockCopyClusterLengths_E;
        int in_block_copy_cluster_lengths_n1 =
            tunable->InBlockCopyClusterLengths_N1;
        int in_block_copy_cluster_lengths_b =
            tunable->InBlockCopyClusterLengths_B;
        int in_block_copy_cluster_lengths_n2 =
            tunable->InBlockCopyClusterLengths_N2;
        int wei_block_copy_cluster_lengths_e =
            tunable->WeiBlockCopyClusterLengths_E;
        int wei_block_copy_cluster_lengths_k =
            tunable->WeiBlockCopyClusterLengths_K;

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

        // const auto KBlockWork = K / KPerBlock;
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

    result_t run(const args_t *arg, const igemm_v4r1_dynamic_tunable_t *tunable,
                 hipModule_t module, float *p_in, float *p_wei, float *p_out,
                 int warmup, int repeat) {
        if (!tunable_is_valid(arg, tunable)) {
            result_t result;
            result.return_code = -1;
            return result;
        }
        hipFunction_t kernel_func;
        std::string kernel_name = get_kernel_name(tunable);
        // printf("kernel:%s\n", kernel_name.c_str());
        HIP_CALL(
            hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));

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
};

#endif