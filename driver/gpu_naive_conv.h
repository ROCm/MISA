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
#ifndef __GPU_NAIVE_CONV_H
#define __GPU_NAIVE_CONV_H

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <assert.h>

#ifndef HIP_CALL
#define HIP_CALL(call)                                                         \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            printf("[hiperror](%d) fail to call %s,(%s)", (int)err, #call,     \
                   hipGetErrorString(err));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)
#endif

static inline size_t gpu_naive_conv_out_size(size_t in_size, size_t pad,
                                         size_t dilation, size_t ksize,
                                         size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

static struct {
    hipModule_t     module;
    hipFunction_t   kernel_naive_conv_fwd_nchw_fp32;
    hipFunction_t   kernel_naive_conv_bwd_nchw_fp32;
    hipFunction_t   kernel_naive_conv_wrw_nchw_fp32;
    hipFunction_t   kernel_naive_conv_fwd_ncdhw_fp32;
    hipFunction_t   kernel_naive_conv_bwd_ncdhw_fp32;
    hipFunction_t   kernel_naive_conv_wrw_ncdhw_fp32;

    hipFunction_t   kernel_naive_conv_fwd_nchw_fp16;
    hipFunction_t   kernel_naive_conv_bwd_nchw_fp16;
    hipFunction_t   kernel_naive_conv_wrw_nchw_fp16;
    hipFunction_t   kernel_naive_conv_fwd_ncdhw_fp16;
    hipFunction_t   kernel_naive_conv_bwd_ncdhw_fp16;
    hipFunction_t   kernel_naive_conv_wrw_ncdhw_fp16;

    hipFunction_t   kernel_naive_conv_fwd_nhwc_fp32;
    hipFunction_t   kernel_naive_conv_bwd_nhwc_fp32;
    hipFunction_t   kernel_naive_conv_wrw_nhwc_fp32;
    hipFunction_t   kernel_naive_conv_fwd_ndhwc_fp32;
    hipFunction_t   kernel_naive_conv_bwd_ndhwc_fp32;
    hipFunction_t   kernel_naive_conv_wrw_ndhwc_fp32;
} the_gpu_handle;


static inline void gpu_naive_conv_init(const char * hsaco){
    static int inited = 0;
    if(!inited){
        HIP_CALL(hipModuleLoad(&the_gpu_handle.module, hsaco));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_fwd_nchw_fp32,  the_gpu_handle.module, "naive_conv_fwd_nchw_fp32"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_bwd_nchw_fp32,  the_gpu_handle.module, "naive_conv_bwd_nchw_fp32"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_wrw_nchw_fp32,  the_gpu_handle.module, "naive_conv_wrw_nchw_fp32"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_fwd_ncdhw_fp32, the_gpu_handle.module, "naive_conv_fwd_ncdhw_fp32"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_bwd_ncdhw_fp32, the_gpu_handle.module, "naive_conv_bwd_ncdhw_fp32"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_wrw_ncdhw_fp32, the_gpu_handle.module, "naive_conv_wrw_ncdhw_fp32"));

        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_fwd_nchw_fp16,  the_gpu_handle.module, "naive_conv_fwd_nchw_fp16"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_bwd_nchw_fp16,  the_gpu_handle.module, "naive_conv_bwd_nchw_fp16"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_wrw_nchw_fp16,  the_gpu_handle.module, "naive_conv_wrw_nchw_fp16"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_fwd_ncdhw_fp16, the_gpu_handle.module, "naive_conv_fwd_ncdhw_fp16"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_bwd_ncdhw_fp16, the_gpu_handle.module, "naive_conv_bwd_ncdhw_fp16"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_wrw_ncdhw_fp16, the_gpu_handle.module, "naive_conv_wrw_ncdhw_fp16"));

        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_fwd_nhwc_fp32,  the_gpu_handle.module, "naive_conv_fwd_nhwc_fp32"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_bwd_nhwc_fp32,  the_gpu_handle.module, "naive_conv_bwd_nhwc_fp32"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_wrw_nhwc_fp32,  the_gpu_handle.module, "naive_conv_wrw_nhwc_fp32"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_fwd_ndhwc_fp32, the_gpu_handle.module, "naive_conv_fwd_ndhwc_fp32"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_bwd_ndhwc_fp32, the_gpu_handle.module, "naive_conv_bwd_ndhwc_fp32"));
        HIP_CALL(hipModuleGetFunction(&the_gpu_handle.kernel_naive_conv_wrw_ndhwc_fp32, the_gpu_handle.module, "naive_conv_wrw_ndhwc_fp32"));

        inited = 1;
    }
}

typedef struct {
    void * p_in;
    void * p_wei;
    void * p_out;
    int hi;
    int wi;
    int n;
    int k_per_group;
    int c_per_group;
    int ho;
    int wo;
    int sy;
    int sx;
    int dy;
    int dx;
    int py;
    int px;
    int fy;
    int fx;
    int group;
} __attribute__((packed)) naive_conv_2d_karg_t;

typedef struct {
    void * p_in;
    void * p_wei;
    void * p_out;
    int di;
    int hi;
    int wi;
    int n;
    int k_per_group;
    int c_per_group;
    int do_;
    int ho;
    int wo;
    int sz;
    int sy;
    int sx;
    int dz;
    int dy;
    int dx;
    int pz;
    int py;
    int px;
    int fz;
    int fy;
    int fx;
    int group;
} __attribute__((packed)) naive_conv_3d_karg_t;


static inline void gpu_naive_conv_fwd_nchw_fp32(void *src, void *filter,
                                       void *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy, size_t group)
{
    assert(group != 0 && c % group == 0 && k % group == 0);
    size_t ho = gpu_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = gpu_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group  = k / group;
    size_t c_per_group  = c / group;
    naive_conv_2d_karg_t karg;
    karg.p_in           = src;
    karg.p_wei          = filter;
    karg.p_out          = dst;
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size = n * k_per_group * group;
    HIP_CALL(hipHccModuleLaunchKernel(the_gpu_handle.kernel_naive_conv_fwd_nchw_fp32, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

}

static inline void gpu_naive_conv_bwd_nchw_fp32(void *src, void *filter,
                                       void *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy, size_t group)
{
    assert(group != 0 && c % group == 0 && k % group == 0);
    size_t ho = gpu_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = gpu_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group  = k / group;
    size_t c_per_group  = c / group;
    naive_conv_2d_karg_t karg;
    karg.p_in           = src;
    karg.p_wei          = filter;
    karg.p_out          = dst;
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size = n * c_per_group * group;
    HIP_CALL(hipHccModuleLaunchKernel(the_gpu_handle.kernel_naive_conv_bwd_nchw_fp32, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

}

static inline void gpu_naive_conv_wrw_nchw_fp32(void *src, void *filter,
                                       void *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy, size_t group)
{
    assert(group != 0 && c % group == 0 && k % group == 0);
    size_t ho = gpu_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = gpu_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group  = k / group;
    size_t c_per_group  = c / group;
    naive_conv_2d_karg_t karg;
    karg.p_in           = src;
    karg.p_wei          = filter;
    karg.p_out          = dst;
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size = k_per_group * group;
    HIP_CALL(hipHccModuleLaunchKernel(the_gpu_handle.kernel_naive_conv_wrw_nchw_fp32, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

}

static inline void gpu_naive_conv_fwd_nchw_fp16(void *src, void *filter,
                                       void *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy, size_t group)
{
    assert(group != 0 && c % group == 0 && k % group == 0);
    size_t ho = gpu_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = gpu_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group  = k / group;
    size_t c_per_group  = c / group;
    naive_conv_2d_karg_t karg;
    karg.p_in           = src;
    karg.p_wei          = filter;
    karg.p_out          = dst;
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size = n * k_per_group * group;
    HIP_CALL(hipHccModuleLaunchKernel(the_gpu_handle.kernel_naive_conv_fwd_nchw_fp16, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

}

static inline void gpu_naive_conv_bwd_nchw_fp16(void *src, void *filter,
                                       void *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy, size_t group)
{
    assert(group != 0 && c % group == 0 && k % group == 0);
    size_t ho = gpu_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = gpu_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group  = k / group;
    size_t c_per_group  = c / group;
    naive_conv_2d_karg_t karg;
    karg.p_in           = src;
    karg.p_wei          = filter;
    karg.p_out          = dst;
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size = n * c_per_group * group;
    HIP_CALL(hipHccModuleLaunchKernel(the_gpu_handle.kernel_naive_conv_bwd_nchw_fp16, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

}

static inline void gpu_naive_conv_wrw_nchw_fp16(void *src, void *filter,
                                       void *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy, size_t group)
{
    assert(group != 0 && c % group == 0 && k % group == 0);
    size_t ho = gpu_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = gpu_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group  = k / group;
    size_t c_per_group  = c / group;
    naive_conv_2d_karg_t karg;
    karg.p_in           = src;
    karg.p_wei          = filter;
    karg.p_out          = dst;
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size = k_per_group * group;
    HIP_CALL(hipHccModuleLaunchKernel(the_gpu_handle.kernel_naive_conv_wrw_nchw_fp16, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

}


static inline void gpu_naive_conv_fwd_nhwc_fp32(void *src, void *filter,
                                       void *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy, size_t group)
{
    assert(group != 0 && c % group == 0 && k % group == 0);
    size_t ho = gpu_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = gpu_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group  = k / group;
    size_t c_per_group  = c / group;
    naive_conv_2d_karg_t karg;
    karg.p_in           = src;
    karg.p_wei          = filter;
    karg.p_out          = dst;
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size = group * n * ho;
    HIP_CALL(hipHccModuleLaunchKernel(the_gpu_handle.kernel_naive_conv_fwd_nhwc_fp32, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

}

static inline void gpu_naive_conv_bwd_nhwc_fp32(void *src, void *filter,
                                       void *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy, size_t group)
{
    assert(group != 0 && c % group == 0 && k % group == 0);
    size_t ho = gpu_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = gpu_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group  = k / group;
    size_t c_per_group  = c / group;
    naive_conv_2d_karg_t karg;
    karg.p_in           = src;
    karg.p_wei          = filter;
    karg.p_out          = dst;
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size = group * n * h;
    HIP_CALL(hipHccModuleLaunchKernel(the_gpu_handle.kernel_naive_conv_bwd_nhwc_fp32, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

}

static inline void gpu_naive_conv_wrw_nhwc_fp32(void *src, void *filter,
                                       void *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy, size_t group)
{
    assert(group != 0 && c % group == 0 && k % group == 0);
    size_t ho = gpu_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = gpu_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group  = k / group;
    size_t c_per_group  = c / group;
    naive_conv_2d_karg_t karg;
    karg.p_in           = src;
    karg.p_wei          = filter;
    karg.p_out          = dst;
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size = group * k_per_group;
    HIP_CALL(hipHccModuleLaunchKernel(the_gpu_handle.kernel_naive_conv_wrw_nhwc_fp32, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

}

static inline void gpu_naive_conv_fwd_ndhwc_fp32(void *src, void *filter, void *dst,
                                       size_t n, size_t w, size_t h, size_t d, size_t c, size_t k,
                                       size_t fx, size_t fy, size_t fz, size_t px, size_t py, size_t pz,
                                       size_t sx, size_t sy, size_t sz, size_t dx, size_t dy, size_t dz, size_t group)
{
    assert(group != 0 && c % group == 0 && k % group == 0);
    size_t do_= gpu_naive_conv_out_size(d, pz, dz, fz, sz);
    size_t ho = gpu_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = gpu_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group  = k / group;
    size_t c_per_group  = c / group;
    naive_conv_3d_karg_t karg;
    karg.p_in           = src;
    karg.p_wei          = filter;
    karg.p_out          = dst;
    karg.di             = static_cast<int>(d);
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.do_            = static_cast<int>(do_);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sz             = static_cast<int>(sz);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dz             = static_cast<int>(dz);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.pz             = static_cast<int>(pz);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fz             = static_cast<int>(fz);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size = group * n * do_;
    HIP_CALL(hipHccModuleLaunchKernel(the_gpu_handle.kernel_naive_conv_fwd_ndhwc_fp32, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

}

static inline void gpu_naive_conv_bwd_ndhwc_fp32(void *src, void *filter, void *dst,
                                       size_t n, size_t w, size_t h, size_t d, size_t c, size_t k,
                                       size_t fx, size_t fy, size_t fz, size_t px, size_t py, size_t pz,
                                       size_t sx, size_t sy, size_t sz, size_t dx, size_t dy, size_t dz, size_t group)
{
    assert(group != 0 && c % group == 0 && k % group == 0);
    size_t do_= gpu_naive_conv_out_size(d, pz, dz, fz, sz);
    size_t ho = gpu_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = gpu_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group  = k / group;
    size_t c_per_group  = c / group;
    naive_conv_3d_karg_t karg;
    karg.p_in           = src;
    karg.p_wei          = filter;
    karg.p_out          = dst;
    karg.di             = static_cast<int>(d);
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.do_            = static_cast<int>(do_);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sz             = static_cast<int>(sz);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dz             = static_cast<int>(dz);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.pz             = static_cast<int>(pz);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fz             = static_cast<int>(fz);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size = group * n * d;
    HIP_CALL(hipHccModuleLaunchKernel(the_gpu_handle.kernel_naive_conv_bwd_ndhwc_fp32, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

}

static inline void gpu_naive_conv_wrw_ndhwc_fp32(void *src, void *filter, void *dst,
                                       size_t n, size_t w, size_t h, size_t d, size_t c, size_t k,
                                       size_t fx, size_t fy, size_t fz, size_t px, size_t py, size_t pz,
                                       size_t sx, size_t sy, size_t sz, size_t dx, size_t dy, size_t dz, size_t group)
{
    assert(group != 0 && c % group == 0 && k % group == 0);
    size_t do_= gpu_naive_conv_out_size(d, pz, dz, fz, sz);
    size_t ho = gpu_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = gpu_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group  = k / group;
    size_t c_per_group  = c / group;
    naive_conv_3d_karg_t karg;
    karg.p_in           = src;
    karg.p_wei          = filter;
    karg.p_out          = dst;
    karg.di             = static_cast<int>(d);
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.do_            = static_cast<int>(do_);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sz             = static_cast<int>(sz);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dz             = static_cast<int>(dz);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.pz             = static_cast<int>(pz);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fz             = static_cast<int>(fz);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size = group * k_per_group;
    HIP_CALL(hipHccModuleLaunchKernel(the_gpu_handle.kernel_naive_conv_wrw_ndhwc_fp32, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

}

#endif
