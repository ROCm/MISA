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

#ifndef __REFERENCE_NAIVE_CONV_DRIVER_H
#define __REFERENCE_NAIVE_CONV_DRIVER_H

#include <string>
#include <sstream>
#include <ostream>
#include <assert.h>
#include <algorithm>
#include <numeric>
#include <unistd.h>

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
} __attribute__((packed)) reference_naive_conv_2d_karg_t;

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
} __attribute__((packed)) reference_naive_conv_3d_karg_t;

typedef struct {
    std::string direction;
    std::string tensor_layout;
    std::string precision;
}reference_naive_conv_ctrl_t;

static inline std::string reference_naive_conv_get_kernel_name(const reference_naive_conv_ctrl_t * ctrl)
{
    std::ostringstream kernel_name;
    kernel_name << "naive_conv_";
    if(ctrl->direction == "fwd")
        kernel_name << "fwd_";
    else if(ctrl->direction == "bwd")
        kernel_name << "bwd_";
    else if(ctrl->direction == "wrw")
        kernel_name << "wrw_";
    else
        assert(0);

    if(ctrl->tensor_layout == "nchw")
        kernel_name << "nchw_";
    else if(ctrl->tensor_layout == "ncdhw")
        kernel_name << "ncdhw_";
    else
        assert(0);

    if(ctrl->precision == "fp32")
        kernel_name << "fp32";
    else if(ctrl->precision == "fp16")
        kernel_name << "fp16";
    else if(ctrl->precision == "bf16")
        kernel_name << "bf16";
    else
        assert(0);

    return kernel_name.str();
}

class reference_naive_conv_t {
public:
    result_t run(const args_t *arg, const reference_naive_conv_ctrl_t *ctrl,
                 hipModule_t module, void *p_in, void *p_wei, void *p_out,
                 int warmup, int repeat)
    {
        int di = arg->get_int("in_d");
        int hi = arg->get_int("in_h");
        int wi = arg->get_int("in_w");
        int n  = arg->get_int("batchsize");
        int k  = arg->get_int("out_channels");
        int c  = arg->get_int("in_channels");

        int sz = arg->get_int("conv_stride_d");
        int sy = arg->get_int("conv_stride_h");
        int sx = arg->get_int("conv_stride_w");
        int dz = arg->get_int("dilation_d");
        int dy = arg->get_int("dilation_h");
        int dx = arg->get_int("dilation_w");
        int pz = arg->get_int("pad_d");
        int py = arg->get_int("pad_h");
        int px = arg->get_int("pad_w");
        int fz = arg->get_int("fil_d");
        int fy = arg->get_int("fil_h");
        int fx = arg->get_int("fil_w");
        int group = arg->get_int("group_count");
        int do_= conv_out_size(di, pz, dz, fz, sz);
        int ho = conv_out_size(hi, py, dy, fy, sy);
        int wo = conv_out_size(wi, px, dx, fx, sx);
        int c_per_group = c / group;
        int k_per_group = k / group;

        reference_naive_conv_2d_karg_t conv_2d_karg;
        conv_2d_karg.p_in           = p_in;
        conv_2d_karg.p_wei          = p_wei;
        conv_2d_karg.p_out          = p_out;
        conv_2d_karg.hi             = hi;
        conv_2d_karg.wi             = wi;
        conv_2d_karg.n              = n;
        conv_2d_karg.k_per_group    = k_per_group;
        conv_2d_karg.c_per_group    = c_per_group;
        conv_2d_karg.ho             = ho;
        conv_2d_karg.wo             = wo;
        conv_2d_karg.sy             = sy;
        conv_2d_karg.sx             = sx;
        conv_2d_karg.dy             = dy;
        conv_2d_karg.dx             = dx;
        conv_2d_karg.py             = py;
        conv_2d_karg.px             = px;
        conv_2d_karg.fy             = fy;
        conv_2d_karg.fx             = fx;
        conv_2d_karg.group          = group;
        size_t conv_2d_karg_size = sizeof(conv_2d_karg);

        reference_naive_conv_3d_karg_t conv_3d_karg;
        conv_3d_karg.p_in           = p_in;
        conv_3d_karg.p_wei          = p_wei;
        conv_3d_karg.p_out          = p_out;
        conv_3d_karg.di             = di;
        conv_3d_karg.hi             = hi;
        conv_3d_karg.wi             = wi;
        conv_3d_karg.n              = n;
        conv_3d_karg.k_per_group    = k_per_group;
        conv_3d_karg.c_per_group    = c_per_group;
        conv_3d_karg.do_            = do_;
        conv_3d_karg.ho             = ho;
        conv_3d_karg.wo             = wo;
        conv_3d_karg.sz             = sz;
        conv_3d_karg.sy             = sy;
        conv_3d_karg.sx             = sx;
        conv_3d_karg.dz             = dz;
        conv_3d_karg.dy             = dy;
        conv_3d_karg.dx             = dx;
        conv_3d_karg.pz             = pz;
        conv_3d_karg.py             = py;
        conv_3d_karg.px             = px;
        conv_3d_karg.fz             = fz;
        conv_3d_karg.fy             = fy;
        conv_3d_karg.fx             = fx;
        conv_3d_karg.group          = group;
        size_t conv_3d_karg_size = sizeof(conv_3d_karg);

        int block_size = 256;
        int grid_size = 0;
        if(ctrl->direction == "fwd")
            grid_size = n * k;
        else if(ctrl->direction == "bwd")
            grid_size = n * c;
        else if(ctrl->direction == "wrw")
            grid_size = k;
        else
            assert(0);
        
        int spatial_dim = ctrl->tensor_layout == "nchw" ? 2 : 3;

        hipFunction_t kernel_func;
        std::string kernel_name = reference_naive_conv_get_kernel_name(ctrl);
        HIP_CALL( hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));


        auto launch_kernel = [&]() -> float {
            if(spatial_dim == 2){
                // printf("launch fwd block:%d, grid:%d\n", block_size, grid_size);
                // dump_fwd_karg(&karg);
                void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &conv_2d_karg,
                            HIP_LAUNCH_PARAM_BUFFER_SIZE, &conv_2d_karg_size,
                            HIP_LAUNCH_PARAM_END};
                float ms = .0;

                hipEvent_t start;
                hipEvent_t stop;
                hipEventCreate(&start);
                hipEventCreate(&stop);

                HIP_CALL(hipHccModuleLaunchKernel(kernel_func, grid_size * block_size, 1, 1,
                                                block_size, 1, 1, 0, 0, NULL,
                                                (void **)&config, start, stop));

                hipEventSynchronize(stop);
                hipEventElapsedTime(&ms, start, stop);
                hipEventDestroy(start);
                hipEventDestroy(stop);
                return ms;
            }else{
                // printf("launch fwd block:%d, grid:%d\n", block_size, grid_size);
                // dump_fwd_karg(&karg);
                void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &conv_3d_karg,
                            HIP_LAUNCH_PARAM_BUFFER_SIZE, &conv_3d_karg_size,
                            HIP_LAUNCH_PARAM_END};
                float ms = .0;

                hipEvent_t start;
                hipEvent_t stop;
                hipEventCreate(&start);
                hipEventCreate(&stop);

                HIP_CALL(hipHccModuleLaunchKernel(kernel_func, grid_size * block_size, 1, 1,
                                                block_size, 1, 1, 0, 0, NULL,
                                                (void **)&config, start, stop));

                hipEventSynchronize(stop);
                hipEventElapsedTime(&ms, start, stop);
                hipEventDestroy(start);
                hipEventDestroy(stop);
                return ms;
            }
        };

        for (int i = 0; i < warmup; i++) {
            launch_kernel();
        }

        std::vector<float> duration_list;
        for (int i = 0; i < repeat; i++) {
            float d = launch_kernel();
            duration_list.push_back(d);
        }

        // remove min and max from list, then do average
        auto imin = std::min_element(begin(duration_list), end(duration_list));
        duration_list.erase(imin);
        auto imax = std::max_element(begin(duration_list), end(duration_list));
        duration_list.erase(imax);
        assert(duration_list.size() == (repeat - 2));
        float avg_duration = std::accumulate(duration_list.begin(), duration_list.end(), (float).0) / duration_list.size();

        usleep(1000 * 5);

        result_t result;
        result.return_code = 0;
        result.duration_ms = avg_duration;
        result.kernel_name = kernel_name;
        return result;
    }
};

#endif
