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

#ifndef __DIRECT_CONV_DRIVER_H
#define __DIRECT_CONV_DRIVER_H


#include <string>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <numeric>

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

#define _VEC_C_ 8

static inline size_t gpu_conv_out_size(size_t in_size, size_t pad,
                                         size_t dilation, size_t ksize,
                                         size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

static inline int env_get_int(const char *var_name, int default_int) {
    char *v = getenv(var_name);
    int r = default_int;
    if (v)
        r = atoi(v);
    return r;
}

typedef struct {
    void *   p_in;
    void *   p_wei;
    void *   p_out;
    uint32_t hi;
    uint32_t wi;
    uint32_t n;
    uint32_t k_per_group;
    uint32_t c_per_group;
    uint32_t ho;
    uint32_t wo;
    uint32_t sy;
    uint32_t sx;
    uint32_t dy;
    uint32_t dx;
    uint32_t py;
    uint32_t px;
    uint32_t fy;
    uint32_t fx;
    uint32_t group;
    uint32_t vec_c;
    uint32_t batch_m;
    uint32_t stride_m;
    float    alpha;
    float    beta;
    float    gamma;
    uint32_t magic_0;
    uint32_t magic_1;
    uint32_t magic_2;
    uint32_t shift_pack_0;
} __attribute__((packed)) igemm_fwd_cnhwc_karg_t;
static inline void dump_igemm_fwd_cnhwc_2d_karg(igemm_fwd_cnhwc_karg_t * karg)
{
    std::cout<<"p_in:"<<karg->p_in<<", ";
    std::cout<<"p_wei:"<<karg->p_wei<<", ";
    std::cout<<"p_out:"<<karg->p_out<<", ";
    std::cout<<"hi:"<<karg->hi<<", ";
    std::cout<<"wi:"<<karg->wi<<", ";
    std::cout<<"n:"<<karg->n<<", ";
    std::cout<<"k_per_group:"<<karg->k_per_group<<", ";
    std::cout<<"c_per_group:"<<karg->c_per_group<<", ";
    std::cout<<"ho:"<<karg->ho<<", ";
    std::cout<<"wo:"<<karg->wo<<", ";
    std::cout<<"sy:"<<karg->sy<<", ";
    std::cout<<"sx:"<<karg->sx<<", ";
    std::cout<<"dy:"<<karg->dy<<", ";
    std::cout<<"dx:"<<karg->dx<<", ";
    std::cout<<"py:"<<karg->py<<", ";
    std::cout<<"px:"<<karg->px<<", ";
    std::cout<<"fy:"<<karg->fy<<", ";
    std::cout<<"fx:"<<karg->fx<<", ";
    std::cout<<"group:"<<karg->group<<", ";
    std::cout<<"vec_c:"<<karg->vec_c<<", ";
    std::cout<<"batch_m:"<<karg->batch_m<<", ";
    std::cout<<"stride_m:"<<karg->stride_m<<", ";
    std::cout<<"alpha:"<<karg->alpha<<", ";
    std::cout<<"beta:"<<karg->beta<<", ";
    std::cout<<"gamma:"<<karg->gamma<<", ";
    std::cout<<"magic_0:"<<karg->magic_0<<", ";
    std::cout<<"magic_1:"<<karg->magic_1<<", ";
    std::cout<<"magic_2:"<<karg->magic_2<<", ";
    std::cout<<"shift_pack_0:"<<karg->shift_pack_0<<std::endl;
}

typedef struct {
    std::string kernel_name;
    std::string data_type;
    uint32_t m_per_block;
    uint32_t n_per_block;
    uint32_t k_per_block;
    uint32_t block_size;
    uint32_t r;
    uint32_t occupancy;
} igemm_fwd_cnhwc_kernel_info_t;

igemm_fwd_cnhwc_kernel_info_t igemm_fwd_cnhwc_kernel_list [] = 
{
    {"igemm_fwd_gtcx_cnhwc_fp16_ex0_bt256x128x32_wt32x32x8_ws2x1_wr2x2"  , "fp16",  256,  128, 32, 256, 1, 2},
};

class igemm_fwd_cnhwc_t {
public:
    int num_cu;
    int gcn_arch = 0;
    igemm_fwd_cnhwc_t(){
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        num_cu = dev_prop.multiProcessorCount;
        gcn_arch = dev_prop.gcnArch;
        if(gcn_arch >= 1000)
            num_cu *= 2;
    }
    ~igemm_fwd_cnhwc_t(){}
    std::string get_kernel_name(const igemm_fwd_cnhwc_kernel_info_t *kernel_info) {
        return kernel_info->kernel_name;
    }

    bool is_valid(const args_t *arg, igemm_fwd_cnhwc_kernel_info_t * kernel_info)
    {
        size_t hi = arg->get_int("in_h");
        size_t wi = arg->get_int("in_w");
        size_t n = arg->get_int("batchsize");
        size_t k = arg->get_int("out_channels");
        size_t c = arg->get_int("in_channels");

        size_t sy = arg->get_int("conv_stride_h");
        size_t sx = arg->get_int("conv_stride_w");
        size_t dy = arg->get_int("dilation_h");
        size_t dx = arg->get_int("dilation_w");
        size_t py = arg->get_int("pad_h");
        size_t px = arg->get_int("pad_w");
        size_t fy = arg->get_int("fil_h");
        size_t fx = arg->get_int("fil_w");
        size_t ho = gpu_conv_out_size(hi, py, dy, fy, sy);
        size_t wo = gpu_conv_out_size(wi, px, dx, fx, sx);
        size_t group = arg->get_int("group_count");

        assert(c % group == 0 && k % group == 0);

        assert(group != 0 && c % group == 0 && k % group == 0);

        size_t k_per_group  = k / group;
        size_t c_per_group  = c / group;

        if(k_per_group % kernel_info->n_per_block != 0)
            return false;
        
        return true;
    }

    result_t run(const args_t *arg,  hipModule_t module, igemm_fwd_cnhwc_kernel_info_t * kernel_info,
                 void *p_in, void *p_wei, void *p_out,
                 int warmup, int repeat, const driverDataType_t& data_type) {
        if(!is_valid(arg, kernel_info)){
            result_t result;
            result.return_code = -1;
            return result;
        }
        size_t hi = arg->get_int("in_h");
        size_t wi = arg->get_int("in_w");
        size_t n = arg->get_int("batchsize");
        size_t k = arg->get_int("out_channels");
        size_t c = arg->get_int("in_channels");

        size_t sy = arg->get_int("conv_stride_h");
        size_t sx = arg->get_int("conv_stride_w");
        size_t dy = arg->get_int("dilation_h");
        size_t dx = arg->get_int("dilation_w");
        size_t py = arg->get_int("pad_h");
        size_t px = arg->get_int("pad_w");
        size_t fy = arg->get_int("fil_h");
        size_t fx = arg->get_int("fil_w");
        size_t ho = gpu_conv_out_size(hi, py, dy, fy, sy);
        size_t wo = gpu_conv_out_size(wi, px, dx, fx, sx);
        size_t group = arg->get_int("group_count");

        size_t vec_c = _VEC_C_;

        assert(c % (group * vec_c) == 0 && k % (group * vec_c) == 0);

        assert(group != 0 && c % (group * vec_c) == 0 && k % (group * vec_c) == 0);

        size_t k_per_group  = k / group;
        size_t c_per_group  = c / group / vec_c;
        igemm_fwd_cnhwc_karg_t karg;
        karg.p_in           = p_in;
        karg.p_wei          = p_wei;
        karg.p_out          = p_out;
        karg.hi             = static_cast<int>(hi);
        karg.wi             = static_cast<int>(wi);
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
        karg.vec_c          = static_cast<int>(vec_c);
        size_t karg_size    = sizeof(karg);

        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                            HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                            HIP_LAUNCH_PARAM_END};

        hipFunction_t kernel_func;
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_info->kernel_name.c_str()));

        int block_size  = kernel_info->block_size;
        int num_gemm_m  = (n * ho * wo + kernel_info->m_per_block - 1) / kernel_info->m_per_block;
        int num_gemm_n  = (k_per_group + kernel_info->n_per_block - 1) / kernel_info->n_per_block;

        // int grid_size = num_gemm_m * num_gemm_n;

        printf("s_p_out=%x\n", p_out);

        printf("launch fwd block:%d, grid:[%d,%d]\n", block_size, num_gemm_m, num_gemm_n);
        // dump_igemm_fwd_cnhwc_2d_karg(&karg);

        auto launch_fwd = [&]() -> float {
            void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};
            float ms = .0;

            hipEvent_t start;
            hipEvent_t stop;
            hipEventCreate(&start);
            hipEventCreate(&stop);

            // for hipHccModuleLaunchKernel/hipExtModuleLaunchKernel, the grid_size is in unit of workitem
            HIP_CALL(hipHccModuleLaunchKernel(kernel_func, num_gemm_m * block_size, num_gemm_n, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, start, stop));

            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms, start, stop);
            hipEventDestroy(start);
            hipEventDestroy(stop);

            return ms;
        };

        for (int i = 0; i < warmup; i++) {
            launch_fwd();
        }

        if(env_get_int("DBG_MODE", 0) != 0){
            printf("workspace debug \r\n");
            float* gemmc_host_check = (float* )malloc(k * n * ho * wo * sizeof(float));
            printf("gemmc_host_check size=%d\n",  k * n * ho * wo * sizeof(float));
            printf("copy output\n");
            hipMemcpy(gemmc_host_check, p_out, k * n * ho * wo * sizeof(float16), hipMemcpyDeviceToHost);

            for (int i_check = 0; i_check < (0+block_size); i_check++)
            {
                float16 *gemmc_host_check_fp16 = (float16 *)gemmc_host_check;
                float16 check_num0 = gemmc_host_check_fp16[i_check*2];
                float16 check_num1 = gemmc_host_check_fp16[i_check*2+1];
                float check_num0_fp32 = (float)check_num0;
                float check_num1_fp32 = (float)check_num1;
                printf("[%d]th var to monitor:[%f, %d, fp16(%f, %f)]\r\n", i_check, gemmc_host_check[i_check], ((int *)gemmc_host_check)[i_check], check_num0_fp32, check_num1_fp32);
            }
            printf("s_p_out=%x\n", p_out);
            printf("workspace debug end \r\n");
            free(gemmc_host_check);
        }

        std::vector<float> duration_list;
        for (int i = 0; i < repeat; i++) {
            float d = launch_fwd();
            duration_list.push_back(d);
        }

        // remove min and max from list, then do average
        auto imin = std::min_element(begin(duration_list), end(duration_list));
        duration_list.erase(imin);
        auto imax = std::max_element(begin(duration_list), end(duration_list));
        duration_list.erase(imax);
        assert(duration_list.size() == (repeat - 2));
        float avg_duration = std::accumulate(duration_list.begin(), duration_list.end(), (float).0) / duration_list.size();

        usleep(1000 * 1);

        result_t result;
        result.return_code = 0;
        result.duration_ms = avg_duration;
        result.kernel_name = kernel_info->kernel_name;
        return result;
    }
};

#endif
