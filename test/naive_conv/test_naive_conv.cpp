
#include <vector>
#include <string>
#include <assert.h>
#include <chrono>
#include <functional>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <vector>
#include <float.h>
#include <cmath>
#include <iostream>

#define NAIVE_CONV_THREADED
#include "naive_conv.h"
#include "gpu_naive_conv.h"

#define HIP_CALL(call)                                                         \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            printf("[hiperror](%d) fail to call %s,(%s)", (int)err, #call,     \
                   hipGetErrorString(err));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static inline int env_get_int(const char *var_name, int default_int) {
    char *v = getenv(var_name);
    int r = default_int;
    if (v)
        r = atoi(v);
    return r;
}

static int gen_rand_integer()
{
    static int inited = 0;
    if(inited == 0)
    {
        std::srand(std::time(nullptr));
        inited = 1;
    }
    return std::rand();
}


static inline char *env_get_str(const char *var_name, char *default_str) {
    char *v = getenv(var_name);
    if (v)
        return v;
    return default_str;
}

template <typename T>
struct distribution_t{
};

template <>
struct distribution_t<int>{
    distribution_t(int min, int max) : distribution(min, max) {}
    template<class URNG>
    int operator()(URNG & rng){ return distribution(rng);}
    std::uniform_int_distribution<int> distribution;
};
template <>
struct distribution_t<float>{
    distribution_t(float min, float max) : distribution(min, max) {}
    template<class URNG>
    float operator()(URNG & rng){ return distribution(rng);}
    std::uniform_real_distribution<float> distribution;
};

template <typename Dst_T, typename Src_T>
void block_wise_rand_generator(Dst_T *p, int tid, int block_size, int total_size, Src_T min, Src_T max, Src_T scale)
{
    std::mt19937 rng(std::chrono::system_clock::now()
                        .time_since_epoch()
                        .count() +
                    std::hash<std::thread::id>()(std::this_thread::get_id()));
    distribution_t<Src_T> distribution(min,max);
    for (int i = tid; i < total_size; i += block_size) {
        p[i] = static_cast<Dst_T>(scale * distribution(rng));
    }
}

template <typename Dst_T, typename Src_T>
void gen_rand_vector(Dst_T *vec, size_t vec_size, Src_T fmin, Src_T fmax, Src_T scale = 1) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads < 4)
        num_threads = 4;
    // printf("total threads:%d\n",num_threads);
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.push_back(std::thread(block_wise_rand_generator<Dst_T, Src_T>,
            vec, t, num_threads, vec_size, fmin, fmax, scale));
    }
    for (auto &th : threads)
        th.join();
}

static inline bool valid_float(float p)
{
    return !(std::isnan(p) || std::isinf(p));
}
#ifndef ABS
#define ABS(b) ((b) > 0 ? (b) : -1 * (b))
#endif
static inline bool valid_vector(const float *ref, const float *pred, int n,
                                double nrms = 1.5e-6) {
    double s0 = 0.0;
    double s1 = 0.0;
    int igemm_per_pixel_check = env_get_int("PER_PIXEL_CHECK", 0);
    int igemm_per_pixel_check_print = env_get_int("PER_PIXEL_CHECK_PRINT", 1);
    int pp_err = 0;

    for (int i = 0; i < n; ++i) {
        if(!(valid_float(ref[i]) && valid_float(pred[i]))){
            printf(" invalid float at %4d, ref:%f, pred:%f\n", i, ref[i], pred[i]);
            return -1;
        }
        double ri = (double)ref[i];
        double pi = (double)pred[i];
        double d = ri - pi;
        double dd = d * d;
        double rr = 2.0 * ri * ri;
        s0 += dd;
        s1 += rr;
        if(igemm_per_pixel_check){
            double delta = ABS(ABS(ri - pi) / ri);
            printf("[%d] ref:%lf, pred:%lf(0x%08x) [%s]\n", i, ri, pi, ((uint32_t *)pred)[i], delta > 3e-5? "N":"Y");
            if (delta > 3e-5) {
                if(igemm_per_pixel_check_print){
                    if (pp_err < 100)
                        printf("diff at %4d, ref:%lf, pred:%lf(0x%08x), d:%lf\n", i, ri,
                            pi, ((uint32_t *)pred)[i], delta);
                }
                pp_err++;
            }

        }
    }
    // printf("\nnrms:%lf, s0:%lf, s1:%lf, expected_nrms is %1f\n",sqrt(s0/s1),s0,s1,nrms);
    fflush(stdout);
    return (sqrt(s0 / s1) < nrms)
#ifdef PER_PIXEL_CHECK
           && (pp_err == 0)
#endif
        ;
}

class test_conv_t{
public:
    static int conv_out_size(int in_size, int pad, int dilation, int ksize, int stride)
    {
        return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
    }

    static std::vector<int> get_image_depth() { return {8, 10}; }

    static std::vector<int> get_image_size() { return {11, 17}; }

    static std::vector<int> get_channel_size() { return {3, 8}; }

    static std::vector<int> get_filter_depth() { return {1, 3}; }

    static std::vector<int> get_filter_size() { return {1, 3}; }

    static std::vector<int> get_stride_depth() { return {1, 2}; }

    static std::vector<int> get_dilation_depth() { return {1}; }

    static std::vector<int> get_stride_dilation_size() { return {1, 2}; }

    static std::vector<int> get_pad_depth() { return {0, 1}; }

    static std::vector<int> get_pad_size() { return {0, 1}; }

    static std::vector<int> get_group_size() { return {1, 2}; }

    static std::vector<int> get_batch_size() { return {1, 2}; }

    template <typename F>
    void iterate_conv_2d(F f)
    {
        for(int c : get_channel_size())
        {
            for(int hi : get_image_size())
            {
                for(int wi : get_image_size())
                {   
                    for(int fy : get_filter_size())
                    {
                        for(int fx : get_filter_size())
                        {
                            for(int py : get_pad_size())
                            {
                                for(int px : get_pad_size())
                                {
                                    for(int sy : get_stride_dilation_size())
                                    {
                                        for(int sx : get_stride_dilation_size())
                                        {
                                            int n = get_batch_size()[gen_rand_integer() % 2];
                                            int g = get_group_size()[gen_rand_integer() % 2];
                                            int k = get_channel_size()[gen_rand_integer() % 2];
                                            int dy =
                                                get_stride_dilation_size()[gen_rand_integer() % 2];
                                            int dx =
                                                get_stride_dilation_size()[gen_rand_integer() % 2];
                                            int ho = conv_out_size(hi, py, dy, fy, sy);
                                            int wo = conv_out_size(wi, px, dx, fx, sx);

                                            if(fy > hi || fx > wi || (fy - 1) < py ||
                                               (fx - 1) < px || ho <= 0 || wo <= 0 || c % g != 0 ||
                                               k % g != 0)
                                               continue;
                                            if((fx == 3 && fy == 5) || (fx == 5 && fy == 3))
                                                continue;
                                            f(n, wi, hi, c, k, fx, fy, px, py, sx, sy, dx, dy, g);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename F>
    void iterate_conv_3d(F f)
    {
        for(int c : get_channel_size())
        {
            for(int fy : get_filter_size())
            {
                for(int fx : get_filter_size())
                {
                    for(int py : get_pad_size())
                    {
                        for(int px : get_pad_size())
                        {
                            for(int sy : get_stride_dilation_size())
                            {
                                for(int sx : get_stride_dilation_size())
                                {
                                    for(int dy : get_stride_dilation_size())
                                    {
                                        for(int dx : get_stride_dilation_size())
                                        {
                                            int n   = get_batch_size()[gen_rand_integer() % 2];
                                            int g   = get_group_size()[gen_rand_integer() % 2];
                                            int k   = get_channel_size()[gen_rand_integer() % 2];
                                            int di  = get_image_depth()[gen_rand_integer() % 2];
                                            int hi  = get_image_size()[gen_rand_integer() % 2];
                                            int wi  = get_image_size()[gen_rand_integer() % 2];
                                            int fz  = get_filter_depth()[gen_rand_integer() % 2];
                                            int pz  = get_pad_depth()[gen_rand_integer() % 2];
                                            int sz  = get_stride_depth()[gen_rand_integer() % 2];
                                            int dz  = get_dilation_depth()[0];
                                            int ho  = conv_out_size(hi, py, dy, fy, sy);
                                            int wo  = conv_out_size(wi, px, dx, fx, sx);
                                            int do_ = conv_out_size(di, pz, dz, fz, sz);
                                            if(fy > hi || fx > wi || fz > di || (fy - 1) < py ||
                                               (fx - 1) < px || (fz - 1) < pz || ho <= 0 ||
                                               wo <= 0 || do_ <= 0 || c % g != 0 || k % g != 0)
                                               continue;
                                            if((fx == 3 && fy == 5) || (fx == 5 && fy == 3))
                                                continue;
                                            f(n,
                                              di,
                                              wi,
                                              hi,
                                              c,
                                              k,
                                              fz,
                                              fx,
                                              fy,
                                              pz,
                                              px,
                                              py,
                                              sz,
                                              sx,
                                              sy,
                                              dz,
                                              dx,
                                              dy,
                                              g);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void run_2d(std::string layout, std::string precision){
        auto run_conv_2d = [&](int n,
                               int wi,
                               int hi,
                               int c,
                               int k,
                               int fx,
                               int fy,
                               int px,
                               int py,
                               int sx,
                               int sy,
                               int dx,
                               int dy,
                               int g)
        {
            int ho  = conv_out_size(hi, py, dy, fy, sy);
            int wo  = conv_out_size(wi, px, dx, fx, sx);
            
            int data_byte = 4;
            if(precision == "fp16")
                data_byte = 2;

            // init host side
            float *host_input = (float *)malloc(n * c * hi * wi * sizeof(float));
            float *host_weight = (float *)malloc(g * (k/g) * (c/g) * fy * fx * sizeof(float));
            float *host_output = (float *)malloc(n * k * ho * wo * sizeof(float));

            void *device_input;
            void *device_weight;
            void *device_output;

            HIP_CALL(hipMalloc(&device_input, n * c * hi * wi * data_byte));
            HIP_CALL(hipMalloc(&device_weight, g * (k/g) * (c/g) * fy * fx * data_byte));
            HIP_CALL(hipMalloc(&device_output, n * k * ho * wo * data_byte));
            bool is_valid;

            std::cout << "n:" << n << ", c:" << c << ", hi:" << hi << ", wi:" << wi << ", k:" << k
                      << ", ho:" << ho << ", wo:" << wo << ", fy:" << fy << ",fx:" << fx
                      << ", py:" << py << ", px:" << px << ", sy:" << sy << ", sx:" << sx
                      << ", dy:" << dy << ",dx:" << dx << ", g:" << g << ", "<<layout << ", "<< precision;

            if(precision == "fp32"){
                if(layout == "nchw"){
                    // ok not to test now
                }else if(layout == "nhwc") {
                    // fwd
                    float *device_output_to_host = (float *)malloc(n * k * ho * wo * sizeof(float));
                    gen_rand_vector<float, float>(host_input, n * c * hi * wi, 0.0, 1.0);
                    gen_rand_vector<float, float>(host_weight, g * (k/g) * (c/g) * fy * fx, -0.5, 0.5);

                    naive_conv_fwd_nhwc(host_input, host_weight, host_output, n, wi, hi, c, k, fx, fy,
                                       px, py, sx, sy, dx, dy, g);

                    HIP_CALL(hipMemcpy(device_input, host_input,
                       n * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
                    HIP_CALL(hipMemcpy(device_weight, host_weight,
                            g * (k/g) * (c/g) * fy * fx * sizeof(float), hipMemcpyHostToDevice));
                    
                    gpu_naive_conv_fwd_nhwc_fp32(device_input, device_weight, device_output,
                                        n, wi, hi, c, k, fx, fy, px, py, sx, sy, dx, dy, g);
                    HIP_CALL(hipDeviceSynchronize());
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output,
                                        n * k * ho * wo * sizeof(float),
                                        hipMemcpyDeviceToHost));

                    is_valid = valid_vector(host_output, device_output_to_host,
                                            n * k * ho * wo, 1.5e-6);
                    assert(is_valid);
                    std::cout<<" fwd:"<<(is_valid?"y":"n");
                    free(device_output_to_host);

                    // bwd
                    float *device_input_to_host = (float *)malloc(n * c * hi * wi * sizeof(float));
                    gen_rand_vector<float, float>(host_output, n * k * ho * wo, 0.0, 1.0);
                    gen_rand_vector<float, float>(host_weight, g * (k/g) * (c/g) * fy * fx, -0.5, 0.5);
                    gen_rand_vector<float, float>(host_input, n * c * hi * wi, 999999., 9999999.);

                    naive_conv_bwd_nhwc(host_input, host_weight, host_output, n, wi, hi, c, k, fx, fy,
                                       px, py, sx, sy, dx, dy, g);


                    HIP_CALL(hipMemcpy(device_output, host_output,
                            n * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));
                    HIP_CALL(hipMemcpy(device_weight, host_weight,
                            g * (k/g) * (c/g) * fy * fx * sizeof(float), hipMemcpyHostToDevice));
                    gpu_naive_conv_bwd_nhwc_fp32(device_input, device_weight, device_output,
                                        n, wi, hi, c, k, fx, fy, px, py, sx, sy, dx, dy, g);
                    HIP_CALL(hipDeviceSynchronize());
                    HIP_CALL(hipMemcpy(device_input_to_host, device_input,
                                        n * c * hi * wi * sizeof(float),
                                        hipMemcpyDeviceToHost));
                    
                    is_valid = valid_vector(host_input, device_input_to_host,
                                            n * c * hi * wi, 1.5e-6);
                    assert(is_valid);
                    std::cout<<" bwd:"<<(is_valid?"y":"n");
                    free(device_input_to_host);

                    // wrw
                    float *device_weight_to_host = (float *)malloc(g * (k/g) * (c/g) * fy * fx * sizeof(float));
                    gen_rand_vector<float, float>(host_input, n * c * hi * wi, 0.0, 1.0);
                    gen_rand_vector<float, float>(host_output, n * k * ho * wo, -0.5, 0.5);

                    naive_conv_wrw_nhwc(host_input, host_weight, host_output, n, wi, hi, c, k, fx, fy,
                                       px, py, sx, sy, dx, dy, g);

                    HIP_CALL(hipMemcpy(device_input, host_input,
                            n * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
                    HIP_CALL(hipMemcpy(device_output, host_output,
                            n * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));
                    gpu_naive_conv_wrw_nhwc_fp32(device_input, device_weight, device_output,
                                        n, wi, hi, c, k, fx, fy, px, py, sx, sy, dx, dy, g);
                    HIP_CALL(hipDeviceSynchronize());
                    HIP_CALL(hipMemcpy(device_weight_to_host, device_weight,
                                        g * (k/g) * (c/g) * fy * fx * sizeof(float),
                                        hipMemcpyDeviceToHost));
                    is_valid = valid_vector(host_weight, device_weight_to_host,
                                            g * (k/g) * (c/g) * fy * fx, 1e-4);
                    assert(is_valid);
                    std::cout<<" wrw:"<<(is_valid?"y":"n");
                    free(device_weight_to_host);
                    std::cout<<std::endl;
                }
            }

            free(host_input);
            free(host_weight);
            free(host_output);

            hipFree(device_input);
            hipFree(device_weight);
            hipFree(device_output);
        };

        iterate_conv_2d(run_conv_2d);
    }

    void run_3d(std::string layout, std::string precision){
        auto run_conv_3d = [&](int n,
                               int di,
                               int wi,
                               int hi,
                               int c,
                               int k,
                               int fz,
                               int fx,
                               int fy,
                               int pz,
                               int px,
                               int py,
                               int sz,
                               int sx,
                               int sy,
                               int dz,
                               int dx,
                               int dy,
                               int g)
        {
            int do_ = conv_out_size(di, pz, dz, fz, sz);
            int ho  = conv_out_size(hi, py, dy, fy, sy);
            int wo  = conv_out_size(wi, px, dx, fx, sx);

            int data_byte = 4;
            if(precision == "fp16")
                data_byte = 2;

            // init host side
            float *host_input = (float *)malloc(n * c * di * hi * wi * sizeof(float));
            float *host_weight = (float *)malloc(g * (k/g) * (c/g) * fz * fy * fx * sizeof(float));
            float *host_output = (float *)malloc(n * k * do_ * ho * wo * sizeof(float));

            void *device_input;
            void *device_weight;
            void *device_output;

            HIP_CALL(hipMalloc(&device_input, n * c * di * hi * wi * data_byte));
            HIP_CALL(hipMalloc(&device_weight, g * (k/g) * (c/g) * fz * fy * fx * data_byte));
            HIP_CALL(hipMalloc(&device_output, n * k * do_ * ho * wo * data_byte));
            bool is_valid;

            std::cout << "n:" << n << ", c:" << c << ", di:" << di << ", hi:" << hi << ", wi:" << wi
                      << ", k:" << k << ", do:" << do_ << ", ho:" << ho << ", wo:" << wo
                      << ", fz:" << fz << ", fy:" << fy << ",fx:" << fx << ", pz:" << pz
                      << ", py:" << py << ", px:" << px << ", sz:" << sz << ", sy:" << sy
                      << ", sx:" << sx << ", dz:" << dz << ", dy:" << dy << ", dx:" << dx
                      << ", g:" << g << ", "<<layout << ", "<< precision;

            if(precision == "fp32"){
                if(layout == "ncdhw"){
                    // ok not to test now
                }else if(layout == "ndhwc") {
                    // fwd
                    float *device_output_to_host = (float *)malloc(n * k * do_ * ho * wo * sizeof(float));
                    gen_rand_vector<float, float>(host_input, n * c * di * hi * wi, 0.0, 1.0);
                    gen_rand_vector<float, float>(host_weight, g * (k/g) * (c/g) * fz * fy * fx, -0.5, 0.5);

                    naive_conv_fwd_ndhwc(host_input, host_weight, host_output, n, wi, hi, di, c, k, fx, fy, fz,
                                       px, py, pz, sx, sy, sz, dx, dy, dz, g);

                    HIP_CALL(hipMemcpy(device_input, host_input,
                       n * c * di * hi * wi * sizeof(float), hipMemcpyHostToDevice));
                    HIP_CALL(hipMemcpy(device_weight, host_weight,
                            g * (k/g) * (c/g) * fz * fy * fx * sizeof(float), hipMemcpyHostToDevice));
                    
                    gpu_naive_conv_fwd_ndhwc_fp32(device_input, device_weight, device_output,
                                        n, wi, hi, di, c, k, fx, fy, fz, px, py, pz, sx, sy, sz, dx, dy, dz, g);
                    HIP_CALL(hipDeviceSynchronize());
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output,
                                        n * k * do_ * ho * wo * sizeof(float),
                                        hipMemcpyDeviceToHost));
                    
                    is_valid = valid_vector(host_output, device_output_to_host,
                                            n * k * do_ * ho * wo, 1.5e-6);
                    std::cout<<" fwd:"<<(is_valid?"y":"n");
                    assert(is_valid);
                    free(device_output_to_host);

                    // bwd
                    float *device_input_to_host = (float *)malloc(n * c * di * hi * wi * sizeof(float));
                    gen_rand_vector<float, float>(host_output, n * k * do_ * ho * wo, 0.0, 1.0);
                    gen_rand_vector<float, float>(host_weight, g * (k/g) * (c/g) * fz * fy * fx, -0.5, 0.5);
                    gen_rand_vector<float, float>(host_input, n * c * di * hi * wi, 999999., 9999999.);

                    naive_conv_bwd_ndhwc(host_input, host_weight, host_output,
                            n, wi, hi, di, c, k, fx, fy, fz, px, py, pz, sx, sy, sz, dx, dy, dz, g);


                    HIP_CALL(hipMemcpy(device_output, host_output,
                            n * k * do_ * ho * wo * sizeof(float), hipMemcpyHostToDevice));
                    HIP_CALL(hipMemcpy(device_weight, host_weight,
                            g * (k/g) * (c/g) * fz * fy * fx * sizeof(float), hipMemcpyHostToDevice));
                    gpu_naive_conv_bwd_ndhwc_fp32(device_input, device_weight, device_output,
                                        n, wi, hi, di, c, k, fx, fy, fz, px, py, pz, sx, sy, sz, dx, dy, dz, g);
                    HIP_CALL(hipDeviceSynchronize());
                    HIP_CALL(hipMemcpy(device_input_to_host, device_input,
                                        n * c * di * hi * wi * sizeof(float),
                                        hipMemcpyDeviceToHost));
                    
                    is_valid = valid_vector(host_input, device_input_to_host,
                                            n * c * di * hi * wi, 1.5e-6);
                    std::cout<<" bwd:"<<(is_valid?"y":"n");
                    assert(is_valid);
                    free(device_input_to_host);

                    // wrw
                    float *device_weight_to_host = (float *)malloc(g * (k/g) * (c/g) * fz * fy * fx * sizeof(float));
                    gen_rand_vector<float, float>(host_input, n * c * di * hi * wi, 0.0, 1.0);
                    gen_rand_vector<float, float>(host_output, n * k * do_ * ho * wo, -0.5, 0.5);

                    naive_conv_wrw_ndhwc(host_input, host_weight, host_output,
                                        n, wi, hi, di, c, k, fx, fy, fz, px, py, pz, sx, sy, sz, dx, dy, dz, g);

                    HIP_CALL(hipMemcpy(device_input, host_input,
                            n * c * di * hi * wi * sizeof(float), hipMemcpyHostToDevice));
                    HIP_CALL(hipMemcpy(device_output, host_output,
                            n * k * do_ * ho * wo * sizeof(float), hipMemcpyHostToDevice));
                    gpu_naive_conv_wrw_ndhwc_fp32(device_input, device_weight, device_output,
                                        n, wi, hi, di, c, k, fx, fy, fz, px, py, pz, sx, sy, sz, dx, dy, dz, g);
                    HIP_CALL(hipDeviceSynchronize());
                    HIP_CALL(hipMemcpy(device_weight_to_host, device_weight,
                                        g * (k/g) * (c/g) * fz * fy * fx * sizeof(float),
                                        hipMemcpyDeviceToHost));
                    is_valid = valid_vector(host_weight, device_weight_to_host,
                                            g * (k/g) * (c/g) * fz * fy * fx, 1e-4);
                    std::cout<<" wrw:"<<(is_valid?"y":"n");
                    assert(is_valid);
                    free(device_weight_to_host);
                    std::cout<<std::endl;
                }
            }

            free(host_input);
            free(host_weight);
            free(host_output);

            hipFree(device_input);
            hipFree(device_weight);
            hipFree(device_output);
        };

        iterate_conv_3d(run_conv_3d);
    }
};

#define     GPU_NAIVE_CONV_HSACO    "naive_conv.hsaco"
int main(){
    char *gpu_naive_conv_hsaco = env_get_str("GPU_NAIVE_CONV_HSACO", GPU_NAIVE_CONV_HSACO);
    gpu_naive_conv_init(gpu_naive_conv_hsaco);
    test_conv_t test_conv;
    test_conv.run_2d("nhwc", "fp32");
    test_conv.run_3d("ndhwc", "fp32");
}