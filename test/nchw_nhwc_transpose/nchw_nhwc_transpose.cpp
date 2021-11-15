
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
#include "gpu_nchw_nhwc_transpose.h"

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
struct distribution_t<int8_t>{
    distribution_t(int min, int max) : distribution(min, max) {}
    template<class URNG>
    int8_t operator()(URNG & rng){
        int value = distribution(rng);
        return *reinterpret_cast<int8_t*>(&value);
        //return 0xf;
    }
    std::uniform_int_distribution<int> distribution;
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
    distribution_t<Dst_T> distribution(min,max);
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

static inline bool valid_vector_binary(int8_t *ref, int8_t *pred, size_t bytes) {
    int igemm_per_pixel_check = env_get_int("PER_PIXEL_CHECK", 0);
    size_t err = 0;
    for(size_t i = 0; i < bytes ; i++){
        // {
        //     uint32_t r = 0;
        //     uint32_t p = 0;
        //     memcpy(reinterpret_cast<void*>(&r), reinterpret_cast<void*>(&ref[i]), 1);
        //     memcpy(reinterpret_cast<void*>(&p), reinterpret_cast<void*>(&pred[i]), 1);
        //     printf("%7d, ref:0x%x, pred:0x%x, %s\n", i, r, p, r==p?"y":"n");
        // }
        if(ref[i] != pred[i]){
            err ++;
            if(igemm_per_pixel_check){
                uint32_t r = 0;
                uint32_t p = 0;
                memcpy(reinterpret_cast<void*>(&r), reinterpret_cast<void*>(&ref[i]), 1);
                memcpy(reinterpret_cast<void*>(&p), reinterpret_cast<void*>(&pred[i]), 1);
                printf("fail at %d, ref:0x%x, pred:0x%x\n", i, r, p);
            }
        }
    }
    return err == 0;
}

template<typename T>
void cpu_nchw2nhwc(T * dst, T * src, uint64_t N, uint64_t C, uint64_t H, uint64_t W)
{
    for(uint64_t i_n = 0; i_n < N; i_n++){
        for(uint64_t i_h = 0; i_h < H; i_h++){
            for(uint64_t i_w = 0; i_w < W; i_w++){
                for(uint64_t i_c =0 ; i_c < C; i_c++){
                    uint64_t idx_nhwc = i_n * H * W * C + i_h * W * C + i_w * C + i_c;
                    uint64_t idx_nchw = i_n * C * H * W + i_c * H * W + i_h * W + i_w;
                    dst[idx_nhwc] = src[idx_nchw];
                }
            }
        }
    }
}

template<typename T>
void cpu_nhwc2nchw(T * dst, T * src, uint64_t N, uint64_t C, uint64_t H, uint64_t W)
{
    for(uint64_t i_n = 0; i_n < N; i_n++){
        for(uint64_t i_c =0 ; i_c < C; i_c++){
            for(uint64_t i_h = 0; i_h < H; i_h++){
                for(uint64_t i_w = 0; i_w < W; i_w++){
                    uint64_t idx_nhwc = i_n * H * W * C + i_h * W * C + i_w * C + i_c;
                    uint64_t idx_nchw = i_n * C * H * W + i_c * H * W + i_h * W + i_w;
                    dst[idx_nchw] = src[idx_nhwc];
                }
            }
        }
    }
}

#define WARMUP 3
#define REPEAT 7
#define     TRANSPOSE_HSACO    "batched_transpose.hsaco"

int main(int argc, char ** argv){
    if(argc < 5){
        printf("%s N C H W\n", argv[0]);
        return -1;
    }
    int warmup = env_get_int("IGEMM_WARMUP", WARMUP);
    int repeat = env_get_int("IGEMM_REPEAT", REPEAT);

    uint64_t N = std::stoull(std::string(argv[1]));
    uint64_t C = std::stoull(std::string(argv[2]));
    uint64_t H = std::stoull(std::string(argv[3]));
    uint64_t W = std::stoull(std::string(argv[4]));

    size_t size_byte = 4;
    char * fp = env_get_str("FP", "32");
    std::string fp_str(fp);
    if(fp_str == "32")
        size_byte = 4;
    else if(fp_str == "16")
        size_byte = 2;
    else if(fp_str == "8")
        size_byte = 1;
    else{
        printf("error FP:%s\n", fp);
        return -1;
    }

    char * hsaco = env_get_str("TRANSPOSE_HSACO", TRANSPOSE_HSACO);
    gpu_nhwc_nchw_transpose_init(hsaco);

    void * src_cpu = malloc(N*C*H*W*size_byte);
    void * dst_cpu = malloc(N*C*H*W*size_byte);
    void * dst_gpu_valid = malloc(N*C*H*W*size_byte);

    void * src_gpu;
    void * dst_gpu;

    HIP_CALL(hipMalloc(&src_gpu, N*C*H*W*size_byte));
    HIP_CALL(hipMalloc(&dst_gpu, N*C*H*W*size_byte));

    gen_rand_vector<int8_t>(reinterpret_cast<int8_t*>(src_cpu), N*C*H*W*size_byte, -116, 121);

    HIP_CALL(hipMemcpy(src_gpu, src_cpu, N*C*H*W*size_byte, hipMemcpyHostToDevice));


    auto launch_gpu_nchw2nhwc = [&](const transpose_kernel_param_t * kparam){
        if(fp_str == "32")
            gpu_nchw2nhwc<float>(reinterpret_cast<float*>(dst_gpu), reinterpret_cast<float*>(src_gpu), N, C, H, W, kparam);
        else if(fp_str == "16")
            gpu_nchw2nhwc<ushort>(reinterpret_cast<ushort*>(dst_gpu), reinterpret_cast<ushort*>(src_gpu), N, C, H, W, kparam);
        else if(fp_str == "8")
            gpu_nchw2nhwc<int8_t>(reinterpret_cast<int8_t*>(dst_gpu), reinterpret_cast<int8_t*>(src_gpu), N, C, H, W, kparam);
    };

    auto launch_gpu_nhwc2nchw = [&](const transpose_kernel_param_t * kparam){
        if(fp_str == "32")
            gpu_nhwc2nchw<float>(reinterpret_cast<float*>(dst_gpu), reinterpret_cast<float*>(src_gpu), N, C, H, W, kparam);
        else if(fp_str == "16")
            gpu_nhwc2nchw<ushort>(reinterpret_cast<ushort*>(dst_gpu), reinterpret_cast<ushort*>(src_gpu), N, C, H, W, kparam);
        else if(fp_str == "8")
            gpu_nhwc2nchw<int8_t>(reinterpret_cast<int8_t*>(dst_gpu), reinterpret_cast<int8_t*>(src_gpu), N, C, H, W, kparam);
    };

    auto launch_cpu_nchw2nhwc = [&](){
        if(fp_str == "32")
            cpu_nchw2nhwc<float>(reinterpret_cast<float*>(dst_cpu), reinterpret_cast<float*>(src_cpu), N, C, H, W);
        else if(fp_str == "16")
            cpu_nchw2nhwc<ushort>(reinterpret_cast<ushort*>(dst_cpu), reinterpret_cast<ushort*>(src_cpu), N, C, H, W);
        else if(fp_str == "8")
            cpu_nchw2nhwc<int8_t>(reinterpret_cast<int8_t*>(dst_cpu), reinterpret_cast<int8_t*>(src_cpu), N, C, H, W);
    };

    auto launch_cpu_nhwc2nchw = [&](){
        if(fp_str == "32")
            cpu_nhwc2nchw<float>(reinterpret_cast<float*>(dst_cpu), reinterpret_cast<float*>(src_cpu), N, C, H, W);
        else if(fp_str == "16")
            cpu_nhwc2nchw<ushort>(reinterpret_cast<ushort*>(dst_cpu), reinterpret_cast<ushort*>(src_cpu), N, C, H, W);
        else if(fp_str == "8")
            cpu_nhwc2nchw<int8_t>(reinterpret_cast<int8_t*>(dst_cpu), reinterpret_cast<int8_t*>(src_cpu), N, C, H, W);
    };

    auto test_nchw2nhwc = [&](const transpose_kernel_param_t *transpose_kparam){
        // nchw2nhwc
        float kernel_time = 0;
        bool valid = false;

        bool is_kernel_valid = gpu_nchw2nhwc_is_kernel_valid(N, C, H, W, transpose_kparam);

        if(is_kernel_valid){
            hipEvent_t start, stop;
            HIP_CALL(hipMemset(dst_gpu, 0, N*C*H*W*size_byte));

            for(int i=0; i< warmup; i++){
                launch_gpu_nchw2nhwc(transpose_kparam);
            }

            HIP_CALL(hipEventCreate(&start));
            HIP_CALL(hipEventCreate(&stop));
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipEventRecord(start, 0) );

            for(int i=0; i< repeat; i++){
                launch_gpu_nchw2nhwc(transpose_kparam);
            }
            
            HIP_CALL(hipEventRecord(stop, 0) );
            HIP_CALL(hipEventSynchronize(stop) );
            HIP_CALL(hipEventElapsedTime(&kernel_time, start, stop) );
            HIP_CALL(hipEventDestroy(start) );
            HIP_CALL(hipEventDestroy(stop) );

            kernel_time = kernel_time / repeat;

            launch_cpu_nchw2nhwc();

            HIP_CALL(hipMemcpy(dst_gpu_valid, dst_gpu, N*C*H*W*size_byte, hipMemcpyDeviceToHost));

            valid = valid_vector_binary(reinterpret_cast<int8_t*>(dst_cpu), reinterpret_cast<int8_t*>(dst_gpu_valid), N*C*H*W*size_byte);
        }

        double flop_cnt = 2 * N*C*H*W*size_byte;
        double bw = is_kernel_valid ? flop_cnt / kernel_time / 1e6 : 0;

        printf("[nchw2nhwc fp%s] N:%llu, C:%llu, H:%llu, W:%llu, flop:%.0f, time:%fms, bw:%.4fGB/s, valid:%s (%dx%d, %dx%d, %dx%d)\n",
            fp_str.c_str(), N, C, H, W, flop_cnt, kernel_time, bw, is_kernel_valid ? (valid ? "y" : "n") : "x",
            transpose_kparam->tile_x, transpose_kparam->tile_y, transpose_kparam->pack_x, transpose_kparam->pack_y, transpose_kparam->ediv_x, transpose_kparam->ediv_y);
        fflush(stdout);

        return valid && is_kernel_valid ? kernel_time : FLT_MAX;
    };

    auto test_nhwc2nchw = [&](const transpose_kernel_param_t *transpose_kparam){
        // nhwc2nchw
        float kernel_time = 0;
        bool valid = false;

        bool is_kernel_valid = gpu_nhwc2nchw_is_kernel_valid(N, C, H, W, transpose_kparam);

        if(is_kernel_valid){
            hipEvent_t start, stop;
            HIP_CALL(hipMemset(dst_gpu, 0, N*C*H*W*size_byte));

            for(int i=0; i< warmup; i++){
                launch_gpu_nhwc2nchw(transpose_kparam);
            }

            HIP_CALL(hipEventCreate(&start));
            HIP_CALL(hipEventCreate(&stop));
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipEventRecord(start, 0) );

            for(int i=0; i< repeat; i++){
                launch_gpu_nhwc2nchw(transpose_kparam);
            }

            HIP_CALL(hipEventRecord(stop, 0) );
            HIP_CALL(hipEventSynchronize(stop) );
            HIP_CALL(hipEventElapsedTime(&kernel_time, start, stop) );
            HIP_CALL(hipEventDestroy(start) );
            HIP_CALL(hipEventDestroy(stop) );

            kernel_time = kernel_time / repeat;

            launch_cpu_nhwc2nchw();

            HIP_CALL(hipMemcpy(dst_gpu_valid, dst_gpu, N*C*H*W*size_byte, hipMemcpyDeviceToHost));

            valid = valid_vector_binary(reinterpret_cast<int8_t*>(dst_cpu), reinterpret_cast<int8_t*>(dst_gpu_valid), N*C*H*W*size_byte);
        }

        double flop_cnt = 2 * N*C*H*W*size_byte;
        double bw = is_kernel_valid ? flop_cnt / kernel_time / 1e6 : 0;

        printf("[nhwc2nchw fp%s] N:%llu, C:%llu, H:%llu, W:%llu, flop:%.0f, time:%fms, bw:%.4fGB/s, valid:%s (%dx%d, %dx%d, %dx%d)\n",
            fp_str.c_str(), N, C, H, W, flop_cnt, kernel_time, bw, is_kernel_valid ? (valid ? "y" : "n") : "x",
            transpose_kparam->tile_x, transpose_kparam->tile_y, transpose_kparam->pack_x, transpose_kparam->pack_y, transpose_kparam->ediv_x, transpose_kparam->ediv_y);
        fflush(stdout);

        return valid && is_kernel_valid ? kernel_time : FLT_MAX;
    };

    auto get_transpose_all_kernel = [&](){
        if(fp_str == "32")
            return transpose_kernel_get_all_param_t<4>::get();
        else if(fp_str == "16")
            return transpose_kernel_get_all_param_t<2>::get();
        else if(fp_str == "8")
            return transpose_kernel_get_all_param_t<1>::get();
        else
            assert(false);
    };


    float min_nchw2nhwc_time = FLT_MAX;
    transpose_kernel_param_t min_nchw2nhwc_kparam;
    for(auto kparam : get_transpose_all_kernel()){
        float current_time = test_nchw2nhwc(&kparam);
        if(current_time < min_nchw2nhwc_time){
            min_nchw2nhwc_time = current_time;
            min_nchw2nhwc_kparam = kparam;
        }
    }
    printf("                    -> min time:%fms, kparam: %dx%d, %dx%d, %dx%d\n", min_nchw2nhwc_time,
        min_nchw2nhwc_kparam.tile_x, min_nchw2nhwc_kparam.tile_y, min_nchw2nhwc_kparam.pack_x, min_nchw2nhwc_kparam.pack_y, min_nchw2nhwc_kparam.ediv_x, min_nchw2nhwc_kparam.ediv_y);
    fflush(stdout);
    printf("-------------------------\n");

    float min_nhwc2nchw_time = FLT_MAX;
    transpose_kernel_param_t min_nhwc2nchw_kparam;
    for(auto kparam : get_transpose_all_kernel()){
        float current_time = test_nhwc2nchw(&kparam);
        if(current_time < min_nhwc2nchw_time){
            min_nhwc2nchw_time = current_time;
            min_nhwc2nchw_kparam = kparam;
        }
    }
    printf("                    -> min time:%fms, kparam: %dx%d, %dx%d, %dx%d\n", min_nhwc2nchw_time,
        min_nhwc2nchw_kparam.tile_x, min_nhwc2nchw_kparam.tile_y, min_nhwc2nchw_kparam.pack_x, min_nhwc2nchw_kparam.pack_y, min_nhwc2nchw_kparam.ediv_x, min_nhwc2nchw_kparam.ediv_y);
    fflush(stdout);

    free(src_cpu);
    free(dst_cpu);
    free(dst_gpu_valid);
    hipFree(src_gpu);
    hipFree(dst_gpu);
}
