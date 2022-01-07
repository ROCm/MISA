
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

// #define NAIVE_CONV_THREADED
#include "naive_conv.h"
#include "naive_tiled_conv.h"

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

static inline bool valid_vector_exact(const float *ref, const float *pred, int n)
{
    const uint32_t * d_r = reinterpret_cast<const uint32_t*>(ref);
    const uint32_t * d_p = reinterpret_cast<const uint32_t*>(pred);
    uint32_t err_cnt = 0;

    for(int i=0; i<n; i++){
        if(d_r[i] != d_p[i] && err_cnt < 100){
            printf("diff at %4d, r:%u(%f), p:%u(%f)\n", i, d_r[i], ref[i], d_p[i], pred[i]);
            err_cnt ++;
        }
    }
    return err_cnt == 0;
}

#define VAR_ENV(v, e) v = env_get_int(e, v)

class test_conv_t{
public:
    static int conv_out_size(int in_size, int pad, int dilation, int ksize, int stride)
    {
        return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
    }

    static std::vector<int> get_image_depth() { return {8, 10}; }

    static std::vector<int> get_image_size() { return {1, 3, 7, 16}; }

    static std::vector<int> get_channel_size() { return {3, 4, 8}; }

    static std::vector<int> get_filter_depth() { return {1, 3, 5}; }

    static std::vector<int> get_filter_size() { return {1, 3, 4, 5}; }

    static std::vector<int> get_stride_depth() { return {1, 2}; }

    static std::vector<int> get_dilation_depth() { return {1}; }

    static std::vector<int> get_stride_dilation_size() { return {1, 2, 3}; }

    static std::vector<int> get_pad_depth() { return {0, 1}; }

    static std::vector<int> get_pad_size() { return {0, 1, 2}; }

    static std::vector<int> get_group_size() { return {1, 2}; }

    static std::vector<int> get_batch_size() { return {1, 2}; }

    template <typename F>
    void iterate_conv_2d(F f)
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
                                        int c = get_channel_size()[gen_rand_integer() % 3];
                                        int n = get_batch_size()[gen_rand_integer() % 2];
                                        int g = get_group_size()[gen_rand_integer() % 2];
                                        int k = get_channel_size()[gen_rand_integer() % 3];
                                        int dy =
                                            get_stride_dilation_size()[gen_rand_integer() % 3];
                                        int dx =
                                            get_stride_dilation_size()[gen_rand_integer() % 3];

                                        c = env_get_int("C", c);
                                        hi = env_get_int("HI", hi);
                                        wi = env_get_int("WI", wi);
                                        fy = env_get_int("FY", fy);

                                        VAR_ENV(c,  "C");
                                        VAR_ENV(hi, "HI");
                                        VAR_ENV(wi, "WI");
                                        VAR_ENV(fy, "FY");
                                        VAR_ENV(fx, "FX");
                                        VAR_ENV(py, "PY");
                                        VAR_ENV(px, "PX");
                                        VAR_ENV(sy, "SY");
                                        VAR_ENV(sx, "SX");
                                        VAR_ENV(n,  "N");
                                        VAR_ENV(g,  "G");
                                        VAR_ENV(k,  "K");
                                        VAR_ENV(dy, "DY");
                                        VAR_ENV(dx, "DX");

                                        int ho = conv_out_size(hi, py, dy, fy, sy);
                                        int wo = conv_out_size(wi, px, dx, fx, sx);

                                        if(fy > hi || fx > wi || (fy - 1) < py ||
                                            (fx - 1) < px || ho <= 0 || wo <= 0 || c % g != 0 ||
                                            k % g != 0)
                                            continue;

                                        // conv_out_size, denorminator should >= 0
                                        if((hi + 2 * py) < (dy * (fy - 1) + 1) || (wi + 2 * px) < (dx * (fx - 1) + 1))
                                            continue;

                                        if(fx == 4 || fy == 4){
                                            if(gen_rand_integer() % 2 == 0)
                                                continue;   // randomly ignore some case
                                        }

                                        int ty = gen_rand_integer() % ho + 1;
                                        int tx = gen_rand_integer() % wo + 1;

                                        VAR_ENV(ty, "TY");
                                        VAR_ENV(tx, "TX");

                                        f(n, wi, hi, c, k, fx, fy, px, py, sx, sy, dx, dy, g, tx, ty);

                                        int single_run = env_get_int("SRUN", 0);
                                        if(single_run)
                                            return ;
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
                               int g,
                               int tx,
                               int ty)
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

            bool is_valid;

            std::cout << "n:" << n << ", c:" << c << ", hi:" << hi << ", wi:" << wi << ", k:" << k
                      << ", ho:" << ho << ", wo:" << wo << ", fy:" << fy << ", fx:" << fx
                      << ", py:" << py << ", px:" << px << ", sy:" << sy << ", sx:" << sx
                      << ", dy:" << dy << ", dx:" << dx << ", g:" << g << ", "<<layout << ", "<< precision;

            std::cout << ", tx:"<<tx<<", ty:"<<ty<<","<<std::flush;

            if(precision == "fp32"){
                if(layout == "nchw"){
                    float *host_output_tiled = (float *)malloc(n * k * ho * wo * sizeof(float));
                    memset(host_output_tiled, 0, n * k * ho * wo * sizeof(float));
                    memset(host_output, 0, n * k * ho * wo * sizeof(float));
                    //gen_rand_vector<float, float>(host_input, n * c * hi * wi, 0.0, 1.0);
                    //gen_rand_vector<float, float>(host_weight, g * (k/g) * (c/g) * fy * fx, -0.5, 0.5);

                    gen_rand_vector<float, int>(host_input, n * c * hi * wi, -5, 5);
                    gen_rand_vector<float, int>(host_weight, g * (k/g) * (c/g) * fy * fx, -5, 5);

                    naive_conv_fwd_nchw(host_input, host_weight, host_output, n, wi, hi, c, k, fx, fy,
                                       px, py, sx, sy, dx, dy, g);
                    naive_tiled_conv_fwd_nchw(host_input, host_weight, host_output_tiled, n, wi, hi, c, k, fx, fy,
                                       px, py, sx, sy, dx, dy, g, tx, ty);

                    is_valid = valid_vector_exact(host_output, host_output_tiled, n * k * ho * wo);

                    std::cout<<" valid:"<<(is_valid?"y":"n");
                    std::cout<<std::endl;
                    std::cout << std::flush;
                    free(host_output_tiled);
                    assert(is_valid);
                }else if(layout == "nhwc") {
                   
                    
                }
            }

            free(host_input);
            free(host_weight);
            free(host_output);
        };

        iterate_conv_2d(run_conv_2d);
    }
};

int main(){
    test_conv_t test_conv;
    test_conv.run_2d("nchw", "fp32");
}