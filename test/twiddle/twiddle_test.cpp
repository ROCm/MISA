#include "utility.h"
#include "fft.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

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

static inline char *env_get_str(const char *var_name, char *default_str) {
    char *v = getenv(var_name);
    if (v)
        return v;
    return default_str;
}


#ifndef HSACO
#define HSACO "twiddle.hsaco"
#endif

typedef struct{
    float * in;
    float * out;
}twiddle_fft_karg_t;

int main(int argc, char ** argv)
{
    char *hsaco = env_get_str("HSACO", HSACO);
    int fft_length = 8;
    std::string direction = "fwd";
    if(argc > 1){
        fft_length = atoi(argv[1]);
        direction = std::string(argv[2]);
        assert(direction == "fwd" || direction == "bwd");
    }

    float * sequence_origin = new float[fft_length * 2];
    float * sequence        = new float[fft_length * 2];
    float * sequence_host   = new float[fft_length * 2];
    float * sequence_in_dev;
    float * sequence_out_dev;

    rand_vec(sequence_origin, 2*fft_length);
    
    for(int i=0; i<(fft_length * 2); i++)
        sequence[i] = sequence_origin[i];

    HIP_CALL(hipMalloc(&sequence_in_dev, fft_length * 2 * sizeof(float)));
    HIP_CALL(hipMalloc(&sequence_out_dev, fft_length * 2 * sizeof(float)));

    HIP_CALL(hipMemcpy(sequence_in_dev, sequence,
                       fft_length * 2 * sizeof(float), hipMemcpyHostToDevice));

    hipModule_t module;
    HIP_CALL(hipModuleLoad(&module, hsaco));

    std::string kernel_name = std::string("twiddle_fft") + std::to_string(fft_length) + "_" + direction;
    hipFunction_t kernel_func;
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));

    twiddle_fft_karg_t karg;
    karg.in = sequence_in_dev;
    karg.out = sequence_out_dev;
    size_t karg_size = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size = 1;

    HIP_CALL(hipHccModuleLaunchKernel(kernel_func, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));
    HIP_CALL(hipMemcpy(sequence_host, sequence_out_dev, fft_length * 2 * sizeof(float), hipMemcpyDeviceToHost));

    {
        // the gpu fft operation result in brev order, we need to order back to do check
        std::vector<size_t> brev_list;
        bit_reverse_permute(log2(fft_length), brev_list);
        float * tmp = new float[fft_length * 2];
        for(int i=0; i<(fft_length * 2); i++) tmp[i] = sequence_host[i];
        for(int i=0; i<fft_length; i++){
            // brev back
            sequence_host[2 * i]        = tmp[2 * brev_list[i]];
            sequence_host[2 * i + 1]    = tmp[2 * brev_list[i] + 1];
            if(direction == "bwd"){
                sequence_host[2 * i]        /= fft_length;
                sequence_host[2 * i + 1]    /= fft_length;
            }
        }
        delete [] tmp;
    }

    // CPU run
    if(direction == "fwd")
        fft_cooley_tukey_r(sequence, fft_length);
    else
        ifft_cooley_tukey_r(sequence, fft_length);

    int err_cnt = valid_vector(sequence, sequence_host, fft_length * 2);
    int rtn_cnt = 0;

    printf("[%s] valid:%s\n", kernel_name.c_str(), err_cnt==0?"y":"n");
    if(err_cnt != 0){
        dump_vector_as_py_array(sequence_origin, 2*fft_length);
        rtn_cnt = -1;
    }
    delete [] sequence_origin;
    delete [] sequence;
    delete [] sequence_host;
    hipFree(sequence_in_dev);
    hipFree(sequence_out_dev);
    return rtn_cnt;
}
