#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C"
__global__ __launch_bounds__(256,2)
void tensor_cast_fp16_fp32_1d(half* output, float* input, int thread_length, int total_length)
{
    float in_data;
    half out_data;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    int offset = bid * thread_length * 256 + tid * thread_length;

    if(offset + thread_length > total_length){
        for(int i = offset; i < total_length; i++){
            in_data = input[i];
            out_data = (half)(in_data);
            *(output + i) = out_data;
        }
    }
    else{
        for(int i = 0; i < thread_length; i++){
            in_data = input[offset + i];
            out_data = (half)(in_data);
            *(output + offset + i) = out_data;
        }
    }
}