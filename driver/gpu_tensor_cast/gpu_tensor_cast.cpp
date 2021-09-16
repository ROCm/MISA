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

    int offset = bid * thread_length * 256;
    int block_end = offset + thread_length * 256; 

    if(block_end <= total_length)
    {
        for(int i = offset; i < block_end; i += 256)
        {
            in_data = input[i + tid];
            out_data = (half)(in_data);
            *(output + i + tid) = out_data;
        }
    }
    else
    {
        for(int i = offset; i < total_length; i += 256)
        {
            int index = min(i + tid, total_length - 1);
            in_data = input[index];
            out_data = (half)(in_data);
            *(output + index) = out_data;
        }
    }
}