#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C"
__global__ __launch_bounds__(256,2)
void tensor_cast_fp16_fp32_1d(half* output, float* input, int total_length)
{
    constexpr auto unroll_length = 8;
    float vec_in_data[unroll_length];
    half vec_out_data[unroll_length];
    float *tmp_in;
    half *tmp_out;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int block_size = blockDim.x;

    int offset = bid * unroll_length * 256;
    int block_end = offset + unroll_length * 256; 

    if(block_end <= total_length)
    {
        tmp_in = input + offset + tid;
        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            vec_in_data[i] = *(tmp_in);
            tmp_in += 1 * 256;
        }

        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            vec_out_data[i] = (half)(vec_in_data[i]);
        }
        
        tmp_out = output + offset + tid * 1;
        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            *(tmp_out) = vec_out_data[i];
            tmp_out += 1 * 256;
        }
    }
    else
    {
        float in_data;
        half out_data;
        for(int i = offset; i < total_length; i += 256)
        {
            int index = min(i + tid, total_length - 1);
            in_data = input[index];
            out_data = (half)(in_data);
            *(output + index) = out_data;
        }
    }
}