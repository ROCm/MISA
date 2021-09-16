#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C"
__global__ __launch_bounds__(256,2)
void tensor_cast_fp16_fp32_1d(half* output, float* input, int total_length)
{
    float vec_in_data[8];
    half vec_out_data[8];
    float *tmp_in;
    half *tmp_out;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    int offset = bid * 8 * 256;
    int block_end = offset + 8 * 256; 

    if(block_end <= total_length)
    {
        tmp_in = input + offset + tid;
        vec_in_data[0] = *(tmp_in);
        tmp_in += 1 * 256;
        vec_in_data[1] = *(tmp_in);
        tmp_in += 1 * 256;
        vec_in_data[2] = *(tmp_in);
        tmp_in += 1 * 256;
        vec_in_data[3] = *(tmp_in);
        tmp_in += 1 * 256;
        vec_in_data[4] = *(tmp_in);
        tmp_in += 1 * 256;
        vec_in_data[5] = *(tmp_in);
        tmp_in += 1 * 256;
        vec_in_data[6] = *(tmp_in);
        tmp_in += 1 * 256;
        vec_in_data[7] = *(tmp_in);

        vec_out_data[0] = (half)(vec_in_data[0]);
        vec_out_data[1] = (half)(vec_in_data[1]);
        vec_out_data[2] = (half)(vec_in_data[2]);
        vec_out_data[3] = (half)(vec_in_data[3]);
        vec_out_data[4] = (half)(vec_in_data[4]);
        vec_out_data[5] = (half)(vec_in_data[5]);
        vec_out_data[6] = (half)(vec_in_data[6]);
        vec_out_data[7] = (half)(vec_in_data[7]);
        
        tmp_out = output + offset + tid * 1;
        *(tmp_out) = vec_out_data[0];
        tmp_out += 1 * 256;
        *(tmp_out) = vec_out_data[1];
        tmp_out += 1 * 256;
        *(tmp_out) = vec_out_data[2];
        tmp_out += 1 * 256;
        *(tmp_out) = vec_out_data[3];
        tmp_out += 1 * 256;
        *(tmp_out) = vec_out_data[4];
        tmp_out += 1 * 256;
        *(tmp_out) = vec_out_data[5];
        tmp_out += 1 * 256;
        *(tmp_out) = vec_out_data[6];
        tmp_out += 1 * 256;
        *(tmp_out) = vec_out_data[7];
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