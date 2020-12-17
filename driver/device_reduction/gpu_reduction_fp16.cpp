#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C"
__global__ __launch_bounds__(256,2)
void wrw_reduction_fp16(half* output, half* input, int out_length, int in_stride, int n_groups)
{
    half2 vec_in;
    half2 vec_out;
    int i_len, i_groups;
    
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    int offset = bid * out_length * 256 + tid * out_length;
    
    half* local_in = input + offset;
    half* local_out = output + offset;
    
    for (i_len = 0; i_len < out_length; i_len += 2)
    {
        vec_out = (half2)0;
        for (i_groups = 0; i_groups < n_groups; i_groups++)
        {
            vec_in = *(half2* )(local_in + i_len + in_stride * i_groups);
            vec_out += vec_in;
        }
        *(half2 *)(local_out + i_len) = vec_out;
    }
}
