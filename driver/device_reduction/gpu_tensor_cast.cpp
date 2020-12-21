#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C"
__global__ __launch_bounds__(256,2)
void tensor_cast_fp16_fp32_1d(half* output, float* input, int length)
{
    float in_data;
    half out_data;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    int offset = bid * length * 256 + tid * length;

    for(int i = 0; i < length; i++){
        in_data = input[offset + i];
        out_data = (half)(in_data);
        *(output + offset + i) = out_data;
    }
}