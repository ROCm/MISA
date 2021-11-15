#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define MIOPEN_USE_RNE_BFLOAT16

typedef union _cvt_bf16_fp32
{
    uint u32;
    ushort2 ushortx2;
    ushort ushortvec[2];
    float f32;
} _cvt_bf16_fp32_t;

inline __device__ __host__ ushort __float_to_bfloat16(float src_val)
{
    _cvt_bf16_fp32_t target_val;
    target_val.f32 = src_val;

    if((~target_val.u32 & 0x7f800000) == 0) // Inf or NaN
    {
        if((target_val.u32 & 0xffff) != 0)
        {
            target_val.u32 |= 0x10000; // Preserve signaling NaN
        }
    }
    else
    {
#ifdef MIOPEN_USE_RNE_BFLOAT16
        target_val.u32 += (0x7fff + (target_val.ushortvec[1] & 1));
#endif // MIOPEN_USE_RNE_BFLOAT16
    }
    return target_val.ushortvec[1];
}

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

extern "C"
__global__ __launch_bounds__(256,2)
void tensor_cast_bf16_fp32_1d(ushort* output, float* input, int total_length)
{
    constexpr auto unroll_length = 8;
    float vec_in_data[unroll_length];
    ushort vec_out_data[unroll_length];
    float *tmp_in;
    ushort *tmp_out;

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
            vec_out_data[i] = __float_to_bfloat16(vec_in_data[i]);
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
        ushort out_data;
        for(int i = offset; i < total_length; i += 256)
        {
            int index = min(i + tid, total_length - 1);
            in_data = input[index];
            out_data = __float_to_bfloat16(in_data);
            *(output + index) = out_data;
        }
    }
}
