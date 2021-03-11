#!/bin/sh
# to launch from top of generator
ARCH=gfx1030
rm -rf out
mkdir out

/opt/rocm/hip/bin/hipcc -Idriver -std=c++14 -lpthread test/inference/test_inference.cpp -o out/test_inference.exe || exit 1
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=$ARCH -mcumode -Itest/inference/kernel/ test/inference/kernel/igemm_fwd_btm_nhwc_fp16.s -o out/igemm_fwd_btm_nhwc_fp16.hsaco || exit 1
/opt/rocm/hip/bin/hipcc -x hip --cuda-gpu-arch=$ARCH --cuda-device-only -c -O3 driver/gpu_naive_conv/naive_conv.cpp -o out/naive_conv.hsaco




