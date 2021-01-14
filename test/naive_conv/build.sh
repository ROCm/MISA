#!/bin/sh
# to launch from top of generator
ARCH=gfx908
rm -rf out
mkdir out

/opt/rocm/hip/bin/hipcc -Idriver -std=c++14 -lpthread test/naive_conv/test_naive_conv.cpp -o out/test_naive_conv.exe || exit 1
/opt/rocm/hip/bin/hipcc -x hip --cuda-gpu-arch=$ARCH --cuda-device-only -c -O3 driver/gpu_naive_conv/naive_conv.cpp -o out/naive_conv.hsaco


