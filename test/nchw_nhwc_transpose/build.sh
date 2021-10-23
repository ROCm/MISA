#!/bin/sh
# to launch from top of generator
ARCH=gfx908
rm -rf out
mkdir out

/opt/rocm/hip/bin/hipcc --cuda-host-only -DUSE_MAGIC_DIV -Idriver -std=c++14 -lpthread test/nchw_nhwc_transpose/nchw_nhwc_transpose.cpp -o out/nchw_nhwc_transpose.exe || exit 1
/opt/rocm/hip/bin/hipcc -x hip --cuda-gpu-arch=$ARCH --cuda-device-only -gline-tables-only -save-temps=out/ -c -O3 driver/gpu_nchw_nhwc_transpose/nchw_nhwc_transpose.cpp -o out/nchw_nhwc_transpose.hsaco || exit 1
# /opt/rocm/llvm/bin/llvm-objdump --disassemble --mcpu=$ARCH  out/nchw_nhwc_transpose.hsaco > out/nchw_nhwc_transpose.hsaco.dump.s
