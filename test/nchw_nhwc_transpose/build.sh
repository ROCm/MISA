#!/bin/sh
# to launch from top of generator
ARCH=gfx908
rm -rf out
mkdir out

/opt/rocm/hip/bin/hipcc --cuda-host-only -DUSE_MAGIC_DIV -Idriver -std=c++14 -lpthread test/nchw_nhwc_transpose/nchw_nhwc_transpose.cpp -o out/nchw_nhwc_transpose.exe || exit 1
/opt/rocm/hip/bin/hipcc -x hip --cuda-gpu-arch=$ARCH --cuda-device-only -gline-tables-only  -c -O3 driver/gpu_batched_transpose/batched_transpose.cpp -o out/batched_transpose.hsaco || exit 1
# /opt/rocm/llvm/bin/llvm-objdump --disassemble --mcpu=$ARCH  out/nchw_nhwc_transpose.hsaco > out/nchw_nhwc_transpose.hsaco.dump.s
/opt/rocm/llvm/bin/clang-offload-bundler --type=o  --targets=hipv4-amdgcn-amd-amdhsa--$ARCH  --inputs=out/batched_transpose.hsaco --outputs=out/batched_transpose.hsaco.o --unbundle
/opt/rocm/llvm/bin/llvm-objdump --disassemble --mcpu=$ARCH  out/batched_transpose.hsaco.o > out/batched_transpose.dump.s
/opt/rocm/llvm/bin/llvm-readobj --notes --elf-output-style=LLVM  out/batched_transpose.hsaco.o > out/batched_transpose.dump.metadata