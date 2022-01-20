#!/bin/sh
# to launch from top of generator
ARCH=gfx1030
rm -rf out
mkdir out

/opt/rocm/hip/bin/hipcc --cuda-host-only -DUSE_MAGIC_DIV -Idriver -std=c++20 -lpthread test/tensor_reorder/tensor_reorder.cpp -o out/tensor_reorder.exe || exit 1
/opt/rocm/hip/bin/hipcc  -std=c++14 -x hip --cuda-gpu-arch=$ARCH --cuda-device-only -gline-tables-only  -c -O3 driver/gpu_batched_transpose/batched_transpose.cpp -o out/batched_transpose.hsaco || exit 1
# /opt/rocm/llvm/bin/llvm-objdump --disassemble --mcpu=$ARCH  out/tensor_reorder.hsaco > out/tensor_reorder.hsaco.dump.s
/opt/rocm/llvm/bin/clang-offload-bundler --type=o  --targets=hipv4-amdgcn-amd-amdhsa--$ARCH  --inputs=out/batched_transpose.hsaco --outputs=out/batched_transpose.hsaco.o --unbundle
/opt/rocm/llvm/bin/llvm-objdump --disassemble --mcpu=$ARCH  out/batched_transpose.hsaco.o > out/batched_transpose.dump.s
/opt/rocm/llvm/bin/llvm-readobj --notes --elf-output-style=LLVM  out/batched_transpose.hsaco.o > out/batched_transpose.dump.metadata

./out/tensor_reorder.exe  5 25 1023 1025
