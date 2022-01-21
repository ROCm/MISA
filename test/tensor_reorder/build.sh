#!/bin/sh
# to launch from top of generator
ARCH=gfx1030
rm -rf out
mkdir out

/opt/rocm/hip/bin/hipcc --cuda-host-only -DUSE_MAGIC_DIV -Idriver -std=c++17 -lpthread test/tensor_reorder/tensor_reorder.cpp -o out/tensor_reorder.exe || exit 1
/opt/rocm/hip/bin/hipcc  -std=c++17 -x hip --cuda-gpu-arch=$ARCH --cuda-device-only -gline-tables-only  -c -O3 driver/gpu_batched_transpose/batched_transpose.cpp -o out/batched_transpose.hsaco || exit 1
/opt/rocm/hip/bin/hipcc  -std=c++17 -x hip --cuda-gpu-arch=$ARCH --cuda-device-only -gline-tables-only  -c -O3 driver/gpu_tensor_reorder/general_tensor_reorder.cpp -o out/general_tensor_reorder.hsaco || exit 1

/opt/rocm/llvm/bin/clang-offload-bundler --type=o  --targets=hipv4-amdgcn-amd-amdhsa--$ARCH  --inputs=out/batched_transpose.hsaco --outputs=out/batched_transpose.hsaco.o --unbundle
/opt/rocm/llvm/bin/llvm-objdump --disassemble --mcpu=$ARCH  out/batched_transpose.hsaco.o > out/batched_transpose.dump.s
/opt/rocm/llvm/bin/llvm-readobj --notes --elf-output-style=LLVM  out/batched_transpose.hsaco.o > out/batched_transpose.dump.metadata
/opt/rocm/llvm/bin/clang-offload-bundler --type=o  --targets=hipv4-amdgcn-amd-amdhsa--$ARCH  --inputs=out/general_tensor_reorder.hsaco --outputs=out/general_tensor_reorder.hsaco.o --unbundle
/opt/rocm/llvm/bin/llvm-objdump --disassemble --mcpu=$ARCH  out/general_tensor_reorder.hsaco.o > out/general_tensor_reorder.dump.s
/opt/rocm/llvm/bin/llvm-readobj --notes --elf-output-style=LLVM  out/general_tensor_reorder.hsaco.o > out/general_tensor_reorder.dump.metadata


#Completeness & Correctness test
export FP=32
./out/tensor_reorder.exe  4 32 1024 1024
./out/tensor_reorder.exe  4 32 127 129
./out/tensor_reorder.exe  32 129 224 224
./out/tensor_reorder.exe  1024 1024 3 3
./out/tensor_reorder.exe  256 257 16 32

export FP=16
./out/tensor_reorder.exe  4 32 1024 1024
./out/tensor_reorder.exe  4 32 127 129
./out/tensor_reorder.exe  32 129 224 224
./out/tensor_reorder.exe  1024 1024 3 3
./out/tensor_reorder.exe  256 257 16 32

export FP=8
./out/tensor_reorder.exe  4 32 1024 1024
./out/tensor_reorder.exe  4 32 127 129
./out/tensor_reorder.exe  32 129 224 224
./out/tensor_reorder.exe  1024 1024 3 3
./out/tensor_reorder.exe  256 257 16 32
