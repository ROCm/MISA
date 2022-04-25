#!/bin/sh
# to launch from top of generator
DIR=test/persistent_workgroup/
ARCH=gfx1030
rm -rf out
mkdir out
USE_PERSISTENT_WORKGROUP=1

/opt/rocm/hip/bin/hipcc -DUSE_PERSISTENT_WORKGROUP=$USE_PERSISTENT_WORKGROUP --amdgpu-target=$ARCH -Idriver -std=c++14 -lpthread $DIR/test_persistent_workgroup.cpp -o out/test_persistent_workgroup.exe || exit 1
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=$ARCH -Wa,-defsym,USE_PERSISTENT_WORKGROUP=$USE_PERSISTENT_WORKGROUP -I$DIR/kernel/ $DIR/kernel/igemm_fwd_gtcn2_nchwc_kcyxc_fp16x8.s -o out/igemm_fwd_gtcn2_nchwc_kcyxc_fp16x8.hsaco || exit 1
/opt/rocm/llvm/bin/llvm-objdump --disassemble --mcpu=$ARCH  out/igemm_fwd_gtcn2_nchwc_kcyxc_fp16x8.hsaco > out/igemm_fwd_gtcn2_nchwc_kcyxc_fp16x8.dump.s
/opt/rocm/hip/bin/hipcc -x hip --cuda-gpu-arch=$ARCH --cuda-device-only -c -O3 driver/gpu_naive_conv/naive_conv.cpp -o out/naive_conv.hsaco
