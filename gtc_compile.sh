#!/bin/sh
if [ $# -ne 1 ]
then 
    echo "please give this script a direction"
    echo "now I use bwd as default"
    DIR=fwd
else
    DIR=$1
fi

# Flag enables fwd, bwd, wrw convolutions
if [ "${DIR}" = "fwd" ]
then
    FORW=1
elif [ "${DIR}" = "bwd" ]
then
    FORW=2
elif [ "${DIR}" = "wrw" ]
then
    FORW=4
else
    echo "wrong direction"
    exit 1
fi

# host compilation
export TEST_KERNEL=igemm_${DIR}_gtc_gfx1030_nchwc_fp16
rm out/conv_driver.exe
rm out/${TEST_KERNEL}.hsaco

/opt/rocm/hip/bin/hipcc -DUSE_MAGIC_DIV=1 \
-DIGEMM_CONFIG_FILE="\"./config/${TEST_KERNEL}.config\"" \
-DIGEMM_HSACO="\"${TEST_KERNEL}.hsaco\"" -DIGEMM_WRW_USE_ATOMIC_ADD=1 \
-Wno-sign-compare -Wno-unused-variable -Wno-write-strings -Wno-unused-function -Wno-deprecated-declarations \
-Wno-return-type -Wno-uninitialized -Wno-non-c-typedef-for-linkage -Wno-format -std=c++14 \
-DUSE_GPU_NAIVE_CONV  -DUSE_HALF_HPP -DUSE_HALF driver/conv_driver.cpp driver/perf/gmap.cpp -I driver/ -o out/conv_driver.exe

# device compilation
arch=gfx1030
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=$arch -Iout out/${TEST_KERNEL}.s -o out/${TEST_KERNEL}.hsaco
/opt/rocm/hip/bin/hipcc -x hip --cuda-gpu-arch=$arch --cuda-device-only -c -O3 driver/gpu_naive_conv/naive_conv.cpp -o out/naive_conv.hsaco

