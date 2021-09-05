
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
export TEST_KERNEL=igemm_${DIR}_gtc_gfx908_nhwc_fp16_test
rm out/conv_driver.exe
rm out/${TEST_KERNEL}.hsaco

/opt/rocm/hip/bin/hipcc -DUSE_MAGIC_DIV=1 \
-DIGEMM_CONFIG_FILE="\"/dockerx/MIOpen/igemm_codegen/igemm_codegen_fp16/igemmgen_nhwc_fp32_wrw/config/${TEST_KERNEL}.config\"" \
-DIGEMM_HSACO="\"${TEST_KERNEL}.hsaco\"" -DIGEMM_WRW_USE_ATOMIC_ADD=1 \
-Wno-sign-compare -Wno-unused-variable -Wno-write-strings -Wno-unused-function -Wno-deprecated-declarations \
-Wno-return-type -Wno-uninitialized -Wno-non-c-typedef-for-linkage -Wno-format -std=c++14 \
-DUSE_GPU_NAIVE_CONV  -DUSE_HALF_HPP -DUSE_HALF driver/conv_driver.cpp driver/perf/gmap.cpp -I driver/ -o out/conv_driver.exe

# device compilation
arch=gfx908
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=$arch -Iout out/${TEST_KERNEL}.s -o out/${TEST_KERNEL}.hsaco

#/opt/rocm-3.5.0/llvm/bin/clang++  -std=c++14 -mcpu=$arch -Werror -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic \
#-Wno-conversion -Wno-double-promotion -Wno-exit-time-destructors -Wno-extra-semi -Wno-float-conversion \
#-Wno-gnu-anonymous-struct -Wno-gnu-zero-variadic-macro-arguments -Wno-missing-prototypes -Wno-nested-anon-types \
#-Wno-padded -Wno-return-std-move-in-c++11 -Wno-shorten-64-to-32 -Wno-sign-conversion -Wno-unknown-warning-option \
#-Wno-unused-command-line-argument -Wno-weak-vtables -Wno-covered-switch-default -Wno-disabled-macro-expansion \
#-Wno-undefined-reinterpret-cast --cuda-gpu-arch=$arch --cuda-device-only -c -O3  -Wno-unused-command-line-argument \
#-I. -x hip --hip-device-lib-path=/opt/rocm/lib -mllvm -amdgpu-early-inline-all=true -mllvm \
#-amdgpu-function-calls=false -D__HIP_ROCclr__=1 -isystem /opt/rocm-3.5.0/hip/../include \
#-isystem /opt/rocm/llvm/lib/clang/11.0.0/include/.. -D__HIP_PLATFORM_HCC__=1 -D__HIP_ROCclr__=1 \
#-isystem /opt/rocm-3.5.0/hip/include -isystem /opt/rocm/include --hip-device-lib-path=/opt/rocm/lib \
#--hip-link -mllvm -amdgpu-enable-global-sgpr-addr -mllvm --amdgpu-spill-vgpr-to-agpr=0 \
#out/wrw_reduction_hip.hip.cc -o out/wrw_reduction_hip.hip.cc.o -Wno-old-style-cast -Wno-cast-align
#
#/opt/rocm-3.5.0/llvm/bin/clang-offload-bundler --type=o --targets=hip-amdgcn-amd-amdhsa-$arch \
#--inputs=out/wrw_reduction_hip.hip.cc.o --outputs=out/wrw_reduction_hip.hip.cc.o.hsaco --unbundle