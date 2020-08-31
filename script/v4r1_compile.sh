#!/bin/sh

# host compilation
rm out/conv_driver.exe
rm out/igemm_bwd_gtc.hsaco
g++ -D__HIP_PLATFORM_HCC__=1 -I/opt/rocm/hip/include -I/opt/rocm/hcc/include -I/opt/rocm/hsa/include -Wall \
-O2 -std=c++11 -I/opt/intel/inteloneapi/oneDNN/latest/cpu_gomp//include -DUSE_XDNN \
-DIGEMM_CONFIG_FILE="\"/dockerx/MIOpen_task/wrw_igemmgen_gtc/config/igemm_bwd_gtc.config\"" -DIGEMM_HSACO="\"igemm_bwd_gtc.hsaco\"" \
driver/conv_driver.cpp -L/opt/rocm/lib -L/opt/rocm/lib64 -Wl,-rpath=/opt/rocm/lib -ldl -lm -lpthread -Wl,--whole-archive \
-lamdhip64 -lhsa-runtime64 -lhsakmt -Wl,--no-whole-archive -L/opt/intel/inteloneapi/oneDNN/latest/cpu_gomp//lib -ldnnl \
-Wno-sign-compare -Wno-unused-variable -Wno-write-strings -Wno-unused-function \
-Wl,-rpath=/opt/intel/inteloneapi/oneDNN/latest/cpu_gomp//lib -o out/conv_driver.exe

# device compilation
arch=gfx908
/opt/rocm-3.5.0/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=$arch out/igemm_bwd_gtc.s -o out/igemm_bwd_gtc.hsaco

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