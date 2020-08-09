#!/bin/sh

# host compilation
rm out/conv_driver.exe
rm out/igemm_v4r1_dynamic.hsaco
g++ -D__HIP_PLATFORM_HCC__=1 -I/opt/rocm/hip/include -I/opt/rocm/hcc/include \
-I/opt/rocm/hsa/include -Wall -O2 -std=c++11 \
driver/conv_driver.cpp -L/opt/rocm/hcc/lib -L/opt/rocm/lib \
-L/opt/rocm/lib64 -Wl,-rpath=/opt/rocm/hcc/lib:/opt/rocm/lib \
-ldl -lm -lpthread -Wl,--whole-archive -lhip_hcc \
-lhsa-runtime64 -lhsakmt -Wl,--no-whole-archive -o out/conv_driver.exe

#-DIGEMM_CONFIG_FILE="//home//shaowang//Desktop//learning//iGEMMGen_v4r1_1x1//config//igemm_v4r1_dynamic.config" \
#-DIGEMM_HSACO="igemm_v4r1_dynamic.hsaco" \

# device compilation
arch=gfx908
/opt/rocm-3.5.0/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=$arch out/igemm_v4r1_dynamic_xdlops.s -o out/igemm_v4r1_dynamic.hsaco

/opt/rocm-3.5.0/llvm/bin/clang++  -std=c++14 -mcpu=$arch -Werror -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic \
-Wno-conversion -Wno-double-promotion -Wno-exit-time-destructors -Wno-extra-semi -Wno-float-conversion \
-Wno-gnu-anonymous-struct -Wno-gnu-zero-variadic-macro-arguments -Wno-missing-prototypes -Wno-nested-anon-types \
-Wno-padded -Wno-return-std-move-in-c++11 -Wno-shorten-64-to-32 -Wno-sign-conversion -Wno-unknown-warning-option \
-Wno-unused-command-line-argument -Wno-weak-vtables -Wno-covered-switch-default -Wno-disabled-macro-expansion \
-Wno-undefined-reinterpret-cast --cuda-gpu-arch=$arch --cuda-device-only -c -O3  -Wno-unused-command-line-argument \
-I. -x hip --hip-device-lib-path=/opt/rocm/lib -mllvm -amdgpu-early-inline-all=true -mllvm \
-amdgpu-function-calls=false -D__HIP_ROCclr__=1 -isystem /opt/rocm-3.5.0/hip/../include \
-isystem /opt/rocm/llvm/lib/clang/11.0.0/include/.. -D__HIP_PLATFORM_HCC__=1 -D__HIP_ROCclr__=1 \
-isystem /opt/rocm-3.5.0/hip/include -isystem /opt/rocm/include --hip-device-lib-path=/opt/rocm/lib \
--hip-link -mllvm -amdgpu-enable-global-sgpr-addr -mllvm --amdgpu-spill-vgpr-to-agpr=0 \
out/wrw_reduction_hip.hip.cc -o out/wrw_reduction_hip.hip.cc.o -Wno-old-style-cast -Wno-cast-align

/opt/rocm-3.5.0/llvm/bin/clang-offload-bundler --type=o --targets=hip-amdgcn-amd-amdhsa-$arch \
--inputs=out/wrw_reduction_hip.hip.cc.o --outputs=out/wrw_reduction_hip.hip.cc.o.hsaco --unbundle