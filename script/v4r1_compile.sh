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
/opt/rocm-3.5.0/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx900 out/igemm_v4r1_dynamic.s -o out/igemm_v4r1_dynamic.hsaco
#/home/shaowang/llvm_local/bin/clang -x assembler -target amdgcn--amdhsa -mcpu=gfx900 out/igemm_v4r1_dynamic.s -o out/igemm_v4r1_dynamic.hsaco