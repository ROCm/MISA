#!/bin/sh
# to launch from top of generator
ARCH=gfx908
rm -rf out
mkdir out

AS_CLAGS="-Wa,-defsym,activ_mode=0" # PASTHRU = 0      
#AS_CLAGS="-Wa,-defsym,activ_mode=1" # LOGISTIC = 1     
#AS_CLAGS="-Wa,-defsym,activ_mode=2" # TANH = 2         
#AS_CLAGS="-Wa,-defsym,activ_mode=3" # RELU = 3         
#AS_CLAGS="-Wa,-defsym,activ_mode=4" # SOFTRELU = 4     
#AS_CLAGS="-Wa,-defsym,activ_mode=5" # ABS = 5          
#AS_CLAGS="-Wa,-defsym,activ_mode=6" # POWER = 6        
#AS_CLAGS="-Wa,-defsym,activ_mode=7" # CLIPPED_RELU = 7 
#AS_CLAGS="-Wa,-defsym,activ_mode=8" # LEAKY_RELU = 8   
#AS_CLAGS="-Wa,-defsym,activ_mode=9" # ELU = 9          

/opt/rocm/hip/bin/hipcc --amdgpu-target=$ARCH -Idriver -std=c++14 -lpthread test/fwd_cnhwc_test/test_driver.cpp -o out/test_driver.exe || exit 1
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=$ARCH -mcumode $AS_CLAGS -Itest/fwd_cnhwc_test/kernel/ test/fwd_cnhwc_test/kernel/igemm_fwd_cnhwc_gfx908_fp16.asm -o out/igemm_fwd_cnhwc_gfx908_fp16.hsaco || exit 1
/opt/rocm/hip/bin/hipcc -x hip --cuda-gpu-arch=$ARCH --cuda-device-only -c -O3 driver/gpu_naive_conv/naive_conv.cpp -o out/naive_conv.hsaco




