#!/bin/sh
# run this at top directory
export HSACO=./out/igemm_fwd_gtcn2_nchwc_kcyxc_fp16x8.hsaco
export GPU_NAIVE_CONV_HSACO=./out/naive_conv.hsaco
# export IGEMM_LOG_FASTEST_CONFIG=1
./out/test_persistent_workgroup.exe  convfp16x8 -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NCHWC --fil_layout NCHWC --out_layout NCHWC -g 1 -F 1 -t 1  
./out/test_persistent_workgroup.exe  convfp16x8 -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 --in_layout NCHWC --fil_layout NCHWC --out_layout NCHWC -g 1 -F 1 -t 1
# ./out/test_persistent_workgroup.exe  convfp16x8 -n 128 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 --in_layout NCHWC --fil_layout NCHWC --out_layout NCHWC -g 1 -F 1 -t 1
