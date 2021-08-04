#!/bin/sh
# run this at top directory
export HSACO=./out/igemm_fwd_cnhwc_gfx908_fp16.hsaco
export GPU_NAIVE_CONV_HSACO=./out/naive_conv.hsaco
export IGEMM_LOG_FASTEST_CONFIG=1
./out/test_driver.exe convfp16 -t 1 -n 16 -c 4096 -H 16 -W 16 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 1
