#!/bin/sh
# run this at top directory
export HSACO=./out/igemm_fwd_btm_nhwc_fp16.hsaco
export GPU_NAIVE_CONV_HSACO=./out/naive_conv.hsaco
export IGEMM_LOG_FASTEST_CONFIG=1
./out/test_inference.exe convfp16 -t 1 -n 1 -c 8 -H 1080 -W 1920 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convfp16 -t 1 -n 1 -c 16 -H 540 -W 960 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convfp16 -t 1 -n 1 -c 16 -H 270 -W 480 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convfp16 -t 1 -n 1 -c 16 -H 135 -W 240 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convfp16 -t 1 -n 1 -c 16 -H 270 -W 480 -k 16 -y 1 -x 1 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convfp16 -t 1 -n 1 -c 16 -H 270 -W 480 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convfp16 -t 1 -n 1 -c 16 -H 540 -W 960 -k 16 -y 1 -x 1 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convfp16 -t 1 -n 1 -c 16 -H 540 -W 960 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convfp16 -t 1 -n 1 -c 16 -H 1080 -W 1920 -k 16 -y 1 -x 1 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convfp16 -t 1 -n 1 -c 16 -H 1080 -W 1920 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convfp16 -t 1 -n 1 -c 16 -H 1080 -W 1920 -k 4 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1

export HSACO=./out/igemm_fwd_btm_nhwc_int8.hsaco
./out/test_inference.exe convint8 -t 1 -n 1 -c 8 -H 1080 -W 1920 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convint8 -t 1 -n 1 -c 16 -H 540 -W 960 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convint8 -t 1 -n 1 -c 16 -H 270 -W 480 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convint8 -t 1 -n 1 -c 16 -H 135 -W 240 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convint8 -t 1 -n 1 -c 16 -H 270 -W 480 -k 16 -y 1 -x 1 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convint8 -t 1 -n 1 -c 16 -H 270 -W 480 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convint8 -t 1 -n 1 -c 16 -H 540 -W 960 -k 16 -y 1 -x 1 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convint8 -t 1 -n 1 -c 16 -H 540 -W 960 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convint8 -t 1 -n 1 -c 16 -H 1080 -W 1920 -k 16 -y 1 -x 1 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convint8 -t 1 -n 1 -c 16 -H 1080 -W 1920 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1
./out/test_inference.exe convint8 -t 1 -n 1 -c 16 -H 1080 -W 1920 -k 4 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1

