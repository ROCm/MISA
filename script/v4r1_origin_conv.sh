#!/bin/bash -x
set -v
export IGEMM_HSACO=out/igemm_wrw_gtc_gfx908.hsaco
#./out/conv_driver.exe conv -n 128 -c 128 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4 
#./out/conv_driver.exe conv -n 128 -c 128 -H 16 -W 16 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4 
#./out/conv_driver.exe conv -n 2 -c 256 -H 100 -W 104 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4 -V 1
#./out/conv_driver.exe conv -n 2 -c 256 -H 10 -W 20 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
#./out/conv_driver.exe conv -n 2 -c 256 -H 12 -W 18 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
#./out/conv_driver.exe conv -n 2 -c 256 -H 13 -W 10 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
#./out/conv_driver.exe conv -n 2 -c 1024 -H 34 -W 84 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
#./out/conv_driver.exe conv -n 2 -c 2048 -H 21 -W 42 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
#./out/conv_driver.exe conv -n 2 -c 1024 -H 40 -W 52 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
#./out/conv_driver.exe conv -n 2 -c 1024 -H 42 -W 68 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
#./out/conv_driver.exe conv -n 64 -c 256 -H 16 -W 16 -k 256 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4 -V 1
./out/conv_driver.exe conv -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F 4
#./out/conv_driver.exe conv -n 64 -c 256 -H 34 -W 34 -k 256 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 128 -H 35 -W 35 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 64 -c 1536 -H 8 -W 8 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 2048 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 832 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 1280 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 512 -H 14 -W 14 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 64 -c 1536 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 256 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 832 -H 7 -W 7 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 768 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 528 -H 14 -W 14 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 528 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 832 -H 7 -W 7 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 288 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 48 -H 7 -W 7 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 128 -H 17 -W 17 -k 128 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -F 4
#./out/conv_driver.exe conv -n 128 -c 128 -H 17 -W 17 -k 128 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4