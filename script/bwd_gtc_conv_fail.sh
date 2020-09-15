#!/bin/sh
export IGEMM_HSACO=out/igemm_bwd_gtc_gfx908.hsaco
./out/conv_driver.exe conv -n 64 -c 256 -H 34 -W 34 -k 256 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1
#./out/conv_driver.exe conv -n 128 -c 1024 -H 8 -W 8 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 2


