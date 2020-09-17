
#!/bin/sh
export IGEMM_HSACO=out/igemm_fwd_gtc_gfx908_all_ordered.hsaco
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=0

# Flag enables fwd, bwd, wrw convolutions
FORW=1
Verify=1

set -x 

#resnext101
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 1024 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW -V $Verify 
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 1024 -H 28 -W 28 -k 1024 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 2048 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 2048 -H 7 -W 7 -k 2048 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 3 -H 224 -W 224 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 512 -H 56 -W 56 -k 512 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW -V $Verify

#inception4 batch_size=128 
./out/conv_driver.exe conv -n 128 -c 128 -H 17 -W 17 -k 128 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 128 -H 17 -W 17 -k 128 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 128 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 128 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 1280 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 1280 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 1280 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 1280 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 160 -H 17 -W 17 -k 160 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 160 -H 17 -W 17 -k 160 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 160 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 160 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 192 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 192 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 192 -H 35 -W 35 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 192 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 192 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 2048 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 2048 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 2048 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 2048 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 256 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 256 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 288 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 288 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 288 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify

./out/conv_driver.exe conv -n 128 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 384 -H 8 -W 8 -k 384 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 384 -H 8 -W 8 -k 384 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 448 -H 8 -W 8 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 48 -H 35 -W 35 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 64 -H 73 -W 73 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 768 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 768 -H 17 -W 17 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 768 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 80 -H 73 -W 73 -k 192 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 128 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify

./out/conv_driver.exe conv -n 128 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW -V $Verify


#inception3 batch_size=64 
./out/conv_driver.exe conv -n 64 -c 1024 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 1024 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 1024 -H 17 -W 17 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 1024 -H 17 -W 17 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 1536 -H 8 -W 8 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 1536 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 160 -H 73 -W 73 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 192 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 192 -H 17 -W 17 -k 224 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 192 -H 35 -W 35 -k 224 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 192 -H 71 -W 71 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 224 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 224 -H 17 -W 17 -k 256 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 224 -H 35 -W 35 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 256 -H 17 -W 17 -k 256 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 256 -H 17 -W 17 -k 320 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify

./out/conv_driver.exe conv -n 64 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 320 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 384 -H 35 -W 35 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 384 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 384 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 384 -H 35 -W 35 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 384 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 384 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 384 -H 8 -W 8 -k 448 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 448 -H 8 -W 8 -k 512 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 512 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 512 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 64 -H 147 -W 147 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 64 -H 73 -W 73 -k 64 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 64 -H 73 -W 73 -k 64 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 64 -H 73 -W 73 -k 96 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW -V $Verify

./out/conv_driver.exe conv -n 64 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW -V $Verify

#resnet50 
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 64 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify

./out/conv_driver.exe conv -n 64 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify

./out/conv_driver.exe conv -n 32 -c 1024 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 1024 -H 14 -W 14 -k 1024 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 1024 -H 28 -W 28 -k 1024 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 2048 -H 14 -W 14 -k 2048 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 2048 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 2048 -H 7 -W 7 -k 2048 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 256 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 256 -H 56 -W 56 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify

./out/conv_driver.exe conv -n 32 -c 3 -H 224 -W 224 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify

./out/conv_driver.exe conv -n 32 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 512 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 512 -H 28 -W 28 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 512 -H 56 -W 56 -k 512 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver.exe conv -n 32 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify

