
#!/bin/sh
if [ $# -ne 1 ]
then 
    echo "please give this script a direction"
    echo "now I use bwd as default"
    DIR=bwd
else
    DIR=$1
fi
export IGEMM_HSACO=out/igemm_${DIR}_gtc_gfx908.hsaco
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=1

# Flag enables fwd, bwd, wrw convolutions
if [ "${DIR}" = "fwd" ]
then
    FORW=1
elif [ "${DIR}" = "bwd" ]
then
    FORW=2
elif [ "${DIR}" = "wrw" ]
then
    FORW=4
else
    echo "wrong direction"
    exit 1
fi

# only forward support gemm_k_padding
if [ $FORW = 1 ]
then
    ./out/conv_driver.exe conv -n 64 -c 3 -H 224 -W 224 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
    ./out/conv_driver.exe conv -n 128 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
    ./out/conv_driver.exe conv -n 64 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
    ./out/conv_driver.exe conv -n 64 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW

    ./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 1024 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe conv -n 64 -c 1024 -H 7 -W 7 -k 1024 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe conv -n 64 -c 128 -H 56 -W 56 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe conv -n 64 -c 256 -H 28 -W 28 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe conv -n 64 -c 512 -H 14 -W 14 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 512 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 32 -F $FORW
    #exit 1

fi

#resnext101
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 2048 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
# ./out/conv_driver.exe conv -n 64 -c 3 -H 224 -W 224 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW


#inception4 batch_size=128
./out/conv_driver.exe conv -n 128 -c 128 -H 17 -W 17 -k 128 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 128 -H 17 -W 17 -k 128 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 128 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 128 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 1280 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 1280 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 1280 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 1280 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 160 -H 17 -W 17 -k 160 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 160 -H 17 -W 17 -k 160 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 160 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 160 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 192 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 192 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 192 -H 35 -W 35 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 192 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 192 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 2048 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 2048 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 2048 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 2048 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 256 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 256 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 288 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 288 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 288 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
# ./out/conv_driver.exe conv -n 128 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 384 -H 8 -W 8 -k 384 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 384 -H 8 -W 8 -k 384 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 448 -H 8 -W 8 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 48 -H 35 -W 35 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 64 -H 73 -W 73 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 768 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 768 -H 17 -W 17 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 768 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 80 -H 73 -W 73 -k 192 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 128 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW


#inception3 batch_size=64
./out/conv_driver.exe conv -n 64 -c 1024 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 1024 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 1024 -H 17 -W 17 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 1024 -H 17 -W 17 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 1536 -H 8 -W 8 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 1536 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 160 -H 73 -W 73 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 192 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 192 -H 17 -W 17 -k 224 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 192 -H 35 -W 35 -k 224 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 192 -H 71 -W 71 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 224 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 224 -H 17 -W 17 -k 256 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 224 -H 35 -W 35 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 256 -H 17 -W 17 -k 256 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 256 -H 17 -W 17 -k 320 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
# ./out/conv_driver.exe conv -n 64 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 320 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 384 -H 35 -W 35 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 384 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 384 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 384 -H 35 -W 35 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 384 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 384 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 384 -H 8 -W 8 -k 448 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 448 -H 8 -W 8 -k 512 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 512 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 512 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 64 -H 147 -W 147 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 64 -H 73 -W 73 -k 64 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 64 -H 73 -W 73 -k 64 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 64 -H 73 -W 73 -k 96 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe conv -n 64 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW


#resnet50
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
# ./out/conv_driver.exe conv -n 64 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW

#from v4r1_origin_conv.sh
./out/conv_driver.exe conv -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 256 -H 34 -W 34 -k 256 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 128 -H 35 -W 35 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 1536 -H 8 -W 8 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 2048 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 832 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 1280 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 512 -H 14 -W 14 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 64 -c 1536 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 256 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 832 -H 7 -W 7 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 768 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 528 -H 14 -W 14 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 528 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 832 -H 7 -W 7 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 288 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 48 -H 7 -W 7 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 128 -H 17 -W 17 -k 128 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe conv -n 128 -c 128 -H 17 -W 17 -k 128 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW


#mask rcnn
./out/conv_driver.exe conv -n 2 -c 256 -H 12 -W 18 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 2 -c 1024 -H 34 -W 84 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 2 -c 1024 -H 40 -W 52 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 2 -c 256 -H 100 -W 104 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 2 -c 256 -H 10 -W 20 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 2 -c 64 -H 71 -W 83 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 2 -c 64 -H 59 -W 57 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 4 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 4 -c 256 -H 28 -W 28 -k 256 -y 2 -x 2 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 3 -c 256 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 1 -c 256 -H 32 -W 64 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 1 -c 64 -H 17 -W 17 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW


#retina net bs=16
./out/conv_driver.exe conv -n 16 -c 256 -H 12 -W 12 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 16 -c 256 -H 134 -W 77 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe conv -n 16 -c 256 -H 71 -W 101 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
