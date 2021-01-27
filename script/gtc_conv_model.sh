
#!/bin/sh
if [ $# -ne 2 ]
then 
    echo "please give this script a direction(fwd, bwd or wrw) and a precision(fp32 or fp16)"
    echo "now I use fwd and fp32 as default"
    echo "use me as: sh script/gtc_conv_model.sh fwd/bwd/wrw fp32/fp16"
    DIR=fwd
    PRECISION=fp32
else
    DIR=$1
    PRECISION=$2
fi
if [ "${PRECISION}" = "fp32" ]
then
    export IGEMM_HSACO=out/igemm_${DIR}_gtc_gfx908.hsaco
elif [ "${PRECISION}" = "fp16" ]
then
    export IGEMM_HSACO=out/igemm_${DIR}_gtc_gfx908_${PRECISION}.hsaco
else
    echo "wrong precision, I will use fp32 as default"
    export IGEMM_HSACO=out/igemm_${DIR}_gtc_gfx908.hsaco
fi
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
    echo "wrong direction, I will use fwd as default"
    FORW=1
fi

# Flag enables fp32 or fp16
if [ "${PRECISION}" = "fp32" ]
then
    CONV=conv
elif [ "${PRECISION}" = "fp16" ]
then
    CONV=convfp16
else
    echo "wrong precision, I will use fp32 as default"
    CONV=conv
fi

# only forward support gemm_k_padding
if [ $FORW = 1 ]
then
    ./out/conv_driver.exe ${CONV} -n 64 -c 3 -H 224 -W 224 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
    ./out/conv_driver.exe ${CONV} -n 128 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
    ./out/conv_driver.exe ${CONV} -n 64 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
    ./out/conv_driver.exe ${CONV} -n 64 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW

    ./out/conv_driver.exe ${CONV} -n 64 -c 1024 -H 14 -W 14 -k 1024 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe ${CONV} -n 64 -c 1024 -H 7 -W 7 -k 1024 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe ${CONV} -n 64 -c 128 -H 56 -W 56 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe ${CONV} -n 64 -c 256 -H 28 -W 28 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe ${CONV} -n 64 -c 256 -H 56 -W 56 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe ${CONV} -n 64 -c 512 -H 14 -W 14 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe ${CONV} -n 64 -c 512 -H 28 -W 28 -k 512 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 32 -F $FORW
    #exit 1

fi

# efficiency compare with hip
# resnet50
echo "resnet 50 bs==256"
./out/conv_driver.exe ${CONV} -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
if [ $FORW = 1 ]
then
    ./out/conv_driver.exe ${CONV} -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
fi

echo "resnet 50 bs==128"
./out/conv_driver.exe ${CONV} -n 128 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
if [ $FORW = 1 ]
then
    ./out/conv_driver.exe ${CONV} -n 128 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
fi

#resnext101
echo "resnext101"
./out/conv_driver.exe ${CONV} -n 64 -c 1024 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 2048 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 256 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 3 -H 224 -W 224 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 512 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW


#inception4 batch_size=128
echo "inception4 batch_size=128"
./out/conv_driver.exe ${CONV} -n 128 -c 128 -H 17 -W 17 -k 128 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 128 -H 17 -W 17 -k 128 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 128 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 128 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 1280 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 1280 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 1280 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 1280 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 160 -H 17 -W 17 -k 160 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 160 -H 17 -W 17 -k 160 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 160 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 160 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 192 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 192 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 192 -H 35 -W 35 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 192 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 192 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 2048 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 2048 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 2048 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 2048 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 256 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 256 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 288 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 288 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 288 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
if [ $FORW = 1 ]
then
    ./out/conv_driver.exe ${CONV} -n 128 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
fi
./out/conv_driver.exe ${CONV} -n 128 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 384 -H 8 -W 8 -k 384 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 384 -H 8 -W 8 -k 384 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 448 -H 8 -W 8 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 48 -H 35 -W 35 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 64 -H 73 -W 73 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 768 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 768 -H 17 -W 17 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 768 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 80 -H 73 -W 73 -k 192 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW


#inception3 batch_size=64
echo "inception3 batch_size=64"
./out/conv_driver.exe ${CONV} -n 64 -c 1024 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 1024 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 1024 -H 17 -W 17 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 1024 -H 17 -W 17 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 1536 -H 8 -W 8 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 1536 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 160 -H 73 -W 73 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 192 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 192 -H 17 -W 17 -k 224 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 192 -H 35 -W 35 -k 224 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 192 -H 71 -W 71 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 224 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 224 -H 17 -W 17 -k 256 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 224 -H 35 -W 35 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 256 -H 17 -W 17 -k 256 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 256 -H 17 -W 17 -k 320 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
if [ $FORW = 1 ]
then
    ./out/conv_driver.exe ${CONV} -n 64 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
fi
./out/conv_driver.exe ${CONV} -n 64 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 320 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 384 -H 35 -W 35 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 384 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 384 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 384 -H 35 -W 35 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 384 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 384 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 384 -H 8 -W 8 -k 448 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 448 -H 8 -W 8 -k 512 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 512 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 512 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 64 -H 147 -W 147 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 64 -H 73 -W 73 -k 64 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 64 -H 73 -W 73 -k 64 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 64 -H 73 -W 73 -k 96 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW


#resnet50
echo "resnet50 bs=64"
./out/conv_driver.exe ${CONV} -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
if [ $FORW = 1 ]
then
    ./out/conv_driver.exe ${CONV} -n 64 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
fi
./out/conv_driver.exe ${CONV} -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW

#from v4r1_origin_conv.sh
./out/conv_driver.exe ${CONV} -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 256 -H 34 -W 34 -k 256 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 128 -H 35 -W 35 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 1536 -H 8 -W 8 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 2048 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 832 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 1280 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 512 -H 14 -W 14 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 64 -c 1536 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 256 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 832 -H 7 -W 7 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 768 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 528 -H 14 -W 14 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 528 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 832 -H 7 -W 7 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 288 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 48 -H 7 -W 7 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 128 -H 17 -W 17 -k 128 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 128 -c 128 -H 17 -W 17 -k 128 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW


#mask rcnn
echo "mask rcnn"
./out/conv_driver.exe ${CONV} -n 2 -c 256 -H 12 -W 18 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 2 -c 1024 -H 34 -W 84 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 2 -c 1024 -H 40 -W 52 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 2 -c 256 -H 100 -W 104 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 2 -c 256 -H 10 -W 20 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 2 -c 64 -H 71 -W 83 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 2 -c 64 -H 59 -W 57 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 4 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 4 -c 256 -H 28 -W 28 -k 256 -y 2 -x 2 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 3 -c 256 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 1 -c 256 -H 32 -W 64 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 1 -c 64 -H 17 -W 17 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW


#retina net bs=16
echo "retina net bs=16"
./out/conv_driver.exe ${CONV} -n 16 -c 256 -H 12 -W 12 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 16 -c 256 -H 134 -W 77 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
./out/conv_driver.exe ${CONV} -n 16 -c 256 -H 71 -W 101 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
