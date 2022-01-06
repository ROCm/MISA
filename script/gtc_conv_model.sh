
#!/bin/sh
if [ $# -ge 1 ] ; then
    DIR=$1
else
    DIR=bwd
fi

if [ $# -ge 2 ] ; then
    LAYOUT=$2
else
    LAYOUT="nchw"
fi

if [ $# -ge 3 ] ; then
    PREC=$3
else
    PREC="fp32"
fi

if [ $# -ge 4 ] ; then
    ARCH=$4
else
    ARCH="gfx908"
fi

if [ "${LAYOUT}" = "nchw" ] ; then
    LAYOUT_HSACO=""
    LAYOUT_ARG=""
elif [ "${LAYOUT}" = "nhwc" ] ; then
    LAYOUT_HSACO="_nhwc"
    LAYOUT_ARG="--in_layout NHWC --fil_layout NHWC --out_layout NHWC"
elif [ "${LAYOUT}" = "nchwc" ] ; then
    LAYOUT_HSACO="_nchwc"
    LAYOUT_ARG="--in_layout NCHWC --fil_layout CHWNC --out_layout NCHWC"
else
    echo "wrong layout: ${LAYOUT}"
    exit 1
fi

if [ "${PREC}" = "fp32" ] ; then
    PREC_HSACO=""
    CONV="conv"
elif [ "${PREC}" = "fp16" ] ; then
    PREC_HSACO="_fp16"
    CONV="convfp16"
elif [ "${PREC}" = "int8" ] ; then
    PREC_HSACO="_int8"
    CONV="convint8"
elif [ "${PREC}" = "bf16" ] ; then
    PREC_HSACO="_bf16"
    CONV="convbfp16"
else
    echo "wrong precision: ${PREC}"
    exit 1
fi

if [ "${ARCH}" != "gfx90a" ] && [ "${ARCH}" != "gfx908" ] && [ "${ARCH}" != "gfx1030" ] ; then
    echo "wrong arch: ${ARCH}"
    exit 1
fi

echo IGEMM_HSACO=out/igemm_${DIR}_gtc_${ARCH}${LAYOUT_HSACO}${PREC_HSACO}.hsaco
export IGEMM_HSACO=out/igemm_${DIR}_gtc_${ARCH}${LAYOUT_HSACO}${PREC_HSACO}.hsaco
export IGEMM_TENSOR_CAST_HSACO=out/igemm_gtc_tensor_cast.hsaco
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_SLEEP_MS=117
export PER_PIXEL_CHECK=0

export DBG_MODE=0
export IGEMM_ASSERT_WHEN_INVALID=1
export IGEMM_WARMUP=1
export IGEMM_REPEAT=4
export IGEMM_GKS_ITERATIVE=1
export IGEMM_BENCH_CSV=1

# Flag enables fwd, bwd, wrw convolutions
if [ "${DIR}" = "fwd" ] ; then
    FORW=1
elif [ "${DIR}" = "bwd" ] ; then
    FORW=2
elif [ "${DIR}" = "wrw" ] ; then
    FORW=4
else
    echo "wrong direction"
    exit 1
fi

#./out/conv_driver.exe $CONV -n 1024 -c 1 -H 512 -W 1024 -k 1 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F ${FORW} ${LAYOUT_ARG}
#./out/conv_driver.exe $CONV -n 4096 -c 1 -H 512 -W 1024 -k 1 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F ${FORW} ${LAYOUT_ARG}

#./out/conv_driver.exe $CONV -n 64 -c 1024 -H 14 -W 14 -k 1024 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 32 -F ${FORW} ${LAYOUT_ARG}

#./out/conv_driver.exe $CONV -n 64 -c 128 -H 56 -W 56 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F $FORW ${LAYOUT_ARG}

#./out/conv_driver.exe $CONV -n 64 -c 1024 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 4 -F $FORW ${LAYOUT_ARG}

#./out/conv_driver.exe ${CONV} -n 1 -c 3 -H 32 -W 32 -k 1 -y 11 -x 11 -p 1 -q 1 -u 2 -v 2 -l 2 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}

#./out/conv_driver.exe ${CONV} -n 400 -c 256 -H 7 -W 7 -k 1024 -y 7 -x 7 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
#./out/conv_driver.exe ${CONV} -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}

#./out/conv_driver.exe ${CONV} -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}

#./out/conv_driver.exe ${CONV} -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}


#./out/conv_driver.exe ${CONV} -n 16 -c 4096 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
#./out/conv_driver.exe ${CONV} -n 256 -c 4096 -H 14 -W 14 -k 8192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}

#exit 1
# only forward support gemm_k_padding
#if [ $FORW = 1 ]
if [ 0 = 1 ] ; then
    ./out/conv_driver.exe $CONV -n 64 -c 3 -H 224 -W 224 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
    ./out/conv_driver.exe $CONV -n 128 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
    ./out/conv_driver.exe $CONV -n 64 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW
    ./out/conv_driver.exe $CONV -n 64 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW

    ./out/conv_driver.exe $CONV -n 64 -c 1024 -H 14 -W 14 -k 1024 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe $CONV -n 64 -c 1024 -H 7 -W 7 -k 1024 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe $CONV -n 64 -c 128 -H 56 -W 56 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe $CONV -n 64 -c 256 -H 28 -W 28 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe $CONV -n 64 -c 256 -H 56 -W 56 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe $CONV -n 64 -c 512 -H 14 -W 14 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F $FORW
    ./out/conv_driver.exe $CONV -n 64 -c 512 -H 28 -W 28 -k 512 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 32 -F $FORW
    #exit 1
fi

#resnext101
echo "=============================================================== resnext101"
./out/conv_driver.exe $CONV -n 64 -c 1024 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 2048 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 256 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 3 -H 224 -W 224 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 512 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
sleep 2

echo "=============================================================== inception4 batch_size=128"
#inception4 batch_size=128
./out/conv_driver.exe $CONV -n 128 -c 128 -H 17 -W 17 -k 128 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 128 -H 17 -W 17 -k 128 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 128 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 128 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 1280 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 1280 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 1280 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 1280 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 160 -H 17 -W 17 -k 160 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 160 -H 17 -W 17 -k 160 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 160 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 160 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 192 -H 17 -W 17 -k 192 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 192 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 192 -H 35 -W 35 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 192 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 192 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 2048 -H 8 -W 8 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 2048 -H 8 -W 8 -k 320 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 2048 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 2048 -H 8 -W 8 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 256 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 256 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 288 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 288 -H 35 -W 35 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 288 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 384 -H 8 -W 8 -k 384 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 384 -H 8 -W 8 -k 384 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 448 -H 8 -W 8 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 48 -H 35 -W 35 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 64 -H 73 -W 73 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 768 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 768 -H 17 -W 17 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 768 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 80 -H 73 -W 73 -k 192 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
sleep 2

echo "=============================================================== inception3 batch_size=64"
#inception3 batch_size=64
./out/conv_driver.exe $CONV -n 64 -c 1024 -H 17 -W 17 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 1024 -H 17 -W 17 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 1024 -H 17 -W 17 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 1024 -H 17 -W 17 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 1536 -H 8 -W 8 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 1536 -H 8 -W 8 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 160 -H 73 -W 73 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 192 -H 17 -W 17 -k 192 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 192 -H 17 -W 17 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 192 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 192 -H 17 -W 17 -k 224 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 192 -H 35 -W 35 -k 224 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 192 -H 71 -W 71 -k 192 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 224 -H 17 -W 17 -k 224 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 224 -H 17 -W 17 -k 256 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 224 -H 35 -W 35 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 256 -H 17 -W 17 -k 256 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 256 -H 17 -W 17 -k 320 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 3 -H 299 -W 299 -k 32 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 32 -H 147 -W 147 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 32 -H 149 -W 149 -k 32 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 320 -H 17 -W 17 -k 320 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 384 -H 35 -W 35 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 384 -H 35 -W 35 -k 384 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 384 -H 35 -W 35 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 384 -H 35 -W 35 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 384 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 384 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 384 -H 8 -W 8 -k 448 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 448 -H 8 -W 8 -k 512 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 512 -H 8 -W 8 -k 256 -y 1 -x 3 -p 0 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 512 -H 8 -W 8 -k 256 -y 3 -x 1 -p 1 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 64 -H 147 -W 147 -k 96 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 64 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 64 -H 73 -W 73 -k 64 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 64 -H 73 -W 73 -k 64 -y 7 -x 1 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 64 -H 73 -W 73 -k 96 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 96 -H 35 -W 35 -k 96 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1  -F $FORW ${LAYOUT_ARG}
sleep 2

echo "=============================================================== resnet50"
#resnet50
./out/conv_driver.exe $CONV -n 64 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 64 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW ${LAYOUT_ARG}
sleep 2

echo "=============================================================== ssd"
./out/conv_driver.exe $CONV -n 120 -c 3 -H 300 -W 300 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 64 -H 75 -W 75 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 64 -H 75 -W 75 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 64 -H 75 -W 75 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 64 -H 75 -W 75 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 64 -H 75 -W 75 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 64 -H 75 -W 75 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 64 -H 75 -W 75 -k 128 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 64 -H 75 -W 75 -k 128 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 128 -H 38 -W 38 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 128 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 512 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 512 -H 19 -W 19 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 19 -W 19 -k 512 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 512 -H 10 -W 10 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 128 -H 10 -W 10 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 5 -W 5 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 128 -H 5 -W 5 -k 256 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 3 -W 3 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 128 -H 3 -W 3 -k 256 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 38 -W 38 -k 340 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 512 -H 19 -W 19 -k 510 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 512 -H 10 -W 10 -k 510 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 5 -W 5 -k 510 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 3 -W 3 -k 340 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 120 -c 256 -H 1 -W 1 -k 340 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F $FORW ${LAYOUT_ARG}

echo "=============================================================== mask rcnn"
#mask rcnn
./out/conv_driver.exe $CONV -n 2 -c 256 -H 12 -W 18 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 2 -c 1024 -H 34 -W 84 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 2 -c 1024 -H 40 -W 52 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 2 -c 256 -H 100 -W 104 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 2 -c 256 -H 10 -W 20 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 2 -c 64 -H 71 -W 83 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 2 -c 64 -H 59 -W 57 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 4 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 4 -c 256 -H 28 -W 28 -k 256 -y 2 -x 2 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 3 -c 256 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 1 -c 256 -H 32 -W 64 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 1 -c 64 -H 17 -W 17 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F $FORW ${LAYOUT_ARG}


#retina net bs=16
#./out/conv_driver.exe $CONV -n 16 -c 256 -H 12 -W 12 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
#./out/conv_driver.exe $CONV -n 16 -c 256 -H 134 -W 77 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
#./out/conv_driver.exe $CONV -n 16 -c 256 -H 71 -W 101 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F $FORW
