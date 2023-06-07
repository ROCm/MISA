
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

if [[  "${LAYOUT}" = "nchw" ]] ; then
    LAYOUT_HSACO=""
    LAYOUT_ARG=""
elif [[  "${LAYOUT}" = "nhwc" ]] ; then
    LAYOUT_HSACO="_nhwc"
    LAYOUT_ARG="--in_layout NHWC --fil_layout NHWC --out_layout NHWC"
elif [[  "${LAYOUT}" = "nchwc_kcyxc" ]] ; then
    LAYOUT_HSACO="_nchwc"
    LAYOUT_ARG="--in_layout NCHWC --fil_layout NCHWC --out_layout NCHWC"
elif [[  "${LAYOUT}" = "nchwc_cyxkc" ]] ; then
    LAYOUT_HSACO="_nchwc"
    LAYOUT_ARG="--in_layout NCHWC --fil_layout CHWNC --out_layout NCHWC"
else
    echo "wrong layout: ${LAYOUT}"
    exit 1
fi

if [[ "${PREC}" = "fp32" ]] ; then
    PREC_HSACO=""
    CONV="conv"
elif [[ "${PREC}" = "int4"* ]] ; then
    PREC_HSACO="_${PREC}"
    CONV="conv${PREC}"
elif [[  "${PREC}" = "fp16"* ]] ; then
    PREC_HSACO="_${PREC}"
    CONV="conv${PREC}"
elif [[  "${PREC}" = "int8"* ]] ; then
    PREC_HSACO="_${PREC}"
    CONV="conv${PREC}"
elif [[ "${PREC}" = "bf16"* ]] ; then
    PREC_HSACO="_${PREC}"
    CONV="convbfp16${PREC:4}"
else
    echo "wrong precision: ${PREC}"
    exit 1
fi

if [ "${ARCH}" != "gfx90a" ] && [ "${ARCH}" != "gfx908" ] && [ "${ARCH}" != "gfx1030" ] && [ "${ARCH}" != "gfx940" ]; then
    echo "wrong arch: ${ARCH}"
    exit 1
fi

echo IGEMM_HSACO=out/igemm_${DIR}_gtc_${ARCH}${LAYOUT_HSACO}${PREC_HSACO}.hsaco
export IGEMM_HSACO=out/igemm_${DIR}_gtc_${ARCH}${LAYOUT_HSACO}${PREC_HSACO}.hsaco
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_TENSOR_CAST_HSACO=out/igemm_gtc_tensor_cast.hsaco
if [ "${ARCH}" = "gfx90a" ]; then
    export IGEMM_SCLK_MHZ=1700
elif [ "${ARCH}" = "gfx908" ]; then
    export IGEMM_SCLK_MHZ=1502
elif [ "${ARCH}" = "gfx1030" ] ; then
    export IGEMM_SCLK_MHZ=2450
fi
export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_SLEEP_MS=117
export IGEMM_BENCH_CSV=1
export IGEMM_RAND_INT=0
export PRINT_NRMS=0

rm bench_model.csv

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

BATCH=256

# resnet 50 bottleneck [3, 4, 6, 3]
# stage 1 x 3
./out/conv_driver.exe $CONV -n $BATCH -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

# stage 2 x 4
./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 128 -H 56 -W 56 -k 128 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

./out/conv_driver.exe $CONV -n $BATCH -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

./out/conv_driver.exe $CONV -n $BATCH -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

./out/conv_driver.exe $CONV -n $BATCH -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

# stage 3 x 6
./out/conv_driver.exe $CONV -n $BATCH -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 28 -W 28 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

./out/conv_driver.exe $CONV -n $BATCH -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

./out/conv_driver.exe $CONV -n $BATCH -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

./out/conv_driver.exe $CONV -n $BATCH -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

./out/conv_driver.exe $CONV -n $BATCH -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

./out/conv_driver.exe $CONV -n $BATCH -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

# stage 4 x 3
./out/conv_driver.exe $CONV -n $BATCH -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 512 -H 14 -W 14 -k 512 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

./out/conv_driver.exe $CONV -n $BATCH -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}

./out/conv_driver.exe $CONV -n $BATCH -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n $BATCH -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
