
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

if [ "${ARCH}" != "gfx90a" ] && [ "${ARCH}" != "gfx908" ] && [ "${ARCH}" != "gfx1030" ] ; then
    echo "wrong arch: ${ARCH}"
    exit 1
fi

echo IGEMM_HSACO=out/igemm_${DIR}_gtc_${ARCH}${LAYOUT_HSACO}${PREC_HSACO}_fsr.hsaco
export IGEMM_HSACO=out/igemm_${DIR}_gtc_${ARCH}${LAYOUT_HSACO}${PREC_HSACO}_fsr.hsaco
export IGEMM_TENSOR_CAST_HSACO=out/igemm_gtc_tensor_cast.hsaco
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_SLEEP_MS=117
export PER_PIXEL_CHECK=1
export PER_PIXEL_CHECK_PRINT=1

export DBG_MODE=1
export IGEMM_ASSERT_WHEN_INVALID=1
export IGEMM_WARMUP=1
export IGEMM_REPEAT=4
export IGEMM_GKS_ITERATIVE=1
#export IGEMM_BENCH_CSV=1
export IGEMM_RAND_INT=1

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

./out/conv_driver.exe $CONV -n 1 -c 8 -H 1080 -W 1920 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -t 1 -F $FORW ${LAYOUT_ARG}
#./out/conv_driver.exe $CONV -n 1 -c 16 -H 135 -W 240 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -t 1 -F $FORW ${LAYOUT_ARG}
