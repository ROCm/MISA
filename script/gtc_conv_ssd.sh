
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
    LAYOUT_ARG="${LAYOUT_ARG}"
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

echo "=============================================================== ssd bs120"
if [ ${DIR} = 2 ] || [ "${ARCH}" = "gfx1030" ]
then
echo "no first layer for bwd data"
else
./out/conv_driver.exe ${CONV} -n 120 -c 3 -H 300 -W 300 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
fi
./out/conv_driver.exe ${CONV} -n 120 -c 64 -H 75 -W 75 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 64 -H 75 -W 75 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 64 -H 75 -W 75 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 64 -H 75 -W 75 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 64 -H 75 -W 75 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 64 -H 75 -W 75 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 64 -H 75 -W 75 -k 128 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 64 -H 75 -W 75 -k 128 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 128 -H 38 -W 38 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 128 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 512 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 512 -H 19 -W 19 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 19 -W 19 -k 512 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 512 -H 10 -W 10 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 128 -H 10 -W 10 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 5 -W 5 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 128 -H 5 -W 5 -k 256 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 3 -W 3 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 128 -H 3 -W 3 -k 256 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 38 -W 38 -k 344 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 512 -H 19 -W 19 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 512 -H 10 -W 10 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 5 -W 5 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 3 -W 3 -k 344 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
./out/conv_driver.exe ${CONV} -n 120 -c 256 -H 1 -W 1 -k 344 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 ${LAYOUT_ARG} -m conv -g 1 -F ${FORW} 
