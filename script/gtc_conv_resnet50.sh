
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

if [ "${LAYOUT}" = "nchw" ] ; then
    LAYOUT_HSACO=""
    LAYOUT_ARG=""
elif [ "${LAYOUT}" = "nhwc" ] ; then
    LAYOUT_HSACO="_nhwc"
    LAYOUT_ARG="--in_layout NHWC --fil_layout NHWC --out_layout NHWC"
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

echo IGEMM_HSACO=out/igemm_${DIR}_gtc_gfx908${LAYOUT_HSACO}${PREC_HSACO}.hsaco
export IGEMM_HSACO=out/igemm_${DIR}_gtc_gfx908${LAYOUT_HSACO}${PREC_HSACO}.hsaco
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_SCLK_MHZ=1283
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

echo "=============================================================== resnet50 bs256"
./out/conv_driver.exe $CONV -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
sleep 2
echo "=============================================================== resnet50 bs128"
./out/conv_driver.exe $CONV -n 128 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -t 1 -F $FORW ${LAYOUT_ARG}
./out/conv_driver.exe $CONV -n 128 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -t 1 -F $FORW ${LAYOUT_ARG}
