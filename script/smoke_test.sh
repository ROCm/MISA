#!/bin/bash
# need use posix cmopatible bash

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

export IGEMM_HSACO=out/igemm_${DIR}_gtc_${ARCH}${LAYOUT_HSACO}${PREC_HSACO}.hsaco
export IGEMM_TENSOR_CAST_HSACO=out/igemm_gtc_tensor_cast.hsaco
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_SLEEP_MS=117
export PER_PIXEL_CHECK=0

export IGEMM_RAND_INT=1
export IGEMM_ASSERT_WHEN_INVALID=1
export IGEMM_WARMUP=1
export IGEMM_REPEAT=4

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

EXE=./out/conv_driver.exe

batch_size=( 1, 2 )
image_size=( 14 32 55 )
channel_size=( 64 32 )
group_size=( 1 )
stride_size=( 1 2 3 )
dilation_size=( 1 2 3 )
pad_size=( 0 1 2 3 )
filter_size=( 7 5 4 3 2 1 )
tile_x_size=( 0 4 8 11 )
tile_y_size=( 0 4 8 11 )


for tile_x in "${tile_x_size[@]}"; do
for tile_y in "${tile_y_size[@]}"; do
for n  in "${batch_size[@]}"; do
for c  in "${channel_size[@]}"; do
for k  in "${channel_size[@]}"; do
for hi in "${image_size[@]}"; do
for wi in "${image_size[@]}"; do
for fy in "${filter_size[@]}"; do
for fx in "${filter_size[@]}"; do
for py in "${pad_size[@]}"; do
for px in "${pad_size[@]}"; do
for sy in "${stride_size[@]}"; do
for sx in "${stride_size[@]}"; do
for dy in "${dilation_size[@]}"; do
for dx in "${dilation_size[@]}"; do
for g  in "${group_size[@]}"; do


#  (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
ho=$(( ( $hi + 2 * $py - $dy * ( $fy - 1 ) - 1 ) / $sy + 1  ))
wo=$(( ( $wi + 2 * $px - $dx * ( $fx - 1 ) - 1 ) / $sx + 1  ))
if (( $ho <= 0 || $wo <= 0 || $fy > $hi || $fx > $wi || ($fy - 1) < $py || ($fx - 1) < $px || $dy > $fy || $dx > $fx || $c % $g != 0 || $k % $g != 0 )); then
continue
fi
if (( ( $hi + 2 * $py - $dy * ( $fy - 1 ) - 1 ) < 0 || ( $wi + 2 * $px - $dx * ( $fx - 1 ) - 1 ) < 0 )); then
# negetive integer division in bash seems different from c language. e.g -1/2=-1 in C, but =0 in bash
continue
fi
if (( $ho < $tile_y || $wo < $tile_x )); then
continue
fi

echo "${CONV} -n $n -c $c -H $hi -W $wi -k $k -y $fy -x $fx -p $py -q $px -u $sy -v $sx -l $dy -j $dx -g $g -F $FORW ${LAYOUT_ARG} (ho:$ho, wo:$wo, tile_y:$tile_y, tile_x:$tile_x) "

TILE_X=$tile_x TILE_Y=$tile_y $EXE $CONV -n $n -c $c -H $hi -W $wi -k $k -y $fy -x $fx -p $py -q $px -u $sy -v $sx -l $dy -j $dx -g $g -F $FORW ${LAYOUT_ARG} || exit 1
sleep 0.1

done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done