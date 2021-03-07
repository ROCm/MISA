#!/bin/bash
# need use posix cmopatible bash

if [ $# -ne 2 ]
then 
    if [ $# -eq 1 ]
    then 
        echo "please give this script a precision(fp32 or fp16)"
        echo "now I use fp32 as default"
        echo "use me as: /bin/bash script/smoke_test.sh fwd/bwd/wrw fp32/fp16"
        DIR=$1
        PRECISION=fp32
    else
        echo "please give this script a direction(fwd, bwd or wrw) and a precision(fp32 or fp16)"
        echo "now I use fwd and fp32 as default"
        echo "use me as: /bin/bash script/smoke_test.sh fwd/bwd/wrw fp32/fp16"
        DIR=fwd
        PRECISION=fp32
    fi
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
export IGEMM_ASSERT_WHEN_INVALID=1

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

EXE=./out/conv_driver.exe

batch_size=( 2 )
image_size=( 14 32 55 )
channel_size=( 64 32 )
group_size=( 1 2 4 )
stride_size=( 1 2 3 )
dilation_size=( 1 2 3 )
pad_size=( 0 1 2 3 )
filter_size=( 7 5 4 3 2 1 )

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

echo "${CONV} -n $n -c $c -H $hi -W $wi -k $k -y $fy -x $fx -p $py -q $px -u $sy -v $sx -l $dy -j $dx -g $g -F $FORW  (ho:$ho, wo:$wo)"

$EXE $CONV -n $n -c $c -H $hi -W $wi -k $k -y $fy -x $fx -p $py -q $px -u $sy -v $sx -l $dy -j $dx -g $g -F $FORW || exit 1


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
