#!/bin/sh
#set -v
if [ $# -ge 1 ] ; then
    SCRIPT_NAME=$1
else
    SCRIPT_NAME="gtc_conv_resnet50.sh"
fi

if [ $# -ge 2 ] ; then
    PREC=$2
else
    PREC="fp32"
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

echo $SCRIPT_NAME

rm bench_model.csv

dir_list="wrw bwd fwd"

for dir in $dir_list
do 
python igemm_codegen.py config/igemm_${dir}_gtc_gfx908_nhwc${PREC_HSACO}.config
sh script/${SCRIPT_NAME} ${dir} nhwc ${PREC} 2>&1 | tee ${SCRIPT_NAME}_${dir}_${PREC}.log
done 
