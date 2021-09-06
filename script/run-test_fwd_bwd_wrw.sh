#!/bin/sh
set -v
if [ $# -ge 1 ] ; then
    SCRIPT_NAME=$1
else
    SCRIPT_NAME="gtc_conv_resnet50.sh"
fi

echo $SCRIPT_NAME

rm bench_model.csv

dir_list="wrw bwd fwd"

for dir in $dir_list
do 
python igemm_codegen.py config/igemm_${dir}_gtc_gfx908_nhwc_fp16.config
sh script/${SCRIPT_NAME} ${dir} nhwc fp16
done 
