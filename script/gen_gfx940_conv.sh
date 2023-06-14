#!/bin/sh
# call this at top
OUT=out
KERNELS=igemm_gtc_xdlops_nhwc_gfx940

rm -rf $KERNELS ; mkdir $KERNELS
mkdir -p $KERNELS/fwd_fp32
mkdir -p $KERNELS/fwd_fp16
mkdir -p $KERNELS/fwd_bf16
mkdir -p $KERNELS/bwd_fp32
mkdir -p $KERNELS/bwd_fp16
mkdir -p $KERNELS/bwd_bf16
mkdir -p $KERNELS/wrw_fp32
mkdir -p $KERNELS/wrw_fp16
mkdir -p $KERNELS/wrw_bf16

python3 igemm_codegen.py -s config/igemm_fwd_gtc_gfx940_nhwc.config      ; cp $OUT/*.s $OUT/*.inc $KERNELS/fwd_fp32
python3 igemm_codegen.py -s config/igemm_fwd_gtc_gfx940_nhwc_fp16.config ; cp $OUT/*.s $OUT/*.inc $KERNELS/fwd_fp16
python3 igemm_codegen.py -s config/igemm_fwd_gtc_gfx940_nhwc_bf16.config ; cp $OUT/*.s $OUT/*.inc $KERNELS/fwd_bf16

python3 igemm_codegen.py -s config/igemm_bwd_gtc_gfx940_nhwc.config      ; cp $OUT/*.s $OUT/*.inc $KERNELS/bwd_fp32
python3 igemm_codegen.py -s config/igemm_bwd_gtc_gfx940_nhwc_fp16.config ; cp $OUT/*.s $OUT/*.inc $KERNELS/bwd_fp16
python3 igemm_codegen.py -s config/igemm_bwd_gtc_gfx940_nhwc_bf16.config ; cp $OUT/*.s $OUT/*.inc $KERNELS/bwd_bf16

python3 igemm_codegen.py -s config/igemm_wrw_gtc_gfx940_nhwc.config      ; cp $OUT/*.s $OUT/*.inc $KERNELS/wrw_fp32
python3 igemm_codegen.py -s config/igemm_wrw_gtc_gfx940_nhwc_fp16.config ; cp $OUT/*.s $OUT/*.inc $KERNELS/wrw_fp16
python3 igemm_codegen.py -s config/igemm_wrw_gtc_gfx940_nhwc_bf16.config ; cp $OUT/*.s $OUT/*.inc $KERNELS/wrw_bf16