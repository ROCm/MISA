; pay attention to register bank of v_c, v_b
.macro .fma_1x16_int8x4 v_c, v_a, v_b
    v_dot4c_i32_i8 v[\v_c+0 ], v[\v_a], v[\v_b+0 ]
    v_dot4c_i32_i8 v[\v_c+1 ], v[\v_a], v[\v_b+1 ]
    v_dot4c_i32_i8 v[\v_c+2 ], v[\v_a], v[\v_b+2 ]
    v_dot4c_i32_i8 v[\v_c+3 ], v[\v_a], v[\v_b+3 ]
    v_dot4c_i32_i8 v[\v_c+4 ], v[\v_a], v[\v_b+4 ]
    v_dot4c_i32_i8 v[\v_c+5 ], v[\v_a], v[\v_b+5 ]
    v_dot4c_i32_i8 v[\v_c+6 ], v[\v_a], v[\v_b+6 ]
    v_dot4c_i32_i8 v[\v_c+7 ], v[\v_a], v[\v_b+7 ]
    v_dot4c_i32_i8 v[\v_c+8 ], v[\v_a], v[\v_b+8 ]
    v_dot4c_i32_i8 v[\v_c+9 ], v[\v_a], v[\v_b+9 ]
    v_dot4c_i32_i8 v[\v_c+10], v[\v_a], v[\v_b+10]
    v_dot4c_i32_i8 v[\v_c+11], v[\v_a], v[\v_b+11]
    v_dot4c_i32_i8 v[\v_c+12], v[\v_a], v[\v_b+12]
    v_dot4c_i32_i8 v[\v_c+13], v[\v_a], v[\v_b+13]
    v_dot4c_i32_i8 v[\v_c+14], v[\v_a], v[\v_b+14]
    v_dot4c_i32_i8 v[\v_c+15], v[\v_a], v[\v_b+15]
.endm

.macro .fma_1x8_int8x4 v_c, v_a, v_b
    v_dot4c_i32_i8 v[\v_c+0 ], v[\v_a], v[\v_b+0 ]
    v_dot4c_i32_i8 v[\v_c+1 ], v[\v_a], v[\v_b+1 ]
    v_dot4c_i32_i8 v[\v_c+2 ], v[\v_a], v[\v_b+2 ]
    v_dot4c_i32_i8 v[\v_c+3 ], v[\v_a], v[\v_b+3 ]
    v_dot4c_i32_i8 v[\v_c+4 ], v[\v_a], v[\v_b+4 ]
    v_dot4c_i32_i8 v[\v_c+5 ], v[\v_a], v[\v_b+5 ]
    v_dot4c_i32_i8 v[\v_c+6 ], v[\v_a], v[\v_b+6 ]
    v_dot4c_i32_i8 v[\v_c+7 ], v[\v_a], v[\v_b+7 ]
.endm

.macro .fma_1x4_int8x4 v_c, v_a, v_b
    v_dot4c_i32_i8 v[\v_c+0 ], v[\v_a], v[\v_b+0 ]
    v_dot4c_i32_i8 v[\v_c+1 ], v[\v_a], v[\v_b+1 ]
    v_dot4c_i32_i8 v[\v_c+2 ], v[\v_a], v[\v_b+2 ]
    v_dot4c_i32_i8 v[\v_c+3 ], v[\v_a], v[\v_b+3 ]
.endm

.macro .mdiv_u32_ss s_quot s_numer s_magic s_shift s_tmp
    s_mul_hi_u32 s[\s_tmp], s[\s_magic], s[\s_numer]
    s_add_u32 s[\s_tmp], s[\s_tmp], s[\s_numer]
    s_lshr_b32 s[\s_quot], s[\s_tmp], s[\s_shift]
.endm

.macro .mdiv_u32_rem_ss s_rem s_quot s_numer s_magic s_shift s_denom s_tmp
    .mdiv_u32_ss \s_quot,\s_numer,\s_magic,\s_shift,\s_tmp
    s_mul_i32 s[\s_tmp], s[\s_denom], s[\s_quot]
    s_sub_u32 s[\s_rem], s[\s_numer], s[\s_tmp]
.endm

.macro .mdiv_u32_vs v_quot v_numer s_magic s_shift v_tmp
    v_mul_hi_u32 v[\v_tmp], s[\s_magic], v[\v_numer]
    v_add_nc_u32 v[\v_tmp], v[\v_tmp], v[\v_numer]
    v_lshrrev_b32 v[\v_quot], s[\s_shift], v[\v_tmp]
.endm

.macro .mdiv_u32_rem_vs v_rem v_quot v_numer s_magic s_shift s_denom v_tmp
    .mdiv_u32_vs \v_quot,\v_numer,\s_magic,\s_shift,\v_tmp
    v_mul_lo_u32 v[\v_tmp], s[\s_denom], v[\v_quot]
    v_sub_nc_u32 v[\v_rem], v[\v_numer], v[\v_tmp]
.endm

.macro .pack_i8x4_i32_r1 v_d, v_src, s_0xff
    v_and_b32 v[\v_src+ 0], s[\s_0xff], v[\v_src+ 0]
    v_and_b32 v[\v_src+ 1], s[\s_0xff], v[\v_src+ 1]
    v_and_b32 v[\v_src+ 2], s[\s_0xff], v[\v_src+ 2]
    v_lshlrev_b32 v[\v_src+ 3], 24, v[\v_src+ 3]
    v_lshlrev_b32 v[\v_src+ 1],  8, v[\v_src+ 1]
    v_lshlrev_b32 v[\v_src+ 2], 16, v[\v_src+ 2]
    v_or_b32 v[\v_d], v[\v_src+ 0], v[\v_src+ 3]
    v_or3_b32 v[\v_d], v[\v_d], v[\v_src+ 1], v[\v_src+ 2]
.endm

.macro .pack_i8x4_i32_r2 v_d, v_src, s_0xff
    v_and_b32 v[\v_src+ 0], s[\s_0xff], v[\v_src+ 0]
    v_lshlrev_b32 v[\v_src+ 3], 24, v[\v_src+ 3]

    v_and_b32 v[\v_src+ 4], s[\s_0xff], v[\v_src+ 4]
    v_lshlrev_b32 v[\v_src+ 7], 24, v[\v_src+ 7]

    v_and_b32 v[\v_src+ 1], s[\s_0xff], v[\v_src+ 1]
    v_and_b32 v[\v_src+ 2], s[\s_0xff], v[\v_src+ 2]

    v_or_b32 v[\v_d+ 0], v[\v_src+ 0], v[\v_src+ 3]

    v_and_b32 v[\v_src+ 5], s[\s_0xff], v[\v_src+ 5]

    v_and_b32 v[\v_src+ 6], s[\s_0xff], v[\v_src+ 6]
    v_or_b32 v[\v_d+ 1], v[\v_src+ 4], v[\v_src+ 7]

    v_lshlrev_b32 v[\v_src+ 1],  8, v[\v_src+ 1]
    v_lshlrev_b32 v[\v_src+ 2], 16, v[\v_src+ 2]
    v_lshlrev_b32 v[\v_src+ 5],  8, v[\v_src+ 5]
    v_lshlrev_b32 v[\v_src+ 6], 16, v[\v_src+ 6]

    v_or3_b32 v[\v_d+ 0], v[\v_d+ 0], v[\v_src+ 1], v[\v_src+ 2]
    v_or3_b32 v[\v_d+ 1], v[\v_d+ 1], v[\v_src+ 5], v[\v_src+ 6]
.endm

;.macro .pack_i8x4_i32_r4 v_d, v_src, s_0xff
;    v_and_b32 v[\v_src+ 0], s[\s_0xff], v[\v_src+ 0]
;    v_and_b32 v[\v_src+ 1], s[\s_0xff], v[\v_src+ 1]
;    v_and_b32 v[\v_src+ 2], s[\s_0xff], v[\v_src+ 2]
;    v_lshlrev_b32 v[\v_src+ 3], 24, v[\v_src+ 3]
;    v_lshlrev_b32 v[\v_src+ 1],  8, v[\v_src+ 1]
;    v_lshlrev_b32 v[\v_src+ 2], 16, v[\v_src+ 2]
;    v_or_b32 v[\v_d+ 0], v[\v_src+ 0], v[\v_src+ 3]
;    v_or3_b32 v[\v_d+ 0], v[\v_d+ 0], v[\v_src+ 1], v[\v_src+ 2]
;
;    v_and_b32 v[\v_src+ 4], s[\s_0xff], v[\v_src+ 4]
;    v_and_b32 v[\v_src+ 5], s[\s_0xff], v[\v_src+ 5]
;    v_and_b32 v[\v_src+ 6], s[\s_0xff], v[\v_src+ 6]
;    v_lshlrev_b32 v[\v_src+ 7], 24, v[\v_src+ 7]
;    v_lshlrev_b32 v[\v_src+ 5],  8, v[\v_src+ 5]
;    v_lshlrev_b32 v[\v_src+ 6], 16, v[\v_src+ 6]
;    v_or_b32 v[\v_d+ 1], v[\v_src+ 4], v[\v_src+ 7]
;    v_or3_b32 v[\v_d+ 1], v[\v_d+ 1], v[\v_src+ 5], v[\v_src+ 6]
;
;    v_and_b32 v[\v_src+ 8], s[\s_0xff], v[\v_src+ 8]
;    v_and_b32 v[\v_src+ 9], s[\s_0xff], v[\v_src+ 9]
;    v_and_b32 v[\v_src+10], s[\s_0xff], v[\v_src+10]
;    v_lshlrev_b32 v[\v_src+11], 24, v[\v_src+11]
;    v_lshlrev_b32 v[\v_src+ 9],  8, v[\v_src+ 9]
;    v_lshlrev_b32 v[\v_src+10], 16, v[\v_src+10]
;    v_or_b32 v[\v_d+ 2], v[\v_src+ 8], v[\v_src+11]
;    v_or3_b32 v[\v_d+ 2], v[\v_d+ 2], v[\v_src+ 9], v[\v_src+10]
;
;    v_and_b32 v[\v_src+12], s[\s_0xff], v[\v_src+12]
;    v_and_b32 v[\v_src+13], s[\s_0xff], v[\v_src+13]
;    v_and_b32 v[\v_src+14], s[\s_0xff], v[\v_src+14]
;    v_lshlrev_b32 v[\v_src+15], 24, v[\v_src+15]
;    v_lshlrev_b32 v[\v_src+13],  8, v[\v_src+13]
;    v_lshlrev_b32 v[\v_src+14], 16, v[\v_src+14]
;    v_or_b32 v[\v_d+ 3], v[\v_src+12], v[\v_src+15]
;    v_or3_b32 v[\v_d+ 3], v[\v_d+ 3], v[\v_src+13], v[\v_src+14]
;.endm

.macro .pack_i8x4_i32_r4 v_d, v_src, s_0xff
    v_and_b32 v[\v_src+ 0], s[\s_0xff], v[\v_src+ 0]
    v_lshlrev_b32 v[\v_src+ 3], 24, v[\v_src+ 3]
    v_and_b32 v[\v_src+ 4], s[\s_0xff], v[\v_src+ 4]
    v_lshlrev_b32 v[\v_src+ 7], 24, v[\v_src+ 7]

    v_and_b32 v[\v_src+ 8], s[\s_0xff], v[\v_src+ 8]
    v_lshlrev_b32 v[\v_src+11], 24, v[\v_src+11]
    v_and_b32 v[\v_src+12], s[\s_0xff], v[\v_src+12]
    v_lshlrev_b32 v[\v_src+15], 24, v[\v_src+15]

    v_or_b32 v[\v_d+ 0], v[\v_src+ 0], v[\v_src+ 3]
    v_or_b32 v[\v_d+ 1], v[\v_src+ 4], v[\v_src+ 7]
    v_or_b32 v[\v_d+ 2], v[\v_src+ 8], v[\v_src+11]

    v_and_b32 v[\v_src+ 1], s[\s_0xff], v[\v_src+ 1]
    v_or_b32 v[\v_d+ 3], v[\v_src+12], v[\v_src+15]

    v_and_b32 v[\v_src+ 2], s[\s_0xff], v[\v_src+ 2]
    v_and_b32 v[\v_src+ 5], s[\s_0xff], v[\v_src+ 5]
    v_and_b32 v[\v_src+ 6], s[\s_0xff], v[\v_src+ 6]
    v_and_b32 v[\v_src+ 9], s[\s_0xff], v[\v_src+ 9]
    v_and_b32 v[\v_src+10], s[\s_0xff], v[\v_src+10]
    v_and_b32 v[\v_src+13], s[\s_0xff], v[\v_src+13]
    v_and_b32 v[\v_src+14], s[\s_0xff], v[\v_src+14]

    v_lshlrev_b32 v[\v_src+ 1],  8, v[\v_src+ 1]
    v_lshlrev_b32 v[\v_src+ 2], 16, v[\v_src+ 2]

    v_lshlrev_b32 v[\v_src+ 5],  8, v[\v_src+ 5]
    v_lshlrev_b32 v[\v_src+ 6], 16, v[\v_src+ 6]

    v_lshlrev_b32 v[\v_src+ 9],  8, v[\v_src+ 9]
    v_lshlrev_b32 v[\v_src+10], 16, v[\v_src+10]

    v_lshlrev_b32 v[\v_src+13],  8, v[\v_src+13]
    v_lshlrev_b32 v[\v_src+14], 16, v[\v_src+14]

    v_or3_b32 v[\v_d+ 0], v[\v_d+ 0], v[\v_src+ 1], v[\v_src+ 2]
    v_or3_b32 v[\v_d+ 1], v[\v_d+ 1], v[\v_src+ 5], v[\v_src+ 6]
    v_or3_b32 v[\v_d+ 2], v[\v_d+ 2], v[\v_src+ 9], v[\v_src+10]
    v_or3_b32 v[\v_d+ 3], v[\v_d+ 3], v[\v_src+13], v[\v_src+14]
.endm


.macro .v_clear_nc vid, num
    _v = \vid
    .rept \num
        v_mov_b32 v[_v], 0
        _v = _v + 1
    .endr
.endm

.macro exp_f_float base, sign, vtmp //e^x = 2^(xlog2e)
    .if \sign < 0
        v_mov_b32 v[\vtmp], 0xbfb8aa3b //-log2e
    .else
        v_mov_b32 v[\vtmp], 0x3fb8aa3b //log2e
    .endif
    v_mul_f32 v[\base], v[\base], v[\vtmp]
    v_exp_f32 v[\base], v[\base]
.endm

.macro ln_f_float base, vtmp // ln(x) = log2x * 1 / (log2e)
    v_log_f32 v[\base], v[\base]
    v_mov_b32 v[\vtmp], 0x3f317218 // 1/(log2e)
    v_mul_f32 v[\base], v[\base], v[\vtmp]
.endm

MIOPEN_NEURON_PASTHRU = 0      // x
MIOPEN_NEURON_LOGISTIC = 1     // 1 / (1 + e^-x)	//Sigmoid
MIOPEN_NEURON_TANH = 2         // beta * tanh(alpha * x)
MIOPEN_NEURON_RELU = 3         // max(0, x)
MIOPEN_NEURON_SOFTRELU = 4     // log(1 + e^x)   // bonomial normal log likelihood
MIOPEN_NEURON_ABS = 5          // abs(x)
MIOPEN_NEURON_POWER = 6        // (alpha + beta * x )^gamma
MIOPEN_NEURON_CLIPPED_RELU = 7 // min(alpha, max(0, x))
MIOPEN_NEURON_LEAKY_RELU = 8   // alpha * x | x <= 0; x | x > 0
MIOPEN_NEURON_ELU = 9          // alpha * (e^x - 1) | x <= 0; x | x > 0

.ifnotdef activ_mode
      activ_mode = MIOPEN_NEURON_PASTHRU
.endif

EPSILON_float = 0x358637bd

.macro .activ_f32 base, activ_mode, alpha, beta, gamma, vtmp0, vtmp1
    .if \activ_mode == MIOPEN_NEURON_LOGISTIC //1 / (1 + e^-x)
        exp_f_float \base, -1, \vtmp0
        v_add_f32 v[\base], 1.0, v[\base]
        v_rcp_f32 v[\base], v[\base]
    .elseif \activ_mode == MIOPEN_NEURON_TANH // \beta * tanh(\alpha * x)
        v_mul_f32 v[\base], s[\alpha], v[\base]
        v_mul_f32 v[\base], 2.0, v[\base]
        exp_f_float \base, 1, \vtmp0
        v_add_f32 v[\base], 1.0, v[\base]
        v_rcp_f32 v[\base], v[\base]
        v_mul_f32 v[\base], 2.0, v[\base]
        v_sub_f32 v[\base], 1.0, v[\base]
        v_mov_b32 v[\vtmp0], 1.0
        v_mul_f32 v[\base], s[\beta], v[\base]
    .elseif \activ_mode == MIOPEN_NEURON_RELU //max(0, x)
        v_max_f32 v[\base], v[\base], 0
    .elseif \activ_mode == MIOPEN_NEURON_SOFTRELU //log(1 + e^x)
        exp_f_float \base, 1, \vtmp0
        v_add_f32 v[\base], 1.0, v[\base]
        ln_f_float \base, \vtmp0
    .elseif \activ_mode == MIOPEN_NEURON_ABS //abs(x)
        v_max_f32 v[\base], v[\base], -v[\base]
    .elseif \activ_mode == MIOPEN_NEURON_POWER //(\alpha + \beta * x )^\gamma
        v_mul_f32 v[\base], s[\beta], v[\base]
        v_add_f32 v[\base], s[\alpha], v[\base]
        v_mov_b32 v[\vtmp0], v[\base]
        v_log_f32 v[\base], v[\base]
        v_mul_f32 v[\base], s[\gamma], v[\base]
        v_exp_f32 v[\base], v[\base]
        v_cmp_lt_f32 EPSILON_float, v[\vtmp0]
        v_cndmask_b32 v[\base], 0, v[\base]
    .elseif \activ_mode == MIOPEN_NEURON_CLIPPED_RELU //min(\alpha, max(0, x))
        v_max_f32 v[\base], v[\base], 0
        v_min_f32 v[\base], s[\alpha], v[\base] 
    .elseif \activ_mode == MIOPEN_NEURON_LEAKY_RELU //\alpha * x | x <= 0; x | x > 0
        v_cmp_lt_f32 0, v[\base]
        v_mov_b32 v[\vtmp0], s[\alpha]
        v_cndmask_b32 v[\vtmp0], v[\vtmp0], 1.0
        v_mul_f32 v[\base], v[\base], v[\vtmp0]
    .elseif \activ_mode == MIOPEN_NEURON_ELU //\alpha * (e^x - 1) | x <= 0; x | x > 0
        v_cmp_lt_f32 0, v[\base]
        v_mov_b32 v[\vtmp1], v[\base]
        exp_f_float \base, 1, \vtmp0
        v_add_f32 v[\base], -1.0, v[\base]
        v_mul_f32 v[\base], s[\alpha], v[\base]
        v_cndmask_b32 v[\base], v[\base], v[\vtmp1]
    .endif
.endm

.macro .activ_int32 base, activ_mode, alpha, beta, gamma, vtmp0, vtmp1
    v_cvt_f32_i32 v[\base], v[\base]
    .activ_f32 \base, \activ_mode, \alpha, \beta, \gamma, \vtmp0, \vtmp1
    v_cvt_i32_f32 v[\base], v[\base]
.endm

.include "igemm_fwd_btm_nhwc_int8_256x004.asm"
.include "igemm_fwd_btm_nhwc_int8_256x008.asm"
.include "igemm_fwd_btm_nhwc_int8_512x008.asm"
.include "igemm_fwd_btm_nhwc_int8_512x016.asm"
.include "igemm_fwd_btm_nhwc_int8_1024x016.asm"

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: igemm_fwd_btm_nhwc_int8_256x4x16_r1
    .symbol: igemm_fwd_btm_nhwc_int8_256x4x16_r1.kd
    .sgpr_count: 68
    .vgpr_count: 108
    .kernarg_segment_align: 8
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .reqd_workgroup_size : [64, 1, 1]
    .max_flat_workgroup_size: 64
    .args:
    - { .name: p_in      , .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_wei     , .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_out     , .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: hi        , .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32}
    - { .name: wi        , .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32}
    - { .name: n         , .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
    - { .name: k         , .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32}
    - { .name: c         , .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32}
    - { .name: ho        , .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32}
    - { .name: wo        , .size: 4, .offset:  48, .value_kind: by_value, .value_type: i32}
    - { .name: stride_h  , .size: 4, .offset:  52, .value_kind: by_value, .value_type: i32}
    - { .name: stride_w  , .size: 4, .offset:  56, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_h, .size: 4, .offset:  60, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_w, .size: 4, .offset:  64, .value_kind: by_value, .value_type: i32}
    - { .name: pad_h     , .size: 4, .offset:  68, .value_kind: by_value, .value_type: i32}
    - { .name: pad_w     , .size: 4, .offset:  72, .value_kind: by_value, .value_type: i32}
    - { .name: y         , .size: 4, .offset:  76, .value_kind: by_value, .value_type: i32}
    - { .name: x         , .size: 4, .offset:  80, .value_kind: by_value, .value_type: i32}
    - { .name: group     , .size: 4, .offset:  84, .value_kind: by_value, .value_type: i32}
    - { .name: batch_m   , .size: 4, .offset:  88, .value_kind: by_value, .value_type: i32}
    - { .name: stride_m  , .size: 4, .offset:  92, .value_kind: by_value, .value_type: i32}
    - { .name: alpha     , .size: 4, .offset:  96, .value_kind: by_value, .value_type: f32}
    - { .name: beta      , .size: 4, .offset: 100, .value_kind: by_value, .value_type: f32}
    - { .name: gamma     , .size: 4, .offset: 104, .value_kind: by_value, .value_type: f32}
    - { .name: magic_0   , .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset: 112, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset: 116, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 120, .value_kind: by_value, .value_type: i32}
    - { .name: __pack_0  , .size: 4, .offset: 124, .value_kind: by_value, .value_type: i32}
  - .name: igemm_fwd_btm_nhwc_int8_256x8x16_r1
    .symbol: igemm_fwd_btm_nhwc_int8_256x8x16_r1.kd
    .sgpr_count: 68
    .vgpr_count: 80
    .kernarg_segment_align: 8
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 2048
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .reqd_workgroup_size : [128, 1, 1]
    .max_flat_workgroup_size: 128
    .args:
    - { .name: p_in      , .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_wei     , .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_out     , .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: hi        , .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32}
    - { .name: wi        , .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32}
    - { .name: n         , .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
    - { .name: k         , .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32}
    - { .name: c         , .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32}
    - { .name: ho        , .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32}
    - { .name: wo        , .size: 4, .offset:  48, .value_kind: by_value, .value_type: i32}
    - { .name: stride_h  , .size: 4, .offset:  52, .value_kind: by_value, .value_type: i32}
    - { .name: stride_w  , .size: 4, .offset:  56, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_h, .size: 4, .offset:  60, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_w, .size: 4, .offset:  64, .value_kind: by_value, .value_type: i32}
    - { .name: pad_h     , .size: 4, .offset:  68, .value_kind: by_value, .value_type: i32}
    - { .name: pad_w     , .size: 4, .offset:  72, .value_kind: by_value, .value_type: i32}
    - { .name: y         , .size: 4, .offset:  76, .value_kind: by_value, .value_type: i32}
    - { .name: x         , .size: 4, .offset:  80, .value_kind: by_value, .value_type: i32}
    - { .name: group     , .size: 4, .offset:  84, .value_kind: by_value, .value_type: i32}
    - { .name: batch_m   , .size: 4, .offset:  88, .value_kind: by_value, .value_type: i32}
    - { .name: stride_m  , .size: 4, .offset:  92, .value_kind: by_value, .value_type: i32}
    - { .name: alpha     , .size: 4, .offset:  96, .value_kind: by_value, .value_type: f32}
    - { .name: beta      , .size: 4, .offset: 100, .value_kind: by_value, .value_type: f32}
    - { .name: gamma     , .size: 4, .offset: 104, .value_kind: by_value, .value_type: f32}
    - { .name: magic_0   , .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset: 112, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset: 116, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 120, .value_kind: by_value, .value_type: i32}
    - { .name: __pack_0  , .size: 4, .offset: 124, .value_kind: by_value, .value_type: i32}
  - .name: igemm_fwd_btm_nhwc_int8_512x8x16_r1
    .symbol: igemm_fwd_btm_nhwc_int8_512x8x16_r1.kd
    .sgpr_count: 68
    .vgpr_count: 124
    .kernarg_segment_align: 8
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 2048
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .reqd_workgroup_size : [128, 1, 1]
    .max_flat_workgroup_size: 128
    .args:
    - { .name: p_in      , .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_wei     , .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_out     , .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: hi        , .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32}
    - { .name: wi        , .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32}
    - { .name: n         , .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
    - { .name: k         , .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32}
    - { .name: c         , .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32}
    - { .name: ho        , .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32}
    - { .name: wo        , .size: 4, .offset:  48, .value_kind: by_value, .value_type: i32}
    - { .name: stride_h  , .size: 4, .offset:  52, .value_kind: by_value, .value_type: i32}
    - { .name: stride_w  , .size: 4, .offset:  56, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_h, .size: 4, .offset:  60, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_w, .size: 4, .offset:  64, .value_kind: by_value, .value_type: i32}
    - { .name: pad_h     , .size: 4, .offset:  68, .value_kind: by_value, .value_type: i32}
    - { .name: pad_w     , .size: 4, .offset:  72, .value_kind: by_value, .value_type: i32}
    - { .name: y         , .size: 4, .offset:  76, .value_kind: by_value, .value_type: i32}
    - { .name: x         , .size: 4, .offset:  80, .value_kind: by_value, .value_type: i32}
    - { .name: group     , .size: 4, .offset:  84, .value_kind: by_value, .value_type: i32}
    - { .name: batch_m   , .size: 4, .offset:  88, .value_kind: by_value, .value_type: i32}
    - { .name: stride_m  , .size: 4, .offset:  92, .value_kind: by_value, .value_type: i32}
    - { .name: alpha     , .size: 4, .offset:  96, .value_kind: by_value, .value_type: f32}
    - { .name: beta      , .size: 4, .offset: 100, .value_kind: by_value, .value_type: f32}
    - { .name: gamma     , .size: 4, .offset: 104, .value_kind: by_value, .value_type: f32}
    - { .name: magic_0   , .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset: 112, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset: 116, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 120, .value_kind: by_value, .value_type: i32}
    - { .name: __pack_0  , .size: 4, .offset: 124, .value_kind: by_value, .value_type: i32}
  - .name: igemm_fwd_btm_nhwc_int8_512x16x8_r2
    .symbol: igemm_fwd_btm_nhwc_int8_512x16x8_r2.kd
    .sgpr_count: 68
    .vgpr_count: 140
    .kernarg_segment_align: 8
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .reqd_workgroup_size : [128, 1, 1]
    .max_flat_workgroup_size: 128
    .args:
    - { .name: p_in      , .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_wei     , .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_out     , .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: hi        , .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32}
    - { .name: wi        , .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32}
    - { .name: n         , .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
    - { .name: k         , .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32}
    - { .name: c         , .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32}
    - { .name: ho        , .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32}
    - { .name: wo        , .size: 4, .offset:  48, .value_kind: by_value, .value_type: i32}
    - { .name: stride_h  , .size: 4, .offset:  52, .value_kind: by_value, .value_type: i32}
    - { .name: stride_w  , .size: 4, .offset:  56, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_h, .size: 4, .offset:  60, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_w, .size: 4, .offset:  64, .value_kind: by_value, .value_type: i32}
    - { .name: pad_h     , .size: 4, .offset:  68, .value_kind: by_value, .value_type: i32}
    - { .name: pad_w     , .size: 4, .offset:  72, .value_kind: by_value, .value_type: i32}
    - { .name: y         , .size: 4, .offset:  76, .value_kind: by_value, .value_type: i32}
    - { .name: x         , .size: 4, .offset:  80, .value_kind: by_value, .value_type: i32}
    - { .name: group     , .size: 4, .offset:  84, .value_kind: by_value, .value_type: i32}
    - { .name: batch_m   , .size: 4, .offset:  88, .value_kind: by_value, .value_type: i32}
    - { .name: stride_m  , .size: 4, .offset:  92, .value_kind: by_value, .value_type: i32}
    - { .name: alpha     , .size: 4, .offset:  96, .value_kind: by_value, .value_type: f32}
    - { .name: beta      , .size: 4, .offset: 100, .value_kind: by_value, .value_type: f32}
    - { .name: gamma     , .size: 4, .offset: 104, .value_kind: by_value, .value_type: f32}
    - { .name: magic_0   , .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset: 112, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset: 116, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 120, .value_kind: by_value, .value_type: i32}
    - { .name: __pack_0  , .size: 4, .offset: 124, .value_kind: by_value, .value_type: i32}
  - .name: igemm_fwd_btm_nhwc_int8_512x16x16_r2
    .symbol: igemm_fwd_btm_nhwc_int8_512x16x16_r2.kd
    .sgpr_count: 68
    .vgpr_count: 188
    .kernarg_segment_align: 8
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .reqd_workgroup_size : [128, 1, 1]
    .max_flat_workgroup_size: 128
    .args:
    - { .name: p_in      , .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_wei     , .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_out     , .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: hi        , .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32}
    - { .name: wi        , .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32}
    - { .name: n         , .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
    - { .name: k         , .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32}
    - { .name: c         , .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32}
    - { .name: ho        , .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32}
    - { .name: wo        , .size: 4, .offset:  48, .value_kind: by_value, .value_type: i32}
    - { .name: stride_h  , .size: 4, .offset:  52, .value_kind: by_value, .value_type: i32}
    - { .name: stride_w  , .size: 4, .offset:  56, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_h, .size: 4, .offset:  60, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_w, .size: 4, .offset:  64, .value_kind: by_value, .value_type: i32}
    - { .name: pad_h     , .size: 4, .offset:  68, .value_kind: by_value, .value_type: i32}
    - { .name: pad_w     , .size: 4, .offset:  72, .value_kind: by_value, .value_type: i32}
    - { .name: y         , .size: 4, .offset:  76, .value_kind: by_value, .value_type: i32}
    - { .name: x         , .size: 4, .offset:  80, .value_kind: by_value, .value_type: i32}
    - { .name: group     , .size: 4, .offset:  84, .value_kind: by_value, .value_type: i32}
    - { .name: batch_m   , .size: 4, .offset:  88, .value_kind: by_value, .value_type: i32}
    - { .name: stride_m  , .size: 4, .offset:  92, .value_kind: by_value, .value_type: i32}
    - { .name: alpha     , .size: 4, .offset:  96, .value_kind: by_value, .value_type: f32}
    - { .name: beta      , .size: 4, .offset: 100, .value_kind: by_value, .value_type: f32}
    - { .name: gamma     , .size: 4, .offset: 104, .value_kind: by_value, .value_type: f32}
    - { .name: magic_0   , .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset: 112, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset: 116, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 120, .value_kind: by_value, .value_type: i32}
    - { .name: __pack_0  , .size: 4, .offset: 124, .value_kind: by_value, .value_type: i32}
  - .name: igemm_fwd_btm_nhwc_int8_1024x16x8_r2
    .symbol: igemm_fwd_btm_nhwc_int8_1024x16x8_r2.kd
    .sgpr_count: 68
    .vgpr_count: 244
    .kernarg_segment_align: 8
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .reqd_workgroup_size : [128, 1, 1]
    .max_flat_workgroup_size: 128
    .args:
    - { .name: p_in      , .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_wei     , .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_out     , .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: hi        , .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32}
    - { .name: wi        , .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32}
    - { .name: n         , .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
    - { .name: k         , .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32}
    - { .name: c         , .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32}
    - { .name: ho        , .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32}
    - { .name: wo        , .size: 4, .offset:  48, .value_kind: by_value, .value_type: i32}
    - { .name: stride_h  , .size: 4, .offset:  52, .value_kind: by_value, .value_type: i32}
    - { .name: stride_w  , .size: 4, .offset:  56, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_h, .size: 4, .offset:  60, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_w, .size: 4, .offset:  64, .value_kind: by_value, .value_type: i32}
    - { .name: pad_h     , .size: 4, .offset:  68, .value_kind: by_value, .value_type: i32}
    - { .name: pad_w     , .size: 4, .offset:  72, .value_kind: by_value, .value_type: i32}
    - { .name: y         , .size: 4, .offset:  76, .value_kind: by_value, .value_type: i32}
    - { .name: x         , .size: 4, .offset:  80, .value_kind: by_value, .value_type: i32}
    - { .name: group     , .size: 4, .offset:  84, .value_kind: by_value, .value_type: i32}
    - { .name: batch_m   , .size: 4, .offset:  88, .value_kind: by_value, .value_type: i32}
    - { .name: stride_m  , .size: 4, .offset:  92, .value_kind: by_value, .value_type: i32}
    - { .name: alpha     , .size: 4, .offset:  96, .value_kind: by_value, .value_type: f32}
    - { .name: beta      , .size: 4, .offset: 100, .value_kind: by_value, .value_type: f32}
    - { .name: gamma     , .size: 4, .offset: 104, .value_kind: by_value, .value_type: f32}
    - { .name: magic_0   , .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset: 112, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset: 116, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 120, .value_kind: by_value, .value_type: i32}
    - { .name: __pack_0  , .size: 4, .offset: 124, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata
