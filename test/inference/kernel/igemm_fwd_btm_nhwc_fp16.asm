; pay attention to register bank of v_c, v_b
.macro .fma_1x16_fp16 v_c, v_a, v_b
    v_dot2c_f32_f16 v[\v_c+0 ], v[\v_a], v[\v_b+0 ]
    v_dot2c_f32_f16 v[\v_c+1 ], v[\v_a], v[\v_b+1 ]
    v_dot2c_f32_f16 v[\v_c+2 ], v[\v_a], v[\v_b+2 ]
    v_dot2c_f32_f16 v[\v_c+3 ], v[\v_a], v[\v_b+3 ]
    v_dot2c_f32_f16 v[\v_c+4 ], v[\v_a], v[\v_b+4 ]
    v_dot2c_f32_f16 v[\v_c+5 ], v[\v_a], v[\v_b+5 ]
    v_dot2c_f32_f16 v[\v_c+6 ], v[\v_a], v[\v_b+6 ]
    v_dot2c_f32_f16 v[\v_c+7 ], v[\v_a], v[\v_b+7 ]
    v_dot2c_f32_f16 v[\v_c+8 ], v[\v_a], v[\v_b+8 ]
    v_dot2c_f32_f16 v[\v_c+9 ], v[\v_a], v[\v_b+9 ]
    v_dot2c_f32_f16 v[\v_c+10], v[\v_a], v[\v_b+10]
    v_dot2c_f32_f16 v[\v_c+11], v[\v_a], v[\v_b+11]
    v_dot2c_f32_f16 v[\v_c+12], v[\v_a], v[\v_b+12]
    v_dot2c_f32_f16 v[\v_c+13], v[\v_a], v[\v_b+13]
    v_dot2c_f32_f16 v[\v_c+14], v[\v_a], v[\v_b+14]
    v_dot2c_f32_f16 v[\v_c+15], v[\v_a], v[\v_b+15]
.endm

.macro .fma_1x8_fp16 v_c, v_a, v_b
    v_dot2c_f32_f16 v[\v_c+0 ], v[\v_a], v[\v_b+0 ]
    v_dot2c_f32_f16 v[\v_c+1 ], v[\v_a], v[\v_b+1 ]
    v_dot2c_f32_f16 v[\v_c+2 ], v[\v_a], v[\v_b+2 ]
    v_dot2c_f32_f16 v[\v_c+3 ], v[\v_a], v[\v_b+3 ]
    v_dot2c_f32_f16 v[\v_c+4 ], v[\v_a], v[\v_b+4 ]
    v_dot2c_f32_f16 v[\v_c+5 ], v[\v_a], v[\v_b+5 ]
    v_dot2c_f32_f16 v[\v_c+6 ], v[\v_a], v[\v_b+6 ]
    v_dot2c_f32_f16 v[\v_c+7 ], v[\v_a], v[\v_b+7 ]
.endm

.macro .fma_1x4_fp16 v_c, v_a, v_b
    v_dot2c_f32_f16 v[\v_c+0 ], v[\v_a], v[\v_b+0 ]
    v_dot2c_f32_f16 v[\v_c+1 ], v[\v_a], v[\v_b+1 ]
    v_dot2c_f32_f16 v[\v_c+2 ], v[\v_a], v[\v_b+2 ]
    v_dot2c_f32_f16 v[\v_c+3 ], v[\v_a], v[\v_b+3 ]
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

.macro .v_clear_nc vid, num
    _v = \vid
    .rept \num
        v_mov_b32 v[_v], 0
        _v = _v + 1
    .endr
.endm

.include "igemm_fwd_btm_nhwc_fp16_128x016.asm"
.include "igemm_fwd_btm_nhwc_fp16_256x016.asm"
.include "igemm_fwd_btm_nhwc_fp16_256x008.asm"
.include "igemm_fwd_btm_nhwc_fp16_512x008.asm"
.include "igemm_fwd_btm_nhwc_fp16_1024x008.asm"

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: igemm_fwd_btm_nhwc_fp16_128x16x16_r3
    .symbol: igemm_fwd_btm_nhwc_fp16_128x16x16_r3.kd
    .sgpr_count: 60
    .vgpr_count: 74
    .kernarg_segment_align: 8
    .kernarg_segment_size: 112
    .group_segment_fixed_size: 13056
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
    - { .name: magic_0   , .size: 4, .offset:  96, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset: 100, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset: 104, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
  - .name: igemm_fwd_btm_nhwc_fp16_256x16x16_r3
    .symbol: igemm_fwd_btm_nhwc_fp16_256x16x16_r3.kd
    .sgpr_count: 60
    .vgpr_count: 112
    .kernarg_segment_align: 8
    .kernarg_segment_size: 112
    .group_segment_fixed_size: 13056
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
    - { .name: magic_0   , .size: 4, .offset:  96, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset: 100, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset: 104, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
  - .name: igemm_fwd_btm_nhwc_fp16_256x8x16_r2
    .symbol: igemm_fwd_btm_nhwc_fp16_256x8x16_r2.kd
    .sgpr_count: 64
    .vgpr_count: 128
    .kernarg_segment_align: 8
    .kernarg_segment_size: 112
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
    - { .name: magic_0   , .size: 4, .offset:  96, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset: 100, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset: 104, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
  - .name: igemm_fwd_btm_nhwc_fp16_512x8x16_r2
    .symbol: igemm_fwd_btm_nhwc_fp16_512x8x16_r2.kd
    .sgpr_count: 64
    .vgpr_count: 188
    .kernarg_segment_align: 8
    .kernarg_segment_size: 112
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
    - { .name: magic_0   , .size: 4, .offset:  96, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset: 100, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset: 104, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
  - .name: igemm_fwd_btm_nhwc_fp16_512x8x8_r1
    .symbol: igemm_fwd_btm_nhwc_fp16_512x8x8_r1.kd
    .sgpr_count: 64
    .vgpr_count: 124
    .kernarg_segment_align: 8
    .kernarg_segment_size: 112
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
    - { .name: magic_0   , .size: 4, .offset:  96, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset: 100, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset: 104, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
  - .name: igemm_fwd_btm_nhwc_fp16_1024x8x8_r1
    .symbol: igemm_fwd_btm_nhwc_fp16_1024x8x8_r1.kd
    .sgpr_count: 64
    .vgpr_count: 212
    .kernarg_segment_align: 8
    .kernarg_segment_size: 112
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
    - { .name: magic_0   , .size: 4, .offset:  96, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset: 100, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset: 104, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata
