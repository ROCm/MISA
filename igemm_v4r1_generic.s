.macro .v_u32_div v_q, v_n, v_d, v_tmp4, s_tmp4
     v_cvt_f32_u32     v[\v_tmp4+0],   v[\v_d]
     v_rcp_f32         v[\v_tmp4+0],   v[\v_tmp4+0]
     v_mul_f32         v[\v_tmp4+0],   0x4f800000, v[\v_tmp4+0]
     v_cvt_u32_f32     v[\v_tmp4+0],   v[\v_tmp4+0]
     v_mul_lo_u32      v[\v_tmp4+1],   v[\v_d],      v[\v_tmp4+0]
     v_mul_hi_u32      v[\v_tmp4+2],   v[\v_d],      v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+3],   vcc, 0,     v[\v_tmp4+1]
     v_cmp_ne_i32      s[\s_tmp4:\s_tmp4+1], 0,          v[\v_tmp4+2]
     v_cndmask_b32     v[\v_tmp4+1],   v[\v_tmp4+3],   v[\v_tmp4+1],   s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+1],   v[\v_tmp4+1],   v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
     v_add_co_u32      v[\v_tmp4+0],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
     v_cndmask_b32     v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_tmp4+2],   s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_n]
     v_mul_lo_u32      v[\v_tmp4+1],   v[\v_tmp4+0],   v[\v_d]
     v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_n],      v[\v_tmp4+1]
     v_cmp_ge_u32      s[\s_tmp4:\s_tmp4+1], v[\v_n],      v[\v_tmp4+1]
     v_cmp_ge_u32      s[\s_tmp4+2:\s_tmp4+3], v[\v_tmp4+2],   v[\v_d]
     v_add_co_u32      v[\v_tmp4+2],   vcc, 1, v[\v_tmp4+0]
     s_and_b64         s[\s_tmp4+2:\s_tmp4+3], s[\s_tmp4:\s_tmp4+1], s[\s_tmp4+2:\s_tmp4+3]
     v_add_co_u32      v[\v_tmp4+1],   vcc, -1,    v[\v_tmp4+0]
     v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+0],   v[\v_tmp4+2],      s[\s_tmp4+2:\s_tmp4+3]
     v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+1],   v[\v_tmp4+2],      s[\s_tmp4:\s_tmp4+1]
     v_cmp_ne_i32      vcc,          0,          v[\v_d]
     v_cndmask_b32     v[\v_q],      -1,         v[\v_tmp4+2],      vcc
.endm

.macro .v_u32_div_vs v_q, v_n, s_d, v_tmp4, s_tmp4
     v_cvt_f32_u32     v[\v_tmp4+0],   s[\s_d]
     v_rcp_f32         v[\v_tmp4+0],   v[\v_tmp4+0]
     v_mul_f32         v[\v_tmp4+0],   0x4f800000, v[\v_tmp4+0]
     v_cvt_u32_f32     v[\v_tmp4+0],   v[\v_tmp4+0]
     v_mul_lo_u32      v[\v_tmp4+1],   s[\s_d],      v[\v_tmp4+0]
     v_mul_hi_u32      v[\v_tmp4+2],   s[\s_d],      v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+3],   vcc, 0,     v[\v_tmp4+1]
     v_cmp_ne_i32      s[\s_tmp4:\s_tmp4+1], 0,          v[\v_tmp4+2]
     v_cndmask_b32     v[\v_tmp4+1],   v[\v_tmp4+3],   v[\v_tmp4+1],   s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+1],   v[\v_tmp4+1],   v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
     v_add_co_u32      v[\v_tmp4+0],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
     v_cndmask_b32     v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_tmp4+2],   s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_n]
     v_mul_lo_u32      v[\v_tmp4+1],   s[\s_d],     v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_n],      v[\v_tmp4+1]
     v_cmp_ge_u32      s[\s_tmp4:\s_tmp4+1], v[\v_n],      v[\v_tmp4+1]
     v_cmp_le_u32      s[\s_tmp4+2:\s_tmp4+3],  s[\s_d],    v[\v_tmp4+2]
     v_add_co_u32      v[\v_tmp4+2],   vcc, 1, v[\v_tmp4+0]
     s_and_b64         s[\s_tmp4+2:\s_tmp4+3], s[\s_tmp4:\s_tmp4+1], s[\s_tmp4+2:\s_tmp4+3]
     v_add_co_u32      v[\v_tmp4+1],   vcc, -1,    v[\v_tmp4+0]
     v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+0],   v[\v_tmp4+2],      s[\s_tmp4+2:\s_tmp4+3]
     v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+1],   v[\v_tmp4+2],      s[\s_tmp4:\s_tmp4+1]
     v_cmp_ne_i32      vcc,          s[\s_d],   0
     v_cndmask_b32     v[\v_q],      -1,         v[\v_tmp4+2],      vcc
.endm

.macro .v_u32_div_ss v_q, s_n, s_d, v_tmp4, s_tmp4
     v_cvt_f32_u32     v[\v_tmp4+0],   s[\s_d]
     v_rcp_f32         v[\v_tmp4+0],   v[\v_tmp4+0]
     v_mul_f32         v[\v_tmp4+0],   0x4f800000, v[\v_tmp4+0]
     v_cvt_u32_f32     v[\v_tmp4+0],   v[\v_tmp4+0]
     v_mul_lo_u32      v[\v_tmp4+1],   s[\s_d],      v[\v_tmp4+0]
     v_mul_hi_u32      v[\v_tmp4+2],   s[\s_d],      v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+3],   vcc, 0,     v[\v_tmp4+1]
     v_cmp_ne_i32      s[\s_tmp4:\s_tmp4+1], 0,          v[\v_tmp4+2]
     v_cndmask_b32     v[\v_tmp4+1],   v[\v_tmp4+3],   v[\v_tmp4+1],   s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+1],   v[\v_tmp4+1],   v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
     v_add_co_u32      v[\v_tmp4+0],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
     v_cndmask_b32     v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_tmp4+2],   s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+0],   s[\s_n],   v[\v_tmp4+0]
     v_mul_lo_u32      v[\v_tmp4+1],   s[\s_d],     v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2],   vcc,        s[\s_n],      v[\v_tmp4+1]
     v_cmp_ge_u32      s[\s_tmp4:\s_tmp4+1], s[\s_n],      v[\v_tmp4+1]
     v_cmp_le_u32      s[\s_tmp4+2:\s_tmp4+3],  s[\s_d],    v[\v_tmp4+2]
     v_add_co_u32      v[\v_tmp4+2],   vcc, 1, v[\v_tmp4+0]
     s_and_b64         s[\s_tmp4+2:\s_tmp4+3], s[\s_tmp4:\s_tmp4+1], s[\s_tmp4+2:\s_tmp4+3]
     v_add_co_u32      v[\v_tmp4+1],   vcc, -1,    v[\v_tmp4+0]
     v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+0],   v[\v_tmp4+2],      s[\s_tmp4+2:\s_tmp4+3]
     v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+1],   v[\v_tmp4+2],      s[\s_tmp4:\s_tmp4+1]
     v_cmp_ne_i32      vcc,          s[\s_d],   0
     v_cndmask_b32     v[\v_q],      -1,         v[\v_tmp4+2],      vcc
.endm

.macro .v_clear_nc vid, num
    _v = \vid
    .rept \num
        v_mov_b32 v[_v], 0
        _v = _v + 1
    .endr
.endm

; hard coded thread {1,2,1,4} for e_n1_b_n2
.macro .v_in_load_e_n1_b_n2 v_dst, s_p_buf_in, v_in_os, s_in_stride_n1, s_in_stride_n2, v_flag, s_tmp4
    .v_clear_nc \v_dst, 8
    v_cmp_eq_u32 vcc, 1, v[\v_flag]
    s_and_saveexec_b64 s[\s_tmp4+2:\s_tmp4+3], vcc
    ; {0,0,0,0}
    buffer_load_dword v[\v_dst], v[\v_in_os], s[\s_p_buf_in:\s_p_buf_in+3], 0 offen
    s_mov_b32 s[\s_tmp4], s[\s_in_stride_n2]
    ; {0,0,0,1}
    buffer_load_dword v[\v_dst+1], v[\v_in_os], s[\s_p_buf_in:\s_p_buf_in+3], s[\s_tmp4] offen
    s_add_u32 s[\s_tmp4], s[\s_tmp4], s[\s_in_stride_n2]
    ; {0,0,0,2}
    buffer_load_dword v[\v_dst+2], v[\v_in_os], s[\s_p_buf_in:\s_p_buf_in+3], s[\s_tmp4] offen
    s_add_u32 s[\s_tmp4], s[\s_tmp4], s[\s_in_stride_n2]
    ; {0,0,0,3}
    buffer_load_dword v[\v_dst+3], v[\v_in_os], s[\s_p_buf_in:\s_p_buf_in+3], s[\s_tmp4] offen
    s_mov_b32 s[\s_tmp4], s[\s_in_stride_n1]
    ; {0,1,0,0}
    buffer_load_dword v[\v_dst+4], v[\v_in_os], s[\s_p_buf_in:\s_p_buf_in+3], s[\s_tmp4] offen
    s_add_u32 s[\s_tmp4], s[\s_tmp4], s[\s_in_stride_n2]
    ; {0,1,0,1}
    buffer_load_dword v[\v_dst+5], v[\v_in_os], s[\s_p_buf_in:\s_p_buf_in+3], s[\s_tmp4] offen
    s_add_u32 s[\s_tmp4], s[\s_tmp4], s[\s_in_stride_n2]
    ; {0,1,0,2}
    buffer_load_dword v[\v_dst+6], v[\v_in_os], s[\s_p_buf_in:\s_p_buf_in+3], s[\s_tmp4] offen
    s_add_u32 s[\s_tmp4], s[\s_tmp4], s[\s_in_stride_n2]
    ; {0,1,0,3}
    buffer_load_dword v[\v_dst+7], v[\v_in_os], s[\s_p_buf_in:\s_p_buf_in+3], s[\s_tmp4] offen
    s_or_b64 exec, exec, s[\s_tmp4+2:\s_tmp4+3]
.endm

; update v_flag
.macro .v_in_set_flag v_flag, v_in_ihi, v_in_iwi, s_hi, s_wi, s_tmp2
    ;   flag: 0<= * <wi
    v_cmp_le_i32 vcc, 0, v[\v_in_ihi]
    v_cmp_gt_i32 s[\s_tmp2:\s_tmp2+1], s[\s_hi], v[\v_in_ihi]
    s_and_b64 vcc, vcc, s[\s_tmp2:\s_tmp2+1]
    v_cndmask_b32 v[\v_flag], 0, 1, vcc
    ;   flag: 0<= * <wi
    v_cmp_le_i32 vcc, 0, v[\v_in_iwi]
    v_cmp_gt_i32 s[\s_tmp2:\s_tmp2+1], s[\s_wi], v[\v_in_iwi]
    s_and_b64 vcc, vcc, s[\s_tmp2:\s_tmp2+1]
    v_cndmask_b32 v[\v_flag], 0, v[\v_flag], vcc
.endm

; update v_in_os, v_flag, update v_in_ic, v_in_iy, v_in_ix (zero or possitive), v_in_ihi, v_in_iwi (negative, zero, possitive)
.macro .v_in_move_slice_window_e_cyx v_in_os, v_in_ic, v_in_iy, v_in_ix, v_in_ihi, v_in_iwi, v_flag, s_hi, s_wi, s_y, s_x, s_in_stride_c, s_dilation_h, s_dilation_w, s_in_ic, s_in_iy, s_in_ix, v_idc, v_idy, v_idx, s_tmp2
    ; record old ic, iy, ix
    v_mov_b32 v[\v_idx], v[\v_in_ix]
    v_mov_b32 v[\v_idy], v[\v_in_iy]
    v_mov_b32 v[\v_idc], v[\v_in_ic]

    ; update ix, calculate idx, carry-out to iy
    v_add_u32 v[\v_in_ix], s[\s_in_ix], v[\v_in_ix]
    v_cmp_le_u32 vcc, s[\s_x], v[\v_in_ix]
    s_and_saveexec_b64 s[\s_tmp2:\s_tmp2+1], vcc
    v_subrev_u32 v[\v_in_ix], s[\s_x], v[\v_in_ix]  
    v_add_u32 v[\v_in_iy], 1, v[\v_in_iy]
    s_or_b64 exec, exec, s[\s_tmp2:\s_tmp2+1]
    v_sub_i32 v[\v_idx], v[\v_in_ix], v[\v_idx]

    ; update iy, calculate idy, carry-out to ic
    v_add_u32 v[\v_in_iy], s[\s_in_iy], v[\v_in_iy]
    v_cmp_le_u32 vcc, s[\s_y], v[\v_in_iy]
    s_and_saveexec_b64 s[\s_tmp2:\s_tmp2+1], vcc
    v_subrev_u32 v[\v_in_iy], s[\s_y], v[\v_in_iy]
    v_add_u32 v[\v_in_ic], 1, v[\v_in_ic]
    s_or_b64 exec, exec, s[\s_tmp2:\s_tmp2+1]
    v_sub_i32 v[\v_idy], v[\v_in_iy], v[\v_idy]

    ; update ic, calculate idc, ignore overflow check
    v_add_u32 v[\v_in_ic], s[\s_in_ic], v[\v_in_ic]
    v_sub_u32 v[\v_idc], v[\v_in_ic], v[\v_idc]

    ; calculate offset: idc*(s_hi*s_wi) + idy*s_dilation_h*s_wi + idx*s_dilation_w
    ; we use i24 as multiplier, for 24bit(-8388607 ~ 8388608) is enough for index
    ; also, update ihi, iwi here
    v_mul_i32_i24 v[\v_idy], s[\s_dilation_h], v[\v_idy]
    v_mul_i32_i24 v[\v_idx], s[\s_dilation_w], v[\v_idx]
    v_add_i32 v[\v_in_ihi], v[\v_idy], v[\v_in_ihi]
    v_add_i32 v[\v_in_iwi], v[\v_idx], v[\v_in_iwi]
    v_mul_i32_i24 v[\v_idy], s[\s_wi], v[\v_idy]

    v_add_i32 v[\v_idx], v[\v_idx], v[\v_idy]
    v_mul_lo_u32 v[\v_idc], s[\s_in_stride_c], v[\v_idc]
    v_add_i32 v[\v_idc], v[\v_idc], v[\v_idx]
    v_lshl_add_u32 v[\v_in_os], v[\v_idc], 2, v[\v_in_os]   ; indeed, v_idc here must be possitive

    ; update v_flag
    .v_in_set_flag \v_flag, \v_in_ihi, \v_in_iwi, \s_hi, \s_wi, \s_tmp2
.endm

; update v_wei_os, update v_wei_ic, v_wei_iy, v_wei_ix (zero or possitive)
.macro .v_wei_move_slice_window_e_cyx v_wei_os, v_wei_ic, v_wei_iy, v_wei_ix, s_y, s_x, s_wei_stride_c, s_wei_ic, s_wei_iy, s_wei_ix, v_idc, v_idy, v_idx, s_tmp2
    ; record old ic, iy, ix
    v_mov_b32 v[\v_idx], v[\v_wei_ix]
    v_mov_b32 v[\v_idy], v[\v_wei_iy]
    v_mov_b32 v[\v_idc], v[\v_wei_ic]

    ; update ix, calculate idx, carry-out to iy
    v_add_u32 v[\v_wei_ix], s[\s_wei_ix], v[\v_wei_ix]
    v_cmp_le_u32 vcc, s[\s_x], v[\v_wei_ix]
    s_and_saveexec_b64 s[\s_tmp2:\s_tmp2+1], vcc
    v_subrev_u32 v[\v_wei_ix], s[\s_x], v[\v_wei_ix]
    v_add_u32 v[\v_wei_iy], 1, v[\v_wei_iy]
    s_or_b64 exec, exec, s[\s_tmp2:\s_tmp2+1]
    v_sub_i32 v[\v_idx], v[\v_wei_ix], v[\v_idx]

    ; update iy, calculate idy, carry-out to ic
    v_add_u32 v[\v_wei_iy], s[\s_wei_iy], v[\v_wei_iy]
    v_cmp_le_u32 vcc, s[\s_y], v[\v_wei_iy]
    s_and_saveexec_b64 s[\s_tmp2:\s_tmp2+1], vcc
    v_subrev_u32 v[\v_wei_iy], s[\s_y], v[\v_wei_iy]
    v_add_u32 v[\v_wei_ic], 1, v[\v_wei_ic]
    s_or_b64 exec, exec, s[\s_tmp2:\s_tmp2+1]
    v_sub_i32 v[\v_idy], v[\v_wei_iy], v[\v_idy]

    ; update ic, calculate idc, ignore overflow check
    v_add_u32 v[\v_wei_ic], s[\s_wei_ic], v[\v_wei_ic]
    v_sub_u32 v[\v_idc], v[\v_wei_ic], v[\v_idc]

    ; calculate offset: idc*(s_y*s_x) + idy*s_x + idx
    ; we use i24 as multiplier, for 24bit(-8388607 ~ 8388608) is enough for index
    v_mad_i32_i24 v[\v_idy], s[\s_x], v[\v_idy], v[\v_idx]
    v_mul_lo_u32 v[\v_idc], s[\s_wei_stride_c], v[\v_idc]
    v_add_i32 v[\v_idc], v[\v_idc], v[\v_idy]
    v_lshl_add_u32 v[\v_wei_os], v[\v_idc], 2, v[\v_wei_os]  ; indeed, idc here must be possitive
.endm

.macro .v_wei_load_e_k v_dst, s_p_buf_wei, v_wei_os, s_wei_stride_k, s_tmp2
    buffer_load_dwordx4 v[\v_dst+0:\v_dst+3], v[\v_wei_os], s[\s_p_buf_wei:\s_p_buf_wei+3], 0 offen
    buffer_load_dwordx4 v[\v_dst+4:\v_dst+7], v[\v_wei_os], s[\s_p_buf_wei:\s_p_buf_wei+3], s[\s_wei_stride_k] offen
.endm

.macro .v_write1d_4_strided v_src, s_p_buf_dst, v_dst_os, s_dst_diff, s_dst_os
    buffer_store_dword v[\v_src+0], v[\v_dst_os], s[\s_p_buf_dst:\s_p_buf_dst+3], s[\s_dst_os] offen
    s_add_u32 s[\s_dst_os], s[\s_dst_os], s[\s_dst_diff]
    buffer_store_dword v[\v_src+1], v[\v_dst_os], s[\s_p_buf_dst:\s_p_buf_dst+3], s[\s_dst_os] offen
    s_add_u32 s[\s_dst_os], s[\s_dst_os], s[\s_dst_diff]
    buffer_store_dword v[\v_src+2], v[\v_dst_os], s[\s_p_buf_dst:\s_p_buf_dst+3], s[\s_dst_os] offen
    s_add_u32 s[\s_dst_os], s[\s_dst_os], s[\s_dst_diff]
    buffer_store_dword v[\v_src+3], v[\v_dst_os], s[\s_p_buf_dst:\s_p_buf_dst+3], s[\s_dst_os] offen
.endm

.macro .v_write2d_2_4_strided v_src, s_p_dst, v_dst_os, s_dst_diff1d, s_dst_diff2d, s_dst_os_2
    .v_write1d_4_strided \v_src+0, \s_p_dst, \v_dst_os, \s_dst_diff1d, \s_dst_os_2
    s_add_u32 s[\s_dst_os_2+1], s[\s_dst_os_2+1], s[\s_dst_diff2d]
    s_mov_b32 s[\s_dst_os_2], s[\s_dst_os_2+1]
    .v_write1d_4_strided \v_src+4, \s_p_dst, \v_dst_os, \s_dst_diff1d, \s_dst_os_2
.endm

.macro .v_write3d_4_2_4_strided v_src, s_p_dst, v_dst_os, s_dst_diff1d, s_dst_diff2d, s_dst_diff3d, s_dst_os_3
    .v_write2d_2_4_strided \v_src+0, \s_p_dst, \v_dst_os, \s_dst_diff1d, \s_dst_diff2d, \s_dst_os_3
    s_add_u32 s[\s_dst_os_3+2], s[\s_dst_os_3+2], s[\s_dst_diff3d]
    s_mov_b32 s[\s_dst_os_3+1], s[\s_dst_os_3+2]
    s_mov_b32 s[\s_dst_os_3], s[\s_dst_os_3+1]
    .v_write2d_2_4_strided \v_src+8, \s_p_dst, \v_dst_os, \s_dst_diff1d, \s_dst_diff2d, \s_dst_os_3
    s_add_u32 s[\s_dst_os_3+2], s[\s_dst_os_3+2], s[\s_dst_diff3d]
    s_mov_b32 s[\s_dst_os_3+1], s[\s_dst_os_3+2]
    s_mov_b32 s[\s_dst_os_3], s[\s_dst_os_3+1]
    .v_write2d_2_4_strided \v_src+16, \s_p_dst, \v_dst_os, \s_dst_diff1d, \s_dst_diff2d, \s_dst_os_3
    s_add_u32 s[\s_dst_os_3+2], s[\s_dst_os_3+2], s[\s_dst_diff3d]
    s_mov_b32 s[\s_dst_os_3+1], s[\s_dst_os_3+2]
    s_mov_b32 s[\s_dst_os_3], s[\s_dst_os_3+1]
    .v_write2d_2_4_strided \v_src+24, \s_p_dst, \v_dst_os, \s_dst_diff1d, \s_dst_diff2d, \s_dst_os_3
.endm

.macro .v_write4d_2_4_2_4_strided v_src, s_p_dst, v_dst_os, s_dst_diff1d, s_dst_diff2d, s_dst_diff3d, s_dst_diff4d, s_dst_os_4
    .v_write3d_4_2_4_strided \v_src, \s_p_dst, \v_dst_os, \s_dst_diff1d, \s_dst_diff2d, \s_dst_diff3d, \s_dst_os_4
    s_add_u32 s[\s_dst_os_4+3], s[\s_dst_os_4+3], s[\s_dst_diff4d]
    s_mov_b32 s[\s_dst_os_4+2], s[\s_dst_os_4+3]
    s_mov_b32 s[\s_dst_os_4+1], s[\s_dst_os_4+2]
    s_mov_b32 s[\s_dst_os_4], s[\s_dst_os_4+1]
    .v_write3d_4_2_4_strided \v_src+32, \s_p_dst, \v_dst_os, \s_dst_diff1d, \s_dst_diff2d, \s_dst_diff3d, \s_dst_os_4
.endm

; hard coded thread {2, 4, 2, 1, 4} for k0_k1_n1_b_n2
.macro .v_out_write_k0_k1_n1_b_n2 v_src, s_p_out, v_out_os, s_out_stride_k0, s_out_stride_k1, s_out_stride_n1, s_out_stride_n2, s_dst_os_4
    .v_write4d_2_4_2_4_strided \v_src, \s_p_out, \v_out_os, \s_out_stride_n2, \s_out_stride_n1, \s_out_stride_k1, \s_out_stride_k0, \s_dst_os_4
.endm

.macro .v_fma_4x4_s8 v_c, v_a, v_b
    v_mac_f32 v[\v_c+0] , v[\v_a+0], v[\v_b+0]
    v_mac_f32 v[\v_c+1] , v[\v_a+0], v[\v_b+1]
    v_mac_f32 v[\v_c+2] , v[\v_a+0], v[\v_b+2]
    v_mac_f32 v[\v_c+3] , v[\v_a+0], v[\v_b+3]

    v_mac_f32 v[\v_c+8] , v[\v_a+1], v[\v_b+0]
    v_mac_f32 v[\v_c+9] , v[\v_a+1], v[\v_b+1]
    v_mac_f32 v[\v_c+10], v[\v_a+1], v[\v_b+2]
    v_mac_f32 v[\v_c+11], v[\v_a+1], v[\v_b+3]

    v_mac_f32 v[\v_c+16], v[\v_a+2], v[\v_b+0]
    v_mac_f32 v[\v_c+17], v[\v_a+2], v[\v_b+1]
    v_mac_f32 v[\v_c+18], v[\v_a+2], v[\v_b+2]
    v_mac_f32 v[\v_c+19], v[\v_a+2], v[\v_b+3]

    v_mac_f32 v[\v_c+24], v[\v_a+3], v[\v_b+0]
    v_mac_f32 v[\v_c+25], v[\v_a+3], v[\v_b+1]
    v_mac_f32 v[\v_c+26], v[\v_a+3], v[\v_b+2]
    v_mac_f32 v[\v_c+27], v[\v_a+3], v[\v_b+3]
.endm

; inner 2x2 sub block
.macro .v_fma_8x8_nk nk, v_c, v_a, v_b, v_a_os, v_b_os, k_a_os, k_a_stride, k_b_os, k_b_stride
    .itr_k = 0
    .rept \nk
        .if .itr_k == 0
            ds_read_b128 v[\v_a+0:\v_a+3], v[\v_a_os], offset:\k_a_os
            ds_read_b128 v[\v_b+0:\v_b+3], v[\v_b_os], offset:\k_b_os
            ds_read_b128 v[\v_b+4:\v_b+7], v[\v_b_os], offset:\k_b_os + (\k_b_stride / 2)
            ds_read_b128 v[\v_a+4:\v_a+7], v[\v_a_os], offset:\k_a_os + (\k_a_stride / 2)
        .endif

        s_waitcnt lgkmcnt(2)
        .v_fma_4x4_s8 \v_c+0, \v_a+0, \v_b+0
        s_waitcnt lgkmcnt(1)
        .v_fma_4x4_s8 \v_c+4, \v_a+0, \v_b+4
        .if .itr_k != (\nk - 1)
            ds_read_b128 v[\v_a+0:\v_a+3], v[\v_a_os], offset:\k_a_os + (.itr_k + 1) * \k_a_stride
            s_waitcnt lgkmcnt(1)
        .else
            s_waitcnt lgkmcnt(0)
        .endif
        .v_fma_4x4_s8 \v_c+32, \v_a+4, \v_b+0
        .if .itr_k != (\nk - 1)
            ds_read_b128 v[\v_b+0:\v_b+3], v[\v_b_os], offset:\k_b_os + (.itr_k + 1) * \k_b_stride
        .endif
        .v_fma_4x4_s8 \v_c+36, \v_a+4, \v_b+4

        .if .itr_k != (\nk - 1)
            ds_read_b128 v[\v_b+4:\v_b+7], v[\v_b_os], offset:\k_b_os + (.itr_k + 1) * \k_b_stride + (\k_b_stride / 2)
            ds_read_b128 v[\v_a+4:\v_a+7], v[\v_a_os], offset:\k_a_os + (.itr_k + 1) * \k_a_stride + (\k_a_stride / 2)
        .endif
        .itr_k = .itr_k + 1
    .endr
.endm
;******************************************************************************************

.hsa_code_object_version 2,1
.hsa_code_object_isa
.text
.p2align 8
.amdgpu_hsa_kernel igemm_v4r1_generic

; kernarg offset
.set k_p_in,                8
.set k_p_wei,               16
.set k_p_out,               24
.set k_hi,                  32
.set k_wi,                  36
.set k_n,                   40
.set k_k,                   44
.set k_c,                   48
.set k_ho,                  52
.set k_wo,                  56
.set k_stride_h,            60
.set k_stride_w,            64
.set k_dilation_h,          68
.set k_dilation_w,          72
.set k_pad_h,               76
.set k_pad_w,               80
.set k_y,                   84
.set k_x,                   88
.set k_end,                 92

; sgpr
.set s_ka,                  0
.set s_bx,                  2
.set s_p_in,                4
.set s_p_wei,               6
.set s_hi,                  8
.set s_wi,                  9
.set s_n,                   10
.set s_k,                   11
.set s_c,                   12
.set s_ho,                  13
.set s_wo,                  14
.set s_stride_h,            15
.set s_stride_w,            16
.set s_dilation_h,          17
.set s_dilation_w,          18
.set s_pad_h,               19
.set s_pad_w,               20
.set s_y,                   21
.set s_x,                   22
.set s_p_out,               24
.set s_block_ik,            26
.set s_block_ib,            27
.set s_in_stride_c,         28
.set s_in_stride_n2,        29
.set s_in_stride_n1,        30
.set s_in_ic,               31
.set s_in_iy,               32
.set s_in_ix,               33
.set s_wei_stride_c,        34
.set s_wei_stride_k,        35
.set s_wei_ic,              s_in_ic     ; weight&input ic, iy, ix from EPerBlock is the same
.set s_wei_iy,              s_in_iy
.set s_wei_ix,              s_in_ix
.set s_out_stride_k0,       36
.set s_out_stride_k1,       37
.set s_out_stride_n1,       38
.set s_out_stride_n2,       39
.set s_kitr,                0
.set s_tmp,                 40
.set s_p_buf_in,            s_p_in      ; 4 sgpr used for MUBUF
.set s_p_buf_wei,           44
.set s_p_buf_out,           s_p_out
.set s_end,                 47

; vgpr
.set v_c,                   0
.set v_a,                   64
.set v_b,                   72
.set v_la,                  80
.set v_lb,                  88
.set v_in_os,               96
.set v_wei_os,              97
.set v_sta_os,              98
.set v_stb_os,              99
.set v_lda_os,              100
.set v_ldb_os,              101
.set v_out_os,              102
.set v_flag,                103
.set v_in_ic,               104
.set v_in_iy,               105
.set v_in_ix,               106
.set v_in_in0,              31
.set v_in_iho,              32
.set v_in_iwo,              33
.set v_in_ihi,              107
.set v_in_iwi,              108
.set v_wei_ic,              109
.set v_wei_iy,              110
.set v_wei_ix,              111
.set v_in_ie,               40
.set v_in_ib,               41
.set v_wei_ie,              42
.set v_wei_ik,              43
.set v_out_ik0,             44
.set v_out_ik1,             45
.set v_out_ib,              46
.set v_gemm_in,             47
.set v_gemm_im,             48
.set v_tmp,                 60
.set v_idc,                 112
.set v_idy,                 113
.set v_idx,                 114
.set v_end,                 114

; tunable parameters, note here is only for record
.set t_b_per_block,                         16
.set t_k_per_block,                         128
.set t_e_per_block,                         16
.set t_gemm_n_repeat,                       2
.set t_gemm_m_per_thread_subc,              4
.set t_gemm_n_per_thread_subc,              4
.set t_gemm_m_level0_cluster,               4
.set t_gemm_n_level0_cluster,               4
.set t_gemm_m_level1_cluster,               4
.set t_gemm_n_level1_cluster,               4
.set t_gemm_k_per_thread_loop,              1
.set t_gemm_data_per_read_a,                4
.set t_gemm_data_per_read_b,                4
.set t_in_block_copy_sub_lengths_e,         1
.set t_in_block_copy_sub_lengths_n1,        2
.set t_in_block_copy_sub_lengths_b,         1
.set t_in_block_copy_sub_lengths_n2,        4
.set t_in_block_copy_cluster_lengths_e,     16
.set t_in_block_copy_cluster_lengths_n1,    1
.set t_in_block_copy_cluster_lengths_b,     16
.set t_in_block_copy_cluster_lengths_n2,    1
; TODO: in access order
.set t_in_block_copy_srcdata_per_read_b,    1
.set t_in_block_copy_dstdata_per_write_n2,  4
.set t_wei_block_copy_sub_Lengths_e,        4
.set t_wei_block_copy_sub_lengths_k,        2
.set t_wei_block_copy_cluster_lengths_e,    4
.set t_wei_block_copy_cluster_lengths_k,    64
; TODO: wei access order
.set t_wei_block_copy_srcdata_per_read_e,   4
.set t_wei_block_copy_dstdata_per_write_k,  2

igemm_v4r1_generic:
    .amd_kernel_code_t
        enable_sgpr_kernarg_segment_ptr = 1                 ;
        user_sgpr_count = 2
        enable_sgpr_workgroup_id_x = 1                      ;        blockIdx.x
        enable_vgpr_workitem_id = 0
        is_ptr64 = 1
        float_mode = 192
        workgroup_group_segment_byte_size = 16384 * 2
        kernarg_segment_byte_size = k_end
        wavefront_sgpr_count = s_end+1+2*3                  ; VCC, FLAT_SCRATCH and XNACK must be counted
        workitem_vgpr_count = v_end+1
        granulated_workitem_vgpr_count = v_end/4            ; (workitem_vgpr_count-1)/4
        granulated_wavefront_sgpr_count = (s_end+2*3)/8     ; (wavefront_sgpr_count-1)/8
    .end_amd_kernel_code_t
    s_load_dwordx4  s[s_p_in:s_p_in+3],         s[s_ka:s_ka+1],     0+k_p_in
    s_load_dwordx2  s[s_p_out:s_p_out+1],       s[s_ka:s_ka+1],     0+k_p_out
    s_load_dwordx8  s[s_hi:s_hi+7],             s[s_ka:s_ka+1],     0+k_hi
    s_load_dwordx4  s[s_stride_w:s_stride_w+3], s[s_ka:s_ka+1],     0+k_stride_w
    s_load_dwordx2  s[s_pad_w:s_pad_w+1],       s[s_ka:s_ka+1],     0+k_pad_w
    s_load_dword    s[s_x],                     s[s_ka:s_ka+1],     0+k_x

    ;   in e_n1_b_n2 cluster: {16,1,16,1}, {1,2,1,4}, order:{0,1,3,2}
    v_and_b32 v[v_in_ib], 15, v0                            ; TUNABLE
    v_lshrrev_b32 v[v_in_ie], 4, v0                         ; TUNABLE
    ;   wei e_k cluster: {4, 64}, {4, 2}, order:{1,0}
    v_lshrrev_b32 v[v_tmp], 2, v0
    v_lshlrev_b32 v[v_wei_ik], 1, v[v_tmp]                  ; TUNABLE
    v_and_b32 v[v_tmp+1], 3, v0
    v_lshlrev_b32 v[v_wei_ie], 2, v[v_tmp+1]                ; TUNABLE
    s_waitcnt lgkmcnt(0)

    ;   note here not consider sizeof(float)
    s_mul_i32 s[s_out_stride_k1], s[s_ho], s[s_wo]
    s_lshl_b32 s[s_out_stride_k0], s[s_out_stride_k1], 6    ; TUNABLE
    s_mul_i32 s[s_out_stride_n2], s[s_k], s[s_out_stride_k1]
    s_lshl_b32 s[s_out_stride_n1], s[s_out_stride_n2], 2    ; TUNABLE
    s_mul_i32 s[s_in_stride_c], s[s_hi], s[s_wi]
    s_mul_i32 s[s_in_stride_n2], s[s_c], s[s_in_stride_c]
    ;s_lshl_b32 s[s_in_stride_n1], s[s_in_stride_n2], 2      ; TUNABLE
    s_mul_i32 s[s_wei_stride_c], s[s_y], s[s_x]
    s_mul_i32 s[s_wei_stride_k], s[s_c], s[s_wei_stride_c]
    s_mov_b64 s[s_p_buf_wei:s_p_buf_wei+1], s[s_p_wei:s_p_wei+1]
    s_mov_b32 s[s_p_buf_in+2], 0xffffffff
    s_mov_b32 s[s_p_buf_in+3], 0x27000
    s_mov_b32 s[s_p_buf_wei+2], 0xffffffff
    s_mov_b32 s[s_p_buf_wei+3], 0x27000

    s_lshr_b32 s[s_tmp], s[s_n], 3                          ; TUNABLE   N0 = N / (N1 * N2);
    s_mul_i32 s[s_tmp+1], s[s_out_stride_k1], s[s_tmp]      ;           B = N0 * Ho * Wo;
    s_lshr_b32 s[0], s[s_tmp+1], 4                          ; TUNABLE   BBlockWork = B / BPerBlock;
    .v_u32_div_ss 10, s_bx, 0, v_tmp, s_tmp                 ;           KBlockID = get_block_1d_id() / BBlockWork;
    v_readfirstlane_b32 s[s_tmp], v[10]
    s_mul_i32 s[s_tmp+2], s[s_tmp], s[0]                    ;           BBlockID = get_block_1d_id() % BBlockWork;
    s_sub_i32 s[s_tmp+1], s[s_bx], s[s_tmp+2]

    s_lshl_b32 s[s_block_ik], s[s_tmp], 7                   ; TUNABLE   k_block_data_on_global = KBlockID * KPerBlock;
    s_lshl_b32 s[s_block_ib], s[s_tmp+1], 4                 ; TUNABLE   b_block_data_on_global = BBlockID * BPerBlock;

    ; input
    ;
    ; from tensor transform, input tensor is divided into folowing dim iterator (current position):
    ;               ic, iy, ix, in0, iho, iwo
    ; from iy, ix, iho, iwo, can get the input width, height iterator:
    ;               -> ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
    ;               -> iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w
    ; hence, can calculate input offset from above iterator:
    ;       in_offset: in0 * (8*s_c*s_hi*s_wi) + ic * (s_hi*s_wi) + ihi * s_wi + iwi
    ;
    ; for each MoveSliceWindow, need move <EPerBlock, 0, 0, 0>, the diff can be divided into:
    ;               dc, dy, dx      (e=c*y*x, can all be sgpr)
    ; new c, y, x iterator:
    ;               ix_new = (ix+dx)%s_x
    ;               iy_new = (iy+dy+(ix+dx)/s_x)%s_y
    ;               ic_new = (ic+dc+(iy+dy+(ix+dx)/s_x)/s_y)   (no check overflow)
    ; hence the iterator diff (may have negative):
    ;               idx = ix_new - ix
    ;               idy = iy_new - iy
    ;               idc = ic_new - ic
    ;
    ; hence for offset, the diff should be:
    ;   in_offset_diff: idc*(s_hi*s_wi) + idy*s_dilation_h*s_wi + idx*s_dilation_w
    ;
    ;   note here:
    ;       1) idc can only be 0 or possitive, idx, idy can be negative, possitive, 0
    ;       2) ic, iy, ix need be updated to ic_new, iy_new, ix_new
    ;
    ; 1. e_n1_b_n2:b
    ;   1). transform: b -> n0*ho*wo
    v_add_u32 v[1], s[s_block_ib], v[v_in_ib]
    .v_u32_div_vs v_in_in0, 1, s_out_stride_k1, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp], s[s_out_stride_k1], v[v_in_in0]
    v_sub_u32 v[1], v[1], v[v_tmp]
    .v_u32_div_vs v_in_iho, 1, s_wo, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp], s[s_wo], v[v_in_iho]
    v_sub_u32 v[v_in_iwo], v[1], v[v_tmp]

    ; 2. e_n1_b_n2:e
    ;   1). e -> c*y*x
    .v_u32_div_vs v_in_ic, v_in_ie, s_wei_stride_c, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp], s[s_wei_stride_c], v[v_in_ic]
    v_sub_u32 v[1], v[v_in_ie], v[v_tmp]
    .v_u32_div_vs v_in_iy, 1, s_x, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp], s[s_x], v[v_in_iy]
    v_sub_u32 v[v_in_ix], v[1], v[v_tmp]

    ; 2) iho, iwo, iy, ix -> hip, wip
    v_mul_lo_u32 v[v_tmp], s[s_stride_h], v[v_in_iho]
    v_mul_lo_u32 v[v_tmp+1], s[s_stride_w], v[v_in_iwo]
    v_mul_lo_u32 v[v_tmp+2], s[s_dilation_h], v[v_in_iy]
    v_mul_lo_u32 v[v_tmp+3], s[s_dilation_w], v[v_in_ix]

    ; 3). hip, wip -> hi, wi
    v_add_u32 v[v_tmp], v[v_tmp], v[v_tmp+2]
    v_add_u32 v[v_tmp+1], v[v_tmp+1], v[v_tmp+3]
    v_sub_i32 v[v_in_ihi], v[v_tmp], s[s_pad_h]
    v_sub_i32 v[v_in_iwi], v[v_tmp+1], s[s_pad_w]

    ; set input flag
    .v_in_set_flag v_flag, v_in_ihi, v_in_iwi, s_hi, s_wi, s_tmp

    ; in offset: from ihi, iwi, ic, in, calculate v_in_os
    v_mul_lo_u32 v[v_tmp], s[s_wi], v[v_in_ihi]
    v_add_u32 v[v_tmp], v[v_tmp], v[v_in_iwi]
    v_mul_lo_u32 v[v_tmp+1], s[s_in_stride_c], v[v_in_ic]
    v_add_u32 v[v_tmp], v[v_tmp], v[v_tmp+1]
    v_lshlrev_b32 v[v_tmp+1], 3, v[v_in_in0]                ; TUNABLE
    v_mul_lo_u32 v[v_tmp+1], s[s_in_stride_n2], v[v_tmp+1]
    v_add_lshl_u32 v[v_in_os], v[v_tmp], v[v_tmp+1], 2

    s_lshl_b32 s[s_in_stride_n2], s[s_in_stride_n2], 2
    s_lshl_b32 s[s_in_stride_n1], s[s_in_stride_n2], 2      ; TUNABLE

    ; load input from global
    .v_in_load_e_n1_b_n2 v_b, s_p_buf_in, v_in_os, s_in_stride_n1, s_in_stride_n2, v_flag, s_tmp

    ; calculate SliceWindow EPerBlock(e=c*y*x). this is same for both input/weight
    s_mov_b32 s[1], t_e_per_block                           ; TUNABLE
    .v_u32_div_ss 1, 1, s_wei_stride_c, v_tmp, s_tmp
    v_readfirstlane_b32 s[s_in_ic], v[1]
    s_mul_i32 s[s_tmp], s[s_wei_stride_c], s[s_in_ic]
    s_sub_i32 s[1], s[1], s[s_tmp]
    .v_u32_div_ss 1, 1, s_x, v_tmp, s_tmp
    v_readfirstlane_b32 s[s_in_iy], v[1]
    s_mul_i32 s[s_tmp], s[s_x], s[s_in_iy]
    s_sub_i32 s[s_in_ix], s[1], s[s_tmp]

    ; c mapping, GetBeginOfThreadMatrixC. calculate output mapping here
    v_and_b32 v[2], 15, v0                                  ; TUNABLE
    v_and_b32 v[v_tmp], 3, v[2]
    v_lshrrev_b32 v[v_tmp+1], 2, v[2]

    v_lshrrev_b32 v[6], 4, v0
    v_and_b32 v[v_tmp+2], 3, v[6]
    v_lshrrev_b32 v[v_tmp+3], 2, v[6]

    v_lshl_or_b32 v[v_gemm_in], v[v_tmp+2], 2, v[v_tmp]               ; in
    v_lshl_or_b32 v[v_gemm_im], v[v_tmp+3], 2, v[v_tmp+1]             ; im
    v_lshlrev_b32 v[v_ldb_os], 2+2, v[v_gemm_in]
    v_lshlrev_b32 v[v_lda_os], 2+2, v[v_gemm_im]

    ; weight
    ;
    ; from tensor transform, weight tensor is divided into following dim iterator (current position):
    ;               ic, iy, ix, ik
    ; hence, can calculate weight offset from above iterator:
    ;       in_offset: ik*(s_c*s_y*s_x) + ic*(s_y*s_x) + iy*s_x + ix
    ;
    ; for each MoveSliceWindow, need move <EPerBlock, 0>, the diff can be divided into:
    ;               dc, dy, dx      (e=c*y*x, can all be sgpr)
    ; new c, y, x iterator:
    ;               ix_new = (ix+dx)%s_x
    ;               iy_new = (iy+dy+(ix+dx)/s_x)%s_y
    ;               ic_new = (ic+dc+(iy+dy+(ix+dx)/s_x)/s_y)   (no check overflow)
    ; hence the iterator diff (may have negative):
    ;               idx = ix_new - ix
    ;               idy = iy_new - iy
    ;               idc = ic_new - ic
    ;
    ; hence for offset, the diff should be:
    ;   wei_offset_diff: idc*(s_y*s_x) + idy*s_x + idx
    ;
    ;   note here:
    ;       1) idx can only be 0 or possitive, idx, idy can be negative, possitive, 0
    ;       2) ic, iy, ix need be updated to ic_new, iy_new, ix_new
    ;
    ; e_k:e->c*y*x
    .v_u32_div_vs v_wei_ic, v_wei_ie, s_wei_stride_c, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp], s[s_wei_stride_c], v[v_wei_ic]
    v_sub_u32 v[1], v[v_wei_ie], v[v_tmp]
    .v_u32_div_vs v_wei_iy, 1, s_x, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp], s[s_x], v[v_wei_iy]
    v_sub_u32 v[v_wei_ix], v[1], v[v_tmp]

    ; offset: from ic, iy, ix, ik, calculate v_wei_os
    v_mul_lo_u32 v[v_tmp], s[s_wei_stride_c], v[v_wei_ic]
    v_mul_lo_u32 v[v_tmp+1], s[s_x], v[v_wei_iy]
    v_add3_u32 v[v_wei_os], v[v_tmp], v[v_tmp+1], v[v_wei_ix]

    v_add_u32 v[v_tmp], s[s_block_ik], v[v_wei_ik]
    v_mul_lo_u32 v[v_tmp+1], s[s_wei_stride_k], v[v_tmp]
    v_add_lshl_u32 v[v_wei_os], v[v_wei_os], v[v_tmp+1], 2

    s_lshl_b32 s[s_wei_stride_k], s[s_wei_stride_k], 2

    ; load wei from global
    .v_wei_load_e_k v_a, s_p_buf_wei, v_wei_os, s_wei_stride_k, s_tmp

    ; out diff
    ; k_thread_data_on_global = k_block_data_on_global + c_thread_mtx_on_block.row;
    ; k_thread_data_on_global / K1
    ; k_thread_data_on_global % K1
    v_lshlrev_b32 v[v_tmp+1], 2, v[v_gemm_im]           ; TUNABLE
    v_add_u32 v[v_tmp], s[s_block_ik], v[v_tmp+1]
    v_lshrrev_b32 v[v_out_ik0], 6, v[v_tmp]
    v_and_b32 v[v_out_ik1], 63, v[v_tmp]

    ; b_thread_data_on_global = b_block_data_on_global + c_thread_mtx_on_block.col / N2;
    v_add_u32 v[v_out_ib], s[s_block_ib], v[v_gemm_in]
    .v_u32_div_vs 21, v_out_ib, s_out_stride_k1, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp+1], s[s_out_stride_k1], v21
    v_sub_u32 v22, v[v_out_ib], v[v_tmp+1]
    .v_u32_div_vs 23, 22, s_wo, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp+1], s[s_wo], v23
    v_sub_u32 v24, v22, v[v_tmp+1]
    ; 21:n0, 23:ho, 24:wo

    v_mul_lo_u32 v[v_tmp], s[s_wo], v23
    s_mul_i32 s[s_tmp], s[s_k], s[s_out_stride_k1]
    v_add_u32 v[v_out_os], v[v_tmp], v24
    s_lshl_b32 s[s_tmp+1], s[s_tmp], 3
    v_mul_lo_u32 v[v_tmp], s[s_tmp+1], v21
    v_add_u32 v[v_out_os], v[v_out_os], v[v_tmp]

    s_lshl_b32 s[s_out_stride_k0], s[s_out_stride_k0], 2
    v_lshl_or_b32 v[v_tmp], v[v_out_ik0], 6, v[v_out_ik1]
    s_lshl_b32 s[s_out_stride_n1], s[s_out_stride_n1], 2
    v_mul_lo_u32 v[v_tmp+1], s[s_out_stride_k1], v[v_tmp]
    s_lshl_b32 s[s_out_stride_n2], s[s_out_stride_n2], 2
    v_add_u32 v[v_out_os], v[v_out_os], v[v_tmp+1]
    s_lshl_b32 s[s_out_stride_k1], s[s_out_stride_k1], 2
    v_lshlrev_b32 v[v_out_os], 2, v[v_out_os]

    ; in lds offset block_e_n1_b_n2:{16,2,16,4}, {16, 1, 16, 1}
    v_lshlrev_b32 v[v_tmp], 7, v[v_in_ie]
    v_lshl_or_b32 v[v_tmp+1], v[v_in_ib], 2, v[v_tmp]
    v_lshlrev_b32 v[v_stb_os], 2, v[v_tmp+1]

    ; wei lds offset block_e_k:{16, 128}, {4, 64}
    v_lshl_or_b32 v[v_tmp], v[v_wei_ie], 7, v[v_wei_ik]
    v_lshlrev_b32 v[v_sta_os], 2, v[v_tmp]

    s_mov_b32 s[s_p_buf_out+2], 0xffffffff
    s_mov_b32 s[s_p_buf_out+3], 0x27000
    .v_clear_nc v_c, 64

    ; start FMA loop
    s_waitcnt vmcnt(2)
    ds_write_b128 v[v_stb_os], v[v_b:v_b+3]  offset:0
    ds_write_b128 v[v_stb_os], v[v_b+4:v_b+7]  offset:64*4

    s_waitcnt vmcnt(0)
    v_swap_b32 v[v_a+1], v[v_a+4]       ; caution! swap is half speed
    v_swap_b32 v[v_a+3], v[v_a+6]
    ds_write2st64_b64 v[v_sta_os], v[v_a+0:v_a+1], v[v_a+4:v_a+5], offset0:16+0, offset1:16+1
    ds_write2st64_b64 v[v_sta_os], v[v_a+2:v_a+3], v[v_a+6:v_a+7], offset0:16+2, offset1:16+3

    ; E = C * Y * X
    s_mul_i32 s[s_tmp], s[s_c], s[s_wei_stride_c]
    s_sub_i32 s[s_kitr], s[s_tmp], t_e_per_block
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_v4r1_generic_fma_end


    .v_in_move_slice_window_e_cyx v_in_os, v_in_ic, v_in_iy, v_in_ix, v_in_ihi, v_in_iwi, v_flag, s_hi, s_wi, s_y, s_x, s_in_stride_c, s_dilation_h, s_dilation_w, s_in_ic, s_in_iy, s_in_ix, v_idc, v_idy, v_idx, s_tmp
    .v_wei_move_slice_window_e_cyx v_wei_os, v_wei_ic, v_wei_iy, v_wei_ix, s_y, s_x, s_wei_stride_c, s_wei_ic, s_wei_iy, s_wei_ix, v_idc, v_idy, v_idx, s_tmp
    v_xor_b32 v[v_stb_os], 0x4000, v[v_stb_os]
    v_xor_b32 v[v_sta_os], 0x4000, v[v_sta_os]
    s_waitcnt lgkmcnt(0)
    s_barrier

    .v_in_load_e_n1_b_n2 v_b, s_p_buf_in, v_in_os, s_in_stride_n1, s_in_stride_n2, v_flag, s_tmp
    .v_wei_load_e_k v_a, s_p_buf_wei, v_wei_os, s_wei_stride_k, s_tmp

L_igemm_v4r1_generic_fma_body:
    ; do fma accumulate
    .itr_k = 0
    .rept 15
        .if .itr_k == 0
            ds_read_b128 v[v_la+0:v_la+3], v[v_lda_os], offset:8192
            ds_read_b128 v[v_lb+0:v_lb+3], v[v_ldb_os], offset:0
            ds_read_b128 v[v_lb+4:v_lb+7], v[v_ldb_os], offset:0 + (512 / 2)
            ds_read_b128 v[v_la+4:v_la+7], v[v_lda_os], offset:8192 + (512 / 2)
        .endif

        s_waitcnt lgkmcnt(2)
        .v_fma_4x4_s8 v_c+0, v_la+0, v_lb+0

        s_waitcnt lgkmcnt(1)
        .v_fma_4x4_s8 v_c+4, v_la+0, v_lb+4

        ds_read_b128 v[v_la+0:v_la+3], v[v_lda_os], offset:8192 + (.itr_k + 1) * 512
        s_waitcnt lgkmcnt(1)
        .v_fma_4x4_s8 v_c+32, v_la+4, v_lb+0

        ds_read_b128 v[v_lb+0:v_lb+3], v[v_ldb_os], offset:0 + (.itr_k + 1) * 512
        .v_fma_4x4_s8 v_c+36, v_la+4, v_lb+4

        ds_read_b128 v[v_lb+4:v_lb+7], v[v_ldb_os], offset:0 + (.itr_k + 1) * 512 + (512 / 2)
        ds_read_b128 v[v_la+4:v_la+7], v[v_lda_os], offset:8192 + (.itr_k + 1) * 512 + (512 / 2)

        .itr_k = .itr_k + 1
    .endr

    ; last unrool
    v_xor_b32 v[v_ldb_os], 0x4000, v[v_ldb_os]
    v_xor_b32 v[v_lda_os], 0x4000, v[v_lda_os]
    s_waitcnt lgkmcnt(2)
    .v_fma_4x4_s8 v_c+0, v_la+0, v_lb+0
    s_waitcnt lgkmcnt(1)
    .v_fma_4x4_s8 v_c+4, v_la+0, v_lb+4

    s_waitcnt vmcnt(2)
    ds_write_b128 v[v_stb_os], v[v_b:v_b+3]  offset:0
    ds_write_b128 v[v_stb_os], v[v_b+4:v_b+7]  offset:64*4

    s_waitcnt vmcnt(0)
    v_swap_b32 v[v_a+1], v[v_a+4]       ; caution! swap is half speed
    v_swap_b32 v[v_a+3], v[v_a+6]
    ds_write2st64_b64 v[v_sta_os], v[v_a+0:v_a+1], v[v_a+4:v_a+5], offset0:16+0, offset1:16+1
    ds_write2st64_b64 v[v_sta_os], v[v_a+2:v_a+3], v[v_a+6:v_a+7], offset0:16+2, offset1:16+3

    s_sub_i32 s[s_kitr], s[s_kitr], t_e_per_block
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_v4r1_generic_fma_finishing

    .v_in_move_slice_window_e_cyx v_in_os, v_in_ic, v_in_iy, v_in_ix, v_in_ihi, v_in_iwi, v_flag, s_hi, s_wi, s_y, s_x, s_in_stride_c, s_dilation_h, s_dilation_w, s_in_ic, s_in_iy, s_in_ix, v_idc, v_idy, v_idx, s_tmp
    .v_wei_move_slice_window_e_cyx v_wei_os, v_wei_ic, v_wei_iy, v_wei_ix, s_y, s_x, s_wei_stride_c, s_wei_ic, s_wei_iy, s_wei_ix, v_idc, v_idy, v_idx, s_tmp

    s_waitcnt lgkmcnt(4)
    .v_fma_4x4_s8 v_c+32, v_la+4, v_lb+0
    v_xor_b32 v[v_stb_os], 0x4000, v[v_stb_os]
    v_xor_b32 v[v_sta_os], 0x4000, v[v_sta_os]

    s_waitcnt lgkmcnt(0)
    s_barrier

    .v_in_load_e_n1_b_n2 v_b, s_p_buf_in, v_in_os, s_in_stride_n1, s_in_stride_n2, v_flag, s_tmp
    .v_wei_load_e_k v_a, s_p_buf_wei, v_wei_os, s_wei_stride_k, s_tmp

    .v_fma_4x4_s8 v_c+36, v_la+4, v_lb+4
    s_branch L_igemm_v4r1_generic_fma_body
L_igemm_v4r1_generic_fma_finishing:
    s_waitcnt lgkmcnt(4)
    .v_fma_4x4_s8 v_c+32, v_la+4, v_lb+0
    .v_fma_4x4_s8 v_c+36, v_la+4, v_lb+4
L_igemm_v4r1_generic_fma_end:
    s_mov_b32 s[s_tmp], 0
    s_mov_b32 s[s_tmp+1], 0
    s_mov_b32 s[s_tmp+2], 0
    s_mov_b32 s[s_tmp+3], 0

    s_waitcnt lgkmcnt(0)
    s_barrier
    .v_fma_8x8_nk 16, v_c, v_la, v_lb, v_lda_os, v_ldb_os, 8192, 512, 0, 512

L_igemm_v4r1_generic_fma_out:
    .v_out_write_k0_k1_n1_b_n2 v_c, s_p_buf_out, v_out_os, s_out_stride_k0, s_out_stride_k1, s_out_stride_n1, s_out_stride_n2, s_tmp

    s_endpgm
