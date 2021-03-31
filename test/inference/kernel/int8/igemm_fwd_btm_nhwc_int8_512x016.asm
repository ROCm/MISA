.set k_p_in, 0
.set k_p_wei, 8
.set k_p_out, 16
.set k_hi, 24
.set k_wi, 28
.set k_n, 32
.set k_k, 36
.set k_c, 40
.set k_ho, 44
.set k_wo, 48
.set k_stride_h, 52
.set k_stride_w, 56
.set k_dilation_h, 60
.set k_dilation_w, 64
.set k_pad_h, 68
.set k_pad_w, 72
.set k_y, 76
.set k_x, 80
.set k_group, 84
.set k_batch_m, 88
.set k_stride_m, 92
.set k_alpha, 96
.set k_beta, 100
.set k_gamma, 104
.set k_magic_0, 108
.set k_magic_1, 112
.set k_magic_2, 116
.set k_shift_pack_0, 120
.set k_n_dword, 16

.set s_ka, 0
.set s_bx, 2            ; bx, ho*wo
.set s_block_ig, 3      ; by, group
.set s_block_in, 4      ; bz, batch
.set s_p_in, 6
.set s_p_wei, 8
.set s_p_out, 10
.set s_hi, 16
.set s_wi, 17
.set s_n, 18
.set s_k, 19
.set s_c, 20
.set s_ho, 21
.set s_wo, 22
.set s_stride_h, 23
.set s_stride_w, 24
.set s_dilation_h, 25
.set s_dilation_w, 26
.set s_pad_h, 27
.set s_pad_w, 28
.set s_y, 29
.set s_x, 30
.set s_group, 31
.set s_batch_m, 32
.set s_stride_m, 33
.set s_alpha, 34
.set s_beta, 35
.set s_gamma, 36
.set s_magic_0, 37
.set s_magic_1, 38
.set s_magic_2, 39
.set s_shift_pack_0, 40
.set s__pack_0, 41
.set s_shift_m0, 42
.set s_shift_m1, s_shift_pack_0
.set s_shift_m2, 43
.set s_in_stride_wi, 12
.set s_in_stride_n, 13
.set s_wei_stride_k, 14
.set s_out_stride_wo, 15
.set s_out_stride_n, 44
.set s_in_diff_hi, 45
.set s_in_diff_wi, 46
.set s_dilation_w_x, 47
.set s_move_slice_k_ix, 48

.set s_kitr, 1
.set s_wei_offset, 49
.set s_out_stride, s_wei_offset
.set s_sld_b_stride, 50
.set s_br, 51
.set s_ib_stride, 52
.set s_block_ik, 53
.set s_block_ib, 54
.set s_0xff, 55
.set s_tmp, 56
.set s_end, 62

; magic_0: x
; magic_1: wo

.set v_c,               0
.set v_sld_b_os,        64
.set v_ax,              65
.set v_ay,              81
.set v_ib,              97
.set v_b,               98
.set v_gld_b,           v_b
.set v_wei_iy_list,     v_b+8
.set v_wei_ix_list,     v_b+10
.set v_wei_flag,        v_b+12
.set v_wei_os,          v_b+14
.set v_tmp,             v_b+16
.set v_wei_ik,          v_ay
.set v_wei_ic,          v_ay+1
.set v_wei_ie,          v_ay+2
.set v_wei_flag_ik,     v_ay+3
.set v_sst_b_os,        v_ay+4
.set v_in_os,           162
.set v_in_ihi,          166
.set v_in_iwi,          170
.set v_in_flag,         174
.set v_out_os,          178
.set v_out_flag,        182
.set v_tid,             186
.set v_end,             188
.set v_c_buf,           v_b

; short wide igemv
.text
.globl igemm_fwd_btm_nhwc_int8_512x16x16_r2
.p2align 8

.type igemm_fwd_btm_nhwc_int8_512x16x16_r2,@function
igemm_fwd_btm_nhwc_int8_512x16x16_r2:
    s_load_dwordx2  s[s_p_in+0:s_p_in+1],    s[s_ka+0:s_ka+1],    0+k_p_in
    s_load_dwordx4  s[s_p_wei+0:s_p_wei+3],  s[s_ka+0:s_ka+1],    0+k_p_wei
    s_load_dwordx16 s[s_hi+0:s_hi+15],    s[s_ka+0:s_ka+1],    0+k_hi
    s_load_dwordx8  s[s_batch_m:s_batch_m+7],    s[s_ka+0:s_ka+1],    0+k_batch_m
    s_load_dword  s[s_shift_pack_0],    s[s_ka+0:s_ka+1],    0+k_shift_pack_0
    v_mov_b32       v[v_tid], v0
    s_mov_b32 s[s_ib_stride], 128
    s_mov_b32 s[s_0xff], 0xff

    ; calculate wei offset, 16x8, 16 for k, 8 for yxc, 8 for yx, 1 for c
    v_lshrrev_b32 v[v_wei_ik], 3, v0
    s_mov_b32 s[s_tmp], k_n_dword*4 * 4
    v_and_b32 v[v_wei_ie], 7, v0                            ; yx
    ;s_lshl_b32 s[s_block_ig], s[s_block_ig], 1
    v_mov_b32 v[v_wei_ic], 0
    ;s_lshl_b32 s[s_block_in], s[s_block_in], 1
    ;v_lshrrev_b32 v[v_tmp+4], 1, v0
    v_mov_b32 v[v_ib], v0
    v_mul_u32_u24 v[v_tmp+5],   s[s_tmp]  ,v[v_wei_ie]
    v_lshlrev_b32 v[v_sst_b_os], 2, v[v_wei_ik]             ; store, k*n*k_pack, ds_write2 if possible, n*k_pack->16dword, pad to x
    v_mov_b32 v[v_sld_b_os], 0                              ; load   
    v_lshlrev_b32 v[v_wei_ic], 4, v[v_wei_ic]               ; 16xc, k_pack, 4x dword
    v_add_nc_u32 v[v_sst_b_os], v[v_sst_b_os], v[v_tmp+5]   ; note, do not use or due to pad

    s_waitcnt lgkmcnt(0)
    s_bfe_u32 s[s_shift_m2], s[s_shift_pack_0], 0x00080010      ; offset:16, width:8
    s_lshr_b32 s[s_tmp+3], s[s_k], 4
    s_bfe_u32 s[s_shift_m0], s[s_shift_pack_0], 0x00080000      ; offset:0, width:8
    .mdiv_u32_rem_ss s_tmp+4,s_tmp+5,s_bx,s_magic_2,s_shift_m2,s_tmp+3,s_tmp
    s_lshl_b32 s[s_block_ib], s[s_tmp+5], 9                     ; 512
    s_lshl_b32 s[s_block_ik], s[s_tmp+4], 4
    v_add_nc_u32 v[v_ib], s[s_block_ib],  v[v_ib]
    s_mul_i32 s[s_tmp], s[s_x], s[s_c]
    v_add_nc_u32 v[v_wei_ik], s[s_block_ik], v[v_wei_ik]

    v_mad_u32_u24 v[v_tmp+1], s[s_c], v[v_wei_ie], v[v_wei_ic]
    s_mul_i32 s[s_wei_stride_k], s[s_tmp], s[s_y]
    s_lshl_b32 s[s_wei_offset], s[s_c], 3+0                     ; 8x s_c, int8
    s_mul_i32 s[s_tmp+5], s[s_wei_stride_k], s[s_k]
    v_mad_u32_u24 v[v_wei_os], s[s_wei_stride_k], v[v_wei_ik], v[v_tmp+1]
    s_mul_i32 s[s_tmp+2], s[s_block_ig], s[s_tmp+5]
    v_cmp_gt_u32 s[s_k], v[v_wei_ik]
    s_add_u32 s[s_p_wei], s[s_p_wei], s[s_tmp+2]
    v_cndmask_b32 v[v_wei_flag_ik], 0, 1
    s_addc_u32 s[s_p_wei+1], s[s_p_wei+1], 0
    ;v_lshlrev_b32 v[v_wei_os], 1, v[v_wei_os]

    ; divide x
    .mdiv_u32_rem_vs v_wei_ix_list+0,v_wei_iy_list+0,v_wei_ie,s_magic_0,s_shift_m0,s_x,v_tmp
    v_add_nc_u32 v[v_wei_os+1], s[s_wei_offset], v[v_wei_os+0]
    v_add_nc_u32 v[v_wei_ie], 8, v[v_wei_ie]
    v_cmp_gt_u32 s[s_y], v[v_wei_iy_list+0]
    v_cndmask_b32 v[v_wei_flag+0], 0, v[v_wei_flag_ik]
    v_cmp_gt_u32 s[s_x], v[v_wei_ix_list+0]
    v_cndmask_b32 v[v_wei_flag+0], 0, v[v_wei_flag+0]

    .mdiv_u32_rem_vs v_wei_ix_list+1,v_wei_iy_list+1,v_wei_ie,s_magic_0,s_shift_m0,s_x,v_tmp
    v_cmp_gt_u32 s[s_y], v[v_wei_iy_list+1]
    v_cndmask_b32 v[v_wei_flag+1], 0, v[v_wei_flag_ik]
    v_cmp_gt_u32 s[s_x], v[v_wei_ix_list+1]
    v_cndmask_b32 v[v_wei_flag+1], 0, v[v_wei_flag+1]

    v_cmpx_le_u32 1, v[v_wei_flag+0]
    global_load_dwordx4 v[v_gld_b+0:v_gld_b+3], v[v_wei_os+0], s[s_p_wei:s_p_wei+1]
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_wei_flag+1]
    global_load_dwordx4 v[v_gld_b+4:v_gld_b+7], v[v_wei_os+1], s[s_p_wei:s_p_wei+1]
    s_mov_b64 exec, -1

    s_mov_b32 s[s_tmp+5], 32*k_n_dword*4       ; stride for wei sst offset. 8 thread for gemm_k, each thread store 4 c, hence 8*4=32 gemm_k

    ; calculate in offset
    s_mul_i32 s[s_in_stride_wi], s[s_c], s[s_group]
    s_bfe_u32 s[s_shift_m1], s[s_shift_pack_0], 0x00080008 ; offset:8, width:8
    s_mul_i32 s[s_tmp+2], s[s_wi], s[s_in_stride_wi]
    s_mul_i32 s[s_tmp+0], s[s_block_ig], s[s_c]
    s_mul_i32 s[s_in_stride_n], s[s_hi], s[s_tmp+2]
    s_mul_i32 s[s_tmp+3], s[s_block_in], s[s_in_stride_n]
    ;s_lshl_b32 s[s_in_stride_wi], s[s_in_stride_wi], 1
    s_add_u32 s[s_tmp+0], s[s_tmp+0], s[s_tmp+3]
    v_add_nc_u32 v[v_sst_b_os+1], s[s_tmp+5], v[v_sst_b_os+0]

    .mdiv_u32_rem_vs v_in_iwi,v_in_ihi,v_ib,s_magic_1,s_shift_m1,s_wo,v_tmp
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp+0]
    v_add_nc_u32 v[v_tmp+5], s[s_ib_stride], v[v_ib]
    s_addc_u32 s[s_p_in+1], s[s_p_in+1], 0
    v_mul_lo_u32 v[v_in_ihi], s[s_stride_h], v[v_in_ihi]
    .v_clear_nc v_ax, 4
    v_sub_nc_i32 v[v_in_ihi], v[v_in_ihi], s[s_pad_h]
    v_mul_lo_u32 v[v_in_iwi], s[s_stride_w], v[v_in_iwi]
    .v_clear_nc v_ax+4, 4
    v_sub_nc_i32 v[v_in_iwi], v[v_in_iwi], s[s_pad_w]

    .mdiv_u32_rem_vs v_in_iwi+1,v_in_ihi+1,v_tmp+5,s_magic_1,s_shift_m1,s_wo,v_tmp

    v_mul_lo_u32 v[v_tmp], s[s_wi], v[v_in_ihi]
    v_add_nc_u32 v[v_tmp+5], s[s_ib_stride], v[v_tmp+5]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi]
    v_cndmask_b32 v[v_in_flag], 0, 1
    v_add_nc_u32 v[v_tmp], v[v_in_iwi], v[v_tmp]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi]
    v_cndmask_b32 v[v_in_flag], 0, v[v_in_flag]
    v_mul_lo_u32 v[v_in_os], s[s_in_stride_wi], v[v_tmp]

    v_mul_lo_u32 v[v_in_ihi+1], s[s_stride_h], v[v_in_ihi+1]
    v_sub_nc_i32 v[v_in_ihi+1], v[v_in_ihi+1], s[s_pad_h]
    v_mul_lo_u32 v[v_in_iwi+1], s[s_stride_w], v[v_in_iwi+1]
    v_sub_nc_i32 v[v_in_iwi+1], v[v_in_iwi+1], s[s_pad_w]

    v_cmpx_le_u32 1, v[v_in_flag+0]
    global_load_dwordx4 v[v_ax+ 0:v_ax+ 3], v[v_in_os+0], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    v_mul_lo_u32 v[v_tmp], s[s_wi], v[v_in_ihi+1]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, 1
    v_add_nc_u32 v[v_tmp], v[v_in_iwi+1], v[v_tmp]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, v[v_in_flag+1]
    v_mul_lo_u32 v[v_in_os+1], s[s_in_stride_wi], v[v_tmp]

    v_cmpx_le_u32 1, v[v_in_flag+1]
    global_load_dwordx4 v[v_ax+ 4:v_ax+ 7], v[v_in_os+1], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    .mdiv_u32_rem_vs v_in_iwi+2,v_in_ihi+2,v_tmp+5,s_magic_1,s_shift_m1,s_wo,v_tmp
    v_add_nc_u32 v[v_tmp+5], s[s_ib_stride], v[v_tmp+5]
    v_mul_lo_u32 v[v_in_ihi+2], s[s_stride_h], v[v_in_ihi+2]
    .v_clear_nc v_ax+8, 4
    v_sub_nc_i32 v[v_in_ihi+2], v[v_in_ihi+2], s[s_pad_h]
    v_mul_lo_u32 v[v_in_iwi+2], s[s_stride_w], v[v_in_iwi+2]
    .v_clear_nc v_ax+12, 4
    v_sub_nc_i32 v[v_in_iwi+2], v[v_in_iwi+2], s[s_pad_w]

    .mdiv_u32_rem_vs v_in_iwi+3,v_in_ihi+3,v_tmp+5,s_magic_1,s_shift_m1,s_wo,v_tmp
    v_mul_lo_u32 v[v_in_ihi+3], s[s_stride_h], v[v_in_ihi+3]
    v_sub_nc_i32 v[v_in_ihi+3], v[v_in_ihi+3], s[s_pad_h]
    v_mul_lo_u32 v[v_in_iwi+3], s[s_stride_w], v[v_in_iwi+3]
    v_sub_nc_i32 v[v_in_iwi+3], v[v_in_iwi+3], s[s_pad_w]

    v_mul_lo_u32 v[v_tmp], s[s_wi], v[v_in_ihi+2]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, 1
    v_add_nc_u32 v[v_tmp], v[v_in_iwi+2], v[v_tmp]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, v[v_in_flag+2]
    v_mul_lo_u32 v[v_in_os+2], s[s_in_stride_wi], v[v_tmp]

    v_cmpx_le_u32 1, v[v_in_flag+2]
    global_load_dwordx4 v[v_ax+ 8:v_ax+11], v[v_in_os+2], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    v_mul_lo_u32 v[v_tmp], s[s_wi], v[v_in_ihi+3]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, 1
    v_add_nc_u32 v[v_tmp], v[v_in_iwi+3], v[v_tmp]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, v[v_in_flag+3]
    v_mul_lo_u32 v[v_in_os+3], s[s_in_stride_wi], v[v_tmp]

    v_cmpx_le_u32 1, v[v_in_flag+3]
    global_load_dwordx4 v[v_ax+12:v_ax+15], v[v_in_os+3], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    s_mul_i32 s[s_br], s[s_wo], s[s_ho]

    s_mul_i32 s[s_out_stride_wo], s[s_k], s[s_group]
    s_mul_i32 s[s_in_diff_wi], s[s_dilation_w], s[s_in_stride_wi]
    s_mov_b32 s[s_move_slice_k_ix], 0

    s_mul_i32 s[s_out_stride_n], s[s_br], s[s_out_stride_wo]
    s_mul_i32 s[s_tmp+1], s[s_block_ig], s[s_k]
    s_mul_i32 s[s_tmp+4], s[s_block_in], s[s_out_stride_n]
    ;s_lshl_b32 s[s_tmp+5], s[s_block_ik], 0
    s_add_u32 s[s_tmp+1], s[s_tmp+1], s[s_tmp+4]
    s_add_u32 s[s_tmp+1], s[s_tmp+1], s[s_block_ik]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp+1]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], 0

    ; calculate diffs, for y, x
    s_sub_i32 s[s_tmp+3], s[s_x], 1
    s_mul_i32 s[s_tmp], s[s_in_diff_wi], s[s_tmp+3]
    s_mul_i32 s[s_tmp+1], s[s_in_stride_wi], s[s_wi]
    s_mul_i32 s[s_tmp+1], s[s_tmp+1], s[s_dilation_h]
    s_sub_i32 s[s_in_diff_hi], s[s_tmp+1], s[s_tmp]
    s_mul_i32 s[s_dilation_w_x], s[s_dilation_w], s[s_tmp+3]
    s_mul_i32 s[s_dilation_w_x], s[s_dilation_w_x], -1


    v_add_nc_u32 v[v_tmp+5], s[s_ib_stride], v[v_ib]
    s_mul_i32 s[s_out_stride], s[s_stride_m], s[s_out_stride_wo]

    ;s_lshl_b32 s[s_out_stride], s[s_out_stride], 1
    ;s_lshl_b32 s[s_out_stride_n], s[s_out_stride_n], 1

    ; output offset
    v_mul_lo_u32 v[v_out_os], s[s_k], v[v_ib]
    ;v_lshlrev_b32 v[v_out_os], 1, v[v_out_os]
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os]
    v_cndmask_b32 v[v_out_flag], 0, 1
    v_add_nc_u32 v[v_tmp+4], s[s_ib_stride], v[v_tmp+5]

    v_mul_lo_u32 v[v_out_os+1], s[s_k], v[v_tmp+5]
    ;v_lshlrev_b32 v[v_out_os+1], 1, v[v_out_os+1]
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+1]
    v_cndmask_b32 v[v_out_flag+1], 0, 1
    v_add_nc_u32 v[v_tmp+5], s[s_ib_stride], v[v_tmp+4]

    v_mul_lo_u32 v[v_out_os+2], s[s_k], v[v_tmp+4]
    ;v_lshlrev_b32 v[v_out_os+2], 1, v[v_out_os+2]
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+2]
    v_cndmask_b32 v[v_out_flag+2], 0, 1

    v_mul_lo_u32 v[v_out_os+3], s[s_k], v[v_tmp+5]
    ;v_lshlrev_b32 v[v_out_os+3], 1, v[v_out_os+3]
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+3]
    v_cndmask_b32 v[v_out_flag+3], 0, 1

    s_mov_b32 s[s_sld_b_stride],    k_n_dword*4*4

    s_waitcnt vmcnt(4)

    v_cmpx_le_u32 1, v[v_wei_flag+0]
    ds_write2_b32 v[v_sst_b_os+0], v[v_gld_b+0], v[v_gld_b+1], offset0:k_n_dword*0  offset1:k_n_dword*1
    ds_write2_b32 v[v_sst_b_os+0], v[v_gld_b+2], v[v_gld_b+3], offset0:k_n_dword*2  offset1:k_n_dword*3
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_wei_flag+1]
    ds_write2_b32 v[v_sst_b_os+1], v[v_gld_b+4], v[v_gld_b+5], offset0:k_n_dword*0  offset1:k_n_dword*1
    ds_write2_b32 v[v_sst_b_os+1], v[v_gld_b+6], v[v_gld_b+7], offset0:k_n_dword*2  offset1:k_n_dword*3
    s_mov_b64 exec, -1

    .v_clear_nc v_c, 64

    s_waitcnt lgkmcnt(0)
    s_barrier

    ds_read_b128 v[v_b+ 0:v_b+ 3], v[v_sld_b_os], offset:k_n_dword*4*0 + 0*4
    ds_read_b128 v[v_b+ 4:v_b+ 7], v[v_sld_b_os], offset:k_n_dword*4*0 + 4*4
    ds_read_b128 v[v_b+ 8:v_b+11], v[v_sld_b_os], offset:k_n_dword*4*0 + 8*4
    ds_read_b128 v[v_b+12:v_b+15], v[v_sld_b_os], offset:k_n_dword*4*0 +12*4
    ds_read_b128 v[v_b+16:v_b+19], v[v_sld_b_os], offset:k_n_dword*4*1 + 0*4
    ds_read_b128 v[v_b+20:v_b+23], v[v_sld_b_os], offset:k_n_dword*4*1 + 4*4
    ds_read_b128 v[v_b+24:v_b+27], v[v_sld_b_os], offset:k_n_dword*4*1 + 8*4
    ds_read_b128 v[v_b+28:v_b+31], v[v_sld_b_os], offset:k_n_dword*4*1 +12*4

    ds_read_b128 v[v_b+32:v_b+35], v[v_sld_b_os], offset:k_n_dword*4*2 + 0*4
    ds_read_b128 v[v_b+36:v_b+39], v[v_sld_b_os], offset:k_n_dword*4*2 + 4*4
    ds_read_b128 v[v_b+40:v_b+43], v[v_sld_b_os], offset:k_n_dword*4*2 + 8*4
    ds_read_b128 v[v_b+44:v_b+47], v[v_sld_b_os], offset:k_n_dword*4*2 +12*4
    ds_read_b128 v[v_b+48:v_b+51], v[v_sld_b_os], offset:k_n_dword*4*3 + 0*4
    ds_read_b128 v[v_b+52:v_b+55], v[v_sld_b_os], offset:k_n_dword*4*3 + 4*4
    ds_read_b128 v[v_b+56:v_b+59], v[v_sld_b_os], offset:k_n_dword*4*3 + 8*4
    ds_read_b128 v[v_b+60:v_b+63], v[v_sld_b_os], offset:k_n_dword*4*3 +12*4

    s_sub_i32 s[s_kitr], s[s_wei_stride_k], 16
    v_add_nc_u32 v[v_sld_b_os], s[s_sld_b_stride], v[v_sld_b_os]            ; accumulate sld_b_os

    s_cmp_gt_i32 s[s_kitr], 0
    
    s_cbranch_scc0 L_igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_end

L_igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_body:
    ; accumulate im

    ; a buffer x
    ;--- start move slice window
    s_add_u32 s[s_move_slice_k_ix], 1, s[s_move_slice_k_ix]
    s_cmp_le_u32 s[s_x], s[s_move_slice_k_ix]
    s_cselect_b32 s[s_tmp], s[s_dilation_w_x], s[s_dilation_w]
    s_cselect_b32 s[s_tmp+1], s[s_in_diff_hi], s[s_in_diff_wi]
    v_add_nc_u32 v[v_in_iwi+0], s[s_tmp], v[v_in_iwi+0]
    v_add_nc_u32 v[v_in_iwi+1], s[s_tmp], v[v_in_iwi+1]
    v_add_nc_u32 v[v_in_iwi+2], s[s_tmp], v[v_in_iwi+2]
    v_add_nc_u32 v[v_in_iwi+3], s[s_tmp], v[v_in_iwi+3]
    v_add_nc_u32 v[v_in_os+0], s[s_tmp+1], v[v_in_os+0]
    v_add_nc_u32 v[v_in_os+1], s[s_tmp+1], v[v_in_os+1]
    v_add_nc_u32 v[v_in_os+2], s[s_tmp+1], v[v_in_os+2]
    v_add_nc_u32 v[v_in_os+3], s[s_tmp+1], v[v_in_os+3]
    s_cbranch_scc0 igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_acc_yx_x_end_1
    s_mov_b32 s[s_move_slice_k_ix], 0
    v_add_nc_i32 v[v_in_ihi+0], s[s_dilation_h], v[v_in_ihi+0]
    v_add_nc_i32 v[v_in_ihi+1], s[s_dilation_h], v[v_in_ihi+1]
    v_add_nc_i32 v[v_in_ihi+2], s[s_dilation_h], v[v_in_ihi+2]
    v_add_nc_i32 v[v_in_ihi+3], s[s_dilation_h], v[v_in_ihi+3]
igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_acc_yx_x_end_1:
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+0]
    v_cndmask_b32 v[v_in_flag+0], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, 1
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+0]
    v_cndmask_b32 v[v_in_flag+0], 0, v[v_in_flag+0]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, v[v_in_flag+1]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, v[v_in_flag+2]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, v[v_in_flag+3]
    ;--- end move slice window

    ;s_waitcnt vmcnt(0)
    .v_clear_nc v_ay, 8
    v_cmpx_le_u32 1, v[v_in_flag+0]
    global_load_dwordx4 v[v_ay+ 0:v_ay+ 3], v[v_in_os+0], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_in_flag+1]
    global_load_dwordx4 v[v_ay+ 4:v_ay+ 7], v[v_in_os+1], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1
    .v_clear_nc v_ay+8, 8
    v_cmpx_le_u32 1, v[v_in_flag+2]
    global_load_dwordx4 v[v_ay+ 8:v_ay+11], v[v_in_os+2], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_in_flag+3]
    global_load_dwordx4 v[v_ay+12:v_ay+15], v[v_in_os+3], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    s_waitcnt vmcnt(4) lgkmcnt(8)
    .fma_1x16_int8x4 v_c+ 0, v_ax + 0, v_b + 0
    .fma_1x16_int8x4 v_c+16, v_ax + 4, v_b + 0
    .fma_1x16_int8x4 v_c+32, v_ax + 8, v_b + 0
    .fma_1x16_int8x4 v_c+48, v_ax +12, v_b + 0

    .fma_1x16_int8x4 v_c+ 0, v_ax + 1, v_b +16
    .fma_1x16_int8x4 v_c+16, v_ax + 5, v_b +16
    .fma_1x16_int8x4 v_c+32, v_ax + 9, v_b +16
    .fma_1x16_int8x4 v_c+48, v_ax +13, v_b +16


    ds_read_b128 v[v_b+ 0:v_b+ 3], v[v_sld_b_os], offset:k_n_dword*4*0 + 0*4
    ds_read_b128 v[v_b+ 4:v_b+ 7], v[v_sld_b_os], offset:k_n_dword*4*0 + 4*4
    ds_read_b128 v[v_b+ 8:v_b+11], v[v_sld_b_os], offset:k_n_dword*4*0 + 8*4
    ds_read_b128 v[v_b+12:v_b+15], v[v_sld_b_os], offset:k_n_dword*4*0 +12*4
    ds_read_b128 v[v_b+16:v_b+19], v[v_sld_b_os], offset:k_n_dword*4*1 + 0*4
    ds_read_b128 v[v_b+20:v_b+23], v[v_sld_b_os], offset:k_n_dword*4*1 + 4*4
    ds_read_b128 v[v_b+24:v_b+27], v[v_sld_b_os], offset:k_n_dword*4*1 + 8*4
    ds_read_b128 v[v_b+28:v_b+31], v[v_sld_b_os], offset:k_n_dword*4*1 +12*4

    s_waitcnt lgkmcnt(8)
    .fma_1x16_int8x4 v_c+ 0, v_ax + 2, v_b +32
    .fma_1x16_int8x4 v_c+16, v_ax + 6, v_b +32
    .fma_1x16_int8x4 v_c+32, v_ax +10, v_b +32
    .fma_1x16_int8x4 v_c+48, v_ax +14, v_b +32

    .fma_1x16_int8x4 v_c+ 0, v_ax + 3, v_b +48
    .fma_1x16_int8x4 v_c+16, v_ax + 7, v_b +48
    .fma_1x16_int8x4 v_c+32, v_ax +11, v_b +48
    .fma_1x16_int8x4 v_c+48, v_ax +15, v_b +48


    ds_read_b128 v[v_b+32:v_b+35], v[v_sld_b_os], offset:k_n_dword*4*2 + 0*4
    ds_read_b128 v[v_b+36:v_b+39], v[v_sld_b_os], offset:k_n_dword*4*2 + 4*4
    ds_read_b128 v[v_b+40:v_b+43], v[v_sld_b_os], offset:k_n_dword*4*2 + 8*4
    ds_read_b128 v[v_b+44:v_b+47], v[v_sld_b_os], offset:k_n_dword*4*2 +12*4
    ds_read_b128 v[v_b+48:v_b+51], v[v_sld_b_os], offset:k_n_dword*4*3 + 0*4
    ds_read_b128 v[v_b+52:v_b+55], v[v_sld_b_os], offset:k_n_dword*4*3 + 4*4
    ds_read_b128 v[v_b+56:v_b+59], v[v_sld_b_os], offset:k_n_dword*4*3 + 8*4
    ds_read_b128 v[v_b+60:v_b+63], v[v_sld_b_os], offset:k_n_dword*4*3 +12*4

    s_sub_i32 s[s_kitr], s[s_kitr], 16
    v_add_nc_u32 v[v_sld_b_os], s[s_sld_b_stride], v[v_sld_b_os]            ; accumulate sld_b_os
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_end_1

    ; a buffer y
    ;--- start move slice window
    s_add_u32 s[s_move_slice_k_ix], 1, s[s_move_slice_k_ix]
    s_cmp_le_u32 s[s_x], s[s_move_slice_k_ix]
    s_cselect_b32 s[s_tmp], s[s_dilation_w_x], s[s_dilation_w]
    s_cselect_b32 s[s_tmp+1], s[s_in_diff_hi], s[s_in_diff_wi]
    v_add_nc_u32 v[v_in_iwi+0], s[s_tmp], v[v_in_iwi+0]
    v_add_nc_u32 v[v_in_iwi+1], s[s_tmp], v[v_in_iwi+1]
    v_add_nc_u32 v[v_in_iwi+2], s[s_tmp], v[v_in_iwi+2]
    v_add_nc_u32 v[v_in_iwi+3], s[s_tmp], v[v_in_iwi+3]
    v_add_nc_u32 v[v_in_os+0], s[s_tmp+1], v[v_in_os+0]
    v_add_nc_u32 v[v_in_os+1], s[s_tmp+1], v[v_in_os+1]
    v_add_nc_u32 v[v_in_os+2], s[s_tmp+1], v[v_in_os+2]
    v_add_nc_u32 v[v_in_os+3], s[s_tmp+1], v[v_in_os+3]
    s_cbranch_scc0 igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_acc_yx_x_end_2
    s_mov_b32 s[s_move_slice_k_ix], 0
    v_add_nc_i32 v[v_in_ihi+0], s[s_dilation_h], v[v_in_ihi+0]
    v_add_nc_i32 v[v_in_ihi+1], s[s_dilation_h], v[v_in_ihi+1]
    v_add_nc_i32 v[v_in_ihi+2], s[s_dilation_h], v[v_in_ihi+2]
    v_add_nc_i32 v[v_in_ihi+3], s[s_dilation_h], v[v_in_ihi+3]
igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_acc_yx_x_end_2:
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+0]
    v_cndmask_b32 v[v_in_flag+0], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, 1
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+0]
    v_cndmask_b32 v[v_in_flag+0], 0, v[v_in_flag+0]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, v[v_in_flag+1]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, v[v_in_flag+2]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, v[v_in_flag+3]
    ;--- end move slice window

    .v_clear_nc v_ax, 8
    v_cmpx_le_u32 1, v[v_in_flag+0]
    global_load_dwordx4 v[v_ax+ 0:v_ax+ 3], v[v_in_os+0], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_in_flag+1]
    global_load_dwordx4 v[v_ax+ 4:v_ax+ 7], v[v_in_os+1], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1
    .v_clear_nc v_ax+8, 8
    v_cmpx_le_u32 1, v[v_in_flag+2]
    global_load_dwordx4 v[v_ax+ 8:v_ax+11], v[v_in_os+2], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_in_flag+3]
    global_load_dwordx4 v[v_ax+12:v_ax+15], v[v_in_os+3], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    s_waitcnt vmcnt(4) lgkmcnt(8)
    .fma_1x16_int8x4 v_c+ 0, v_ay + 0, v_b + 0
    .fma_1x16_int8x4 v_c+16, v_ay + 4, v_b + 0
    .fma_1x16_int8x4 v_c+32, v_ay + 8, v_b + 0
    .fma_1x16_int8x4 v_c+48, v_ay +12, v_b + 0

    .fma_1x16_int8x4 v_c+ 0, v_ay + 1, v_b +16
    .fma_1x16_int8x4 v_c+16, v_ay + 5, v_b +16
    .fma_1x16_int8x4 v_c+32, v_ay + 9, v_b +16
    .fma_1x16_int8x4 v_c+48, v_ay +13, v_b +16

    ds_read_b128 v[v_b+ 0:v_b+ 3], v[v_sld_b_os], offset:k_n_dword*4*0 + 0*4
    ds_read_b128 v[v_b+ 4:v_b+ 7], v[v_sld_b_os], offset:k_n_dword*4*0 + 4*4
    ds_read_b128 v[v_b+ 8:v_b+11], v[v_sld_b_os], offset:k_n_dword*4*0 + 8*4
    ds_read_b128 v[v_b+12:v_b+15], v[v_sld_b_os], offset:k_n_dword*4*0 +12*4
    ds_read_b128 v[v_b+16:v_b+19], v[v_sld_b_os], offset:k_n_dword*4*1 + 0*4
    ds_read_b128 v[v_b+20:v_b+23], v[v_sld_b_os], offset:k_n_dword*4*1 + 4*4
    ds_read_b128 v[v_b+24:v_b+27], v[v_sld_b_os], offset:k_n_dword*4*1 + 8*4
    ds_read_b128 v[v_b+28:v_b+31], v[v_sld_b_os], offset:k_n_dword*4*1 +12*4

    s_waitcnt lgkmcnt(8)
    .fma_1x16_int8x4 v_c+ 0, v_ay + 2, v_b +32
    .fma_1x16_int8x4 v_c+16, v_ay + 6, v_b +32
    .fma_1x16_int8x4 v_c+32, v_ay +10, v_b +32
    .fma_1x16_int8x4 v_c+48, v_ay +14, v_b +32

    .fma_1x16_int8x4 v_c+ 0, v_ay + 3, v_b +48
    .fma_1x16_int8x4 v_c+16, v_ay + 7, v_b +48
    .fma_1x16_int8x4 v_c+32, v_ay +11, v_b +48
    .fma_1x16_int8x4 v_c+48, v_ay +15, v_b +48

    ds_read_b128 v[v_b+32:v_b+35], v[v_sld_b_os], offset:k_n_dword*4*2 + 0*4
    ds_read_b128 v[v_b+36:v_b+39], v[v_sld_b_os], offset:k_n_dword*4*2 + 4*4
    ds_read_b128 v[v_b+40:v_b+43], v[v_sld_b_os], offset:k_n_dword*4*2 + 8*4
    ds_read_b128 v[v_b+44:v_b+47], v[v_sld_b_os], offset:k_n_dword*4*2 +12*4
    ds_read_b128 v[v_b+48:v_b+51], v[v_sld_b_os], offset:k_n_dword*4*3 + 0*4
    ds_read_b128 v[v_b+52:v_b+55], v[v_sld_b_os], offset:k_n_dword*4*3 + 4*4
    ds_read_b128 v[v_b+56:v_b+59], v[v_sld_b_os], offset:k_n_dword*4*3 + 8*4
    ds_read_b128 v[v_b+60:v_b+63], v[v_sld_b_os], offset:k_n_dword*4*3 +12*4

    s_sub_i32 s[s_kitr], s[s_kitr], 16
    v_add_nc_u32 v[v_sld_b_os], s[s_sld_b_stride], v[v_sld_b_os]            ; accumulate sld_b_os
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc1 L_igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_body

L_igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_end:
    s_waitcnt vmcnt(0)

    v_mov_b32 v[v_ay + 0], v[v_ax + 0]
    v_mov_b32 v[v_ay + 1], v[v_ax + 1]
    v_mov_b32 v[v_ay + 2], v[v_ax + 2]
    v_mov_b32 v[v_ay + 3], v[v_ax + 3]
    v_mov_b32 v[v_ay + 4], v[v_ax + 4]
    v_mov_b32 v[v_ay + 5], v[v_ax + 5]
    v_mov_b32 v[v_ay + 6], v[v_ax + 6]
    v_mov_b32 v[v_ay + 7], v[v_ax + 7]
    v_mov_b32 v[v_ay + 8], v[v_ax + 8]
    v_mov_b32 v[v_ay + 9], v[v_ax + 9]
    v_mov_b32 v[v_ay +10], v[v_ax +10]
    v_mov_b32 v[v_ay +11], v[v_ax +11]
    v_mov_b32 v[v_ay +12], v[v_ax +12]
    v_mov_b32 v[v_ay +13], v[v_ax +13]
    v_mov_b32 v[v_ay +14], v[v_ax +14]
    v_mov_b32 v[v_ay +15], v[v_ax +15]

L_igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_end_1:
    s_waitcnt vmcnt(0)

    s_sub_i32 s[s_batch_m], s[s_batch_m], 1
    v_add_nc_u32 v[v_ib], s[s_stride_m],  v[v_ib]

    s_cmp_gt_i32 s[s_batch_m], 0
    s_cbranch_scc0 L_igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_end_not_load_next
    ; --- start move slice for batch m
    ; ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
    ; iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w
    ; we will update v_in_os below, so use this as v_tmp
    .mdiv_u32_rem_vs v_in_iwi,v_in_ihi,v_ib,s_magic_1,s_shift_m1,s_wo,v_in_os
    v_mul_u32_u24 v[v_in_ihi], s[s_stride_h], v[v_in_ihi]
    .v_clear_nc v_ax, 4
    v_add_nc_u32 v[v_in_flag+1], s[s_ib_stride], v[v_ib]
    v_sub_nc_i32 v[v_in_ihi], v[v_in_ihi], s[s_pad_h]
    v_mul_u32_u24 v[v_in_iwi], s[s_stride_w], v[v_in_iwi]
    .v_clear_nc v_ax+4, 4
    v_sub_nc_i32 v[v_in_iwi], v[v_in_iwi], s[s_pad_w]

    .mdiv_u32_rem_vs v_in_iwi+1,v_in_ihi+1,v_in_flag+1,s_magic_1,s_shift_m1,s_wo,v_in_os+1

    v_mul_u32_u24 v[v_in_os], s[s_wi], v[v_in_ihi]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi]
    v_cndmask_b32 v[v_in_flag], 0, 1
    v_add_nc_u32 v[v_in_os], v[v_in_iwi], v[v_in_os]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi]
    v_cndmask_b32 v[v_in_flag], 0, v[v_in_flag]
    v_mul_lo_u32 v[v_in_os], s[s_in_stride_wi], v[v_in_os]
    
    v_mul_u32_u24 v[v_in_ihi+1], s[s_stride_h], v[v_in_ihi+1]
    v_sub_nc_i32 v[v_in_ihi+1], v[v_in_ihi+1], s[s_pad_h]
    v_mul_u32_u24 v[v_in_iwi+1], s[s_stride_w], v[v_in_iwi+1]
    v_sub_nc_i32 v[v_in_iwi+1], v[v_in_iwi+1], s[s_pad_w]

    v_add_nc_u32 v[v_in_flag+2], s[s_ib_stride], v[v_in_flag+1]

    v_cmpx_le_u32 1, v[v_in_flag+0]
    global_load_dwordx4 v[v_ax+ 0:v_ax+ 3], v[v_in_os], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    v_mul_u32_u24 v[v_in_os+1], s[s_wi], v[v_in_ihi+1]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, 1
    v_add_nc_u32 v[v_in_os+1], v[v_in_iwi+1], v[v_in_os+1]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, v[v_in_flag+1]
    v_mul_lo_u32 v[v_in_os+1], s[s_in_stride_wi], v[v_in_os+1]

    v_cmpx_le_u32 1, v[v_in_flag+1]
    global_load_dwordx4 v[v_ax+ 4:v_ax+ 7], v[v_in_os+1], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    .mdiv_u32_rem_vs v_in_iwi+2,v_in_ihi+2,v_in_flag+2,s_magic_1,s_shift_m1,s_wo,v_in_os+2
    v_add_nc_u32 v[v_in_flag+3], s[s_ib_stride], v[v_in_flag+2]
    v_mul_lo_u32 v[v_in_ihi+2], s[s_stride_h], v[v_in_ihi+2]
    .v_clear_nc v_ax+8, 4
    v_sub_nc_i32 v[v_in_ihi+2], v[v_in_ihi+2], s[s_pad_h]
    v_mul_lo_u32 v[v_in_iwi+2], s[s_stride_w], v[v_in_iwi+2]
    .v_clear_nc v_ax+12, 4
    v_sub_nc_i32 v[v_in_iwi+2], v[v_in_iwi+2], s[s_pad_w]

    .mdiv_u32_rem_vs v_in_iwi+3,v_in_ihi+3,v_in_flag+3,s_magic_1,s_shift_m1,s_wo,v_in_os+3
    v_mul_lo_u32 v[v_in_ihi+3], s[s_stride_h], v[v_in_ihi+3]
    v_sub_nc_i32 v[v_in_ihi+3], v[v_in_ihi+3], s[s_pad_h]
    v_mul_lo_u32 v[v_in_iwi+3], s[s_stride_w], v[v_in_iwi+3]
    v_sub_nc_i32 v[v_in_iwi+3], v[v_in_iwi+3], s[s_pad_w]

    v_mul_lo_u32 v[v_in_os+2], s[s_wi], v[v_in_ihi+2]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, 1
    v_add_nc_u32 v[v_in_os+2], v[v_in_iwi+2], v[v_in_os+2]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, v[v_in_flag+2]
    v_mul_lo_u32 v[v_in_os+2], s[s_in_stride_wi], v[v_in_os+2]

    v_cmpx_le_u32 1, v[v_in_flag+2]
    global_load_dwordx4 v[v_ax+ 8:v_ax+11], v[v_in_os+2], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    v_mul_lo_u32 v[v_in_os+3], s[s_wi], v[v_in_ihi+3]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, 1
    v_add_nc_u32 v[v_in_os+3], v[v_in_iwi+3], v[v_in_os+3]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, v[v_in_flag+3]
    v_mul_lo_u32 v[v_in_os+3], s[s_in_stride_wi], v[v_in_os+3]

    v_cmpx_le_u32 1, v[v_in_flag+3]
    global_load_dwordx4 v[v_ax+12:v_ax+15], v[v_in_os+3], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    s_mov_b32 s[s_move_slice_k_ix], 0

L_igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_end_not_load_next:
    ; --- end move slice for batch m

    s_waitcnt lgkmcnt(8)
    .fma_1x16_int8x4 v_c+ 0, v_ay + 0, v_b + 0
    .fma_1x16_int8x4 v_c+16, v_ay + 4, v_b + 0
    .fma_1x16_int8x4 v_c+32, v_ay + 8, v_b + 0
    .fma_1x16_int8x4 v_c+48, v_ay +12, v_b + 0

    .fma_1x16_int8x4 v_c+ 0, v_ay + 1, v_b +16
    .fma_1x16_int8x4 v_c+16, v_ay + 5, v_b +16
    .fma_1x16_int8x4 v_c+32, v_ay + 9, v_b +16
    .fma_1x16_int8x4 v_c+48, v_ay +13, v_b +16

    s_waitcnt lgkmcnt(0)
    .fma_1x16_int8x4 v_c+ 0, v_ay + 2, v_b +32
    .fma_1x16_int8x4 v_c+16, v_ay + 6, v_b +32
    .fma_1x16_int8x4 v_c+32, v_ay +10, v_b +32
    .fma_1x16_int8x4 v_c+48, v_ay +14, v_b +32

    .fma_1x16_int8x4 v_c+ 0, v_ay + 3, v_b +48
    .fma_1x16_int8x4 v_c+16, v_ay + 7, v_b +48
    .fma_1x16_int8x4 v_c+32, v_ay +11, v_b +48
    .fma_1x16_int8x4 v_c+48, v_ay +15, v_b +48

    v_mov_b32 v[v_sld_b_os], 0                                  ; reset to start
 
    .activ_int32 v_c + 0, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 0+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .pack_i8x4_i32_r4 v_c_buf+ 0, v_c+ 0, s_0xff
    .activ_int32 v_c + 0+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 0+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .pack_i8x4_i32_r4 v_c_buf+ 4, v_c+16, s_0xff
    v_cmpx_le_u32 1, v[v_out_flag]
    global_store_dwordx4 v[v_out_os], v[v_c_buf+0:v_c_buf+3], s[s_p_out:s_p_out+1]
    s_mov_b64 exec, -1

    v_cmpx_le_u32 1, v[v_out_flag+1]
    global_store_dwordx4 v[v_out_os+1], v[v_c_buf+ 4:v_c_buf+ 7], s[s_p_out:s_p_out+1]
    s_mov_b64 exec, -1

    .activ_int32 v_c + 0+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 0+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .pack_i8x4_i32_r4 v_c_buf+ 8, v_c+32, s_0xff

    .activ_int32 v_c + 0+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 0+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .pack_i8x4_i32_r4 v_c_buf+12, v_c+48, s_0xff

    v_cmpx_le_u32 1, v[v_out_flag+2]
    global_store_dwordx4 v[v_out_os+2], v[v_c_buf+ 8:v_c_buf+11], s[s_p_out:s_p_out+1]
    s_mov_b64 exec, -1

    v_cmpx_le_u32 1, v[v_out_flag+3]
    global_store_dwordx4 v[v_out_os+3], v[v_c_buf+12:v_c_buf+15], s[s_p_out:s_p_out+1]
    s_mov_b64 exec, -1

    s_cmp_le_i32 s[s_batch_m], 0

    s_cbranch_scc1 L_igemm_fwd_btm_nhwc_int8_512x16x16_r2_end
    ds_read_b128 v[v_b+ 0:v_b+ 3], v[v_sld_b_os], offset:k_n_dword*4*0 + 0*4
    ds_read_b128 v[v_b+ 4:v_b+ 7], v[v_sld_b_os], offset:k_n_dword*4*0 + 4*4
    ds_read_b128 v[v_b+ 8:v_b+11], v[v_sld_b_os], offset:k_n_dword*4*0 + 8*4
    ds_read_b128 v[v_b+12:v_b+15], v[v_sld_b_os], offset:k_n_dword*4*0 +12*4
    ds_read_b128 v[v_b+16:v_b+19], v[v_sld_b_os], offset:k_n_dword*4*1 + 0*4
    ds_read_b128 v[v_b+20:v_b+23], v[v_sld_b_os], offset:k_n_dword*4*1 + 4*4
    ds_read_b128 v[v_b+24:v_b+27], v[v_sld_b_os], offset:k_n_dword*4*1 + 8*4
    ds_read_b128 v[v_b+28:v_b+31], v[v_sld_b_os], offset:k_n_dword*4*1 +12*4

    ds_read_b128 v[v_b+32:v_b+35], v[v_sld_b_os], offset:k_n_dword*4*2 + 0*4
    ds_read_b128 v[v_b+36:v_b+39], v[v_sld_b_os], offset:k_n_dword*4*2 + 4*4
    ds_read_b128 v[v_b+40:v_b+43], v[v_sld_b_os], offset:k_n_dword*4*2 + 8*4
    ds_read_b128 v[v_b+44:v_b+47], v[v_sld_b_os], offset:k_n_dword*4*2 +12*4
    ds_read_b128 v[v_b+48:v_b+51], v[v_sld_b_os], offset:k_n_dword*4*3 + 0*4
    ds_read_b128 v[v_b+52:v_b+55], v[v_sld_b_os], offset:k_n_dword*4*3 + 4*4
    ds_read_b128 v[v_b+56:v_b+59], v[v_sld_b_os], offset:k_n_dword*4*3 + 8*4
    ds_read_b128 v[v_b+60:v_b+63], v[v_sld_b_os], offset:k_n_dword*4*3 +12*4

    .v_clear_nc v_c, 64
    v_add_nc_u32 v[v_sld_b_os], s[s_sld_b_stride], v[v_sld_b_os]            ; accumulate sld_b_os

    v_add_nc_u32 v[v_out_os], s[s_out_stride], v[v_out_os]
    s_sub_i32 s[s_kitr], s[s_wei_stride_k], 16
    v_add_nc_u32 v[v_out_os+1], s[s_out_stride], v[v_out_os+1]
    v_add_nc_u32 v[v_out_os+2], s[s_out_stride], v[v_out_os+2]
    v_add_nc_u32 v[v_out_os+3], s[s_out_stride], v[v_out_os+3]
    
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os]
    v_cndmask_b32 v[v_out_flag], 0, 1
    s_cmp_gt_i32 s[s_kitr], 0
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+1]
    v_cndmask_b32 v[v_out_flag+1], 0, 1
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+2]
    v_cndmask_b32 v[v_out_flag+2], 0, 1
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+3]
    v_cndmask_b32 v[v_out_flag+3], 0, 1

    s_cbranch_scc0 L_igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_end
    s_branch L_igemm_fwd_btm_nhwc_int8_512x16x16_r2_fma_body
L_igemm_fwd_btm_nhwc_int8_512x16x16_r2_end:
    s_endpgm

; LDS: 2 * 4    * 4  * 128
;      r2  4dword 4    threads
.rodata
.p2align 6
.amdhsa_kernel igemm_fwd_btm_nhwc_int8_512x16x16_r2
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 188
    .amdhsa_next_free_sgpr 62
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
    .amdhsa_wavefront_size32 1
    .amdhsa_workgroup_processor_mode 0
.end_amdhsa_kernel

;----------------------------------------------------------------------------------
.set k_p_in, 0
.set k_p_wei, 8
.set k_p_out, 16
.set k_hi, 24
.set k_wi, 28
.set k_n, 32
.set k_k, 36
.set k_c, 40
.set k_ho, 44
.set k_wo, 48
.set k_stride_h, 52
.set k_stride_w, 56
.set k_dilation_h, 60
.set k_dilation_w, 64
.set k_pad_h, 68
.set k_pad_w, 72
.set k_y, 76
.set k_x, 80
.set k_group, 84
.set k_batch_m, 88
.set k_stride_m, 92
.set k_alpha, 96
.set k_beta, 100
.set k_gamma, 104
.set k_magic_0, 108
.set k_magic_1, 112
.set k_magic_2, 116
.set k_shift_pack_0, 120
.set k_n_dword, 16

.set s_ka, 0
.set s_bx, 2            ; bx, ho*wo
.set s_block_ig, 3      ; by, group
.set s_block_in, 4      ; bz, batch
.set s_p_in, 6
.set s_p_wei, 8
.set s_p_out, 10
.set s_hi, 16
.set s_wi, 17
.set s_n, 18
.set s_k, 19
.set s_c, 20
.set s_ho, 21
.set s_wo, 22
.set s_stride_h, 23
.set s_stride_w, 24
.set s_dilation_h, 25
.set s_dilation_w, 26
.set s_pad_h, 27
.set s_pad_w, 28
.set s_y, 29
.set s_x, 30
.set s_group, 31
.set s_batch_m, 32
.set s_stride_m, 33
.set s_alpha, 34
.set s_beta, 35
.set s_gamma, 36
.set s_magic_0, 37
.set s_magic_1, 38
.set s_magic_2, 39
.set s_shift_pack_0, 40
.set s__pack_0, 41
.set s_shift_m0, 42
.set s_shift_m1, s_shift_pack_0
.set s_shift_m2, 43
.set s_in_stride_wi, 12
.set s_in_stride_n, 13
.set s_wei_stride_k, 14
.set s_out_stride_wo, 15
.set s_out_stride_n, 44
.set s_in_diff_hi, 45
.set s_in_diff_wi, 46
.set s_dilation_w_x, 47
.set s_move_slice_k_ix, 48

.set s_kitr, 1
.set s_wei_offset, 49
.set s_out_stride, s_wei_offset
.set s_sld_b_stride, 50
.set s_br, 51
.set s_ib_stride, 52
.set s_block_ik, 53
.set s_block_ib, 54
.set s_0xff, 55
.set s_tmp, 56
.set s_end, 62

; magic_0: x
; magic_1: wo

.set v_c,               0
.set v_sld_b_os,        64
.set v_ax,              65
.set v_ay,              73
.set v_ib,              81
.set v_b,               82
.set v_gld_b,           v_b
.set v_wei_iy_list,     v_b+8
.set v_wei_ix_list,     v_b+10
.set v_wei_flag,        v_b+12
.set v_wei_os,          v_b+14
.set v_tmp,             v_b+16
.set v_wei_ik,          v_ay
.set v_wei_ic,          v_ay+1
.set v_wei_ie,          v_ay+2
.set v_wei_flag_ik,     v_ay+3
.set v_sst_b_os,        v_ay+4
.set v_in_os,           114
.set v_in_ihi,          118
.set v_in_iwi,          122
.set v_in_flag,         126
.set v_out_os,          130
.set v_out_flag,        134
.set v_tid,             138
.set v_end,             140
.set v_c_buf,           v_b

; short wide igemv
.text
.globl igemm_fwd_btm_nhwc_int8_512x16x8_r2
.p2align 8

.type igemm_fwd_btm_nhwc_int8_512x16x8_r2,@function
igemm_fwd_btm_nhwc_int8_512x16x8_r2:
    s_load_dwordx2  s[s_p_in+0:s_p_in+1],    s[s_ka+0:s_ka+1],    0+k_p_in
    s_load_dwordx4  s[s_p_wei+0:s_p_wei+3],  s[s_ka+0:s_ka+1],    0+k_p_wei
    s_load_dwordx16 s[s_hi+0:s_hi+15],    s[s_ka+0:s_ka+1],    0+k_hi
    s_load_dwordx8  s[s_batch_m:s_batch_m+7],    s[s_ka+0:s_ka+1],    0+k_batch_m
    s_load_dword  s[s_shift_pack_0],    s[s_ka+0:s_ka+1],    0+k_shift_pack_0
    v_mov_b32       v[v_tid], v0
    s_mov_b32 s[s_ib_stride], 128
    s_mov_b32 s[s_0xff], 0xff

    ; calculate wei offset, 16x8, 16 for k, 8 for yxc, 8 for yx, 1 for c
    v_lshrrev_b32 v[v_wei_ik], 3, v0
    s_mov_b32 s[s_tmp], k_n_dword*4 * 2
    v_and_b32 v[v_wei_ie], 7, v0                            ; yx
    ;s_lshl_b32 s[s_block_ig], s[s_block_ig], 1
    v_mov_b32 v[v_wei_ic], 0
    ;s_lshl_b32 s[s_block_in], s[s_block_in], 1
    ;v_lshrrev_b32 v[v_tmp+4], 1, v0
    v_mov_b32 v[v_ib], v0
    v_mul_u32_u24 v[v_tmp+5],   s[s_tmp]  ,v[v_wei_ie]
    v_lshlrev_b32 v[v_sst_b_os], 2, v[v_wei_ik]             ; store, k*n*k_pack, ds_write2 if possible, n*k_pack->16dword, pad to x
    v_mov_b32 v[v_sld_b_os], 0                              ; load   
    v_lshlrev_b32 v[v_wei_ic], 4, v[v_wei_ic]               ; 16xc, k_pack, 4x dword
    v_add_nc_u32 v[v_sst_b_os], v[v_sst_b_os], v[v_tmp+5]   ; note, do not use or due to pad

    s_waitcnt lgkmcnt(0)
    s_bfe_u32 s[s_shift_m2], s[s_shift_pack_0], 0x00080010      ; offset:16, width:8
    s_lshr_b32 s[s_tmp+3], s[s_k], 4
    s_bfe_u32 s[s_shift_m0], s[s_shift_pack_0], 0x00080000      ; offset:0, width:8
    .mdiv_u32_rem_ss s_tmp+4,s_tmp+5,s_bx,s_magic_2,s_shift_m2,s_tmp+3,s_tmp
    s_lshl_b32 s[s_block_ib], s[s_tmp+5], 9                     ; 512
    s_lshl_b32 s[s_block_ik], s[s_tmp+4], 4
    v_add_nc_u32 v[v_ib], s[s_block_ib],  v[v_ib]
    s_mul_i32 s[s_tmp], s[s_x], s[s_c]
    v_add_nc_u32 v[v_wei_ik], s[s_block_ik], v[v_wei_ik]

    v_mad_u32_u24 v[v_tmp+1], s[s_c], v[v_wei_ie], v[v_wei_ic]
    s_mul_i32 s[s_wei_stride_k], s[s_tmp], s[s_y]
    s_lshl_b32 s[s_wei_offset], s[s_c], 3+0                     ; 8x s_c, int8
    s_mul_i32 s[s_tmp+5], s[s_wei_stride_k], s[s_k]
    v_mad_u32_u24 v[v_wei_os], s[s_wei_stride_k], v[v_wei_ik], v[v_tmp+1]
    s_mul_i32 s[s_tmp+2], s[s_block_ig], s[s_tmp+5]
    v_cmp_gt_u32 s[s_k], v[v_wei_ik]
    s_add_u32 s[s_p_wei], s[s_p_wei], s[s_tmp+2]
    v_cndmask_b32 v[v_wei_flag_ik], 0, 1
    s_addc_u32 s[s_p_wei+1], s[s_p_wei+1], 0
    ;v_lshlrev_b32 v[v_wei_os], 1, v[v_wei_os]

    ; divide x
    .mdiv_u32_rem_vs v_wei_ix_list+0,v_wei_iy_list+0,v_wei_ie,s_magic_0,s_shift_m0,s_x,v_tmp
    v_add_nc_u32 v[v_wei_os+1], s[s_wei_offset], v[v_wei_os+0]
    v_add_nc_u32 v[v_wei_ie], 8, v[v_wei_ie]
    v_cmp_gt_u32 s[s_y], v[v_wei_iy_list+0]
    v_cndmask_b32 v[v_wei_flag+0], 0, v[v_wei_flag_ik]
    v_cmp_gt_u32 s[s_x], v[v_wei_ix_list+0]
    v_cndmask_b32 v[v_wei_flag+0], 0, v[v_wei_flag+0]

    .mdiv_u32_rem_vs v_wei_ix_list+1,v_wei_iy_list+1,v_wei_ie,s_magic_0,s_shift_m0,s_x,v_tmp
    v_cmp_gt_u32 s[s_y], v[v_wei_iy_list+1]
    v_cndmask_b32 v[v_wei_flag+1], 0, v[v_wei_flag_ik]
    v_cmp_gt_u32 s[s_x], v[v_wei_ix_list+1]
    v_cndmask_b32 v[v_wei_flag+1], 0, v[v_wei_flag+1]

    v_cmpx_le_u32 1, v[v_wei_flag+0]
    global_load_dwordx2 v[v_gld_b+0:v_gld_b+1], v[v_wei_os+0], s[s_p_wei:s_p_wei+1]
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_wei_flag+1]
    global_load_dwordx2 v[v_gld_b+2:v_gld_b+3], v[v_wei_os+1], s[s_p_wei:s_p_wei+1]
    s_mov_b64 exec, -1

    s_mov_b32 s[s_tmp+5], 16*k_n_dword*4       ; stride for wei sst offset. 8 thread for gemm_k, each thread store 2 c, hence 8*2=16 gemm_k

    ; calculate in offset
    s_mul_i32 s[s_in_stride_wi], s[s_c], s[s_group]
    s_bfe_u32 s[s_shift_m1], s[s_shift_pack_0], 0x00080008 ; offset:8, width:8
    s_mul_i32 s[s_tmp+2], s[s_wi], s[s_in_stride_wi]
    s_mul_i32 s[s_tmp+0], s[s_block_ig], s[s_c]
    s_mul_i32 s[s_in_stride_n], s[s_hi], s[s_tmp+2]
    s_mul_i32 s[s_tmp+3], s[s_block_in], s[s_in_stride_n]
    ;s_lshl_b32 s[s_in_stride_wi], s[s_in_stride_wi], 1
    s_add_u32 s[s_tmp+0], s[s_tmp+0], s[s_tmp+3]
    v_add_nc_u32 v[v_sst_b_os+1], s[s_tmp+5], v[v_sst_b_os+0]

    .mdiv_u32_rem_vs v_in_iwi,v_in_ihi,v_ib,s_magic_1,s_shift_m1,s_wo,v_tmp
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp+0]
    v_add_nc_u32 v[v_tmp+5], s[s_ib_stride], v[v_ib]
    s_addc_u32 s[s_p_in+1], s[s_p_in+1], 0
    v_mul_lo_u32 v[v_in_ihi], s[s_stride_h], v[v_in_ihi]
    .v_clear_nc v_ax, 4
    v_sub_nc_i32 v[v_in_ihi], v[v_in_ihi], s[s_pad_h]
    v_mul_lo_u32 v[v_in_iwi], s[s_stride_w], v[v_in_iwi]
    ;.v_clear_nc v_ax+4, 4
    v_sub_nc_i32 v[v_in_iwi], v[v_in_iwi], s[s_pad_w]

    .mdiv_u32_rem_vs v_in_iwi+1,v_in_ihi+1,v_tmp+5,s_magic_1,s_shift_m1,s_wo,v_tmp

    v_mul_lo_u32 v[v_tmp], s[s_wi], v[v_in_ihi]
    v_add_nc_u32 v[v_tmp+5], s[s_ib_stride], v[v_tmp+5]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi]
    v_cndmask_b32 v[v_in_flag], 0, 1
    v_add_nc_u32 v[v_tmp], v[v_in_iwi], v[v_tmp]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi]
    v_cndmask_b32 v[v_in_flag], 0, v[v_in_flag]
    v_mul_lo_u32 v[v_in_os], s[s_in_stride_wi], v[v_tmp]

    v_mul_lo_u32 v[v_in_ihi+1], s[s_stride_h], v[v_in_ihi+1]
    v_sub_nc_i32 v[v_in_ihi+1], v[v_in_ihi+1], s[s_pad_h]
    v_mul_lo_u32 v[v_in_iwi+1], s[s_stride_w], v[v_in_iwi+1]
    v_sub_nc_i32 v[v_in_iwi+1], v[v_in_iwi+1], s[s_pad_w]

    v_cmpx_le_u32 1, v[v_in_flag+0]
    global_load_dwordx2 v[v_ax+ 0:v_ax+ 1], v[v_in_os+0], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    v_mul_lo_u32 v[v_tmp], s[s_wi], v[v_in_ihi+1]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, 1
    v_add_nc_u32 v[v_tmp], v[v_in_iwi+1], v[v_tmp]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, v[v_in_flag+1]
    v_mul_lo_u32 v[v_in_os+1], s[s_in_stride_wi], v[v_tmp]

    v_cmpx_le_u32 1, v[v_in_flag+1]
    global_load_dwordx2 v[v_ax+ 2:v_ax+ 3], v[v_in_os+1], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    .mdiv_u32_rem_vs v_in_iwi+2,v_in_ihi+2,v_tmp+5,s_magic_1,s_shift_m1,s_wo,v_tmp
    v_add_nc_u32 v[v_tmp+5], s[s_ib_stride], v[v_tmp+5]
    v_mul_lo_u32 v[v_in_ihi+2], s[s_stride_h], v[v_in_ihi+2]
    .v_clear_nc v_ax+4, 4
    v_sub_nc_i32 v[v_in_ihi+2], v[v_in_ihi+2], s[s_pad_h]
    v_mul_lo_u32 v[v_in_iwi+2], s[s_stride_w], v[v_in_iwi+2]
    ;.v_clear_nc v_ax+12, 4
    v_sub_nc_i32 v[v_in_iwi+2], v[v_in_iwi+2], s[s_pad_w]

    .mdiv_u32_rem_vs v_in_iwi+3,v_in_ihi+3,v_tmp+5,s_magic_1,s_shift_m1,s_wo,v_tmp
    v_mul_lo_u32 v[v_in_ihi+3], s[s_stride_h], v[v_in_ihi+3]
    v_sub_nc_i32 v[v_in_ihi+3], v[v_in_ihi+3], s[s_pad_h]
    v_mul_lo_u32 v[v_in_iwi+3], s[s_stride_w], v[v_in_iwi+3]
    v_sub_nc_i32 v[v_in_iwi+3], v[v_in_iwi+3], s[s_pad_w]

    v_mul_lo_u32 v[v_tmp], s[s_wi], v[v_in_ihi+2]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, 1
    v_add_nc_u32 v[v_tmp], v[v_in_iwi+2], v[v_tmp]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, v[v_in_flag+2]
    v_mul_lo_u32 v[v_in_os+2], s[s_in_stride_wi], v[v_tmp]

    v_cmpx_le_u32 1, v[v_in_flag+2]
    global_load_dwordx2 v[v_ax+ 4:v_ax+ 5], v[v_in_os+2], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    v_mul_lo_u32 v[v_tmp], s[s_wi], v[v_in_ihi+3]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, 1
    v_add_nc_u32 v[v_tmp], v[v_in_iwi+3], v[v_tmp]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, v[v_in_flag+3]
    v_mul_lo_u32 v[v_in_os+3], s[s_in_stride_wi], v[v_tmp]

    v_cmpx_le_u32 1, v[v_in_flag+3]
    global_load_dwordx2 v[v_ax+ 6:v_ax+ 7], v[v_in_os+3], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    s_mul_i32 s[s_br], s[s_wo], s[s_ho]

    s_mul_i32 s[s_out_stride_wo], s[s_k], s[s_group]
    s_mul_i32 s[s_in_diff_wi], s[s_dilation_w], s[s_in_stride_wi]
    s_mov_b32 s[s_move_slice_k_ix], 0

    s_mul_i32 s[s_out_stride_n], s[s_br], s[s_out_stride_wo]
    s_mul_i32 s[s_tmp+1], s[s_block_ig], s[s_k]
    s_mul_i32 s[s_tmp+4], s[s_block_in], s[s_out_stride_n]
    ;s_lshl_b32 s[s_tmp+5], s[s_block_ik], 0
    s_add_u32 s[s_tmp+1], s[s_tmp+1], s[s_tmp+4]
    s_add_u32 s[s_tmp+1], s[s_tmp+1], s[s_block_ik]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp+1]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], 0

    ; calculate diffs, for y, x
    s_sub_i32 s[s_tmp+3], s[s_x], 1
    s_mul_i32 s[s_tmp], s[s_in_diff_wi], s[s_tmp+3]
    s_mul_i32 s[s_tmp+1], s[s_in_stride_wi], s[s_wi]
    s_mul_i32 s[s_tmp+1], s[s_tmp+1], s[s_dilation_h]
    s_sub_i32 s[s_in_diff_hi], s[s_tmp+1], s[s_tmp]
    s_mul_i32 s[s_dilation_w_x], s[s_dilation_w], s[s_tmp+3]
    s_mul_i32 s[s_dilation_w_x], s[s_dilation_w_x], -1


    v_add_nc_u32 v[v_tmp+5], s[s_ib_stride], v[v_ib]
    s_mul_i32 s[s_out_stride], s[s_stride_m], s[s_out_stride_wo]

    ;s_lshl_b32 s[s_out_stride], s[s_out_stride], 1
    ;s_lshl_b32 s[s_out_stride_n], s[s_out_stride_n], 1

    ; output offset
    v_mul_lo_u32 v[v_out_os], s[s_k], v[v_ib]
    ;v_lshlrev_b32 v[v_out_os], 1, v[v_out_os]
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os]
    v_cndmask_b32 v[v_out_flag], 0, 1
    v_add_nc_u32 v[v_tmp+4], s[s_ib_stride], v[v_tmp+5]

    v_mul_lo_u32 v[v_out_os+1], s[s_k], v[v_tmp+5]
    ;v_lshlrev_b32 v[v_out_os+1], 1, v[v_out_os+1]
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+1]
    v_cndmask_b32 v[v_out_flag+1], 0, 1
    v_add_nc_u32 v[v_tmp+5], s[s_ib_stride], v[v_tmp+4]

    v_mul_lo_u32 v[v_out_os+2], s[s_k], v[v_tmp+4]
    ;v_lshlrev_b32 v[v_out_os+2], 1, v[v_out_os+2]
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+2]
    v_cndmask_b32 v[v_out_flag+2], 0, 1

    v_mul_lo_u32 v[v_out_os+3], s[s_k], v[v_tmp+5]
    ;v_lshlrev_b32 v[v_out_os+3], 1, v[v_out_os+3]
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+3]
    v_cndmask_b32 v[v_out_flag+3], 0, 1

    s_mov_b32 s[s_sld_b_stride],    k_n_dword*4*2

    s_waitcnt vmcnt(4)

    v_cmpx_le_u32 1, v[v_wei_flag+0]
    ds_write2_b32 v[v_sst_b_os+0], v[v_gld_b+0], v[v_gld_b+1], offset0:k_n_dword*0  offset1:k_n_dword*1
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_wei_flag+1]
    ds_write2_b32 v[v_sst_b_os+1], v[v_gld_b+2], v[v_gld_b+3], offset0:k_n_dword*0  offset1:k_n_dword*1
    s_mov_b64 exec, -1

    .v_clear_nc v_c, 64

    s_waitcnt lgkmcnt(0)
    s_barrier

    ds_read_b128 v[v_b+ 0:v_b+ 3], v[v_sld_b_os], offset:k_n_dword*4*0 + 0*4
    ds_read_b128 v[v_b+ 4:v_b+ 7], v[v_sld_b_os], offset:k_n_dword*4*0 + 4*4
    ds_read_b128 v[v_b+ 8:v_b+11], v[v_sld_b_os], offset:k_n_dword*4*0 + 8*4
    ds_read_b128 v[v_b+12:v_b+15], v[v_sld_b_os], offset:k_n_dword*4*0 +12*4
    ds_read_b128 v[v_b+16:v_b+19], v[v_sld_b_os], offset:k_n_dword*4*1 + 0*4
    ds_read_b128 v[v_b+20:v_b+23], v[v_sld_b_os], offset:k_n_dword*4*1 + 4*4
    ds_read_b128 v[v_b+24:v_b+27], v[v_sld_b_os], offset:k_n_dword*4*1 + 8*4
    ds_read_b128 v[v_b+28:v_b+31], v[v_sld_b_os], offset:k_n_dword*4*1 +12*4

    s_sub_i32 s[s_kitr], s[s_wei_stride_k], 8
    v_add_nc_u32 v[v_sld_b_os], s[s_sld_b_stride], v[v_sld_b_os]            ; accumulate sld_b_os

    s_cmp_gt_i32 s[s_kitr], 0
    
    s_cbranch_scc0 L_igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_end

L_igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_body:
    ; accumulate im

    ; a buffer x
    ;--- start move slice window
    s_add_u32 s[s_move_slice_k_ix], 1, s[s_move_slice_k_ix]
    s_cmp_le_u32 s[s_x], s[s_move_slice_k_ix]
    s_cselect_b32 s[s_tmp], s[s_dilation_w_x], s[s_dilation_w]
    s_cselect_b32 s[s_tmp+1], s[s_in_diff_hi], s[s_in_diff_wi]
    v_add_nc_u32 v[v_in_iwi+0], s[s_tmp], v[v_in_iwi+0]
    v_add_nc_u32 v[v_in_iwi+1], s[s_tmp], v[v_in_iwi+1]
    v_add_nc_u32 v[v_in_iwi+2], s[s_tmp], v[v_in_iwi+2]
    v_add_nc_u32 v[v_in_iwi+3], s[s_tmp], v[v_in_iwi+3]
    v_add_nc_u32 v[v_in_os+0], s[s_tmp+1], v[v_in_os+0]
    v_add_nc_u32 v[v_in_os+1], s[s_tmp+1], v[v_in_os+1]
    v_add_nc_u32 v[v_in_os+2], s[s_tmp+1], v[v_in_os+2]
    v_add_nc_u32 v[v_in_os+3], s[s_tmp+1], v[v_in_os+3]
    s_cbranch_scc0 igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_acc_yx_x_end_1
    s_mov_b32 s[s_move_slice_k_ix], 0
    v_add_nc_i32 v[v_in_ihi+0], s[s_dilation_h], v[v_in_ihi+0]
    v_add_nc_i32 v[v_in_ihi+1], s[s_dilation_h], v[v_in_ihi+1]
    v_add_nc_i32 v[v_in_ihi+2], s[s_dilation_h], v[v_in_ihi+2]
    v_add_nc_i32 v[v_in_ihi+3], s[s_dilation_h], v[v_in_ihi+3]
igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_acc_yx_x_end_1:
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+0]
    v_cndmask_b32 v[v_in_flag+0], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, 1
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+0]
    v_cndmask_b32 v[v_in_flag+0], 0, v[v_in_flag+0]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, v[v_in_flag+1]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, v[v_in_flag+2]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, v[v_in_flag+3]
    ;--- end move slice window

    ;s_waitcnt vmcnt(0)
    .v_clear_nc v_ay, 4
    v_cmpx_le_u32 1, v[v_in_flag+0]
    global_load_dwordx2 v[v_ay+ 0:v_ay+ 1], v[v_in_os+0], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_in_flag+1]
    global_load_dwordx2 v[v_ay+ 2:v_ay+ 3], v[v_in_os+1], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1
    .v_clear_nc v_ay+4, 4
    v_cmpx_le_u32 1, v[v_in_flag+2]
    global_load_dwordx2 v[v_ay+ 4:v_ay+ 5], v[v_in_os+2], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_in_flag+3]
    global_load_dwordx2 v[v_ay+ 6:v_ay+ 7], v[v_in_os+3], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    s_waitcnt vmcnt(4) lgkmcnt(4)
    .fma_1x16_int8x4 v_c+ 0, v_ax + 0, v_b + 0
    .fma_1x16_int8x4 v_c+16, v_ax + 2, v_b + 0
    .fma_1x16_int8x4 v_c+32, v_ax + 4, v_b + 0
    .fma_1x16_int8x4 v_c+48, v_ax + 6, v_b + 0

    ds_read_b128 v[v_b+ 0:v_b+ 3], v[v_sld_b_os], offset:k_n_dword*4*0 + 0*4
    ds_read_b128 v[v_b+ 4:v_b+ 7], v[v_sld_b_os], offset:k_n_dword*4*0 + 4*4
    ds_read_b128 v[v_b+ 8:v_b+11], v[v_sld_b_os], offset:k_n_dword*4*0 + 8*4
    ds_read_b128 v[v_b+12:v_b+15], v[v_sld_b_os], offset:k_n_dword*4*0 +12*4

    s_waitcnt lgkmcnt(4)
    .fma_1x16_int8x4 v_c+ 0, v_ax + 1, v_b +16
    .fma_1x16_int8x4 v_c+16, v_ax + 3, v_b +16
    .fma_1x16_int8x4 v_c+32, v_ax + 5, v_b +16
    .fma_1x16_int8x4 v_c+48, v_ax + 7, v_b +16

    ds_read_b128 v[v_b+16:v_b+19], v[v_sld_b_os], offset:k_n_dword*4*1 + 0*4
    ds_read_b128 v[v_b+20:v_b+23], v[v_sld_b_os], offset:k_n_dword*4*1 + 4*4
    ds_read_b128 v[v_b+24:v_b+27], v[v_sld_b_os], offset:k_n_dword*4*1 + 8*4
    ds_read_b128 v[v_b+28:v_b+31], v[v_sld_b_os], offset:k_n_dword*4*1 +12*4

    s_sub_i32 s[s_kitr], s[s_kitr], 8
    v_add_nc_u32 v[v_sld_b_os], s[s_sld_b_stride], v[v_sld_b_os]            ; accumulate sld_b_os
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_end_1

    ; a buffer y
    ;--- start move slice window
    s_add_u32 s[s_move_slice_k_ix], 1, s[s_move_slice_k_ix]
    s_cmp_le_u32 s[s_x], s[s_move_slice_k_ix]
    s_cselect_b32 s[s_tmp], s[s_dilation_w_x], s[s_dilation_w]
    s_cselect_b32 s[s_tmp+1], s[s_in_diff_hi], s[s_in_diff_wi]
    v_add_nc_u32 v[v_in_iwi+0], s[s_tmp], v[v_in_iwi+0]
    v_add_nc_u32 v[v_in_iwi+1], s[s_tmp], v[v_in_iwi+1]
    v_add_nc_u32 v[v_in_iwi+2], s[s_tmp], v[v_in_iwi+2]
    v_add_nc_u32 v[v_in_iwi+3], s[s_tmp], v[v_in_iwi+3]
    v_add_nc_u32 v[v_in_os+0], s[s_tmp+1], v[v_in_os+0]
    v_add_nc_u32 v[v_in_os+1], s[s_tmp+1], v[v_in_os+1]
    v_add_nc_u32 v[v_in_os+2], s[s_tmp+1], v[v_in_os+2]
    v_add_nc_u32 v[v_in_os+3], s[s_tmp+1], v[v_in_os+3]
    s_cbranch_scc0 igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_acc_yx_x_end_2
    s_mov_b32 s[s_move_slice_k_ix], 0
    v_add_nc_i32 v[v_in_ihi+0], s[s_dilation_h], v[v_in_ihi+0]
    v_add_nc_i32 v[v_in_ihi+1], s[s_dilation_h], v[v_in_ihi+1]
    v_add_nc_i32 v[v_in_ihi+2], s[s_dilation_h], v[v_in_ihi+2]
    v_add_nc_i32 v[v_in_ihi+3], s[s_dilation_h], v[v_in_ihi+3]
igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_acc_yx_x_end_2:
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+0]
    v_cndmask_b32 v[v_in_flag+0], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, 1
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+0]
    v_cndmask_b32 v[v_in_flag+0], 0, v[v_in_flag+0]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, v[v_in_flag+1]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, v[v_in_flag+2]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, v[v_in_flag+3]
    ;--- end move slice window

    .v_clear_nc v_ax, 4
    v_cmpx_le_u32 1, v[v_in_flag+0]
    global_load_dwordx2 v[v_ax+ 0:v_ax+ 1], v[v_in_os+0], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_in_flag+1]
    global_load_dwordx2 v[v_ax+ 2:v_ax+ 3], v[v_in_os+1], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1
    .v_clear_nc v_ax+4, 4
    v_cmpx_le_u32 1, v[v_in_flag+2]
    global_load_dwordx2 v[v_ax+ 4:v_ax+ 5], v[v_in_os+2], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_in_flag+3]
    global_load_dwordx2 v[v_ax+ 6:v_ax+ 7], v[v_in_os+3], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    s_waitcnt vmcnt(4) lgkmcnt(4)
    .fma_1x16_int8x4 v_c+ 0, v_ay + 0, v_b + 0
    .fma_1x16_int8x4 v_c+16, v_ay + 2, v_b + 0
    .fma_1x16_int8x4 v_c+32, v_ay + 4, v_b + 0
    .fma_1x16_int8x4 v_c+48, v_ay + 6, v_b + 0

    ds_read_b128 v[v_b+ 0:v_b+ 3], v[v_sld_b_os], offset:k_n_dword*4*0 + 0*4
    ds_read_b128 v[v_b+ 4:v_b+ 7], v[v_sld_b_os], offset:k_n_dword*4*0 + 4*4
    ds_read_b128 v[v_b+ 8:v_b+11], v[v_sld_b_os], offset:k_n_dword*4*0 + 8*4
    ds_read_b128 v[v_b+12:v_b+15], v[v_sld_b_os], offset:k_n_dword*4*0 +12*4

    s_waitcnt lgkmcnt(4)
    .fma_1x16_int8x4 v_c+ 0, v_ay + 1, v_b +16
    .fma_1x16_int8x4 v_c+16, v_ay + 3, v_b +16
    .fma_1x16_int8x4 v_c+32, v_ay + 5, v_b +16
    .fma_1x16_int8x4 v_c+48, v_ay + 7, v_b +16

    ds_read_b128 v[v_b+16:v_b+19], v[v_sld_b_os], offset:k_n_dword*4*1 + 0*4
    ds_read_b128 v[v_b+20:v_b+23], v[v_sld_b_os], offset:k_n_dword*4*1 + 4*4
    ds_read_b128 v[v_b+24:v_b+27], v[v_sld_b_os], offset:k_n_dword*4*1 + 8*4
    ds_read_b128 v[v_b+28:v_b+31], v[v_sld_b_os], offset:k_n_dword*4*1 +12*4


    s_sub_i32 s[s_kitr], s[s_kitr], 8
    v_add_nc_u32 v[v_sld_b_os], s[s_sld_b_stride], v[v_sld_b_os]            ; accumulate sld_b_os
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc1 L_igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_body

L_igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_end:
    s_waitcnt vmcnt(0)

    v_mov_b32 v[v_ay + 0], v[v_ax + 0]
    v_mov_b32 v[v_ay + 1], v[v_ax + 1]
    v_mov_b32 v[v_ay + 2], v[v_ax + 2]
    v_mov_b32 v[v_ay + 3], v[v_ax + 3]
    v_mov_b32 v[v_ay + 4], v[v_ax + 4]
    v_mov_b32 v[v_ay + 5], v[v_ax + 5]
    v_mov_b32 v[v_ay + 6], v[v_ax + 6]
    v_mov_b32 v[v_ay + 7], v[v_ax + 7]

L_igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_end_1:
    s_waitcnt vmcnt(0)

    s_sub_i32 s[s_batch_m], s[s_batch_m], 1
    v_add_nc_u32 v[v_ib], s[s_stride_m],  v[v_ib]

    s_cmp_gt_i32 s[s_batch_m], 0
    s_cbranch_scc0 L_igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_end_not_load_next
    ; --- start move slice for batch m
    ; ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
    ; iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w
    ; we will update v_in_os below, so use this as v_tmp
    .mdiv_u32_rem_vs v_in_iwi,v_in_ihi,v_ib,s_magic_1,s_shift_m1,s_wo,v_in_os
    v_mul_u32_u24 v[v_in_ihi], s[s_stride_h], v[v_in_ihi]
    .v_clear_nc v_ax, 2
    v_add_nc_u32 v[v_in_flag+1], s[s_ib_stride], v[v_ib]
    v_sub_nc_i32 v[v_in_ihi], v[v_in_ihi], s[s_pad_h]
    v_mul_u32_u24 v[v_in_iwi], s[s_stride_w], v[v_in_iwi]
    .v_clear_nc v_ax+2, 2
    v_sub_nc_i32 v[v_in_iwi], v[v_in_iwi], s[s_pad_w]

    .mdiv_u32_rem_vs v_in_iwi+1,v_in_ihi+1,v_in_flag+1,s_magic_1,s_shift_m1,s_wo,v_in_os+1

    v_mul_u32_u24 v[v_in_os], s[s_wi], v[v_in_ihi]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi]
    v_cndmask_b32 v[v_in_flag], 0, 1
    v_add_nc_u32 v[v_in_os], v[v_in_iwi], v[v_in_os]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi]
    v_cndmask_b32 v[v_in_flag], 0, v[v_in_flag]
    v_mul_lo_u32 v[v_in_os], s[s_in_stride_wi], v[v_in_os]
    
    v_mul_u32_u24 v[v_in_ihi+1], s[s_stride_h], v[v_in_ihi+1]
    v_sub_nc_i32 v[v_in_ihi+1], v[v_in_ihi+1], s[s_pad_h]
    v_mul_u32_u24 v[v_in_iwi+1], s[s_stride_w], v[v_in_iwi+1]
    v_sub_nc_i32 v[v_in_iwi+1], v[v_in_iwi+1], s[s_pad_w]

    v_add_nc_u32 v[v_in_flag+2], s[s_ib_stride], v[v_in_flag+1]

    v_cmpx_le_u32 1, v[v_in_flag+0]
    global_load_dwordx2 v[v_ax+ 0:v_ax+ 1], v[v_in_os], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    v_mul_u32_u24 v[v_in_os+1], s[s_wi], v[v_in_ihi+1]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, 1
    v_add_nc_u32 v[v_in_os+1], v[v_in_iwi+1], v[v_in_os+1]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, v[v_in_flag+1]
    v_mul_lo_u32 v[v_in_os+1], s[s_in_stride_wi], v[v_in_os+1]

    v_cmpx_le_u32 1, v[v_in_flag+1]
    global_load_dwordx2 v[v_ax+ 2:v_ax+ 3], v[v_in_os+1], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    .mdiv_u32_rem_vs v_in_iwi+2,v_in_ihi+2,v_in_flag+2,s_magic_1,s_shift_m1,s_wo,v_in_os+2
    v_add_nc_u32 v[v_in_flag+3], s[s_ib_stride], v[v_in_flag+2]
    v_mul_lo_u32 v[v_in_ihi+2], s[s_stride_h], v[v_in_ihi+2]
    .v_clear_nc v_ax+4, 2
    v_sub_nc_i32 v[v_in_ihi+2], v[v_in_ihi+2], s[s_pad_h]
    v_mul_lo_u32 v[v_in_iwi+2], s[s_stride_w], v[v_in_iwi+2]
    .v_clear_nc v_ax+6, 2
    v_sub_nc_i32 v[v_in_iwi+2], v[v_in_iwi+2], s[s_pad_w]

    .mdiv_u32_rem_vs v_in_iwi+3,v_in_ihi+3,v_in_flag+3,s_magic_1,s_shift_m1,s_wo,v_in_os+3
    v_mul_lo_u32 v[v_in_ihi+3], s[s_stride_h], v[v_in_ihi+3]
    v_sub_nc_i32 v[v_in_ihi+3], v[v_in_ihi+3], s[s_pad_h]
    v_mul_lo_u32 v[v_in_iwi+3], s[s_stride_w], v[v_in_iwi+3]
    v_sub_nc_i32 v[v_in_iwi+3], v[v_in_iwi+3], s[s_pad_w]

    v_mul_lo_u32 v[v_in_os+2], s[s_wi], v[v_in_ihi+2]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, 1
    v_add_nc_u32 v[v_in_os+2], v[v_in_iwi+2], v[v_in_os+2]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+2]
    v_cndmask_b32 v[v_in_flag+2], 0, v[v_in_flag+2]
    v_mul_lo_u32 v[v_in_os+2], s[s_in_stride_wi], v[v_in_os+2]

    v_cmpx_le_u32 1, v[v_in_flag+2]
    global_load_dwordx2 v[v_ax+ 4:v_ax+ 5], v[v_in_os+2], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    v_mul_lo_u32 v[v_in_os+3], s[s_wi], v[v_in_ihi+3]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, 1
    v_add_nc_u32 v[v_in_os+3], v[v_in_iwi+3], v[v_in_os+3]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+3]
    v_cndmask_b32 v[v_in_flag+3], 0, v[v_in_flag+3]
    v_mul_lo_u32 v[v_in_os+3], s[s_in_stride_wi], v[v_in_os+3]

    v_cmpx_le_u32 1, v[v_in_flag+3]
    global_load_dwordx2 v[v_ax+ 6:v_ax+ 7], v[v_in_os+3], s[s_p_in:s_p_in+1]
    s_mov_b64 exec, -1

    s_mov_b32 s[s_move_slice_k_ix], 0

L_igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_end_not_load_next:
    ; --- end move slice for batch m

    s_waitcnt lgkmcnt(4)
    .fma_1x16_int8x4 v_c+ 0, v_ay + 0, v_b + 0
    .fma_1x16_int8x4 v_c+16, v_ay + 2, v_b + 0
    .fma_1x16_int8x4 v_c+32, v_ay + 4, v_b + 0
    .fma_1x16_int8x4 v_c+48, v_ay + 6, v_b + 0

    s_waitcnt lgkmcnt(0)
    .fma_1x16_int8x4 v_c+ 0, v_ay + 1, v_b +16
    .fma_1x16_int8x4 v_c+16, v_ay + 3, v_b +16
    .fma_1x16_int8x4 v_c+32, v_ay + 5, v_b +16
    .fma_1x16_int8x4 v_c+48, v_ay + 7, v_b +16

    v_mov_b32 v[v_sld_b_os], 0                                  ; reset to start
 
    .activ_int32 v_c + 0, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 0+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+8, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .pack_i8x4_i32_r4 v_c_buf+ 0, v_c+ 0, s_0xff

    .activ_int32 v_c + 0+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+16, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 0+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+24, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .pack_i8x4_i32_r4 v_c_buf+ 4, v_c+16, s_0xff
    v_cmpx_le_u32 1, v[v_out_flag]
    global_store_dwordx4 v[v_out_os], v[v_c_buf+0:v_c_buf+3], s[s_p_out:s_p_out+1]
    s_mov_b64 exec, -1

    v_cmpx_le_u32 1, v[v_out_flag+1]
    global_store_dwordx4 v[v_out_os+1], v[v_c_buf+ 4:v_c_buf+ 7], s[s_p_out:s_p_out+1]
    s_mov_b64 exec, -1

    .activ_int32 v_c + 0+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+32, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 0+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+40, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .pack_i8x4_i32_r4 v_c_buf+ 8, v_c+32, s_0xff

    .activ_int32 v_c + 0+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+48, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 0+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 1+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 2+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 3+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 4+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 5+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 6+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_int32 v_c + 7+56, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .pack_i8x4_i32_r4 v_c_buf+12, v_c+48, s_0xff

    v_cmpx_le_u32 1, v[v_out_flag+2]
    global_store_dwordx4 v[v_out_os+2], v[v_c_buf+ 8:v_c_buf+11], s[s_p_out:s_p_out+1]
    s_mov_b64 exec, -1

    v_cmpx_le_u32 1, v[v_out_flag+3]
    global_store_dwordx4 v[v_out_os+3], v[v_c_buf+12:v_c_buf+15], s[s_p_out:s_p_out+1]
    s_mov_b64 exec, -1

    s_cmp_le_i32 s[s_batch_m], 0

    s_cbranch_scc1 L_igemm_fwd_btm_nhwc_int8_512x16x8_r2_end
    ds_read_b128 v[v_b+ 0:v_b+ 3], v[v_sld_b_os], offset:k_n_dword*4*0 + 0*4
    ds_read_b128 v[v_b+ 4:v_b+ 7], v[v_sld_b_os], offset:k_n_dword*4*0 + 4*4
    ds_read_b128 v[v_b+ 8:v_b+11], v[v_sld_b_os], offset:k_n_dword*4*0 + 8*4
    ds_read_b128 v[v_b+12:v_b+15], v[v_sld_b_os], offset:k_n_dword*4*0 +12*4
    ds_read_b128 v[v_b+16:v_b+19], v[v_sld_b_os], offset:k_n_dword*4*1 + 0*4
    ds_read_b128 v[v_b+20:v_b+23], v[v_sld_b_os], offset:k_n_dword*4*1 + 4*4
    ds_read_b128 v[v_b+24:v_b+27], v[v_sld_b_os], offset:k_n_dword*4*1 + 8*4
    ds_read_b128 v[v_b+28:v_b+31], v[v_sld_b_os], offset:k_n_dword*4*1 +12*4

    .v_clear_nc v_c, 64
    v_add_nc_u32 v[v_sld_b_os], s[s_sld_b_stride], v[v_sld_b_os]            ; accumulate sld_b_os

    v_add_nc_u32 v[v_out_os], s[s_out_stride], v[v_out_os]
    s_sub_i32 s[s_kitr], s[s_wei_stride_k], 8
    v_add_nc_u32 v[v_out_os+1], s[s_out_stride], v[v_out_os+1]
    v_add_nc_u32 v[v_out_os+2], s[s_out_stride], v[v_out_os+2]
    v_add_nc_u32 v[v_out_os+3], s[s_out_stride], v[v_out_os+3]
    
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os]
    v_cndmask_b32 v[v_out_flag], 0, 1
    s_cmp_gt_i32 s[s_kitr], 0
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+1]
    v_cndmask_b32 v[v_out_flag+1], 0, 1
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+2]
    v_cndmask_b32 v[v_out_flag+2], 0, 1
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+3]
    v_cndmask_b32 v[v_out_flag+3], 0, 1

    s_cbranch_scc0 L_igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_end
    s_branch L_igemm_fwd_btm_nhwc_int8_512x16x8_r2_fma_body
L_igemm_fwd_btm_nhwc_int8_512x16x8_r2_end:
    s_endpgm

; LDS: 2 * 4    * 4  * 128
;      r2  4dword 4    threads
.rodata
.p2align 6
.amdhsa_kernel igemm_fwd_btm_nhwc_int8_512x16x8_r2
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 140
    .amdhsa_next_free_sgpr 62
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
    .amdhsa_wavefront_size32 1
    .amdhsa_workgroup_processor_mode 0
.end_amdhsa_kernel
