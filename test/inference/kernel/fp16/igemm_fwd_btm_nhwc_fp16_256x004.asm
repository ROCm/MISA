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
.set k_n_dword, 4

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
.set v_c_buf,           v_c
.set v_sld_b_os,        8
.set v_ax,              9
.set v_ay,              25
.set v_ib,              41
.set v_b,               42
.set v_gld_b,           v_b
.set v_wei_iy_list,     v_b+4
.set v_wei_ix_list,     v_b+5
.set v_wei_flag,        v_b+6
.set v_wei_os,          v_b+7
.set v_tmp,             v_b+8
.set v_wei_ik,          v_ay
.set v_wei_ic,          v_ay+1
.set v_wei_ie,          v_ay+2
.set v_wei_flag_ik,     v_ay+3
.set v_sst_b_os,        v_ay+4
.set v_in_os,           74
.set v_in_ihi,          76
.set v_in_iwi,          78
.set v_in_flag,         80
.set v_out_os,          82
.set v_out_flag,        84
.set v_tid,             86
.set v_end,             88

; short wide igemv
.text
.globl igemm_fwd_btm_nhwc_fp16_256x4x16_r1
.p2align 8

.type igemm_fwd_btm_nhwc_fp16_256x4x16_r1,@function
igemm_fwd_btm_nhwc_fp16_256x4x16_r1:
    s_load_dwordx2  s[s_p_in+0:s_p_in+1],    s[s_ka+0:s_ka+1],    0+k_p_in
    s_load_dwordx4  s[s_p_wei+0:s_p_wei+3],  s[s_ka+0:s_ka+1],    0+k_p_wei
    s_load_dwordx16 s[s_hi+0:s_hi+15],    s[s_ka+0:s_ka+1],    0+k_hi
    s_load_dwordx8  s[s_batch_m:s_batch_m+7],    s[s_ka+0:s_ka+1],    0+k_batch_m
    s_load_dword  s[s_shift_pack_0],    s[s_ka+0:s_ka+1],    0+k_shift_pack_0
    v_mov_b32       v[v_tid], v0
    s_mov_b32 s[s_ib_stride], 128

    ; calculate wei offset, 4x32, 4 for k, 32 for yxc, 16 for yx, 2 for c
    v_lshrrev_b32 v[v_wei_ik], 5, v0
    s_mov_b32 s[s_tmp], k_n_dword*4 * 4                             ; 9 dword per row, 4 row
    v_and_b32 v[v_tmp+5], 31, v0
    s_lshl_b32 s[s_block_ig], s[s_block_ig], 1
    v_and_b32 v[v_wei_ic], 1, v0
    s_lshl_b32 s[s_block_in], s[s_block_in], 1
    v_lshrrev_b32 v[v_tmp+4], 1, v0
    v_mov_b32 v[v_ib], v0
    v_mul_u32_u24 v[v_tmp+5],   s[s_tmp]  ,v[v_tmp+5]
    v_lshlrev_b32 v[v_sst_b_os], 2, v[v_wei_ik]             ; store, k*n*k_pack, ds_write2 if possible, n*k_pack->16dword, pad to x
    v_mov_b32 v[v_sld_b_os], 0                              ; load   
    v_lshlrev_b32 v[v_wei_ic], 3, v[v_wei_ic]               ; 8xc, k_pack, 4x dword
    v_and_b32 v[v_wei_ie], 15, v[v_tmp+4]                    ; yx
    v_add_nc_u32 v[v_sst_b_os], v[v_sst_b_os], v[v_tmp+5]   ; note, do not use or due to pad

    s_waitcnt lgkmcnt(0)
    s_bfe_u32 s[s_shift_m2], s[s_shift_pack_0], 0x00080010      ; offset:16, width:8
    s_lshr_b32 s[s_tmp+3], s[s_k], 2
    s_bfe_u32 s[s_shift_m0], s[s_shift_pack_0], 0x00080000      ; offset:0, width:8
    .mdiv_u32_rem_ss s_tmp+4,s_tmp+5,s_bx,s_magic_2,s_shift_m2,s_tmp+3,s_tmp
    s_lshl_b32 s[s_block_ib], s[s_tmp+5], 8                     ; 256
    s_lshl_b32 s[s_block_ik], s[s_tmp+4], 2
    v_add_nc_u32 v[v_ib], s[s_block_ib],  v[v_ib]
    s_mul_i32 s[s_tmp], s[s_x], s[s_c]
    v_add_nc_u32 v[v_wei_ik], s[s_block_ik], v[v_wei_ik]


    v_mad_u32_u24 v[v_tmp+1], s[s_c], v[v_wei_ie], v[v_wei_ic]
    s_mul_i32 s[s_wei_stride_k], s[s_tmp], s[s_y]
    s_lshl_b32 s[s_wei_offset], s[s_c], 3+1                     ; 8x s_c, half
    s_mul_i32 s[s_tmp+5], s[s_wei_stride_k], s[s_k]
    v_mad_u32_u24 v[v_wei_os], s[s_wei_stride_k], v[v_wei_ik], v[v_tmp+1]
    s_mul_i32 s[s_tmp+2], s[s_block_ig], s[s_tmp+5]
    v_cmp_gt_u32 s[s_k], v[v_wei_ik]
    s_add_u32 s[s_p_wei], s[s_p_wei], s[s_tmp+2]
    v_cndmask_b32 v[v_wei_flag_ik], 0, 1
    s_addc_u32 s[s_p_wei+1], s[s_p_wei+1], 0
    v_lshlrev_b32 v[v_wei_os], 1, v[v_wei_os]

    ; divide x
    .mdiv_u32_rem_vs v_wei_ix_list+0,v_wei_iy_list+0,v_wei_ie,s_magic_0,s_shift_m0,s_x,v_tmp
    ;v_add_nc_u32 v[v_wei_os+1], s[s_wei_offset], v[v_wei_os+0]
    ;v_add_nc_u32 v[v_wei_ie], 8, v[v_wei_ie]
    v_cmp_gt_u32 s[s_y], v[v_wei_iy_list+0]
    v_cndmask_b32 v[v_wei_flag+0], 0, v[v_wei_flag_ik]
    v_cmp_gt_u32 s[s_x], v[v_wei_ix_list+0]
    v_cndmask_b32 v[v_wei_flag+0], 0, v[v_wei_flag+0]

    v_cmpx_le_u32 1, v[v_wei_flag+0]
    global_load_dwordx4 v[v_gld_b+0:v_gld_b+3], v[v_wei_os+0], s[s_p_wei:s_p_wei+1]
    s_mov_b64 exec, -1

    ; s_mov_b32 s[s_tmp+5], 128*k_n_dword*4       ; stride for wei sst offset. 32 thread for gemm_k, each thread store 4 c, hence 32*4=128 gemm_k

    ; calculate in offset
    s_mul_i32 s[s_in_stride_wi], s[s_c], s[s_group]
    s_bfe_u32 s[s_shift_m1], s[s_shift_pack_0], 0x00080008 ; offset:8, width:8
    s_mul_i32 s[s_tmp+2], s[s_wi], s[s_in_stride_wi]
    s_mul_i32 s[s_tmp+0], s[s_block_ig], s[s_c]
    s_mul_i32 s[s_in_stride_n], s[s_hi], s[s_tmp+2]
    s_mul_i32 s[s_tmp+3], s[s_block_in], s[s_in_stride_n]
    s_lshl_b32 s[s_in_stride_wi], s[s_in_stride_wi], 1
    s_add_u32 s[s_tmp+0], s[s_tmp+0], s[s_tmp+3]
    ;v_add_nc_u32 v[v_sst_b_os+1], s[s_tmp+5], v[v_sst_b_os+0]

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
    ; v_add_nc_u32 v[v_tmp+5], s[s_ib_stride], v[v_tmp+5]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi]
    v_cndmask_b32 v[v_in_flag], 0, 1
    v_add_nc_u32 v[v_tmp], v[v_in_iwi], v[v_tmp]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi]
    v_cndmask_b32 v[v_in_flag], 0, v[v_in_flag]
    v_mul_lo_u32 v[v_in_os], s[s_in_stride_wi], v[v_tmp]

    v_mul_lo_u32 v[v_in_ihi+1], s[s_stride_h], v[v_in_ihi+1]
    .v_clear_nc v_ax+8, 4
    v_sub_nc_i32 v[v_in_ihi+1], v[v_in_ihi+1], s[s_pad_h]
    v_mul_lo_u32 v[v_in_iwi+1], s[s_stride_w], v[v_in_iwi+1]
    .v_clear_nc v_ax+12, 4
    v_sub_nc_i32 v[v_in_iwi+1], v[v_in_iwi+1], s[s_pad_w]

    v_cmpx_le_u32 1, v[v_in_flag]
    global_load_dwordx4 v[v_ax+0:v_ax+3], v[v_in_os], s[s_p_in:s_p_in+1]
    global_load_dwordx4 v[v_ax+4:v_ax+7], v[v_in_os], s[s_p_in:s_p_in+1] offset:16
    s_mov_b64 exec, -1

    v_mul_lo_u32 v[v_tmp], s[s_wi], v[v_in_ihi+1]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, 1
    v_add_nc_u32 v[v_tmp], v[v_in_iwi+1], v[v_tmp]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, v[v_in_flag+1]
    v_mul_lo_u32 v[v_in_os+1], s[s_in_stride_wi], v[v_tmp]

    v_cmpx_le_u32 1, v[v_in_flag+1]
    global_load_dwordx4 v[v_ax+ 8:v_ax+11], v[v_in_os+1], s[s_p_in:s_p_in+1]
    global_load_dwordx4 v[v_ax+12:v_ax+15], v[v_in_os+1], s[s_p_in:s_p_in+1] offset:16
    s_mov_b64 exec, -1

    s_mul_i32 s[s_br], s[s_wo], s[s_ho]

    s_mul_i32 s[s_out_stride_wo], s[s_k], s[s_group]
    s_mul_i32 s[s_in_diff_wi], s[s_dilation_w], s[s_in_stride_wi]
    s_mov_b32 s[s_move_slice_k_ix], 0

    s_mul_i32 s[s_out_stride_n], s[s_br], s[s_out_stride_wo]
    s_mul_i32 s[s_tmp+1], s[s_block_ig], s[s_k]
    s_mul_i32 s[s_tmp+4], s[s_block_in], s[s_out_stride_n]
    s_lshl_b32 s[s_tmp+5], s[s_block_ik], 1
    s_add_u32 s[s_tmp+1], s[s_tmp+1], s[s_tmp+4]
    s_add_u32 s[s_tmp+1], s[s_tmp+1], s[s_tmp+5]
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

    s_lshl_b32 s[s_out_stride], s[s_out_stride], 1
    s_lshl_b32 s[s_out_stride_n], s[s_out_stride_n], 1

    ; output offset
    v_mul_lo_u32 v[v_out_os], s[s_k], v[v_ib]
    v_lshlrev_b32 v[v_out_os], 1, v[v_out_os]
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os]
    v_cndmask_b32 v[v_out_flag], 0, 1
    v_add_nc_u32 v[v_tmp+4], s[s_ib_stride], v[v_tmp+5]

    v_mul_lo_u32 v[v_out_os+1], s[s_k], v[v_tmp+5]
    v_lshlrev_b32 v[v_out_os+1], 1, v[v_out_os+1]
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+1]
    v_cndmask_b32 v[v_out_flag+1], 0, 1

    s_mov_b32 s[s_sld_b_stride],    k_n_dword*8*4

    s_waitcnt vmcnt(4)

    v_cmpx_le_u32 1, v[v_wei_flag+0]
    ds_write2_b32 v[v_sst_b_os+0], v[v_gld_b+0], v[v_gld_b+1], offset0:k_n_dword*0  offset1:k_n_dword*1
    ds_write2_b32 v[v_sst_b_os+0], v[v_gld_b+2], v[v_gld_b+3], offset0:k_n_dword*2  offset1:k_n_dword*3
    s_mov_b64 exec, -1

    .v_clear_nc v_c, 8

    s_waitcnt lgkmcnt(0)
    s_barrier

    ds_read_b128 v[v_b+ 0:v_b+ 3], v[v_sld_b_os], offset:k_n_dword*4*0
    ds_read_b128 v[v_b+ 4:v_b+ 7], v[v_sld_b_os], offset:k_n_dword*4*1
    ds_read_b128 v[v_b+ 8:v_b+11], v[v_sld_b_os], offset:k_n_dword*4*2
    ds_read_b128 v[v_b+12:v_b+15], v[v_sld_b_os], offset:k_n_dword*4*3
    ds_read_b128 v[v_b+16:v_b+19], v[v_sld_b_os], offset:k_n_dword*4*4
    ds_read_b128 v[v_b+20:v_b+23], v[v_sld_b_os], offset:k_n_dword*4*5
    ds_read_b128 v[v_b+24:v_b+27], v[v_sld_b_os], offset:k_n_dword*4*6
    ds_read_b128 v[v_b+28:v_b+31], v[v_sld_b_os], offset:k_n_dword*4*7

    s_sub_i32 s[s_kitr], s[s_wei_stride_k], 16
    v_add_nc_u32 v[v_sld_b_os], s[s_sld_b_stride], v[v_sld_b_os]            ; accumulate sld_b_os

    s_cmp_gt_i32 s[s_kitr], 0
    
    s_cbranch_scc0 L_igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_end

L_igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_body:
    ; accumulate im

    ; a buffer x
    ;--- start move slice window
    s_add_u32 s[s_move_slice_k_ix], 1, s[s_move_slice_k_ix]
    s_cmp_le_u32 s[s_x], s[s_move_slice_k_ix]
    s_cselect_b32 s[s_tmp], s[s_dilation_w_x], s[s_dilation_w]
    s_cselect_b32 s[s_tmp+1], s[s_in_diff_hi], s[s_in_diff_wi]
    v_add_nc_u32 v[v_in_iwi+0], s[s_tmp], v[v_in_iwi+0]
    v_add_nc_u32 v[v_in_iwi+1], s[s_tmp], v[v_in_iwi+1]
    v_add_nc_u32 v[v_in_os+0], s[s_tmp+1], v[v_in_os+0]
    v_add_nc_u32 v[v_in_os+1], s[s_tmp+1], v[v_in_os+1]
    s_cbranch_scc0 igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_acc_yx_x_end_1
    s_mov_b32 s[s_move_slice_k_ix], 0
    v_add_nc_i32 v[v_in_ihi+0], s[s_dilation_h], v[v_in_ihi+0]
    v_add_nc_i32 v[v_in_ihi+1], s[s_dilation_h], v[v_in_ihi+1]
igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_acc_yx_x_end_1:
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+0]
    v_cndmask_b32 v[v_in_flag+0], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, 1
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+0]
    v_cndmask_b32 v[v_in_flag+0], 0, v[v_in_flag+0]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, v[v_in_flag+1]
    ;--- end move slice window

    ;s_waitcnt vmcnt(0)
    .v_clear_nc v_ay, 16
    v_cmpx_le_u32 1, v[v_in_flag+0]
    global_load_dwordx4 v[v_ay+0:v_ay+3], v[v_in_os], s[s_p_in:s_p_in+1]
    global_load_dwordx4 v[v_ay+4:v_ay+7], v[v_in_os], s[s_p_in:s_p_in+1] offset:16
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_in_flag+1]
    global_load_dwordx4 v[v_ay+ 8:v_ay+11], v[v_in_os+1], s[s_p_in:s_p_in+1]
    global_load_dwordx4 v[v_ay+12:v_ay+15], v[v_in_os+1], s[s_p_in:s_p_in+1] offset:16
    s_mov_b64 exec, -1

    s_waitcnt vmcnt(4) lgkmcnt(4)
    .fma_1x4_fp16 v_c+ 0, v_ax + 0, v_b + 0
    .fma_1x4_fp16 v_c+ 4, v_ax + 8, v_b + 0

    .fma_1x4_fp16 v_c+ 0, v_ax + 1, v_b + 4
    .fma_1x4_fp16 v_c+ 4, v_ax + 9, v_b + 4

    .fma_1x4_fp16 v_c+ 0, v_ax + 2, v_b + 8
    .fma_1x4_fp16 v_c+ 4, v_ax +10, v_b + 8

    .fma_1x4_fp16 v_c+ 0, v_ax + 3, v_b +12
    .fma_1x4_fp16 v_c+ 4, v_ax +11, v_b +12

    ds_read_b128 v[v_b+ 0:v_b+ 3], v[v_sld_b_os], offset:k_n_dword*4*0
    ds_read_b128 v[v_b+ 4:v_b+ 7], v[v_sld_b_os], offset:k_n_dword*4*1
    ds_read_b128 v[v_b+ 8:v_b+11], v[v_sld_b_os], offset:k_n_dword*4*2
    ds_read_b128 v[v_b+12:v_b+15], v[v_sld_b_os], offset:k_n_dword*4*3

    s_waitcnt lgkmcnt(4)
    .fma_1x4_fp16 v_c+ 0, v_ax + 4, v_b +16
    .fma_1x4_fp16 v_c+ 4, v_ax +12, v_b +16

    .fma_1x4_fp16 v_c+ 0, v_ax + 5, v_b +20
    .fma_1x4_fp16 v_c+ 4, v_ax +13, v_b +20

    .fma_1x4_fp16 v_c+ 0, v_ax + 6, v_b +24
    .fma_1x4_fp16 v_c+ 4, v_ax +14, v_b +24

    .fma_1x4_fp16 v_c+ 0, v_ax + 7, v_b +28
    .fma_1x4_fp16 v_c+ 4, v_ax +15, v_b +28

    ds_read_b128 v[v_b+16:v_b+19], v[v_sld_b_os], offset:k_n_dword*4*4
    ds_read_b128 v[v_b+20:v_b+23], v[v_sld_b_os], offset:k_n_dword*4*5
    ds_read_b128 v[v_b+24:v_b+27], v[v_sld_b_os], offset:k_n_dword*4*6
    ds_read_b128 v[v_b+28:v_b+31], v[v_sld_b_os], offset:k_n_dword*4*7

    s_sub_i32 s[s_kitr], s[s_kitr], 16
    v_add_nc_u32 v[v_sld_b_os], s[s_sld_b_stride], v[v_sld_b_os]            ; accumulate sld_b_os
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_end_1

    ; a buffer y
    ;--- start move slice window
    s_add_u32 s[s_move_slice_k_ix], 1, s[s_move_slice_k_ix]
    s_cmp_le_u32 s[s_x], s[s_move_slice_k_ix]
    s_cselect_b32 s[s_tmp], s[s_dilation_w_x], s[s_dilation_w]
    s_cselect_b32 s[s_tmp+1], s[s_in_diff_hi], s[s_in_diff_wi]
    v_add_nc_u32 v[v_in_iwi+0], s[s_tmp], v[v_in_iwi+0]
    v_add_nc_u32 v[v_in_iwi+1], s[s_tmp], v[v_in_iwi+1]
    v_add_nc_u32 v[v_in_os+0], s[s_tmp+1], v[v_in_os+0]
    v_add_nc_u32 v[v_in_os+1], s[s_tmp+1], v[v_in_os+1]
    s_cbranch_scc0 igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_acc_yx_x_end_2
    s_mov_b32 s[s_move_slice_k_ix], 0
    v_add_nc_i32 v[v_in_ihi+0], s[s_dilation_h], v[v_in_ihi+0]
    v_add_nc_i32 v[v_in_ihi+1], s[s_dilation_h], v[v_in_ihi+1]
igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_acc_yx_x_end_2:
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+0]
    v_cndmask_b32 v[v_in_flag+0], 0, 1
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, 1

    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+0]
    v_cndmask_b32 v[v_in_flag+0], 0, v[v_in_flag+0]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, v[v_in_flag+1]
    ;--- end move slice window

    ; s_waitcnt vmcnt(0)
    .v_clear_nc v_ax, 16
    v_cmpx_le_u32 1, v[v_in_flag+0]
    global_load_dwordx4 v[v_ax+0:v_ax+3], v[v_in_os], s[s_p_in:s_p_in+1]
    global_load_dwordx4 v[v_ax+4:v_ax+7], v[v_in_os], s[s_p_in:s_p_in+1] offset:16
    s_mov_b64 exec, -1
    v_cmpx_le_u32 1, v[v_in_flag+1]
    global_load_dwordx4 v[v_ax+ 8:v_ax+11], v[v_in_os+1], s[s_p_in:s_p_in+1]
    global_load_dwordx4 v[v_ax+12:v_ax+15], v[v_in_os+1], s[s_p_in:s_p_in+1] offset:16
    s_mov_b64 exec, -1

    s_waitcnt vmcnt(4) lgkmcnt(4)
    .fma_1x4_fp16 v_c+ 0, v_ay + 0, v_b + 0
    .fma_1x4_fp16 v_c+ 4, v_ay + 8, v_b + 0

    .fma_1x4_fp16 v_c+ 0, v_ay + 1, v_b + 4
    .fma_1x4_fp16 v_c+ 4, v_ay + 9, v_b + 4

    .fma_1x4_fp16 v_c+ 0, v_ay + 2, v_b + 8
    .fma_1x4_fp16 v_c+ 4, v_ay +10, v_b + 8

    .fma_1x4_fp16 v_c+ 0, v_ay + 3, v_b +12
    .fma_1x4_fp16 v_c+ 4, v_ay +11, v_b +12

    ds_read_b128 v[v_b+ 0:v_b+ 3], v[v_sld_b_os], offset:k_n_dword*4*0
    ds_read_b128 v[v_b+ 4:v_b+ 7], v[v_sld_b_os], offset:k_n_dword*4*1
    ds_read_b128 v[v_b+ 8:v_b+11], v[v_sld_b_os], offset:k_n_dword*4*2
    ds_read_b128 v[v_b+12:v_b+15], v[v_sld_b_os], offset:k_n_dword*4*3

    s_waitcnt lgkmcnt(4)
    .fma_1x4_fp16 v_c+ 0, v_ay + 4, v_b +16
    .fma_1x4_fp16 v_c+ 4, v_ay +12, v_b +16

    .fma_1x4_fp16 v_c+ 0, v_ay + 5, v_b +20
    .fma_1x4_fp16 v_c+ 4, v_ay +13, v_b +20

    .fma_1x4_fp16 v_c+ 0, v_ay + 6, v_b +24
    .fma_1x4_fp16 v_c+ 4, v_ay +14, v_b +24

    .fma_1x4_fp16 v_c+ 0, v_ay + 7, v_b +28
    .fma_1x4_fp16 v_c+ 4, v_ay +15, v_b +28

    ds_read_b128 v[v_b+16:v_b+19], v[v_sld_b_os], offset:k_n_dword*4*4
    ds_read_b128 v[v_b+20:v_b+23], v[v_sld_b_os], offset:k_n_dword*4*5
    ds_read_b128 v[v_b+24:v_b+27], v[v_sld_b_os], offset:k_n_dword*4*6
    ds_read_b128 v[v_b+28:v_b+31], v[v_sld_b_os], offset:k_n_dword*4*7

    s_sub_i32 s[s_kitr], s[s_kitr], 16
    v_add_nc_u32 v[v_sld_b_os], s[s_sld_b_stride], v[v_sld_b_os]            ; accumulate sld_b_os
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc1 L_igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_body

L_igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_end:
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

L_igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_end_1:
    s_waitcnt vmcnt(0)

    s_sub_i32 s[s_batch_m], s[s_batch_m], 1
    v_add_nc_u32 v[v_ib], s[s_stride_m],  v[v_ib]

    s_cmp_gt_i32 s[s_batch_m], 0
    s_cbranch_scc0 L_igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_end_not_load_next
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
    .v_clear_nc v_ax+8, 4
    v_sub_nc_i32 v[v_in_ihi+1], v[v_in_ihi+1], s[s_pad_h]
    v_mul_u32_u24 v[v_in_iwi+1], s[s_stride_w], v[v_in_iwi+1]
    .v_clear_nc v_ax+12, 4
    v_sub_nc_i32 v[v_in_iwi+1], v[v_in_iwi+1], s[s_pad_w]

    ; v_add_nc_u32 v[v_in_flag+2], s[s_ib_stride], v[v_in_flag+1]

    v_cmpx_le_u32 1, v[v_in_flag]
    global_load_dwordx4 v[v_ax+0:v_ax+3], v[v_in_os], s[s_p_in:s_p_in+1]
    global_load_dwordx4 v[v_ax+4:v_ax+7], v[v_in_os], s[s_p_in:s_p_in+1] offset:16
    s_mov_b64 exec, -1

    v_mul_u32_u24 v[v_in_os+1], s[s_wi], v[v_in_ihi+1]
    v_cmp_gt_u32 s[s_hi], v[v_in_ihi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, 1
    v_add_nc_u32 v[v_in_os+1], v[v_in_iwi+1], v[v_in_os+1]
    v_cmp_gt_u32 s[s_wi], v[v_in_iwi+1]
    v_cndmask_b32 v[v_in_flag+1], 0, v[v_in_flag+1]
    v_mul_lo_u32 v[v_in_os+1], s[s_in_stride_wi], v[v_in_os+1]

    v_cmpx_le_u32 1, v[v_in_flag+1]
    global_load_dwordx4 v[v_ax+ 8:v_ax+11], v[v_in_os+1], s[s_p_in:s_p_in+1]
    global_load_dwordx4 v[v_ax+12:v_ax+15], v[v_in_os+1], s[s_p_in:s_p_in+1] offset:16
    s_mov_b64 exec, -1

    s_mov_b32 s[s_move_slice_k_ix], 0

L_igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_end_not_load_next:
    ; --- end move slice for batch m

    s_waitcnt lgkmcnt(4)

    .fma_1x4_fp16 v_c+ 0, v_ay + 0, v_b + 0
    .fma_1x4_fp16 v_c+ 4, v_ay + 8, v_b + 0

    .fma_1x4_fp16 v_c+ 0, v_ay + 1, v_b + 4
    .fma_1x4_fp16 v_c+ 4, v_ay + 9, v_b + 4

    .fma_1x4_fp16 v_c+ 0, v_ay + 2, v_b + 8
    .fma_1x4_fp16 v_c+ 4, v_ay +10, v_b + 8

    .fma_1x4_fp16 v_c+ 0, v_ay + 3, v_b +12
    .fma_1x4_fp16 v_c+ 4, v_ay +11, v_b +12

    s_waitcnt lgkmcnt(0)
    .fma_1x4_fp16 v_c+ 0, v_ay + 4, v_b +16
    .fma_1x4_fp16 v_c+ 4, v_ay +12, v_b +16

    .fma_1x4_fp16 v_c+ 0, v_ay + 5, v_b +20
    .fma_1x4_fp16 v_c+ 4, v_ay +13, v_b +20

    .fma_1x4_fp16 v_c+ 0, v_ay + 6, v_b +24
    .fma_1x4_fp16 v_c+ 4, v_ay +14, v_b +24

    .fma_1x4_fp16 v_c+ 0, v_ay + 7, v_b +28
    .fma_1x4_fp16 v_c+ 4, v_ay +15, v_b +28


    v_mov_b32 v[v_sld_b_os], 0                                  ; reset to start
    .activ_f32 v_c + 0, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_f32 v_c + 1, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_f32 v_c + 2, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_f32 v_c + 3, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_f32 v_c + 4, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_f32 v_c + 5, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_f32 v_c + 6, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    .activ_f32 v_c + 7, activ_mode, s_alpha, s_beta, s_gamma, v_tmp+0, v_tmp+1
    v_cvt_f16_f32 v[v_c + 0], v[v_c + 0]
    v_cvt_f16_f32 v[v_c + 1], v[v_c + 1]
    v_cvt_f16_f32 v[v_c + 2], v[v_c + 2]
    v_cvt_f16_f32 v[v_c + 3], v[v_c + 3]
    v_cvt_f16_f32 v[v_c + 4], v[v_c + 4]
    v_cvt_f16_f32 v[v_c + 5], v[v_c + 5]
    v_cvt_f16_f32 v[v_c + 6], v[v_c + 6]
    v_cvt_f16_f32 v[v_c + 7], v[v_c + 7]

    v_pack_b32_f16 v[v_c_buf+0], v[v_c+ 0], v[v_c+ 1]
    v_pack_b32_f16 v[v_c_buf+1], v[v_c+ 2], v[v_c+ 3]
    v_pack_b32_f16 v[v_c_buf+2], v[v_c+ 4], v[v_c+ 5]
    v_pack_b32_f16 v[v_c_buf+3], v[v_c+ 6], v[v_c+ 7]


    v_cmpx_le_u32 1, v[v_out_flag]
    global_store_dwordx2 v[v_out_os], v[v_c_buf+0:v_c_buf+1], s[s_p_out:s_p_out+1]
    s_mov_b64 exec, -1

    v_cmpx_le_u32 1, v[v_out_flag+1]
    global_store_dwordx2 v[v_out_os+1], v[v_c_buf+2:v_c_buf+3], s[s_p_out:s_p_out+1]
    s_mov_b64 exec, -1

    s_cmp_le_i32 s[s_batch_m], 0

    s_cbranch_scc1 L_igemm_fwd_btm_nhwc_fp16_256x4x16_r1_end
    ds_read_b128 v[v_b+ 0:v_b+ 3], v[v_sld_b_os], offset:k_n_dword*4*0
    ds_read_b128 v[v_b+ 4:v_b+ 7], v[v_sld_b_os], offset:k_n_dword*4*1
    ds_read_b128 v[v_b+ 8:v_b+11], v[v_sld_b_os], offset:k_n_dword*4*2
    ds_read_b128 v[v_b+12:v_b+15], v[v_sld_b_os], offset:k_n_dword*4*3
    ds_read_b128 v[v_b+16:v_b+19], v[v_sld_b_os], offset:k_n_dword*4*4
    ds_read_b128 v[v_b+20:v_b+23], v[v_sld_b_os], offset:k_n_dword*4*5
    ds_read_b128 v[v_b+24:v_b+27], v[v_sld_b_os], offset:k_n_dword*4*6
    ds_read_b128 v[v_b+28:v_b+31], v[v_sld_b_os], offset:k_n_dword*4*7

    .v_clear_nc v_c, 8
    v_add_nc_u32 v[v_sld_b_os], s[s_sld_b_stride], v[v_sld_b_os]            ; accumulate sld_b_os

    v_add_nc_u32 v[v_out_os], s[s_out_stride], v[v_out_os]
    s_sub_i32 s[s_kitr], s[s_wei_stride_k], 16
    v_add_nc_u32 v[v_out_os+1], s[s_out_stride], v[v_out_os+1]
    
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os]
    v_cndmask_b32 v[v_out_flag], 0, 1
    s_cmp_gt_i32 s[s_kitr], 0
    v_cmp_gt_u32 s[s_out_stride_n], v[v_out_os+1]
    v_cndmask_b32 v[v_out_flag+1], 0, 1

    s_cbranch_scc0 L_igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_end
    s_branch L_igemm_fwd_btm_nhwc_fp16_256x4x16_r1_fma_body
L_igemm_fwd_btm_nhwc_fp16_256x4x16_r1_end:
    s_endpgm

; LDS: 1 * 4    * 4  * 128
;      r1  4dword 4    threads
.rodata
.p2align 6
.amdhsa_kernel igemm_fwd_btm_nhwc_fp16_256x4x16_r1
    .amdhsa_group_segment_fixed_size 2048
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 88
    .amdhsa_next_free_sgpr 62
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
    .amdhsa_wavefront_size32 1
    .amdhsa_workgroup_processor_mode 0
.end_amdhsa_kernel
