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
.set k_magic_0, 88
.set k_magic_1, 92
.set k_magic_2, 96
.set k_magic_3, 100
.set k_magic_4, 104
.set k_magic_5, 108
.set k_shift_pack_0, 112
.set k_shift_pack_1, 116
.set k_gemm_k_global_split, 120
.set k__pack_0, 124
.set k_end, 128
.set k_gload_in_c_stride, 16

.set s_ka, 0
.set s_bx, 2
.set s_by, 3
.set s_p_in, 4
.set s_p_wei, 8
.set s_p_out, 12
.set s_hi, 16
.set s_wi, 17
.set s_n, 18
.set s_k, 19
.set s_c, 20
.set s_group, 21
.set s_in_stride_c, 22
.set s_in_stride_n, 23
.set s_wei_stride_c, 24
.set s_wei_stride_k, 25
.set s_out_stride_k, 26
.set s_out_stride_n, 27
.set s_block_gtc_ig, 28
.set s_block_gtc_ik, 29
.set s_block_gtc_inb, 30
.set s_move_slice_k_stride_c, 31
.set s_knum, 3
.set s_dim_br, 32
.set s_dim_mp, 33
.set s_dim_mr, 34
.set s_dim_np, 35
.set s_gemm_k_num_c, 35
.set s_in_diff_hi, 29
.set s_in_diff_wi, 28
.set s_dilation_w_x, 36
.set s_move_slice_k_ix, 32
.set s_flag_need_acc_yx, 33
.set s_kitr, 1
.set s_in_step, 37
.set s_wei_step, 38
.set s_magic_0, 6
.set s_magic_1, 7
.set s_magic_2, 14
.set s_magic_3, 15
.set s_shift_pack_0, 40
.set s_tmp, 42
.set s_end, 48

.set v_c, 0  ; coalescing:32, needed:0, resuable:66
.set v_a, 0
.set v_b, 16
.set v_gld_a, 24
.set v_gld_b, 40
.set v_sst_a_os, 48
.set v_sld_a_os, 49
.set v_sst_b_os, 50
.set v_sld_b_os, 51
.set v_in_os, 52
.set v_in_ihi_list, 56
.set v_in_iwi_list, 60
.set v_in_flag, 64
.set v_in_flag_n, 68
.set v_wei_os, 69
.set v_out_os, 70
.set v_gtc_ic, 71
.set v_in_inb, 72
.set v_in_in, 73
.set v_wei_ik, 74
.set v_co_sst, 73
.set v_co_sld, 75
.set v_out_flag, 74
.set v_out_inb, 72
.set v_gemm_in, 76
.set v_gemm_im, 77
.set v_co_sub_m_index, 77
.set v_co_sub_n_index, 76
.set v_tmp, 78
.set v_wei_tmp_pack, 23
.set v_wei_flag, 78
.set v_end, 128

.set a_c, 0
.set a_end, 128

.text
.globl igemm_fwd_gtcx_cnhwc_fp16_ex0_bt256x128x32_wt32x32x8_ws2x1_wr2x2
.p2align 8
.type igemm_fwd_gtcx_cnhwc_fp16_ex0_bt256x128x32_wt32x32x8_ws2x1_wr2x2,@function
;         c   nhw-vecc 
; t-len   1     4*8
; c-len   4     64
;         c   k-vecc
; t-len   1     2*8
; c-len   4     64
; vec_c == vec_k == 8
igemm_fwd_gtcx_cnhwc_fp16_ex0_bt256x128x32_wt32x32x8_ws2x1_wr2x2:
    s_load_dwordx2  s[s_p_in+0:s_p_in+1],    s[s_ka+0:s_ka+1],    0+k_p_in
    s_load_dwordx2  s[s_p_wei+0:s_p_wei+1],   s[s_ka+0:s_ka+1],    0+k_p_wei
    s_load_dwordx2  s[s_p_out+0:s_p_out+1],   s[s_ka+0:s_ka+1],    0+k_p_out
    s_load_dwordx4 s[s_hi+0:s_hi+3],    s[s_ka+0:s_ka+1],    0+k_hi
    s_load_dword s[s_c],    s[s_ka+0:s_ka+1],    0+k_c
    s_load_dword s[s_group],    s[s_ka+0:s_ka+1],    0+k_group
    s_load_dwordx2 s[s_magic_0+0:s_magic_0+1],  s[s_ka+0:s_ka+1],  0+k_magic_0
    s_load_dwordx2 s[s_magic_2+0:s_magic_2+1],  s[s_ka+0:s_ka+1],  0+k_magic_2
    s_load_dword s[s_shift_pack_0], s[s_ka+0:s_ka+1],  0+k_shift_pack_0
    ; in(c, nhw-vecc) thread_lengths: (1, 4-8), cluster_length: (4, 64), vec_c = kpack = 8
    v_lshrrev_b32 v[v_gtc_ic], 6, v0 ; v_gtc_ic = tid / 64
    v_and_b32 v[v_in_inb], 63, v0 ; v_in_inb = tid % 64
    v_lshlrev_b32 v[v_in_inb], 3, v[v_in_inb] ; v_in_inb = v_in_inb * 4 * 8
    ; wei(c, k-vecc) thread_length: (1, 2-8), cluster_length: (4, 64)
    v_and_b32 v[v_wei_ik], 63, v0
    v_lshlrev_b32 v[v_wei_ik], 3, v[v_wei_ik] ; v_wei_ik = v_wei_ik * 2 * 8

    s_mov_b32 s[s_p_in+2], 0xffffffff
    s_mov_b32 s[s_p_in+3], 0x27000
    s_mov_b32 s[s_p_wei+2], 0xffffffff
    s_mov_b32 s[s_p_wei+3], 0x27000
    s_mov_b32 s[s_p_out+2], 0xffffffff
    s_mov_b32 s[s_p_out+3], 0x27000

    s_waitcnt lgkmcnt(0)

    s_mov_b32 s[s_knum], s[s_c]

    ; calculate stride
    s_mul_i32 s[s_in_stride_c], s[s_wi], s[s_hi] ; s_in_stride_c = s_wi * s_hi
    s_mul_i32 s[s_in_stride_c], s[s_in_stride_c], s[s_n] ; s_in_stride_c = s_in_stride_c * s_n
    s_lshl_b32 s[s_in_stride_c], s[s_in_stride_c], 3 ; s_in_stride_c = s_in_stride_c * vecc ; vecc is 8 now 
    
    s_lshl_b32 s[s_wei_stride_c], s[s_k], 3 ; s_wei_stride_c = s_k * vecc ; vecc is 8 now

    s_mov_b32 s[s_out_stride_k], s[s_in_stride_c] ; for 1x1 stride 1 cases, s_out_stride_k = s_in_stride_c

    ; calculate block_im, block_in. gemm m per block: 256, gemm n per block:128
    s_lshl_b32 s[s_block_gtc_ik], s[s_by], 7 ; block_ik = blockId.y * 128
    s_lshl_b32 s[s_block_gtc_inb], s[s_bx], 8 ; block_inb = blockId.x * 256

    ; calculate weight offset
    s_lshl_b32 s[s_tmp], s[s_block_gtc_ik], 3 ; block_inb = blockId.y * 128 * 8
    v_add_u32 v[v_wei_ik], v[v_wei_ik], s[s_tmp]
    v_mul_lo_u32 v[v_gtc_ic], v[v_gtc_ic], s[s_wei_stride_c]
    v_add_lshl_u32 v[v_wei_os], v[v_wei_ik], v[v_gtc_ic], 1

    ; global load weight
    buffer_load_dwordx4 v[v_gld_b:v_gld_b+3], v[v_wei_os], s[s_p_wei:s_p_wei+3], 0 offen offset:0
    buffer_load_dwordx4 v[v_gld_b+4:v_gld_b+4+3], v[v_wei_os], s[s_p_wei:s_p_wei+3], 0 offen offset:2048

    ; calculate input offset
    s_lshl_b32 s[s_tmp], s[s_block_gtc_inb], 3 ; block_inb = blockId.x * 256 * 8
    v_add_u32 v[v_tmp], v[v_in_inb], s[s_tmp]
    v_mul_lo_u32 v[v_tmp+1], v[v_gtc_ic], s[s_in_stride_c]
    v_add_lshl_u32 v[v_in_os], v[v_tmp], v[v_tmp+1], 1

    ; global load input 
    buffer_load_dwordx4 v[v_gld_a:v_gld_a+3], v[v_in_os], s[s_p_in:s_p_in+3], 0 offen offset:0
    buffer_load_dwordx4 v[v_gld_a+4:v_gld_a+4+3], v[v_in_os], s[s_p_in:s_p_in+3], 0 offen offset:2048
    buffer_load_dwordx4 v[v_gld_a+8:v_gld_a+8+3], v[v_in_os], s[s_p_in:s_p_in+3], 0 offen offset:4096
    buffer_load_dwordx4 v[v_gld_a+12:v_gld_a+12+3], v[v_in_os], s[s_p_in:s_p_in+3], 0 offen offset:6144

    ; shared mem store offset: input
    v_lshl_add_u32 v[v_tmp], v[v_gtc_ic], 11, v[v_in_inb]
    v_lshlrev_b32 v[v_sst_a_os], 1, v[v_tmp]

    ; shared mem store offset: weight
    v_lshl_add_u32 v[v_tmp], v[v_gtc_ic], 10, v[v_wei_ik]
    v_lshlrev_b32 v[v_sst_b_os], 1, v[v_tmp]
    v_add_u32 v[v_sst_b_os], 16384, v[v_sst_b_os]

    ; shared mem load offset: input and weight
    v_and_b32 v[v_gemm_in], 31, v0           ; v_gemm_in = tid % 32 
    v_and_b32 v[v_gemm_im], 31, v0           ; v_gemm_im = tid % 32 
    v_lshlrev_b32 v[v_gemm_in], 3, v[v_gemm_in]   ; shift left k_pack:8
    v_lshlrev_b32 v[v_gemm_im], 3, v[v_gemm_im]   ; shift left k_pack:8
    v_lshrrev_b32 v[v_tmp+5], 5, v0
    v_and_b32 v[v_tmp+0], 1, v[v_tmp+5]         ; block_k_per_wave index v_tmp0 = (tid / 32) % 1
    v_lshl_or_b32 v[v_gemm_in], v[v_tmp+0], 10, v[v_gemm_in]
    v_lshl_or_b32 v[v_gemm_im], v[v_tmp+0], 11, v[v_gemm_im]
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5]
    v_and_b32 v[v_tmp + 2], 1, v[v_tmp+5]  ; waves_per_n index
    v_lshl_or_b32 v[v_gemm_in], v[v_tmp + 2], 8, v[v_gemm_in]
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5]
    v_and_b32 v[v_tmp + 3], 1, v[v_tmp+5]  ; waves_per_m index
    v_lshl_or_b32 v[v_gemm_im], v[v_tmp + 3], 9, v[v_gemm_im]

    v_lshlrev_b32 v[v_sld_b_os], 1, v[v_gemm_in]
    v_lshlrev_b32 v[v_sld_a_os], 1, v[v_gemm_im]
    v_add_u32 v[v_sld_b_os], 16384, v[v_sld_b_os]

    v_mov_b32 v[v_tmp+5], v0

    ;-----------------TODO Part-------------------------------------------
    ; xdlops mapping, get dst matrix gemm index
    v_and_b32 v[v_tmp+0], 31, v[v_tmp+5] ; v_tmp0 = tid % 32
    v_lshrrev_b32 v[v_tmp+5], 5, v[v_tmp+5] ; v_tmp5 = tid / 32
    v_and_b32 v[v_tmp+1], 1, v[v_tmp+5] ; v_tmp1 = tid / 32 % 2
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5] ; v_tmp5 = tid / 32 / 2
    v_mov_b32 v[v_co_sst], v[v_tmp+0] ; v_co_sst = v_tmp0 = tid % 32
    v_lshlrev_b32 v[v_co_sld], 2, v[v_tmp+1] ; v_co_sld = v_tmp1 * 4 = tid / 32 % 2 * 4
    v_and_b32 v[v_tmp+0], 1, v[v_tmp+5] ; v_tmp0 = tid / 32 / 2 % 2
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5] ; v_tmp5 = tid / 32 / 2 / 2
    v_and_b32 v[v_tmp+1], 1, v[v_tmp+5] ; v_tmp1 = tid / 32 / 2 / 2 % 2
    v_lshl_or_b32 v[v_co_sst], v[v_tmp+0], 5, v[v_co_sst] ; v_co_sst = tid % 32 + tid / 32 / 2 % 2 * 32
    v_lshl_or_b32 v[v_co_sld], v[v_tmp+1], 6, v[v_co_sld] ; v_co_sld = v_co_sld + tid / 32 / 2 / 2 % 2 * 64

    v_mov_b32 v[v_gemm_in], v[v_co_sst]
    v_mov_b32 v[v_gemm_im], v[v_co_sld]
    ; init_co_lds_offset for xdlops
    v_lshrrev_b32 v[v_tmp], 2, v[v_gemm_im]
    v_and_b32 v[v_tmp],  1 v[v_tmp]   ; thread id of lanegroup_m_per_cluster
    v_lshlrev_b32 v[v_co_sst], 2, v[v_tmp]
    v_lshrrev_b32 v[v_tmp+2], 6, v[v_gemm_im]  ; thread id of waves_per_m
    v_lshl_or_b32 v[v_co_sst], v[v_tmp+2], 6, v[v_co_sst]
    v_lshl_or_b32 v[v_co_sst], v[v_co_sst], 7, v[v_gemm_in]
    v_lshlrev_b32 v[v_co_sst], 1, v[v_co_sst]
    v_lshlrev_b32 v[v_co_sld], 2, v[0]
    ; init_co_sub_m_index xdlops, block_size:256, macro-tile:256x128 sub_m_index:[0, 1, 2, 3]
    ; g_mr:2, g_ms:1, g_mw:1, g_mb:1, g_mt:1 | l_mr:1, l_ms:2, l_mw:1, l_mb:4, l_mt:4 | n_mc:2, n_ml:1, n_mv:2
    ; nd_stride:[4, 2, 1, 4, 1, 2, 2, 1]
    v_lshlrev_b32 v[v_tmp], 1, v[0]
    v_lshrrev_b32 v[v_co_sub_m_index], 7, v[v_tmp]  ; get tid along m
    v_and_b32 v[v_tmp+0], 3, v[v_co_sub_m_index]                   ; => x_mt
    v_mov_b32 v[v_co_sub_m_index], v[v_tmp+0]      ; => accumulate x_mt
    ; init_co_sub_n_index xdlops
    v_lshlrev_b32 v[v_tmp], 3, v[0] ; vector write out = vecc = 8
    v_and_b32 v[v_co_sub_n_index], 127, v[v_tmp]
    ;----------------------------------------TODO Part END--------------------------------

    ;----------------------------------------TODO Part START--------------------------------
    ; output pointer offset 
    s_lshr_b32 s[s_block_gtc_ik], s[s_block_gtc_ik], 3
    s_lshl_b32 s[s_tmp+3], s[s_block_gtc_ik], 1
    s_mul_i32 s[s_tmp], s[s_out_stride_k], s[s_tmp+3]
    s_mul_hi_u32 s[s_tmp+1], s[s_out_stride_k], s[s_tmp+3]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], s[s_tmp+1]
    ; output offset
    s_lshl_b32 s[s_out_stride_wo], s[s_out_stride_wo], 1
    v_add_u32 v[v_out_inb], s[s_block_gtc_inb], v[v_co_sub_m_index]   ; total n*ho*wo
    v_mul_lo_u32 v[v_out_os], s[s_out_stride_wo], v[v_out_inb]
    v_lshlrev_b32 v[v_tmp], 1, v[v_co_sub_n_index]
    v_add_u32 v[v_out_os], v[v_out_os], v[v_tmp]
    ;----------------------------------------TODO Part END--------------------------------

    ; move slice step
    s_lshl_b32 s[s_in_step], s[s_in_stride_c], 5 ; s_in_step = s_in_stride_c * (32/8) * 2
    s_lshl_b32 s[s_wei_step], s[s_wei_stride_c], 5 ; s_in_step = s_in_stride_c * (32/8) * 2

    ; start MFMA loop, 32x32 wave tile with 2x2 repeat, 2x1 step, k_pack:8
    s_waitcnt vmcnt(4)
    ds_write_b128 v[v_sst_b_os], v[v_gld_b+0:v_gld_b+0+3] 
    ds_write_b128 v[v_sst_b_os], v[v_gld_b+4:v_gld_b+4+3] offset:2048

    s_waitcnt vmcnt(0)
    ds_write_b128 v[v_sst_a_os], v[v_gld_a+0:v_gld_a+0+3] 
    ds_write_b128 v[v_sst_a_os], v[v_gld_a+4:v_gld_a+4+3] offset:2048
    ds_write_b128 v[v_sst_a_os], v[v_gld_a+8:v_gld_a+8+3] offset:4096
    ds_write_b128 v[v_sst_a_os], v[v_gld_a+12:v_gld_a+12+3] offset:6144

    .v_clear_acc_c a_c, 128

    ; make sure acc WAR harzard, at least 1 nop for src_c
    s_sub_i32 s[s_kitr], s[s_knum], 4 ; 32 / 8
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_fwd_gtcx_cnhwc_fp16_ex0_bt256x128x32_wt32x32x8_ws2x1_wr2x2_mfma_end

    v_add_u32 v[v_wei_os], s[s_wei_step], v[v_wei_os]
    v_add_u32 v[v_in_os], s[s_in_step], v[v_in_os]

    s_waitcnt lgkmcnt(0)
    s_barrier

L_igemm_fwd_gtcx_cnhwc_fp16_ex0_bt256x128x32_wt32x32x8_ws2x1_wr2x2_mfma_body:
    ds_read_b128 v[v_a:v_a+3], v[v_sld_a_os] ; i_k 0 repeat 0 step 0
    ds_read_b128 v[v_b:v_b+3], v[v_sld_b_os] ; i_k 0 repeat 0
    ds_read_b128 v[v_b+4:v_b+4+3], v[v_sld_b_os] offset: 1024 ; i_k 0 repeat 1 
    ds_read_b128 v[v_a+4:v_a+4+3], v[v_sld_a_os] offset: 256 ; i_k 0 repeat 0 step 1

    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x8f16 a[a_c+0:a_c+15], v[v_a+0:v_a+1], v[v_b+0:v_b+1], a[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16
    buffer_load_dwordx4 v[v_gld_b:v_gld_b+3], v[v_wei_os], s[s_p_wei:s_p_wei+3], 0 offen offset:0
    v_mfma_f32_32x32x8f16 a[a_c+0:a_c+15], v[v_a+2:v_a+3], v[v_b+2:v_b+3], a[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16

    ds_read_b128 v[v_a+8:v_a+8+3], v[v_sst_a_os] offset: 2048 ; i_k 0 repeat 1 step 0
    ds_read_b128 v[v_a+12:v_a+12+3], v[v_sst_a_os] offset: 2304 ; i_k 0 repeat 1 step 1

    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x8f16 a[a_c+16:a_c+31], v[v_a+4:v_a+4+1], v[v_b+0:v_b+1], a[a_c+16:a_c+31]     ; repeat:0x0, step:1x0, num_a_c:16
    buffer_load_dwordx4 v[v_gld_b:v_gld_b+3], v[v_wei_os], s[s_p_wei:s_p_wei+3], 0 offen offset:2048
    v_mfma_f32_32x32x8f16 a[a_c+16:a_c+31], v[v_a+4+2:v_a+4+3], v[v_b+2:v_b+3], a[a_c+16:a_c+31]     ; repeat:0x0, step:1x0, num_a_c:16

    ds_read_b128 v[v_b+4:v_b+4+3], v[v_sld_b_os] offset: 2048 ; i_k 1 repeat 0
    v_mfma_f32_32x32x8f16 a[a_c+32:a_c+47], v[v_a+0:v_a+1], v[v_b+4:v_b+4+1], a[a_c+32:a_c+47]     ; repeat:0x1, step:0x0, num_a_c:16





L_igemm_fwd_gtcx_cnhwc_fp16_ex0_bt256x128x32_wt32x32x8_ws2x1_wr2x2_mfma_end:



    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel igemm_fwd_gtcx_cnhwc_fp16_ex0_bt256x128x32_wt32x32x8_ws2x1_wr2x2
    .amdhsa_group_segment_fixed_size 32768
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 128
    .amdhsa_next_free_sgpr 48
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel