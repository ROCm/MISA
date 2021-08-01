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
.set s_in_stride_wi, 22
.set s_in_stride_n, 23
.set s_wei_stride_k0, 24
.set s_wei_stride_k, 25
.set s_out_stride_wo, 26
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
.set s_in_offset, 37
.set s_wei_offset, 38
.set s_magic_0, 6
.set s_magic_1, 7
.set s_magic_2, 14
.set s_magic_3, 15
.set s_shift_pack_0, 40
.set s_tmp, 42
.set s_end, 48

.set v_c, 0  ; coalescing:32, needed:0, resuable:60
.set v_a, 0
.set v_b, 8
.set v_gld_a, 24
.set v_gld_b, 32
.set v_sst_a_os, 48
.set v_sld_a_os, 49
.set v_sst_b_os, 50
.set v_sld_b_os, 51
.set v_in_os, 52
.set v_in_ihi_list, 54
.set v_in_iwi_list, 56
.set v_in_flag, 58
.set v_in_flag_n, 60
.set v_wei_os, 61
.set v_out_os, 62
.set v_gtc_ic, 63
.set v_in_inb, 64
.set v_in_in, 65
.set v_wei_ik, 66
.set v_co_sst, 65
.set v_co_sld, 67
.set v_out_flag, 66
.set v_out_inb, 64
.set v_gemm_in, 68
.set v_gemm_im, 69
.set v_co_sub_m_index, 69
.set v_co_sub_n_index, 68
.set v_tmp, 70
.set v_wei_tmp_pack, 23
.set v_wei_flag, 70
.set v_end, 128

.set a_c, 0
.set a_end, 128

.text
.globl igemm_fwd_gtcx_cnhwc_fp16_ex0_bt128x256x32_wt32x32x8_ws1x2_wr2x2
.p2align 8
.type igemm_fwd_gtcx_cnhwc_fp16_ex0_bt128x256x32_wt32x32x8_ws1x2_wr2x2,@function
igemm_fwd_gtcx_cnhwc_fp16_ex0_bt128x256x32_wt32x32x8_ws1x2_wr2x2:
    s_load_dwordx2  s[s_p_in+0:s_p_in+1],    s[s_ka+0:s_ka+1],    0+k_p_in
    s_load_dwordx2  s[s_p_wei+0:s_p_wei+1],   s[s_ka+0:s_ka+1],    0+k_p_wei
    s_load_dwordx2  s[s_p_out+0:s_p_out+1],   s[s_ka+0:s_ka+1],    0+k_p_out
    s_load_dwordx4 s[s_hi+0:s_hi+3],    s[s_ka+0:s_ka+1],    0+k_hi
    s_load_dword s[s_c],    s[s_ka+0:s_ka+1],    0+k_c
    s_load_dword s[s_group],    s[s_ka+0:s_ka+1],    0+k_group
    s_load_dwordx2 s[s_magic_0+0:s_magic_0+1],  s[s_ka+0:s_ka+1],  0+k_magic_0
    s_load_dwordx2 s[s_magic_2+0:s_magic_2+1],  s[s_ka+0:s_ka+1],  0+k_magic_2
    s_load_dword s[s_shift_pack_0], s[s_ka+0:s_ka+1],  0+k_shift_pack_0
    ; in(e, c, nb0, nb1) thread_lengths: 1x8x2x1, cluster_length: 1x4x1x64, k_pack:8
    v_mov_b32 v[v_tmp], v0
    v_and_b32 v[v_gtc_ic], 3, v[v_tmp]
    v_lshlrev_b32 v[v_gtc_ic], 3, v[v_gtc_ic]
    v_lshrrev_b32 v[v_tmp], 2, v[v_tmp]
    v_and_b32 v[v_in_inb], 63, v[v_tmp]
    ; wei(e, c, k0, k1) thread_length: 1x8x4x1, cluster_length: 1x4x1x64, k_pack:8
    v_lshrrev_b32 v[v_tmp], 2, v0
    v_and_b32 v[v_wei_ik], 63, v[v_tmp]

    s_waitcnt lgkmcnt(0)


    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel igemm_fwd_gtcx_cnhwc_fp16_ex0_bt128x256x32_wt32x32x8_ws1x2_wr2x2
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