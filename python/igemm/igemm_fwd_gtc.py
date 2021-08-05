################################################################################
# 
#  MIT License
# 
#  Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
# 
################################################################################
# pylint: disable=maybe-no-member
from ..codegen import *
from ..operations import *
from .igemm_base import *


IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1 = 0
IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0 = 1
IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B = 4
IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N1B_N0 = 5
IGEMM_FWD_GTC_GLOBAL_LOAD_TA_ORDER_K_M= 0
IGEMM_FWD_GTC_GLOBAL_LOAD_TA_ORDER_M_K= 1

def _find_non_1_index_in_list(list_object):
    result_list = list()
    for idx, item in enumerate(list_object):
        assert type(item) is int
        if item != 1:
            result_list.append(idx)
    return result_list

class macro_igemm_fwd_gtc_set_flag_hw(macro_base_t):
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_flag")
        self.declare_arg("v_ih")
        self.declare_arg("v_iw")
        self.declare_arg("s_h")
        self.declare_arg("s_w")
    def name(self):
        return '.v_set_flag_hw'

    def expr(self):
        self._emit(f"v_cmp_gt_u32 vcc, s[{self.s_h()}], v[{self.v_ih()}]")
        self._emit(f"v_cndmask_b32 v[{self.v_flag()}], 0, 1, vcc")
        self._emit(f"v_cmp_gt_u32 vcc, s[{self.s_w()}], v[{self.v_iw()}]")
        self._emit(f"v_cndmask_b32 v[{self.v_flag()}], 0, v[{self.v_flag()}], vcc")

class macro_igemm_fwd_gtc_set_flag_c(macro_base_t):
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_flag")
        self.declare_arg("v_move_slice_k_ic1")
        self.declare_arg("s_c")

    def name(self):
        return '.v_set_flag_hw'

    def expr(self):
        self._emit(f"v_cmp_gt_u32 vcc, s[{self.s_c()}], v[{self.v_move_slice_k_ic1()}]")
        self._emit(f"v_cndmask_b32 v[{self.v_flag()}], 0, v[{self.v_flag()}], vcc")


class macro_igemm_fwd_gtc_in_update_hw_t(macro_base_t):
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_in_ihi")
        self.declare_arg("v_in_iwi")
        self.declare_arg("v_in_iho")
        self.declare_arg("v_in_iwo")
        self.declare_arg("v_in_iy")
        self.declare_arg("v_in_ix")
        self.declare_arg("s_dilation_h")
        self.declare_arg("s_dilation_w")
    def name(self):
        return '.v_fwd_gtc_in_update_hw'

    def expr(self):
        self._emit(f"; ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h,   here make sure iho <- iho * s_stride_h - s_pad_h before hand")
        self._emit(f"; iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w,   here make sure iwo <- iwo * s_stride_w - s_pad_w before hand")
        self._emit(f"v_mad_i32_i24 v[{self.v_in_ihi()}], s[{self.s_dilation_h()}], v[{self.v_in_iy()}], v[{self.v_in_iho()}]")
        self._emit(f"v_mad_i32_i24 v[{self.v_in_iwi()}], s[{self.s_dilation_w()}], v[{self.v_in_ix()}], v[{self.v_in_iwo()}]")


class macro_igemm_fwd_gtc_in_update_os_t(macro_base_t):
    def __init__(self, mc, data_byte, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.data_byte = data_byte
        self.declare_arg("v_in_os")
        self.declare_arg("v_in_os_base")
        self.declare_arg("v_in_ihi")
        self.declare_arg("v_in_iwi")
        self.declare_arg("s_wi")
        self.declare_arg("v_tmp")
    def name(self):
        return '.v_fwd_gtc_in_update_os'

    def expr(self):
        self._emit(f"v_mad_u32_u24 v[{self.v_tmp()}], v[{self.v_in_ihi()}], s[{self.s_wi()}], v[{self.v_in_iwi()}]")
        self._emit(f"v_lshl_add_u32 v[{self.v_in_os()}], v[{self.v_tmp()}], {igemm_log2(self.data_byte)}, v[{self.v_in_os_base()}]")

class macro_igemm_fwd_gtc_move_slice_window_ta_t(macro_base_t):
    '''
    weight no need a specific function to update its offset. it is updated here.
    '''
    def __init__(self, mc, data_byte, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.data_byte = data_byte
        self.declare_arg("v_wei_os")
        self.declare_arg("s_move_slice_k_c1e")
    def name(self):
        return '.v_fwd_gtc_move_slice_window_ta'

    def expr(self):
        self._emit(f"; move slice window for weight")
        self._emit(f"v_add_u32 v[{self.v_wei_os()}],  s[{self.s_move_slice_k_c1e()}], v[{self.v_wei_os()}]")

class macro_igemm_fwd_gtc_move_slice_window_k_y_x_tb_t(macro_base_t):
    def __init__(self, mc, tunable, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.tunable = tunable
        self.declare_arg("v_move_slice_k_ic1")
        self.declare_arg("v_move_slice_k_iy")
        self.declare_arg("v_move_slice_k_ix")
        self.declare_arg("s_gemm_k_num_c1")
        self.declare_arg("s_gemm_k_num_y")
        self.declare_arg("s_gemm_k_num_x")
        self.declare_arg("s_move_slice_k_c1")
        self.declare_arg("s_move_slice_k_y")
        self.declare_arg("s_move_slice_k_x")
        self.declare_arg("v_in_os_base")
        self.declare_arg("s_in_stride_c")
        self.declare_arg("s_in_stride_c_c1")
        self.declare_arg("s_in_stride_c_c0_c1_diff")
    def name(self):
        return '.v_fwd_gtc_move_slice_window_k_y_x_tb'

    def init_stride_c(self, s_in_stride_c, s_in_stride_c_c1, s_in_stride_c_c0_c1_diff, s_move_slice_k_c1):
        '''
        s_in_stride_c, s_move_slice_k_c1 is known value, want to compute other
        '''
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4

        ca_c0, ca_c1e, ca_k0, ca_k1  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        cb_c0, cb_c1e, cb_n0, cb_n1b = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(t_ta) == 4 and len(t_tb) == 4

        ta_c0, ta_c1e, ta_k0, ta_k1   = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        tb_c0, tb_c1e, tb_n0, tb_n1b  = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        nb_c0, nb_c1e = cb_c0 * tb_c0, cb_c1e * tb_c1e
        unmerge_sub_c = self.tunable.unmerge_sub_c
        assert unmerge_sub_c % nb_c0 == 0
        unmerge_sub_tb_c1 = unmerge_sub_c // nb_c0
        assert nb_c1e % unmerge_sub_tb_c1 == 0

        diff_c0_c1 = self.tunable.gemm_k_per_block - unmerge_sub_tb_c1 # !!! the diff of 2 unmerged dimension (like K=K0*K1)

        with self._deferred_context():
            self._emit(f"s_mul_i32 s[{s_in_stride_c_c0_c1_diff}], {diff_c0_c1}, s[{s_in_stride_c}]")
            self._emit(f"s_mul_i32 s[{s_in_stride_c_c1}], s[{s_move_slice_k_c1}], s[{s_in_stride_c}]  ; might be 0 or larger")
        return self._get_deferred()

    def expr(self):
        # k0, k1e is unmerge.  k1e is merged from k1, e
        self._emit(f"v_add_u32 v[{self.v_move_slice_k_ix()}], s[{self.s_move_slice_k_x()}], v[{self.v_move_slice_k_ix()}]")
        self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_gemm_k_num_x()}], v[{self.v_move_slice_k_ix()}]")
        self._emit(f"v_subrev_u32 v[{self.v_move_slice_k_ix()}], s[{self.s_gemm_k_num_x()}], v[{self.v_move_slice_k_ix()}]")
        self._emit(f"v_add_u32 v[{self.v_move_slice_k_iy()}], 1, v[{self.v_move_slice_k_iy()}]")
        self._emit(f"s_mov_b64 exec, -1")
        self._emit_empty_line()
        self._emit(f"v_add_u32 v[{self.v_move_slice_k_iy()}], s[{self.s_move_slice_k_y()}], v[{self.v_move_slice_k_iy()}]")
        self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_gemm_k_num_y()}], v[{self.v_move_slice_k_iy()}]")
        self._emit(f"v_subrev_u32 v[{self.v_move_slice_k_iy()}], s[{self.s_gemm_k_num_y()}], v[{self.v_move_slice_k_iy()}]")
        self._emit(f"v_add_u32 v[{self.v_move_slice_k_ic1()}], 1, v[{self.v_move_slice_k_ic1()}]")
        self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_c()}], v[{self.v_in_os_base()}]")
        # self._emit(f"v_add_u32 v[{self.v_wei_os_base()}], s[{self.s_wei_stride_k()}], v[{self.v_wei_os_base()}]")
        self._emit(f"s_mov_b64 exec, -1")
        self._emit_empty_line()
        self._emit(f"v_add_u32 v[{self.v_move_slice_k_ic1()}], s[{self.s_move_slice_k_c1()}], v[{self.v_move_slice_k_ic1()}]")
        self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_c_c1()}], v[{self.v_in_os_base()}]")
        # self._emit(f"v_add_u32 v[{self.v_wei_os_base()}], s[{self.s_wei_stride_k_c1()}], v[{self.v_wei_os_base()}]")

class macro_igemm_fwd_gtc_move_slice_window_k_y_x_tb_c0_gt_1_t(macro_base_t):
    def __init__(self, mc, tunable, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.tunable = tunable
        self.declare_arg("v_move_slice_k_ic1")
        self.declare_arg("v_move_slice_k_iy")
        self.declare_arg("v_move_slice_k_ix")
        self.declare_arg("s_gemm_k_num_c1")
        self.declare_arg("s_gemm_k_num_y")
        self.declare_arg("s_gemm_k_num_x")
        self.declare_arg("s_move_slice_k_c1")
        self.declare_arg("s_move_slice_k_y")
        self.declare_arg("s_move_slice_k_x")
        self.declare_arg("v_in_os_base")
        self.declare_arg("s_in_stride_c")
        self.declare_arg("s_in_stride_c_c1")
        self.declare_arg("s_in_stride_c_c0_c1_diff")
    def name(self):
        return '.v_fwd_gtc_move_slice_window_k_y_x_tb_c0_gt_1'

    def init_stride_c(self, s_in_stride_c, s_in_stride_c_c1, s_in_stride_c_c0_c1_diff, s_move_slice_k_c1):
        '''
        s_in_stride_c, s_move_slice_k_c1 is known value, want to compute other
        '''
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4

        ca_c0, ca_c1e, ca_k0, ca_k1  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        cb_c0, cb_c1e, cb_n0, cb_n1b = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(t_ta) == 4 and len(t_tb) == 4

        ta_c0, ta_c1e, ta_k0, ta_k1   = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        tb_c0, tb_c1e, tb_n0, tb_n1b  = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        nb_c0, nb_c1e = cb_c0 * tb_c0, cb_c1e * tb_c1e
        unmerge_sub_c = self.tunable.unmerge_sub_c
        assert unmerge_sub_c % nb_c0 == 0
        unmerge_sub_tb_c1 = unmerge_sub_c // nb_c0
        assert nb_c1e % unmerge_sub_tb_c1 == 0

        diff_c0_c1 = self.tunable.gemm_k_per_block - unmerge_sub_tb_c1 # !!! the diff of 2 unmerged dimension (like K=K0*K1)

        with self._deferred_context():
            self._emit(f"s_mul_i32 s[{s_in_stride_c_c0_c1_diff}], {diff_c0_c1}, s[{s_in_stride_c}]")
            self._emit(f"s_mul_i32 s[{s_in_stride_c_c1}], s[{s_move_slice_k_c1}], s[{s_in_stride_c}]  ; might be 0 or larger")
        return self._get_deferred()

    def expr(self):
        # k0, k1e is unmerge.  k1e is merged from k1, e
        self._emit(f"v_add_u32 v[{self.v_move_slice_k_ix()}], s[{self.s_move_slice_k_x()}], v[{self.v_move_slice_k_ix()}]")
        self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_gemm_k_num_x()}], v[{self.v_move_slice_k_ix()}]")
        self._emit(f"v_subrev_u32 v[{self.v_move_slice_k_ix()}], s[{self.s_gemm_k_num_x()}], v[{self.v_move_slice_k_ix()}]")
        self._emit(f"v_add_u32 v[{self.v_move_slice_k_iy()}], 1, v[{self.v_move_slice_k_iy()}]")
        self._emit(f"s_mov_b64 exec, -1")
        self._emit_empty_line()
        self._emit(f"v_add_u32 v[{self.v_move_slice_k_iy()}], s[{self.s_move_slice_k_y()}], v[{self.v_move_slice_k_iy()}]")
        self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_gemm_k_num_y()}], v[{self.v_move_slice_k_iy()}]")
        self._emit(f"v_subrev_u32 v[{self.v_move_slice_k_iy()}], s[{self.s_gemm_k_num_y()}], v[{self.v_move_slice_k_iy()}]")
        self._emit(f"v_add_u32 v[{self.v_move_slice_k_ic1()}], 1, v[{self.v_move_slice_k_ic1()}]")
        self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_c()}], v[{self.v_in_os_base()}]")
        # self._emit(f"v_add_u32 v[{self.v_wei_os_base()}], s[{self.s_wei_stride_k()}], v[{self.v_wei_os_base()}]")
        self._emit(f"s_mov_b64 exec, -1")
        self._emit_empty_line()
        self._emit(f"v_add_u32 v[{self.v_move_slice_k_ic1()}], s[{self.s_move_slice_k_c1()}], v[{self.v_move_slice_k_ic1()}]")
        self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_c_c1()}], v[{self.v_in_os_base()}]")
        
        self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_gemm_k_num_c1()}], v[{self.v_move_slice_k_ic1()}]")
        self._emit(f"v_subrev_u32 v[{self.v_move_slice_k_ic1()}], s[{self.s_gemm_k_num_c1()}], v[{self.v_move_slice_k_ic1()}]")
        self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_c_c0_c1_diff()}], v[{self.v_in_os_base()}]")
        self._emit(f"s_mov_b64 exec, -1")
        self._emit_empty_line()

class macro_igemm_fwd_gtc_move_slice_window_k_tb_t(macro_base_t):
    def __init__(self, mc, tunable, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.tunable = tunable
        self.declare_arg("v_in_os")
        self.declare_arg("v_move_slice_k_ic1")
        self.declare_arg("s_gemm_k_num_c1")
        self.declare_arg("s_move_slice_k_c1")
        self.declare_arg("s_in_stride_c")
        self.declare_arg("s_in_stride_c_c1")
        self.declare_arg("s_in_stride_c_c0_c1_diff")
    def name(self):
        return '.v_fwd_gtc_move_slice_window_k_tb'

    def init_stride_c(self, s_in_stride_c, s_in_stride_c_c1, s_in_stride_c_c0_c1_diff, s_move_slice_k_c1):
        '''
        s_in_stride_c, s_move_slice_k_c1 is known value, want to compute other
        '''
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths

        

        assert len(c_ta) == 4 and len(c_tb) == 4

        ca_c0, ca_c1e, ca_k0, ca_k1  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        cb_c0, cb_c1e, cb_n0, cb_n1b = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(t_ta) == 4 and len(t_tb) == 4

        ta_c0, ta_c1e, ta_k0, ta_k1   = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        tb_c0, tb_c1e, tb_n0, tb_n1b  = t_tb[0], t_tb[1], t_tb[2], t_tb[3]


        nb_c0, nb_c1e = cb_c0 * tb_c0, cb_c1e * tb_c1e
        unmerge_sub_c = self.tunable.unmerge_sub_c
        assert unmerge_sub_c % nb_c0 == 0
        unmerge_sub_tb_c1 = unmerge_sub_c // nb_c0
        assert nb_c1e % unmerge_sub_tb_c1 == 0

        diff_c0_c1 = self.tunable.gemm_k_per_block - unmerge_sub_tb_c1 # !!! the diff of 2 unmerged dimension (like K=K0*K1)

        with self._deferred_context():
            self._emit(f"s_mul_i32 s[{s_in_stride_c_c0_c1_diff}], {diff_c0_c1}, s[{s_in_stride_c}]")
            self._emit(f"s_mul_i32 s[{s_in_stride_c_c1}], s[{s_move_slice_k_c1}], s[{s_in_stride_c}]  ; might be 0 or larger")
        return self._get_deferred()

    def expr(self):
        self._emit(f"v_add_u32 v[{self.v_move_slice_k_ic1()}], s[{self.s_move_slice_k_c1()}], v[{self.v_move_slice_k_ic1()}]")
        self._emit(f"v_add_u32 v[{self.v_in_os()}], s[{self.s_in_stride_c_c1()}], v[{self.v_in_os()}]")
        self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_gemm_k_num_c1()}], v[{self.v_move_slice_k_ic1()}]")
        self._emit(f"v_subrev_u32 v[{self.v_move_slice_k_ic1()}], s[{self.s_gemm_k_num_c1()}], v[{self.v_move_slice_k_ic1()}]")
        self._emit(f"v_add_u32 v[{self.v_in_os()}], s[{self.s_in_stride_c_c0_c1_diff()}], v[{self.v_in_os()}]")
        self._emit(f"s_mov_b64 exec, -1")


class macro_igemm_fwd_gtc_move_slice_window_k_1d_tb_t(macro_base_t):
    def __init__(self, mc, tunable, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.tunable = tunable
        self.declare_arg("v_in_os")
        self.declare_arg("s_move_slice_k_c1")
        self.declare_arg("s_in_stride_c")
        self.declare_arg("s_in_stride_c_c1")

    def name(self):
        return '.v_fwd_gtc_move_slice_window_k_1d_tb'

    def init_stride_c(self, s_in_stride_c, s_in_stride_c_c1, s_move_slice_k_c1):
        '''
        s_in_stride_c, s_move_slice_k_c1 is known value, want to compute other
        '''
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4

        ca_c0, ca_c1e, ca_k0, ca_k1  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        cb_c0, cb_c1e, cb_n0, cb_n1b = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(t_ta) == 4 and len(t_tb) == 4

        ta_c0, ta_c1e, ta_k0, ta_k1   = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        tb_c0, tb_c1e, tb_n0, tb_n1b  = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        nb_c0, nb_c1e = cb_c0 * tb_c0, cb_c1e * tb_c1e
        unmerge_sub_c = self.tunable.unmerge_sub_c
        assert unmerge_sub_c % nb_c0 == 0
        unmerge_sub_tb_c1 = unmerge_sub_c // nb_c0
        assert nb_c1e % unmerge_sub_tb_c1 == 0

        with self._deferred_context():
            self._emit(f"s_mul_i32 s[{s_in_stride_c_c1}], s[{s_move_slice_k_c1}], s[{s_in_stride_c}]  ; might be 0 or larger")
        return self._get_deferred()

    def expr(self):
        self._emit(f"v_add_u32 v[{self.v_in_os()}], s[{self.s_in_stride_c_c1()}], v[{self.v_in_os()}]")


class igemm_fwd_gtc_t(mc_base_t):
    '''
                      tensor a (wei)                   tensor b (in)
    thread_lengths  : ta_c0, ta_c1e, ta_k0, ta_k1,     tb_c0, tb_c1e, tb_n0, tb_n1b
    cluster_lengths : ca_c0, ca_c1e, ca_k0, ca_k1,     cb_c0, cb_c1e, cb_n0, cb_n1b

    for wei, we always want to load GemmK(c0, c1e) first, then GemmM(k0, k1)
    indeed, c0*c1e should be treated as a single dimension

    '''
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
        self.global_load_in = self.global_load_in_t(mc, self)
        self.global_load_wei = self.global_load_wei_t(mc, self)
        self.shared_store_in = self.shared_store_in_t(mc, self)
        self.shared_store_wei = self.shared_store_wei_t(mc, self)

        wei_thread_copy_index, in_thread_copy_index = self.get_thread_copy_index()
        self.in_thread_copy_ndim = len(in_thread_copy_index)
        self.wei_thread_copy_ndim = len(wei_thread_copy_index)
        assert self.in_thread_copy_ndim in (0, 1, 2)
        assert self.wei_thread_copy_ndim in (0, 1, 2)

        '''
         in generic tensor contraction, gemm_m direction always is *good* dimension, fwd:k0*k1, bwd:c0*c1, wrw:k0*k1
         hence we always want to split coalescing groups along m direction, to store c matrix
        '''
        self.coalescing_store_groups = igemm_next_pow2(self.tunable.coalescing_store_groups)
        if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            assert (self.tunable.gemm_m_per_thread * self.tunable.gemm_m_repeat) % self.coalescing_store_groups == 0, \
                f"coalescing store groups should be divided by thread m {self.tunable.gemm_m_per_thread}x{self.tunable.gemm_m_repeat}"

            ctrl_thread_mapping = ctrl_thread_mapping_t()
                    #                        ->      MR x  NR x ML1 x NL1 x ML0 x NL0
            ctrl_thread_mapping.thread_lengths = [self.tunable.gemm_m_repeat, self.tunable.gemm_n_repeat, 1, 1, self.tunable.gemm_m_per_thread, self.tunable.gemm_n_per_thread]
            ctrl_thread_mapping.cluster_lengths = [1, 1, self.tunable.gemm_m_level1_cluster, self.tunable.gemm_n_level1_cluster, self.tunable.gemm_m_level0_cluster, self.tunable.gemm_n_level0_cluster]
            self.thread_mapping = igemm_thread_mapping_t(self.mc, ctrl_thread_mapping)

            ctrl_coalescing_store = ctrl_coalescing_store_t()
            ctrl_coalescing_store.ctm = ctrl_thread_mapping
            ctrl_coalescing_store.coalescing_groups = self.coalescing_store_groups
            ctrl_coalescing_store.data_byte = amdgpu_precision_data_byte(self.tunable.precision)

            ctrl_coalescing_store.vector_write_out = 1                      # TODO: some cases this can be set to other value
            ctrl_coalescing_store.block_size = self.tunable.block_size

            gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()
            na_c0, na_c1e, na_k0, na_k1, nb_c0, nb_c1e, nb_n0, nb_n1b = self.get_dims_lengths()
            ctrl_coalescing_store.gemm_m_m0_m1 = [na_k0, na_k1]
            if gemm_m_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0:
                ctrl_coalescing_store.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M1_M0

            ctrl_coalescing_store.adjust_optimal_coalescing_groups()        # in m1_m0 order, must adjust 
            self.coalescing_store = igemm_coalescing_store_t(mc, ctrl_coalescing_store)

        else:
            def flatten(x):
                from functools import reduce
                return reduce(lambda a, b: a*b, x, 1)
            ctrl_xdlops_mapping = get_ctrl_xdlops_mapping_from_wave_tile(self.tunable.gemm_m_per_block, self.tunable.gemm_n_per_block, self.tunable.wave_tile_m, self.tunable.wave_tile_n, self.tunable.wave_tile_k,
                    self.tunable.wave_repeat_m, self.tunable.wave_repeat_n, self.tunable.wave_step_m, self.tunable.wave_step_n, self.tunable.block_size // AMDGPU_WAVE_SIZE, self.tunable.precision)
            self.xdlops_mapping = igemm_xdlops_mapping_t(self.mc, ctrl_xdlops_mapping)
            assert flatten(ctrl_xdlops_mapping.acc_c_per_thread_m()) % self.coalescing_store_groups == 0, \
                f"coalescing store groups should be divided by agpr per thread in m direction {ctrl_xdlops_mapping.acc_c_per_thread_m()}"

            ctrl_coalescing_store_xdlops = ctrl_coalescing_store_xdlops_t()
            ctrl_coalescing_store_xdlops.cxm = ctrl_xdlops_mapping
            ctrl_coalescing_store_xdlops.coalescing_groups = self.coalescing_store_groups
            ctrl_coalescing_store_xdlops.data_byte = amdgpu_precision_data_byte(self.tunable.precision)

            ctrl_coalescing_store_xdlops.vector_write_out = 1                      # TODO: some cases this can be set to other value
            ctrl_coalescing_store_xdlops.block_size = self.tunable.block_size
        
            gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()
            na_c0, na_c1e, na_k0, na_k1, nb_c0, nb_c1e, nb_n0, nb_n1b = self.get_dims_lengths()
            ctrl_coalescing_store_xdlops.gemm_m_m0_m1 = [na_k0, na_k1]
            if gemm_m_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0:
                # we may consider not suppor this mode
                ctrl_coalescing_store_xdlops.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M1_M0
            ctrl_coalescing_store_xdlops.adjust_optimal_coalescing_groups()        # in m1_m0 order, must adjust 
            self.coalescing_store = igemm_coalescing_store_xdlops_t(mc, ctrl_coalescing_store_xdlops)



        
        self.label_out = f"L_{self.name()}_out"
        self.dict_shifted_stride = dict()

        self.karg = self.kernel_karg_t(mc, self)
        self.sgpr = self.kernel_sgpr_t(mc, self)
        self.vgpr = self.kernel_vgpr_t(mc, self)
        if self.tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            self.agpr = self.kernel_agpr_t(mc, self)
    
    def name(self):
        return igemm_gtc_encode_kernel_name(self.tunable, self.mc.arch_config.arch)
    
    def try_shift_stride(self, gpr, shifter):
        assert type(gpr) is sym_t
        with self._deferred_context():
            if gpr.label not in self.dict_shifted_stride:
                self.dict_shifted_stride[gpr.label] = gpr
                self._emit(f"s_lshl_b32 s[{gpr()}], s[{gpr()}], {shifter}")
        return self._get_deferred()
    
    def get_lds_gemm_m_gemm_n_order(self):
        def need_reverse_order(x0, x1):
            if x0 != 1 and x1 == 1:
                return True
            if x0 > x1:
                return True
            return False

        ta_c0, ta_c1e, ta_k0, ta_k1, tb_c0, tb_c1e, tb_n0, tb_n1b = self.get_thread_lengths()

        gemm_n_order = IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B
        if self.tunable.allow_lds_reorder:
            if need_reverse_order(tb_n0, tb_n1b):
                gemm_n_order = IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N1B_N0
                assert False, "maybe not correct"

        gemm_m_order = IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1
        if self.tunable.allow_lds_reorder:
            if need_reverse_order(ta_k0, ta_k1):
                gemm_m_order = IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0
                assert False, "maybe not correct"

        return gemm_m_order, gemm_n_order


    class global_load_in_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_wei_2d_global_load, m_in_2d_global_load = outer.get_macro_global_load()
            return m_in_2d_global_load.get_issues()

        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr

            m_wei_2d_global_load, m_in_2d_global_load = self.outer.get_macro_global_load()
            s_in_stride_d0, s_in_stride_d1, s_wei_stride_d0, s_wei_stride_d1 = self.outer.get_symbol_global_load_s_stride_d0_d1()
            with self._deferred_context():
                self._emit(f"; load input")
                if self.outer.tunable.nxe != 0:
                    self._emit(f".v_clear_nc {v.v_gld_b()}, {m_in_2d_global_load.ctrl.length_d0 * m_in_2d_global_load.ctrl.length_d1}")
                    self._emit(f"v_cmp_eq_u32 vcc, 1, v[{v.v_in_flag()}]")
                    self._emit(f"s_and_saveexec_b64 s[{s.s_tmp(4)}:{s.s_tmp(5)}], vcc")
                if self.outer.tunable.precache_soffset:
                    self._emit(m_in_2d_global_load(v.v_gld_b(), s.s_p_in(), v.v_in_os(), s_in_stride_d0(), s_in_stride_d1(), s.s_in_offset()))
                else:
                    self._emit(m_in_2d_global_load(v.v_gld_b(), s.s_p_in(), v.v_in_os(), s_in_stride_d0(), s_in_stride_d1(), s.s_tmp()))
                if self.outer.tunable.nxe != 0:
                    self._emit(f"s_or_b64 exec, exec, s[{s.s_tmp(4)}:{s.s_tmp(5)}]")
            return self._get_deferred()

    def is_1d_move_slice_k(self):
        '''
        this now only meaning for input tensor
        '''
        na_c0, na_c1e, na_k0, na_k1, nb_c0, nb_c1e, nb_n0, nb_n1b = self.get_dims_lengths()
        if self.tunable.nxe != 0:
            return False        # if not nxe 0, it is possible that we can do move slice, but that will lead to extra index calculation
        if nb_c1e != 1 and nb_c0 == 1:
            return True
        # it is meanless to let n_c1e==1 and n_c0!=1
        return False

    class global_load_wei_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_wei_2d_global_load, m_in_2d_global_load  = self.outer.get_macro_global_load()
            return m_wei_2d_global_load.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr

            m_wei_2d_global_load, m_in_2d_global_load = self.outer.get_macro_global_load()
            s_in_stride_d0, s_in_stride_d1, s_wei_stride_d0, s_wei_stride_d1 = self.outer.get_symbol_global_load_s_stride_d0_d1()
            with self._deferred_context():
                self._emit(f"; load weight")
                # self._emit(f".v_clear_nc {v.v_gld_a()}, {m_wei_2d_global_load.ctrl.length_d0 * m_wei_2d_global_load.ctrl.length_d1}")
                if self.outer.tunable.precache_soffset:
                    self._emit(m_wei_2d_global_load(v.v_gld_a(), s.s_p_wei(), v.v_wei_os(), s_wei_stride_d0(), s_wei_stride_d1(), s.s_wei_offset()))
                else:
                    self._emit(m_wei_2d_global_load(v.v_gld_a(), s.s_p_wei(), v.v_wei_os(), s_wei_stride_d0(), s_wei_stride_d1(), s.s_tmp()))
            return self._get_deferred() 

    class shared_store_in_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_in_2d_shared_store, m_wei_2d_shared_store = self.outer.get_macro_shared_store()
            return  m_in_2d_shared_store.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr
            m_in_2d_shared_store, m_wei_2d_shared_store = self.outer.get_macro_shared_store()
            with self._deferred_context():
                self._emit(m_in_2d_shared_store(v.v_gld_b(), v.v_sst_b_os()))
            return self._get_deferred()

    class shared_store_wei_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_in_2d_shared_store, m_wei_2d_shared_store = self.outer.get_macro_shared_store()
            return m_wei_2d_shared_store.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr
            m_in_2d_shared_store, m_wei_2d_shared_store = self.outer.get_macro_shared_store()
            with self._deferred_context():
                self._emit(m_wei_2d_shared_store(v.v_gld_a(), v.v_sst_a_os()))
            return self._get_deferred()



    class kernel_karg_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
            self.k_p_in       = sym_t('k_p_in'          ,0)
            self.k_p_wei      = sym_t('k_p_wei'         ,8)
            self.k_p_out      = sym_t('k_p_out'         ,16)
            self.k_hi         = sym_t('k_hi'            ,24)
            self.k_wi         = sym_t('k_wi'            ,28)
            self.k_n          = sym_t('k_n'             ,32)
            self.k_k          = sym_t('k_k'             ,36)
            self.k_c          = sym_t('k_c'             ,40)
            self.k_ho         = sym_t('k_ho'            ,44)
            self.k_wo         = sym_t('k_wo'            ,48)
            self.k_stride_h   = sym_t('k_stride_h'      ,52)
            self.k_stride_w   = sym_t('k_stride_w'      ,56)
            self.k_dilation_h = sym_t('k_dilation_h'    ,60)
            self.k_dilation_w = sym_t('k_dilation_w'    ,64)
            self.k_pad_h      = sym_t('k_pad_h'         ,68)
            self.k_pad_w      = sym_t('k_pad_w'         ,72)
            self.k_y          = sym_t('k_y'             ,76)
            self.k_x          = sym_t('k_x'             ,80)
            self.k_group      = sym_t('k_group'         ,84)
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self.k_magic_0      = sym_t('k_magic_0'         ,88)
                self.k_magic_1      = sym_t('k_magic_1'         ,92)
                self.k_magic_2      = sym_t('k_magic_2'         ,96)
                self.k_magic_3      = sym_t('k_magic_3'         ,100)
                self.k_magic_4      = sym_t('k_magic_4'         ,104)
                self.k_magic_5      = sym_t('k_magic_5'         ,108)
                self.k_magic_6      = sym_t('k_magic_6'         ,112)
                self.k_shift_pack_0 = sym_t('k_shift_pack_0'    ,116)
                self.k_shift_pack_1 = sym_t('k_shift_pack_1'    ,120)
                self.k__pack_0      = sym_t('k__pack_0'         ,124)
                self.k_end          = sym_t('k_end'             ,128)
            else:
                self.k_end          = sym_t('k_end'             ,88)

        def get_count(self):
            return self.k_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('k_'):
                    self._emit(v.declare())

    class kernel_sgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            ta_c0, ta_c1e, ta_k0, ta_k1, tb_c0, tb_c1e, tb_n0, tb_n1b = outer.get_thread_lengths()
            sseq                          = gpr_sequencer_t()
            self.outer                    = outer
            self.s_ka                     = sym_t('s_ka'                      , sseq(2))
            self.s_bx                     = sym_t('s_bx'                      , sseq(1))
            self.s_by                     = sym_t('s_by'                      , sseq(1))
            self.s_p_in                   = sym_t('s_p_in'                    , sseq(4))
            self.s_p_wei                  = sym_t('s_p_wei'                   , sseq(4))
            self.s_p_out                  = sym_t('s_p_out'                   , sseq(4))
            self.s_hi                     = sym_t('s_hi'                      , sseq(1))
            self.s_wi                     = sym_t('s_wi'                      , sseq(1))
            self.s_n                      = sym_t('s_n'                       , sseq(1))
            self.s_k                      = sym_t('s_k'                       , sseq(1))    # this is indeed k_per_group
            self.s_c                      = sym_t('s_c'                       , sseq(1))    # this is indeed c_per_group
            if outer.tunable.nxe != 0:
                self.s_ho                 = sym_t('s_ho'                      , sseq(1))
                self.s_wo                 = sym_t('s_wo'                      , sseq(1))
                self.s_stride_h           = sym_t('s_stride_h'                , sseq(1))
                self.s_stride_w           = sym_t('s_stride_w'                , sseq(1))
                self.s_dilation_h         = sym_t('s_dilation_h'              , sseq(1))
                self.s_dilation_w         = sym_t('s_dilation_w'              , sseq(1))
                self.s_pad_h              = sym_t('s_pad_h'                   , sseq(1))
                self.s_pad_w              = sym_t('s_pad_w'                   , sseq(1))
                self.s_y                  = sym_t('s_y'                       , sseq(1))
                self.s_x                  = sym_t('s_x'                       , sseq(1))
            self.s_group                  = sym_t('s_group'                   , sseq(1))

            # stride for wei
            if outer.tunable.nxe != 0:
                self.s_wei_stride_c       = sym_t('s_wei_stride_c'            , sseq(1))
                if ta_c0 != 1:
                    self.s_wei_stride_c1e = sym_t('s_wei_stride_c1e'          , sseq(1))
                self.s_wei_stride_k       = sym_t('s_wei_stride_k'            , sseq(1))
            if ta_k0 != 1:
                self.s_wei_stride_k0      = sym_t('s_wei_stride_k0'           , sseq(1))

            # stride for in
            if outer.tunable.nxe == 0:
                self.s_stride_hw          = sym_t('s_stride_hw'               , sseq(1))
            self.s_in_stride_c            = sym_t('s_in_stride_c'             , sseq(1))

            self.s_in_stride_n            = sym_t('s_in_stride_n'             , sseq(1))
            if tb_c0 != 1:
                self.s_in_stride_c0       = sym_t('s_in_stride_c0'            , sseq(1))
            if tb_n0 != 1:
                self.s_in_stride_n0       = sym_t('s_in_stride_n0'            , sseq(1))

            # stride for out
            #if outer.tunable.nxe != 0:
            self.s_out_stride_k           = sym_t('s_out_stride_k'            , sseq(1))
            self.s_out_stride_n           = sym_t('s_out_stride_n'            , sseq(1))
            if outer.tunable.gemm_n_unmerge_cluster:
                self.s_out_stride_n0      = sym_t('s_out_stride_n0'           , sseq(1))

            self.s_in_stride_c_c1         = sym_t("s_in_stride_c_c1"          , sseq(1))
            self.s_in_stride_c_c0_c1_diff = sym_t("s_in_stride_c_c0_c1_diff"  , sseq(1))

            self.s_block_gtc_ig           = sym_t("s_block_gtc_ig"            , sseq(1))
            self.s_block_gtc_ik           = sym_t("s_block_gtc_ik"            , sseq(1))
            self.s_block_gtc_in0          = sym_t("s_block_gtc_in0"           , sseq(1))
            self.s_block_gtc_in1b         = sym_t("s_block_gtc_in1b"          , sseq(1))

            self.s_move_slice_k_c1e       = sym_t("s_move_slice_k_c1e"        , sseq(1))
            if outer.tunable.nxe != 0:
                self.s_move_slice_k_c1    = sym_t("s_move_slice_k_c1"         , sseq(1))
                self.s_move_slice_k_y     = sym_t("s_move_slice_k_y"          , sseq(1))
                self.s_move_slice_k_x     = sym_t("s_move_slice_k_x"          , self.s_block_gtc_ig.value)

            self.s_knum                   = sym_t("s_knum"                    , 3)
            self.s_gemm_k_num_c1          = sym_t("s_gemm_k_num_c1"           , sseq(1))
            if outer.tunable.nxe != 0:
                self.s_gemm_k_num_y       = sym_t("s_gemm_k_num_y"            , self.s_y.value)
                self.s_gemm_k_num_x       = sym_t("s_gemm_k_num_x"            , self.s_x.value)

            if outer.tunable.nxe != 0:
                self.s_dim_b               = sym_t("s_dim_b"                  , self.s_move_slice_k_y.value)

            self.s_kitr                    = sym_t("s_kitr"                   , 1)
            if outer.tunable.precache_soffset:
                m_wei_2d_global_load, m_in_2d_global_load         = outer.get_macro_global_load()
                in_npc = m_in_2d_global_load.get_num_precache_soffset()
                wei_npc = m_wei_2d_global_load.get_num_precache_soffset()
                self.s_in_offset           = sym_t("s_in_offset"              ,sseq(in_npc))   # if this number is zero, it is also OK, since we would not use
                self.s_wei_offset          = sym_t("s_wei_offset"             ,sseq(wei_npc))
            self.s_k_padded                = sym_t("s_k_padded"             ,sseq(1))

            # TODO: this sgpr allocation is a mess
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                # allocate several sgpr to hold magic/shift value.
                self.s_shift_pack_0        = sym_t("s_shift_pack_0"           ,self.s_p_out.value + 2)
                self.s_shift_pack_1        = sym_t("s_shift_pack_1"           ,self.s_p_out.value + 3)

                self.s_magic_2             = sym_t("s_magic_2"                ,self.s_in_stride_c_c1.value)    # when load, loadx4 with magic_0/1
                self.s_magic_3             = sym_t("s_magic_3"                ,self.s_in_stride_c_c0_c1_diff.value) # when load, loadx4 with magic_0/1

                self.s_magic_4             = sym_t("s_magic_4"                ,self.s_move_slice_k_c1e.value)
                self.s_magic_5             = sym_t("s_magic_5"                ,self.s_gemm_k_num_c1.value)
                self.s_magic_6             = sym_t("s_magic_6"                ,self.s_block_gtc_in0.value)

            self.s_tmp                     = sym_t("s_tmp"                    ,sseq(6, 2))
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self.s_magic_0             = sym_t("s_magic_0"                ,self.s_p_wei.value + 2)
                self.s_magic_1             = sym_t("s_magic_1"                ,self.s_p_wei.value + 3)

            self.s_end                     = sym_t("s_end"                    ,sseq())

        def get_count(self):
            return self.s_end.value

        def emit(self):
            assert self.s_end.value <= amdgpu_sgpr_limit(self.mc.arch_config.arch), f"s_end:{self.s_end.value}, tunable:{self.outer.tunable.serialize()}"
            for k, v in self.__dict__.items():
                if k.startswith('s_'):
                    self._emit(v.declare())

    class kernel_vgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
            is_vgpr_acc_c = outer.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS
            vseq = gpr_sequencer_t()
            if is_vgpr_acc_c:
                self.v_c             = sym_t("v_c"            ,vseq(outer.tunable.num_vgpr_accumulate_c))
                v_c_num              = vseq()
            else:
                v_c_resuable_num     = outer.tunable.num_vgpr_accumulate_a + outer.tunable.num_vgpr_accumulate_b + \
                                        outer.tunable.num_global_load_a + outer.tunable.num_global_load_b + \
                                        16       # from v_sst_a_os to v_co_sst
                v_c_coalescing_num   = outer.tunable.num_agpr_accumulate_c // outer.coalescing_store_groups
                v_c_needed           = (v_c_coalescing_num - v_c_resuable_num) if (v_c_coalescing_num - v_c_resuable_num) > 0 else 0

                v_c_needed           = v_c_needed if v_c_needed > 2 else 2  # let at least 2
                self.v_c             = sym_t("v_c"            ,vseq(v_c_needed), f"coalescing:{v_c_coalescing_num}, needed:{v_c_needed}, resuable:{v_c_resuable_num}")

            self.v_a                 = sym_t("v_a"            ,vseq(outer.tunable.num_vgpr_accumulate_a))
            self.v_b                 = sym_t("v_b"            ,vseq(outer.tunable.num_vgpr_accumulate_b))
            self.v_gld_a             = sym_t("v_gld_a"        ,vseq(outer.tunable.num_global_load_a))
            self.v_gld_b             = sym_t("v_gld_b"        ,vseq(outer.tunable.num_global_load_b))
            self.v_sst_a_os          = sym_t("v_sst_a_os"     ,vseq(1))
            self.v_sst_b_os          = sym_t("v_sst_b_os"     ,vseq(1))
            self.v_sld_a_os          = sym_t("v_sld_a_os"     ,vseq(1))
            self.v_sld_b_os          = sym_t("v_sld_b_os"     ,vseq(1))
            self.v_in_os             = sym_t("v_in_os"        ,vseq(1))
            self.v_in_os_base        = sym_t("v_in_os_base"   ,vseq(1))
            if outer.tunable.nxe != 0:
                self.v_in_flag       = sym_t("v_in_flag"      ,vseq(1))
            self.v_wei_os            = sym_t("v_wei_os"       ,vseq(1))

            self.v_gtc_ta_ik1        = sym_t("v_gtc_ta_ik1"   ,vseq(1))
            self.v_gtc_ta_ik0        = sym_t("v_gtc_ta_ik0"   ,vseq(1))
            self.v_gtc_ta_ic1e       = sym_t("v_gtc_ta_ic1e"  ,vseq(1))
            self.v_gtc_ta_ic0        = sym_t("v_gtc_ta_ic0"   ,vseq(1))

            self.v_gtc_tb_in1b       = sym_t("v_gtc_tb_in1b"  ,vseq(1))
            self.v_gtc_tb_in0        = sym_t("v_gtc_tb_in0"   ,vseq(1))
            self.v_gtc_tb_ic1e       = sym_t("v_gtc_tb_ic1e"  ,vseq(1))
            # self.v_gtc_tb_ic0        = sym_t("v_gtc_tb_ic0"   ,vseq(1))
            self.v_gtc_tb_in1        = sym_t("v_gtc_tb_in1"   ,vseq(1))
            self.v_gtc_tb_ib         = sym_t("v_gtc_tb_ib"    ,vseq(1))
            if outer.tunable.nxe != 0:
                self.v_gtc_tb_ic1     = sym_t("v_gtc_tb_ic1"  ,vseq(1))

            self.v_co_sst            = sym_t("v_co_sst"       ,vseq(1))
            self.v_co_sld            = sym_t("v_co_sld"       ,vseq(1))

            self.v_out_os            = sym_t("v_out_os"       ,vseq(1))
            if outer.tunable.nxe != 0:
                self.v_out_flag      = sym_t("v_out_flag"     ,vseq(1))
            self.v_out_in0           = sym_t("v_out_in0"      ,vseq(1))
            self.v_out_in1b          = sym_t("v_out_in1b"     ,vseq(1))
            self.v_out_in1           = sym_t("v_out_in1"      ,vseq(1))

            self.v_in_iho           = sym_t("v_in_iho"        ,vseq(1))
            self.v_in_iwo           = sym_t("v_in_iwo"        ,vseq(1))
            self.v_in_ihi           = sym_t("v_in_ihi"        ,vseq(1))
            self.v_in_iwi           = sym_t("v_in_iwi"        ,vseq(1))
            if outer.tunable.nxe != 0:
                self.v_in_iy            = sym_t("v_in_iy"     ,vseq(1))
                self.v_in_ix            = sym_t("v_in_ix"     ,vseq(1))

            self.v_move_slice_k_ic1  = sym_t("v_move_slice_k_ic1" , self.v_gtc_tb_ic1.value if outer.tunable.nxe != 0 else self.v_gtc_tb_ic1e.value)
            if outer.tunable.nxe != 0:
                self.v_move_slice_k_iy = sym_t("v_move_slice_k_iy", self.v_in_iy.value)
                self.v_move_slice_k_ix = sym_t("v_move_slice_k_ix", self.v_in_ix.value)

            self.v_gemm_in       = sym_t("v_gemm_in"      , vseq(1))
            self.v_gemm_im       = sym_t("v_gemm_im"      , vseq(1))

            self.v_out_iho        = sym_t("v_out_iho" ,vseq(1))
            self.v_out_iwo        = sym_t("v_out_iwo" ,vseq(1))
            self.v_co_sub_m_index = sym_t("v_co_sub_m_index" ,vseq(1))
            self.v_co_sub_n_index = sym_t("v_co_sub_n_index" ,vseq(1))

            self.v_cur_k          = sym_t("v_cur_k" ,vseq(1))

            self.v_tmp           = sym_t("v_tmp"          ,vseq(6, 2))
            total_vgpr           = vseq()
            if outer.tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
                # if xdlops agpr is larger than vgpr usage, must change vgpr count to agpr
                total_vgpr       = max(total_vgpr, outer.tunable.num_agpr_accumulate_c)
            self.v_end           = sym_t("v_end"          ,total_vgpr)

        def get_count(self):
            return self.v_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('v_'):
                    self._emit(v.declare())

    class kernel_agpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            assert outer.tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS, 'only xdlops can use agpr'
            self.outer         = outer
            aseq = gpr_sequencer_t()
            self.a_c           = sym_t("a_c",          aseq(outer.tunable.num_agpr_accumulate_c))
            self.a_end         = sym_t("a_end",        aseq())

        def get_count(self):
            return self.a_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('a_'):
                    self._emit(v.declare())

    def get_thread_lengths(self):
        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(t_ta) == 4 and len(t_tb) == 4

        ta_c0, ta_c1e, ta_k0, ta_k1   = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        tb_c0, tb_c1e, tb_n0, tb_n1b  = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        if self.tunable.nxe == 0:
            # tb_c1e now can have length. for simplicity, c0 always be 1 for wei/inp
            assert ta_c0 == 1
            assert tb_c0 == 1
        else:
            #assert ta_c0 == 1, "wei not using c0. for wei treat c1e as c*e, single dimension"
            #assert tb_c0 == 1
            pass
            # there should be a case that tb_c1e not be 1 and nxe != 0. this should only be allowed in 1x1 and with stride or dilation or pad
            # assert tb_c0 == 1 and tb_c1e == 1, "input no need to use c0/c1e per thread"

        return ta_c0, ta_c1e, ta_k0, ta_k1, tb_c0, tb_c1e, tb_n0, tb_n1b    # ta K M, tb K N

    def get_cluster_lengths(self):
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4

        ca_c0, ca_c1e, ca_k0, ca_k1  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        cb_c0, cb_c1e, cb_n0, cb_n1b = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        assert ca_c0 == 1, "wei not using c0. for wei treat c1e as c*e"
        assert cb_c0 == 1, "input no need to use c0 cluster length"

        return ca_c0, ca_c1e, ca_k0, ca_k1, cb_c0, cb_c1e, cb_n0, cb_n1b

    def get_dims_lengths(self):
        ta_c0, ta_c1e, ta_k0, ta_k1, tb_c0, tb_c1e, tb_n0, tb_n1b = self.get_thread_lengths()
        ca_c0, ca_c1e, ca_k0, ca_k1, cb_c0, cb_c1e, cb_n0, cb_n1b = self.get_cluster_lengths()

        na_c0, na_c1e, na_k0, na_k1  = ta_c0 * ca_c0, ta_c1e * ca_c1e, ta_k0 * ca_k0, ta_k1 * ca_k1
        nb_c0, nb_c1e, nb_n0, nb_n1b = tb_c0 * cb_c0, tb_c1e * cb_c1e, tb_n0 * cb_n0, tb_n1b * cb_n1b

        return na_c0, na_c1e, na_k0, na_k1, nb_c0, nb_c1e, nb_n0, nb_n1b

    def get_thread_copy_dims(self):
        ta_c0, ta_c1e, ta_k0, ta_k1, tb_c0, tb_c1e, tb_n0, tb_n1b = self.get_thread_lengths()
        #wei_thread_copy_dims    = [ta_c0, ta_c1e, ta_k0, ta_k1]
        wei_thread_copy_dims    = [ta_k0, ta_k1, ta_c0, ta_c1e]     # always reordered!
        in_thread_copy_dims    = [tb_c0, tb_c1e, tb_n0, tb_n1b]

        return wei_thread_copy_dims, in_thread_copy_dims

    def get_thread_copy_index(self):
        wei_thread_copy_dims, in_thread_copy_dims = self.get_thread_copy_dims()
        wei_thread_copy_index   = _find_non_1_index_in_list(wei_thread_copy_dims)
        in_thread_copy_index   = _find_non_1_index_in_list(in_thread_copy_dims)
        '''
        if thread lengths both dimension is 1, means every thread only copy one pixel.
        we need support this also
        '''
        return wei_thread_copy_index, in_thread_copy_index

    def get_macro_global_load(self):
        '''
        NOTICE: for wei, always load GemmK(c0*c1e) first, then (k0*k1)
        '''
        inline = True if self.tunable.fma_interleave else False
        ta_c0, ta_c1e, ta_k0, ta_k1, tb_c0, tb_c1e, tb_n0, tb_n1b = self.get_thread_lengths()
        na_c0, na_c1e, na_k0, na_k1, nb_c0, nb_c1e, nb_n0, nb_n1b = self.get_dims_lengths()

        wei_thread_copy_dims, in_thread_copy_dims = self.get_thread_copy_dims()
        wei_thread_copy_index, in_thread_copy_index = self.get_thread_copy_index()
        ctrl_wei_gld = ctrl_2d_global_load_t()
        ctrl_in_gld = ctrl_2d_global_load_t()

        ctrl_wei_gld.vector_d1 = utility_gcd(ta_c1e, 4) if ta_c1e != 1 else 1
        ctrl_in_gld.vector_d1 = utility_gcd(tb_n1b, 4) if tb_n1b != 1 else 1

        if self.wei_thread_copy_ndim == 2:
            # [ta_k0, ta_k1, ta_c0, ta_c1e]
            # if wei_thread_copy_index[0] in (0, 1) and wei_thread_copy_index[1] in (2, 3):
            #     # reorder when global load. we need to order back into LDS
            #     ctrl_wei_gld.length_d0 = wei_thread_copy_dims[wei_thread_copy_index[1]]
            #     ctrl_wei_gld.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]
            # else:
            ctrl_wei_gld.length_d0 = wei_thread_copy_dims[wei_thread_copy_index[0]]
            ctrl_wei_gld.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[1]]
        elif self.wei_thread_copy_ndim == 1:
            ctrl_wei_gld.length_d0 = 1
            ctrl_wei_gld.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]
        else:
            ctrl_wei_gld.length_d0 = 1
            ctrl_wei_gld.length_d1 = wei_thread_copy_dims[-1]

        if self.in_thread_copy_ndim == 2:
            ctrl_in_gld.length_d0 = in_thread_copy_dims[in_thread_copy_index[0]]
            ctrl_in_gld.length_d1 = in_thread_copy_dims[in_thread_copy_index[1]]
        elif self.in_thread_copy_ndim == 1:
            ctrl_in_gld.length_d0 = 1
            ctrl_in_gld.length_d1 = in_thread_copy_dims[in_thread_copy_index[0]]
        else:
            ctrl_in_gld.length_d0 = 1
            ctrl_in_gld.length_d1 = in_thread_copy_dims[-1]

        if self.tunable.precache_soffset:
            return macro_igemm_2d_global_load_precache_soffset_t(self.mc, ctrl_wei_gld, inline), \
                    macro_igemm_2d_global_load_precache_soffset_t(self.mc, ctrl_in_gld, inline)
        else:
            return macro_igemm_2d_global_load_t(self.mc, ctrl_wei_gld, inline),  macro_igemm_2d_global_load_t(self.mc, ctrl_in_gld, inline)


    def get_macro_shared_store(self):
        wei_thread_copy_dims, in_thread_copy_dims = self.get_thread_copy_dims()
        wei_thread_copy_index, in_thread_copy_index = self.get_thread_copy_index()
        na_c0, na_c1e, na_k0, na_k1, nb_c0, nb_c1e, nb_n0, nb_n1b = self.get_dims_lengths()
        ta_c0, ta_c1e, ta_k0, ta_k1, tb_c0, tb_c1e, tb_n0, tb_n1b = self.get_thread_lengths()
        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()

        # give the LDS strides of wei dimensions [ta_k0, ta_k1, ta_c0, ta_c1e]
        if gemm_m_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1:
            wei_stride_list = [na_k1, 1, na_c1e*na_k0*na_k1, na_k0*na_k1]
        else:
            wei_stride_list = [1, na_k0, na_c1e*na_k0*na_k1, na_k0*na_k1]

        # give the LDS strides of in dimensions [tb_c0, tb_c1e, tb_n0, tb_n1b]
        if gemm_n_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B:
            in_stride_list = [nb_c1e*nb_n0*nb_n1b, nb_n0*nb_n1b, nb_n1b, 1]
        else:
            in_stride_list = [nb_c1e*nb_n0*nb_n1b, nb_n0*nb_n1b, 1, nb_n0]

        # print(f"__ wei_stride_list:{wei_stride_list}, wei_thread_copy_index:{wei_thread_copy_index}")

        wei_sst_ctrl = ctrl_2d_shared_store_t()
        wei_sst_ctrl.src_order = 1                  # for weight, always reverse order in register.
        wei_sst_ctrl.v_tmp = self.vgpr.v_tmp
        in_sst_ctrl = ctrl_2d_shared_store_t()
        if self.wei_thread_copy_ndim == 2:
            # [ta_k0, ta_k1, ta_c0, ta_c1e]
            if wei_thread_copy_index[0] in (0, 1) and wei_thread_copy_index[1] in (2, 3):
                # when store into LDS, reorder back. indeed we always wish this pattern, if ndim is 2
                wei_sst_ctrl.length_d0 = wei_thread_copy_dims[wei_thread_copy_index[1]]
                wei_sst_ctrl.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]
            else:
                wei_sst_ctrl.length_d0 = wei_thread_copy_dims[wei_thread_copy_index[0]]
                wei_sst_ctrl.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[1]]
                wei_sst_ctrl.need_transpose = 0
            if gemm_m_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1:
                wei_sst_ctrl.vector_d1 = ta_k1
            else:
                assert False, "tobe implement"
                wei_sst_ctrl.vector_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]

            if wei_thread_copy_index[0] in (0, 1) and wei_thread_copy_index[1] in (2, 3):
                wei_sst_ctrl.stride_d0 = wei_stride_list[wei_thread_copy_index[1]] * data_byte
                wei_sst_ctrl.stride_d1 = wei_stride_list[wei_thread_copy_index[0]] * data_byte
            else:
                wei_sst_ctrl.stride_d0 = wei_stride_list[wei_thread_copy_index[0]] * data_byte
                wei_sst_ctrl.stride_d1 = wei_stride_list[wei_thread_copy_index[1]] * data_byte

        elif self.wei_thread_copy_ndim == 1:
            wei_sst_ctrl.length_d0 = 1
            wei_sst_ctrl.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]

            if (gemm_m_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1 and ta_k1 != 1) or \
                (gemm_m_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0 and ta_k0 != 1):
                wei_sst_ctrl.vector_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]
            else:
                wei_sst_ctrl.vector_d1 = 1

            wei_sst_ctrl.stride_d0 = 1
            wei_sst_ctrl.stride_d1 = wei_stride_list[wei_thread_copy_index[0]] * data_byte
            if wei_sst_ctrl.length_d1 == 8 and wei_sst_ctrl.vector_d1 != 1:
                # assert False
                # TODO: this is indeed not optimal. may consider shuffle in the future.
                wei_sst_ctrl.length_d0 = 2
                wei_sst_ctrl.length_d1 = 4
                wei_sst_ctrl.vector_d1 = 4
                wei_sst_ctrl.stride_d0 = 4 * data_byte
        else:
            wei_sst_ctrl.length_d0 = 1
            wei_sst_ctrl.length_d1 = wei_thread_copy_dims[-1]

            if (gemm_m_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1 and ta_k1 != 1) or \
                (gemm_m_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0 and ta_k0 != 1):
                wei_sst_ctrl.vector_d1 = wei_thread_copy_dims[-1]
            else:
                wei_sst_ctrl.vector_d1 = 1

            wei_sst_ctrl.stride_d0 = 1
            wei_sst_ctrl.stride_d1 = wei_stride_list[-1] * data_byte
            if wei_sst_ctrl.length_d1 == 8 and wei_sst_ctrl.vector_d1 != 1:
                # assert False
                # TODO: this is indeed not optimal. may consider shuffle in the future.
                wei_sst_ctrl.length_d0 = 2
                wei_sst_ctrl.length_d1 = 4
                wei_sst_ctrl.vector_d1 = 4
                wei_sst_ctrl.stride_d0 = 4 * data_byte

        # [tb_c0, tb_c1e, tb_n0, tb_n1b]
        if self.in_thread_copy_ndim == 2:
            in_sst_ctrl.length_d0 = in_thread_copy_dims[in_thread_copy_index[0]]
            in_sst_ctrl.length_d1 = in_thread_copy_dims[in_thread_copy_index[1]]
            if gemm_n_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B:
                in_sst_ctrl.vector_d1 = tb_n1b
            else:
                in_sst_ctrl.vector_d1 = in_thread_copy_dims[in_thread_copy_index[1]]
            #in_sst_ctrl.vector_d1 = t_n1b
            in_sst_ctrl.stride_d0 = in_stride_list[in_thread_copy_index[0]] * data_byte
            in_sst_ctrl.stride_d1 = in_stride_list[in_thread_copy_index[1]] * data_byte
            #in_sst_ctrl.stride_d1 = 1
        elif self.in_thread_copy_ndim == 1:
            in_sst_ctrl.length_d0 = 1
            in_sst_ctrl.length_d1 = in_thread_copy_dims[in_thread_copy_index[0]]
            if (gemm_n_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B and tb_n1b != 1) or \
                (gemm_n_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N1B_N0 and tb_n0 != 1):
                in_sst_ctrl.vector_d1 = in_thread_copy_dims[in_thread_copy_index[0]]
            else:
                in_sst_ctrl.vector_d1 = 1
            in_sst_ctrl.stride_d0 = 1
            in_sst_ctrl.stride_d1 = in_stride_list[in_thread_copy_index[0]] * data_byte
            if in_sst_ctrl.length_d1 == 8 and in_sst_ctrl.vector_d1 != 1:
                # assert False
                # TODO: this is indeed not optimal. may consider shuffle in the future.
                in_sst_ctrl.length_d0 = 2
                in_sst_ctrl.length_d1 = 4
                in_sst_ctrl.vector_d1 = 4
                in_sst_ctrl.stride_d0 = 4 * data_byte
        else:
            in_sst_ctrl.length_d0 = 1
            in_sst_ctrl.length_d1 = in_thread_copy_dims[-1]
            if (gemm_n_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B and tb_n1b != 1) or \
                (gemm_n_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N1B_N0 and tb_n0 != 1):
                in_sst_ctrl.vector_d1 = in_thread_copy_dims[-1]
            else:
                in_sst_ctrl.vector_d1 = 1
            in_sst_ctrl.stride_d0 = 1
            in_sst_ctrl.stride_d1 = in_stride_list[-1] * data_byte
            if in_sst_ctrl.length_d1 == 8 and in_sst_ctrl.vector_d1 != 1:
                # assert False
                # TODO: this is indeed not optimal. may consider shuffle in the future.
                in_sst_ctrl.length_d0 = 2
                in_sst_ctrl.length_d1 = 4
                in_sst_ctrl.vector_d1 = 4
                in_sst_ctrl.stride_d0 = 4 * data_byte

        # print(f"in_sst_ctrl.vector_d1:{in_sst_ctrl.vector_d1}, wei_sst_ctrl.vector_d1:{wei_sst_ctrl.vector_d1}")
        # print(f"wei_sst_ctrl, {wei_sst_ctrl.serialize()}")
        inline = True if self.tunable.fma_interleave else False 
        return macro_igemm_2d_shared_store_t(self.mc, in_sst_ctrl, inline), macro_igemm_2d_shared_store_t(self.mc, wei_sst_ctrl, inline)

    # computation macro
    def get_macro_in_update_hw(self):
        inline = True if self.tunable.fma_interleave else False
        return macro_igemm_fwd_gtc_in_update_hw_t(self.mc, inline)

    def get_macro_in_update_os(self):
        inline = True if self.tunable.fma_interleave else False
        return macro_igemm_fwd_gtc_in_update_os_t(self.mc, amdgpu_precision_data_byte(self.tunable.precision), inline)

    def get_macro_move_slice_window(self):
        inline = True if self.tunable.fma_interleave else False
        move_slice_window_ta = macro_igemm_fwd_gtc_move_slice_window_ta_t(self.mc, self.tunable, inline)
        if self.tunable.nxe != 0:
            if self.tunable.tensor_b_thread_lengths[0] > 1:
                return move_slice_window_ta, macro_igemm_fwd_gtc_move_slice_window_k_y_x_tb_c0_gt_1_t(self.mc, self.tunable, inline)
            else:
                return move_slice_window_ta, macro_igemm_fwd_gtc_move_slice_window_k_y_x_tb_t(self.mc, self.tunable, inline)
        else:
            if self.is_1d_move_slice_k():
                return move_slice_window_ta, macro_igemm_fwd_gtc_move_slice_window_k_1d_tb_t(self.mc, self.tunable, inline)
            else:
                return move_slice_window_ta, macro_igemm_fwd_gtc_move_slice_window_k_tb_t(self.mc, self.tunable, inline)


    def get_macro_set_flag_hw(self):
        inline = True if self.tunable.fma_interleave else False
        return macro_igemm_fwd_gtc_set_flag_hw(self.mc, inline)

    def get_macro_set_flag_c(self):
        inline = True if self.tunable.fma_interleave else False
        return macro_igemm_fwd_gtc_set_flag_c(self.mc, inline)

    def get_symbol_global_load_s_stride_d0_d1(self):
        ta_c0, ta_c1e, ta_k0, ta_k1, tb_c0, tb_c1e, tb_n0, tb_n1b = self.get_thread_lengths()
        # get the symbol object that load 2d may use
        s = self.sgpr
        s_dummy = sym_t("s_dummy")
        wei_thread_copy_index, in_thread_copy_index  = self.get_thread_copy_index()

        # [tb_c0, tb_c1e, tb_n0, tb_n1b]
        in_stride_gprs = [s.s_in_stride_c0 if tb_c0 != 1 else s_dummy,
                    s.s_in_stride_c if self.tunable.nxe != 0 else s.s_in_stride_c,  # TODO: both case is just the same
                    s.s_in_stride_n0 if tb_n0 != 1 else s_dummy,
                    s_dummy]

        # [ta_k0, ta_k1, ta_c0, ta_c1e]
        wei_stride_gprs = [s.s_wei_stride_k0 if ta_k0 != 1 else s_dummy,
                    s.s_wei_stride_k if self.tunable.nxe != 0 else s.s_c,
                    s.s_wei_stride_c1e if ta_c0 != 1 else s_dummy,
                    s_dummy]
        
        # print(f" ___ wei_thread_copy_index:{wei_thread_copy_index}")

        if self.in_thread_copy_ndim == 2:
            s_in_stride_d0 = in_stride_gprs[in_thread_copy_index[0]]
            s_in_stride_d1 = in_stride_gprs[in_thread_copy_index[1]]
        elif self.in_thread_copy_ndim == 1:
            s_in_stride_d0 = s_dummy
            s_in_stride_d1 = in_stride_gprs[in_thread_copy_index[0]]
        else:
            s_in_stride_d0 = s_dummy
            s_in_stride_d1 = in_stride_gprs[-1]

        if self.wei_thread_copy_ndim == 2:
            # print(f" ____ wei_thread_copy_index:{len(wei_thread_copy_index)}")
            s_wei_stride_d0 = wei_stride_gprs[wei_thread_copy_index[0]]
            s_wei_stride_d1 = wei_stride_gprs[wei_thread_copy_index[1]]
        elif self.wei_thread_copy_ndim == 1:
            s_wei_stride_d0 = s_dummy
            s_wei_stride_d1 = wei_stride_gprs[wei_thread_copy_index[0]]
        else:
            s_wei_stride_d0 = s_dummy
            s_wei_stride_d1 = wei_stride_gprs[-1]

        return s_in_stride_d0, s_in_stride_d1, s_wei_stride_d0, s_wei_stride_d1


    def get_kernel_code(self):
        kernel_code = amdgpu_kernel_code_t({
                'enable_sgpr_kernarg_segment_ptr'   :   1,
                'enable_sgpr_workgroup_id_x'        :   1,
                'enable_sgpr_workgroup_id_y'        :   1,
                'enable_vgpr_workitem_id'           :   0,
                'workgroup_group_segment_byte_size' :   self.tunable.lds_total,
                'kernarg_segment_byte_size'         :   self.karg.get_count(),
                'wavefront_sgpr_count'              :   self.sgpr.get_count() + 2*3,
                'workitem_vgpr_count'               :   self.vgpr.get_count()
                })
        return kernel_code

    def get_kernel_args(self):
        '''
        float *p_in;
        float *p_wei;
        float *p_out;
        int hi;
        int wi;
        int n;
        int k;
        int c;
        int ho;
        int wo;
        int stride_h;
        int stride_w;
        int dilation_h;
        int dilation_w;
        int pad_h;
        int pad_w;
        int y;
        int x;
        int group;
        /* if use magic division */
        uint32_t magic_0;           // denom: sa=0: n*b / n_per_block, sa=1: k / m_per_block
        uint32_t magic_1;           // denom: ((n / nb_n0) * b) / nb_n1b
        uint32_t magic_2;           // denom: y*x, if nxe==0 not used
        uint32_t magic_3;           // denom: x, if nxe==0 not used
        uint32_t magic_4;           // denom: b
        uint32_t magic_5;           // denom: wo
        uint32_t magic_6;           // denom: n*b*k / (m_per_block*n_per_block)
        uint32_t shift_pack_0;
        uint32_t shift_pack_1;
        uint32_t __pack_0;
        '''
        kas = []
        # name: {}, .size: {}, .offset: {}, .value_kind: {}, .value_type
        kas.append(amdgpu_kernel_arg_t('p_in'           , 8,   0, 'global_buffer','f32',address_space='global',is_const='true'))
        kas.append(amdgpu_kernel_arg_t('p_wei'          , 8,   8, 'global_buffer','f32',address_space='global',is_const='true'))
        kas.append(amdgpu_kernel_arg_t('p_out'          , 8,  16, 'global_buffer','f32',address_space='global',is_const='false'))
        kas.append(amdgpu_kernel_arg_t('hi'             , 4,  24, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('wi'             , 4,  28, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('n'              , 4,  32, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('k'              , 4,  36, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('c'              , 4,  40, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('ho'             , 4,  44, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('wo'             , 4,  48, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('stride_h'       , 4,  52, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('stride_w'       , 4,  56, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_h'     , 4,  60, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_w'     , 4,  64, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('pad_h'          , 4,  68, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('pad_w'          , 4,  72, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('y'              , 4,  76, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('x'              , 4,  80, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('group'          , 4,  84, 'by_value','i32'))
        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            kas.append(amdgpu_kernel_arg_t('magic_0'        , 4,  88, 'by_value','i32'))
            kas.append(amdgpu_kernel_arg_t('magic_1'        , 4,  92, 'by_value','i32'))
            kas.append(amdgpu_kernel_arg_t('magic_2'        , 4,  96, 'by_value','i32'))
            kas.append(amdgpu_kernel_arg_t('magic_3'        , 4, 100, 'by_value','i32'))
            kas.append(amdgpu_kernel_arg_t('magic_4'        , 4, 104, 'by_value','i32'))
            kas.append(amdgpu_kernel_arg_t('magic_5'        , 4, 108, 'by_value','i32'))
            kas.append(amdgpu_kernel_arg_t('magic_6'        , 4, 112, 'by_value','i32'))
            kas.append(amdgpu_kernel_arg_t('shift_pack_0'   , 4, 116, 'by_value','i32'))
            kas.append(amdgpu_kernel_arg_t('shift_pack_1'   , 4, 120, 'by_value','i32'))
            kas.append(amdgpu_kernel_arg_t('__pack_0'       , 4, 124, 'by_value','i32'))
        else:
            pass
        return kas

    def get_kernel_info(self):
        kernel_code = self.get_kernel_code()
        kernel_args = self.get_kernel_args()
        kernel_info = amdgpu_kernel_info_t(kernel_code, self.name(), self.tunable.block_size, kernel_args)
        return kernel_info

    def get_kernel_macros(self):
        kernel_macros = []
        for attrs in dir(self):
            if attrs.startswith('get_macro_'):
                functor = getattr(self, attrs)
                rtn = functor()
                if rtn is None:
                    continue

                # here we follow the convention in code:
                # #1. for macro like emit class, use emit() to generate macro definition, use __call__() to call this macro
                # #2. for non-macro like emit class, which might want to "inline-ed" into normal code, no emit() is defined, just __call__().
                # hence need to check if has attr name "emit". if not have, it is type #2, no need to do emit() before hand.
                if type(rtn) is tuple:
                    for e in rtn:
                        #if hasattr(e, 'emit'):
                        if not e.is_inline():
                            #continue
                            kernel_macros.extend([m for m in rtn])
                else:
                    #if hasattr(rtn, 'emit'):
                    if not e.is_inline():
                        #continue
                        kernel_macros.append(rtn)
        return kernel_macros

    def emit_kernel_prologue(self):
        s = self.sgpr
        v = self.vgpr
        k = self.karg
        gemm_m_unmerge_cluster = self.tunable.gemm_m_unmerge_cluster
        gemm_n_unmerge_cluster = self.tunable.gemm_n_unmerge_cluster
        gemm_k_unmerge_cluster = self.tunable.gemm_k_unmerge_cluster

        assert gemm_m_unmerge_cluster == 0 and gemm_k_unmerge_cluster == 0, 'in fwd, gemm_m/k unmerge_cluster no need to change'

        ta_c0, ta_c1e, ta_k0, ta_k1, tb_c0, tb_c1e, tb_n0, tb_n1b = self.get_thread_lengths()
        ca_c0, ca_c1e, ca_k0, ca_k1, cb_c0, cb_c1e, cb_n0, cb_n1b = self.get_cluster_lengths()
        na_c0, na_c1e, na_k0, na_k1, nb_c0, nb_c1e, nb_n0, nb_n1b = self.get_dims_lengths()

        unmerge_sub_n = self.tunable.unmerge_sub_n
        if gemm_n_unmerge_cluster == 0:
            assert unmerge_sub_n % nb_n0 == 0, f"unmerge_sub_n:{unmerge_sub_n}, nb_n0:{nb_n0}"
            unmerge_sub_n1 = unmerge_sub_n // nb_n0
            assert nb_n1b % unmerge_sub_n1 == 0, f"nb_n1b:{nb_n1b}, unmerge_sub_n1:{unmerge_sub_n1}"
        elif gemm_n_unmerge_cluster == 1:
            assert cb_n0 == 1 and cb_n1b != 1 and tb_n0 != 1 and tb_n1b == 1, "current implementation only support this stratagy"
            unmerge_sub_n1 = unmerge_sub_n
        else:
            assert False, f"unsupported gemm_n_unmerge_cluster:{self.tunable.gemm_n_unmerge_cluster}"

        # c0*c1e is gemm_k for fwd, we do it for in/wei seperatedly
        #unmerge_sub_ta_c  = self.tunable.unmerge_sub_c
        #unmerge_sub_ta_c1 = unmerge_sub_ta_c // na_c0
        unmerge_sub_tb_c  = self.tunable.unmerge_sub_c
        unmerge_sub_tb_c1 = unmerge_sub_tb_c // nb_c0

        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        m_in_update_hw = self.get_macro_in_update_hw()
        m_in_update_os = self.get_macro_in_update_os()
        # m_wei_update_os   = self.get_macro_wei_update_os()
        # m_wei_update_yx   = self.get_macro_wei_update_yx()
        m_set_flag_hw     = self.get_macro_set_flag_hw()
        m_set_flag_c      = self.get_macro_set_flag_c()
        s_in_stride_d0, s_in_stride_d1, s_wei_stride_d0, s_wei_stride_d1 = self.get_symbol_global_load_s_stride_d0_d1()

        m_wei_2d_global_load, m_in_2d_global_load = self.get_macro_global_load()

        tc_index_dispatcher = igemm_thread_cluster_index_dispatcher_t(self.mc)
        tc_index_accumulator = igemm_thread_cluster_index_accumulator_t(self.mc)

        m_int_div_rem_vv = macro_int_div_rem_vv_t(self.mc)
        m_int_div_rem_vs = macro_int_div_rem_vs_t(self.mc)
        m_int_div_rem_ss = macro_int_div_rem_ss_t(self.mc)

        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            m_mdiv_u32_vs = macro_mdiv_u32_rem_vs_t(self.mc)
            m_mdiv_u32_ss = macro_mdiv_u32_rem_ss_t(self.mc)

        gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()
        s_dummy = sym_t("s_dummy")

        global_load_ta_order = IGEMM_FWD_GTC_GLOBAL_LOAD_TA_ORDER_M_K   # for fwd, it seems always K dimension first is better

        # start emit
        #self._emit(f"; unmerge_sub_k:{unmerge_sub_k}, unmerge_sub_k1:{unmerge_sub_k1}, unmerge_sub_n:{unmerge_sub_n}, unmerge_sub_n1:{unmerge_sub_n1}")
        self._emit(f"; gemm_m_unmerge_cluster:{gemm_m_unmerge_cluster}, gemm_n_unmerge_cluster:{gemm_n_unmerge_cluster}, gemm_k_unmerge_cluster:{gemm_k_unmerge_cluster}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_in((0,1))}],    s[{s.s_ka((0, 1))}],    0+{k.k_p_in()}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_wei((0,1))}],   s[{s.s_ka((0, 1))}],    0+{k.k_p_wei()}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_out((0,1))}],   s[{s.s_ka((0, 1))}],    0+{k.k_p_out()}")
        if self.tunable.nxe != 0:
            self._emit(f"s_load_dwordx8 s[{s.s_hi((0, 7))}],    s[{s.s_ka((0, 1))}],    0+{k.k_hi()}")
            self._emit(f"s_load_dwordx8 s[{s.s_stride_w((0, 7))}],    s[{s.s_ka((0, 1))}],    0+{k.k_stride_w()}")
        else:
            self._emit(f"s_load_dwordx4 s[{s.s_hi((0, 3))}],    s[{s.s_ka((0, 1))}],    0+{k.k_hi()}")
            self._emit(f"s_load_dword s[{s.s_c()}],    s[{s.s_ka((0, 1))}],    0+{k.k_c()}")
            self._emit(f"s_load_dword s[{s.s_group()}],    s[{s.s_ka((0, 1))}],    0+{k.k_group()}")

        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            self._emit(f"s_load_dwordx2 s[{s.s_magic_0((0, 1))}],  s[{s.s_ka((0, 1))}],  0+{k.k_magic_0()}")
            self._emit(f"s_load_dwordx2 s[{s.s_tmp((2, 3))}],  s[{s.s_ka((0, 1))}],  0+{k.k_magic_2()}")
            self._emit(f"s_load_dwordx2 s[{s.s_tmp((4, 5))}],  s[{s.s_ka((0, 1))}],  0+{k.k_magic_4()}")
            self._emit(f"s_load_dword s[{s.s_magic_6()}],  s[{s.s_ka((0, 1))}],  0+{k.k_magic_6()}")
            self._emit(f"s_load_dwordx2 s[{s.s_shift_pack_0((0, 1))}], s[{s.s_ka((0, 1))}],  0+{k.k_shift_pack_0()}")

        self._emit(f"; wei(c0, c1e, k0, k1) thread_lengths: {ta_c0}x{ta_c1e}x{ta_k0}x{ta_k1}, cluster_lengths:{ca_c0}x{ca_c1e}x{ca_k0}x{ca_k1}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        if global_load_ta_order == IGEMM_FWD_GTC_GLOBAL_LOAD_TA_ORDER_M_K:
            self._emit(tc_index_dispatcher(v.v_gtc_ta_ic1e(),   v.v_tmp(),  ca_c1e, ta_c1e))
            if ca_c0 != 1:
                self._emit(tc_index_dispatcher(v.v_gtc_ta_ic0(),    v.v_tmp(),  ca_c0,  ta_c0))
            else:
                self._emit(f"v_mov_b32 v[{v.v_gtc_ta_ic0()}], 0")

            self._emit(tc_index_dispatcher(v.v_gtc_ta_ik1(),    v.v_tmp(),  ca_k1,  ta_k1))
            self._emit(tc_index_dispatcher(v.v_gtc_ta_ik0(),    v.v_tmp(),  ca_k0,  ta_k0, True))

        else:
            self._emit(tc_index_dispatcher(v.v_gtc_ta_ik1(),    v.v_tmp(),  ca_k1,  ta_k1))
            self._emit(tc_index_dispatcher(v.v_gtc_ta_ik0(),    v.v_tmp(),  ca_k0,  ta_k0))
            if ca_c0 != 1:
                self._emit(tc_index_dispatcher(v.v_gtc_ta_ic1e(),   v.v_tmp(),  ca_c1e, ta_c1e))
                self._emit(tc_index_dispatcher(v.v_gtc_ta_ic0(),    v.v_tmp(),  ca_c0,  ta_c0,  True))
            else:
                self._emit(tc_index_dispatcher(v.v_gtc_ta_ic1e(),   v.v_tmp(),  ca_c1e, ta_c1e, True))
                self._emit(f"v_mov_b32 v[{v.v_gtc_ta_ic0()}], 0")
        # assert ta_c0 == 1 and ca_c0 == 1, "re-assert again to make sure for weight no copy in c0 dimension"
        self._emit_empty_line()

        self._emit(f"; in(c0, c1e, n0, n1b), thread_lengths: {tb_c0}x{tb_c1e}x{tb_n0}x{tb_n1b}, cluster_lengths:{cb_c0}x{cb_c1e}x{cb_n0}x{cb_n1b}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        self._emit(tc_index_dispatcher(v.v_gtc_tb_in1b(),   v.v_tmp(),  cb_n1b, tb_n1b))
        self._emit(tc_index_dispatcher(v.v_gtc_tb_in0(),    v.v_tmp(),  cb_n0,  tb_n0))
        self._emit(tc_index_dispatcher(v.v_gtc_tb_ic1e(),   v.v_tmp(),  cb_c1e, tb_c1e,  True))
        # self._emit(tc_index_dispatcher(v.v_gtc_tb_ic0(),    v.v_tmp(),  ca_c0,  ta_c0,  True))
        self._emit_empty_line()

        self._emit(f"s_mov_b32 s[{s.s_p_in(2)}], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_in(3)}], 0x27000")

        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit_empty_line()
        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            self._emit(f"s_mov_b32 s[{s.s_magic_2()}], s[{s.s_tmp(2)}]")
            self._emit(f"s_mov_b32 s[{s.s_magic_3()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_mov_b32 s[{s.s_magic_4()}], s[{s.s_tmp(4)}]")
            self._emit(f"s_mov_b32 s[{s.s_magic_5()}], s[{s.s_tmp(5)}]")
        self._emit(f"; calculate index")

        if self.tunable.nxe != 0:
            # stride for wei
            self._emit(f"s_mul_i32 s[{s.s_wei_stride_c()}], s[{s.s_y()}], s[{s.s_x()}]")
            self._emit(f"s_mul_i32 s[{s.s_wei_stride_k()}], s[{s.s_c()}], s[{s.s_wei_stride_c()}]")
            if ta_c0 != 1:
                self._emit(f"s_lshl_b32 s[{s.s_wei_stride_c1e()}], s[{s.s_wei_stride_c()}], {utility_log2(na_c1e)}")
            if ta_k0 != 1:
                self._emit(f"s_lshl_b32 s[{s.s_wei_stride_k0()}], s[{s.s_wei_stride_k()}], {utility_log2(na_k1)}")

            # stride for in
            self._emit(f"s_mul_i32 s[{s.s_in_stride_c()}], s[{s.s_hi()}], s[{s.s_wi()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_c()}], s[{s.s_group()}]")
            self._emit(f"s_mul_i32 s[{s.s_in_stride_n()}], s[{s.s_tmp()}], s[{s.s_in_stride_c()}]")
            if tb_c0 != 1:
                self._emit(f"s_lshl_b32 s[{s.s_in_stride_c0()}], s[{s.s_in_stride_c()}], {utility_log2(unmerge_sub_tb_c1)}")
            if tb_n0 != 1:
                if gemm_n_unmerge_cluster == 0:
                    self._emit(f"s_lshl_b32 s[{s.s_in_stride_n0()}], s[{s.s_in_stride_n()}], {utility_log2(unmerge_sub_n1)} ")
                else:
                    self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {utility_log2(nb_n0)}")
                    self._emit(f"s_mul_i32 s[{s.s_in_stride_n0()}], s[{s.s_in_stride_n()}], s[{s.s_tmp()}],")

            # stride for out
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_group()}], s[{s.s_k()}]")
            self._emit(f"s_mul_i32 s[{s.s_out_stride_k()}], s[{s.s_ho()}], s[{s.s_wo()}]")
            self._emit(f"s_mul_i32 s[{s.s_out_stride_n()}], s[{s.s_tmp()}], s[{s.s_out_stride_k()}]")
            if gemm_n_unmerge_cluster == 1:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {igemm_log2(nb_n0)}")
                self._emit(f"s_mul_i32 s[{s.s_out_stride_n0()}], s[{s.s_out_stride_n()}], s[{s.s_tmp()}]")

        else:
            # stride for wei
            if ta_k0 != 1:
                self._emit(f"s_lshl_b32 s[{s.s_wei_stride_k0()}], s[{s.s_c()}], {utility_log2(na_k1)}")

            self._emit(f"s_mul_i32 s[{s.s_stride_hw()}], s[{s.s_hi()}], s[{s.s_wi()}]")             # both in/out
            self._emit(f"s_mov_b32 s[{s.s_out_stride_k()}],       s[{s.s_stride_hw()}]")
            self._emit(f"s_mov_b32 s[{s.s_in_stride_c()}],       s[{s.s_stride_hw()}]")

            # stride for in
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_c()}], s[{s.s_group()}]")
            self._emit(f"s_mul_i32 s[{s.s_in_stride_n()}], s[{s.s_tmp()}], s[{s.s_stride_hw()}]")
            if tb_c0 != 1:
                self._emit(f"s_lshl_b32 s[{s.s_in_stride_c0()}], s[{s.s_in_stride_c()}], {utility_log2(unmerge_sub_tb_c1)}")
            if tb_n0 != 1:
                if gemm_n_unmerge_cluster == 0:
                    self._emit(f"s_lshl_b32 s[{s.s_in_stride_n0()}], s[{s.s_in_stride_n()}], {utility_log2(unmerge_sub_n1)} ")
                else:
                    self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {utility_log2(nb_n0)}")
                    self._emit(f"s_mul_i32 s[{s.s_in_stride_n0()}], s[{s.s_in_stride_n()}], s[{s.s_tmp()}]")

            # stride for out
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_group()}], s[{s.s_k()}]")
            self._emit(f"s_mul_i32 s[{s.s_out_stride_n()}], s[{s.s_tmp()}], s[{s.s_stride_hw()}]")
            if gemm_n_unmerge_cluster == 1:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {igemm_log2(nb_n0)}")
                self._emit(f"s_mul_i32 s[{s.s_out_stride_n0()}], s[{s.s_out_stride_n()}], s[{s.s_tmp()}]")

        # calculate batch split and accumulate the base pointer for input/output
        self._emit(f"s_mul_i32  s[{s.s_tmp(0)}], s[{s.s_n()}], s[{s.s_in_stride_n()}]")
        self._emit(f"s_mul_i32  s[{s.s_tmp(1)}], s[{s.s_n()}], s[{s.s_out_stride_n()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(4)}], s[{s.s_tmp(0)}], {igemm_log2(data_byte)}")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(5)}], s[{s.s_tmp(1)}], {igemm_log2(data_byte)}")

        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_by()}], s[{s.s_tmp(4)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_by()}], s[{s.s_tmp(4)}]")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], s[{s.s_tmp(1)}]")

        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_by()}], s[{s.s_tmp(5)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_by()}], s[{s.s_tmp(5)}]")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], s[{s.s_tmp(1)}]")

        # early init s_knum in case shifted
        if self.tunable.nxe != 0:
            self._emit(f"s_mul_i32 s[{s.s_knum()}], s[{s.s_wei_stride_c()}], s[{s.s_c()}]")
        else:
            self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_c()}]")

        # warp around the really dim_b length, in case pad
        if self.tunable.nxe != 0:
            self._emit(f"s_add_u32 s[{s.s_tmp()}], {self.tunable.nxb - 1}, s[{s.s_out_stride_k()}]")
            self._emit(f"s_lshr_b32 s[{s.s_tmp(1)}], s[{s.s_tmp()}], {igemm_log2(self.tunable.nxb)}")
            self._emit(f"s_lshl_b32 s[{s.s_dim_b()}], s[{s.s_tmp(1)}], {igemm_log2(self.tunable.nxb)}")

        # for gemm_m pad
        self._emit_empty_line()
        self._emit(f"; pad k if need")
        self._emit(f"s_add_u32 s[{s.s_tmp()}], {self.tunable.gemm_m_per_block - 1}, s[{s.s_k()}]")
        self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_m_per_block)}")
        self._emit(f"s_lshl_b32 s[{s.s_k_padded()}], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_m_per_block)}")

        self._emit_empty_line()
        self._emit(f"; gemm_m_per_block:{self.tunable.gemm_m_per_block}, gemm_n_per_block:{self.tunable.gemm_n_per_block}, source_access_order:{self.tunable.source_access_order}")

        # calculate group index
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_dim_b() if self.tunable.nxe != 0 else s.s_stride_hw()}], s[{s.s_n()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_tmp()}], s[{s.s_k_padded()}]")
        self._emit(f"s_lshr_b32 s[0], s[{s.s_tmp(1)}], {igemm_log2(self.tunable.gemm_m_per_block * self.tunable.gemm_n_per_block)}")
        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_1()}], 0x00080010 ; offset:16, width:8")
            self._emit(m_mdiv_u32_ss(s.s_tmp(4), s.s_block_gtc_ig(), s.s_bx(), s.s_magic_6(), s.s_tmp(3), '0', s.s_tmp()))
        else:
            self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_block_gtc_ig(), s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))

        # s.s_tmp(4)=> rem, gemm_m, gemm_n, s.s_block_gtc_ig()=> quo, group
        self._emit(f"s_mov_b32 s[{s.s_bx()}], s[{s.s_tmp(4)}]")

        if self.tunable.source_access_order == IGEMM_GTC_TUNABLE_SOURCE_ACCESS_ORDER_GEMM_M_GEMM_N:
            if self.tunable.nxe != 0:
                self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_dim_b()}], s[{s.s_n()}]")
            else:
                self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_stride_hw()}], s[{s.s_n()}]")

            self._emit(f"s_lshr_b32 s[0], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_n_per_block)}")
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080000 ; offset:0, width:8")
                self._emit(m_mdiv_u32_ss(s.s_tmp(4), s.s_tmp(5), s.s_bx(), s.s_magic_0(), s.s_tmp(3), '0', s.s_tmp()))
            else:
                self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_tmp(5), s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
        else:
            self._emit(f"s_lshr_b32 s[0], s[{s.s_k_padded()}], {igemm_log2(self.tunable.gemm_m_per_block)}")
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080000 ; offset:0, width:8")
                self._emit(m_mdiv_u32_ss(s.s_tmp(5), s.s_tmp(4), s.s_bx(), s.s_magic_0(), s.s_tmp(3), '0', s.s_tmp()))
            else:
                self._emit(m_int_div_rem_ss(s.s_tmp(5), s.s_tmp(4), s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))

        self._emit(f"; s_tmp+4:block_gtc_in, s_tmp+5:block_gtc_im")

        ## gemm_m_unmerge_cluster is always 0 
        self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ik()}], s[{s.s_tmp(5)}], {igemm_log2(self.tunable.gemm_m_per_block)}")

        if gemm_n_unmerge_cluster == 0:
            if self.tunable.nxe != 0:
                if unmerge_sub_n1 == 1:
                    self._emit(f"s_lshr_b32 s[0], s[{s.s_dim_b()}], {igemm_log2(nb_n1b)} ; total number of n1b")
                else:
                    if unmerge_sub_n1 == nb_n1b:
                        self._emit(f"s_mov_b32 s[0], s[{s.s_dim_b()}] ; total number of n1b")
                    else:
                        self._emit(f"s_lshr_b32 s[0], s[{s.s_dim_b()}], {igemm_log2(nb_n1b // unmerge_sub_n1)}  ; total number of n1b")
            else:
                if unmerge_sub_n1 == 1:
                    self._emit(f"s_lshr_b32 s[0], s[{s.s_stride_hw()}], {igemm_log2(nb_n1b)} ; total number of n1b")
                else:
                    if unmerge_sub_n1 == nb_n1b:
                        self._emit(f"s_mov_b32 s[0], s[{s.s_stride_hw()}] ; total number of n1b")
                    else:
                        self._emit(f"s_lshr_b32 s[0], s[{s.s_stride_hw()}], {igemm_log2(nb_n1b // unmerge_sub_n1)}  ; total number of n1b")
        else:
            if self.tunable.nxe != 0:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {igemm_log2(nb_n0)}")
                self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_dim_b()}], s[{s.s_tmp()}]")
                self._emit(f"s_lshr_b32 s[0], s[{s.s_tmp(1)}], {igemm_log2(nb_n1b)}")
            else:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {igemm_log2(nb_n0)}")
                self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_stride_hw()}], s[{s.s_tmp()}]")
                self._emit(f"s_lshr_b32 s[0], s[{s.s_tmp(1)}], {igemm_log2(nb_n1b)}")

        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080008 ; offset:8, width:8")
            self._emit(m_mdiv_u32_ss(s.s_block_gtc_in1b(), s.s_block_gtc_in0(), s.s_tmp(4), s.s_magic_1(), s.s_tmp(3), '0', s.s_tmp()))
        else:
            self._emit(m_int_div_rem_ss(s.s_block_gtc_in1b(), s.s_block_gtc_in0(), s.s_tmp(4), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
        if nb_n1b != 1:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_in1b()}], s[{s.s_block_gtc_in1b()}], {igemm_log2(nb_n1b)}")
        if nb_n0 != 1:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_in0()}], s[{s.s_block_gtc_in0()}], {igemm_log2(nb_n0)}")
        self._emit_empty_line()

        # in transform
        self._emit(f"; in c1e transform")
        if self.tunable.nxe != 0:
            if cb_c1e == 1:
                #assert False, "this is not wished and may introduce wrong machine code"
                # TODO: this case is indeed same as below. this should only be allowed in cases that 1x1 with stride/dilation
                if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                    self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080010 ; offset:16, width:8")
                    self._emit(m_mdiv_u32_vs(v.v_tmp(4), v.v_gtc_tb_ic1(), v.v_gtc_tb_ic1e(), s.s_magic_2(), s.s_tmp(3), s.s_wei_stride_c(), v.v_tmp()))
                    self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080018 ; offset:24, width:8")
                    self._emit(m_mdiv_u32_vs(v.v_in_ix(), v.v_in_iy(), v.v_tmp(4), s.s_magic_3(), s.s_tmp(3), s.s_x(), v.v_tmp()))
                else:
                    self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_tb_ic1(), v.v_gtc_tb_ic1e(), s.s_wei_stride_c(), v.v_tmp(), s.s_tmp()))
                    self._emit(m_int_div_rem_vs(v.v_in_ix(), v.v_in_iy(), v.v_tmp(4), s.s_x(), v.v_tmp(), s.s_tmp()))
            else:
                if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                    self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080010 ; offset:16, width:8")
                    self._emit(m_mdiv_u32_vs(v.v_tmp(4), v.v_gtc_tb_ic1(), v.v_gtc_tb_ic1e(), s.s_magic_2(), s.s_tmp(3), s.s_wei_stride_c(), v.v_tmp()))
                    self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080018 ; offset:24, width:8")
                    self._emit(m_mdiv_u32_vs(v.v_in_ix(), v.v_in_iy(), v.v_tmp(4), s.s_magic_3(), s.s_tmp(3), s.s_x(), v.v_tmp()))
                else:
                    self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_tb_ic1(), v.v_gtc_tb_ic1e(), s.s_wei_stride_c(), v.v_tmp(), s.s_tmp()))
                    self._emit(m_int_div_rem_vs(v.v_in_ix(), v.v_in_iy(), v.v_tmp(4), s.s_x(), v.v_tmp(), s.s_tmp()))
        else:
            # self._emit(f"v_mov_b32 v[{v.v_gtc_tb_ic1()}], v[{v.v_gtc_tb_ic1e()}]")
            pass


        self._emit(f"; in n1b transform")
        if cb_n1b == 1:
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_in1b()}]")
        else:
            self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_in1b()}], v[{v.v_gtc_tb_in1b()}]")
        if self.tunable.nxe != 0:
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_1()}], 0x00080000 ; offset:0, width:8")
                self._emit(m_mdiv_u32_vs(v.v_tmp(4), v.v_gtc_tb_in1(), v.v_tmp(5), s.s_magic_4(), s.s_tmp(3), s.s_dim_b(), v.v_tmp()))
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_1()}], 0x00080008 ; offset:8, width:8")
                self._emit(m_mdiv_u32_vs(v.v_in_iwo(), v.v_in_iho(), v.v_tmp(4), s.s_magic_5(), s.s_tmp(3), s.s_wo(), v.v_tmp()))
            else:
                self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_tb_in1(), v.v_tmp(5), s.s_dim_b(), v.v_tmp(), s.s_tmp()))
                self._emit(m_int_div_rem_vs(v.v_in_iwo(), v.v_in_iho(), v.v_tmp(4), s.s_wo(), v.v_tmp(), s.s_tmp()))
            self._emit(f"v_mul_lo_u32 v[{v.v_in_iho()}], s[{s.s_stride_h()}], v[{v.v_in_iho()}]")
            self._emit(f"v_sub_i32 v[{v.v_in_iho()}], v[{v.v_in_iho()}], s[{s.s_pad_h()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_in_iwo()}], s[{s.s_stride_w()}], v[{v.v_in_iwo()}]")
            self._emit(f"v_sub_i32 v[{v.v_in_iwo()}], v[{v.v_in_iwo()}], s[{s.s_pad_w()}]")
            self._emit(m_in_update_hw(v.v_in_ihi(), v.v_in_iwi(), v.v_in_iho(), v.v_in_iwo(), v.v_in_iy(), v.v_in_ix(), s.s_dilation_h(), s.s_dilation_w()))
            self._emit_empty_line()
        else:
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_1()}], 0x00080000 ; offset:0, width:8")
                self._emit(m_mdiv_u32_vs(v.v_tmp(4), v.v_gtc_tb_in1(), v.v_tmp(5), s.s_magic_4(), s.s_tmp(3), s.s_stride_hw(), v.v_tmp()))
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_1()}], 0x00080008 ; offset:8, width:8")
                self._emit(m_mdiv_u32_vs(v.v_in_iwi(), v.v_in_ihi(), v.v_tmp(4), s.s_magic_5(), s.s_tmp(3), s.s_wi(), v.v_tmp()))
            else:
                self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_tb_in1(), v.v_tmp(5), s.s_stride_hw(), v.v_tmp(), s.s_tmp()))
                self._emit(m_int_div_rem_vs(v.v_in_iwi(), v.v_in_ihi(),  v.v_tmp(4), s.s_wi(), v.v_tmp(), s.s_tmp()))

        self._emit(f"; calculate in offset")
        # compute group distance
        self._emit(f"s_mul_i32 s[{s.s_tmp(5)}], s[{s.s_c()}], s[{s.s_in_stride_c()}]")
        self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ig()}], s[{s.s_block_gtc_ig()}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(5)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(5)}]")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], s[{s.s_tmp(1)}]")
        if gemm_n_unmerge_cluster == 0:
            self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], s[{s.s_block_gtc_in0()}], {igemm_log2(unmerge_sub_n1 * data_byte)}")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_in_stride_n()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_in_stride_n()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
            self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], s[{s.s_tmp(1)}]")
        else:
            pass
        self._emit_empty_line()

        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_tb_ic1() if self.tunable.nxe != 0 else v.v_gtc_tb_ic1e()}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_in_stride_c()}], v[{v.v_tmp()}]")
        if gemm_n_unmerge_cluster == 0:
            self._emit(tc_index_accumulator(v.v_tmp(1), v.v_gtc_tb_in0(), v.v_gtc_tb_in1(), cb_n0, cb_n1b, 0, unmerge_sub_n1))
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_in_stride_n()}], v[{v.v_tmp(1)}]")
        else:
            # no in0
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_in_stride_n()}], v[{v.v_gtc_tb_in1()}]")

        if self.tunable.nxe != 0:
            self._emit(f"v_add_lshl_u32 v[{v.v_in_os_base()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
            self._emit(m_in_update_os(v.v_in_os(), v.v_in_os_base(), v.v_in_ihi(), v.v_in_iwi(), s.s_wi(), v.v_tmp()))
            self._emit(m_set_flag_hw(v.v_in_flag(), v.v_in_ihi(), v.v_in_iwi(), s.s_hi(), s.s_wi()))
            if self.tunable.tensor_b_cluster_lengths[0] == 1:
                self._emit(m_set_flag_c(v.v_in_flag(), v.v_gtc_tb_ic1(), s.s_c()))
        else:
            self._emit(f"v_add_lshl_u32 v[{v.v_tmp(4)}], v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
            self._emit(m_in_update_os(v.v_in_os(), v.v_tmp(4), v.v_in_ihi(), v.v_in_iwi(), s.s_wi(), v.v_tmp()))
        self._emit_empty_line()

        if self.in_thread_copy_ndim != 1:
            if s_in_stride_d0 != s_dummy:
                self._emit(self.try_shift_stride(s_in_stride_d0, igemm_log2(data_byte)))
        if s_in_stride_d1 != s_dummy:
            self._emit(self.try_shift_stride(s_in_stride_d1, igemm_log2(data_byte)))
        self._emit_empty_line()

        if self.tunable.precache_soffset:
            self._emit(m_in_2d_global_load.init_precache_soffset(s_in_stride_d0(), s_in_stride_d1(), s.s_in_offset(), s.s_tmp()))

        # load in
        self._emit(self.global_load_in())
        self._emit_empty_line()
        #self._emit(f"s_mov_b32 s[{s.s_p_wei(2)}], 0xffffffff")
	    # config weight range
        self._emit("; config for weight range")
        self._emit(f"s_mul_i32 s[{s.s_p_wei(2)}], s[{s.s_wei_stride_k() if self.tunable.nxe != 0 else s.s_c()}], s[{s.s_k()}]")
        self._emit(f"s_lshl_b32 s[{s.s_p_wei(2)}], s[{s.s_p_wei(2)}], {igemm_log2(data_byte)}")
        self._emit(f"s_mov_b32 s[{s.s_p_wei(3)}], 0x27000")

        self._emit(f"; calculate wei offset")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_k()}], s[{s.s_wei_stride_k() if self.tunable.nxe != 0 else s.s_c()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
        self._emit(f"s_add_u32 s[{s.s_p_wei()}], s[{s.s_p_wei()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_wei(1)}], s[{s.s_p_wei(1)}], s[{s.s_tmp(1)}]")
        #if self.tunable.nxe != 0:
        # one important thing is we let wei=k*c*y*x, c*y*x -> e, treat e as a single dimension
        self._emit(tc_index_accumulator(v.v_tmp(), v.v_gtc_ta_ik0(), v.v_gtc_ta_ik1(), ca_k0, ca_k1, na_k0, na_k1))
        self._emit(f"v_add_u32 v[{v.v_cur_k()}], s[{s.s_block_gtc_ik()}], v[{v.v_tmp()}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_k() if self.tunable.nxe != 0 else s.s_c()}], v[{v.v_cur_k()}]")
        if ca_c0 != 1:
            self._emit(tc_index_accumulator(v.v_tmp(2), v.v_gtc_ta_ic0(), v.v_gtc_ta_ic1e(), ca_c0, ca_c1e, na_c0, na_c1e))
            self._emit(f"v_add_lshl_u32 v[{v.v_wei_os()}], v[{v.v_tmp()}], v[{v.v_tmp(2)}], {igemm_log2(data_byte)}")
        else:
            self._emit(f"v_add_lshl_u32 v[{v.v_wei_os()}], v[{v.v_tmp()}], v[{v.v_gtc_ta_ic1e()}], {igemm_log2(data_byte)}")

            # self._emit(m_wei_update_os(v.v_wei_os(), v.v_wei_os_base(), v.v_wei_iy(), v.v_wei_ix(), s.s_x(), v.v_tmp()))
        #else:
        #    self._emit(tc_index_accumulator(v.v_tmp(), v.v_gtc_ik0(), v.v_gtc_ta_ik1(), ca_k0, ca_k1, na_k0, na_k1))
        #    self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_ik()}], v[{v.v_tmp()}]")
        #    self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_k()}], v[{v.v_tmp(5)}]")
        #    self._emit(f"v_add_lshl_u32 v[{v.v_wei_os()}], v[{v.v_tmp()}], v[{v.v_gtc_ic1e()}], {igemm_log2(data_byte)}")

        self._emit_empty_line()
        if self.wei_thread_copy_ndim != 1:
            if s_wei_stride_d0 != s_dummy:
                #self._emit(f"s_lshl_b32 s[{s_wei_stride_d0()}], s[{s_wei_stride_d0()}], {igemm_log2(data_byte)}")
                self._emit(self.try_shift_stride(s_wei_stride_d0, igemm_log2(data_byte)))
        if s_wei_stride_d1 != s_dummy:
            #self._emit(f"s_lshl_b32 s[{s_wei_stride_d1()}], s[{s_wei_stride_d1()}], {igemm_log2(data_byte)}")
            self._emit(self.try_shift_stride(s_wei_stride_d1, igemm_log2(data_byte)))
        self._emit_empty_line()

        if self.tunable.precache_soffset:
            self._emit(m_wei_2d_global_load.init_precache_soffset(s_wei_stride_d0(), s_wei_stride_d1(), s.s_wei_offset(), s.s_tmp()))
            
        self._emit(self.global_load_wei())
        self._emit_empty_line()


        if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], v0")
            self._emit(self.thread_mapping(v.v_gemm_in(), v.v_gemm_im(), v.v_tmp(5), v.v_tmp()))
        else:
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], v0")
            self._emit(self.xdlops_mapping.get_gemm_index_for_src_matrix(v.v_gemm_in(), v.v_gemm_im(), v.v_tmp(5), v.v_tmp()))
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], v0")
            self._emit(self.xdlops_mapping.get_gemm_index_for_dst_matrix(v.v_co_sst(), v.v_co_sld(), v.v_tmp(5), v.v_tmp()))

        self._emit(f"; LDS store, in: c0,c1e,n0,n1b: {tb_c0}x{tb_c1e}x{tb_n0}x{tb_n1b}, {cb_c0}x{cb_c1e}x{cb_n0}x{cb_n1b}, order:{gemm_n_order}")
        if gemm_n_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B:
            if cb_n1b == 1:
                # TODO: remove this path, not possible go here
                assert cb_n0 != 1
                self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(nb_n1b)},  v[{v.v_gtc_tb_in0()}]")
            else:
                if cb_n0 == 1:
                    self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_tb_in1b()}]")
                else:
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_tb_in0()}], {igemm_log2(nb_n1b)}, v[{v.v_gtc_tb_in1b()}]")
        else:
            assert tb_n0 != 1
            if cb_n1b == 1:
                # this is not prefered
                assert cb_n0 != 1
                self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_tb_in0()}]")
            else:
                if cb_n0 == 1:
                    self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(nb_n0)}, v[{v.v_gtc_tb_in1b()}]")
                else:
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_tb_in1b()}], {igemm_log2(nb_n0)}, v[{v.v_gtc_tb_in0()}]")

        if cb_c1e != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_tb_ic1e()}], {igemm_log2(nb_n0*nb_n1b)}, v[{v.v_tmp()}]")
        #if cb_c0 != 1:
        #    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_tb_ic0()}], {igemm_log2(nb_c1e*nb_n0*nb_n1b)}, v[{v.v_tmp()}]")

        self._emit(f"v_lshlrev_b32 v[{v.v_sst_b_os()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")
        self._emit(f"v_add_u32 v[{v.v_sst_b_os()}], {self.tunable.lds_a_np2}, v[{v.v_sst_b_os()}]")
        self._emit_empty_line()

        self._emit(f"; LDS store, wei: c0,c1e,c0,c1: {ta_c0}x{ta_c1e}x{ta_k0}x{ta_k1}, {ca_c0}x{ca_c1e}x{ca_k0}x{ca_k1}, order:{gemm_m_order}")
        if gemm_m_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1:
            if ca_k1 == 1:
                assert ca_k0 != 1
                self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(na_k1)}, v[{v.v_gtc_ta_ik0}]")
            else:
                if ca_k0 == 1:
                    self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_ta_ik1()}]")
                else:
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ta_ik0()}], {igemm_log2(na_k1)}, v[{v.v_gtc_ta_ik1()}]")
        else:
            if ca_k1 == 1:
                assert ca_k0 != 1
                self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_ta_ik0}]")
            else:
                if ca_k0 == 1:
                    self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(na_k0)}, v[{v.v_gtc_ta_ik1()}]")
                else:
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ta_ik1()}], {igemm_log2(na_k0)}, v[{v.v_gtc_ta_ik0()}]")

        if ca_c1e != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ta_ic1e()}], {igemm_log2(na_k0*na_k1)}, v[{v.v_tmp()}]")
        if ca_c0 != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ta_ic0()}], {igemm_log2(na_c1e*na_k0*na_k1)}, v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_sst_a_os()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")
        self._emit_empty_line()

        self._emit(f"; LDS load")
        self._emit(f"v_lshlrev_b32 v[{v.v_sld_b_os()}], {igemm_log2(data_byte)}, v[{v.v_gemm_in()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_sld_a_os()}], {igemm_log2(data_byte)}, v[{v.v_gemm_im()}]")
        self._emit(f"v_add_u32 v[{v.v_sld_b_os()}], {self.tunable.lds_a_np2}, v[{v.v_sld_b_os()}]")
        self._emit_empty_line()

        if self.tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            self._emit(f"v_mov_b32 v[{v.v_gemm_in()}], v[{v.v_co_sst()}]")
            self._emit(f"v_mov_b32 v[{v.v_gemm_im()}], v[{v.v_co_sld()}]")
        self._emit(self.coalescing_store.init_co_lds_offset(v.v_co_sst(), v.v_co_sld(), v.v_gemm_im(), v.v_gemm_in(), '0', v.v_tmp()))
        self._emit(self.coalescing_store.init_co_sub_m_index(v.v_co_sub_m_index(), '0', v.v_tmp()))
        self._emit(self.coalescing_store.init_co_sub_n_index(v.v_co_sub_n_index(), '0', v.v_tmp()))
        self._emit_empty_line()

        self._emit(f"; output offset")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_k()}], s[{s.s_out_stride_k()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], s[{s.s_tmp(1)}]")
        if gemm_n_unmerge_cluster == 0:
            self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], s[{s.s_block_gtc_in0()}], {igemm_log2(unmerge_sub_n1 * data_byte)}")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_out_stride_n()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_out_stride_n()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
            self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], s[{s.s_tmp(1)}]")
        else:
            pass
        self._emit_empty_line()
        self._emit(f"s_lshl_b32 s[{s.s_tmp()}+3], s[{s.s_block_gtc_ik()}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_out_stride_k()}], s[{s.s_tmp()}+3]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp()}+1], s[{s.s_out_stride_k()}], s[{s.s_tmp()}+3]")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out()}+1], s[{s.s_p_out()}+1], s[{s.s_tmp()}+1]")
        self._emit_empty_line()
        self._emit(f"; compute v_co_sub_n_index along n0 x n1b : {nb_n0}x{nb_n1b}")
        if gemm_n_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B:
            if nb_n1b != 1:
                self._emit(f"v_and_b32 v[{v.v_out_in1b()}], {nb_n1b - 1}, v[{v.v_co_sub_n_index()}]     ; => N1B")
                if nb_n0 != 1:
                    self._emit(f"v_lshrrev_b32 v[{v.v_out_in0()}], {igemm_log2(nb_n1b)}, v[{v.v_co_sub_n_index()}]  ; => N0")
            else:
                assert nb_n0 == self.tunable.block_size
                assert False, "un implemented, should rarely be used"
        else:
            if nb_n0 != 1:
                self._emit(f"v_and_b32 v[{v.v_out_in0()}], {nb_n0 - 1}, v[{v.v_co_sub_n_index()}]     ; => N0")
                if nb_n1b != 1:
                    self._emit(f"v_lshrrev_b32 v[{v.v_out_in1b()}], {igemm_log2(nb_n0)}, v[{v.v_co_sub_n_index()}]   ; => N1B")
                else:
                    assert False, "un implemented, should rarely be used"
            else:
                if nb_n1b != 1:
                    self._emit(f"v_mov_b32 v[{v.v_out_in1b()}], v[{v.v_co_sub_n_index()}]   ; => N1B")
                else:
                    assert False, "un implemented, should rarely be used"

        self._emit(f";   compute from n1b")
        self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_in1b()}], v[{v.v_out_in1b()}]")
        if self.tunable.nxe != 0:
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_1()}], 0x00080000 ; offset:0, width:8")
                self._emit(m_mdiv_u32_vs(v.v_tmp(4), v.v_out_in1(), v.v_tmp(5), s.s_magic_4(), s.s_tmp(3), s.s_dim_b(), v.v_tmp()))
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_1()}], 0x00080008 ; offset:8, width:8")
                self._emit(m_mdiv_u32_vs(v.v_out_iwo(), v.v_out_iho(), v.v_tmp(4), s.s_magic_5(), s.s_tmp(3), s.s_wo(), v.v_tmp()))
            else:
                self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_out_in1(), v.v_tmp(5), s.s_dim_b(), v.v_tmp(), s.s_tmp()))
                self._emit(m_int_div_rem_vs(v.v_out_iwo(), v.v_out_iho(), v.v_tmp(4), s.s_wo(), v.v_tmp(), s.s_tmp()))
            self._emit_empty_line()
        else:
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_1()}], 0x00080000 ; offset:0, width:8")
                self._emit(m_mdiv_u32_vs(v.v_tmp(4), v.v_out_in1(), v.v_tmp(5), s.s_magic_4(), s.s_tmp(3), s.s_stride_hw(), v.v_tmp()))
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_1()}], 0x00080008 ; offset:8, width:8")
                self._emit(m_mdiv_u32_vs(v.v_out_iwo(), v.v_out_iho(), v.v_tmp(4), s.s_magic_5(), s.s_tmp(3), s.s_wi(), v.v_tmp()))
            else:
                self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_out_in1(), v.v_tmp(5), s.s_stride_hw(), v.v_tmp(), s.s_tmp()))
                self._emit(m_int_div_rem_vs(v.v_out_iwo(), v.v_out_iho(), v.v_tmp(4), s.s_wi(), v.v_tmp(), s.s_tmp()))
            self._emit_empty_line()
        self._emit_empty_line()
        self._emit(f"; add in_in0, in_in1")
        if nb_n0 != 1:
            if gemm_n_unmerge_cluster == 0:
                self._emit(f"v_lshl_or_b32 v[{v.v_tmp(1)}], v[{v.v_out_in0()}], {igemm_log2(unmerge_sub_n1)}, v[{v.v_out_in1()}]")
                self._emit(f"v_mul_lo_u32 v[{v.v_out_os()}], s[{s.s_out_stride_n()}], v[{v.v_tmp(1)}]")
            else:
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_out_stride_n()}], v[{v.v_out_in1()}]")
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_out_stride_n0()}], v[{v.v_out_in0()}]")
                self._emit(f"v_add_u32 v[{v.v_out_os()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}]")
        else:
            self._emit(f"v_mul_lo_u32 v[{v.v_out_os()}], s[{s.s_out_stride_n()}], v[{v.v_out_in1()}]")

        self._emit(f"; add i_k")
        ## gemm_m_unmerge_cluster is always 0
        if gemm_m_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1:
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_out_stride_k()}], v[{v.v_co_sub_m_index()}]")
        else:
            if na_k0 == 1:
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_out_stride_k()}], v[{v.v_co_sub_m_index()}]")
            else:
                if na_k1 == 1:
                    self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_out_stride_k()}], v[{v.v_co_sub_m_index()}]")
                else:
                    self._emit(f"v_and_b32 v[{v.v_tmp()}], {na_k0 - 1}, v[{v.v_co_sub_m_index()}]        ; => k0")
                    self._emit(f"v_lshrrev_b32 v[{v.v_tmp(1)}], {igemm_log2(na_k0)}, v[{v.v_co_sub_m_index()}]       ; => k1")
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp(1)}], v[{v.v_tmp()}], {igemm_log2(na_k1)}, v[{v.v_tmp(1)}]")
                    self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_out_stride_k()}], v[{v.v_tmp(1)}]")

        self._emit(f"v_add_u32 v[{v.v_out_os()}], v[{v.v_out_os()}], v[{v.v_tmp()}]")
        self._emit(f"; add ho, wo")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_wo() if self.tunable.nxe != 0 else s.s_wi()}], v[{v.v_out_iho()}]")
        self._emit(f"v_add3_u32 v[{v.v_out_os()}], v[{v.v_out_os()}], v[{v.v_tmp(1)}], v[{v.v_out_iwo()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_out_os()}], {igemm_log2(data_byte)}, v[{v.v_out_os()}]")
        if self.tunable.nxe != 0:
            self._emit(m_set_flag_hw(v.v_out_flag(), v.v_out_iho(), v.v_out_iwo(), s.s_ho(), s.s_wo()))

        self._emit(f"; move slice stride")
        assert na_c0 * na_c1e == self.tunable.gemm_k_per_block and nb_c0 * nb_c1e == self.tunable.gemm_k_per_block

        if self.tunable.nxe != 0:
            assert na_c0 * na_c1e == nb_c0 * nb_c1e
            self._emit(f"s_mov_b32 s[{s.s_move_slice_k_c1e()}], {na_c0 * na_c1e}")
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080010 ; offset:16, width:8")
                self._emit(m_mdiv_u32_ss(s.s_tmp(4), s.s_move_slice_k_c1(), s.s_move_slice_k_c1e(), s.s_magic_2(), s.s_tmp(3), s.s_wei_stride_c(), s.s_tmp()))
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080018 ; offset:24, width:8")
                self._emit(m_mdiv_u32_ss(s.s_move_slice_k_x(), s.s_move_slice_k_y(), s.s_tmp(4), s.s_magic_3(), s.s_tmp(3), s.s_x(), s.s_tmp()))
            else:
                self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_move_slice_k_c1(), s.s_move_slice_k_c1e(), s.s_wei_stride_c(), v.v_tmp(4), v.v_tmp(), s.s_tmp()))
                self._emit(m_int_div_rem_ss(s.s_move_slice_k_x(), s.s_move_slice_k_y(), s.s_tmp(4), s.s_x(), v.v_tmp(4), v.v_tmp(), s.s_tmp()))
        else:
            assert na_c1e == nb_c1e
            #self._emit(f"s_mov_b32 s[{s.s_move_slice_k_c1()}], {nb_c1e}")
            self._emit(f"s_mov_b32 s[{s.s_move_slice_k_c1e()}], {nb_c1e}")
        self._emit_empty_line()

        m_move_slice_window_ta, m_move_slice_window_tb = self.get_macro_move_slice_window()

        if self.tunable.nxe != 0:
            # assert s.s_out_stride_k.label not in self.dict_shifted_stride and s.s_wei_stride_k.label not in self.dict_shifted_stride
            if s.s_in_stride_c.label not in self.dict_shifted_stride:
                self._emit(m_move_slice_window_tb.init_stride_c(s.s_in_stride_c(), s.s_in_stride_c_c1(),
                                                        s.s_in_stride_c_c0_c1_diff(), s.s_move_slice_k_c1()))
            else:
                self._emit(f"s_lshr_b32 s[{s.s_tmp(3)}], s[{s.s_in_stride_c()}], {utility_log2(data_byte)}")
                self._emit(m_move_slice_window_tb.init_stride_c(s.s_tmp(3), s.s_in_stride_c_c1(),
                                                        s.s_in_stride_c_c0_c1_diff(), s.s_move_slice_k_c1()))
        else:
            if self.is_1d_move_slice_k():
                self._emit(m_move_slice_window_tb.init_stride_c(s.s_stride_hw(), s.s_in_stride_c_c1(),  s.s_move_slice_k_c1e()))
            else:
                self._emit(m_move_slice_window_tb.init_stride_c(s.s_stride_hw(), s.s_in_stride_c_c1(), 
                                                        s.s_in_stride_c_c0_c1_diff(), s.s_move_slice_k_c1e()))


        if not self.is_1d_move_slice_k():
            self._emit(f"s_mov_b32 s[{s.s_gemm_k_num_c1()}], {unmerge_sub_tb_c1}")
        #if self.tunable.nxe != 0:
        #    self._emit(f"s_mul_i32 s[{s.s_knum()}], s[{s.s_wei_stride_c()}], s[{s.s_c()}]")
        #else:
        #    self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_c()}]")
        self._emit_empty_line()

        self._emit(self.try_shift_stride(s.s_in_stride_c_c1, igemm_log2(data_byte)))
        #self._emit(self.try_shift_stride(s.s_wei_stride_k_k1, igemm_log2(data_byte)))
        self._emit(self.try_shift_stride(s.s_in_stride_c_c0_c1_diff, igemm_log2(data_byte)))
        #self._emit(self.try_shift_stride(s.s_wei_stride_k_k0_k1_diff, igemm_log2(data_byte)))

        if self.tunable.nxe != 0:
            self._emit(self.try_shift_stride(s.s_in_stride_c, igemm_log2(data_byte)))
            self._emit(self.try_shift_stride(s.s_wei_stride_k, igemm_log2(data_byte)))
            self._emit(self.try_shift_stride(s.s_out_stride_k, igemm_log2(data_byte)))
        else:
            self._emit(self.try_shift_stride(s.s_in_stride_c, igemm_log2(data_byte)))
            self._emit(self.try_shift_stride(s.s_c, igemm_log2(data_byte)))
            self._emit(self.try_shift_stride(s.s_out_stride_k, igemm_log2(data_byte)))

        self._emit(self.try_shift_stride(s.s_move_slice_k_c1e, igemm_log2(data_byte)))
        self._emit(f"s_mov_b32 s[{s.s_p_out(2)}], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_out(3)}], 0x27000")


    def emit_kernel_fma_main_loop(self):
        s = self.sgpr
        v = self.vgpr
        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        m_move_slice_window_ta, m_move_slice_window_tb = self.get_macro_move_slice_window()
        def move_slice_window_b():
            if self.tunable.nxe != 0:
                m_in_update_os        = self.get_macro_in_update_os()
                m_in_update_hw        = self.get_macro_in_update_hw()
                m_set_flag_hw         = self.get_macro_set_flag_hw()
                m_set_flag_c          = self.get_macro_set_flag_c()
                with self._deferred_context():
                    self._emit(m_move_slice_window_tb(v.v_move_slice_k_ic1(), v.v_move_slice_k_iy(), v.v_move_slice_k_ix(), s.s_gemm_k_num_c1(), s.s_gemm_k_num_y(), s.s_gemm_k_num_x(),
                            s.s_move_slice_k_c1(), s.s_move_slice_k_y(), s.s_move_slice_k_x(), v.v_in_os_base(),
                            s.s_in_stride_c(), s.s_in_stride_c_c1(), s.s_in_stride_c_c0_c1_diff()))
                    self._emit(m_in_update_hw(v.v_in_ihi(), v.v_in_iwi(), v.v_in_iho(), v.v_in_iwo(), v.v_in_iy(), v.v_in_ix(), s.s_dilation_h(), s.s_dilation_w()))
                    self._emit(m_in_update_os(v.v_in_os(), v.v_in_os_base(), v.v_in_ihi(), v.v_in_iwi(), s.s_wi(), v.v_tmp()))
                    self._emit(m_set_flag_hw(v.v_in_flag(), v.v_in_ihi(), v.v_in_iwi(), s.s_hi(), s.s_wi()))
                    if self.tunable.tensor_b_cluster_lengths[0] == 1:
                        self._emit(m_set_flag_c(v.v_in_flag(), v.v_move_slice_k_ic1(), s.s_c()))
                return self._get_deferred()
            else:
                with self._deferred_context():
                    if self.is_1d_move_slice_k():
                        self._emit(m_move_slice_window_tb(v.v_in_os(), s.s_move_slice_k_c1e(), s.s_in_stride_c(), s.s_in_stride_c_c1()))
                    else:
                        self._emit(m_move_slice_window_tb(v.v_in_os(), v.v_move_slice_k_ic1(), s.s_gemm_k_num_c1(), 
                                s.s_move_slice_k_c1e(), s.s_in_stride_c(), s.s_in_stride_c_c1(), s.s_in_stride_c_c0_c1_diff()))
                return self._get_deferred()

        def move_slice_window_a():
            with self._deferred_context():
                self._emit(m_move_slice_window_ta(v.v_wei_os(), s.s_move_slice_k_c1e()))
            return self._get_deferred()

            # if self.tunable.nxe != 0:
            #     #m_wei_update_os   = self.get_macro_wei_update_os()
            #     #m_wei_update_yx   = self.get_macro_wei_update_yx()
            #     with self._deferred_context():
            #         self._emit(m_move_slice_window_ta(v.v_wei_os(), s.s_move_slice_k_c1e()))
            #         # self._emit(m_wei_update_yx(v.v_wei_iy(), v.v_wei_ix(), v.v_move_slice_k_idsy(), v.v_move_slice_k_idsx(), s.s_dtile_y(), s.s_dtile_x(), v.v_dtile_iy(), v.v_dtile_ix()))
            #         # self._emit(m_wei_update_os(v.v_wei_os(), v.v_wei_os_base(), v.v_wei_iy(), v.v_wei_ix(), s.s_x(), v.v_tmp()))
            #     return self._get_deferred()
            # else:
            #     with self._deferred_context():
            #         # we don't really need do anything for a, in nxe 0 case.
            #         pass
            #     return self._get_deferred()

        if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            fctrl                             = ctrl_fma_main_loop_t()
            fctrl.thread_m                    = self.tunable.thread_tile_m
            fctrl.thread_n                    = self.tunable.thread_tile_n
            fctrl.unroll_k                    = self.tunable.gemm_k_per_block
            fctrl.label_prefix                = self.name()
            fctrl.gemm_m_repeat               = self.tunable.gemm_m_repeat
            fctrl.gemm_m_level0_cluster       = self.tunable.gemm_m_level0_cluster
            fctrl.gemm_m_level1_cluster       = self.tunable.gemm_m_level1_cluster
            fctrl.gemm_n_repeat               = self.tunable.gemm_n_repeat
            fctrl.gemm_n_level0_cluster       = self.tunable.gemm_n_level0_cluster
            fctrl.gemm_n_level1_cluster       = self.tunable.gemm_n_level1_cluster
            fctrl.lds_single_size             = self.tunable.lds_single            # in byte, should be power of 2
            fctrl.lds_buffer_num              = self.tunable.lds_buffer_num

            # functor
            fctrl.global_load_a_functor       = self.global_load_wei
            fctrl.global_load_b_functor       = self.global_load_in
            fctrl.shared_store_a_functor      = self.shared_store_wei
            fctrl.shared_store_b_functor      = self.shared_store_in
            fctrl.shared_load_a_functor       = inst_ds_read_t(self.tunable.thread_sub_tile_m * data_byte)
            fctrl.shared_load_b_functor       = inst_ds_read_t(self.tunable.thread_sub_tile_n * data_byte)
            fctrl.move_slice_window_a_functor = move_slice_window_a
            fctrl.move_slice_window_b_functor = move_slice_window_b

            # sympol type
            fctrl.v_a                         = v.v_a
            fctrl.v_b                         = v.v_b
            fctrl.v_c                         = v.v_c
            fctrl.v_gld_a                     = v.v_gld_a
            fctrl.v_gld_b                     = v.v_gld_b
            fctrl.v_sld_a_os                  = v.v_sld_a_os
            fctrl.v_sld_b_os                  = v.v_sld_b_os
            fctrl.v_sst_a_os                  = v.v_sst_a_os
            fctrl.v_sst_b_os                  = v.v_sst_b_os
            fctrl.s_kitr                      = s.s_kitr
            fctrl.s_knum                      = s.s_knum

            fma_main_loop = fma_main_loop_t(self.mc, fctrl)
            fma_main_loop.emit()
        else:
            a = self.agpr
            fctrl                             = ctrl_mfma_main_loop_t()
            ctrl_xdlops_mapping               = get_ctrl_xdlops_mapping_from_wave_tile(self.tunable.gemm_m_per_block, self.tunable.gemm_n_per_block,
                                                                        self.tunable.wave_tile_m, self.tunable.wave_tile_n, self.tunable.wave_tile_k,
                                                                        self.tunable.wave_repeat_m, self.tunable.wave_repeat_n,
                                                                        self.tunable.wave_step_m, self.tunable.wave_step_n, self.tunable.block_size // AMDGPU_WAVE_SIZE,
                                                                        self.tunable.precision)
            fctrl.cxm                         = ctrl_xdlops_mapping
            fctrl.unroll_k                    = self.tunable.gemm_k_per_block
            fctrl.label_prefix                = self.name()
            fctrl.lds_single_size             = self.tunable.lds_single            # in byte, should be power of 2
            fctrl.lds_buffer_num              = self.tunable.lds_buffer_num
            fctrl.local_prefetch_num          = self.tunable.local_prefetch_num
            fctrl.interleave                  = self.tunable.fma_interleave

            # functor
            fctrl.global_load_a_functor       = self.global_load_wei
            fctrl.global_load_b_functor       = self.global_load_in
            fctrl.shared_store_a_functor      = self.shared_store_wei
            fctrl.shared_store_b_functor      = self.shared_store_in
            if ctrl_xdlops_mapping.wave_step_m == 1:
                fctrl.shared_load_a_functor   = inst_ds_read_t(data_byte)   # xdlops load from LDS always single load
            else:
                assert ctrl_xdlops_mapping.wave_step_m == 2, "currently only support wave_step_m is 2"
                fctrl.shared_load_a_functor   = inst_ds_read2_likely_accumulate_offset_t(self.mc, 2, data_byte, ctrl_xdlops_mapping.wave_tile_m * data_byte, sym_t(self.vgpr.v_tmp(4)))

            if ctrl_xdlops_mapping.wave_step_n == 1:
                fctrl.shared_load_b_functor   = inst_ds_read_t(data_byte)   # xdlops load from LDS always single load
            else:
                assert ctrl_xdlops_mapping.wave_step_n == 2, "currently only support wave_step_n is 2"
                fctrl.shared_load_b_functor   = inst_ds_read2_likely_accumulate_offset_t(self.mc, 2, data_byte, ctrl_xdlops_mapping.wave_tile_n * data_byte, sym_t(self.vgpr.v_tmp(5)))
            fctrl.move_slice_window_a_functor = move_slice_window_a
            fctrl.move_slice_window_b_functor = move_slice_window_b

            # sympol type
            fctrl.v_a                         = v.v_a
            fctrl.v_b                         = v.v_b
            fctrl.a_c                         = a.a_c
            fctrl.v_gld_a                     = v.v_gld_a
            fctrl.v_gld_b                     = v.v_gld_b
            fctrl.v_sld_a_os                  = v.v_sld_a_os
            fctrl.v_sld_b_os                  = v.v_sld_b_os
            fctrl.v_sst_a_os                  = v.v_sst_a_os
            fctrl.v_sst_b_os                  = v.v_sst_b_os
            fctrl.s_kitr                      = s.s_kitr
            fctrl.s_knum                      = s.s_knum

            mfma_main_loop = mfma_main_loop_t(self.mc, fctrl)
            mfma_main_loop.emit()


    def emit_kernel_epilogue(self):
        s = self.sgpr
        v = self.vgpr
        #label_out = f"L_{self.name()}_out"

        if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            # if self.tunable.nxe != 0:
            #     self._emit(self.coalescing_store(v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_in(), v.v_in_os(), None,
            #         s.s_in_stride_c0() if self.tunable.gemm_m_unmerge_cluster == 1 else None, s.s_in_stride_c(), s.s_tmp(), v.v_in_flag()))
            # else:
            #     self._emit(self.coalescing_store(v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_in(), v.v_in_os(), None,
            #         s.s_in_stride_c0() if self.tunable.gemm_m_unmerge_cluster == 1 else None, s.s_in_stride_c(), s.s_tmp()))
            pass
        else:
            a = self.agpr
            
            self._emit(self.coalescing_store(a.a_c(), v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_out(), v.v_out_os(), None,
                    None, s.s_out_stride_k(), s.s_tmp(), v.v_out_flag() if self.tunable.nxe != 0 else None, s.s_k(), v.v_cur_k(), s.s_block_gtc_ik(), v.v_co_sub_m_index(), v.v_tmp()))

        self._emit_front(f"{self.label_out}:")

    def emit_kernel_symbol(self):
        self.karg.emit()
        self._emit_empty_line()
        self.sgpr.emit()
        self._emit_empty_line()
        self.vgpr.emit()
        self._emit_empty_line()
        if self.tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            self.agpr.emit()
            self._emit_empty_line()

    def emit_kernel_header(self):
        kernel_name = self.name()
        self._emit('.text')
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
            self._emit('.globl {}'.format(kernel_name))
        self._emit('.p2align 8')
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
            self._emit('.type {},@function'.format(kernel_name))
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V2:
            self._emit('.amdgpu_hsa_kernel {}'.format(kernel_name))
        self._emit('{}:'.format(kernel_name))

    def emit_kernel_body(self):
        self.emit_kernel_prologue()
        self.emit_kernel_fma_main_loop()
        self.emit_kernel_epilogue()
    def emit_kernel_end(self):
        self._emit('s_endpgm')
    def emit_kernel_footer(self):
        self._emit_empty_line()

    def emit_kernel_amd_kernel_code_t(self):
        amd_kernel_code_t(self.mc, self.get_kernel_info()).emit()
