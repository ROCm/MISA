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


IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1 = 0
IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0 = 1
IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_N_C0_C1E = 4
IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_N_C1E_C0 = 5
IGEMM_WRW_GTC_DEBUG = 0


def _find_non_1_index_in_list(list_object):
    result_list = list()
    for idx, item in enumerate(list_object):
        assert type(item) is int
        if item != 1:
            result_list.append(idx)
    return result_list

class macro_igemm_wrw_gtc_in_update_os_t(macro_base_t):
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
        return '.v_wrw_gtc_in_update_os'
    def expr(self):
        self._emit(f"; from hi, wi, os_base, compute final offset")
        self._emit(f"v_mad_u32_u24 v[{self.v_tmp()}], s[{self.s_wi()}], v[{self.v_in_ihi()}], v[{self.v_in_iwi()}]")
        self._emit(f"v_lshl_add_u32 v[{self.v_in_os()}], v[{self.v_tmp()}], {igemm_log2(self.data_byte)}, v[{self.v_in_os_base()}]")

class macro_igemm_wrw_gtc_in_update_hw_t(macro_base_t):
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_in_ihi")
        self.declare_arg("v_in_iwi")
        self.declare_arg("v_out_iho")
        self.declare_arg("v_out_iwo")
        self.declare_arg("s_stride_h")
        self.declare_arg("s_stride_w")
        self.declare_arg("v_wei_iy")
        self.declare_arg("v_wei_ix")
        self.declare_arg("s_dilation_h")
        self.declare_arg("s_dilation_w")
        self.declare_arg("s_pad_h")
        self.declare_arg("s_pad_w")
        self.declare_arg("v_tmp")
    def name(self):
        return '.v_wrw_gtc_in_update_hw'
    
    def expr(self):
        self._emit(f"; ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h,   here make sure iy <- iy * s_dilation_h - s_pad_h before hand")
        self._emit(f"; iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w,   here make sure ix <- ix * s_dilation_w - s_pad_w before hand")
        self._emit(f"v_mul_lo_u32 v[{self.v_tmp()}], s[{self.s_stride_h()}], v[{self.v_out_iho()}]")
        self._emit(f"v_add_i32 v[{self.v_in_ihi()}], v[{self.v_tmp()}], v[{self.v_wei_iy()}]")
        self._emit(f"v_mul_lo_u32 v[{self.v_tmp(1)}], s[{self.s_stride_w()}], v[{self.v_out_iwo()}]")   
        self._emit(f"v_add_i32 v[{self.v_in_iwi()}], v[{self.v_tmp(1)}], v[{self.v_wei_ix()}]")


class macro_igemm_wrw_gtc_out_update_os_t(macro_base_t):
    def __init__(self, mc, data_byte, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.data_byte = data_byte
        self.declare_arg("v_out_os")
        self.declare_arg("v_out_os_base")
        self.declare_arg("v_out_iho")
        self.declare_arg("v_out_iwo")
        self.declare_arg("s_wo")
        self.declare_arg("v_tmp")
    def name(self):
        return '.v_wrw_gtc_out_update_os'

    def expr(self):
        self._emit(f"; from ho, wo, os_base, compute final offset")
        self._emit(f"v_mad_u32_u24 v[{self.v_tmp()}], s[{self.s_wo()}], v[{self.v_out_iho()}], v[{self.v_out_iwo()}]")
        self._emit(f"v_lshl_add_u32 v[{self.v_out_os()}], v[{self.v_tmp()}], {igemm_log2(self.data_byte)}, v[{self.v_out_os_base()}]")

class macro_igemm_wrw_gtc_out_update_hw_t(macro_base_t):
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_out_iho")
        self.declare_arg("v_out_iwo")
        self.declare_arg("v_out_d_iho")
        self.declare_arg("v_out_d_iwo")
    def name(self):
        return '.v_wrw_gtc_out_update_hw'
   
    def expr(self):
        self._emit(f"v_mov_b32 v[{self.v_out_iho()}], v[{self.v_out_d_iho()}]")
        self._emit(f"v_mov_b32 v[{self.v_out_iwo()}], v[{self.v_out_d_iwo()}]")

class macro_igemm_wrw_gtc_set_flag_hw(macro_base_t):
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


class macro_igemm_wrw_gtc_move_slice_window_n_dsho_dswo(macro_base_t):
    '''
    optimized move slice approach. 
    '''
    def __init__(self, mc, tunable, inline = False):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        macro_base_t.__init__(self, mc, inline)
        self.tunable = tunable
        self.declare_arg("v_move_slice_n_in1")
        self.declare_arg("v_move_slice_n_idsho")
        self.declare_arg("v_move_slice_n_idswo")
        self.declare_arg("s_gemm_k_num_n1")
        self.declare_arg("s_gemm_k_num_dsho")
        self.declare_arg("s_gemm_k_num_dswo")
        self.declare_arg("s_move_slice_n_n1")
        self.declare_arg("s_move_slice_n_dsho")
        self.declare_arg("s_move_slice_n_dswo")
        self.declare_arg("v_in_os_base")
        self.declare_arg("v_out_os_base")
        self.declare_arg("s_in_stride_n")
        self.declare_arg("s_out_stride_n")
        self.declare_arg("s_in_stride_n_n1")
        self.declare_arg("s_out_stride_n_n1")
        self.declare_arg("s_in_stride_n_n0_n1_diff")
        self.declare_arg("s_out_stride_n_n0_n1_diff")

    def name(self):
        return '.s_wrw_gtc_move_slice_window_n_dsho_dswo'

    def init_stride_n(self, s_in_stride_n, s_out_stride_n, s_in_stride_n_n1, s_out_stride_n_n1, s_in_stride_n_n0_n1_diff, s_out_stride_n_n0_n1_diff, s_move_slice_n_n1):
        '''
        s_in_stride_n, s_in_stride_n, s_move_slice_n_n1 is known value, want to compute other
        '''
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths
        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4
        assert len(t_ta) == 4 and len(t_tb) == 4

        c_n0, c_n1b, c_c0, c_c1e  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        _   ,     _, c_k0, c_k1   = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        t_n0, t_n1b, t_c0, t_c1e  = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        _   ,    _,  t_k0, t_k1   = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        n_n0, n_n1b = c_n0 * t_n0, c_n1b * t_n1b
        unmerge_sub_n = self.tunable.unmerge_sub_n
        assert unmerge_sub_n % n_n0 == 0
        unmerge_sub_n1 = unmerge_sub_n // n_n0
        assert n_n1b % unmerge_sub_n1 == 0

        assert n_n0 == 1, "TODO: currently only support such kind of 1d move slice"

        diff_n0_n1 = self.tunable.gemm_k_per_block - unmerge_sub_n1 # !!! the diff of 2 unmerged dimension (like K=K0*K1)

        with self._deferred_context():
            self._emit(f"s_mul_i32 s[{s_in_stride_n_n0_n1_diff}], {diff_n0_n1}, s[{s_in_stride_n}]")
            self._emit(f"s_mul_i32 s[{s_out_stride_n_n0_n1_diff}], {diff_n0_n1}, s[{s_out_stride_n}]")
            self._emit(f"s_mul_i32 s[{s_in_stride_n_n1}], s[{s_move_slice_n_n1}], s[{s_in_stride_n}]  ; might be 0 or larger")
            self._emit(f"s_mul_i32 s[{s_out_stride_n_n1}], s[{s_move_slice_n_n1}], s[{s_out_stride_n}]  ; might be 0 or larger")

        return self._get_deferred()

        '''
    This is indeed a multi-dimension add-carry operation
    e.g. if want to compute a 3d (merged) dimension index [iz, iy, ix], with dimension length of [nz, ny, nx] in each.
    suppose we want to add a specific value this merged dimension.
    1) if want to add 1, it is simple.
        ix += 1
        if ix >= nx:
            ix = 0
            iy += 1     # carry to iy
        if iy >= ny:
            iy = 0
            iz += 1     # carry to iz
        if iz >= nz:
            pass        # the final dimension indeed can be ignored
    
    2) if we want to add N
        # first, find out how many steps in each dimension needed to add
        stride_x = N % nx               # -> usually can store in sgpr
        stride_y = (N//nx) % ny         # -> usually can store in sgpr
        stride_z = (N//(nx*ny)) % nz    # -> usually can store in sgpr

        # then do the add-carry
        ix += stride_x
        if ix >= nx:
            ix -= nx    # ! note here, no longer set 0
            iy += 1     # carry to iy
        iy += stride_y
        if iy >= ny:
            iy -= ny    # ! note here, no longer set 0
            iz += 1     # carry to iz
        iz += stride_z
        if iz >= nz:
            pass        # the final dimension indeed can be ignored
    '''
    def expr(self):
        # n0, n1b is unmerge.  n1b is merged from n1, b
        self._emit(f"v_add_u32 v[{self.v_move_slice_n_idswo()}], s[{self.s_move_slice_n_dswo()}], v[{self.v_move_slice_n_idswo()}]")
        self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_gemm_k_num_dswo()}], v[{self.v_move_slice_n_idswo()}]")
        self._emit(f"v_subrev_u32 v[{self.v_move_slice_n_idswo()}], s[{self.s_gemm_k_num_dswo()}], v[{self.v_move_slice_n_idswo()}]")
        self._emit(f"v_add_u32 v[{self.v_move_slice_n_idsho()}], 1, v[{self.v_move_slice_n_idsho()}]")
        self._emit(f"s_mov_b64 exec, -1")
        self._emit_empty_line()
        self._emit(f"v_add_u32 v[{self.v_move_slice_n_idsho()}], s[{self.s_move_slice_n_dsho()}], v[{self.v_move_slice_n_idsho()}]")
        self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_gemm_k_num_dsho()}], v[{self.v_move_slice_n_idsho()}]")
        self._emit(f"v_subrev_u32 v[{self.v_move_slice_n_idsho()}], s[{self.s_gemm_k_num_dsho()}], v[{self.v_move_slice_n_idsho()}]")
        # self._emit(f"v_add_u32 v[{self.v_move_slice_n_in1()}], 1, v[{self.v_move_slice_n_in1()}]")
        self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_n()}], v[{self.v_in_os_base()}]")
        self._emit(f"v_add_u32 v[{self.v_out_os_base()}], s[{self.s_out_stride_n()}], v[{self.v_out_os_base()}]")
        self._emit(f"s_mov_b64 exec, -1")
        self._emit_empty_line()
        # self._emit(f"v_add_u32 v[{self.v_move_slice_n_in1()}], s[{self.s_move_slice_n_n1()}], v[{self.v_move_slice_n_in1()}]")
        self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_n_n1()}], v[{self.v_in_os_base()}]")
        self._emit(f"v_add_u32 v[{self.v_out_os_base()}], s[{self.s_out_stride_n_n1()}], v[{self.v_out_os_base()}]")
        # self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_gemm_k_num_n1()}], v[{self.v_move_slice_n_in1()}]")
        # self._emit(f"v_subrev_u32 v[{self.v_move_slice_n_in1()}], s[{self.s_gemm_k_num_n1()}], v[{self.v_move_slice_n_in1()}]")
        # self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_n_n0_n1_diff()}], v[{self.v_in_os_base()}]")
        # self._emit(f"v_add_u32 v[{self.v_out_os_base()}], s[{self.s_out_stride_n_n0_n1_diff()}], v[{self.v_out_os_base()}]")
        # self._emit(f"s_mov_b64 exec, -1")
        # self._emit_empty_line()



class macro_igemm_wrw_gtc_move_slice_window_n_dsho_dswo_check_last_dim(macro_base_t):
    '''
    when n1 carry, need check n0. used in pad image size
    '''
    def __init__(self, mc, tunable, inline = False):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        macro_base_t.__init__(self, mc, inline)
        self.tunable = tunable
        self.declare_arg("v_move_slice_n_in1")
        self.declare_arg("v_move_slice_n_idsho")
        self.declare_arg("v_move_slice_n_idswo")
        self.declare_arg("s_gemm_k_num_n1")
        self.declare_arg("s_gemm_k_num_dsho")
        self.declare_arg("s_gemm_k_num_dswo")
        self.declare_arg("s_move_slice_n_n1")
        self.declare_arg("s_move_slice_n_dsho")
        self.declare_arg("s_move_slice_n_dswo")
        self.declare_arg("v_in_os_base")
        self.declare_arg("v_out_os_base")
        self.declare_arg("s_in_stride_n")
        self.declare_arg("s_out_stride_n")
        self.declare_arg("s_in_stride_n_n1")
        self.declare_arg("s_out_stride_n_n1")
        self.declare_arg("s_in_stride_n_n0_n1_diff")
        self.declare_arg("s_out_stride_n_n0_n1_diff")
        self.declare_arg("v_move_slice_n_in0")
        self.declare_arg("v_tmp")
        self.declare_arg("s_sub_n")         # use real value after split k
        self.declare_arg("v_flag_n")        # if n0 out of range.

    def name(self):
        return '.s_wrw_gtc_move_slice_window_n_dsho_dswo_check_last_dim'

    def init_stride_n(self, s_in_stride_n, s_out_stride_n, s_in_stride_n_n1, s_out_stride_n_n1, s_in_stride_n_n0_n1_diff, s_out_stride_n_n0_n1_diff, s_move_slice_n_n1):
        '''
        s_in_stride_n, s_in_stride_n, s_move_slice_n_n1 is known value, want to compute other
        '''
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths
        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4
        assert len(t_ta) == 4 and len(t_tb) == 4

        c_n0, c_n1b, c_c0, c_c1e  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        _   ,     _, c_k0, c_k1   = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        t_n0, t_n1b, t_c0, t_c1e  = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        _   ,    _,  t_k0, t_k1   = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        n_n0, n_n1b = c_n0 * t_n0, c_n1b * t_n1b
        unmerge_sub_n = self.tunable.unmerge_sub_n
        assert unmerge_sub_n % n_n0 == 0
        unmerge_sub_n1 = unmerge_sub_n // n_n0
        assert n_n1b % unmerge_sub_n1 == 0

        diff_n0_n1 = self.tunable.gemm_k_per_block - unmerge_sub_n1 # !!! the diff of 2 unmerged dimension (like K=K0*K1)

        with self._deferred_context():
            self._emit(f"s_mul_i32 s[{s_in_stride_n_n0_n1_diff}], {diff_n0_n1}, s[{s_in_stride_n}]")
            self._emit(f"s_mul_i32 s[{s_out_stride_n_n0_n1_diff}], {diff_n0_n1}, s[{s_out_stride_n}]")
            self._emit(f"s_mul_i32 s[{s_in_stride_n_n1}], s[{s_move_slice_n_n1}], s[{s_in_stride_n}]  ; might be 0 or larger")
            self._emit(f"s_mul_i32 s[{s_out_stride_n_n1}], s[{s_move_slice_n_n1}], s[{s_out_stride_n}]  ; might be 0 or larger")

        return self._get_deferred()

    def expr(self):
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths
        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4
        assert len(t_ta) == 4 and len(t_tb) == 4
        c_n0, c_n1b, c_c0, c_c1e  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        _   ,     _, c_k0, c_k1   = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        t_n0, t_n1b, t_c0, t_c1e  = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        _   ,    _,  t_k0, t_k1   = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        n_n0, n_n1b = c_n0 * t_n0, c_n1b * t_n1b
        unmerge_sub_n = self.tunable.unmerge_sub_n
        assert unmerge_sub_n % n_n0 == 0
        unmerge_sub_n1 = unmerge_sub_n // n_n0
        assert n_n1b % unmerge_sub_n1 == 0

        tc_index_accumulator = igemm_thread_cluster_index_accumulator_t(self.mc)

        assert n_n0 == 1, "TODO: currently only support such kind of 1d move slice"

        # n0, n1b is unmerge.  n1b is merged from n1, b
        self._emit(f"v_add_u32 v[{self.v_move_slice_n_idswo()}], s[{self.s_move_slice_n_dswo()}], v[{self.v_move_slice_n_idswo()}]")
        self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_gemm_k_num_dswo()}], v[{self.v_move_slice_n_idswo()}]")
        self._emit(f"v_subrev_u32 v[{self.v_move_slice_n_idswo()}], s[{self.s_gemm_k_num_dswo()}], v[{self.v_move_slice_n_idswo()}]")
        self._emit(f"v_add_u32 v[{self.v_move_slice_n_idsho()}], 1, v[{self.v_move_slice_n_idsho()}]")
        self._emit(f"s_mov_b64 exec, -1")
        self._emit_empty_line()
        self._emit(f"v_add_u32 v[{self.v_move_slice_n_idsho()}], s[{self.s_move_slice_n_dsho()}], v[{self.v_move_slice_n_idsho()}]")
        self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_gemm_k_num_dsho()}], v[{self.v_move_slice_n_idsho()}]")
        self._emit(f"v_subrev_u32 v[{self.v_move_slice_n_idsho()}], s[{self.s_gemm_k_num_dsho()}], v[{self.v_move_slice_n_idsho()}]")
        self._emit(f"v_add_u32 v[{self.v_move_slice_n_in1()}], 1, v[{self.v_move_slice_n_in1()}]")
        self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_n()}], v[{self.v_in_os_base()}]")
        self._emit(f"v_add_u32 v[{self.v_out_os_base()}], s[{self.s_out_stride_n()}], v[{self.v_out_os_base()}]")
        self._emit(f"s_mov_b64 exec, -1")
        self._emit_empty_line()
        self._emit(f"v_add_u32 v[{self.v_move_slice_n_in1()}], s[{self.s_move_slice_n_n1()}], v[{self.v_move_slice_n_in1()}]")
        self._emit(f"v_cmpx_gt_u32 vcc, s[{self.s_sub_n()}], v[{self.v_move_slice_n_in1()}]")
        self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_n_n1()}], v[{self.v_in_os_base()}]")
        self._emit(f"v_add_u32 v[{self.v_out_os_base()}], s[{self.s_out_stride_n_n1()}], v[{self.v_out_os_base()}]")
        #self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_n_n1()}], v[{self.v_in_os_base()}]")
        #self._emit(f"v_add_u32 v[{self.v_out_os_base()}], s[{self.s_out_stride_n_n1()}], v[{self.v_out_os_base()}]")
        #self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_gemm_k_num_n1()}], v[{self.v_move_slice_n_in1()}]")
        # self._emit(f"v_subrev_u32 v[{self.v_move_slice_n_in1()}], s[{self.s_gemm_k_num_n1()}], v[{self.v_move_slice_n_in1()}]")
        # self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_n_n0_n1_diff()}], v[{self.v_in_os_base()}]")
        # self._emit(f"v_add_u32 v[{self.v_out_os_base()}], s[{self.s_out_stride_n_n0_n1_diff()}], v[{self.v_out_os_base()}]")
        # self._emit(f"v_add_u32 v[{self.v_move_slice_n_in0()}], 1, v[{self.v_move_slice_n_in0()}]")  # to check n0
        # self._emit(f"v_mov_b32 v[{self.v_flag_n()}], 0")
        #self._emit(f"v_cndmask_b32 v[{self.v_flag_n()}], 0, 1, vcc")
        self._emit(f"s_mov_b64 exec, -1")
        self._emit(f"v_cndmask_b32 v[{self.v_flag_n()}], 0, 1, vcc")
        # self._emit_empty_line()
        # self._emit(tc_index_accumulator(self.v_tmp(), self.v_move_slice_n_in0(), self.v_move_slice_n_in1(), c_n0, c_n1b, 0, unmerge_sub_n1))
        # self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_n()}], v[{self.v_tmp()}]")
        # self._emit(f"v_mov_b32 v[{self.v_flag_n()}], 0")
        # self._emit(f"s_mov_b64 exec, -1")
        # self._emit_empty_line()


class igemm_wrw_gtc_t(mc_base_t):
    '''
    k -> k0, k1
    c -> c0, c1
    n -> n0, n1
    ho, wo -> b
    x, y -> e

    gemm_m -> k0*k1
    gemm_k -> n0*n1b
    gemm_n -> c0*c1e

    tensor a: n0*n1b*k0*k1
    tensor b: n0*n1b*c0*c1e

              thread_lengths            cluster_lengths
    tensor a: t_n0*t_n1b*t_k0*t_k1      c_n0*c_n1b*c_k0*c_k1
    tensor b: t_n0*t_n1b*t_c0*t_c1e     c_n0*c_n1b*c_c0*c_c1e

                      tensor a                      tensor b
    thread_lengths  : t_n0, t_n1b, t_k0, t_k1   t_n0, t_n1b, t_c0, t_c1e
    cluster_lengths : c_n0, c_n1b, c_k0, c_k1   c_n0, c_n1b, c_c0, c_c1e

    for the n1b, c1e thread_lengths no longer check per thread stride in n1*b or k1*e
    but cluster lengths will check.

    '''
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
        self.global_load_in = self.global_load_in_t(mc, self)
        self.global_load_out = self.global_load_out_t(mc, self)
        self.shared_store_in = self.shared_store_in_t(mc, self)
        self.shared_store_out = self.shared_store_out_t(mc, self)

        in_thread_copy_index, out_thread_copy_index = self.get_thread_copy_index()
        self.in_thread_copy_ndim = len(in_thread_copy_index)
        self.out_thread_copy_ndim = len(out_thread_copy_index)
        assert self.in_thread_copy_ndim in (0, 1, 2)
        assert self.out_thread_copy_ndim in (0, 1, 2)

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
            ctrl_coalescing_store.precision = self.tunable.precision

            ctrl_coalescing_store.vector_write_out = 1                      # TODO: some cases this can be set to other value
            ctrl_coalescing_store.block_size = self.tunable.block_size

            gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()
            n_c0, n_c1, n_k0, n_k1e, n_n0, n_n1b = self.get_dims_lengths()
            ctrl_coalescing_store.gemm_m_m0_m1 = [n_c0, n_c1]
            if gemm_m_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0:
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
            ctrl_coalescing_store_xdlops.gemm_k_global_split = self.tunable.gemm_k_global_split
            ctrl_coalescing_store_xdlops.coalescing_groups = self.coalescing_store_groups
            ctrl_coalescing_store_xdlops.precision = self.tunable.precision

            ctrl_coalescing_store_xdlops.vector_write_out = 1                      # TODO: some cases this can be set to other value
            ctrl_coalescing_store_xdlops.block_size = self.tunable.block_size
        
            gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()
            n_c0, n_c1, n_k0, n_k1e, n_n0, n_n1b = self.get_dims_lengths()
            ctrl_coalescing_store_xdlops.gemm_m_m0_m1 = [n_c0, n_c1]
            if gemm_m_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0:
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

        t_c0, t_c1, t_k0, t_k1e, t_n0, t_n1b = self.get_thread_lengths()

        gemm_n_order = IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_N_C0_C1E
        if self.tunable.allow_lds_reorder:
            if need_reverse_order(t_n0, t_n1b):
                gemm_n_order = IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_N_C1E_C0

        gemm_m_order = IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1
        if self.tunable.allow_lds_reorder:
            if need_reverse_order(t_c0, t_c1):
                gemm_m_order = IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0

        return gemm_m_order, gemm_n_order

    class global_load_in_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_in_2d_global_load, m_out_2d_global_load = self.outer.get_macro_global_load()
            return m_in_2d_global_load.get_issues()

        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr

            m_in_2d_global_load, m_out_2d_global_load = self.outer.get_macro_global_load()
            s_in_stride_d0, s_in_stride_d1, s_out_stride_d0, s_out_stride_d1 = self.outer.get_symbol_global_load_s_stride_d0_d1()
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

    class global_load_out_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_in_2d_global_load, m_out_2d_global_load = self.outer.get_macro_global_load()
            return m_out_2d_global_load.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr

            m_in_2d_global_load, m_out_2d_global_load = self.outer.get_macro_global_load()
            s_in_stride_d0, s_in_stride_d1, s_out_stride_d0, s_out_stride_d1 = self.outer.get_symbol_global_load_s_stride_d0_d1()
            with self._deferred_context():
                self._emit(f"; load output")
                if self.outer.tunable.nxe != 0:
                    self._emit(f".v_clear_nc {v.v_gld_a()}, {m_out_2d_global_load.ctrl.length_d0 * m_out_2d_global_load.ctrl.length_d1}")
                    self._emit(f"v_cmp_eq_u32 vcc, 1, v[{v.v_out_flag()}]")
                    self._emit(f"s_and_saveexec_b64 s[{s.s_tmp((4, 5))}], vcc")
                if self.outer.tunable.precache_soffset:
                    self._emit(m_out_2d_global_load(v.v_gld_a(), s.s_p_out(), v.v_out_os(), s_out_stride_d0(), s_out_stride_d1(), s.s_out_offset()))
                else:
                    self._emit(m_out_2d_global_load(v.v_gld_a(), s.s_p_out(), v.v_out_os(), s_out_stride_d0(), s_out_stride_d1(), s.s_tmp()))
                if self.outer.tunable.nxe != 0:
                    self._emit(f"s_or_b64 exec, exec, s[{s.s_tmp((4, 5))}]")
            return self._get_deferred() 

    class shared_store_in_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_in_2d_shared_store, m_out_2d_shared_store = self.outer.get_macro_shared_store()
            return  m_in_2d_shared_store.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr
            m_in_2d_shared_store, m_out_2d_shared_store = self.outer.get_macro_shared_store()
            with self._deferred_context():
                self._emit(m_in_2d_shared_store(v.v_gld_b(), v.v_sst_b_os()))
            return self._get_deferred()

    class shared_store_out_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_in_2d_shared_store, m_out_2d_shared_store = self.outer.get_macro_shared_store()
            return m_out_2d_shared_store.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr
            m_in_2d_shared_store, m_out_2d_shared_store = self.outer.get_macro_shared_store()
            with self._deferred_context():
                self._emit(m_out_2d_shared_store(v.v_gld_a(), v.v_sst_a_os()))
            return self._get_deferred()

    class kernel_karg_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer         = outer

            self.k_p_in          = sym_t("k_p_in",          0)
            self.k_p_wei         = sym_t("k_p_wei",         8)
            self.k_p_out         = sym_t("k_p_out",         16)
            self.k_hi            = sym_t("k_hi",            24)
            self.k_wi            = sym_t("k_wi",            28)
            self.k_n             = sym_t("k_n",             32)
            self.k_k             = sym_t("k_k",             36)
            self.k_c             = sym_t("k_c",             40)
            self.k_ho            = sym_t("k_ho",            44)
            self.k_wo            = sym_t("k_wo",            48)
            self.k_stride_h      = sym_t("k_stride_h",      52)
            self.k_stride_w      = sym_t("k_stride_w",      56)
            self.k_dilation_h    = sym_t("k_dilation_h",    60)
            self.k_dilation_w    = sym_t("k_dilation_w",    64)
            self.k_pad_h         = sym_t("k_pad_h",         68)
            self.k_pad_w         = sym_t("k_pad_w",         72)
            self.k_y             = sym_t("k_y",             76)
            self.k_x             = sym_t("k_x",             80)
            self.k_gemm_k_global_split  = sym_t("k_gemm_k_global_split",  84)
            self.k_group         = sym_t("k_group",         88)
            self.k_pack_0        = sym_t("k_pack_0",        92)
            self.k_end           = sym_t("k_end",           96)

        def get_count(self):
            return self.k_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('k_'):
                    self._emit(v.declare())

    class kernel_sgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer

            self.s_ka                      = sym_t("s_ka"                     ,0)
            self.s_bx                      = sym_t("s_bx"                     ,2)
            self.s_p_in                    = sym_t("s_p_in"                   ,4)
            self.s_p_wei                   = sym_t("s_p_wei"                  ,8)
            self.s_p_out                   = sym_t("s_p_out"                  ,12)
            self.s_hi                      = sym_t("s_hi"                     ,16)
            self.s_wi                      = sym_t("s_wi"                     ,17)
            self.s_n                       = sym_t("s_n"                      ,18)
            self.s_k                       = sym_t("s_k"                      ,19)
            self.s_c                       = sym_t("s_c"                      ,20)
            #if outer.tunable.nxe != 0:
            self.s_ho                      = sym_t("s_ho"                     ,21)
            self.s_wo                      = sym_t("s_wo"                     ,22)
            self.s_stride_h                = sym_t("s_stride_h"               ,23)
            self.s_stride_w                = sym_t("s_stride_w"               ,24)
            self.s_dilation_h              = sym_t("s_dilation_h"             ,25)
            self.s_dilation_w              = sym_t("s_dilation_w"             ,26)
            self.s_pad_h                   = sym_t("s_pad_h"                  ,27)
            self.s_pad_w                   = sym_t("s_pad_w"                  ,28)
            self.s_y                       = sym_t("s_y"                      ,29)
            self.s_x                       = sym_t("s_x"                      ,30)
            sseq                           = gpr_sequencer_t(30 + 1)
            self.s_gemmk_split             = sym_t("s_gemmk_split"           ,sseq(1))
            self.s_group                   = sym_t("s_group"                 ,sseq(1))

            self.s_out_stride_k            = sym_t("s_out_stride_k"           ,sseq(1))
            #if outer.tunable.nxe == 0:
            #    self.s_stride_hw           = sym_t("s_stride_hw"              ,sseq(1))
            self.s_out_stride_k0           = sym_t("s_out_stride_k0"          ,sseq(1))
            self.s_out_stride_n            = sym_t("s_out_stride_n"           ,sseq(1))
            self.s_out_stride_n0           = sym_t("s_out_stride_n0"          ,sseq(1))

            #if outer.tunable.gemm_m_unmerge_cluster == 1:
            self.s_in_stride_c0            = sym_t("s_in_stride_c0"           ,sseq(1))
            self.s_in_stride_c             = sym_t("s_in_stride_c"            ,sseq(1))
            #if outer.tunable.gemm_n_unmerge_cluster == 1:
            self.s_in_stride_n0            = sym_t("s_in_stride_n0"           ,sseq(1))
            self.s_in_stride_n             = sym_t("s_in_stride_n"            ,sseq(1))

            self.s_wei_stride_c            = sym_t("s_wei_stride_c"           ,sseq(1))
            if outer.tunable.gemm_n_unmerge_cluster == 1:
                self.s_wei_stride_c0       = sym_t("s_wei_stride_c0"          ,sseq(1))
            self.s_wei_stride_k            = sym_t("s_wei_stride_k"           ,sseq(1))
            if outer.tunable.gemm_n_unmerge_cluster == 1:
                self.s_wei_stride_k0       = sym_t("s_wei_stride_k0"          ,sseq(1))

            #if outer.tunable.nxe != 0:
                #self.s_stride_dslice_hw    = sym_t("s_stride_dslice_hw"       ,sseq(1))
                #self.s_stride_dslice_yx    = sym_t("s_stride_dslice_yx"       ,sseq(1))

            #if outer.tunable.nxe != 0:
            self.s_out_stride_n_n1         = sym_t("s_out_stride_n_n1"        ,sseq(1))
            self.s_out_stride_n_n0_n1_diff = sym_t("s_out_stride_n_n0_n1_diff",sseq(1))
            self.s_in_stride_n_n1          = sym_t("s_in_stride_n_n1"        ,sseq(1))
            self.s_in_stride_n_n0_n1_diff  = sym_t("s_in_stride_n_n0_n1_diff",sseq(1))

            self.s_move_slice_n_n1         = sym_t("s_move_slice_n_n1"        ,sseq(1))

            self.s_move_slice_n_dsho       = sym_t("s_move_slice_n_dsho"           ,sseq(1))
            self.s_move_slice_n_dswo       = sym_t("s_move_slice_n_dswo"           ,sseq(1))

            if outer.tunable.nxe != 0:
                self.s_dim_b               = sym_t("s_dim_b"                   ,sseq(1))

            self.s_block_gtc_ie            = sym_t("s_block_gtc_ie"           ,sseq(1))
            self.s_block_gtc_ik            = sym_t("s_block_gtc_ik"           ,sseq(1))
            self.s_block_gtc_ic0           = sym_t("s_block_gtc_ic0"          ,sseq(1))
            self.s_block_gtc_ic1e          = sym_t("s_block_gtc_ic1e"         ,sseq(1))
            self.s_block_gtc_in            = sym_t('s_block_gtc_in'           ,sseq(1))
            self.s_block_gtc_ig            = sym_t('s_block_gtc_ig'           ,sseq(1))

            self.s_knum                    = sym_t("s_knum"                   ,1)
            self.s_gemm_k_num_n1           = sym_t("s_gemm_k_num_n1"          ,0)
            #if outer.tunable.nxe != 0:
            self.s_gemm_k_num_dsho     = sym_t("s_gemm_k_num_dsho"         ,sseq(1))
            self.s_gemm_k_num_dswo     = sym_t("s_gemm_k_num_dswo"         ,sseq(1))

            self.s_kitr                    = sym_t("s_kitr"                   ,3)
            if outer.tunable.precache_soffset:
                m_out_2d_global_load, m_wei_2d_global_load = outer.get_macro_global_load()
                out_npc = m_out_2d_global_load.get_num_precache_soffset()
                wei_npc = m_wei_2d_global_load.get_num_precache_soffset()
                self.s_in_offset          = sym_t("s_in_offset"             ,sseq(out_npc))   # if this number is zero, it is also OK, since we would not use
                self.s_out_offset          = sym_t("s_out_offset"             ,sseq(wei_npc))
            self.s_sub_n                   = sym_t("s_sub_n"                  ,sseq(1))
            # self.s_group_left              = sym_t("s_group_left"             ,self.s_knum.value)
            if IGEMM_WRW_GTC_DEBUG == 1:
                self.s_dbg                     = sym_t("s_dbg"                    ,sseq(2, 2))
            self.s_k_padded                = sym_t("s_k_padded"             ,sseq(1))
            self.s_tmp                     = sym_t("s_tmp"                    ,sseq(6, 2))
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
            is_vgpr_acc_c = outer.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS
            vseq = gpr_sequencer_t()
            self.outer               = outer
            if is_vgpr_acc_c:
                self.v_c             = sym_t("v_c"            ,vseq(outer.tunable.num_vgpr_accumulate_c))
                v_c_num              = vseq()
            else:
                v_c_resuable_num     = outer.tunable.num_vgpr_accumulate_a + outer.tunable.num_vgpr_accumulate_b + \
                                        outer.tunable.num_global_load_a + outer.tunable.num_global_load_b + \
                                        8       # from v_sst_a_os to v_wei_os
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
            self.v_in_ihi            = sym_t("v_in_ihi"       ,vseq(1))
            self.v_in_iwi            = sym_t("v_in_iwi"       ,vseq(1))
            #if outer.tunable.nxe != 0:
            #    self.v_out_dslice_ih     = sym_t("v_out_dslice_ih",vseq(1))
            #    self.v_out_dslice_iw     = sym_t("v_out_dslice_iw",vseq(1))
            self.v_in_os            = sym_t("v_in_os"       ,vseq(1))
            #if outer.tunable.nxe != 0:
            self.v_in_os_base       = sym_t("v_in_os_base"  ,vseq(1))
            #if outer.tunable.nxe != 0:
            self.v_out_iho            = sym_t("v_out_iho"       ,vseq(1))
            self.v_out_iwo            = sym_t("v_out_iwo"       ,vseq(1))
            self.v_out_os            = sym_t("v_out_os"       ,vseq(1))
            #if outer.tunable.nxe != 0:
            self.v_out_os_base       = sym_t("v_out_os_base"  ,vseq(1))
            if outer.tunable.nxe != 0:
                self.v_in_flag       = sym_t("v_in_flag"      ,vseq(1))
                self.v_out_flag      = sym_t("v_out_flag"     ,vseq(1))
            self.v_co_sst            = sym_t("v_co_sst"       ,vseq(1))
            self.v_co_sld            = sym_t("v_co_sld"       ,vseq(1))
            #if outer.tunable.nxe != 0:
            self.v_wei_flag       = sym_t("v_wei_flag"      ,vseq(1))
            self.v_wei_os             = sym_t("v_wei_os"        ,vseq(1))
            self.v_gtc_ik1           = sym_t("v_gtc_ik1"      ,vseq(1))
            #if outer.tunable.nxe != 0:
            #    self.v_gtc_dslice_iy = sym_t("v_gtc_dslice_iy",vseq(1))
            #    self.v_gtc_dslice_ix = sym_t("v_gtc_dslice_ix",vseq(1))
            if outer.tunable.nxe != 0:
                self.v_move_slice_n_in0  = sym_t("v_move_slice_n_in0" , vseq(1))  # only used in pad image size
                self.v_flag_n            = sym_t("v_flag_n" , vseq(1))  # only used in pad image size
            self.v_move_slice_n_in1  = sym_t("v_move_slice_n_in1" , self.v_wei_flag.value)
            #if outer.tunable.nxe != 0:
            self.v_move_slice_n_idsho = sym_t("v_move_slice_n_idsho", vseq(1))
            self.v_move_slice_n_idswo = sym_t("v_move_slice_n_idswo", vseq(1))

            self.v_wei_iy        = sym_t("v_wei_iy"       ,vseq(1))
            self.v_wei_ix        = sym_t("v_wei_ix"       ,vseq(1))

            self.v_gtc_ic0       = sym_t("v_gtc_ic0"       ,v_c_num - 1  if is_vgpr_acc_c else vseq(1))
            self.v_gtc_ic1e      = sym_t("v_gtc_ic1e"      ,v_c_num - 2  if is_vgpr_acc_c else vseq(1))
            self.v_gtc_ik0       = sym_t("v_gtc_ik0"       ,v_c_num - 3  if is_vgpr_acc_c else vseq(1))
            self.v_gtc_ic1       = sym_t("v_gtc_ic1"       ,v_c_num - 4  if is_vgpr_acc_c else vseq(1))

            self.v_gtc_in0       = sym_t("v_gtc_in0"       ,v_c_num - 8  if is_vgpr_acc_c else vseq(1))
            self.v_gtc_in1b      = sym_t("v_gtc_in1b"      ,v_c_num - 9  if is_vgpr_acc_c else vseq(1))
            self.v_gtc_in1       = sym_t("v_gtc_in1"       ,v_c_num - 10 if is_vgpr_acc_c else vseq(1))
            self.v_gemm_in       = sym_t("v_gemm_in"       ,v_c_num - 11 if is_vgpr_acc_c else vseq(1))
            self.v_gemm_im       = sym_t("v_gemm_im"       ,v_c_num - 12 if is_vgpr_acc_c else vseq(1))

            if is_vgpr_acc_c:
                if v_c_num < 16:
                    self.v_wei_ic0        = sym_t("v_wei_ic0"       ,vseq(1))
                    self.v_wei_ic1e       = sym_t("v_wei_ic1e"      ,vseq(1))
                    self.v_wei_ic1        = sym_t("v_wei_ic1"       ,vseq(1))

                    self.v_co_sub_m_index = sym_t("v_co_sub_m_index" ,vseq(1))
                    self.v_co_sub_n_index = sym_t("v_co_sub_n_index" ,vseq(1))
                else:
                    self.v_wei_ic0        = sym_t("v_wei_ic0"       ,v_c_num - 13)
                    self.v_wei_ic1e       = sym_t("v_wei_ic1e"      ,v_c_num - 14)
                    self.v_wei_ic1        = sym_t("v_wei_ic1"       ,v_c_num - 15)

                    self.v_co_sub_m_index = sym_t("v_co_sub_m_index" ,v_c_num - 18)
                    self.v_co_sub_n_index = sym_t("v_co_sub_n_index" ,v_c_num - 19)
            else:
                self.v_wei_ic0            = sym_t("v_wei_ic0"       ,vseq(1))
                self.v_wei_ic1e           = sym_t("v_wei_ic1e"      ,vseq(1))
                self.v_wei_ic1            = sym_t("v_wei_ic1"       ,vseq(1))
                self.v_co_sub_m_index     = sym_t("v_co_sub_m_index" ,vseq(1))
                self.v_co_sub_n_index     = sym_t("v_co_sub_n_index" ,vseq(1))

            self.v_cur_k          = sym_t("v_cur_k" ,vseq(1))
            self.v_tmp           = sym_t("v_tmp"          ,vseq(8, 2))
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

        t_n0, t_n1b, t_k0, t_k1  = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        _   ,    _,  t_c0, t_c1e = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        return t_k0, t_k1, t_n0, t_n1b, t_c0, t_c1e # M, K, N


    def get_cluster_lengths(self):
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4

        c_n0, c_n1b, c_k0, c_k1  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        _   ,     _, c_c0, c_c1e = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        return c_k0, c_k1, c_n0, c_n1b, c_c0, c_c1e # M, K, N

    def get_dims_lengths(self):
        t_k0, t_k1, t_n0, t_n1b, t_c0, t_c1e = self.get_thread_lengths()
        c_k0, c_k1, c_n0, c_n1b, c_c0, c_c1e = self.get_cluster_lengths()

        n_k0, n_k1, n_n0, n_n1b, n_c0, n_c1e = \
                t_k0*c_k0, t_k1*c_k1, t_n0*c_n0, t_n1b*c_n1b, t_c0*c_c0, t_c1e*c_c1e

        return n_k0, n_k1, n_n0, n_n1b, n_c0, n_c1e

    def get_thread_copy_dims(self):
        t_k0, t_k1, t_n0, t_n1b, t_c0, t_c1e = self.get_thread_lengths()
        in_thread_copy_dims     = [t_n0, t_n1b, t_c0, t_c1e]
        out_thread_copy_dims    = [t_n0, t_n1b, t_k0, t_k1]
        return in_thread_copy_dims, out_thread_copy_dims

    def get_thread_copy_index(self):
        in_thread_copy_dims, out_thread_copy_dims = self.get_thread_copy_dims()
        in_thread_copy_index    = _find_non_1_index_in_list(in_thread_copy_dims)
        #print(in_thread_copy_dims)
        #print(in_thread_copy_index)
        out_thread_copy_index   = _find_non_1_index_in_list(out_thread_copy_dims)
        #assert len(out_thread_copy_index) in (1, 2) and len(wei_thread_copy_index) in (1, 2),\
        #        f'out_thread_copy_dims:{out_thread_copy_dims} wei_thread_copy_dims:{wei_thread_copy_dims}'
        return in_thread_copy_index, out_thread_copy_index

    def get_macro_global_load(self):
        inline = True if self.tunable.fma_interleave else False
        t_k0, t_k1, t_n0, t_n1b, t_c0, t_c1e        = self.get_thread_lengths()
        in_thread_copy_dims, out_thread_copy_dims   = self.get_thread_copy_dims()
        in_thread_copy_index, out_thread_copy_index = self.get_thread_copy_index()

        ctrl_in_gld  = ctrl_2d_global_load_t()
        ctrl_out_gld = ctrl_2d_global_load_t()

        if self.tunable.nxb != 1:
            ctrl_in_gld.vector_d1  = igemm_gcd(t_n1b, 4) if self.tunable.nxe == 0 else 1
            ctrl_out_gld.vector_d1 = igemm_gcd(t_n1b, 4) if self.tunable.nxe == 0 else 1
        else:
            ctrl_in_gld.vector_d1  = 1
            ctrl_out_gld.vector_d1 = 1

        if self.in_thread_copy_ndim == 2:
            if t_n1b == 1:
                ctrl_in_gld.length_d0 = in_thread_copy_dims[in_thread_copy_index[0]]
                ctrl_in_gld.length_d1 = in_thread_copy_dims[in_thread_copy_index[1]]
            else:
                ctrl_in_gld.length_d0 = 1#in_thread_copy_dims[in_thread_copy_index[0]]
                ctrl_in_gld.length_d1 = in_thread_copy_dims[in_thread_copy_index[1]] * t_n1b
            #ctrl_in_gld.length_d0 = in_thread_copy_dims[in_thread_copy_index[0]]
            #ctrl_in_gld.length_d1 = in_thread_copy_dims[in_thread_copy_index[1]]
        elif self.in_thread_copy_ndim == 1:
            ctrl_in_gld.length_d0 = 1
            ctrl_in_gld.length_d1 = in_thread_copy_dims[in_thread_copy_index[0]]
        else:
            ctrl_in_gld.length_d0 = 1
            ctrl_in_gld.length_d1 = in_thread_copy_dims[-1]

        if self.out_thread_copy_ndim == 2:
            if t_n1b == 1:
                ctrl_out_gld.length_d0 = out_thread_copy_dims[out_thread_copy_index[0]]
                ctrl_out_gld.length_d1 = out_thread_copy_dims[out_thread_copy_index[1]]
            else:
                ctrl_out_gld.length_d0 = 1#out_thread_copy_dims[out_thread_copy_index[0]]
                ctrl_out_gld.length_d1 = out_thread_copy_dims[out_thread_copy_index[1]] * t_n1b
            #if t_n0 != 1 and t_n1b == 1:
            #    ctrl_out_gld.src_order = 1              # this reorder seems have little impact...
            #ctrl_out_gld.length_d0 = out_thread_copy_dims[out_thread_copy_index[0]]
            #ctrl_out_gld.length_d1 = out_thread_copy_dims[out_thread_copy_index[1]]
        elif self.out_thread_copy_ndim == 1:
            ctrl_out_gld.length_d0 = 1
            ctrl_out_gld.length_d1 = out_thread_copy_dims[out_thread_copy_index[0]]
        else:
            ctrl_out_gld.length_d0 = 1
            ctrl_out_gld.length_d1 = out_thread_copy_dims[-1]

        if self.tunable.precache_soffset:
            return macro_igemm_2d_global_load_precache_soffset_t(self.mc, ctrl_in_gld, inline), \
                    macro_igemm_2d_global_load_precache_soffset_t(self.mc, ctrl_out_gld, inline)
        else:
            return macro_igemm_2d_global_load_t(self.mc, ctrl_in_gld),  macro_igemm_2d_global_load_t(self.mc, ctrl_out_gld, inline)

    def get_macro_global_store(self):
        return macro_igemm_write_4d_strided_t(self.mc)

    def get_macro_shared_store(self):
        in_thread_copy_dims, out_thread_copy_dims   = self.get_thread_copy_dims()
        in_thread_copy_index, out_thread_copy_index = self.get_thread_copy_index()
        n_k0, n_k1, n_n0, n_n1b, n_c0, n_c1e = self.get_dims_lengths()
        t_k0, t_k1, t_n0, t_n1b, t_c0, t_c1e = self.get_thread_lengths()
        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()

        if gemm_n_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_N_C0_C1E:
            # t_n0, t_n1b, t_c0, t_c1e
            in_stride_list = [n_n1b*n_c0*n_c1e, n_c0*n_c1e, n_c1e, 1]
        else:
            # t_k0, t_k1e, t_n0, t_n1b
            in_stride_list = [n_n1b*n_c0*n_c1e, n_c0*n_c1e, 1, n_c0]


        if gemm_m_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1:
            out_stride_list = [n_n1b*n_k0*n_k1, n_k0*n_k1, n_k1, 1]
        else:
            out_stride_list = [n_n1b*n_k0*n_k1, n_k0*n_k1, 1, n_k0]

        in_sst_ctrl  = ctrl_2d_shared_store_t()
        out_sst_ctrl = ctrl_2d_shared_store_t()

        if self.tunable.nxb != 1:
            vector_in_d1  = igemm_gcd(t_n1b, 4) if self.tunable.nxe == 0 else 1
            vector_out_d1 = igemm_gcd(t_n1b, 4) if self.tunable.nxe == 0 else 1
        else:
            vector_in_d1  = 1
            vector_out_d1 = 1

        #print(f"vector_in_d1={vector_in_d1}, vector_out_d1={vector_in_d1}")

        in_sst_ctrl.src_order = 1 if vector_in_d1 > 1 else 0
        out_sst_ctrl.src_order = 1 if vector_out_d1 > 1 else 0
        in_sst_ctrl.v_tmp = self.vgpr.v_tmp
        out_sst_ctrl.v_tmp = self.vgpr.v_tmp

        #print(f"in_sst_ctrl.src_order={in_sst_ctrl.src_order}, out_sst_ctrl.src_order={out_sst_ctrl.src_order}")

        if self.in_thread_copy_ndim == 2:
            in_sst_ctrl.length_d0 = in_thread_copy_dims[in_thread_copy_index[0]]
            in_sst_ctrl.length_d1 = in_thread_copy_dims[in_thread_copy_index[1]]
            if gemm_n_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_N_C0_C1E:
                in_sst_ctrl.vector_d1 = t_c1e
            else:
                in_sst_ctrl.vector_d1 = in_thread_copy_dims[in_thread_copy_index[1]]
            #in_sst_ctrl.vector_d1 = t_c1
            in_sst_ctrl.stride_d0 = in_stride_list[in_thread_copy_index[0]] * data_byte
            in_sst_ctrl.stride_d1 = in_stride_list[in_thread_copy_index[1]] * data_byte
        elif self.in_thread_copy_ndim == 1:
            in_sst_ctrl.length_d0 = 1
            in_sst_ctrl.length_d1 = in_thread_copy_dims[in_thread_copy_index[0]]

            if (gemm_n_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_N_C0_C1E and t_c1e != 1) or \
                (gemm_n_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_N_C1E_C0 and t_c0 != 1):
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
            if (gemm_n_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_N_C0_C1E and t_c1e != 1) or \
                (gemm_n_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_N_C1E_C0 and t_c0 != 1):
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

        if self.out_thread_copy_ndim == 2:
            out_sst_ctrl.length_d0 = out_thread_copy_dims[out_thread_copy_index[0]]
            out_sst_ctrl.length_d1 = out_thread_copy_dims[out_thread_copy_index[1]]
            if gemm_m_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1:
                out_sst_ctrl.vector_d1 = t_k1
            else:
                out_sst_ctrl.vector_d1 = out_thread_copy_dims[out_thread_copy_index[1]]
            #out_sst_ctrl.vector_d1 = t_n1b
            out_sst_ctrl.stride_d0 = out_stride_list[out_thread_copy_index[0]] * data_byte
            out_sst_ctrl.stride_d1 = out_stride_list[out_thread_copy_index[1]] * data_byte
            #out_sst_ctrl.stride_d1 = 1
        elif self.out_thread_copy_ndim == 1:
            out_sst_ctrl.length_d0 = 1
            out_sst_ctrl.length_d1 = out_thread_copy_dims[out_thread_copy_index[0]]
            if (gemm_m_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1 and t_k1 != 1) or \
                (gemm_m_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0 and t_k0 != 1):
                out_sst_ctrl.vector_d1 = out_thread_copy_dims[out_thread_copy_index[0]]
            else:
                out_sst_ctrl.vector_d1 = 1
            out_sst_ctrl.stride_d0 = 1
            out_sst_ctrl.stride_d1 = out_stride_list[out_thread_copy_index[0]] * data_byte
            if out_sst_ctrl.length_d1 == 8 and out_sst_ctrl.vector_d1 != 1:
                # assert False
                # TODO: this is indeed not optimal. may consider shuffle in the future.
                out_sst_ctrl.length_d0 = 2
                out_sst_ctrl.length_d1 = 4
                out_sst_ctrl.vector_d1 = 4
                out_sst_ctrl.stride_d0 = 4 * data_byte
        else:
            out_sst_ctrl.length_d0 = 1
            out_sst_ctrl.length_d1 = out_thread_copy_dims[-1]

            if (gemm_m_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1 and t_k1 != 1) or \
                (gemm_m_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0 and t_k0 != 1):
                out_sst_ctrl.vector_d1 = out_thread_copy_dims[-1]
            else:
                out_sst_ctrl.vector_d1 = 1

            out_sst_ctrl.stride_d0 = 1
            out_sst_ctrl.stride_d1 = out_stride_list[-1] * data_byte
            if out_sst_ctrl.length_d1 == 8 and out_sst_ctrl.vector_d1 != 1:
                # assert False
                # TODO: this is indeed not optimal. may consider shuffle in the future.
                out_sst_ctrl.length_d0 = 2
                out_sst_ctrl.length_d1 = 4
                out_sst_ctrl.vector_d1 = 4
                out_sst_ctrl.stride_d0 = 4 * data_byte

        # print(f"in_sst_ctrl.vector_d1:{in_sst_ctrl.vector_d1}, out_sst_ctrl.vector_d1:{out_sst_ctrl.vector_d1}")
        inline = True if self.tunable.fma_interleave else False 
        return macro_igemm_2d_shared_store_t(self.mc, in_sst_ctrl, inline), macro_igemm_2d_shared_store_t(self.mc, out_sst_ctrl, inline)

    #def get_macro_shared_load(self):
    #    return None

    def get_macro_in_update_os(self):
        inline = True if self.tunable.fma_interleave else False
        return macro_igemm_wrw_gtc_in_update_os_t(self.mc, amdgpu_precision_data_byte(self.tunable.precision), inline)

    def get_macro_in_update_hw(self):
        inline = True if self.tunable.fma_interleave else False
        if self.tunable.nxb != 0:
            return macro_igemm_wrw_gtc_in_update_hw_t(self.mc, inline)
        return None

    def get_macro_out_update_os(self):
        inline = True if self.tunable.fma_interleave else False
        return macro_igemm_wrw_gtc_out_update_os_t(self.mc, amdgpu_precision_data_byte(self.tunable.precision), inline)
    
    def get_macro_out_update_hw(self):
        inline = True if self.tunable.fma_interleave else False
        #if self.tunable.nxe != 0:
        return macro_igemm_wrw_gtc_out_update_hw_t(self.mc, inline)
        #return None
   
    def get_macro_set_flag_hw(self):
        inline = True if self.tunable.fma_interleave else False
        return macro_igemm_wrw_gtc_set_flag_hw(self.mc, inline)

    def get_macro_move_slice_window(self):
        inline = True if self.tunable.fma_interleave else False
        return macro_igemm_wrw_gtc_move_slice_window_n_dsho_dswo(self.mc, self.tunable, inline)

    def get_macro_move_slice_window_check_last_dim(self):
        inline = True if self.tunable.fma_interleave else False
        return macro_igemm_wrw_gtc_move_slice_window_n_dsho_dswo_check_last_dim(self.mc, self.tunable, inline)

    def get_symbol_global_load_s_stride_d0_d1(self):
        t_k0, t_k1, t_n0, t_n1b, t_c0, t_c1e = self.get_thread_lengths()
        # get the symbol object that load 2d may use
        s = self.sgpr
        s_dummy = sym_t("s_dummy")
        in_thread_copy_index, out_thread_copy_index = self.get_thread_copy_index()

        in_stride_gprs = [s.s_in_stride_n0 if t_n0 != 1 else s_dummy,
                    # s_dummy if self.tunable.nxb != 0 else s.s_in_stride_n,
                    s.s_in_stride_n,
                    s.s_in_stride_c0 if t_c0 != 1 else s_dummy,
                    s_dummy]
        out_stride_gprs = [s.s_out_stride_n0 if t_n0 != 1 else s_dummy,
                    # s_dummy if self.tunable.nxb != 0 else s.s_out_stride_n,
                    s.s_out_stride_n,
                    s.s_out_stride_k0 if t_k0 != 1 else s_dummy,
                    s.s_out_stride_k]
        
        if self.in_thread_copy_ndim == 2:
            s_in_stride_d0 = in_stride_gprs[in_thread_copy_index[0]]
            s_in_stride_d1 = in_stride_gprs[in_thread_copy_index[1]]
        elif self.in_thread_copy_ndim == 1:
            s_in_stride_d0 = s_dummy
            s_in_stride_d1 = in_stride_gprs[in_thread_copy_index[0]]
        else:
            s_in_stride_d0 = s_dummy
            s_in_stride_d1 = in_stride_gprs[-1]

        if self.out_thread_copy_ndim == 2:
            s_out_stride_d0 = out_stride_gprs[out_thread_copy_index[0]]
            s_out_stride_d1 = out_stride_gprs[out_thread_copy_index[1]]
        elif self.out_thread_copy_ndim == 1:
            s_out_stride_d0 = s_dummy
            s_out_stride_d1 = out_stride_gprs[out_thread_copy_index[0]]
        elif self.out_thread_copy_ndim == 0:
            s_out_stride_d0 = s_dummy
            s_out_stride_d1 = s_dummy
        else:
            s_out_stride_d0 = s_dummy
            s_out_stride_d1 = out_stride_gprs[-1]

        #print(f"in_thread_copy_ndim={self.in_thread_copy_ndim}, out_thread_copy_ndim={self.out_thread_copy_ndim}")
        #print(s_in_stride_d0(), s_in_stride_d1(), s_out_stride_d0(), s_out_stride_d1())

        return s_in_stride_d0, s_in_stride_d1, s_out_stride_d0, s_out_stride_d1

    def get_kernel_code(self):
        kernel_code = amdgpu_kernel_code_t({
                'enable_sgpr_kernarg_segment_ptr'   :   1,
                'enable_sgpr_workgroup_id_x'        :   1,
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
            int gemm_k_global_split;
            int group;
            int __pack_0;
        '''
        kas = []
        # name: {}, .size: {}, .offset: {}, .value_kind: {}, .value_type
        kas.append(amdgpu_kernel_arg_t('p_in'          , 8,   0, 'global_buffer','f32',address_space='global',is_const='false'))
        kas.append(amdgpu_kernel_arg_t('p_wei'         , 8,   8, 'global_buffer','f32',address_space='global',is_const='true'))
        kas.append(amdgpu_kernel_arg_t('p_out'         , 8,  16, 'global_buffer','f32',address_space='global',is_const='true'))
        kas.append(amdgpu_kernel_arg_t('hi'            , 4,  24, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('wi'            , 4,  28, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('n'             , 4,  32, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('k'             , 4,  36, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('c'             , 4,  40, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('ho'            , 4,  44, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('wo'            , 4,  48, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('stride_h'      , 4,  52, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('stride_w'      , 4,  56, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_h'    , 4,  60, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_w'    , 4,  64, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('pad_h'         , 4,  68, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('pad_w'         , 4,  72, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('y'             , 4,  76, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('x'             , 4,  80, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('gemm_k_global_split'  , 4,  84, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('group'         , 4,  88, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('__pack_0'      , 4,  92, 'by_value', 'i32'))
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

        t_k0, t_k1, t_n0, t_n1b, t_c0, t_c1e = self.get_thread_lengths()
        c_k0, c_k1, c_n0, c_n1b, c_c0, c_c1e = self.get_cluster_lengths()
        n_k0, n_k1, n_n0, n_n1b, n_c0, n_c1e = self.get_dims_lengths()

        unmerge_sub_c = self.tunable.unmerge_sub_c
        if gemm_n_unmerge_cluster == 0:
            assert unmerge_sub_c % n_c0 == 0, f"unmerge_sub_c:{unmerge_sub_c}, n_c0:{n_c0}"
            #print(f"unmerge_sub_c:{unmerge_sub_c}, n_c0:{n_c0}")
            unmerge_sub_c1 = unmerge_sub_c // n_c0
            #print(f"n_c1e:{n_c1e}, unmerge_sub_c1:{unmerge_sub_c1}")
            assert n_c1e % unmerge_sub_c1 == 0, f"n_c1e:{n_c1e}, unmerge_sub_c1:{unmerge_sub_c1}"
        elif gemm_n_unmerge_cluster == 1:
            assert c_c0 == 1 and n_c1e != 1 and t_c0 != 1 and t_c1e == 1, "current implementation only support this stratagy"
            unmerge_sub_c1 = unmerge_sub_c
        else:
            assert False, f"unsupported gemm_n_unmerge_cluster:{self.tunable.gemm_n_unmerge_cluster}"

        unmerge_sub_n = self.tunable.unmerge_sub_n
        if gemm_k_unmerge_cluster == 0:
            assert unmerge_sub_n % n_n0 == 0, f"unmerge_sub_n:{unmerge_sub_n}, n_n0:{n_n0}"
            unmerge_sub_n1 = unmerge_sub_n // n_n0
            assert n_n1b % unmerge_sub_n1 == 0, f"n_n1b:{n_n1b}, unmerge_sub_n1:{unmerge_sub_n1}"
        elif gemm_k_unmerge_cluster == 1:
            assert c_n0 == 1 and c_n1b != 1 and t_n0 != 1 and t_n1b == 1, "current implementation only support this stratagy"
            unmerge_sub_n1 = unmerge_sub_n
        else:
            assert False, f"unsupported gemm_k_unmerge_cluster:{self.tunable.gemm_k_unmerge_cluster}"

        if gemm_m_unmerge_cluster == 1:
            assert c_k0 == 1 and c_k1 != 1 and t_k0 != 1 and t_k1 == 1, "current implementation only support this stratagy"

        #assert c_n0 == 1 and c_k0 == 1 and c_c0 == 1, "cluster lengths has no meaning to deal with x0"

        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        m_in_update_os   = self.get_macro_in_update_os()
        m_in_update_hw   = self.get_macro_in_update_hw()
        m_out_update_os   = self.get_macro_out_update_os()
        m_out_update_hw   = self.get_macro_out_update_hw()
        m_set_flag_hw     = self.get_macro_set_flag_hw()

        m_in_2d_global_load, m_out_2d_global_load = self.get_macro_global_load()
        s_in_stride_d0, s_in_stride_d1, s_out_stride_d0, s_out_stride_d1 = self.get_symbol_global_load_s_stride_d0_d1()

        tc_index_dispatcher = igemm_thread_cluster_index_dispatcher_t(self.mc)
        tc_index_accumulator = igemm_thread_cluster_index_accumulator_t(self.mc)

        m_int_div_rem_vv = macro_int_div_rem_vv_t(self.mc)
        m_int_div_rem_vs = macro_int_div_rem_vs_t(self.mc)
        m_int_div_rem_ss = macro_int_div_rem_ss_t(self.mc)
        gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()
        s_dummy = sym_t("s_dummy")

        self._emit(f"; unmerge_sub_n:{unmerge_sub_n}, unmerge_sub_n1:{unmerge_sub_n1}, unmerge_sub_c:{unmerge_sub_c}, unmerge_sub_c1:{unmerge_sub_c1}")
        self._emit(f"; gemm_m_unmerge_cluster:{gemm_m_unmerge_cluster}, gemm_n_unmerge_cluster:{gemm_n_unmerge_cluster}, gemm_k_unmerge_cluster:{gemm_k_unmerge_cluster}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_in((0,1))}],       s[{s.s_ka((0, 1))}],    0+{k.k_p_in()}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_wei((0,1))}],      s[{s.s_ka((0, 1))}],    0+{k.k_p_wei()}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_out((0,1))}],      s[{s.s_ka((0, 1))}],    0+{k.k_p_out()}")
        self._emit(f"s_load_dwordx16 s[{s.s_hi((0,15))}],        s[{s.s_ka((0, 1))}],    0+{k.k_hi()}")
        self._emit(f"s_load_dword s[{s.s_group()}],         s[{s.s_ka((0, 1))}],    0+{k.k_group()}")

        # self._emit("; clear vector r")
        # self._emit(".v_clear_nc v_c+1, v_end-1")
        self._emit_empty_line()
        if IGEMM_WRW_GTC_DEBUG == 1:
            self._emit("; debug vgpr")
            self._emit("v_mov_b32 v1, 0")
            self._emit("v_add_lshl_u32 v[v_tmp+6], v0, v1, 2")
            self._emit(";v_lshlrev_b32 v[114], 2, v0 ; every thread write one float")
            self._emit(f"s_load_dwordx2 s[{s.s_dbg((0,1))}], s[s_ka:s_ka+1], k_p_wei")

        self._emit(f"; input, thread(n0,n1b,c0,c1e): {t_n0}x{t_n1b}x{t_c0}x{t_c1e}, cluster(n0,n1b,c0,c1e): {c_n0}x{c_n1b}x{c_c0}x{c_c1e}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        self._emit(tc_index_dispatcher(v.v_gtc_ic1e(),  v.v_tmp(), c_c1e, t_c1e))      # merged dimension no need to do shift per thread here, do shift later
        self._emit(tc_index_dispatcher(v.v_gtc_ic0(),   v.v_tmp(), c_n0,  t_n0))
        self._emit(tc_index_dispatcher(v.v_gtc_in1b(),  v.v_tmp(), c_n1b, t_n1b))      # merged dimension no need to do shift per thread here, do shift later
        self._emit(tc_index_dispatcher(v.v_gtc_in0(),   v.v_tmp(), c_n0,  t_n0, True))
        self._emit_empty_line()
        self._emit(f"; output, thread(n0,n1b,k0,k1): {t_n0}x{t_n1b}x{t_k0}x{t_k1}, cluster(n0,n1b,k0,k1) {c_n0}x{c_n1b}x{c_k0}x{c_k1}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        self._emit(tc_index_dispatcher(v.v_gtc_ik1(), v.v_tmp(), c_k1, t_k1))
        self._emit(tc_index_dispatcher(v.v_gtc_ik0(), v.v_tmp(), c_k0, t_k0, True))
        self._emit_empty_line()
        self._emit(f"s_mov_b32 s[{s.s_p_in(2)}], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_in(3)}], 0x27000")
        self._emit(f"s_mov_b32 s[{s.s_p_wei(2)}], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_wei(3)}], 0x27000")
        self._emit(f"s_mov_b32 s[{s.s_p_out(2)}], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_out(3)}], 0x27000")
        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit_empty_line()

        self._emit(f"; calculate index")

        self._emit(f"s_mul_i32 s[{s.s_in_stride_c()}],      s[{s.s_hi()}],       s[{s.s_wi()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}],  s[{s.s_in_stride_c()}], s[{s.s_c()}]")
        self._emit(f"s_mul_i32 s[{s.s_in_stride_n()}],      s[{s.s_group()}],        s[{s.s_tmp()}]")
        self._emit(f"s_mul_i32 s[{s.s_wei_stride_c()}],       s[{s.s_y()}],       s[{s.s_x()}]")
        self._emit(f"s_mul_i32 s[{s.s_wei_stride_k()}],       s[{s.s_c()}],        s[{s.s_wei_stride_c()}]")
        if gemm_m_unmerge_cluster == 1:
            self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_k()}], {igemm_log2(n_k0)}")
            self._emit(f"s_mul_i32 s[{s.s_wei_stride_k0()}], s[{s.s_wei_stride_k()}], s[{s.s_tmp()}]")
        if gemm_n_unmerge_cluster == 1:
            self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_c()}], {igemm_log2(n_c0)}")
            self._emit(f"s_mul_i32 s[{s.s_wei_stride_c0()}], s[{s.s_wei_stride_c()}], s[{s.s_tmp()}]")
        self._emit(f"s_mul_i32 s[{s.s_out_stride_k()}],      s[{s.s_ho()}],        s[{s.s_wo()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}],  s[{s.s_out_stride_k()}],  s[{s.s_k()}]")
        self._emit(f"s_mul_i32 s[{s.s_out_stride_n()}],      s[{s.s_group()}],        s[{s.s_tmp()}]")

        self._emit("; config for weight range")
        self._emit(f"s_mul_i32 s[{s.s_p_out(2)}], s[{s.s_out_stride_n()}], s[{s.s_n()}]")
        self._emit(f"s_lshl_b32 s[{s.s_p_out(2)}], s[{s.s_p_out(2)}], {igemm_log2(data_byte)}")

        if t_n0 != 1:
            self._emit(f"s_lshl_b32 s[{s.s_in_stride_n0()}], s[{s.s_in_stride_n()}], {igemm_log2(unmerge_sub_n1)}")
            self._emit(f"s_lshl_b32 s[{s.s_out_stride_n0()}], s[{s.s_out_stride_n()}], {igemm_log2(unmerge_sub_n1)}")
        if t_c0 != 1:
            if gemm_n_unmerge_cluster == 0:
                self._emit(f"s_lshl_b32 s[{s.s_in_stride_c0()}], s[{s.s_in_stride_c()}], {igemm_log2(unmerge_sub_c1)}")
            else:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_c()}], {igemm_log2(n_c0)}")
                self._emit(f"s_mul_i32 s[{s.s_in_stride_c0()}], s[{s.s_in_stride_c()}], s[{s.s_tmp()}]")
        if t_k0 != 1:
            if gemm_m_unmerge_cluster == 0:
                self._emit(f"s_lshl_b32 s[{s.s_out_stride_k0()}], s[{s.s_out_stride_k()}], {igemm_log2(n_k1)}")
            else:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_k()}], {igemm_log2(n_k0)}")
                self._emit(f"s_mul_i32 s[{s.s_out_stride_k0()}], s[{s.s_out_stride_k()}], s[{s.s_tmp()}]")

        if self.tunable.nxe != 0:
            self._emit(f"s_add_u32 s[{s.s_tmp()}], {self.tunable.nxb - 1}, s[{s.s_out_stride_k()}]")
            self._emit(f"s_lshr_b32 s[{s.s_tmp(1)}], s[{s.s_tmp()}], {igemm_log2(self.tunable.nxb)}")
            self._emit(f"s_lshl_b32 s[{s.s_dim_b()}], s[{s.s_tmp(1)}], {igemm_log2(self.tunable.nxb)}")

        self._emit(f"; n1b transform")

        self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_in1(), v.v_gtc_in1b(), s.s_dim_b() if self.tunable.nxe != 0 else s.s_out_stride_k() , v.v_tmp(), s.s_tmp()))
        # self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_in1(), v.v_gtc_in1b(), s.s_out_stride_k(), v.v_tmp(), s.s_tmp()))
        self._emit(m_int_div_rem_vs(v.v_move_slice_n_idswo(), v.v_move_slice_n_idsho(), v.v_tmp(4), s.s_wo(), v.v_tmp(), s.s_tmp()))
        m_out_update_hw   = self.get_macro_out_update_hw()
        self._emit(m_out_update_hw(v.v_out_iho(), v.v_out_iwo(), v.v_move_slice_n_idsho(), v.v_move_slice_n_idswo()))
        self._emit(f"v_mov_b32 v[{v.v_move_slice_n_in1()}], v[{v.v_gtc_in1()}]")
        if self.tunable.nxe != 0:
            self._emit(f"v_mov_b32 v[{v.v_move_slice_n_in0()}], v[{v.v_gtc_in0()}]")
            #self._emit(f"v_mov_b32 v[{v.v_flag_n()}], 1")

        self._emit_empty_line()
        self._emit(f"; pad gemm_m if needed")
        self._emit(f"s_add_u32 s[{s.s_tmp()}], {self.tunable.gemm_m_per_block - 1}, s[{s.s_k()}]")
        self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_m_per_block)}")
        self._emit(f"s_lshl_b32 s[{s.s_k_padded()}], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_m_per_block)}")

        self._emit_empty_line()
        self._emit(f"; add block i_n")
        self._emit(f"; gemm_m_per_block:{self.tunable.gemm_m_per_block}, gemm_n_per_block:{self.tunable.gemm_n_per_block}")
        # calculate group index
        self._emit(f"s_lshr_b32 s[0], s[{s.s_wei_stride_k()}], {igemm_log2(self.tunable.gemm_n_per_block)}")
        self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_k_padded()}], {igemm_log2(self.tunable.gemm_m_per_block)}")
        self._emit(f"s_mul_i32 s[1], s[0], s[{s.s_tmp()}]")
        self._emit(f"s_lshl_b32 s[3], s[1], s[{s.s_gemmk_split()}]")
        self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_block_gtc_ig(), s.s_bx(), '3', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
        self._emit(m_int_div_rem_ss(s.s_bx(), s.s_block_gtc_in(), s.s_tmp(4), '1', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
        self._emit(f"s_lshr_b32 s[{s.s_sub_n()}], s[{s.s_n()}], s[{s.s_gemmk_split()}]")
        self._emit(f"s_mul_i32 s[{s.s_block_gtc_in()}], s[{s.s_block_gtc_in()}], s[{s.s_sub_n()}]")

        self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_tmp(5), s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
        self._emit(f"; s_tmp+4:block_gtc_in, s_tmp+5:block_gtc_im")
        if gemm_m_unmerge_cluster == 0:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ik()}], s[{s.s_tmp(5)}], {igemm_log2(self.tunable.gemm_m_per_block)}")
        else:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ik()}], s[{s.s_tmp(5)}], {igemm_log2(self.tunable.gemm_m_per_block // n_k0)}")

        if gemm_n_unmerge_cluster == 0:
            if unmerge_sub_c1 == 1:
                self._emit(f"s_lshr_b32 s[0], s[{s.s_wei_stride_c()}], {igemm_log2(n_c1e)} ; total number of c1e")
            else:
                if unmerge_sub_c1 == n_c1e:
                    self._emit(f"s_mov_b32 s[0], s[{s.s_wei_stride_c()}] ; total number of c1e")
                else:
                    self._emit(f"s_lshr_b32 s[0], s[{s.s_wei_stride_c()}], {igemm_log2(n_c1e // unmerge_sub_c1)}  ; total number of c1e")
        else:
            self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_c()}], {igemm_log2(n_c0)}")
            self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_wei_stride_c()}], s[{s.s_tmp()}]")
            self._emit(f"s_lshr_b32 s[0], s[{s.s_tmp(1)}], {igemm_log2(n_c1e)}")

        self._emit(m_int_div_rem_ss(s.s_block_gtc_ic1e(), s.s_block_gtc_ic0(), s.s_tmp(4), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
        if n_c1e != 1:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ic1e()}], s[{s.s_block_gtc_ic1e()}], {igemm_log2(n_c1e)}")
        if n_c0 != 1:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ic0()}], s[{s.s_block_gtc_ic0()}], {igemm_log2(n_c0)}")

        self._emit_empty_line()

        self._emit(f"; c1e transform")
        if c_c1e == 1:
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_ic1e()}]")
        else:
            self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_ic1e()}], v[{v.v_gtc_ic1e()}]")
        
        self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_ic1(), v.v_tmp(5), s.s_wei_stride_c(), v.v_tmp(), s.s_tmp()))
        self._emit(m_int_div_rem_vs(v.v_wei_ix(), v.v_wei_iy(), v.v_tmp(4), s.s_x(), v.v_tmp(), s.s_tmp()))
        self._emit_empty_line()

        m_in_update_hw   = self.get_macro_in_update_hw()
        # ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h,   here make sure iy <- iy * s_dilation_h - s_pad_h before hand
        # iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w,   here make sure ix <- ix * s_dilation_w - s_pad_w before hand
        self._emit(f"v_mul_u32_u24 v[{v.v_tmp()}], s[{s.s_dilation_h()}], v[{v.v_wei_iy()}]")
        self._emit(f"v_mul_u32_u24 v[{v.v_tmp(1)}], s[{s.s_dilation_w()}], v[{v.v_wei_ix()}]")
        self._emit(f"v_sub_i32 v[{v.v_wei_iy()}], v[{v.v_tmp()}], s[{s.s_pad_h()}]")
        self._emit(f"v_sub_i32 v[{v.v_wei_ix()}], v[{v.v_tmp(1)}], s[{s.s_pad_w()}]")
        self._emit(m_in_update_hw(v.v_in_ihi(), v.v_in_iwi(), v.v_out_iho(), v.v_out_iwo(), s.s_stride_h(), s.s_stride_w(), v.v_wei_iy(), v.v_wei_ix(), s.s_dilation_h(), s.s_dilation_w(), s.s_pad_h(), s.s_pad_w(), v.v_tmp()))
        #self._emit(f"; compute i_in_iwi and i_in_ihi")
        #self._emit(f"; transform iho, iwo, iy, ix -> hip, wip")
        #self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_stride_h()}], v[{v.v_out_iho()}]")
        #self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_stride_w()}], v[{v.v_out_iwo()}]")
        #self._emit(f"v_mul_lo_u32 v[{v.v_tmp(2)}], s[{s.s_dilation_h()}], v[{v.v_wei_iy()}]")
        #self._emit(f"v_mul_lo_u32 v[{v.v_tmp(3)}], s[{s.s_dilation_w()}], v[{v.v_wei_ix()}]")

        #self._emit(f"; transform hip, wip -> hi, wi")
        #self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_tmp()}], v[{v.v_tmp(2)}]")
        #self._emit(f"v_add_u32 v[{v.v_tmp(1)}], v[{v.v_tmp(1)}], v[{v.v_tmp(3)}]")
        #self._emit(f"v_sub_i32 v[{v.v_in_ihi()}], v[{v.v_tmp()}], s[{s.s_pad_h()}]")
        #self._emit(f"v_sub_i32 v[{v.v_in_iwi()}], v[{v.v_tmp(1)}], s[{s.s_pad_w()}]")

        self._emit(f"; calculate input offset")
        # compute group distance
        self._emit(f"s_mul_i32 s[{s.s_tmp(5)}], s[{s.s_c()}], s[{s.s_in_stride_c()}]")
        self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ig()}], s[{s.s_block_gtc_ig()}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(5)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(5)}]")
        self._emit(f"s_sub_u32 s[{s.s_p_in(2)}], s[{s.s_p_in(2)}], s[{s.s_tmp()}]")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], s[{s.s_tmp(1)}]")
        if gemm_n_unmerge_cluster == 0:
            self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], s[{s.s_block_gtc_ic0()}], {igemm_log2(unmerge_sub_c1 * data_byte)}")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_in_stride_c()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_in_stride_c()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_sub_u32 s[{s.s_p_in(2)}], s[{s.s_p_in(2)}], s[{s.s_tmp()}]")
            self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
            self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], s[{s.s_tmp(1)}]")
        else:
            pass # no ic0
        self._emit_empty_line()

        self._emit(tc_index_accumulator(v.v_tmp(), v.v_gtc_in0(), v.v_gtc_in1(), c_n0, c_n1b, 0, unmerge_sub_n1))
        if self.tunable.nxe != 0:
            self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_n()}], v[{v.v_tmp()}]")
            self._emit(f"v_cndmask_b32 v[{v.v_flag_n()}], 0, 1, vcc")
        self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_tmp()}], s[{s.s_block_gtc_in()}]")
        #if self.tunable.nxe != 0:
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_in_stride_n()}], v[{v.v_tmp()}]")
        #else:
        #self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_stride_hw()}], v[{v.v_tmp()}]")
        if gemm_n_unmerge_cluster == 0:
            self._emit(tc_index_accumulator(v.v_tmp(1), v.v_gtc_ic0(), v.v_gtc_ic1(), c_c0, c_c1e, 0, unmerge_sub_c1))
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_in_stride_c()}], v[{v.v_tmp(1)}]")
        else:
            # no in0
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_in_stride_c()}], v[{v.v_gtc_ic1()}]")
        
        self._emit(f"v_add_lshl_u32 v[{v.v_in_os_base()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
        self._emit(m_in_update_os(v.v_in_os(), v.v_in_os_base(), v.v_in_ihi(), v.v_in_iwi(), s.s_wi(), v.v_tmp()))
        if self.tunable.nxe != 0:
            self._emit(m_set_flag_hw(v.v_in_flag(), v.v_in_ihi(), v.v_in_iwi(), s.s_hi(), s.s_wi()))
            self._emit(f"v_and_b32 v[{v.v_in_flag()}], v[{v.v_in_flag()}], v[{v.v_flag_n()}]")
        
        self._emit_empty_line()

        if self.in_thread_copy_ndim != 1:
            if s_in_stride_d0 != s_dummy:
                #self._emit(f"s_lshl_b32 s[{s_out_stride_d0()}], s[{s_out_stride_d0()}], {igemm_log2(data_byte)}")
                self._emit(self.try_shift_stride(s_in_stride_d0, igemm_log2(data_byte)))
        if s_in_stride_d1 != s_dummy:
            #self._emit(f"s_lshl_b32 s[{s_out_stride_d1()}], s[{s_out_stride_d1()}], {igemm_log2(data_byte)}")
            self._emit(self.try_shift_stride(s_in_stride_d1, igemm_log2(data_byte)))
        self._emit_empty_line()

        if self.tunable.precache_soffset:
            #assert type(m_out_2d_global_load) is macro_igemm_2d_global_load_precache_soffset_t
            #init_precache_soffset(s_stride_d0, s_stride_d1, s_offset, s_tmp):
            self._emit(m_in_2d_global_load.init_precache_soffset(s_in_stride_d0(), s_in_stride_d1(), s.s_in_offset(), s.s_tmp()))

        # load out
        self._emit(self.global_load_in())
        self._emit_empty_line()

        self._emit(f"; calculate out offset")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_k()}], s[{s.s_out_stride_k()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
        self._emit(f"s_sub_u32 s[{s.s_p_out(2)}], s[{s.s_p_out(2)}], s[{s.s_tmp()}]")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], s[{s.s_tmp(1)}]")
        self._emit_empty_line()
        self._emit(tc_index_accumulator(v.v_tmp(), v.v_gtc_ik0(), v.v_gtc_ik1(), c_k0, c_k1, n_k0, n_k1))
        self._emit(f"v_add_u32 v[{v.v_cur_k()}], s[{s.s_block_gtc_ik()}], v[{v.v_tmp()}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_out_stride_k()}], v[{v.v_cur_k()}]")
        self._emit(tc_index_accumulator(v.v_tmp(1), v.v_gtc_in0(), v.v_gtc_in1(), c_n0, c_n1b, 0, unmerge_sub_n1))
        self._emit(f"v_add_u32 v[{v.v_tmp(1)}], v[{v.v_tmp(1)}], s[{s.s_block_gtc_in()}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_out_stride_n()}], v[{v.v_tmp(1)}]")
        self._emit(f"v_add_lshl_u32 v[{v.v_out_os_base()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
        self._emit(m_out_update_os(v.v_out_os(), v.v_out_os_base(), v.v_out_iho(), v.v_out_iwo(), s.s_wo(), v.v_tmp()))
        if self.tunable.nxe != 0:
            self._emit(m_set_flag_hw(v.v_out_flag(), v.v_out_iho(), v.v_out_iwo(), s.s_ho(), s.s_wo()))
            self._emit(f"v_and_b32 v[{v.v_out_flag()}], v[{v.v_out_flag()}], v[{v.v_flag_n()}]")
        self._emit_empty_line()

        if self.out_thread_copy_ndim != 1:
            if s_out_stride_d0 != s_dummy:
                #self._emit(f"s_lshl_b32 s[{s_wei_stride_d0()}], s[{s_wei_stride_d0()}], {igemm_log2(data_byte)}")
                self._emit(self.try_shift_stride(s_out_stride_d0, igemm_log2(data_byte)))
        if s_out_stride_d1 != s_dummy:
            #self._emit(f"s_lshl_b32 s[{s_wei_stride_d1()}], s[{s_wei_stride_d1()}], {igemm_log2(data_byte)}")
            self._emit(self.try_shift_stride(s_out_stride_d1, igemm_log2(data_byte)))
        self._emit_empty_line()

        if self.tunable.precache_soffset:
            self._emit(m_out_2d_global_load.init_precache_soffset(s_out_stride_d0(), s_out_stride_d1(), s.s_out_offset(), s.s_tmp()))

        self._emit(self.global_load_out())
        self._emit_empty_line()

        if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], v0")
            self._emit(self.thread_mapping(v.v_gemm_in(), v.v_gemm_im(), v.v_tmp(5), v.v_tmp()))
        else:
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], v0")
            self._emit(self.xdlops_mapping.get_gemm_index_for_src_matrix(v.v_gemm_in(), v.v_gemm_im(), v.v_tmp(5), v.v_tmp()))
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], v0")
            self._emit(self.xdlops_mapping.get_gemm_index_for_dst_matrix(v.v_co_sst(), v.v_co_sld(), v.v_tmp(5), v.v_tmp()))

        self._emit(f"; LDS store, in: n0,n1b,c0,c1e: {t_n0}x{t_n1b}x{t_c0}x{t_c1e}, {c_n0}x{c_n1b}x{c_c0}x{c_c1e}, order:{gemm_n_order}")
        if gemm_n_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_N_C0_C1E:
            if c_c1e == 1:
                # TODO: remove this path, not possible go here
                assert c_c0 != 1
                self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(n_c1e)},  v[{v.v_gtc_ic0()}]")
            else:
                if c_c0 == 1:
                    self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_ic1e()}]")
                else:
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ic0()}], {igemm_log2(n_c1e)}, v[{v.v_gtc_ic1e()}]")
        else:
            assert t_c0 != 1
            if c_c1e == 1:
                # this is not prefered
                assert c_c0 != 1
                self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_ic0()}]")
            else:
                if c_c0 == 1:
                    self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(n_c0)}, v[{v.v_gtc_ic1e()}]")
                else:
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ic1e()}], {igemm_log2(n_c0)}, v[{v.v_gtc_ic0()}]")

        if c_n1b != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_in1b()}], {igemm_log2(n_c0*n_c1e)}, v[{v.v_tmp()}]")
        if c_n0 != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_in0()}], {igemm_log2(n_n1b*n_c0*n_c1e)}, v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_sst_b_os()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")
        self._emit(f"v_add_u32 v[{v.v_sst_b_os()}], {self.tunable.lds_a_np2}, v[{v.v_sst_b_os()}]")
        self._emit_empty_line()

        self._emit(f"; LDS store, out: n0,n1b,k0,k1: {t_n0}x{t_n1b}x{t_k0}x{t_k1}, {c_n0}x{c_n1b}x{c_k0}x{c_k1}, order:{gemm_m_order}")
        if gemm_m_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1:
            if c_k1 == 1:
                assert c_k0 != 1
                self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(n_k1)}, v[{v.v_gtc_ik0}]")
            else:
                if c_k0 == 1:
                    self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_ik1()}]")
                else:
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ik0()}], {igemm_log2(n_k1)}, v[{v.v_gtc_ik1()}]")
        else:
            if c_k1 == 1:
                assert c_k0 != 1
                self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_ik0}]")
            else:
                if c_k0 == 1:
                    self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(n_k0)}, v[{v.v_gtc_ik1()}]")
                else:
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ik1()}], {igemm_log2(n_k0)}, v[{v.v_gtc_ik0()}]")

        if c_n1b != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_in1b()}], {igemm_log2(n_k0*n_k1)}, v[{v.v_tmp()}]")
        if c_n0 != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_in0()}], {igemm_log2(n_n1b*n_k0*n_k1)}, v[{v.v_tmp()}]")
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

        self._emit(f"; weight offset")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_k()}], s[{s.s_wei_stride_k()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
        self._emit(f"s_add_u32 s[{s.s_p_wei()}], s[{s.s_p_wei()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_wei(1)}], s[{s.s_p_wei(1)}], s[{s.s_tmp(1)}]")
        if gemm_n_unmerge_cluster == 0:
            self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], s[{s.s_block_gtc_ic0()}], {igemm_log2(unmerge_sub_c1 * data_byte)}")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_wei_stride_c()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_wei_stride_c()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_add_u32 s[{s.s_p_wei()}], s[{s.s_p_wei()}], s[{s.s_tmp()}]")
            self._emit(f"s_addc_u32 s[{s.s_p_wei(1)}], s[{s.s_p_wei(1)}], s[{s.s_tmp(1)}]")
        else:
            pass
        self._emit_empty_line()
        self._emit(f"s_lshl_b32 s[{s.s_tmp()}+3], s[{s.s_block_gtc_ik()}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_wei_stride_k()}], s[{s.s_tmp()}+3]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp()}+1], s[{s.s_wei_stride_k()}], s[{s.s_tmp()}+3]")
        self._emit(f"s_add_u32 s[{s.s_p_wei()}], s[{s.s_p_wei()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_wei()}+1], s[{s.s_p_wei()}+1], s[{s.s_tmp()}+1]")
        self._emit_empty_line()
        self._emit(f"; compute v_co_sub_n_index along c0 x c1e : {n_c0}x{n_c1e}")
        if gemm_n_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_N_C0_C1E:
            if n_c1e != 1:
                self._emit(f"v_and_b32 v[{v.v_wei_ic1e()}], {n_c1e - 1}, v[{v.v_co_sub_n_index()}]     ; => C1E")
                if n_c0 != 1:
                    self._emit(f"v_lshrrev_b32 v[{v.v_wei_ic0()}], {igemm_log2(n_c1e)}, v[{v.v_co_sub_n_index()}]  ; => C0")
            else:
                assert n_c0 == self.tunable.block_size
                assert False, "un implemented, should rarely be used"
        else:
            if n_c0 != 1:
                self._emit(f"v_and_b32 v[{v.v_wei_ic0()}], {n_c0 - 1}, v[{v.v_co_sub_n_index()}]     ; => C0")
                if n_c1e != 0:
                    self._emit(f"v_lshrrev_b32 v[{v.v_wei_ic1e()}], {igemm_log2(n_c0)}, v[{v.v_co_sub_n_index()}]   ; => C1E")
                else:
                    assert False, "un implemented, should rarely be used"
            else:
                if n_c1e != 0:
                    self._emit(f"v_mov_b32 v[{v.v_wei_ic1e()}], v[{v.v_co_sub_n_index()}]   ; => C1E")
                else:
                    assert False, "un implemented, should rarely be used"

        self._emit(f";   compute from n1b")
        self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_ic1e()}], v[{v.v_wei_ic1e()}]")
        self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_wei_ic1(), v.v_tmp(5), s.s_wei_stride_c(), v.v_tmp(), s.s_tmp()))

        self._emit_empty_line()
        self._emit(f"; add wei_ic0, wei_ic1")
        if n_c0 != 1:
            if gemm_n_unmerge_cluster == 0:
                self._emit(f"v_lshl_or_b32 v[{v.v_tmp(1)}], v[{v.v_wei_ic0()}], {igemm_log2(unmerge_sub_c1)}, v[{v.v_wei_ic1()}]")
                self._emit(f"v_mul_lo_u32 v[{v.v_wei_os()}], s[{s.s_wei_stride_c()}], v[{v.v_tmp(1)}]")
            else:
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_c()}], v[{v.v_wei_ic1()}]")
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_wei_stride_c0()}], v[{v.v_wei_ic0()}]")
                self._emit(f"v_add_u32 v[{v.v_wei_os()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}]")
        else:
            self._emit(f"v_mul_lo_u32 v[{v.v_wei_os()}], s[{s.s_wei_stride_c()}], v[{v.v_wei_ic1()}]")

        self._emit(f"; add i_k")
        if gemm_m_unmerge_cluster == 0:
            if gemm_m_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1:
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_k()}], v[{v.v_co_sub_m_index()}]")
            else:
                if n_k0 == 1:
                    self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_k()}], v[{v.v_co_sub_m_index()}]")
                else:
                    if n_k1 == 1:
                        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_k()}], v[{v.v_co_sub_m_index()}]")
                    else:
                        self._emit(f"v_and_b32 v[{v.v_tmp()}], {n_k0 - 1}, v[{v.v_co_sub_m_index()}]        ; => k0")
                        self._emit(f"v_lshrrev_b32 v[{v.v_tmp(1)}], {igemm_log2(n_k0)}, v[{v.v_co_sub_m_index()}]       ; => k1")
                        self._emit(f"v_lshl_or_b32 v[{v.v_tmp(1)}], v[{v.v_tmp()}], {igemm_log2(n_k1)}, v[{v.v_tmp(1)}]")
                        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_k()}], v[{v.v_tmp(1)}]")
        else:
            if gemm_m_order == IGEMM_WRW_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1:
                self._emit(f"v_and_b32 v[{v.v_tmp()}], {n_k1 - 1}, v[{v.v_co_sub_m_index()}]    ; => k1")
                self._emit(f"v_lshrrev_b32 v[{v.v_tmp(1)}], {igemm_log2(n_k1)}, v[{v.v_co_sub_m_index()}]   ; => k0")
            else:
                self._emit(f"v_and_b32 v[{v.v_tmp(1)}], {n_k0 - 1}, v[{v.v_co_sub_m_index()}]    ; => k0")
                self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], {igemm_log2(n_k0)}, v[{v.v_co_sub_m_index()}]   ; => k1")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_wei_stride_k0()}] ,v[{v.v_tmp(1)}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_k()}] ,v[{v.v_tmp()}]")
            self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}]")


        self._emit(f"v_add_u32 v[{v.v_wei_os()}], v[{v.v_wei_os()}], v[{v.v_tmp()}]")
        self._emit(f"; add y, x")
        #self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_x()}], v[{v.v_wei_iy()}]")
        self._emit(f"v_add_u32 v[{v.v_wei_os()}], v[{v.v_wei_os()}], v[{v.v_tmp(4)}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_wei_os()}], {igemm_log2(data_byte)}, v[{v.v_wei_os()}]")


        self._emit(f"; move slice stride")
        assert n_n0 * n_n1b == self.tunable.gemm_k_per_block
        #if n_k0 != 1:
        #    self._emit(f"s_mov_b32 s[{s.s_move_slice_k_k0}], {n_k0}")
        if self.tunable.nxb != 0:
            self._emit(f"s_mov_b32 s[0], {n_n1b}")
            if self.tunable.nxe != 0:
                self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_move_slice_n_n1(), '0', s.s_dim_b(), v.v_tmp(4), v.v_tmp(), s.s_tmp()))
            else:
                if s.s_out_stride_k.label in self.dict_shifted_stride:
                    self._emit(f"s_lshr_b32 s[{s.s_tmp(5)}], s[{s.s_out_stride_k()}], {igemm_log2(data_byte)}")
                    self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_move_slice_n_n1(), '0', s.s_tmp(5), v.v_tmp(4), v.v_tmp(), s.s_tmp()))
                else:
                    self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_move_slice_n_n1(), '0', s.s_out_stride_k(), v.v_tmp(4), v.v_tmp(), s.s_tmp()))
            #self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_move_slice_n_n1(), '0', s.s_out_stride_k(), v.v_tmp(4), v.v_tmp(), s.s_tmp()))
            self._emit(m_int_div_rem_ss(s.s_move_slice_n_dswo(), s.s_move_slice_n_dsho(), s.s_tmp(4), s.s_wo(), v.v_tmp(4), v.v_tmp(), s.s_tmp()))
        else:
            pass
        self._emit_empty_line()

        m_move_slice_window = self.get_macro_move_slice_window()


        if self.tunable.nxb != 0:
            #assert s.s_out_stride_n.label not in self.dict_shifted_stride and s.s_in_stride_n.label not in self.dict_shifted_stride
            if s.s_out_stride_n.label not in self.dict_shifted_stride:
                _sym_s_out_stride_n = s.s_out_stride_n()
            else:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_out_stride_n()}], {utility_log2(data_byte)}")
                _sym_s_out_stride_n = s.s_tmp()

            if s.s_in_stride_n.label not in self.dict_shifted_stride:
                _sym_s_in_stride_n = s.s_in_stride_n()
            else:
                self._emit(f"s_lshr_b32 s[{s.s_tmp(1)}], s[{s.s_in_stride_n()}], {utility_log2(data_byte)}")
                _sym_s_in_stride_n = s.s_tmp(1)

            self._emit(m_move_slice_window.init_stride_n(_sym_s_in_stride_n, _sym_s_out_stride_n, s.s_in_stride_n_n1(), s.s_out_stride_n_n1(),
                                                        s.s_in_stride_n_n0_n1_diff(), s.s_out_stride_n_n0_n1_diff(), s.s_move_slice_n_n1()))
        else:
            assert False

        self._emit(self.try_shift_stride(s.s_in_stride_n_n1, igemm_log2(data_byte)))
        self._emit(self.try_shift_stride(s.s_out_stride_n_n1, igemm_log2(data_byte)))
        self._emit(self.try_shift_stride(s.s_in_stride_n_n0_n1_diff, igemm_log2(data_byte)))
        self._emit(self.try_shift_stride(s.s_out_stride_n_n0_n1_diff, igemm_log2(data_byte)))

        self._emit(self.try_shift_stride(s.s_in_stride_n, igemm_log2(data_byte)))
        self._emit(self.try_shift_stride(s.s_out_stride_n, igemm_log2(data_byte)))
        self._emit(self.try_shift_stride(s.s_wei_stride_k, igemm_log2(data_byte)))
        if gemm_m_unmerge_cluster == 1:
            self._emit(self.try_shift_stride(s.s_wei_stride_k0, igemm_log2(data_byte)))

        self._emit(f"s_mov_b32 s[{s.s_gemm_k_num_n1()}], {unmerge_sub_n1}")
        #if self.tunable.nxe != 0:
        if self.tunable.nxe != 0:
            self._emit(f"s_mul_i32 s[{s.s_knum()}], s[{s.s_dim_b()}], s[{s.s_sub_n()}]")
        else:
            if s.s_out_stride_k.label in self.dict_shifted_stride:
                self._emit(f"s_lshr_b32 s[{s.s_tmp(5)}], s[{s.s_out_stride_k()}], {igemm_log2(data_byte)}")
                self._emit(f"s_mul_i32 s[{s.s_knum()}],  s[{s.s_tmp(5)}], s[{s.s_sub_n()}]")
            else:
                self._emit(f"s_mul_i32 s[{s.s_knum()}], s[{s.s_out_stride_k()}], s[{s.s_sub_n()}]")
        # self._emit(f"s_mul_i32 s[{s.s_knum()}], s[{s.s_out_stride_k()}], s[{s.s_sub_n()}]")
        #else:
        #    self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_k()}]")

        self._emit_empty_line()

    def emit_kernel_fma_main_loop(self):
        s = self.sgpr
        v = self.vgpr
        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        def move_slice_window_b():
            if self.tunable.nxb != 0:
                
                m_in_update_os       = self.get_macro_in_update_os()
                m_in_update_hw       = self.get_macro_in_update_hw()
                m_set_flag_hw         = self.get_macro_set_flag_hw()
                with self._deferred_context():
                    if self.tunable.nxe != 0:
                        m_move_slice_window   = self.get_macro_move_slice_window_check_last_dim()
                        self._emit(m_move_slice_window(v.v_move_slice_n_in1(), v.v_move_slice_n_idsho(), v.v_move_slice_n_idswo(), s.s_gemm_k_num_n1(), s.s_ho(), s.s_wo(),
                            s.s_move_slice_n_n1(), s.s_move_slice_n_dsho(), s.s_move_slice_n_dswo(), v.v_in_os_base(), v.v_out_os_base(),
                            s.s_in_stride_n(), s.s_out_stride_n(), s.s_in_stride_n_n1(), s.s_out_stride_n_n1(), s.s_in_stride_n_n0_n1_diff(), s.s_out_stride_n_n0_n1_diff(),
                            v.v_move_slice_n_in0(), v.v_tmp(), s.s_sub_n(), v.v_flag_n()))
                        #self._emit(m_in_update_hw(v.v_in_ihi(), v.v_in_iwi(), v.v_move_slice_n_idsho(), v.v_move_slice_n_idswo()))
                        self._emit(m_in_update_hw(v.v_in_ihi(), v.v_in_iwi(), v.v_move_slice_n_idsho(), v.v_move_slice_n_idswo(), s.s_stride_h(), s.s_stride_w(), v.v_wei_iy(), v.v_wei_ix(), s.s_dilation_h(), s.s_dilation_w(), s.s_pad_h(), s.s_pad_w(), v.v_tmp()))
                        self._emit(m_in_update_os(v.v_in_os(), v.v_in_os_base(), v.v_in_ihi(), v.v_in_iwi(), s.s_wi(), v.v_tmp()))
                        self._emit(m_set_flag_hw(v.v_in_flag(), v.v_in_ihi(), v.v_in_iwi(), s.s_hi(), s.s_wi()))
                        self._emit(f"v_and_b32 v[{v.v_in_flag()}], v[{v.v_in_flag()}], v[{v.v_flag_n()}]")
                    else:
                        m_move_slice_window   = self.get_macro_move_slice_window()
                        self._emit(m_move_slice_window(v.v_move_slice_n_in1(), v.v_move_slice_n_idsho(), v.v_move_slice_n_idswo(), s.s_gemm_k_num_n1(), s.s_ho(), s.s_wo(),
                            s.s_move_slice_n_n1(), s.s_move_slice_n_dsho(), s.s_move_slice_n_dswo(), v.v_in_os_base(), v.v_out_os_base(),
                            s.s_in_stride_n(), s.s_out_stride_n(), s.s_in_stride_n_n1(), s.s_out_stride_n_n1(), s.s_in_stride_n_n0_n1_diff(), s.s_out_stride_n_n0_n1_diff()))
                        self._emit(m_in_update_os(v.v_in_os(), v.v_in_os_base(), v.v_move_slice_n_idsho(), v.v_move_slice_n_idswo(), s.s_wi(), v.v_tmp()))
                return self._get_deferred()
            else:
                assert False


        def move_slice_window_a():
            if self.tunable.nxb != 0:
                m_out_update_os   = self.get_macro_out_update_os()
                m_out_update_hw   = self.get_macro_out_update_hw()
                m_set_flag_hw     = self.get_macro_set_flag_hw()
                with self._deferred_context():
                    if self.tunable.nxe != 0:
                        # self._emit(m_out_update_hw(v.v_out_iho(), v.v_out_iwo(), v.v_move_slice_n_idsho(), v.v_move_slice_n_idswo()))
                        # self._emit(m_out_update_os(v.v_out_os(), v.v_out_os_base(), v.v_out_iho(), v.v_out_iwo(), s.s_wo(), v.v_tmp()))
                        # self._emit(m_set_flag_hw(v.v_out_flag(), v.v_out_iho(), v.v_out_iwo(), s.s_ho(), s.s_wo()))
                        self._emit(m_out_update_os(v.v_out_os(), v.v_out_os_base(), v.v_move_slice_n_idsho(), v.v_move_slice_n_idswo(), s.s_wo(), v.v_tmp()))
                        self._emit(m_set_flag_hw(v.v_out_flag(), v.v_move_slice_n_idsho(), v.v_move_slice_n_idswo(), s.s_ho(), s.s_wo()))
                        self._emit(f"v_and_b32 v[{v.v_out_flag()}], v[{v.v_out_flag()}], v[{v.v_flag_n()}]")
                    else:
                        self._emit(m_out_update_os(v.v_out_os(), v.v_out_os_base(), v.v_move_slice_n_idsho(), v.v_move_slice_n_idswo(), s.s_wo(), v.v_tmp()))
                return self._get_deferred()
            else:
                assert False

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
            fctrl.global_load_a_functor       = self.global_load_out
            fctrl.global_load_b_functor       = self.global_load_in
            fctrl.shared_store_a_functor      = self.shared_store_out
            fctrl.shared_store_b_functor      = self.shared_store_in
            fctrl.shared_load_a_functor       = inst_ds_read_t(self.tunable.thread_sub_tile_m * 4)
            fctrl.shared_load_b_functor       = inst_ds_read_t(self.tunable.thread_sub_tile_n * 4)
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
            ctrl_xdlops_mapping               = get_ctrl_xdlops_mapping_from_wave_tile(self.tunable.gemm_m_per_block, self.tunable.gemm_n_per_block,self.tunable.wave_tile_m, self.tunable.wave_tile_n, self.tunable.wave_tile_k,
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
            fctrl.global_load_a_functor       = self.global_load_out
            fctrl.global_load_b_functor       = self.global_load_in
            fctrl.shared_store_a_functor      = self.shared_store_out
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
            if self.tunable.nxb != 0:
                self._emit(self.coalescing_store(v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_wei(), v.v_wei_os(), None,
                    s.s_wei_stride_k0() if self.tunable.gemm_m_unmerge_cluster == 1 else None, s.s_wei_stride_k(), s.s_tmp(), None))
            else:
                assert False
        else:
            a = self.agpr
            if self.tunable.nxb != 0:
                self._emit(self.coalescing_store(a.a_c(), v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_wei(), v.v_wei_os(), None,
                    s.s_wei_stride_k0() if self.tunable.gemm_m_unmerge_cluster == 1 else None, s.s_wei_stride_k(), s.s_tmp(), None, s.s_k(), v.v_cur_k(), s.s_block_gtc_ik(), v.v_co_sub_m_index(), v.v_tmp()))
            else:
                assert False

        if IGEMM_WRW_GTC_DEBUG == 1:
            self._emit_empty_line()
            self._emit(f"s_branch {self.label_out}")
            self._emit("; debug code to cpy vgpr to host")
            if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
                self._emit(f"L_debug_{self.label_out}_0:")
            else: 
                self._emit(f"L_debug_{self.label_out}_1:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_waitcnt vmcnt(0)")
            self._emit("s_barrier")
            self._emit("s_cmp_lg_u32 s[s_bx], 0")
            #self._emit("s_cbranch_scc1  L_program_end_0")
            if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
                self._emit(f"s_cbranch_scc1 L_program_end_{self.label_out}_0")
            else: 
                self._emit(f"s_cbranch_scc1 L_program_end_{self.label_out}_1")
            self._emit(";s_cmp_lg_u32 s[s_wave_id], 0")
            self._emit(";s_cbranch_scc1  L_program_end")
            self._emit(";v_add_co_u32 v34, vcc, 0, v[v_a0+2]")
            self._emit("v_mov_b32 v[v_tmp], s[s_in_offset]")
            self._emit(f"s_mov_b32 s[{s.s_tmp()}], 0")
            self._emit_empty_line()

            self._emit(f"buffer_store_dword v[v_in_os], v[v_tmp+6], s[{s.s_p_wei((0,3))}], s[{s.s_tmp()}] offen")

            self._emit("s_waitcnt vmcnt(0)")
            self._emit("s_barrier")
            self._emit_empty_line()

            if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
                self._emit(f"L_program_end_{self.label_out}_0:")
            else: 
                self._emit(f"L_program_end_{self.label_out}_1:")
            self._emit("s_nop 2")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_waitcnt vmcnt(0)")
            self._emit("s_barrier")

        self._emit_empty_line()
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
