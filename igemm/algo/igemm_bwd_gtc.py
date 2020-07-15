################################################################################
# 
#  MIT License
# 
#  Copyright (c) 2020 Advanced Micro Devices, Inc.
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
from .fma_main_loop import *
from .igemm_base import *
from .global_memory import *
from .shared_memory import *
from .utility import *

def _find_non_1_index_in_list(list_object):
    result_list = list()
    for idx, item in enumerate(list_object):
        assert type(item) is int
        if item != 1:
            result_list.append(idx)
    return result_list

class macro_igemm_bwd_gtc_out_update_os_t(mc_base_t):
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
    def name(self):
        return '.v_bwd_gtc_out_update_os'
    def __call__(self, v_out_os, v_out_os_base, v_out_iho, v_out_iwo, s_wo, v_tmp):
        return '{} {}, {}, {}, {}, {}, {}'.format(self.name(), v_out_os, v_out_os_base, v_out_iho, v_out_iwo, s_wo, v_tmp)
    def emit(self):
        with self._emit_macro_indented('.macro {} v_out_os, v_out_os_base, v_out_iho, v_out_iwo, s_wo, v_tmp'.format(self.name())):
            self._emit(f"; from ho, wo, os_base, compute final offset")
            self._emit(f"v_mad_u32_u24 v[\\v_tmp], s[\\s_wo], v[\\v_out_iho], v[\\v_out_iwo]")
            self._emit(f"v_lshl_add_u32 v[\\v_out_os], v[\\v_tmp], {igemm_log2(amdgpu_precision_data_byte(self.tunable.precision))}, v[\\v_out_os_base]")

class macro_igemm_bwd_gtc_out_update_hw_t(mc_base_t):
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
    def name(self):
        return '.v_bwd_gtc_out_update_hw'
    def __call__(self, v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, s_out_dslice_iy, s_out_dslice_ix, s_dtile_dy, s_dtile_dx, s_tmp2):
        return '{} {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(self.name(),
            v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, s_out_dslice_iy, s_out_dslice_ix, s_dtile_dy, s_dtile_dx, s_tmp2)
    def emit(self):
        with self._emit_macro_indented('.macro {} v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, s_out_dslice_iy, s_out_dslice_ix, s_dtile_dy, s_dtile_dx, s_tmp2'.format(self.name())):
            self._emit(f"; dslice_y,dslice_h -> oh, dslice_x,dslice_w -> ow")
            self._emit(f"s_mul_i32 s[\\s_tmp2], s[\\s_dtile_dy], s[\\s_out_dslice_iy]")
            self._emit(f"v_subrev_u32 v[\\v_out_iho], s[\\s_tmp2], v[\\v_out_dslice_ih]")
            self._emit(f"s_mul_i32 s[\\s_tmp2+1], s[\\s_dtile_dx], s[\\s_out_dslice_ix]")
            self._emit(f"v_subrev_u32 v[\\v_out_iwo], s[\\s_tmp2+1], v[\\v_out_dslice_iw]")

class macro_igemm_bwd_gtc_wei_update_os_t(mc_base_t):
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
    def name(self):
        return '.v_bwd_gtc_wei_update_os'
    def __call__(self, v_wei_os, v_wei_os_base, v_wei_iy, v_wei_ix, s_x, v_tmp):
        return '{} {}, {}, {}, {}, {}, {}'.format(self.name(), v_wei_os, v_wei_os_base, v_wei_iy, v_wei_ix, s_x, v_tmp)
    def emit(self):
        with self._emit_macro_indented('.macro {} v_wei_os, v_wei_os_base, v_wei_iy, v_wei_ix, s_x, v_tmp'.format(self.name())):
            self._emit(f"; from y, x, os_base, compute final offset")
            self._emit(f"v_mad_u32_u24 v[\\v_tmp], v[\\v_wei_iy], s[\\s_x], v[\\v_wei_ix]")
            self._emit(f"v_lshl_add_u32 v[\\v_wei_os], v[\\v_tmp], {igemm_log2(amdgpu_precision_data_byte(self.tunable.precision))}, v[\\v_wei_os_base]")

class macro_igemm_bwd_gtc_wei_update_yx_t(mc_base_t):
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
    def name(self):
        return '.v_bwd_gtc_wei_update_yx'
    def __call__(self, v_wei_iy, v_wei_ix, s_move_slice_k_idsy, s_move_slice_k_idsx, s_dtile_y, s_dtile_x, v_dtile_iy, v_dtile_ix, s_tmp2):
        return '{} {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(self.name(),
            v_wei_iy, v_wei_ix, s_move_slice_k_idsy, s_move_slice_k_idsx, s_dtile_y, s_dtile_x, v_dtile_iy, v_dtile_ix, s_tmp2)
    def emit(self):
        with self._emit_macro_indented('.macro {} v_wei_iy, v_wei_ix, s_move_slice_k_idsy, s_move_slice_k_idsx, s_dtile_y, s_dtile_x, v_dtile_iy, v_dtile_ix, s_tmp2'.format(self.name())):
            self._emit(f"s_mul_i32 s[\\s_tmp2], s[\\s_dtile_y], s[\\s_move_slice_k_idsy]")
            self._emit(f"v_add_u32 v[\\v_wei_iy], s[\\s_tmp2], v[\\v_dtile_iy]")
            self._emit(f"s_mul_i32 s[\\s_tmp2+1], s[\\s_dtile_x], s[\\s_move_slice_k_idsx]")
            self._emit(f"v_add_u32 v[\\v_wei_ix], s[\\s_tmp2+1], v[\\v_dtile_ix]")

class macro_igemm_bwd_gtc_set_flag_hw(mc_base_t):
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
    def name(self):
        return '.v_set_flag_hw'
    def __call__(self, v_flag, v_ih, v_iw, s_h, s_w):
        return '{} {}, {}, {}, {}, {}'.format(self.name(), v_flag, v_ih, v_iw, s_h, s_w)
    def emit(self):
        with self._emit_macro_indented('.macro {} v_flag, v_ih, v_iw, s_h, s_w'.format(self.name())):
            self._emit(f"v_cmp_gt_u32 vcc, s[\\s_h], v[\\v_ih]")
            self._emit(f"v_cndmask_b32 v[\\v_flag], 0, 1, vcc")
            self._emit(f"v_cmp_gt_u32 vcc, s[\\s_w], v[\\v_iw]")
            self._emit(f"v_cndmask_b32 v[\\v_flag], 0, v[\\v_flag], vcc")

class macro_igemm_bwd_gtc_move_slice_window_k_dsy_dsx(mc_base_t):
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
    def name(self):
        return '.s_bwd_gtc_move_slice_window_k_dsy_dsx'
    def __call__(self, s_move_slice_k_ik, s_move_slice_k_idsy, s_move_slice_k_idsx, s_gemm_k_num_dsy, s_gemm_k_num_dsx, k_step_k, k_step_dsy, k_step_dsx, v_out_os_base, v_wei_os_base, s_out_move_slice_stride_k, s_wei_move_slice_stride_k):
        return '{} {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(self.name(),
            s_move_slice_k_ik, s_move_slice_k_idsy, s_move_slice_k_idsx, s_gemm_k_num_dsy, s_gemm_k_num_dsx, k_step_k, k_step_dsy, k_step_dsx, v_out_os_base, v_wei_os_base, s_out_move_slice_stride_k, s_wei_move_slice_stride_k)
    def emit(self):
        with self._emit_macro_indented('.macro {} s_move_slice_k_ik, s_move_slice_k_idsy, s_move_slice_k_idsx, s_gemm_k_num_dsy, s_gemm_k_num_dsx, k_step_k, k_step_dsy, k_step_dsx, v_out_os_base, v_wei_os_base, s_out_move_slice_stride_k, s_wei_move_slice_stride_k'.format(self.name())):
            self._emit(f"; order always x->y->k first")
            self._emit(f"; if k=k0*k1, always iterate k then compute back to k0, k1")
            self._emit(f"; update offset if k need step, for both output, weight")
            self._emit(f"s_add_u32 s[\\s_move_slice_k_idsx], \\k_step_dsx, s[\\s_move_slice_k_idsx]")
            self._emit(f"s_cmp_lt_u32 s[\\s_move_slice_k_idsx], s[\\s_gemm_k_num_dsx]")
            self._emit(f"s_cbranch_scc1 s_bwd_gtc_move_slice_window_k_dsy_dsx_L0_\\@")
            self._emit(f"s_mov_b32 s[\\s_move_slice_k_idsx], 0")
            self._emit(f"s_add_u32 s[\\s_move_slice_k_idsy], \\k_step_dsy, s[\\s_move_slice_k_idsy]")
            self._emit(f"s_bwd_gtc_move_slice_window_k_dsy_dsx_L0_\\@:")

            self._emit(f"s_cmp_lt_u32 s[\\s_move_slice_k_idsy], s[\\s_gemm_k_num_dsy]")
            self._emit(f"s_cbranch_scc1 s_bwd_gtc_move_slice_window_k_dsy_dsx_L1_\\@")
            self._emit(f"v_add_u32 v[\\v_out_os_base], s[\\s_out_move_slice_stride_k], v[\\v_out_os_base]")
            self._emit(f"s_mov_b32 s[\\s_move_slice_k_idsy], 0")
            self._emit(f"v_add_u32 v[\\v_wei_os_base], s[\\s_wei_move_slice_stride_k], v[\\v_wei_os_base]")
            self._emit(f"s_add_u32 s[\\s_move_slice_k_ik], \\k_step_k, s[\\s_move_slice_k_ik]")
            self._emit(f"s_bwd_gtc_move_slice_window_k_dsy_dsx_L1_\\@:")

class igemm_bwd_gtc_t(mc_base_t):
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
        self.karg = self.kernel_karg_t(mc, tunable)
        self.sgpr = self.kernel_sgpr_t(mc, tunable)
        self.vgpr = self.kernel_vgpr_t(mc, tunable)
        self.global_load_out = self.global_load_out_t(mc, self)
        self.global_load_wei = self.global_load_wei_t(mc, self)
        self.shared_store_out = self.shared_store_out_t(mc, self)
        self.shared_store_wei = self.shared_store_wei_t(mc, self)

        out_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()
        self.out_thread_copy_nd = len(out_thread_copy_index)
        self.wei_thread_copy_nd = len(wei_thread_copy_index)
        assert self.out_thread_copy_nd in (1, 2)
        assert self.wei_thread_copy_nd in (1, 2)

    def name(self):
        return igemm_gtc_encode_kernel_name(self.tunable)

    class global_load_out_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_out_2d_global_load, m_wei_2d_global_load = self.outer.get_macro_global_load()
            return m_out_2d_global_load.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr

            m_out_2d_global_load, m_wei_2d_global_load = self.outer.get_macro_global_load()
            s_out_stride_d0, s_out_stride_d1, s_wei_stride_d0, s_wei_stride_d1 = self.outer.get_global_load_s_stride_d0_d1()
            with self._deferred_context():
                self._emit(f"; load output")
                self._emit(f"v_cmp_eq_u32 vcc, 1, v[{v.v_out_flag()}]")
                self._emit(f"s_and_saveexec_b64 s[{s.s_tmp()}+4:{s.s_tmp()}+5], vcc")
                self._emit(m_out_2d_global_load(v.v_gld_b(), s.s_p_out(), v.v_out_os(), s_out_stride_d0(), s_out_stride_d1(), s.s_tmp()))
                self._emit(f"s_or_b64 exec, exec, s[{s.s_tmp()}+4:{s.s_tmp()}+5]")
            return self._get_deferred()

    class global_load_wei_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_out_2d_global_load, m_wei_2d_global_load = self.outer.get_macro_global_load()
            return m_wei_2d_global_load.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr

            m_out_2d_global_load, m_wei_2d_global_load = self.outer.get_macro_global_load()
            s_out_stride_d0, s_out_stride_d1, s_wei_stride_d0, s_wei_stride_d1 = self.outer.get_global_load_s_stride_d0_d1()
            with self._deferred_context():
                self._emit(f"; load weight")
                self._emit(m_wei_2d_global_load(v.v_gld_a(), s.s_p_wei(), v.v_wei_os(), s_wei_stride_d0(), s_wei_stride_d1(), s.s_tmp()))
            return self._get_deferred() 

    class shared_store_out_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_out_2d_shared_store, m_wei_2d_shared_store = self.outer.get_macro_shared_store()
            return  m_out_2d_shared_store.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr
            m_out_2d_shared_store, m_wei_2d_shared_store = self.outer.get_macro_shared_store()
            with self._deferred_context():
                self._emit(m_out_2d_shared_store(v.v_gld_b(), v.v_sst_b_os()))
            return self._get_deferred()

    class shared_store_wei_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_out_2d_shared_store, m_wei_2d_shared_store = self.outer.get_macro_shared_store()
            return m_wei_2d_shared_store.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr
            m_out_2d_shared_store, m_wei_2d_shared_store = self.outer.get_macro_shared_store()
            with self._deferred_context():
                self._emit(m_wei_2d_shared_store(v.v_gld_a(), v.v_sst_a_os()))
            return self._get_deferred()

    class kernel_karg_t(mc_base_t):
        def __init__(self, mc, tunable):
            mc_base_t.__init__(self, mc)
            self.tunable         = tunable

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
            self.k_dtile_iy      = sym_t("k_dtile_iy",      84)
            self.k_dtile_ix      = sym_t("k_dtile_ix",      88)
            self.k_dtile_dy      = sym_t("k_dtile_dy",      92)
            self.k_dtile_dx      = sym_t("k_dtile_dx",      96)
            self.k_dtile_y       = sym_t("k_dtile_y",       100)
            self.k_dtile_x       = sym_t("k_dtile_x",       104)
            self.k_dtile_h       = sym_t("k_dtile_h",       108)
            self.k_dtile_w       = sym_t("k_dtile_w",       112)
            self.k_dslice_y      = sym_t("k_dslice_y",      116)
            self.k_dslice_x      = sym_t("k_dslice_x",      120)
            self.k_dslice_h      = sym_t("k_dslice_h",      124)
            self.k_dslice_w      = sym_t("k_dslice_w",      128)
            self.k_dslice_h_left = sym_t("k_dslice_h_left", 132)
            self.k_dslice_w_left = sym_t("k_dslice_w_left", 136)
            self.k_pack0         = sym_t("k_pack0",         140)
            self.k_end           = sym_t("k_end",           144)

        def get_count(self):
            return self.k_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('k_'):
                    self._emit(v.declare())

    class kernel_sgpr_t(mc_base_t):
        def __init__(self, mc, tunable):
            mc_base_t.__init__(self, mc)
            self.tunable = tunable

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
            self.s_dtile_iy                = sym_t("s_dtile_iy"               ,31)
            self.s_dtile_ix                = sym_t("s_dtile_ix"               ,32)
            self.s_dtile_dy                = sym_t("s_dtile_dy"               ,33)
            self.s_dtile_dx                = sym_t("s_dtile_dx"               ,34)
            self.s_dtile_y                 = sym_t("s_dtile_y"                ,35)
            self.s_dtile_x                 = sym_t("s_dtile_x"                ,36)
            self.s_dtile_h                 = sym_t("s_dtile_h"                ,37)
            self.s_dtile_w                 = sym_t("s_dtile_w"                ,38)
            self.s_dslice_y                = sym_t("s_dslice_y"               ,39)
            self.s_dslice_x                = sym_t("s_dslice_x"               ,40)
            self.s_dslice_h                = sym_t("s_dslice_h"               ,41)
            self.s_dslice_w                = sym_t("s_dslice_w"               ,42)
            self.s_dslice_h_left           = sym_t("s_dslice_h_left"          ,43)
            self.s_dslice_w_left           = sym_t("s_dslice_w_left"          ,44)

            self.s_out_stride_k            = sym_t("s_out_stride_k"           ,45)
            self.s_out_stride_k0           = sym_t("s_out_stride_k0"          ,46)
            self.s_out_stride_n            = sym_t("s_out_stride_n"           ,47)
            self.s_out_stride_n0           = sym_t("s_out_stride_n0"          ,48)
            self.s_out_stride_b0           = sym_t("s_out_stride_b0"          ,49)
            self.s_out_move_slice_stride_k = sym_t("s_out_move_slice_stride_k",50)
            self.s_in_stride_c             = sym_t("s_in_stride_c"            ,51)
            self.s_in_stride_cr            = sym_t("s_in_stride_cr"           ,52)
            self.s_in_stride_n             = sym_t("s_in_stride_n"            ,53)
            self.s_in_stride_nr            = sym_t("s_in_stride_nr"           ,54)

            self.s_wei_stride_c            = sym_t("s_wei_stride_c"           ,55)
            self.s_wei_stride_c0           = sym_t("s_wei_stride_c0"          ,56)
            self.s_wei_stride_k            = sym_t("s_wei_stride_k"           ,57)
            self.s_wei_stride_k0           = sym_t("s_wei_stride_k0"          ,58)
            self.s_wei_move_slice_stride_k = sym_t("s_wei_move_slice_stride_k",59)
            self.s_stride_dslice_hw        = sym_t("s_stride_dslice_hw"       ,60)
            self.s_stride_dslice_yx        = sym_t("s_stride_dslice_yx"       ,61)

            self.s_block_gtc_ib            = sym_t("s_block_gtc_ib"           ,62)
            self.s_block_gtc_ic            = sym_t("s_block_gtc_ic"           ,63)
            self.s_block_gtc_in            = sym_t("s_block_gtc_in"           ,64)
            self.s_knum                    = sym_t("s_knum"                   ,self.s_k.value)

            self.s_move_slice_k_idsy       = sym_t("s_move_slice_k_idsy"      ,0)
            self.s_move_slice_k_idsx       = sym_t("s_move_slice_k_idsx"      ,1)
            self.s_move_slice_k_ik         = sym_t("s_move_slice_k_ik"        ,2)
            self.s_kitr                    = sym_t("s_kitr"                   ,3)
            self.s_tmp                     = sym_t("s_tmp"                    ,66)
            self.s_end                     = sym_t("s_end"                    ,72)

        def get_count(self):
            return self.s_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('s_'):
                    self._emit(v.declare())

    class kernel_vgpr_t(mc_base_t):
        def __init__(self, mc, tunable):
            mc_base_t.__init__(self, mc)
            self.tunable            = tunable
            self.v_c             = sym_t("v_c"            ,0)
            self.v_a             = sym_t("v_a"            ,64)
            self.v_b             = sym_t("v_b"            ,72)
            self.v_gld_a         = sym_t("v_gld_a"        ,80)
            self.v_gld_b         = sym_t("v_gld_b"        ,88)
            self.v_sst_a_os      = sym_t("v_sst_a_os"     ,96)
            self.v_sst_b_os      = sym_t("v_sst_b_os"     ,97)
            self.v_sld_a_os      = sym_t("v_sld_a_os"     ,98)
            self.v_sld_b_os      = sym_t("v_sld_b_os"     ,99)
            self.v_out_iho       = sym_t("v_out_iho"      ,100)
            self.v_out_iwo       = sym_t("v_out_iwo"      ,101)
            self.v_out_dslice_ih = sym_t("v_out_dslice_ih",102)
            self.v_out_dslice_iw = sym_t("v_out_dslice_iw",103)
            self.v_out_os        = sym_t("v_out_os"       ,104)
            self.v_out_os_base   = sym_t("v_out_os_base"  ,105)
            self.v_wei_iy        = sym_t("v_wei_iy"       ,106)
            self.v_wei_ix        = sym_t("v_wei_ix"       ,107)
            self.v_dtile_iy      = sym_t("v_dtile_iy"     ,108)
            self.v_dtile_ix      = sym_t("v_dtile_ix"     ,109)
            self.v_wei_os        = sym_t("v_wei_os"       ,110)
            self.v_wei_os_base   = sym_t("v_wei_os_base"  ,111)
            self.v_out_flag      = sym_t("v_out_flag"     ,112)
            self.v_in_flag       = sym_t("v_in_flag"      ,113)
            self.v_in_os         = sym_t("v_in_os"        ,114)

            self.v_gtc_ic0       = sym_t("v_gtc_ic0"      ,63)
            self.v_gtc_ic1       = sym_t("v_gtc_ic1"      ,62)
            self.v_gtc_in0       = sym_t("v_gtc_in0"      ,61)
            self.v_gtc_in1       = sym_t("v_gtc_in1"      ,60)
            self.v_gtc_ib0       = sym_t("v_gtc_ib0"      ,59)
            self.v_gtc_ib1       = sym_t("v_gtc_ib1"      ,58)
            self.v_gtc_ik0       = sym_t("v_gtc_ik0"      ,57)
            self.v_gtc_ik1       = sym_t("v_gtc_ik1"      ,56)
            self.v_gtc_ie        = sym_t("v_gtc_ie"       ,55)
            self.v_gemm_in       = sym_t("v_gemm_in"      ,54)
            self.v_gemm_im       = sym_t("v_gemm_im"      ,53)

            self.v_in_in0        = sym_t("v_in_in0"       ,52)
            self.v_in_in1        = sym_t("v_in_in1"       ,51)
            self.v_in_ib0        = sym_t("v_in_ib0"       ,50)
            self.v_in_ib1        = sym_t("v_in_ib1"       ,49)
            self.v_in_ic0        = sym_t("v_in_ic0"       ,48)
            self.v_in_ic1        = sym_t("v_in_ic1"       ,47)
            self.v_in_ihi        = sym_t("v_in_ihi"       ,46)
            self.v_in_iwi        = sym_t("v_in_iwi"       ,45)
            self.v_in_dslice_ih  = sym_t("v_in_dslice_ih" ,44)
            self.v_in_dslice_iw  = sym_t("v_in_dslice_iw" ,43)

            self.v_tmp           = sym_t("v_tmp"          ,116)
            self.v_end           = sym_t("v_end"          ,122)

        def get_count(self):
            return self.v_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('v_'):
                    self._emit(v.declare())

    def get_thread_lengths(self):
        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths
        t_k0, t_k1, t_e, t_n0, t_n1, t_b0, t_b1 = t_tb[0], t_tb[1], t_tb[2], t_tb[3], t_tb[4], t_tb[5], t_tb[6]
        _, _, _, t_c0, t_c1 = t_ta[0], t_ta[1], t_ta[2], t_ta[3], t_ta[4]

        assert t_e == 1, 'currently e dimension always not do load'
        assert (t_k0, t_k1, t_e, t_n0, t_n1, t_b0, t_b1).count(1) in (5, 6), 'currently only support at most 2 dimension is non-1'
        assert (t_k0, t_k1, t_e, t_c0, t_c1).count(1) in (3, 4), 'currently only support at most 2 dimension is non-1'

        return t_c0, t_c1, t_k0, t_k1, t_e, t_n0, t_n1, t_b0, t_b1 # M, K, N

    def get_cluster_lengths(self):
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths
        c_k0, c_k1, c_e, c_n0, c_n1, c_b0, c_b1 = c_tb[0], c_tb[1], c_tb[2], c_tb[3], c_tb[4], c_tb[5], c_tb[6]
        _, _, _, c_c0, c_c1 = c_ta[0], c_ta[1], c_ta[2], c_ta[3], c_ta[4]
        return c_c0, c_c1, c_k0, c_k1, c_e, c_n0, c_n1, c_b0, c_b1 # M, K, N

    def get_dims_lengths(self):
        t_c0, t_c1, t_k0, t_k1, t_e, t_n0, t_n1, t_b0, t_b1 = self.get_thread_lengths()
        c_c0, c_c1, c_k0, c_k1, c_e, c_n0, c_n1, c_b0, c_b1 = self.get_cluster_lengths()

        n_c0, n_c1, n_k0, n_k1, n_e, n_n0, n_n1, n_b0, n_b1 = \
                t_c0*c_c0, t_c1*c_c1, t_k0*c_k0, t_k1*c_k1, t_e*c_e, t_n0*c_n0, t_n1*c_n1, t_b0*c_b0, t_b1*c_b1
        return n_c0, n_c1, n_k0, n_k1, n_e, n_n0, n_n1, n_b0, n_b1

    def get_thread_copy_dims(self):
        t_c0, t_c1, t_k0, t_k1, t_e, t_n0, t_n1, t_b0, t_b1 = self.get_thread_lengths()
        out_thread_copy_dims    = [t_k0, t_k1, t_e, t_n0, t_n1, t_b0, t_b1]
        wei_thread_copy_dims    = [t_k0, t_k1, t_e, t_c0, t_c1]
        return out_thread_copy_dims, wei_thread_copy_dims

    def get_thread_copy_index(self):
        out_thread_copy_dims, wei_thread_copy_dims = self.get_thread_copy_dims()
        out_thread_copy_index   = _find_non_1_index_in_list(out_thread_copy_dims)
        wei_thread_copy_index   = _find_non_1_index_in_list(wei_thread_copy_dims)
        assert len(out_thread_copy_index) in (1, 2) and len(wei_thread_copy_index) in (1, 2),\
                f'out_thread_copy_dims:{out_thread_copy_dims} wei_thread_copy_dims:{wei_thread_copy_dims}'
        return out_thread_copy_index, wei_thread_copy_index

    def get_macro_global_load(self):
        t_c0, t_c1, t_k0, t_k1, t_e, t_n0, t_n1, t_b0, t_b1 = self.get_thread_lengths()
        out_thread_copy_dims, wei_thread_copy_dims = self.get_thread_copy_dims()
        out_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()

        ctrl_out_gld = ctrl_2d_global_load_t()
        ctrl_wei_gld = ctrl_2d_global_load_t()

        if self.out_thread_copy_nd == 2:
            ctrl_out_gld.length_d0 = out_thread_copy_dims[out_thread_copy_index[0]]
            ctrl_out_gld.length_d1 = out_thread_copy_dims[out_thread_copy_index[1]]
            ctrl_out_gld.vector_d1 = t_b1 if t_b1 != 1 else 1
        elif self.out_thread_copy_nd == 1:
            ctrl_out_gld.length_d0 = 1
            ctrl_out_gld.length_d1 = out_thread_copy_dims[out_thread_copy_index[0]]
            ctrl_out_gld.vector_d1 = t_b1 if t_b1 != 1 else 1
        else:
            assert False

        if self.wei_thread_copy_nd == 2:
            ctrl_wei_gld.length_d0 = wei_thread_copy_dims[wei_thread_copy_index[0]]
            ctrl_wei_gld.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[1]]
            ctrl_wei_gld.vector_d1 = 1
        elif self.wei_thread_copy_nd == 1:
            ctrl_wei_gld.length_d0 = 1
            ctrl_wei_gld.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]
            ctrl_wei_gld.vector_d1 = 1
        else:
            assert False

        return macro_igemm_2d_global_load_t(self.mc, ctrl_out_gld),  macro_igemm_2d_global_load_t(self.mc, ctrl_wei_gld)


    def get_macro_global_store(self):
        return macro_igemm_write_4d_strided_t(self.mc)

    def get_macro_shared_store(self):
        out_thread_copy_dims, wei_thread_copy_dims = self.get_thread_copy_dims()
        out_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()
        n_c0, n_c1, n_k0, n_k1, n_e, n_n0, n_n1, n_b0, n_b1 = self.get_dims_lengths()
        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        # K0xK1xExN0xB0xB1xN1 n_k0, n_k1, n_e, n_n0, n_b0, n_b1, n_n1
        out_stride_list = [n_k1*n_e*n_n0*n_b0*n_b1*n_n1, n_e*n_n0*n_b0*n_b1*n_n1, n_n0*n_b0*n_b1*n_n1, n_b0*n_b1*n_n1, n_b1*n_n1, n_n1, 1]
        out_reorder_list = [0, 1, 2, 3, 6, 4, 5]
        # K0xK1xExC0xC1, n_k0, n_k1, n_e, n_c0, n_c1
        wei_stride_list = [n_k1*n_e*n_c0*n_c1, n_e*n_c0*n_c1, n_c0*n_c1, n_c1, 1]
        wei_reorder_list = [0, 1, 2, 3, 4]

        # K0xK1xExN0xB0xB1xN1
        out_sst_ctrl = ctrl_2d_shared_store_t()
        wei_sst_ctrl = ctrl_2d_shared_store_t()

        if self.out_thread_copy_nd == 2:
            out_sst_ctrl.length_d0 = out_thread_copy_dims[out_thread_copy_index[0]]
            out_sst_ctrl.length_d1 = out_thread_copy_dims[out_thread_copy_index[1]]
            out_sst_ctrl.vector_d1 = out_thread_copy_dims[out_thread_copy_index[1]]
            out_sst_ctrl.stride_d0 = out_stride_list[out_reorder_list[out_thread_copy_index[0]]] * data_byte
        elif self.out_thread_copy_nd == 1:
            out_sst_ctrl.length_d0 = 1
            out_sst_ctrl.length_d1 = out_thread_copy_dims[out_thread_copy_index[0]]
            out_sst_ctrl.vector_d1 = out_thread_copy_dims[out_thread_copy_index[0]]
            out_sst_ctrl.stride_d0 = 1
        else:
            assert False

        if self.wei_thread_copy_nd == 2:
            wei_sst_ctrl.length_d0 = wei_thread_copy_dims[wei_thread_copy_index[0]]
            wei_sst_ctrl.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[1]]
            wei_sst_ctrl.vector_d1 = wei_thread_copy_dims[wei_thread_copy_index[1]]
            wei_sst_ctrl.stride_d0 = wei_stride_list[wei_reorder_list[wei_thread_copy_index[0]]] * data_byte
        elif self.wei_thread_copy_nd == 1:
            wei_sst_ctrl.length_d0 = 1
            wei_sst_ctrl.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]
            wei_sst_ctrl.vector_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]
            wei_sst_ctrl.stride_d0 = 1
        else:
            assert False

        return macro_igemm_2d_shared_store_t(self.mc, out_sst_ctrl), macro_igemm_2d_shared_store_t(self.mc, wei_sst_ctrl)

    def get_macro_shared_load(self):
        return None

    def get_macro_out_update_os(self):
        return macro_igemm_bwd_gtc_out_update_os_t(self.mc, self.tunable)
    def get_macro_out_update_hw(self):
        return macro_igemm_bwd_gtc_out_update_hw_t(self.mc, self.tunable)
    def get_macro_wei_update_os(self):
        return macro_igemm_bwd_gtc_wei_update_os_t(self.mc, self.tunable)
    def get_macro_wei_update_yx(self):
        return macro_igemm_bwd_gtc_wei_update_yx_t(self.mc, self.tunable)
    def get_macro_set_flag_hw(self):
        return macro_igemm_bwd_gtc_set_flag_hw(self.mc, self.tunable)

    def get_macro_move_slice_window(self):
        return macro_igemm_bwd_gtc_move_slice_window_k_dsy_dsx(self.mc, self.tunable)


    def get_global_load_s_stride_d0_d1(self):
        t_c0, t_c1, t_k0, t_k1, t_e, t_n0, t_n1, t_b0, t_b1 = self.get_thread_lengths()
        # get the symbol object that load 2d may use
        s = self.sgpr
        s_dummy = sym_t("0")
        out_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()
        out_stride_gprs = [s.s_out_stride_k0 if t_k0 != 1 else s_dummy, s.s_out_stride_k, s_dummy,
                    s.s_out_stride_n0 if t_n0 != 1 else s_dummy, s.s_out_stride_n, s.s_out_stride_b0 if t_b0 != 1 else s_dummy, s_dummy]
        wei_stride_gprs = [s.s_wei_stride_k0 if t_k0 != 1 else s_dummy, s.s_wei_stride_k, s_dummy,
                    s.s_wei_stride_c0 if t_c0 != 1 else s_dummy, s.s_wei_stride_c]

        if self.out_thread_copy_nd == 2:
            s_out_stride_d0 = out_stride_gprs[out_thread_copy_index[0]]
            s_out_stride_d1 = out_stride_gprs[out_thread_copy_index[1]]
        elif self.out_thread_copy_nd == 1:
            s_out_stride_d0 = s_dummy
            s_out_stride_d1 = out_stride_gprs[out_thread_copy_index[0]]
        else:
            assert False

        if self.wei_thread_copy_nd == 2:
            s_wei_stride_d0 = wei_stride_gprs[wei_thread_copy_index[0]]
            s_wei_stride_d1 = wei_stride_gprs[wei_thread_copy_index[1]]
        elif self.wei_thread_copy_nd == 1:
            s_wei_stride_d0 = s_dummy
            s_wei_stride_d1 = wei_stride_gprs[wei_thread_copy_index[0]]
        else:
            assert False

        return s_out_stride_d0, s_out_stride_d1, s_wei_stride_d0, s_wei_stride_d1


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
            int dtile_iy;
            int dtile_ix;
            int dtile_dy;
            int dtile_dx;
            int dtile_y;
            int dtile_x;
            int dtile_h;
            int dtile_w;
            int dslice_y;
            int dslice_x;
            int dslice_h;
            int dslice_w;
            int dslice_h_left;
            int dslice_w_left;
            int __pack0;
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
        kas.append(amdgpu_kernel_arg_t('dtile_iy'      , 4,  84, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_ix'      , 4,  88, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_dy'      , 4,  92, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_dx'      , 4,  96, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_y'       , 4, 100, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_x'       , 4, 104, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_h'       , 4, 108, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_w'       , 4, 112, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dslice_y'      , 4, 116, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dslice_x'      , 4, 120, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dslice_h'      , 4, 124, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dslice_w'      , 4, 128, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dslice_h_left' , 4, 132, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dslice_w_left' , 4, 136, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('__pack0'       , 4, 140, 'by_value', 'i32'))
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
                if type(rtn) is tuple:
                    kernel_macros.extend([m for m in rtn])
                else:
                    kernel_macros.append(rtn)
        return kernel_macros

    


    def emit_kernel_prologue(self):
        s = self.sgpr
        v = self.vgpr
        k = self.karg

        t_c0, t_c1, t_k0, t_k1, t_e, t_n0, t_n1, t_b0, t_b1 = self.get_thread_lengths()
        c_c0, c_c1, c_k0, c_k1, c_e, c_n0, c_n1, c_b0, c_b1 = self.get_cluster_lengths()
        n_c0, n_c1, n_k0, n_k1, n_e, n_n0, n_n1, n_b0, n_b1 = self.get_dims_lengths()

        n_k, n_n, n_b, n_c = n_k0*n_k1, n_n0*n_n1, n_b0*n_b1, n_c0*n_c1

        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        m_out_update_os   = self.get_macro_out_update_os()
        m_out_update_hw   = self.get_macro_out_update_hw()
        m_wei_update_os   = self.get_macro_wei_update_os()
        m_wei_update_yx   = self.get_macro_wei_update_yx()
        m_set_flag_hw     = self.get_macro_set_flag_hw()

        m_out_2d_global_load, m_wei_2d_global_load = self.get_macro_global_load()
        s_out_stride_d0, s_out_stride_d1, s_wei_stride_d0, s_wei_stride_d1 = self.get_global_load_s_stride_d0_d1()

        def emit_igemm_index_from_thread_cluster(v_idx, v_tmp, cluster_length, thread_length, is_last_dim = False):
            if cluster_length != 1:
                self._emit(f"v_and_b32 v[{v_idx()}], {cluster_length-1}, v[{v_tmp()}]")
                if not is_last_dim:
                    self._emit(f"v_lshrrev_b32 v[{v_tmp()}], {igemm_log2(cluster_length)}, v[{v_tmp()}]")
                if thread_length != 1:
                    self._emit(f"v_lshlrev_b32 v[{v_idx()}], {igemm_log2(thread_length)}, v[{v_idx()}]")
            else:
                pass
                ''' if is 1, will not specify value! '''
                # self._emit(f"v_mov_b32 v[{v_idx()}], 0")

        def emit_igemm_accumulate_x0_x1(v_x0, v_x1, c_x0, c_x1, n_x0, n_x1):
            if c_x0 == 1 and c_x1 == 1:
                self._emit(f"v_mov_b32 v[{v_x1()}], 0")
            elif c_x0 == 1 and c_x1 != 1:
                pass
            elif c_x0 != 1 and c_x1 == 1:
                if n_x1 == 1:
                    self._emit(f"v_mov_b32 v[{v_x1()}], v[{v_x0()}]")
                else:
                    self._emit(f"v_lshlrev_b32 v[{v_x1()}], {igemm_log2(n_x1)}, v[{v_x0()}]")
            else:
                self._emit(f"v_lshl_or_b32 v[{v_x1()}], v[{v_x0()}], {igemm_log2(n_x1)}, v[{v_x1()}]")


        self._emit(f"s_load_dwordx2  s[{s.s_p_in((0,1))}],       s[{s.s_ka((0, 1))}],    0+{k.k_p_in()}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_wei((0,1))}],      s[{s.s_ka((0, 1))}],    0+{k.k_p_wei()}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_out((0,1))}],      s[{s.s_ka((0, 1))}],    0+{k.k_p_out()}")
        self._emit(f"s_load_dwordx16 s[{s.s_hi((0,15))}],        s[{s.s_ka((0, 1))}],    0+{k.k_hi()}")
        self._emit(f"s_load_dwordx8  s[{s.s_dtile_ix((0,7))}],   s[{s.s_ka((0, 1))}],    0+{k.k_dtile_ix()}")
        self._emit(f"s_load_dwordx4  s[{s.s_dslice_x((0,3))}],   s[{s.s_ka((0, 1))}],    0+{k.k_dslice_x()}")
        self._emit(f"s_load_dword    s[{s.s_dslice_w_left()}],   s[{s.s_ka((0, 1))}],    0+{k.k_dslice_w_left()}")

        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        # output: K0xK1xExN0xN1xB0xB1
        self._emit(f"; output: K0xK1xExN0xN1xB0xB1: {t_k0}x{t_k1}x{t_e}x{t_n0}x{t_n1}x{t_b0}x{t_b1}, slice {c_k0}x{c_k1}x{c_e}x{c_n0}x{c_n1}x{c_b0}x{c_b1}")
        emit_igemm_index_from_thread_cluster(v.v_gtc_ib1, v.v_tmp, c_b1, t_b1)
        emit_igemm_index_from_thread_cluster(v.v_gtc_ib0, v.v_tmp, c_b0, t_b0)
        emit_igemm_index_from_thread_cluster(v.v_gtc_in1, v.v_tmp, c_n1, t_n1)
        emit_igemm_index_from_thread_cluster(v.v_gtc_in0, v.v_tmp, c_n0, t_n0)
        emit_igemm_index_from_thread_cluster(v.v_gtc_ie,  v.v_tmp, c_e,  t_e)
        emit_igemm_index_from_thread_cluster(v.v_gtc_ik1, v.v_tmp, c_k1, t_k1)
        emit_igemm_index_from_thread_cluster(v.v_gtc_ik0, v.v_tmp, c_k0, t_k0, True)
        self._emit_empty_line()
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        self._emit(f"; wei: K0xK1xExC0xC1: {t_k0}x{t_k1}x{t_e}x{t_c0}x{t_c1}, slice {c_k0}x{c_k1}x{c_e}x{c_c0}x{c_c1}")
        emit_igemm_index_from_thread_cluster(v.v_gtc_ic1, v.v_tmp, c_c1, t_c1)
        emit_igemm_index_from_thread_cluster(v.v_gtc_ic0, v.v_tmp, c_c0, t_c0, True)
        self._emit_empty_line()
        self._emit(f"s_mov_b32 s[{s.s_p_in()}  + 2], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_in()}  + 3], 0x27000")
        self._emit(f"s_mov_b32 s[{s.s_p_wei()} + 2], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_wei()} + 3], 0x27000")
        self._emit(f"s_mov_b32 s[{s.s_p_out()} + 2], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_out()} + 3], 0x27000")
        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit_empty_line()
        self._emit(f"; calculate index")
        self._emit(f"s_mul_i32 s[{s.s_out_stride_k()}],      s[{s.s_ho()}],       s[{s.s_wo()}]")
        self._emit(f"s_mul_i32 s[{s.s_out_stride_n()}],      s[{s.s_k()}],        s[{s.s_out_stride_k()}]")
        self._emit(f"s_mul_i32 s[{s.s_in_stride_c()}],       s[{s.s_hi()}],       s[{s.s_wi()}]")
        self._emit(f"s_mul_i32 s[{s.s_in_stride_n()}],       s[{s.s_c()}],        s[{s.s_in_stride_c()}]")
        self._emit(f"s_mul_i32 s[{s.s_wei_stride_c()}],      s[{s.s_y()}],        s[{s.s_x()}]")
        self._emit(f"s_mul_i32 s[{s.s_wei_stride_k()}],      s[{s.s_c()}],        s[{s.s_wei_stride_c()}]")
        self._emit(f"s_mul_i32 s[{s.s_stride_dslice_hw()}],  s[{s.s_dslice_h()}], s[{s.s_dslice_w()}]")
        self._emit(f"s_mul_i32 s[{s.s_stride_dslice_yx()}],  s[{s.s_dslice_y()}], s[{s.s_dslice_x()}]")
        if t_k0 != 1:
            self._emit(f"s_lshl_b32 s[{s.s_out_stride_k0}], s[{s.s_out_stride_k}], {igemm_log2(t_k0)}")
            self._emit(f"s_lshl_b32 s[{s.s_wei_stride_k0}], s[{s.s_wei_stride_k}], {igemm_log2(t_k0)}")
        if t_b0 != 1:
            self._emit(f"s_mov_b32 s[{s.s_out_stride_b0}], {t_b0}")
        if t_n0 != 1:
            self._emit(f"s_lshl_b32 s[{s.s_out_stride_n0}], s[{s.s_out_stride_n()}], {igemm_log2(t_n0)}")
        if t_c0 != 1:
            self._emit(f"s_lshl_b32 s[{s.s_wei_stride_c0}], s[{s.s_wei_stride_c()}], {igemm_log2(t_c0)}")
        self._emit_empty_line()
        self._emit(f"; N0xN1xB0xB1, gemm_m_per_block:{self.tunable.gemm_m_per_block}, gemm_n_per_block:{self.tunable.gemm_n_per_block}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_stride_dslice_hw()}], s[{s.s_n()}]")
        self._emit(f"s_lshr_b32 s[0], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_n_per_block)}")
        self._emit(f".v_u32_div_ss {v.v_tmp()}+5, {s.s_bx()}, 0, {v.v_tmp()}, {s.s_tmp()}")
        self._emit(f"v_readfirstlane_b32 s[{s.s_tmp()}+4], v[{v.v_tmp()}+5]                  ; gemm_m,")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}+2], s[{s.s_tmp()}+4], s[0]")
        self._emit(f"s_sub_i32 s[{s.s_tmp()}+5], s[{s.s_bx()}], s[{s.s_tmp()}+2]                   ; gemm_n, cnt")
        self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ic()}], s[{s.s_tmp()}+4], {igemm_log2(self.tunable.gemm_n_per_block)}")

        if n_b != 1:
            self._emit(f"s_lshr_b32 s[0], s[{s.s_stride_dslice_hw()}], {igemm_log2(n_b)}          ; B:{n_b} per block, total num of B")
        else:
            self._emit(f"s_mov_b32 s[0], s[{s.s_stride_dslice_hw()}]                              ; B:{1} per block, total num of B")

        self._emit(f".v_u32_div_ss {v.v_tmp()}+5, {s.s_tmp()}+5, 0, {v.v_tmp()}, {s.s_tmp()}")
        self._emit(f"v_readfirstlane_b32 s[{s.s_tmp()}], v[{v.v_tmp()}+5]                             ; => N")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}+2], s[{s.s_tmp()}], s[0]")
        self._emit(f"s_sub_i32 s[{s.s_tmp()}+1], s[{s.s_tmp()}+5], s[{s.s_tmp()}+2]                ; => B")

        if n_n != 1:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_in()}], s[{s.s_tmp()}], {igemm_log2(n_n)}")
        else:
            self._emit(f"s_mov_b32 s[{s.s_block_gtc_in()}], s[{s.s_tmp()}]")
        if n_b != 1:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ib()}], s[{s.s_tmp()}+1], {igemm_log2(n_b)}")
        else:
            self._emit(f"s_mov_b32 s[{s.s_block_gtc_ib()}], s[{s.s_tmp()}+1]")

        self._emit_empty_line()
        emit_igemm_accumulate_x0_x1(v.v_gtc_ib0, v.v_gtc_ib1, c_b0, c_b1, n_b0, n_b1)
        emit_igemm_accumulate_x0_x1(v.v_gtc_in0, v.v_gtc_in1, c_n0, c_n1, n_n0, n_n1)
        emit_igemm_accumulate_x0_x1(v.v_gtc_ik0, v.v_gtc_ik1, c_k0, c_k1, n_k0, n_k1)
        self._emit_empty_line()
        # transform b
        self._emit(f"v_add_u32 v[{v.v_tmp()}+5], s[{s.s_block_gtc_ib()}], v[{v.v_gtc_ib1()}]")

        self._emit(f"; calculate output transform, B -> dslice_h*dslice_w. always use ib1")
        self._emit(f".v_u32_div_vs {v.v_out_dslice_ih()}, {v.v_tmp()}+5, {s.s_dslice_w()}, {v.v_tmp()}, {s.s_tmp()}")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_dslice_w()}], v[{v.v_out_dslice_ih()}]")
        self._emit(f"v_sub_u32 v[{v.v_out_dslice_iw()}], v[{v.v_tmp()}+5], v[{v.v_tmp()}]")

        self._emit(f"; iHTildaLeft, iWTildaLeft")
        self._emit(f"v_add_u32 v[{v.v_out_dslice_ih()}], s[{s.s_dslice_h_left()}], v[{v.v_out_dslice_ih()}]")
        self._emit(f"v_add_u32 v[{v.v_out_dslice_iw()}], s[{s.s_dslice_w_left()}], v[{v.v_out_dslice_iw()}]")

        self._emit(f"; dslice_y,dslice_h -> oh, dslice_x,dslice_w -> ow")
        self._emit(f"v_mov_b32 v[{v.v_out_iho()}], v[{v.v_out_dslice_ih()}]")
        self._emit(f"v_mov_b32 v[{v.v_out_iwo()}], v[{v.v_out_dslice_iw()}]")

        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_out_stride_k()}], v[{v.v_gtc_ik1()}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}+1], s[{s.s_out_stride_n()}], v[{v.v_gtc_in1()}]")
        self._emit(f"v_add_lshl_u32 v[{v.v_out_os_base()}], v[{v.v_tmp()}], v[{v.v_tmp()}+1], {igemm_log2(data_byte)}")

        self._emit(f"; n to staticly accumulate into base pointer")
        self._emit(f"s_lshl_b32 s[{s.s_tmp()}+3], s[{s.s_block_gtc_in()}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_out_stride_n()}], s[{s.s_tmp()}+3]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp()}+1], s[{s.s_out_stride_n()}], s[{s.s_tmp()}+3]")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out()}+1], s[{s.s_p_out()}+1], s[{s.s_tmp()}+1]")

        self._emit_empty_line()
        self._emit(m_out_update_os(v.v_out_os(), v.v_out_os_base(), v.v_out_iho(), v.v_out_iwo(), s.s_wo(), v.v_tmp()))
        self._emit(m_set_flag_hw(v.v_out_flag(), v.v_out_iho(), v.v_out_iwo(), s.s_ho(), s.s_wo()))

        self._emit_empty_line()
        if self.out_thread_copy_nd != 1:
            self._emit(f"s_lshl_b32 s[{s_out_stride_d0()}], s[{s_out_stride_d0()}], {igemm_log2(data_byte)}")
        self._emit(f"s_lshl_b32 s[{s_out_stride_d1()}], s[{s_out_stride_d1()}], {igemm_log2(data_byte)}")
        self._emit_empty_line()

        # load out
        self._emit(self.global_load_out())
        self._emit_empty_line()

        emit_igemm_accumulate_x0_x1(v.v_gtc_ic0, v.v_gtc_ic1, c_c0, c_c1, n_c0, n_c1)

        self._emit(f"v_add_u32 v[{v.v_tmp()}+5], s[{s.s_block_gtc_ic()}], v[{v.v_gtc_ic1()}]")
        self._emit(f"v_mov_b32 v[{v.v_dtile_iy()}], s[{s.s_dtile_iy()}]")
        self._emit(f"v_mov_b32 v[{v.v_dtile_ix()}], s[{s.s_dtile_ix()}]")

        self._emit(f"v_mov_b32 v[{v.v_wei_iy()}], s[{s.s_dtile_iy()}]")
        self._emit(f"v_mov_b32 v[{v.v_wei_ix()}], s[{s.s_dtile_ix()}]")
        self._emit_empty_line()
        self._emit(f"; calculate wei offset")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_c()}], v[{v.v_tmp()}+5]")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}+1], s[{s.s_wei_stride_k()}], v[{v.v_gtc_ik1()}]")
        self._emit(f"v_add_lshl_u32 v[{v.v_wei_os_base()}], v[{v.v_tmp()}], v[{v.v_tmp()}+1], {igemm_log2(data_byte)}")
        self._emit(m_wei_update_os(v.v_wei_os(), v.v_wei_os_base(), v.v_wei_iy(), v.v_wei_ix(), s.s_x(), v.v_tmp()))
        self._emit_empty_line()
        if self.wei_thread_copy_nd != 1:
            self._emit(f"s_lshl_b32 s[{s_wei_stride_d0()}], s[{s_wei_stride_d0()}], {igemm_log2(data_byte)}")
        self._emit(f"s_lshl_b32 s[{s_wei_stride_d1()}], s[{s_wei_stride_d1()}], {igemm_log2(data_byte)}")
        self._emit_empty_line()
        # load wei
        self._emit(self.global_load_wei())
        self._emit_empty_line()
        # c thread mapping
        self._emit(f"; c thread mapping ")
        self._emit(f"; ->            MR x  NR x ML1 x NL1 x ML0 x NL0")
        self._emit(f";  cluster       1 x   1 x  {self.tunable.gemm_m_level1_cluster}  x  {self.tunable.gemm_n_level1_cluster}  x  {self.tunable.gemm_m_level0_cluster}  x  {self.tunable.gemm_n_level0_cluster}")
        self._emit(f";  perthrd       {self.tunable.gemm_m_repeat} x   {self.tunable.gemm_n_repeat} x  1  x  1  x  {self.tunable.gemm_m_per_thread}  x  {self.tunable.gemm_n_per_thread}")
        #self._emit(f";  order(thrd):")
        #self._emit(f";              ML1 x NL1 x  MR x ML0 x  NR x NL0")
        assert self.tunable.gemm_m_per_thread != 1 and self.tunable.gemm_n_per_thread != 1

        self._emit(f"v_and_b32 v[{v.v_tmp()}], {self.tunable.gemm_n_level0_cluster - 1}, v0")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(self.tunable.gemm_n_per_thread)}, v[{v.v_tmp()}]         ; => iNL0")
        self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}+5], {igemm_log2(self.tunable.gemm_n_level0_cluster)}, v0")

        self._emit(f"v_and_b32 v[{v.v_tmp()}+1], {self.tunable.gemm_m_level0_cluster - 1}, v[{v.v_tmp()}+5]")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}+1], {igemm_log2(self.tunable.gemm_m_per_thread)}, v[{v.v_tmp()}+1]     ; => iML0")
        self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}+5], {igemm_log2(self.tunable.gemm_m_level0_cluster)}, v[{v.v_tmp()}+5]")

        self._emit(f"v_and_b32 v[{v.v_tmp()}+2],   {self.tunable.gemm_n_level1_cluster - 1}, v[{v.v_tmp()}+5]       ; => iNL1")
        self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}+5], {igemm_log2(self.tunable.gemm_n_level1_cluster)}, v[{v.v_tmp()}+5]")
        self._emit(f"v_and_b32 v[{v.v_tmp()}+3],   {self.tunable.gemm_m_level1_cluster - 1}, v[{v.v_tmp()}+5]       ; => iML1")
        # self._emit(f"; v_lshrrev_b32 v[{v.v_tmp()}+5], {igemm_log2(self.tunable.gemm_m_level1_cluster)}, v[{v.v_tmp()}+5]")

        self._emit(f"v_lshl_or_b32 v[{v.v_gemm_in()}], v[{v.v_tmp()}+2], {igemm_log2(self.tunable.gemm_n_per_thread * self.tunable.gemm_n_level0_cluster)}, v[{v.v_tmp()}]               ; in  (without repeat)")
        self._emit(f"v_lshl_or_b32 v[{v.v_gemm_im()}], v[{v.v_tmp()}+3], {igemm_log2(self.tunable.gemm_m_per_thread * self.tunable.gemm_m_level0_cluster)}, v[{v.v_tmp()}+1]             ; im  (without repeat)")
        self._emit_empty_line()
        # note here, for gemm_n, we always want this order n0xb0xb1xn1
        self._emit(f"; remapping, gemm_im => C0xC1:{n_c0}x{n_c1}, gemm_in => N0xB0xB1xN1:{n_n0}x{n_b0}x{n_b1}x{n_n1}")

        assert n_n1 == self.tunable.gemm_n_per_thread or (n_n1 == 1 and n_b1 == self.tunable.gemm_n_per_thread)
        assert n_c1 == self.tunable.gemm_m_per_thread

        if n_n1 != 1:
            self._emit(f"v_and_b32 v[{v.v_in_in1()}], {n_n1 - 1}, v[{v.v_gemm_in()}]")
            self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], {igemm_log2(n_n1)}, v[{v.v_gemm_in()}]")
        else:
            self._emit(f"v_mov_b32 v[{v.v_in_in1()}], 0")
            #self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gemm_in()}]")
        
        if n_b1 != 1:
            self._emit(f"v_and_b32 v[{v.v_in_ib1()}], {n_b1 - 1}, v[{v.v_tmp()}]")
            self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], {igemm_log2(n_b1)}, v[{v.v_tmp()}]")
        else:
            self._emit(f"v_mov_b32 v[{v.v_in_ib1()}], 0")
            #self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_tmp()}]")

        if n_b0 != 1:
            self._emit(f"v_and_b32 v[{v.v_in_ib0()}], {n_b0 - 1}, v[{v.v_tmp()}]")
            self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], {igemm_log2(n_b0)}, v[{v.v_tmp()}]")
        else:
            self._emit(f"v_mov_b32 v[{v.v_in_ib0()}], 0")
            #self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_tmp()}]")

        if n_n0 != 1:
            self._emit(f"v_and_b32 v[{v.v_in_in0()}], {n_n0 - 1}, v[{v.v_tmp()}]")
            self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], {igemm_log2(n_n0)}, v[{v.v_tmp()}]")
        else:
            self._emit(f"v_mov_b32 v[{v.v_in_in0()}], 0")
            #self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_tmp()}]")

        if n_c1 != 1:
            self._emit(f"v_and_b32 v[{v.v_in_ic1()}], {n_c1 - 1}, v[{v.v_gemm_im()}]")
            self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], {igemm_log2(n_c1)}, v[{v.v_gemm_im()}]")
        else:
            self._emit(f"v_mov_b32 v[{v.v_in_ic1()}], 0")
            #self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gemm_im()}]")

        if n_c0 != 1:
            self._emit(f"v_and_b32 v[{v.v_in_ic0()}], {n_c0 - 1}, v[{v.v_tmp()}]")
            self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], {igemm_log2(n_c0)}, v[{v.v_tmp()}]")
        else:
            self._emit(f"v_mov_b32 v[{v.v_in_ic0()}], 0")
            #self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_tmp()}]")

        self._emit_empty_line()
        self._emit(f"v_lshl_or_b32 v[{v.v_in_ib1()}], v[{v.v_in_ib0()}], {igemm_log2(n_b1)}, v[{v.v_in_ib1()}]")
        self._emit(f"v_lshl_or_b32 v[{v.v_in_in1()}], v[{v.v_in_in0()}], {igemm_log2(n_n1)}, v[{v.v_in_in1()}]")
        self._emit(f"v_lshl_or_b32 v[{v.v_in_ic1()}], v[{v.v_in_ic0()}], {igemm_log2(n_c1)}, v[{v.v_in_ic1()}]")
        self._emit(f"v_add_u32 v[{v.v_in_ib1()}], s[{s.s_block_gtc_ib()}], v[{v.v_in_ib1()}]")
        self._emit_empty_line()
        self._emit(f".v_u32_div_vs {v.v_in_dslice_ih()}, {v.v_in_ib1()}, {s.s_dslice_w()}, {v.v_tmp()}, {s.s_tmp()}")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}+1], s[{s.s_dslice_w()}], v[{v.v_in_dslice_ih()}]")
        self._emit(f"v_sub_u32 v[{v.v_in_dslice_iw()}], v[{v.v_in_ib1()}], v[{v.v_tmp()}+1]")

        self._emit(f"v_add_u32 v[{v.v_in_dslice_ih()}], s[{s.s_dslice_h_left()}], v[{v.v_in_dslice_ih()}]")
        self._emit(f"v_add_u32 v[{v.v_in_dslice_iw()}], s[{s.s_dslice_w_left()}], v[{v.v_in_dslice_iw()}]")
        self._emit_empty_line()
        self._emit(f"; dslice_h,dslice_y -> hip,  dslice_w,dslicw_x -> wip")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_dtile_iy()}], s[{s.s_dilation_h()}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_stride_h()}], v[{v.v_in_dslice_ih()}]")
        self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_tmp()}], v[{v.v_tmp()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}+1], s[{s.s_dtile_ix()}], s[{s.s_dilation_w()}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}+1], s[{s.s_stride_w()}], v[{v.v_in_dslice_iw()}]")
        self._emit(f"v_add_u32 v[{v.v_tmp()}+1], s[{s.s_tmp()}+1], v[{v.v_tmp()}+1]")
        self._emit(f"; v_tmp: hip, v_tmp+1: wip")
        self._emit_empty_line()
        self._emit(f"; hip->h, wip->w")
        self._emit(f"v_sub_i32 v[{v.v_in_ihi()}], v[{v.v_tmp()}], s[{s.s_pad_h()}]")
        self._emit(f"v_sub_i32 v[{v.v_in_iwi()}], v[{v.v_tmp()}+1], s[{s.s_pad_w()}]")
        self._emit_empty_line()
        self._emit(m_set_flag_hw(v.v_in_flag(), v.v_in_ihi(), v.v_in_iwi(), s.s_hi(), s.s_wi()))
        self._emit_empty_line()

        self._emit(f"; input offset")
        self._emit(f"s_lshl_b32 s[{s.s_tmp()}+3], s[{s.s_block_gtc_in()}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_in_stride_n()}], s[{s.s_tmp()}+3]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp()}+1], s[{s.s_in_stride_n()}], s[{s.s_tmp()}+3]")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_in()}+1], s[{s.s_p_in()}+1], s[{s.s_tmp()}+1]")
        self._emit_empty_line()
        self._emit(f"s_lshl_b32 s[{s.s_tmp()}+3], s[{s.s_block_gtc_ic()}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_in_stride_c()}], s[{s.s_tmp()}+3]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp()}+1], s[{s.s_in_stride_c()}], s[{s.s_tmp()}+3]")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_in()}+1], s[{s.s_p_in()}+1], s[{s.s_tmp()}+1]")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_in_stride_n()}], v[{v.v_in_in1()}]")

        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}+1], s[{s.s_in_stride_c()}], v[{v.v_in_ic1()}]")
        self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_tmp()}], v[{v.v_tmp()}+1]")

        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}+1], s[{s.s_wi()}], v[{v.v_in_ihi()}]")
        self._emit(f"v_add3_u32 v[{v.v_in_os()}], v[{v.v_tmp()}], v[{v.v_tmp()}+1], v[{v.v_in_iwi()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_in_os()}], {igemm_log2(data_byte)}, v[{v.v_in_os()}]")
        self._emit_empty_line()
        if n_n1 != 1:
            self._emit(f"s_lshl_b32 s[{s.s_in_stride_n()}], s[{s.s_in_stride_n()}], {igemm_log2(data_byte)}")
        else:
            #self._emit(f"s_lshl_b32 s[{s.s_in_stride_n()}], s[{s.s_in_stride_n()}], {igemm_log2(data_byte)}")
            self._emit(f"s_mov_b32 s[{s.s_in_stride_n()}], {data_byte}            ; B stride. indeed this may not be used")
        
        if n_n0 != 1:
            self._emit(f"s_lshl_b32 s[{s.s_in_stride_nr()}], s[{s.s_in_stride_n()}], {igemm_log2( (n_n0 * n_n1) // 2)}")
        else:
            assert n_b0 != 1
            self._emit(f"s_mov_b32 s[{s.s_in_stride_nr()}], {(n_b0 * n_b1 * data_byte) // 2}")
        self._emit_empty_line()
        self._emit(f"s_lshl_b32 s[{s.s_in_stride_c()}], s[{s.s_in_stride_c()}], {igemm_log2(data_byte)}")
        self._emit(f"s_lshl_b32 s[{s.s_in_stride_cr()}], s[{s.s_in_stride_c()}], {igemm_log2( (n_c0*n_c1) // 2)}")
        self._emit_empty_line()
        self._emit(f"v_lshrrev_b32 v[{v.v_gtc_in0()}], {igemm_log2(n_n1)}, v[{v.v_gtc_in1()}]")
        self._emit(f"; LDS store, order out: K0xK1xExN0xB0xB1xN1: {t_k0}x{t_k1}x{t_e}x{t_n0}x{t_b0}x{t_b1}x{t_n1}, {c_k0}x{c_k1}x{c_e}x{c_n0}x{c_b0}x{c_b1}x{c_n1}")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(n_n1)}, v[{v.v_gtc_ib1()}]")
        self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_in0()}], {igemm_log2(n_b0*n_b1*n_n1)}, v[{v.v_tmp()}]")
        self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}+1], v[{v.v_gtc_ik1()}], {igemm_log2(n_n0*n_b0*n_b1*n_n1)}, v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_sst_b_os()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}+1]")
        self._emit_empty_line()
        self._emit(f"; LDS store, order wei not shuffled, wei: K0xK1xExC0xC1: {t_k0}x{t_k1}x{t_e}x{t_c0}x{t_c1}, {c_k0}x{c_k1}x{c_e}x{c_c0}x{c_c1}")
        self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ik1()}], {igemm_log2(n_c0*n_c1)}, v[{v.v_gtc_ic1()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_sst_a_os()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")

        self._emit(f"v_add_u32 v[{v.v_sst_b_os()}], {self.tunable.lds_a_np2}, v[{v.v_sst_b_os()}]")
        self._emit_empty_line()
        self._emit(f"; LDS load")
        self._emit(f"v_lshlrev_b32 v[{v.v_sld_b_os()}], {igemm_log2(data_byte)}, v[{v.v_gemm_in()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_sld_a_os()}], {igemm_log2(data_byte)}, v[{v.v_gemm_im()}]")

        self._emit(f"v_add_u32 v[{v.v_sld_b_os()}], {self.tunable.lds_a_np2}, v[{v.v_sld_b_os()}]")
        self._emit_empty_line()
        self._emit(f"s_mul_i32 s[{s.s_knum()}], s[{s.s_stride_dslice_yx()}], s[{s.s_k()}]")

        if s_out_stride_d0 == s.s_out_stride_k or s_out_stride_d1 == s.s_out_stride_k:
            # stride K is already mul by data_byte above. TODO: better logic
            self._emit(f"s_lshl_b32 s[{s.s_out_move_slice_stride_k()}], s[{s.s_out_stride_k()}], {igemm_log2(self.tunable.gemm_k_per_block)} ; {self.tunable.gemm_k_per_block}x")
        else:
            self._emit(f"s_lshl_b32 s[{s.s_out_move_slice_stride_k()}], s[{s.s_out_stride_k()}], {igemm_log2(self.tunable.gemm_k_per_block * data_byte)} ; {self.tunable.gemm_k_per_block}x")

        if s_wei_stride_d0 == s.s_wei_stride_k or s_wei_stride_d1 == s.s_wei_stride_k:
            # stride K is already mul by data_byte above. TODO: better logic
            self._emit(f"s_lshl_b32 s[{s.s_wei_move_slice_stride_k()}], s[{s.s_wei_stride_k()}], {igemm_log2(self.tunable.gemm_k_per_block)} ; {self.tunable.gemm_k_per_block}x")
        else:
            self._emit(f"s_lshl_b32 s[{s.s_wei_move_slice_stride_k()}], s[{s.s_wei_stride_k()}], {igemm_log2(self.tunable.gemm_k_per_block * data_byte)} ; {self.tunable.gemm_k_per_block}x")

        self._emit(f"s_mov_b32 s[{s.s_move_slice_k_ik()}], 0")
        self._emit(f"s_mov_b32 s[{s.s_move_slice_k_idsy()}], 0")
        self._emit(f"s_mov_b32 s[{s.s_move_slice_k_idsx()}], 0")
        self._emit_empty_line()

    def emit_kernel_fma_main_loop(self):
        s = self.sgpr
        v = self.vgpr

        def move_slice_window_b():
            m_move_slice_window   = self.get_macro_move_slice_window()
            m_out_update_os       = self.get_macro_out_update_os()
            m_out_update_hw       = self.get_macro_out_update_hw()
            m_set_flag_hw         = self.get_macro_set_flag_hw()
            k_step_k, k_step_dsy, k_step_dsx  =  self.tunable.gemm_k_per_block, 1, 1
            with self._deferred_context():
                self._emit(m_move_slice_window(s.s_move_slice_k_ik(), s.s_move_slice_k_idsy(), s.s_move_slice_k_idsx(), s.s_dslice_y(), s.s_dslice_x(),
                        k_step_k, k_step_dsy, k_step_dsx, v.v_out_os_base(), v.v_wei_os_base(), s.s_out_move_slice_stride_k(), s.s_wei_move_slice_stride_k()))
                self._emit(m_out_update_hw(v.v_out_iho(), v.v_out_iwo(), v.v_out_dslice_ih(), v.v_out_dslice_iw(), s.s_move_slice_k_idsy(), s.s_move_slice_k_idsx(), s.s_dtile_dy(), s.s_dtile_dx(), s.s_tmp()))
                self._emit(m_out_update_os(v.v_out_os(), v.v_out_os_base(), v.v_out_iho(), v.v_out_iwo(), s.s_wo(), v.v_tmp()))
                self._emit(m_set_flag_hw(v.v_out_flag(), v.v_out_iho(), v.v_out_iwo(), s.s_ho(), s.s_wo()))
            return self._get_deferred()

        def move_slice_window_a():
            m_wei_update_os   = self.get_macro_wei_update_os()
            m_wei_update_yx   = self.get_macro_wei_update_yx()
            with self._deferred_context():
                self._emit(m_wei_update_yx(v.v_wei_iy(), v.v_wei_ix(), s.s_move_slice_k_idsy(), s.s_move_slice_k_idsx(), s.s_dtile_y(), s.s_dtile_x(), v.v_dtile_iy(), v.v_dtile_ix(), s.s_tmp()))
                self._emit(m_wei_update_os(v.v_wei_os(), v.v_wei_os_base(), v.v_wei_iy(), v.v_wei_ix(), s.s_x(), v.v_tmp()))
            return self._get_deferred()

        fctrl                             = ctrl_fma_main_loop_t()
        fctrl.thread_m                    = self.tunable.thread_tile_m
        fctrl.thread_n                    = self.tunable.thread_tile_n
        fctrl.unroll_k                    = self.tunable.gemm_k_per_block
        fctrl.label_prefix                = self.name()
        #fctrl.data_type                   = AMDGPU_PRECISION_FP32
        fctrl.gemm_m_repeat               = self.tunable.gemm_m_repeat
        fctrl.gemm_m_level0_cluster       = self.tunable.gemm_m_level0_cluster
        fctrl.gemm_m_level1_cluster       = self.tunable.gemm_m_level1_cluster
        fctrl.gemm_n_repeat               = self.tunable.gemm_n_repeat
        fctrl.gemm_n_level0_cluster       = self.tunable.gemm_n_level0_cluster
        fctrl.gemm_n_level1_cluster       = self.tunable.gemm_n_level1_cluster
        fctrl.lds_single_size             = self.tunable.lds_single            # in byte, should be power of 2
        # fctrl.lds_buffer_num              = 2

        # functor
        fctrl.global_load_a_functor       = self.global_load_wei
        fctrl.global_load_b_functor       = self.global_load_out
        fctrl.shared_store_a_functor      = self.shared_store_wei
        fctrl.shared_store_b_functor      = self.shared_store_out
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

    def emit_kernel_epilogue(self):
        s = self.sgpr
        v = self.vgpr

        in_global_store = self.get_macro_global_store()

        label_out = f"L_{self.name()}_out"
        self._emit(f"v_cmpx_eq_u32 vcc, 1, v[{v.v_in_flag()}]")
        self._emit(f"s_cbranch_execz {label_out}")

        self._emit(f"s_mov_b32 s[{s.s_tmp()}], 0")
        self._emit(f"s_mov_b32 s[{s.s_tmp()}+1], 0")
        self._emit(f"s_mov_b32 s[{s.s_tmp()}+2], 0")
        self._emit(f"s_mov_b32 s[{s.s_tmp()}+3], 0")

        # self._emit(f"_iii = 0")
        # self._emit(f".rept 64")
        # self._emit(f"v_mov_b32 v[_iii], 3.0")
        # self._emit(f"_iii = _iii + 1")
        # self._emit(f".endr")

        self._emit(in_global_store(v.v_c(), s.s_p_in(), v.v_in_os(), s.s_in_stride_n(), s.s_in_stride_nr(), s.s_in_stride_c(), s.s_in_stride_cr(), s.s_tmp(),
            self.tunable.gemm_n_per_thread, self.tunable.gemm_n_repeat, self.tunable.gemm_m_per_thread, self.tunable.gemm_m_repeat))

        self._emit_front(f"{label_out}:")

    def emit_kernel_symbol(self):
        self.karg.emit()
        self._emit_empty_line()
        self.sgpr.emit()
        self._emit_empty_line()
        self.vgpr.emit()
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


    # def emit_kernel(self):
    #     self.emit_kernel_prologue()
    #     self.emit_kernel_fma_main_loop()
    #     self.emit_kernel_epilogue()