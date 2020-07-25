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

import math
from .igemm_base import *
from .shared_memory import *
from .global_memory import *

class ctrl_coalescing_store_t(object):
    def __init__(self):
        self.coalescing_groups = 1
        self.gemm_m_repeat = 1
        self.gemm_m_per_thread = 1
        self.gemm_n_repeat = 1
        self.gemm_n_per_thread = 1

        self.gemm_m_length = 1  # in unit of dword
        self.gemm_n_length = 1  # in unit of dword
        self.data_byte = 4

        self.vector_write_out = 1
        self.block_size = 256

class igemm_coalescing_store_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(mc)
        assert type(ctrl) is ctrl_coalescing_store_t
        assert ctrl.gemm_m_repeat == 2 and ctrl.gemm_n_repeat == 2
        assert ctrl.block_size % ctrl.gemm_n_length
        self.ctrl = ctrl

        self.num_dword_per_group = (ctrl.gemm_m_repeat * ctrl.gemm_m_per_thread * ctrl.gemm_n_repeat * ctrl.gemm_n_per_thread) // ctrl.coalescing_groups
        self.coalescing_groups_in_m_repeat = igemm_gcd(ctrl.gemm_m_repeat, ctrl.coalescing_groups)
        self.group_length_in_m_repeat = ctrl.gemm_m_repeat // self.coalescing_groups_in_m_repeat
        self.coalescing_groups_in_m_per_thread = igemm_gcd(ctrl.gemm_m_per_thread, ctrl.coalescing_groups // self.coalescing_groups_in_m_repeat)
        self.group_length_in_m_per_thread = ctrl.gemm_m_per_thread // self.coalescing_groups_in_m_per_thread
        self.gemm_m_per_linear_block = ctrl.block_size * ctrl.vector_write_out // ctrl.gemm_m_length

        '''
            m_per_thread  glen_in_mpt gs_in_mpt m_per_lin_block  |  pps_in_mpt  pps_in_mr  stride_start_per_group
            4             1           4         1                   x           4          1
            4             2           2         1                   1           4          2
            4             4           1         1                   1           4          x
        ---------------------------------------------------------------------------------------------------
            4             1           4         2                   x           8          1
            4             2           2         2                   x           4          2
            4             4           1         2                   2           4          x
        ---------------------------------------------------------------------------------------------------
            4             1           4         4                   x           16         1
            4             2           2         4                   x           8          2
            4             4           1         4                   x           4          x
        '''



    def name(self):
        return ''

    def init_co_lds_offset(self, v_co_sst, v_co_sld, v_gemm_im, v_gemm_in, v_tid, v_tmp2):
        ctrl = self.ctrl
        with self._deferred_context():
            gemm_m_shrink = ctrl.gemm_m_per_thread // self.coalescing_groups_in_m_per_thread
            if gemm_m_shrink != 1:
                self._emit(f"v_lshrrev_b32 v[{v_tmp2}], {igemm_log2(gemm_m_shrink)}, v[{v_gemm_im}]")
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp2}], {igemm_log2(ctrl.gemm_m_length)}, v[{v_gemm_in}]")
            else:
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_gemm_im}], {igemm_log2(ctrl.gemm_m_length)}, v[{v_gemm_in}]")
            self._emit(f"v_lshlrev_b32 v[{v_co_sst}], {igemm_log2(ctrl.data_byte)}, v[{v_co_sst}]")
            self._emit(f"v_lshlrev_b32 v[{v_co_sld}], {igemm_log2(ctrl.data_byte * ctrl.vector_write_out)}, v[{v_tid}]")

        return self._get_deferred()

    def __call__(self, v_c, v_co_sst, v_co_sld, s_p_out, v_out_offset, s_out_offset, s_gemm_m_stride, s_tmp):
        ctrl = self.ctrl
        v_c = sym_t(v_c)
        v_co_sst = sym_t(v_co_sst)
        v_co_sld = sym_t(v_co_sld)
        s_tmp = sym_t(s_tmp)

        self._emit(f"; coalescing store, gemm_mxn:{ctrl.gemm_m_length}x{ctrl.gemm_n_length}, block:{ctrl.block_size}, m_repeatxm_perthread:{ctrl.gemm_m_repeat}x{ctrl.gemm_m_per_thread}, n_repeatxn_perthread:{ctrl.gemm_n_repeat}x{ctrl.gemm_n_per_thread}")
        self._emit(f"; coalescing_groups:{ctrl.coalescing_groups}, num_dword_per_group:{self.num_dword_per_group}")
        self._emit(f"; coalescing_groups_in_m_repeat:{self.coalescing_groups_in_m_repeat}, group_length_in_m_repeat:{self.group_length_in_m_repeat}, coalescing_groups_in_m_per_thread:{self.coalescing_groups_in_m_per_thread}, group_length_in_m_per_thread:{self.group_length_in_m_per_thread}")
        self._emit(f"; gemm_m_per_linear_block:{self.gemm_m_per_linear_block}, gemm_m_per_pixel:2, gemm_m_per_group_in_m_per_thread:4")

        # mc, vec_count, vec_byte, vec_stride, sst_base=0):
        inst_sst = inst_ds_write2_likely_t(self.mc, 2, ctrl.gemm_n_per_thread * ctrl.data_byte, ctrl.gemm_n_length // 2)
        # mc, vec_count, vec_byte, vec_stride, sld_base = 0):
        inst_sld = inst_ds_read2_likely_t(self.mc, 2, ctrl.vector_write_out, ctrl.block_size * ctrl.vector_write_out * ctrl.data_byte)
        # self, vdata, vaddr, srsrc, soffset, offset):
        inst_gst = inst_buffer_store_dword_t(ctrl.vector_write_out)

        s_out_offset_itr = sym_t(s_tmp(0))

        for i_group_in_m_repeat in range(self.coalescing_groups_in_m_repeat):
            for i_group_in_m_per_thread in range(self.group_length_in_m_per_thread):
                c_group_start_index = (i_group_in_m_repeat *ctrl.gemm_m_per_thread + i_group_in_m_per_thread) * ctrl.gemm_n_per_thread * ctrl.gemm_n_repeat

                self._emit(f"s_barrier")
                for imr in range(self.group_length_in_m_repeat):
                    for imp in range(self.group_length_in_m_per_thread):
                        c_start_current = c_group_start_index + (imr * ctrl.gemm_m_per_thread + imp) * ctrl.gemm_n_per_thread * ctrl.gemm_n_repeat
                        sst_offset = (imr * coalescing_groups_in_m_per_thread + imp) * ctrl.gemm_n_length * ctrl.data_byte
                        # def __call__(self, v_sst, v_src, sst_offset = 0):
                        self._emit(inst_sst(v_co_sst(), v_c(c_start_current), sst_offset))

                self._emit(f"s_waitcnt lgkmcnt(0)")
                self._emit(f"s_barrier")

                issue_list = []
                for i_d in range(self.num_dword_per_group // (2 * ctrl.vector_write_out)):
                    #v_dst, v_sld_os, sld_offset = 0):
                    c_start_current = c_group_start_index + i_d * 2 * ctrl.vector_write_out
                    sld_offset = i_d * 2 * ctrl.block_size * ctrl.vector_write_out * ctrl.data_byte
                    self._emit(inst_sld(v_c(c_start_current), v_co_sld(), sld_offset))
                    issue_list.append(inst_sld.get_issues(sld_offset))
                
                for i_gst in range(self.num_dword_per_group // (ctrl.vector_write_out)):
                    if i_gst % 2 == 0:
                        i_issues =  (i_gst // 2) + 1
                        i_issue_list = issue_list[i_issues:]
                        i_issue_cnt = igemm_flatten_list_accumulate(i_issue_list) if len(i_issue_list) != 0 else 0
                        self._emit(f"s_waitcnt lgkmcnt({i_issue_cnt})")
                    # vdata, vaddr, srsrc, soffset, offset
                    inst_gst(v_c(c_group_start_index + i_gst*ctrl.vector_write_out), s_p_out, s_out_offset_itr(), 0)
