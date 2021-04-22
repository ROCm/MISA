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
from .fma_main_loop import *
from .igemm_base import *
from .global_memory import *
from .shared_memory import *
from .utility import *
from .thread_mapping import *
from .xdlops_mapping import *
from .coalescing_store import *
from .mfma_main_loop import *

IGEMM_WRW_GTC_DEBUG = 1
IGEMM_WRW_GTC_N_SPLIT_FIRST = 1


def _find_non_1_index_in_list(list_object):
    result_list = list()
    for idx, item in enumerate(list_object):
        assert type(item) is int
        if item != 1:
            result_list.append(idx)
    return result_list

class igemm_wrw_gtc_nhwc_t(mc_base_t):
    '''
    k -> k
    c -> c0, c1
    n -> n
    ho, wo -> b
    x, y -> e

    gemm_m -> k
    gemm_k -> n*b
    gemm_n -> ec0*c1

    tensor a: 1*nb*1*k
    tensor b: 1*nb*ec0*c1

              thread_lengths            cluster_lengths
    tensor a: 1*ta_nb*1*ta_k            1*ca_nb*1*ca_k
    tensor b: 1*tb_nb*tb_ec0*tb_c1      1*ca_nb*cb_ec0*cb_c1

                      tensor a                      tensor b
    thread_lengths  : 1, ta_nb, 1, ta_k             1, tb_nb, tb_ec0, tb_c1
    cluster_lengths : 1, ca_nb, 1, ca_k             1, ca_nb, cb_ec0, cb_c1

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
            ctrl_coalescing_store.data_byte = amdgpu_precision_data_byte(self.tunable.precision)

            ctrl_coalescing_store.vector_write_out = 1                      # TODO: some cases this can be set to other value
            ctrl_coalescing_store.block_size = self.tunable.block_size

            self.coalescing_store = igemm_coalescing_store_t(mc, ctrl_coalescing_store)

        else:
            def flatten(x):
                from functools import reduce
                return reduce(lambda a, b: a*b, x, 1)
            ctrl_xdlops_mapping = get_ctrl_xdlops_mapping_from_wave_tile_fp32(self.tunable.gemm_m_per_block, self.tunable.gemm_n_per_block, self.tunable.wave_tile_m, self.tunable.wave_tile_n, self.tunable.wave_tile_k, 
                    self.tunable.wave_repeat_m, self.tunable.wave_repeat_n, self.tunable.wave_step_m, self.tunable.wave_step_n, self.tunable.block_size // AMDGPU_WAVE_SIZE)
            self.xdlops_mapping = igemm_xdlops_mapping_t(self.mc, ctrl_xdlops_mapping)
            assert flatten(ctrl_xdlops_mapping.acc_c_per_thread_m()) % self.coalescing_store_groups == 0, \
                f"coalescing store groups should be divided by agpr per thread in m direction {ctrl_xdlops_mapping.acc_c_per_thread_m()}"

            ctrl_coalescing_store_xdlops = ctrl_coalescing_store_xdlops_t()
            ctrl_coalescing_store_xdlops.cxm = ctrl_xdlops_mapping
            ctrl_coalescing_store_xdlops.gemm_k_global_split = self.tunable.gemm_k_global_split
            ctrl_coalescing_store_xdlops.coalescing_groups = self.coalescing_store_groups
            ctrl_coalescing_store_xdlops.data_byte = amdgpu_precision_data_byte(self.tunable.precision)

            ctrl_coalescing_store_xdlops.vector_write_out = 1                      # TODO: some cases this can be set to other value
            ctrl_coalescing_store_xdlops.block_size = self.tunable.block_size

            na_k, na_nb, nb_nb, nb_ec0, nb_c1  = self.get_dims_lengths()
            ctrl_coalescing_store_xdlops.gemm_m_m0_m1 = [1, na_k]
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
        return igemm_gtc_encode_kernel_name(self.tunable)

    def try_shift_stride(self, gpr, shifter):
        assert type(gpr) is sym_t
        with self._deferred_context():
            if gpr.label not in self.dict_shifted_stride:
                self.dict_shifted_stride[gpr.label] = gpr
                self._emit(f"s_lshl_b32 s[{gpr()}], s[{gpr()}], {shifter}")
        return self._get_deferred()

    class macro_igemm_wrw_gtc_in_out_update_os_t(macro_base_t):
        def __init__(self, mc, data_byte, inline = False):
            macro_base_t.__init__(self, mc, inline)
            self.data_byte = data_byte
            self.declare_arg("v_os")
            self.declare_arg("v_os_base")
            self.declare_arg("v_ih")
            self.declare_arg("v_iw")
            self.declare_arg("s_w")
            self.declare_arg("s_stride_w")
            self.declare_arg("v_tmp")
        def name(self):
            return '.v_wrw_gtc_in_update_os'
        def expr(self):
            self._emit(f"; from hi, wi, os_base, compute final offset")
            self._emit(f"v_mad_u32_u24 v[{self.v_tmp()}], s[{self.s_w()}], v[{self.v_ih()}], v[{self.v_iw()}]")
            self._emit(f"v_mul_lo_u32 v[{self.v_tmp()}], v[{self.v_tmp()}], s[{self.s_stride_w()}]")
            self._emit(f"v_lshl_add_u32 v[{self.v_os()}], v[{self.v_tmp()}], {igemm_log2(self.data_byte)}, v[{self.v_os_base()}]")

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
            #self._emit(f"v_mul_lo_u32 v[{self.v_tmp()}], s[{self.s_stride_h()}], v[{self.v_out_iho()}]")
            #self._emit(f"v_add_i32 v[{self.v_in_ihi()}], v[{self.v_tmp()}], v[{self.v_wei_iy()}]")
            #self._emit(f"v_mul_lo_u32 v[{self.v_tmp(1)}], s[{self.s_stride_w()}], v[{self.v_out_iwo()}]")   
            #self._emit(f"v_add_i32 v[{self.v_in_iwi()}], v[{self.v_tmp(1)}], v[{self.v_wei_ix()}]")
            self._emit(f"v_mad_u32_u24 v[{self.v_in_ihi()}], s[{self.s_stride_h()}], v[{self.v_out_iho()}], v[{self.v_wei_iy()}]")
            self._emit(f"v_mad_u32_u24 v[{self.v_in_iwi()}], s[{self.s_stride_w()}], v[{self.v_out_iwo()}], v[{self.v_wei_ix()}]")

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
            self.declare_arg("v_move_slice_n_idsho")
            self.declare_arg("v_move_slice_n_idswo")
            self.declare_arg("v_gemm_k_num_dsho")
            self.declare_arg("v_gemm_k_num_dswo")
            self.declare_arg("s_move_slice_n_dsho")
            self.declare_arg("s_move_slice_n_dswo")
            self.declare_arg("v_in_os_base")
            self.declare_arg("s_in_stride_n")
            self.declare_arg("s_in_stride_n_n")
            self.declare_arg("s_stride_h")
            self.declare_arg("s_ho_line")
            self.declare_arg("s_wo_line")

        def name(self):
            return '.s_wrw_gtc_move_slice_window_n_dsho_dswo'

        def expr(self):
            # n0, n1b is unmerge.  n1b is merged from n1, b
            self._emit(f"v_add_u32 v[{self.v_move_slice_n_idswo()}], s[{self.s_move_slice_n_dswo()}], v[{self.v_move_slice_n_idswo()}]")
            self._emit(f"v_cmpx_le_i32 vcc, v[{self.v_gemm_k_num_dswo()}], v[{self.v_move_slice_n_idswo()}]")
            self._emit(f"v_subrev_u32 v[{self.v_move_slice_n_idswo()}], s[{self.s_wo_line()}], v[{self.v_move_slice_n_idswo()}]")
            self._emit(f"v_add_u32 v[{self.v_move_slice_n_idsho()}], s[{self.s_stride_h()}], v[{self.v_move_slice_n_idsho()}]")
            self._emit(f"s_mov_b64 exec, -1")
            self._emit_empty_line()
            self._emit(f"v_add_u32 v[{self.v_move_slice_n_idsho()}], s[{self.s_move_slice_n_dsho()}], v[{self.v_move_slice_n_idsho()}]")
            self._emit(f"v_cmpx_le_i32 vcc, v[{self.v_gemm_k_num_dsho()}], v[{self.v_move_slice_n_idsho()}]")
            self._emit(f"v_subrev_u32 v[{self.v_move_slice_n_idsho()}], s[{self.s_ho_line()}], v[{self.v_move_slice_n_idsho()}]")
            self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_n()}], v[{self.v_in_os_base()}]")
            self._emit(f"s_mov_b64 exec, -1")
            self._emit_empty_line()
            self._emit(f"v_add_u32 v[{self.v_in_os_base()}], s[{self.s_in_stride_n_n()}], v[{self.v_in_os_base()}]")

    class global_load_in_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_out_2d_global_load, m_in_2d_global_load = self.outer.get_macro_global_load()
            return m_in_2d_global_load.get_issues()

        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr

            m_out_2d_global_load, m_in_2d_global_load = self.outer.get_macro_global_load()
            s_out_stride_d0, s_out_stride_d1, s_in_stride_d0, s_in_stride_d1 = self.outer.get_symbol_global_load_s_stride_d0_d1()
            with self._deferred_context():
                self._emit(f"; load input")
                if self.outer.tunable.nxe != 0:
                    self._emit(f".v_clear_nc {v.v_gld_b()}, {m_in_2d_global_load.ctrl.length_d0 * m_in_2d_global_load.ctrl.length_d1}")
                    self._emit(f"v_cmpx_eq_u32 vcc, 1, v[{v.v_in_flag()}]")
                if self.outer.tunable.precache_soffset:
                    self._emit(m_in_2d_global_load(v.v_gld_b(), s.s_p_in(), v.v_in_os(), s_in_stride_d0(), s_in_stride_d1(), s.s_in_offset()))
                else:
                    self._emit(m_in_2d_global_load(v.v_gld_b(), s.s_p_in(), v.v_in_os(), s_in_stride_d0(), s_in_stride_d1(), s.s_tmp()))
                if self.outer.tunable.nxe != 0:
                    self._emit(f"s_mov_b64 exec, -1")
            return self._get_deferred()

    class global_load_out_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_out_2d_global_load, m_in_2d_global_load = self.outer.get_macro_global_load()
            return m_out_2d_global_load.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr

            m_out_2d_global_load, m_in_2d_global_load = self.outer.get_macro_global_load()
            s_out_stride_d0, s_out_stride_d1, s_in_stride_d0, s_in_stride_d1 = self.outer.get_symbol_global_load_s_stride_d0_d1()
            with self._deferred_context():
                self._emit(f"; load output")
                if self.outer.tunable.precache_soffset:
                    self._emit(m_out_2d_global_load(v.v_gld_a(), s.s_p_out(), v.v_out_os(), s_out_stride_d0(), s_out_stride_d1(), s.s_out_offset()))
                else:
                    self._emit(m_out_2d_global_load(v.v_gld_a(), s.s_p_out(), v.v_out_os(), s_out_stride_d0(), s_out_stride_d1(), s.s_tmp()))
            return self._get_deferred() 

    class shared_store_in_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_out_2d_shared_store, m_in_2d_shared_store = self.outer.get_macro_shared_store()
            return  m_in_2d_shared_store.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr
            m_out_2d_shared_store, m_in_2d_shared_store = self.outer.get_macro_shared_store()
            with self._deferred_context():
                self._emit(m_in_2d_shared_store(v.v_gld_b(), v.v_sst_b_os()))
            return self._get_deferred()

    class shared_store_out_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_out_2d_shared_store, m_in_2d_shared_store = self.outer.get_macro_shared_store()
            return m_out_2d_shared_store.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr
            m_out_2d_shared_store, m_in_2d_shared_store = self.outer.get_macro_shared_store()
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
            self.s_by                      = sym_t("s_by"                     ,3)
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
            sseq                           = gpr_sequencer_t(30 + 1)
            self.s_gemmk_split             = sym_t("s_gemmk_split"            ,sseq(1))
            self.s_group                   = sym_t("s_group"                  ,sseq(1))

            self.s_ho_x_stride_h           = sym_t("s_ho_x_stride_h"          ,sseq(1))
            self.s_wo_x_stride_w           = sym_t("s_wo_x_stride_w"          ,sseq(1))

            self.s_in_stride_wi            = sym_t("s_in_stride_wi"           ,sseq(1))
            self.s_in_stride_hi            = sym_t("s_in_stride_hi"           ,sseq(1))
            self.s_in_stride_n             = sym_t("s_in_stride_n"            ,sseq(1))
            self.s_out_stride_wo           = sym_t("s_out_stride_wo"          ,sseq(1))
            self.s_out_stride_ho           = sym_t("s_out_stride_ho"          ,sseq(1))
            self.s_out_stride_n            = sym_t("s_out_stride_n"           ,sseq(1))
            
            self.s_wei_stride_k            = sym_t("s_wei_stride_k"           ,sseq(1))

            self.s_out_stride_n_n          = sym_t("s_out_stride_n_n"         ,sseq(1))
            self.s_in_stride_n_n           = sym_t("s_in_stride_n_n"          ,sseq(1))

            self.s_move_slice_n_n          = sym_t("s_move_slice_n"           ,sseq(1))

            self.s_move_slice_n_dsho       = sym_t("s_move_slice_n_dsho"      ,sseq(1))
            self.s_move_slice_n_dswo       = sym_t("s_move_slice_n_dswo"      ,sseq(1))

            self.s_dim_b                   = sym_t("s_dim_b"                  ,sseq(1))
            self.s_dim_e                   = sym_t("s_dim_e"                  ,sseq(1))

            self.s_block_gtc_ie            = sym_t("s_block_gtc_ie"           ,sseq(1))
            self.s_block_gtc_ik            = sym_t("s_block_gtc_ik"           ,sseq(1))
            self.s_block_gtc_ic1           = sym_t("s_block_gtc_ic1"          ,sseq(1))
            self.s_block_gtc_iec0          = sym_t("s_block_gtc_iec0"         ,sseq(1))
            self.s_block_gtc_in            = sym_t('s_block_gtc_in'           ,sseq(1))
            self.s_block_gtc_ig            = sym_t('s_block_gtc_ig'           ,sseq(1))

            self.s_knum                    = sym_t("s_knum"                   ,1)
            self.s_gemm_k_num_n1           = sym_t("s_gemm_k_num_n1"          ,0)
            self.s_gemm_k_num_dsho         = sym_t("s_gemm_k_num_dsho"        ,sseq(1))
            self.s_gemm_k_num_dswo         = sym_t("s_gemm_k_num_dswo"        ,sseq(1))

            self.s_kitr                    = sym_t("s_kitr"                   ,3)
            if outer.tunable.precache_soffset:
                m_out_2d_global_load, m_in_2d_global_load = outer.get_macro_global_load()
                out_npc = m_out_2d_global_load.get_num_precache_soffset()
                in_npc = m_in_2d_global_load.get_num_precache_soffset()
                self.s_in_offset           = sym_t("s_in_offset"              ,sseq(in_npc))   # if this number is zero, it is also OK, since we would not use
                self.s_out_offset          = sym_t("s_out_offset"             ,sseq(out_npc))
            self.s_sub_n                   = sym_t("s_sub_n"                  ,sseq(1))
            if IGEMM_WRW_GTC_DEBUG == 1:
                self.s_dbg                 = sym_t("s_dbg"                    ,sseq(4, 2))
            self.s_k_padded                = sym_t("s_k_padded"               ,sseq(1))
            self.s_c_padded                = sym_t("s_c_padded"               ,sseq(1))
            self.s_out_move_step           = sym_t("s_out_move_step"          ,sseq(1))
            if outer.tunable.nxe == 0:
                self.s_in_move_step        = sym_t("s_in_move_step"           ,sseq(1))
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
            self.outer                    = outer
            if is_vgpr_acc_c:
                self.v_c                  = sym_t("v_c"            ,vseq(outer.tunable.num_vgpr_accumulate_c))
                v_c_num                   = vseq()
            else:
                v_c_resuable_num          = outer.tunable.num_vgpr_accumulate_a + outer.tunable.num_vgpr_accumulate_b + \
                                            outer.tunable.num_vgpr_global_load_a + outer.tunable.num_vgpr_global_load_b + \
                                            14       # from v_sst_a_os to v_co_sst
                v_c_coalescing_num        = outer.tunable.num_agpr_accumulate_c // outer.coalescing_store_groups
                v_c_needed                = (v_c_coalescing_num - v_c_resuable_num) if (v_c_coalescing_num - v_c_resuable_num) > 0 else 0

                v_c_needed                = v_c_needed if v_c_needed > 2 else 2  # let at least 2
                self.v_c                  = sym_t("v_c"            ,vseq(v_c_needed), f"coalescing:{v_c_coalescing_num}, needed:{v_c_needed}, resuable:{v_c_resuable_num}")
            self.v_a                      = sym_t("v_a"            ,vseq(outer.tunable.num_vgpr_accumulate_a))
            self.v_b                      = sym_t("v_b"            ,vseq(outer.tunable.num_vgpr_accumulate_b))
            self.v_gld_a                  = sym_t("v_gld_a"        ,vseq(outer.tunable.num_vgpr_global_load_a))
            self.v_gld_b                  = sym_t("v_gld_b"        ,vseq(outer.tunable.num_vgpr_global_load_b))
            self.v_sst_a_os               = sym_t("v_sst_a_os"     ,vseq(1))
            self.v_sst_b_os               = sym_t("v_sst_b_os"     ,vseq(1))
            self.v_sld_a_os               = sym_t("v_sld_a_os"     ,vseq(1))
            self.v_sld_b_os               = sym_t("v_sld_b_os"     ,vseq(1))
            self.v_in_ihi                 = sym_t("v_in_ihi"       ,vseq(1))
            self.v_in_iwi                 = sym_t("v_in_iwi"       ,vseq(1))
            self.v_in_os                  = sym_t("v_in_os"        ,vseq(1))
            self.v_in_os_base             = sym_t("v_in_os_base"   ,vseq(1))
            self.v_out_iho                = sym_t("v_out_iho"      ,vseq(1))
            self.v_out_iwo                = sym_t("v_out_iwo"      ,vseq(1))
            self.v_in_ihi_max             = sym_t("v_in_ihi_max"   ,vseq(1))
            self.v_in_iwi_max             = sym_t("v_in_iwi_max"   ,vseq(1))
            self.v_gtc_in                 = sym_t("v_gtc_in"       ,vseq(1))
            self.v_out_os                 = sym_t("v_out_os"       ,vseq(1))
            self.v_out_os_base            = sym_t("v_out_os_base"  ,vseq(1))
            if outer.tunable.nxe != 0:
                self.v_in_flag            = sym_t("v_in_flag"      ,vseq(1))
            self.v_co_sst                 = sym_t("v_co_sst"       ,vseq(1))
            self.v_co_sld                 = sym_t("v_co_sld"       ,vseq(1))
            self.v_wei_os                 = sym_t("v_wei_os"       ,vseq(1))

            self.v_wei_iy                 = sym_t("v_wei_iy"        ,vseq(1))
            self.v_wei_ix                 = sym_t("v_wei_ix"        ,vseq(1))
            self.v_wei_c_flag             = sym_t("v_wei_c_flag"    ,vseq(1))

            self.v_gtc_iec0               = sym_t("v_gtc_iec0"      ,v_c_num - 1  if is_vgpr_acc_c else vseq(1))
            self.v_gtc_ic1                = sym_t("v_gtc_ic1"       ,v_c_num - 2  if is_vgpr_acc_c else vseq(1))
            self.v_gtc_ic0                = sym_t("v_gtc_ic0"       ,v_c_num - 3  if is_vgpr_acc_c else vseq(1))
            self.v_gtc_ie                 = sym_t("v_gtc_ie"        ,v_c_num - 3  if is_vgpr_acc_c else vseq(1))
            self.v_gtc_ik                 = sym_t("v_gtc_ik"        ,v_c_num - 4  if is_vgpr_acc_c else vseq(1))

            self.v_gtc_inb_a              = sym_t("v_gtc_inb_a"     ,v_c_num - 9  if is_vgpr_acc_c else vseq(1))
            self.v_gemm_in                = sym_t("v_gemm_in"       ,v_c_num - 11 if is_vgpr_acc_c else vseq(1))
            self.v_gemm_im                = sym_t("v_gemm_im"       ,v_c_num - 12 if is_vgpr_acc_c else vseq(1))

            if is_vgpr_acc_c:
                if v_c_num < 16:
                    self.v_wei_ic1        = sym_t("v_wei_ic1"        ,vseq(1))
                    self.v_wei_iec0       = sym_t("v_wei_iec0"       ,vseq(1))

                    self.v_co_sub_m_index = sym_t("v_co_sub_m_index" ,vseq(1))
                    self.v_co_sub_n_index = sym_t("v_co_sub_n_index" ,vseq(1))
                else:
                    self.v_wei_ic1        = sym_t("v_wei_ic1"        ,v_c_num - 13)
                    self.v_wei_iec0       = sym_t("v_wei_iec0"       ,v_c_num - 14)

                    self.v_co_sub_m_index = sym_t("v_co_sub_m_index" ,v_c_num - 18)
                    self.v_co_sub_n_index = sym_t("v_co_sub_n_index" ,v_c_num - 19)
            else:
                self.v_wei_ic1            = sym_t("v_wei_ic1"        ,vseq(1))
                self.v_wei_iec0           = sym_t("v_wei_iec0"       ,vseq(1))
                self.v_co_sub_m_index     = sym_t("v_co_sub_m_index" ,vseq(1))
                self.v_co_sub_n_index     = sym_t("v_co_sub_n_index" ,vseq(1))

            self.v_cur_k                  = sym_t("v_cur_k"          ,vseq(1))
            self.v_tmp                    = sym_t("v_tmp"            ,vseq(8, 2))
            if IGEMM_WRW_GTC_DEBUG == 1:
                self.v_dbg                = sym_t("v_dbg"            ,vseq(2, 2))
            total_vgpr                    = vseq()
            if outer.tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
                # if xdlops agpr is larger than vgpr usage, must change vgpr count to agpr
                total_vgpr                = max(total_vgpr, outer.tunable.num_agpr_accumulate_c)
            self.v_end                    = sym_t("v_end"          ,total_vgpr)

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

        _, ta_nb,      _, ta_k  = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        _, tb_nb, tb_ec0, tb_c1 = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        return ta_k, ta_nb, tb_nb, tb_ec0, tb_c1 # M, K, N


    def get_cluster_lengths(self):
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4

        _, ca_nb,      _, ca_k  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        _, cb_nb, cb_ec0, cb_c1 = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        return ca_k, ca_nb, cb_nb, cb_ec0, cb_c1 # M, K, N

    def get_dims_lengths(self):
        ta_k, ta_nb, tb_nb, tb_ec0, tb_c1 = self.get_thread_lengths()
        ca_k, ca_nb, cb_nb, cb_ec0, cb_c1 = self.get_cluster_lengths()

        na_k,  na_nb          = ta_k  * ca_k,  ta_nb  * ca_nb
        nb_nb, nb_ec0, nb_c1 = tb_nb * cb_nb, tb_ec0 * cb_ec0, tb_c1 * cb_c1

        return na_k, na_nb, nb_nb, nb_ec0, nb_c1

    def get_thread_copy_dims(self):
        ta_k, ta_nb, tb_nb, tb_ec0, tb_c1 = self.get_thread_lengths()
        in_thread_copy_dims     = [tb_nb, tb_ec0, tb_c1]
        out_thread_copy_dims    = [ta_nb, ta_k]
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

    def get_gemmk_pack(self):
        ta_k, ta_nb, tb_nb, tb_ec0, tb_c1 = self.get_thread_lengths()
        assert ta_nb == tb_nb
        # It will be 1 for the first step
        return igemm_gcd(ta_nb, 4)

    def get_macro_global_load(self):
        inline = True if self.tunable.fma_interleave else False
        ta_k, ta_nb, tb_nb, tb_ec0, tb_c1           = self.get_thread_lengths()
        in_thread_copy_dims, out_thread_copy_dims   = self.get_thread_copy_dims()
        in_thread_copy_index, out_thread_copy_index = self.get_thread_copy_index()

        ctrl_in_gld  = ctrl_2d_global_load_t()
        ctrl_out_gld = ctrl_2d_global_load_t()

        ctrl_in_gld.vector_d1  = igemm_gcd(tb_c1, 4)
        ctrl_out_gld.vector_d1 = igemm_gcd(ta_k, 4)

        if self.in_thread_copy_ndim == 2:
            ctrl_in_gld.length_d0 = in_thread_copy_dims[in_thread_copy_index[0]]
            ctrl_in_gld.length_d1 = in_thread_copy_dims[in_thread_copy_index[1]]
        elif self.in_thread_copy_ndim == 1:
            ctrl_in_gld.length_d0 = 1
            ctrl_in_gld.length_d1 = in_thread_copy_dims[in_thread_copy_index[0]]
        else:
            ctrl_in_gld.length_d0 = 1
            ctrl_in_gld.length_d1 = 1

        if ctrl_in_gld.length_d1 // ctrl_in_gld.vector_d1 > 1:
            ctrl_in_gld.dim_conti_flag = 1

        if self.out_thread_copy_ndim == 2:
            ctrl_out_gld.length_d0 = out_thread_copy_dims[out_thread_copy_index[0]]
            ctrl_out_gld.length_d1 = out_thread_copy_dims[out_thread_copy_index[1]]
        elif self.out_thread_copy_ndim == 1:
            ctrl_out_gld.length_d0 = 1
            ctrl_out_gld.length_d1 = out_thread_copy_dims[out_thread_copy_index[0]]
        else:
            ctrl_out_gld.length_d0 = 1
            ctrl_out_gld.length_d1 = 1

        if ctrl_out_gld.length_d1 // ctrl_out_gld.vector_d1 > 1:
            ctrl_out_gld.dim_conti_flag = 1

        if self.tunable.precache_soffset:
            return macro_igemm_2d_global_load_precache_soffset_t(self.mc, ctrl_out_gld, inline), \
                    macro_igemm_2d_global_load_precache_soffset_t(self.mc, ctrl_in_gld, inline)
        else:
            return macro_igemm_2d_global_load_t(self.mc, ctrl_out_gld),  macro_igemm_2d_global_load_t(self.mc, ctrl_in_gld, inline)

    def get_macro_global_store(self):
        return macro_igemm_write_4d_strided_t(self.mc)

    def get_macro_shared_store(self):
        in_thread_copy_dims, out_thread_copy_dims   = self.get_thread_copy_dims()
        in_thread_copy_index, out_thread_copy_index = self.get_thread_copy_index()
        na_k, na_nb, nb_nb, nb_ec0, nb_c1           = self.get_dims_lengths()
        ta_k, ta_nb, tb_nb, tb_ec0, tb_c1           = self.get_thread_lengths()
        data_byte = amdgpu_precision_data_byte(self.tunable.precision)
        k_pack = self.get_gemmk_pack()
        length_dp_a = igemm_gcd(ta_k, 4)
        length_dp_b = igemm_gcd(tb_c1, 4)

        if not self.tunable.tensor_a_pass_through:
            # out is gemm_k * gemm_n * k_pack
            out_sst_ctrl = ctrl_3d_shared_store_t()
            out_sst_ctrl.length_d0 = 1
            out_sst_ctrl.length_d1 = int(ta_k / length_dp_a) # ta_k
            out_sst_ctrl.length_dp = length_dp_a # k_pack
            out_sst_ctrl.stride_d0 = 1
            out_sst_ctrl.stride_d1 = length_dp_a * data_byte # k_pack * data_byte

        if not self.tunable.tensor_b_pass_through:
            # input is gemm_k * gemm_m * k_pack
            in_sst_ctrl = ctrl_3d_shared_store_t()
            in_sst_ctrl.length_d0 = tb_ec0
            in_sst_ctrl.length_d1 = int(tb_c1 / length_dp_b) # tb_c1
            in_sst_ctrl.length_dp = length_dp_b # k_pack
            in_sst_ctrl.stride_d0 = nb_c1 * k_pack * data_byte
            in_sst_ctrl.stride_d1 = length_dp_b * data_byte # k_pack * data_byte

        inline = True if self.tunable.fma_interleave else False 
        return macro_igemm_3d_shared_store_t(self.mc, out_sst_ctrl, inline) if not self.tunable.tensor_a_pass_through else None, \
            macro_igemm_3d_shared_store_t(self.mc, in_sst_ctrl, inline) if not self.tunable.tensor_b_pass_through else None

    def get_macro_in_out_update_os(self):
        inline = True if self.tunable.fma_interleave else False
        return self.macro_igemm_wrw_gtc_in_out_update_os_t(self.mc, amdgpu_precision_data_byte(self.tunable.precision), inline)

    def get_macro_in_update_hw(self):
        inline = True if self.tunable.fma_interleave else False
        return self.macro_igemm_wrw_gtc_in_update_hw_t(self.mc, inline)

    def get_macro_out_update_hw(self):
        inline = True if self.tunable.fma_interleave else False
        #if self.tunable.nxe != 0:
        return self.macro_igemm_wrw_gtc_out_update_hw_t(self.mc, inline)
        #return None
   
    def get_macro_set_flag_hw(self):
        inline = True if self.tunable.fma_interleave else False
        return self.macro_igemm_wrw_gtc_set_flag_hw(self.mc, inline)

    def get_macro_move_slice_window(self):
        inline = True if self.tunable.fma_interleave else False
        return self.macro_igemm_wrw_gtc_move_slice_window_n_dsho_dswo(self.mc, self.tunable, inline)

    def get_symbol_global_load_s_stride_d0_d1(self):
        ta_k, ta_nb, tb_nb, tb_ec0, tb_c1 = self.get_thread_lengths()
        # get the symbol object that load 2d may use
        s = self.sgpr
        s_dummy = sym_t("s_dummy")
        in_thread_copy_index, out_thread_copy_index = self.get_thread_copy_index()

        out_stride_gprs = [s_dummy,
                    s_dummy,
                    s_dummy,
                    s_dummy]

        in_stride_gprs = [s_dummy,
                    s_dummy,
                    s_dummy,
                    s_dummy]

        if self.in_thread_copy_ndim == 2:
            s_in_stride_d0 = in_stride_gprs[in_thread_copy_index[0]]
            s_in_stride_d1 = in_stride_gprs[in_thread_copy_index[1]]
        elif self.in_thread_copy_ndim == 1:
            s_in_stride_d0 = s_dummy
            s_in_stride_d1 = in_stride_gprs[in_thread_copy_index[0]]
        else:
            s_in_stride_d0 = s_dummy
            s_in_stride_d1 = in_stride_gprs[-1]

        if self.out_thread_copy_ndim == 1:
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

        return s_out_stride_d0, s_out_stride_d1, s_in_stride_d0, s_in_stride_d1

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

        ta_k, ta_nb, tb_nb, tb_ec0, tb_c1 = self.get_thread_lengths()
        ca_k, ca_nb, cb_nb, cb_ec0, cb_c1 = self.get_cluster_lengths()
        na_k, na_nb, nb_nb, nb_ec0, nb_c1 = self.get_dims_lengths()

        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        m_in_update_os   = self.get_macro_in_out_update_os()
        m_in_update_hw   = self.get_macro_in_update_hw()
        m_out_update_os   = self.get_macro_in_out_update_os()
        m_out_update_hw   = self.get_macro_out_update_hw()
        m_set_flag_hw     = self.get_macro_set_flag_hw()

        m_out_2d_global_load, m_in_2d_global_load = self.get_macro_global_load()
        s_out_stride_d0, s_out_stride_d1, s_in_stride_d0, s_in_stride_d1 = self.get_symbol_global_load_s_stride_d0_d1()

        tc_index_dispatcher = igemm_thread_cluster_index_dispatcher_t(self.mc)
        tc_index_accumulator = igemm_thread_cluster_index_accumulator_t(self.mc)

        m_int_div_rem_vv = macro_int_div_rem_vv_t(self.mc)
        m_int_div_rem_vs = macro_int_div_rem_vs_t(self.mc)
        m_int_div_rem_ss = macro_int_div_rem_ss_t(self.mc)
        s_dummy = sym_t("s_dummy")

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
            self._emit(f"v_add_lshl_u32 v[{v.v_dbg()}], v0, v1, 2")
            self._emit(f"s_load_dwordx2 s[{s.s_dbg((0,1))}], s[s_ka:s_ka+1], k_p_wei")
            self._emit(f"s_mov_b32 s[{s.s_dbg(2)}], s[{s.s_bx()}]")

        self._emit(f"; input, thread(1,nb,ec0,c1): {1}x{tb_nb}x{tb_ec0}x{tb_c1}, cluster(1,nb,ec0,c1): {1}x{cb_nb}x{cb_ec0}x{cb_c1}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        self._emit(tc_index_dispatcher(v.v_gtc_ic1(),   v.v_tmp(), cb_c1, tb_c1))      # merged dimension no need to do shift per thread here, do shift later
        self._emit(tc_index_dispatcher(v.v_gtc_iec0(),  v.v_tmp(), cb_ec0,  tb_ec0))
        self._emit(tc_index_dispatcher(v.v_gtc_inb_a(), v.v_tmp(), cb_nb, tb_nb, True))      # merged dimension no need to do shift per thread here, do shift later
        self._emit_empty_line()
        self._emit(f"; output, thread(1,nb,1,k): {1}x{ta_nb}x{1}x{ta_k}, cluster(1,nb,1,k) {1}x{ca_nb}x{1}x{ca_k}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        self._emit(tc_index_dispatcher(v.v_gtc_ik(), v.v_tmp(), ca_k, ta_k, True))
        self._emit_empty_line()
        self._emit(f"s_mov_b32 s[{s.s_p_in(3)}], 0x27000")
        self._emit(f"s_mov_b32 s[{s.s_p_wei(2)}], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_wei(3)}], 0x27000")
        self._emit(f"s_mov_b32 s[{s.s_p_out(3)}], 0x27000")
        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit_empty_line()

        self._emit(f"; calculate index")

        self._emit(f"s_lshr_b32 s[{s.s_sub_n()}], s[{s.s_n()}], s[{s.s_gemmk_split()}]")
        self._emit(f"s_mul_i32 s[{s.s_in_stride_wi()}], s[{s.s_c()}], s[{s.s_group()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_wi()}], s[{s.s_in_stride_wi()}]")
        self._emit(f"s_mul_i32 s[{s.s_in_stride_n()}], s[{s.s_hi()}], s[{s.s_tmp(2)}]")
        if self.tunable.nxe != 0:
            self._emit(f"s_mul_i32 s[{s.s_dim_e()}], s[{s.s_x()}], s[{s.s_y()}]")
            self._emit(f"s_mul_i32 s[{s.s_wei_stride_k()}], s[{s.s_c()}], s[{s.s_dim_e()}]")
        else:
            self._emit(f"s_mov_b32 s[{s.s_wei_stride_k()}], s[{s.s_c()}]")
        
        self._emit(f"s_mul_i32 s[{s.s_out_stride_wo()}], s[{s.s_k()}], s[{s.s_group()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_wo()}], s[{s.s_out_stride_wo()}]")
        self._emit(f"s_mul_i32 s[{s.s_out_stride_n()}], s[{s.s_ho()}], s[{s.s_tmp(2)}]")

        self._emit(f"s_mul_i32 s[{s.s_dim_b()}], s[{s.s_ho()}], s[{s.s_wo()}]")

        if self.tunable.gemm_k_global_split != 0:
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

        if self.tunable.nxe == 1:
            self._emit(f"; n1b transform")
            self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_in(), v.v_gtc_inb_a(), s.s_dim_b(), v.v_tmp(), s.s_tmp()))
            self._emit(m_int_div_rem_vs(v.v_out_iwo(), v.v_out_iho(), v.v_tmp(4), s.s_wo(), v.v_tmp(), s.s_tmp()))
            self._emit_empty_line()

        self._emit(f"; pad gemm_m if needed")
        self._emit(f"s_add_u32 s[{s.s_tmp()}], {self.tunable.gemm_m_per_block - 1}, s[{s.s_k()}]")
        self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_m_per_block)}")
        self._emit(f"s_lshl_b32 s[{s.s_k_padded()}], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_m_per_block)}")

        self._emit(f"; pad c now. TODO: pad merged dimension ec")
        self._emit(f"s_add_u32 s[{s.s_tmp()}], {self.tunable.gemm_n_per_block - 1}, s[{s.s_c()}]")
        self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_n_per_block)}")
        self._emit(f"s_lshl_b32 s[{s.s_c_padded()}], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_n_per_block)}")

        self._emit_empty_line()
        self._emit(f"; add block i_n")
        self._emit(f"; gemm_m_per_block:{self.tunable.gemm_m_per_block}, gemm_n_per_block:{self.tunable.gemm_n_per_block}")
        # calculate group index
        if self.tunable.nxe != 0:
            self._emit(f"s_mul_i32 s[0], s[{s.s_c_padded()}], s[{s.s_dim_e()}]")
            self._emit(f"s_lshr_b32 s[0], s[0], {igemm_log2(self.tunable.gemm_n_per_block)}")
        else:
            self._emit(f"s_lshr_b32 s[0], s[{s.s_c_padded()}], {igemm_log2(self.tunable.gemm_n_per_block)}")
        
        self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_k_padded()}], {igemm_log2(self.tunable.gemm_m_per_block)}")
        self._emit(f"s_mul_i32 s[1], s[0], s[{s.s_tmp()}]")
        
        # gemmk split method
        if IGEMM_WRW_GTC_N_SPLIT_FIRST == 0:
            self._emit(f"s_lshl_b32 s[3], s[1], s[{s.s_gemmk_split()}]")
            self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_block_gtc_ig(), s.s_bx(), '3', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
            self._emit(m_int_div_rem_ss(s.s_bx(), s.s_block_gtc_in(), s.s_tmp(4), '1', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
            self._emit(f"s_mul_i32 s[{s.s_block_gtc_in()}], s[{s.s_block_gtc_in()}], s[{s.s_sub_n()}]")
        else:
            self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], 1, s[{s.s_gemmk_split()}]")
            self._emit(f"s_sub_u32 s[{s.s_tmp(3)}], s[{s.s_tmp(3)}], 1")
            self._emit(f"s_and_b32 s[{s.s_block_gtc_in()}], s[{s.s_bx()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_mul_i32 s[{s.s_block_gtc_in()}], s[{s.s_block_gtc_in()}], s[{s.s_sub_n()}]")
            self._emit(f"s_lshr_b32 s[{s.s_bx()}], s[{s.s_bx()}], s[{s.s_gemmk_split()}]")
            self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_block_gtc_ig(), s.s_bx(), '1', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
            self._emit(f"s_mov_b32 s[{s.s_bx()}], s[{s.s_tmp(4)}]")

        self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_tmp(5), s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
        self._emit(f"; s_tmp+4:block_gtc_in, s_tmp+5:block_gtc_im")
        self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ik()}], s[{s.s_tmp(5)}], {igemm_log2(self.tunable.gemm_m_per_block)}")
        self._emit(f"s_lshr_b32 s[0], s[{s.s_c_padded()}], {igemm_log2(int(nb_c1 / nb_ec0))} ; total number of c1")
        self._emit(m_int_div_rem_ss(s.s_block_gtc_ic1(), s.s_block_gtc_iec0(), s.s_tmp(4), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
        if nb_c1 != 1:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ic1()}], s[{s.s_block_gtc_ic1()}], {igemm_log2(nb_c1)}")
        if nb_ec0 != 1:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_iec0()}], s[{s.s_block_gtc_iec0()}], {igemm_log2(nb_ec0)}")

        self._emit_empty_line()

        self._emit("; config for output and input range")
        self._emit(f"s_add_u32 s[{s.s_tmp()}], s[{s.s_block_gtc_in()}], s[{s.s_sub_n()}]")
        self._emit(f"s_mul_i32 s[{s.s_p_out(2)}], s[{s.s_out_stride_n()}], s[{s.s_tmp()}]")
        self._emit(f"s_lshl_b32 s[{s.s_p_out(2)}], s[{s.s_p_out(2)}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_p_in(2)}], s[{s.s_in_stride_n()}], s[{s.s_tmp()}]")
        self._emit(f"s_lshl_b32 s[{s.s_p_in(2)}], s[{s.s_p_in(2)}], {igemm_log2(data_byte)}")
        
        if self.tunable.nxe == 1:
            self._emit(f"; ec0 transform")
            if cb_ec0 == 1:
                self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_iec0()}]")
            else:
                self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_iec0()}], v[{v.v_gtc_iec0()}]")
            self._emit(m_int_div_rem_vs(v.v_gtc_ie(), v.v_gtc_ic0(), v.v_tmp(5), s.s_dim_e(), v.v_tmp(), s.s_tmp()))
            self._emit(m_int_div_rem_vs(v.v_wei_ix(), v.v_wei_iy(), v.v_gtc_ie(), s.s_x(), v.v_tmp(), s.s_tmp()))
            self._emit_empty_line()

            # ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h,   here make sure iy <- iy * s_dilation_h - s_pad_h before hand
            # iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w,   here make sure ix <- ix * s_dilation_w - s_pad_w before hand
            self._emit(f"v_mul_u32_u24 v[{v.v_tmp()}], s[{s.s_dilation_h()}], v[{v.v_wei_iy()}]")
            self._emit(f"v_mul_u32_u24 v[{v.v_tmp(1)}], s[{s.s_dilation_w()}], v[{v.v_wei_ix()}]")
            self._emit(f"v_sub_i32 v[{v.v_wei_iy()}], v[{v.v_tmp()}], s[{s.s_pad_h()}]")
            self._emit(f"v_sub_i32 v[{v.v_wei_ix()}], v[{v.v_tmp(1)}], s[{s.s_pad_w()}]")

            # move by wi and hi, compute new boundary
            self._emit(f"s_mul_i32 s[{s.s_ho_x_stride_h()}], s[{s.s_ho()}], s[{s.s_stride_h()}]")
            self._emit(f"s_mul_i32 s[{s.s_wo_x_stride_w()}], s[{s.s_wo()}], s[{s.s_stride_w()}]")
            self._emit(f"v_add_i32 v[{v.v_in_ihi_max()}], v[{v.v_wei_iy()}], s[{s.s_ho_x_stride_h()}]")
            self._emit(f"v_add_i32 v[{v.v_in_iwi_max()}], v[{v.v_wei_ix()}], s[{s.s_wo_x_stride_w()}]")

            m_in_update_hw   = self.get_macro_in_update_hw()
            self._emit(m_in_update_hw(v.v_in_ihi(), v.v_in_iwi(), v.v_out_iho(), v.v_out_iwo(), s.s_stride_h(), s.s_stride_w(), v.v_wei_iy(), v.v_wei_ix(), s.s_dilation_h(), s.s_dilation_w(), s.s_pad_h(), s.s_pad_w(), v.v_tmp()))
        else:
            if cb_ec0 == 1:
                self._emit(f"v_mov_b32 v[{v.v_gtc_ic0()}], s[{s.s_block_gtc_iec0()}]")
            else:
                self._emit(f"v_add_u32 v[{v.v_gtc_ic0()}], s[{s.s_block_gtc_iec0()}], v[{v.v_gtc_iec0()}]")

        self._emit(f"; calculate input offset")
        # compute group distance
        self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ig()}], s[{s.s_block_gtc_ig()}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_block_gtc_ig()}], s[{s.s_c()}]")
        self._emit(f"s_sub_u32 s[{s.s_p_in(2)}], s[{s.s_p_in(2)}], s[{s.s_tmp()}]")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], s[{s.s_block_gtc_ic1()}], {igemm_log2(data_byte)}")
        self._emit(f"s_sub_u32 s[{s.s_p_in(2)}], s[{s.s_p_in(2)}], s[{s.s_tmp(3)}]")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp(3)}]")
        self._emit_empty_line()

        self._emit(tc_index_accumulator(v.v_tmp(1), v.v_gtc_ic0(), v.v_gtc_ic1(), cb_ec0, cb_c1, 0, cb_c1))
        if self.tunable.nxe != 0:
            self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_gtc_in()}], s[{s.s_block_gtc_in()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_in_stride_n()}], v[{v.v_tmp()}]")
            self._emit(f"v_add_lshl_u32 v[{v.v_in_os_base()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
            self._emit(m_in_update_os(v.v_in_os(), v.v_in_os_base(), v.v_in_ihi(), v.v_in_iwi(), s.s_wi(), s.s_in_stride_wi(), v.v_tmp()))
            self._emit(m_set_flag_hw(v.v_in_flag(), v.v_in_ihi(), v.v_in_iwi(), s.s_hi(), s.s_wi()))
        else:
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_in_stride_n()}], s[{s.s_block_gtc_in()}]")
            self._emit(f"v_add_lshl_u32 v[{v.v_in_os_base()}], v[{v.v_tmp(1)}], s[{s.s_tmp()}], {igemm_log2(data_byte)}")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], v[{v.v_gtc_inb_a()}], s[{s.s_in_stride_wi()}]")
            self._emit(f"v_lshl_add_u32 v[{v.v_in_os()}], v[{v.v_tmp()}], 2, v[{v.v_in_os_base()}]")
        
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
            if tb_nb * tb_ec0 > 1:
                self._emit(m_in_2d_global_load.init_precache_soffset(s_in_stride_d0(), s_in_stride_d1(), s.s_in_offset(), s.s_tmp()))

        # load out
        self._emit(self.global_load_in())
        self._emit_empty_line()

        self._emit(f"; calculate out offset")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_block_gtc_ig()}], s[{s.s_k()}]")
        self._emit(f"s_sub_u32 s[{s.s_p_out(2)}], s[{s.s_p_out(2)}], s[{s.s_tmp()}]")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
        self._emit_empty_line()
        self._emit(f"v_add_u32 v[{v.v_cur_k()}], s[{s.s_block_gtc_ik()}], v[{v.v_gtc_ik()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_out_stride_n()}], s[{s.s_block_gtc_in()}]")
        self._emit(f"v_add_lshl_u32 v[{v.v_out_os_base()}], v[{v.v_cur_k()}], s[{s.s_tmp()}], {igemm_log2(data_byte)}")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], v[{v.v_gtc_inb_a()}], s[{s.s_out_stride_wo()}]")
        self._emit(f"v_lshl_add_u32 v[{v.v_out_os()}], v[{v.v_tmp()}], 2, v[{v.v_out_os_base()}]")
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
            if ta_nb > 1:
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

        self._emit(f"; LDS store, in: 1,nb,ec0,c1: {1}x{tb_nb}x{tb_ec0}x{tb_c1}, {1}x{cb_nb}x{cb_ec0}x{cb_c1}")
        if nb_ec0 == 1:
            self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_ic1()}]")
        else:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ic0()}], {igemm_log2(nb_c1)}, v[{v.v_gtc_ic1()}]")

        if cb_nb != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_inb_a()}], {igemm_log2(nb_ec0*nb_c1)}, v[{v.v_tmp()}]")

        self._emit(f"v_lshlrev_b32 v[{v.v_sst_b_os()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")
        self._emit(f"v_add_u32 v[{v.v_sst_b_os()}], {self.tunable.lds_a_np2}, v[{v.v_sst_b_os()}]")
        self._emit_empty_line()

        self._emit(f"; LDS store, out: 1,nb,1,k: {1}x{ta_nb}x{1}x{ta_k}, {1}x{ca_nb}x{1}x{ca_k}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_ik()}]")
        if ca_nb != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_inb_a()}], {igemm_log2(na_k)}, v[{v.v_tmp()}]")
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
        self._emit(f"s_lshl_b32 s[{s.s_tmp()}], s[{s.s_block_gtc_ic1()}], {igemm_log2(data_byte)}")
        self._emit(f"s_add_u32 s[{s.s_p_wei()}], s[{s.s_p_wei()}], s[{s.s_tmp()}]")

        self._emit_empty_line()
        self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], s[{s.s_block_gtc_ik()}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_wei_stride_k()}], s[{s.s_tmp(3)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_wei_stride_k()}], s[{s.s_tmp(3)}]")
        self._emit(f"s_add_u32 s[{s.s_p_wei()}], s[{s.s_p_wei()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_wei(1)}], s[{s.s_p_wei(1)}], s[{s.s_tmp(1)}]")
        self._emit_empty_line()

        self._emit(f"; compute v_co_sub_n_index along ec0 x c1 : {nb_ec0}x{nb_c1}")
        if nb_c1 != 1:
            self._emit(f"v_and_b32 v[{v.v_wei_ic1()}], {nb_c1 - 1}, v[{v.v_co_sub_n_index()}]     ; => C1")
            if nb_ec0 != 1:
                self._emit(f"v_lshrrev_b32 v[{v.v_wei_iec0()}], {igemm_log2(nb_c1)}, v[{v.v_co_sub_n_index()}]  ; => EC0")

        self._emit_empty_line()
        self._emit(f"; add wei_ic0, wei_ic1")
        if nb_ec0 != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_wei_os()}], v[{v.v_gtc_ic0()}], {igemm_log2(nb_c1)}, v[{v.v_wei_ic1()}] ; v_wei_os means v_wei_ic now")
            self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_wei_os()}], s[{s.s_block_gtc_ic1()}]")
            self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_c()}], v[{v.v_tmp()}]")
            self._emit(f"v_cndmask_b32 v[{v.v_wei_c_flag()}],  0, 1, vcc")
        else:
            self._emit(f"v_mov_b32 v[{v.v_wei_os()}], v[{v.v_wei_ic1()}] ; v_wei_os means v_wei_ic now")
            self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_wei_os()}], s[{s.s_block_gtc_ic1()}]")
            self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_c()}], v[{v.v_tmp()}]")
            self._emit(f"v_cndmask_b32 v[{v.v_wei_c_flag()}],  0, 1, vcc")

        self._emit(f"; add i_k")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_k()}], v[{v.v_co_sub_m_index()}]")
        self._emit(f"v_add_u32 v[{v.v_wei_os()}], v[{v.v_wei_os()}], v[{v.v_tmp()}]")
        if self.tunable.nxe == 1:
            self._emit(f"; add y, x, so called gtc_ie")
            self._emit(f"v_mad_u32_u24 v[{v.v_wei_os()}], s[{s.s_c()}], v[{v.v_gtc_ie()}], v[{v.v_wei_os()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_wei_os()}], {igemm_log2(data_byte)}, v[{v.v_wei_os()}]")

        self._emit(f"; move slice step for output tensor")
        self._emit(f"s_lshl_b32 s[{s.s_out_move_step()}], s[{s.s_k()}], {igemm_log2(data_byte * na_nb)}")
        if self.tunable.nxe == 0:
            self._emit(f"s_lshl_b32 s[{s.s_in_move_step()}], s[{s.s_c()}], {igemm_log2(data_byte * na_nb)}")

        self._emit(f"; move slice stride")
        assert na_nb == self.tunable.gemm_k_per_block
        if self.tunable.nxe != 0:
            self._emit(f"s_mov_b32 s[0], {na_nb}")
            self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_move_slice_n_n(), '0', s.s_dim_b(), v.v_tmp(4), v.v_tmp(), s.s_tmp()))
            self._emit(m_int_div_rem_ss(s.s_move_slice_n_dswo(), s.s_move_slice_n_dsho(), s.s_tmp(4), s.s_wo(), v.v_tmp(4), v.v_tmp(), s.s_tmp()))
            self._emit_empty_line()
            self._emit(f"; convert dswo and dsho to dswi and dshi, dswi=dswo*stride_w, dshi=dsho*stride_h")
            self._emit(f"s_mul_i32 s[{s.s_move_slice_n_dswo()}], s[{s.s_move_slice_n_dswo()}], s[{s.s_stride_w()}]  ; might be 0 or larger")
            self._emit(f"s_mul_i32 s[{s.s_move_slice_n_dsho()}], s[{s.s_move_slice_n_dsho()}], s[{s.s_stride_h()}]  ; might be 0 or larger")
            self._emit(f"s_mul_i32 s[{s.s_in_stride_n_n()}], s[{s.s_move_slice_n_n()}], s[{s.s_in_stride_n()}]  ; might be 0 or larger")
            self._emit(self.try_shift_stride(s.s_in_stride_n_n, igemm_log2(data_byte)))
            self._emit(self.try_shift_stride(s.s_in_stride_n, igemm_log2(data_byte)))
            self._emit(self.try_shift_stride(s.s_out_stride_n, igemm_log2(data_byte)))
        self._emit(self.try_shift_stride(s.s_wei_stride_k, igemm_log2(data_byte)))

        self._emit(f"s_mul_i32 s[{s.s_knum()}], s[{s.s_dim_b()}], s[{s.s_sub_n()}]")
        self._emit(f"s_add_i32 s[{s.s_knum()}], s[{s.s_knum()}], {self.tunable.gemm_k_per_block - 1}")
        self._emit(f"s_lshr_b32 s[{s.s_knum()}], s[{s.s_knum()}], {igemm_log2(self.tunable.gemm_k_per_block)}")
        self._emit(f"s_lshl_b32 s[{s.s_knum()}], s[{s.s_knum()}], {igemm_log2(self.tunable.gemm_k_per_block)}")
        self._emit_empty_line()

    def emit_kernel_fma_main_loop(self):
        s = self.sgpr
        v = self.vgpr
        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        def move_slice_window_b():
            m_in_update_os       = self.get_macro_in_out_update_os()
            m_in_update_hw       = self.get_macro_in_update_hw()
            m_set_flag_hw         = self.get_macro_set_flag_hw()
            with self._deferred_context():
                if self.tunable.nxe != 0:
                    m_move_slice_window   = self.get_macro_move_slice_window()
                    self._emit(m_move_slice_window(v.v_in_ihi(), v.v_in_iwi(), v.v_in_ihi_max(), v.v_in_iwi_max(),
                        s.s_move_slice_n_dsho(), s.s_move_slice_n_dswo(), v.v_in_os_base(), s.s_in_stride_n(), 
                        s.s_in_stride_n_n(), s.s_stride_h(), s.s_ho_x_stride_h(), s.s_wo_x_stride_w()))
                    #self._emit(m_in_update_hw(v.v_in_ihi(), v.v_in_iwi(), v.v_out_iho(), v.v_out_iwo(), s.s_stride_h(), s.s_stride_w(), v.v_wei_iy(), v.v_wei_ix(), s.s_dilation_h(), s.s_dilation_w(), s.s_pad_h(), s.s_pad_w(), v.v_tmp()))
                    self._emit(m_in_update_os(v.v_in_os(), v.v_in_os_base(), v.v_in_ihi(), v.v_in_iwi(), s.s_wi(), s.s_in_stride_wi(), v.v_tmp()))
                    self._emit(m_set_flag_hw(v.v_in_flag(), v.v_in_ihi(), v.v_in_iwi(), s.s_hi(), s.s_wi()))
                else:
                    self._emit(f"v_add_u32 v[{v.v_in_os()}], v[{v.v_in_os()}], s[{s.s_in_move_step()}]")
            return self._get_deferred()

        def move_slice_window_a():
            with self._deferred_context():
                self._emit(f"v_add_u32 v[{v.v_out_os()}], v[{v.v_out_os()}], s[{s.s_out_move_step()}]")
            return self._get_deferred()

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
            ctrl_xdlops_mapping               = get_ctrl_xdlops_mapping_from_wave_tile_fp32(self.tunable.gemm_m_per_block, self.tunable.gemm_n_per_block,self.tunable.wave_tile_m, self.tunable.wave_tile_n, self.tunable.wave_tile_k,
                                                                        self.tunable.wave_repeat_m, self.tunable.wave_repeat_n,
                                                                        self.tunable.wave_step_m, self.tunable.wave_step_n, self.tunable.block_size // AMDGPU_WAVE_SIZE)
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
            self._emit(self.coalescing_store(v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_wei(), v.v_wei_os(), None,
                    None, s.s_wei_stride_k(), s.s_tmp(), None))
        else:
            a = self.agpr
            self._emit(self.coalescing_store(a.a_c(), v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_wei(), v.v_wei_os(), None,
                    None, s.s_wei_stride_k(), s.s_tmp(), v.v_wei_c_flag(), s.s_k(), v.v_cur_k(), s.s_block_gtc_ik(), v.v_co_sub_m_index(), v.v_tmp()))

        if IGEMM_WRW_GTC_DEBUG == 1:
            self._emit_empty_line()
            self._emit(f"s_branch {self.label_out}")
            self._emit("; debug code to cpy vgpr to host")
            if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
                self._emit(f"L_debug_{self.label_out}_0:")
            else: 
                self._emit(f"L_debug_{self.label_out}_1:")
            self._emit("s_nop 256")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_waitcnt vmcnt(0)")
            self._emit("s_barrier")
            self._emit(f"s_cmp_lg_u32 s[{s.s_dbg(2)}], 0")
            if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
                self._emit(f"s_cbranch_scc1 L_program_end_{self.label_out}_0")
            else: 
                self._emit(f"s_cbranch_scc1 L_program_end_{self.label_out}_1")
            self._emit("v_mov_b32 v[v_tmp], s[s_in_offset]")
            self._emit(f"s_mov_b32 s[{s.s_tmp()}], 0")
            self._emit(f"s_mov_b32 s[{s.s_p_wei()}], s[{s.s_dbg()}]")
            self._emit(f"s_mov_b32 s[{s.s_p_wei(1)}], s[{s.s_dbg(1)}]")
            self._emit_empty_line()

            self._emit(f"buffer_store_dword v[v_in_os], v[{v.v_dbg()}], s[{s.s_p_wei((0,3))}], s[{s.s_tmp()}] offen")

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
