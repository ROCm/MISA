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
from .thread_mapping import *
from .xdlops_mapping import *
from .coalescing_store import *
from .mfma_main_loop import *

IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1 = 0
IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0 = 1
IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B = 4
IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N1B_N0 = 5


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
        self.global_load_out = self.global_load_out_t(mc, self)
        self.global_load_wei = self.global_load_wei_t(mc, self)
        self.shared_store_out = self.shared_store_out_t(mc, self)
        self.shared_store_wei = self.shared_store_wei_t(mc, self)

        wei_thread_copy_index, in_thread_copy_index = self.get_thread_copy_index()
        self.in_thread_copy_ndim = len(in_thread_copy_index)
        self.wei_thread_copy_ndim = len(wei_thread_copy_index)
        assert self.in_thread_copy_ndim in (0, 1, 2)
        assert self.wei_thread_copy_ndim in (0, 1, 2)







        
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




    class kernel_karg_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
            self.k_p_in      = sym_t('k_p_in'         ,0)
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
            self.k_end        = sym_t('k_end'           ,84)

        def get_count(self):
            return self.k_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('k_'):
                    self._emit(v.declare())

    class kernel_sgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            sseq                          = gpr_sequencer_t()
            self.outer                    = outer
            self.s_ka                     = sym_t('s_ka'   , sseq(2))
            self.s_bx                     = sym_t('s_bx'   , sseq(2))
            self.s_p_in                  = sym_t('s_p_in'   , sseq(4))
            self.s_p_wei                  = sym_t('s_p_wei'   , sseq(4))
            self.s_p_out                  = sym_t('s_p_out'   , sseq(4))
            self.s_hi                     = sym_t('s_hi'   , sseq(1))
            self.s_wi                     = sym_t('s_wi'   , sseq(1))
            self.s_n                      = sym_t('s_n'   , sseq(1))
            self.s_k                      = sym_t('s_k'   , sseq(1))
            self.s_c                      = sym_t('s_c'   , sseq(1))
            if outer.tunable.nxe != 0:
                self.s_ho                 = sym_t('s_ho'   , sseq(1))
                self.s_wo                 = sym_t('s_wo'   , sseq(1))
                self.s_stride_h           = sym_t('s_stride_h'   , sseq(1))
                self.s_stride_w           = sym_t('s_stride_w'   , sseq(1))
                self.s_dilation_h         = sym_t('s_dilation_h'   , sseq(1))
                self.s_dilation_w         = sym_t('s_dilation_w'   , sseq(1))
                self.s_pad_h              = sym_t('s_pad_h'   , sseq(1))
                self.s_pad_w              = sym_t('s_pad_w', sseq(1))
                self.s_y                  = sym_t('s_y', sseq(1))
                self.s_x                  = sym_t('s_x', sseq(1))

            # stride for wei
            if self.tunable.nxe != 0:
                self.s_wei_stride_c       = sym_t('s_wei_stride_c', sseq(1))
                self.s_wei_stride_k       = sym_t('s_wei_stride_k', sseq(1))
            if ta_c0 != 1:
                self.s_wei_stride_c0      = sym_t('s_wei_stride_c0', sseq(1))
            if ta_k0 != 1:
                self.s_wei_stride_k0      = sym_t('s_wei_stride_k0', sseq(1))

            # stride for in
            if self.tunable.nxe != 0:
                self.s_in_stride_c       = sym_t('s_in_stride_c', sseq(1))
            else:
                self.s_stride_hw          = sym_t('s_stride_hw', sseq(1))
            self.s_in_stride_n           = sym_t('s_in_stride_n', sseq(1))
            if tb_c0 != 1:
                self.s_in_stride_c0      = sym_t('s_in_stride_c0', sseq(1))
            if tb_n0 != 1:
                self.s_in_stride_n0      = sym_t('s_in_stride_n0', sseq(1))

            # stride for out
            if self.tunable.nxe != 0:
                self.s_out_stride_k       = sym_t('s_out_stride_k', sseq(1))
            self.s_out_stride_n           = sym_t('s_out_stride_n', sseq(1))
            if self.tunable.gemm_n_unmerge_cluster
                self.s_out_stride_n0      = sym_t('s_out_stride_n0', sseq(1))

            self.s_block_gtc_ik            = sym_t("s_block_gtc_ik"           ,sseq(1))
            self.s_block_gtc_in0           = sym_t("s_block_gtc_in0"          ,sseq(1))
            self.s_block_gtc_in1b          = sym_t("s_block_gtc_in1b"         ,sseq(1))


            '''
            if self.tunable.is_1x1():
                self._emit('.set s_in_stride,           {}'.format(s_seq(1)))
            else:
                self._emit('.set s_in_stride_c,         {}'.format(s_seq(1)))
            self._emit('.set s_in_stride_n2,        {}'.format(s_seq(1)))
            self._emit('.set s_in_stride_n1,        {}'.format(s_seq(1)))
            if not(self.tunable.is_1x1()):
                self._emit('.set s_in_ic,               {}'.format(s_seq(1)))
                self._emit('.set s_in_iy,               {}'.format(s_seq(1)))
                self._emit('.set s_in_ix,               {}'.format(s_seq(1)))

            if self.tunable.is_1x1():
                self._emit('.set s_wei_stride,          {}'.format(s_seq(1)))
                self._emit('.set s_wei_stride_k,        {}'.format(s_seq(1)))
            else:
                self._emit('.set s_wei_stride,          {}'.format(s_seq(1)))
                self._emit('.set s_wei_stride_c,        {}'.format(s_seq(1)))
                self._emit('.set s_wei_stride_k,        {}'.format(s_seq(1)))

            self._emit('.set s_out_stride_k0,       {}'.format(s_seq(1)))
            self._emit('.set s_out_stride_k1,       {}'.format(s_seq(1)))
            self._emit('.set s_out_stride_n1,       {}'.format(s_seq(1)))
            self._emit('.set s_out_stride_n2,       {}'.format(s_seq(1)))
            self._emit('.set s_kitr,                0')
            self._emit('.set s_tmp,                 {}'.format(s_seq(4, 4)))
            self._emit('.set s_end,                 {}'.format(s_seq(0)))
            '''
        def get_count(self):
            return self.s_end.value

        def emit(self):
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
                                        outer.tunable.num_vgpr_global_load_a + outer.tunable.num_vgpr_global_load_b + \
                                        8       # from v_sst_a_os to v_wei_os
                v_c_coalescing_num   = outer.tunable.num_agpr_accumulate_c // outer.coalescing_store_groups
                v_c_needed           = (v_c_coalescing_num - v_c_resuable_num) if (v_c_coalescing_num - v_c_resuable_num) > 0 else 0

                v_c_needed           = v_c_needed if v_c_needed > 2 else 2  # let at least 2
                self.v_c             = sym_t("v_c"            ,vseq(v_c_needed), f"coalescing:{v_c_coalescing_num}, needed:{v_c_needed}, resuable:{v_c_resuable_num}")

            self.v_a                 = sym_t("v_a"            ,vseq(outer.tunable.num_vgpr_accumulate_a))
            self.v_b                 = sym_t("v_b"            ,vseq(outer.tunable.num_vgpr_accumulate_b))
            self.v_gld_a             = sym_t("v_gld_a"        ,vseq(outer.tunable.num_vgpr_global_load_a))
            self.v_gld_b             = sym_t("v_gld_b"        ,vseq(outer.tunable.num_vgpr_global_load_b))
            self.v_sst_a_os          = sym_t("v_sst_a_os"     ,vseq(1))
            self.v_sst_b_os          = sym_t("v_sst_b_os"     ,vseq(1))
            self.v_sld_a_os          = sym_t("v_sld_a_os"     ,vseq(1))
            self.v_sld_b_os          = sym_t("v_sld_b_os"     ,vseq(1))
            self.v_in_os             = sym_t("v_in_os"        ,vseq(1))
            self.v_wei_os            = sym_t("v_wei_os"       ,vseq(1))

            self.v_gtc_ta_ik1        = sym_t("v_gtc_ta_ik1"   ,vseq(1))
            self.v_gtc_ta_ik0        = sym_t("v_gtc_ta_ik0"   ,vseq(1))
            self.v_gtc_ta_ic1e       = sym_t("v_gtc_ta_ic1e"  ,vseq(1))
            self.v_gtc_ta_ic0        = sym_t("v_gtc_ta_ic0"   ,vseq(1))

            self.v_gtc_tb_in1b       = sym_t("v_gtc_tb_in1b"  ,vseq(1))
            self.v_gtc_tb_in0        = sym_t("v_gtc_tb_in0"   ,vseq(1))
            self.v_gtc_tb_ic1e       = sym_t("v_gtc_tb_ic1e"  ,vseq(1))
            self.v_gtc_tb_ic0        = sym_t("v_gtc_tb_ic0"   ,vseq(1))
            self.v_gtc_tb_in1        = sym_t("v_gtc_tb_in1"   ,vseq(1))
            self.v_gtc_tb_ib         = sym_t("v_gtc_tb_ib"    ,vseq(1))
            self.v_gtc_tb_ic1        = sym_t("v_gtc_tb_ic1"   ,vseq(1))
            if outer.tunable.nxe != 0:
                #self.v_gtc_tb_iy     = sym_t("v_gtc_tb_iy"    ,vseq(1))
                #self.v_gtc_tb_ix     = sym_t("v_gtc_tb_ix"    ,vseq(1))

            self.v_out_os            = sym_t("v_out_os"       ,vseq(1))

            self.v_in_iho           = sym_t("v_in_iho"      ,vseq(1))
            self.v_in_iwo           = sym_t("v_in_iwo"      ,vseq(1))
            self.v_in_ihi           = sym_t("v_in_ihi"      ,vseq(1))
            self.v_in_iwi           = sym_t("v_in_iwi"      ,vseq(1))
            self.v_in_iy            = sym_t("v_in_iy"       ,vseq(1))
            self.v_in_ix            = sym_t("v_in_ix"       ,vseq(1))

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
            pass
        else:
            assert ta_c0 == 1, "wei not using c0. for wei treat c1e as c*e, single dimension"
            assert tb_c0 == 1, "input can't use c0"

        return ta_c0, ta_c1e, ta_k0, ta_k1, tb_c0, tb_c1e, tb_n0, tb_n1b    # ta K M, tb K N

    def get_cluster_lengths(self):
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4

        ca_c0, ca_c1e, ca_k0, ca_k1  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        cb_c0, cb_c1e, cb_n0, cb_n1b = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        assert ca_c0 == 1, "wei not using c0. for wei treat c1e as c*e"

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

        if gemm_m_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K0_K1:
            # na_c0, na_c1e, na_k0, na_k1
            wei_stride_list = [na_c1e*na_k0*na_k1, na_k0*na_k1, na_k1, 1]
        else:
            # na_c0, na_c1e, na_k1, na_k0
            wei_stride_list = [na_c1e*na_k0*na_k1, na_k0*na_k1, 1, na_k0]

        if gemm_n_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B:
            # nb_c0, nb_c1e, nb_n0, nb_n1b
            in_stride_list = [nb_c1e*nb_n0*nb_n1b, nb_n0*nb_n1b, nb_n1b, 1]
        else:
            # nb_c0, nb_c1e, nb_n1b, nb_n0
            in_stride_list = [nb_c1e*nb_n0*nb_n1b, nb_n0*nb_n1b, 1, nb_n0]

        wei_sst_ctrl = ctrl_2d_shared_store_t()
        wei_sst_ctrl.src_order = 1                  # for weight, always reverse order in register.
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
        inline = True if self.tunable.fma_interleave else False 
        return macro_igemm_2d_shared_store_t(self.mc, in_sst_ctrl, inline), macro_igemm_2d_shared_store_t(self.mc, wei_sst_ctrl, inline)

    # computation macro
    def get_macro_in_update_hw(self):
        inline = True if self.tunable.fma_interleave else False
        return macro_igemm_fwd_gtc_in_update_hw_t(self.mc, inline)

    def get_macro_in_update_os(self):
        inline = True if self.tunable.fma_interleave else False
        return macro_igemm_fwd_gtc_in_update_os_t(self.mc, inline)


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
        int __pack0;
        '''
        kas = []
        # name: {}, .size: {}, .offset: {}, .value_kind: {}, .value_type
        kas.append(amdgpu_kernel_arg_t('p_in'      , 8,  0, 'global_buffer','f32',address_space='global',is_const='true'))
        kas.append(amdgpu_kernel_arg_t('p_wei'      , 8,  8, 'global_buffer','f32',address_space='global',is_const='true'))
        kas.append(amdgpu_kernel_arg_t('p_out'      , 8, 16, 'global_buffer','f32',address_space='global',is_const='false'))
        kas.append(amdgpu_kernel_arg_t('hi'         , 4, 24, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('wi'         , 4, 28, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('n'          , 4, 32, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('k'          , 4, 36, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('c'          , 4, 40, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('ho'         , 4, 44, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('wo'         , 4, 48, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('stride_h'   , 4, 52, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('stride_w'   , 4, 56, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_h' , 4, 60, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_w' , 4, 64, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('pad_h'      , 4, 68, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('pad_w'      , 4, 72, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('y'          , 4, 76, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('x'          , 4, 80, 'by_value','i32'))
        kas.append(amdgpu_kernel_arg_t('__pack0'    , 4, 84, 'by_value','i32'))
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
        unmerge_sub_ta_c  = self.tunable.unmerge_sub_c
        unmerge_sub_ta_c1 = unmerge_sub_ta_c // na_c0
        unmerge_sub_tb_c  = self.tunable.unmerge_sub_c
        unmerge_sub_tb_c1 = unmerge_sub_tb_c // nb_c0


        data_byte = amdgpu_precision_data_byte(self.tunable.precision)


        tc_index_dispatcher = igemm_thread_cluster_index_dispatcher_t(self.mc)
        tc_index_accumulator = igemm_thread_cluster_index_accumulator_t(self.mc)

        m_int_div_rem_vv = macro_int_div_rem_vv_t(self.mc)
        m_int_div_rem_vs = macro_int_div_rem_vs_t(self.mc)
        m_int_div_rem_ss = macro_int_div_rem_ss_t(self.mc)
        gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()
        s_dummy = sym_t("s_dummy")

        m_in_update_hw = self.get_macro_in_update_hw()
        m_in_update_os = self.get_macro_in_update_os()

        # start emit
        #self._emit(f"; unmerge_sub_k:{unmerge_sub_k}, unmerge_sub_k1:{unmerge_sub_k1}, unmerge_sub_n:{unmerge_sub_n}, unmerge_sub_n1:{unmerge_sub_n1}")
        self._emit(f"; gemm_m_unmerge_cluster:{gemm_m_unmerge_cluster}, gemm_n_unmerge_cluster:{gemm_n_unmerge_cluster}, gemm_k_unmerge_cluster:{gemm_k_unmerge_cluster}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_in((0,1))}],      s[{s.s_ka((0, 1))}],    0+{k.k_p_in()}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_wei((0,1))}],      s[{s.s_ka((0, 1))}],    0+{k.k_p_wei()}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_out((0,1))}],      s[{s.s_ka((0, 1))}],    0+{k.k_p_out()}")
        if self.tunable.nxe != 0:
            self._emit(f"s_load_dwordx16 s[{s.s_hi((0, 15))}],        s[{s.s_ka((0, 1))}],    0+{k.k_hi()}")
        else:
            self._emit(f"s_load_dwordx4 s[{s.s_hi((0, 3))}],        s[{s.s_ka((0, 1))}],    0+{k.k_hi()}")
            self._emit(f"s_load_dword s[{s.s_c()}],        s[{s.s_ka((0, 1))}],    0+{k.k_c()}")


        self._emit(f"; wei(c0, c1e, k0, k1) thread_lengths: {ta_c0}x{ta_c1e}x{ta_k0}x{ta_k1}, cluster_lengths:{ca_c0}x{ca_c1e}x{ca_k0}x{ca_k1}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        self._emit(tc_index_dispatcher(v.v_gtc_ta_ik1(),    v.v_tmp(),  ca_k1,  ta_k1))
        self._emit(tc_index_dispatcher(v.v_gtc_ta_ik0(),    v.v_tmp(),  ca_k0,  ta_k0))
        self._emit(tc_index_dispatcher(v.v_gtc_ta_ic1e(),   v.v_tmp(),  ca_c1e, ta_c1e))
        self._emit(tc_index_dispatcher(v.v_gtc_ta_ic0(),    v.v_tmp(),  ca_c0,  ta_c0,  True))
        self._emit_empty_line()

        self._emit(f"; in(c0, c1e, n0, n1b), thread_lengths: {tb_c0}x{tb_c1e}x{tb_n0}x{tb_n1b}, cluster_lengths:{cb_c0}x{cb_c1e}x{cb_n0}x{cb_n1b}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        self._emit(tc_index_dispatcher(v.v_gtc_tb_in1b(),   v.v_tmp(),  ca_n1b, ta_n1b))
        self._emit(tc_index_dispatcher(v.v_gtc_tb_in0(),    v.v_tmp(),  ca_n0,  ta_n0))
        self._emit(tc_index_dispatcher(v.v_gtc_tb_ic1e(),   v.v_tmp(),  ca_c1e, ta_c1e))
        self._emit(tc_index_dispatcher(v.v_gtc_tb_ic0(),    v.v_tmp(),  ca_c0,  ta_c0,  True))
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

        if self.tunable.nxe != 0:
            # stride for wei
            self._emit(f"s_mul_i32 s[{s.s_wei_stride_c()}], s[{s.s_y()}], s[{s.s_x()}]")
            self._emit(f"s_mul_i32 s[{s.s_wei_stride_k()}], s[{s.s_c()}], s[{s.s_wei_stride_c()}]")
            if ta_c0 != 1:
                self._emit(f"s_lshl_b32 s[{s.s_wei_stride_c0()}], s[{s.s_wei_stride_c()}], {utility_log2(unmerge_sub_ta_c1)}")
            if ta_k0 != 1:
                self._emit(f"s_lshl_b32 s[{s.s_wei_stride_k0()}], s[{s.s_wei_stride_k()}], {utility_log2(na_k1)}")

            # stride for in
            self._emit(f"s_mul_i32 s[{s.s_in_stride_c()}], s[{s.s_hi()}], s[{s.s_wi()}]")
            self._emit(f"s_mul_i32 s[{s.s_in_stride_n()}], s[{s.s_c()}], s[{s.s_in_stride_c()}]")
            if tb_c0 != 1:
                self._emit(f"s_lshl_b32 s[{s.s_in_stride_c0}], s[{s.s_in_stride_c()}], {utility_log2(unmerge_sub_tb_c1)}")
            if tb_n0 != 1:
                if gemm_n_unmerge_cluster == 0:
                    self._emit(f"s_lshl_b32 s[{s.s_in_stride_n0}], s[{s.s_in_stride_n()}], {utility_log2(unmerge_sub_n1)} ")
                else:
                    self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {utility_log2(nb_n0)}")
                    self._emit(f"s_mul_i32 s[{s.s_in_stride_n0}], s[{s.s_in_stride_n()}], s[{s.s_tmp()}],")

            # stride for out
            self._emit(f"s_mul_i32 s[{s.s_out_stride_k()}], s[{s.s_ho()}], s[{s.s_wo()}]")
            self._emit(f"s_mul_i32 s[{s.s_out_stride_n()}], s[{s.s_k()}], s[{s.s_out_stride_k()}]")
            if gemm_n_unmerge_cluster == 1:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {igemm_log2(nb_n0)}")
                self._emit(f"s_mul_i32 s[{s.s_out_stride_n0()}], s[{s.s_out_stride_n()}], s[{s.s_tmp()}]")

        else:
            # stride for wei
            if ta_c0 != 1:
                self._emit(f"s_mov_b32 s[{s.s_wei_stride_c0()}], {unmerge_sub_ta_c1}")
            if ta_k0 != 1:
                self._emit(f"s_lshl_b32 s[{s.s_wei_stride_k0()}], s[{s.s_c()}], {utility_log2(na_k1)}")

            self._emit(f"s_mul_i32 s[{s.s_stride_hw()}], s[{s.s_hi()}], s[{s.s_wi()}]")             # both in/out
            # stride for in
            self._emit(f"s_mul_i32 s[{s.s_in_stride_n()}], s[{s.s_c()}], s[{s.s_stride_hw()}]")
            if tb_c0 != 1:
                self._emit(f"s_lshl_b32 s[{s.s_in_stride_c0}], s[{s.s_in_stride_c()}], {utility_log2(unmerge_sub_tb_c1)}")
            if tb_n0 != 1:
                if gemm_n_unmerge_cluster == 0:
                    self._emit(f"s_lshl_b32 s[{s.s_in_stride_n0}], s[{s.s_in_stride_n()}], {utility_log2(unmerge_sub_n1)} ")
                else:
                    self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {utility_log2(nb_n0)}")
                    self._emit(f"s_mul_i32 s[{s.s_in_stride_n0}], s[{s.s_in_stride_n()}], s[{s.s_tmp()}]")

            # stride for out
            self._emit(f"s_mul_i32 s[{s.s_out_stride_n()}], s[{s.s_k()}], s[{s.s_stride_hw()}]")
            if gemm_n_unmerge_cluster == 1:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {igemm_log2(nb_n0)}")
                self._emit(f"s_mul_i32 s[{s.s_out_stride_n0()}], s[{s.s_out_stride_n()}], s[{s.s_tmp()}]")

        self._emit_empty_line()
        self._emit(f"; gemm_m_per_block:{self.tunable.gemm_m_per_block}, gemm_n_per_block:{self.tunable.gemm_n_per_block}")
        if self.tunable.nxe != 0:
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_out_stride_k()}], s[{s.s_n()}]")
        else:
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_stride_hw()}], s[{s.s_n()}]")
        self._emit(f"s_lshr_b32 s[0], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_n_per_block)}")
        self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_tmp(5), s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
        self._emit(f"; s_tmp+4:block_gtc_in, s_tmp+5:block_gtc_im")

        if gemm_m_unmerge_cluster == 0:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ik()}], s[{s.s_tmp(5)}], {igemm_log2(self.tunable.gemm_m_per_block)}")
        else:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ik()}], s[{s.s_tmp(5)}], {igemm_log2(self.tunable.gemm_m_per_block // n_k0)}")
            #self._emit(f"s_mov_b32 s[{s.s_block_gtc_ic0()}], 0")

        if gemm_n_unmerge_cluster == 0:
            if self.tunable.nxe != 0:
                if unmerge_sub_n1 == 1:
                    self._emit(f"s_lshr_b32 s[0], s[{s.s_out_stride_k()}], {igemm_log2(nb_n1b)} ; total number of n1b")
                else:
                    if unmerge_sub_n1 == nb_n1b:
                        self._emit(f"s_mov_b32 s[0], s[{s.s_out_stride_k()}] ; total number of n1b")
                    else:
                        self._emit(f"s_lshr_b32 s[0], s[{s.s_out_stride_k()}], {igemm_log2(nb_n1b // unmerge_sub_n1)}  ; total number of n1b")
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
                self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_out_stride_k()}], s[{s.s_tmp()}]")
                self._emit(f"s_lshr_b32 s[0], s[{s.s_tmp(1)}], {igemm_log2(nb_n1b)}")
            else:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {igemm_log2(nb_n0)}")
                self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_stride_hw()}], s[{s.s_tmp()}]")
                self._emit(f"s_lshr_b32 s[0], s[{s.s_tmp(1)}], {igemm_log2(nb_n1b)}")

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
                assert False, "this is not wished and may introduce wrong machine code"
            else:
                self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_tb_ic1(), v.v_gtc_tb_ic1e(), s.s_wei_stride_c(), v.v_tmp(), s.s_tmp()))
                self._emit(m_int_div_rem_vs(v.v_in_ix(), v.v_in_iy(), v.v_tmp(4), s.s_x(), v.v_tmp(), s.s_tmp()))
        else:
            self._emit(f"v_mov_b32 v[{v.v_gtc_tb_ic1()}], v[{v.v_gtc_tb_ic1e()}]")


        self._emit(f"; in n1b transform")
        if cb_n1b == 1:
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_in1b()}]")
        else:
            self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_in1b()}], v[{v.v_gtc_tb_in1b()}]")
        if self.tunable.nxe != 0:
            self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_tb_in1(), v.v_tmp(5), s.s_out_stride_k(), v.v_tmp(), s.s_tmp()))
            self._emit(m_int_div_rem_vs(v.v_in_iwo(), v.v_in_iho(), v.v_tmp(4), s.s_wo(), v.v_tmp(), s.s_tmp()))
            self._emit(f"v_mul_lo_u32 v[{v.v_in_iho()}], s[{s.s_stride_h()}],v[{v.v_in_iho()}]")
            self._emit(f"v_sub_i32 v[{v.v_in_iho()}], v[{v.v_in_iho()}], s[{s.s_pad_h()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_in_iwo()}], s[{s.s_stride_w()}],v[{v.v_in_iwo()}]")
            self._emit(f"v_sub_i32 v[{v.v_in_iwo()}], v[{v.v_in_iwo()}], s[{s.s_pad_w()}]")
            self._emit(m_in_update_hw(v.v_in_ihi(), v.v_in_iwi(), v.v_in_iho(), v.v_in_iwo(), v.v_in_iy(), v.v_in_ix(), s.s_dilation_h(), s.s_dilation_w()))
            self._emit_empty_line()
        else:
            self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_tb_in1(), v.v_tmp(5), s.s_stride_hw(), v.v_tmp(), s.s_tmp()))
            self._emit(m_int_div_rem_vs(v.v_in_iwi(), v.v_in_ihi(),  v.v_tmp(4), s.s_wi(), v.v_tmp(), s.s_tmp()))

        self._emit(f"; calculate in offset")
        if gemm_n_unmerge_cluster == 0:
            self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], s[{s.s_block_gtc_in0()}], {igemm_log2(unmerge_sub_n1 * data_byte)}")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_in_stride_n()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_in_stride_n()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
            self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], s[{s.s_tmp(1)}]")
        else:
            pass
        self._emit_empty_line()

        self._emit(tc_index_accumulator(v.v_tmp(), v.v_gtc_tb_ic0(), v.v_gtc_tb_ic1(), c_c0, c_c1e, 0, unmerge_sub_tb_c1))
        if self.tunable.nxe != 0:
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_in_stride_c()}], v[{v.v_tmp()}]")
        else:
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_stride_hw()}], v[{v.v_tmp()}]")
        if gemm_n_unmerge_cluster == 0:
            self._emit(tc_index_accumulator(v.v_tmp(1), v.v_gtc_tb_in0(), v.v_gtc_tb_in1(), cb_n0, cb_n1b, 0, unmerge_sub_n1))
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_in_stride_n()}], v[{v.v_tmp(1)}]")
        else:
            # no in0
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_in_stride_n()}], v[{v.v_gtc_tb_in1()}]")

        if self.tunable.nxe != 0:
            self._emit(f"v_add_lshl_u32 v[{v.v_in_os_base()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
            self._emit(m_in_update_os(v.v_in_os(), v.v_in_os_base(), v.v_in_iho(), v.v_in_iwo(), s.s_wi(), v.v_tmp()))
            self._emit(m_set_flag_hw(v.v_out_flag(), v.v_out_iho(), v.v_out_iwo(), s.s_ho(), s.s_wo()))
        else:
            self._emit(f"v_add_lshl_u32 v[{v.v_tmp(4)}], v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
            self._emit(m_in_update_os(v.v_in_os(), v.v_tmp(4), v.v_in_iho(), v.v_in_iwo(), s.s_wi(), v.v_tmp()))
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
        self._emit(self.global_load_in()))
        self._emit_empty_line()



        self._emit(f"; calculate wei offset")
        if self.tunable.nxe != 0:
            # one important thing is we let wei=k*c*y*x, c*y*x -> e, treat e as a single dimension
            self._emit(tc_index_accumulator(v.v_tmp(), v.v_gtc_ik0(), v.v_gtc_ta_ik1(), ca_k0, ca_k1, na_k0, na_k1))
            self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_ik()}], v[{v.v_tmp()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_k()}], v[{v.v_tmp(5)}]")
            self._emit(tc_index_accumulator(v.v_tmp(1), v.v_gtc_ik0(), v.v_gtc_ik1(), c_k0, c_k1e, 0, unmerge_sub_k1))
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_wei_stride_k()}], v[{v.v_tmp(1)}]")
            self._emit(f"v_add_lshl_u32 v[{v.v_wei_os_base()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
            self._emit(m_wei_update_os(v.v_wei_os(), v.v_wei_os_base(), v.v_wei_iy(), v.v_wei_ix(), s.s_x(), v.v_tmp()))
        else:
            self._emit(tc_index_accumulator(v.v_tmp(), v.v_gtc_ic0(), v.v_gtc_ic1(), c_c0, c_c1, n_c0, n_c1))
            self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_block_gtc_ic()}], v[{v.v_tmp()}] ; c index")
            self._emit(tc_index_accumulator(v.v_tmp(1), v.v_gtc_ik0(), v.v_gtc_ik1(), c_k0, c_k1e, 0, unmerge_sub_k1))
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_c()}], v[{v.v_tmp(1)}]")
            self._emit(f"v_add_lshl_u32 v[{v.v_wei_os()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
        self._emit_empty_line()
