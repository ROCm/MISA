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


IGEMM_FWD_GTC_NCHW_PACK_IN_FLAG = 0
# IGEMM_FWD_GTC_NCHW_P_INTERLEAVE_GLD = False     # p tensor interleave
IGEMM_FWD_GTC_NCHW_ACCVGPR_UNIFIED = True   # used in gfx90a
IGEMM_FWD_GTC_NCHWC_DEBUG = 0

def _find_non_1_index_in_list(list_object):
    result_list = list()
    for idx, item in enumerate(list_object):
        assert type(item) is int
        if item != 1:
            result_list.append(idx)
    return result_list

class igemm_fwd_gtc_nchwc_t(mc_base_t):
    '''
                       tensor a (wei)               tensor b (inp)
    thread_lengths  :  1, 1,      1, ta_k-vec-c     1, 1,     tb_nb0, tb_vec-c,
    cluster_lengths :  1, ca_ce,  1, ca_k1          1, cb_ce, 1,      cb_nb1,    
    for a/b tensor, always load vec-c dimension first.
    '''
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
        self.global_load_in = self.global_load_in_t(mc, self)
        self.global_load_wei = self.global_load_wei_t(mc, self)
        self.shared_store_in = self.shared_store_in_t(mc, self)
        self.shared_store_wei = self.shared_store_wei_t(mc, self)

        in_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()
        self.in_thread_copy_ndim = len(in_thread_copy_index)
        self.wei_thread_copy_ndim = len(wei_thread_copy_index)
        assert self.in_thread_copy_ndim in (0, 1, 2)
        assert self.wei_thread_copy_ndim in (0, 1, 2)
        
        self.div_v_const_func = div_u32_vi_t(mc)
        self.div_rem_v_const_func = div_rem_u32_vi_t(mc)

        self.coalescing_store_groups = self.tunable.coalescing_store_groups#igemm_next_pow2(self.tunable.coalescing_store_groups)
        if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            # TODO: add non dlops op
            assert (self.tunable.lanegroup_tile_m * self.tunable.lanegroup_repeat_m) % self.coalescing_store_groups == 0, \
                f"coalescing store groups should be divided by thread m {self.tunable.lanegroup_tile_m}x{self.tunable.lanegroup_repeat_m}"

            ctrl_dotx_mapping = get_ctrl_dotx_mapping_from_wave_tile(self.tunable.gemm_m_per_block, self.tunable.gemm_n_per_block,
                                                                     self.tunable.lanegroup_tile_m, self.tunable.lanegroup_tile_n,
                                                                     self.tunable.lanegroup_wave_m, self.tunable.lanegroup_wave_n,
                                                                     self.tunable.block_size // (self.tunable.lanegroup_wave_m * self.tunable.lanegroup_wave_n * LANEGROUP_SIZE),
                                                                     self.tunable.lanegroup_repeat_m, self.tunable.lanegroup_repeat_n,
                                                                     self.tunable.precision)
            self.dotx_mapping = igemm_dotx_mapping_t(self.mc, ctrl_dotx_mapping)

            ctrl_coalescing_store = ctrl_coalescing_store_dotx_t()
            ctrl_coalescing_store.cdm = ctrl_dotx_mapping
            ctrl_coalescing_store.coalescing_groups = self.coalescing_store_groups
            ctrl_coalescing_store.precision = self.tunable.precision

            ctrl_coalescing_store.vector_store_m = self.tunable.vector_c                      # TODO: some cases this can be set to other value
            ctrl_coalescing_store.vector_fold_m = self.tunable.vector_c
            ctrl_coalescing_store.block_size = self.tunable.block_size
            
            ctrl_coalescing_store.div_v_const_func = self.div_v_const_func
            ctrl_coalescing_store.div_rem_v_const_func = self.div_rem_v_const_func

            self.coalescing_store = igemm_coalescing_store_dotx_t(mc, ctrl_coalescing_store)

        else:
            assert False, "xdlops is not needed for now"

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

    class macro_set_flag_nhw(macro_base_t):
        def __init__(self, mc, inline = False):
            macro_base_t.__init__(self, mc, inline)
            self.declare_arg("v_flag")
            self.declare_arg("v_flag_n")
            self.declare_arg("v_ih")
            self.declare_arg("v_iw")
            self.declare_arg("s_h")
            self.declare_arg("s_w")
        def name(self):
            return '.v_fwd_gtc_nchw_set_flag_nhw'

        def expr(self):
            self._emit(f"v_cmp_gt_u32  s[{self.s_h()}], v[{self.v_ih()}]")
            self._emit(f"v_cndmask_b32 v[{self.v_flag()}], 0, v[{self.v_flag_n()}]")
            self._emit(f"v_cmp_gt_u32  s[{self.s_w()}], v[{self.v_iw()}]")
            self._emit(f"v_cndmask_b32 v[{self.v_flag()}], 0, v[{self.v_flag()}]")

    class macro_move_slice_window_block_wise_1x1_t(macro_base_t):
        def __init__(self, mc, tunable, inline, **options):
            macro_base_t.__init__(self, mc, True)
            self.tunable = tunable
            if tunable.tensor_a_pass_through:
                self.declare_arg("s_in_base")       # 64bit acc
            else:
                self.declare_arg("s_in_offset")     # use this as c itr, since other dimension of input is voffset
            self.declare_arg("v_wei_os")
            self.declare_arg("s_move_slice_k_stride_c")        
            self.declare_arg("s_move_slice_k_stride_gemm_k")    # this is gemm_k * data_byte
            self.options = options

        def name(self):
            return '.v_fwd_gtc_nchw_move_slice_window_block_wise_1x1_{self.tunable.tensor_a_pass_through}_{self.tunable.tensor_b_pass_through}'

        def expr(self):
            if self.tunable.tensor_a_pass_through:
                self._emit(f"s_add_u32 s[{self.s_in_base()}], s[{self.s_move_slice_k_stride_c()}], s[{self.s_in_base()}]")
                self._emit(f"s_addc_u32 s[{self.s_in_base(1)}], 0, s[{self.s_in_base(1)}]")
            else:
                self._emit(f"s_add_u32 s[{self.s_in_offset()}],  s[{self.s_move_slice_k_stride_c()}], s[{self.s_in_offset()}]")
            self._emit(f"v_add_nc_u32 v[{self.v_wei_os()}], s[{self.s_move_slice_k_stride_gemm_k()}], v[{self.v_wei_os()}]")
            self._emit_empty_line()

    class macro_move_slice_window_block_wise_t(macro_base_t):
        def __init__(self, mc, tunable, inline, **options):
            macro_base_t.__init__(self, mc, True)
            self.tunable = tunable
            self.options = options
            # ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
            # iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w

            self.declare_arg("v_gtc_iy_itr") 
            self.declare_arg("v_gtc_ix_itr")
            self.declare_arg("v_gtc_ic_itr")
            self.declare_arg("v_x_ovf_backtrace")
            self.declare_arg("v_x_ovf_acc_in_os")
            self.declare_arg("v_x_ovf_acc_y")
            self.declare_arg("v_y_inc")

            self.declare_arg("v_in_os")
            self.declare_arg("v_in_ihi_list")
            self.declare_arg("v_in_iwi_list")
            self.declare_arg("v_in_flag")
            self.declare_arg("v_in_flag_n")

            self.declare_arg("v_wei_os")

            self.declare_arg("s_x_dilation_w")
            self.declare_arg("s_y_dilation_h")

            self.declare_arg("s_move_slice_k_y")                # (gemm_k / (y*x)) % y  => higher 8 bit of s_y, < 256
            self.declare_arg("s_move_slice_k_x")                # (gemm_k /c ) % x      => higher 8 bit of s_x, < 256
            self.declare_arg("s_move_slice_k_c")                # gemm_k % c            => higher 8 bit of s_c, < 256
            self.declare_arg("s_move_slice_k_stride_gemm_k")    # this is gemm_k * data_byte

            #self.declare_arg("s_diff_in_os_acc_c")              # due to c increment, this is s_move_slice_k_c * data_byte
            #self.declare_arg("s_diff_in_os_ovf_c")              # due to c increment and overflow
            #                                                    #   (s_move_slice_k_c - s_c) * data_byte + s_dilation_w * in_stride_wi
            #self.declare_arg("s_diff_in_os_acc_x")              # due to x increment, this is s_move_slice_k_x * s_dilation_w * in_stride_wi
            #self.declare_arg("s_diff_in_os_ovf_x")              # due to x increment and overflow
            #                                                    #   (s_move_slice_k_x - s_x) * s_dilation_w * in_stride_wi + s_dilation_h * in_stride_hi
            #self.declare_arg("s_diff_in_os_acc_y")              # due to y increment, this is s_move_slice_k_y * s_dilation_h * in_stride_hi

            self.declare_arg("s_diff_in_os_acc_c_y_x")          # due to x, y, c increment,
                                                                #    s_move_slice_k_c * data_byte + s_move_slice_k_x * s_dilation_w * in_stride_wi + s_move_slice_k_y * s_dilation_h * in_stride_hi
            self.declare_arg("s_diff_in_os_ovf_y_acc_c")        # due to c increment and overflow
                                                                #    -s_c * data_byte + s_dilation_w * in_stride_wi
            self.declare_arg("s_diff_in_os_ovf_x_acc_y")        # due to x increment and overflow
                                                                #    -s_x * s_dilation_w * in_stride_wi + s_dilation_h * in_stride_hi

            self.declare_arg("s_diff_in_iwi_acc_x")             # due to x increment, iwi diff, s_move_slice_k_x * s_dilation_w
            self.declare_arg("s_dilation_w")                    # due to c ovf to x + 1, iwi will increment s_dilation_w
            self.declare_arg("s_diff_in_iwi_ovf_x")             # due to x overflow, will decrease s_x * s_dilation_w

            self.declare_arg("s_diff_in_ihi_acc_y")             # due to y increment, ihi diff, s_move_slice_k_y * s_dilation_h
            self.declare_arg("s_dilation_h")                    # due to x ovf to y + 1, ihi will increment s_dilation_h

            self.declare_arg("v_in_os_diff")                    # tmp buffer for calculate in os diff
            self.declare_arg("v_in_ihi_diff")                   # tmp buffer for calculate in ihi diff
            self.declare_arg("v_in_iwi_diff")                   # tmp buffer for calculate in iwi diff

            self.declare_arg("s_x")
            self.declare_arg("s_c")
            self.declare_arg("s_hi")
            self.declare_arg("s_wi")

        def name(self):
            return '.v_fwd_gtc_nchw_move_slice_window_me'

        def expr(self):
            assert "nb_per_thread" in self.options
            nb_per_thread = self.options["nb_per_thread"]
            assert 'm_set_flag_nhw' in self.options
            m_set_flag_nhw = self.options['m_set_flag_nhw']

            #self._emit(f"v_mov_b32 v[{self.v_in_iwi_diff()}], s[{self.s_diff_in_iwi_acc_x()}]")         # this is indeed in later stage ++x
            self._emit(f"v_mov_b32 v[{self.v_in_os_diff()}], s[{self.s_diff_in_os_acc_c_y_x()}]")
            self._emit(f"v_mov_b32 v[{self.v_in_ihi_diff()}], s[{self.s_diff_in_ihi_acc_y()}]")         # this is indeed in later stage ++y
            self._emit_empty_line()

            self._emit(f"v_add_co_u32_e64 v[{self.v_gtc_ix_itr()}], vcc_lo, s[{self.s_move_slice_k_x()}], v[{self.v_gtc_ix_itr()}]")
            self._emit(f"v_cndmask_b32 v[{self.v_x_ovf_backtrace()}], 0, s[{self.s_x_dilation_w()}]")
            self._emit(f"v_cndmask_b32 v[{self.v_x_ovf_acc_in_os()}], 0, s[{self.s_diff_in_os_ovf_x_acc_y()}]")
            self._emit(f"v_cndmask_b32 v[{self.v_x_ovf_acc_y()}], 0, s[{self.s_dilation_h()}]")
            self._emit(f"v_add_nc_u32 v[{self.v_gtc_ix_itr()}], v[{self.v_gtc_ix_itr()}], v[{self.v_x_ovf_backtrace()}]")
            for i in range(nb_per_thread):
                self._emit(f"v_add3_u32 v[{self.v_in_iwi_list(i)}], v[{self.v_x_ovf_backtrace()}], v[{self.v_in_iwi_list(i)}], s[{self.s_diff_in_iwi_acc_x()}]")
            self._emit(f"v_add_nc_u32 v[{self.v_in_os_diff()}], v[{self.v_x_ovf_acc_in_os()}], v[{self.v_in_os_diff()}]")
            self._emit_empty_line()

            self._emit(f"v_add_co_ci_u32 v[{self.v_gtc_iy_itr()}], v[{self.v_gtc_iy_itr()}], s[{self.s_move_slice_k_y()}]")
            self._emit(f"v_cndmask_b32 v[{self.v_x_ovf_backtrace()}], 0, s[{self.s_y_dilation_h()}]")
            self._emit(f"v_cndmask_b32 v[{self.v_x_ovf_acc_in_os()}], 0, s[{self.s_diff_in_os_ovf_y_acc_c()}]")
            self._emit(f"v_add_nc_u32 v[{self.v_gtc_iy_itr()}], v[{self.v_gtc_iy_itr()}], v[{self.v_x_ovf_backtrace()}]")
            self._emit(f"v_add3_u32 v[{self.v_y_inc()}], v[{self.v_in_ihi_diff()}], v[{self.v_x_ovf_backtrace()}], v[{self.v_x_ovf_acc_y()}]")
            for i in range(nb_per_thread):
                self._emit(f"v_add_nc_u32 v[{self.v_in_ihi_list(i)}], v[{self.v_y_inc()}], v[{self.v_in_ihi_list(i)}]")
            self._emit(f"v_add_nc_u32 v[{self.v_in_os_diff()}], v[{self.v_x_ovf_acc_in_os()}], v[{self.v_in_os_diff()}]")
            self._emit_empty_line()

            self._emit(f"v_add_co_ci_u32 v[{self.v_gtc_ic_itr()}], v[{self.v_gtc_ic_itr()}], s[{self.s_move_slice_k_c()}]")
            
            self._emit(f"v_add_nc_u32 v[{self.v_wei_os()}], s[{self.s_move_slice_k_stride_gemm_k()}], v[{self.v_wei_os()}]")
            for i in range(nb_per_thread):
                self._emit(f"v_add_nc_u32 v[{self.v_in_os(i)}], v[{self.v_in_os_diff()}], v[{self.v_in_os(i)}]")

            self._emit(f"v_cmp_gt_u32 s[{self.s_c()}], v[{self.v_gtc_ic_itr()}]")
            self._emit(f"v_cndmask_b32 v[{self.v_in_os_diff()}], 0, 1")

            for i in range(nb_per_thread):
                if IGEMM_FWD_GTC_NCHW_PACK_IN_FLAG:
                    assert False
                else:
                    self._emit(f"v_bfe_u32 v[{self.v_in_ihi_diff()}], v[{self.v_in_flag_n()}], {i}, 1   ; extract flag_n")
                    self._emit(f"v_and_b32 v[{self.v_in_ihi_diff()}], v[{self.v_in_os_diff()}], v[{self.v_in_ihi_diff()}]")
                    self._emit(m_set_flag_nhw(self.v_in_flag(i), self.v_in_ihi_diff(), self.v_in_ihi_list(i), self.v_in_iwi_list(i), self.s_hi(), self.s_wi()))

    class global_load_in_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_wei_2d_global_load, m_in_2d_global_load = self.outer.get_macro_global_load()
            return m_in_2d_global_load.get_issues()

        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr
            k = self.outer.karg
            tunable = self.outer.tunable

            m_wei_2d_global_load, m_in_2d_global_load = self.outer.get_macro_global_load()
            with self._deferred_context():
                self._emit(f"; load input, nxe:{self.outer.tunable.nxe}")
                self._emit(f".v_clear_nc {v.v_gld_b() if tunable.global_prefetch_a_num == 1 else v.v_gld_b_gpf()}, {self.outer.get_num_vgpr_global_load_b()}")
                self._emit(m_in_2d_global_load(v.v_gld_b() if tunable.global_prefetch_a_num == 1 else v.v_gld_b_gpf(),
                    s.s_p_in(), v.v_in_os(),
                    *(None, None, None, None) if tunable.tensor_a_pass_through else (s.s_in_offset(), None, None, None),
                    v.v_in_flag(), v.v_tmp(), None, None))

            return self._get_deferred()

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
                self._emit(m_wei_2d_global_load(v.v_gld_a(), s.s_p_wei(), v.v_wei_os(), s_wei_stride_d0(), s_wei_stride_d1(), s.s_wei_offset()))
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
                self.k_shift_pack_0 = sym_t('k_shift_pack_0'    ,112)
                self.k_shift_pack_1 = sym_t('k_shift_pack_1'    ,116)
                self.k_gemm_k_global_split  = sym_t("k_gemm_k_global_split"  , 120)
                self.k__pack_0              = sym_t("k__pack_0"              , 124)
                self.k_end                  = sym_t('k_end'                  , 128)
            else:
                self.k_end          = sym_t('k_end'             ,88)

            ta_k_vec_c, tb_nb0, tb_nb_vec_c = outer.get_thread_lengths()
            ca_k, ca_ce, cb_ce, cb_nb1 = outer.get_cluster_lengths()
            data_byte = amdgpu_precision_data_byte(outer.tunable.precision)

        def get_count(self):
            return self.k_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('k_'):
                    self._emit(v.declare())

    class kernel_sgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            ta_k_vec_c, tb_nb0, tb_nb_vec_c = outer.get_thread_lengths()
            k_pack = outer.get_k_pack()
            sseq                            = gpr_sequencer_t()
            self.outer                      = outer
            self.s_ka                       = sym_t('s_ka'                      , sseq(2))
            self.s_bx                       = sym_t('s_bx'                      , sseq(1))
            self.s_by                       = sym_t('s_by'                      , sseq(1))
            self.s_p_in                     = sym_t('s_p_in'                    , sseq(4))
            self.s_p_wei                    = sym_t('s_p_wei'                   , sseq(4))
            self.s_p_out                    = sym_t('s_p_out'                   , sseq(4))
            self.s_hi                       = sym_t('s_hi'                      , sseq(1))
            self.s_wi                       = sym_t('s_wi'                      , sseq(1))
            self.s_n                        = sym_t('s_n'                       , sseq(1))
            self.s_k                        = sym_t('s_k'                       , sseq(1))    # this is indeed k_per_group
            self.s_c                        = sym_t('s_c'                       , sseq(1))    # this is indeed c_per_group
            if outer.tunable.nxe != 0:
                self.s_ho                   = sym_t('s_ho'                      , sseq(1))
                self.s_wo                   = sym_t('s_wo'                      , sseq(1))
                self.s_stride_h             = sym_t('s_stride_h'                , sseq(1))
                self.s_stride_w             = sym_t('s_stride_w'                , sseq(1))
                self.s_dilation_h           = sym_t('s_dilation_h'              , sseq(1))
                self.s_dilation_w           = sym_t('s_dilation_w'              , sseq(1))
                self.s_pad_h                = sym_t('s_pad_h'                   , sseq(1))
                self.s_pad_w                = sym_t('s_pad_w'                   , sseq(1))
                self.s_y                    = sym_t('s_y'                       , sseq(1))
                self.s_x                    = sym_t('s_x'                       , sseq(1))
            self.s_group                    = sym_t('s_group'                   , sseq(1))

            self.s_in_stride_c             = sym_t('s_in_stride_c'            , sseq(1))
            self.s_in_stride_n              = sym_t('s_in_stride_n'             , sseq(1))

            if tb_nb0 != 1:
                self.s_in_stride_nb0        = sym_t('s_in_stride_nb0'           , sseq(1))
            self.s_wei_stride_x             = sym_t('s_wei_stride_x'            , sseq(1))

            self.s_out_stride_k             = sym_t('s_out_stride_k'           , sseq(1))
            self.s_out_stride_n             = sym_t('s_out_stride_n'            , sseq(1))

            self.s_block_gtc_ig             = sym_t("s_block_gtc_ig"            , sseq(1))
            self.s_block_gtc_ik             = sym_t("s_block_gtc_ik"            , sseq(1))
            self.s_block_gtc_inb            = sym_t("s_block_gtc_inb"           , sseq(1))

            self.s_move_slice_k_stride_gemm_k   = sym_t("s_move_slice_k_stride_gemm_k"  , sseq(1))
            
            self.s_move_slice_k_stride_c    = sym_t("s_move_slice_k_stride_c"  , sseq(1))

            self.s_knum                     = sym_t("s_knum"                    , 3)

            #if outer.tunable.nxe != 0:
            self.s_dim_br                   = sym_t("s_dim_br"                  , sseq(1))
            self.s_dim_mp                   = sym_t("s_dim_mp"                  , sseq(1))
            self.s_dim_mr                   = sym_t("s_dim_mr"                  , sseq(1))
            self.s_dim_np                   = sym_t("s_dim_np"                  , sseq(1))
            self.s_dim_nr                   = sym_t("s_dim_nr"                  , sseq(1))

            if outer.tunable.gemm_k_global_split:
                self.s_gemm_k_diff_c        = sym_t("s_gemm_k_diff_c"           , self.s_group.value)

            self.s_move_slice_k_y           = sym_t("s_move_slice_k_y"          , sseq(1))
            self.s_move_slice_k_x           = sym_t("s_move_slice_k_x"          , sseq(1))
            self.s_move_slice_k_c           = sym_t("s_move_slice_k_c"          , sseq(1))
            if outer.tunable.nxe != 0:
                self.s_diff_in_os_acc_c_y_x     = sym_t("s_diff_in_os_acc_c_y_x"    , self.s_block_gtc_ig.value)
                self.s_diff_in_os_ovf_y_acc_c   = sym_t("s_diff_in_os_ovf_y_acc_c"  , 0)
                self.s_diff_in_os_ovf_x_acc_y   = sym_t("s_diff_in_os_ovf_x_acc_y"  , self.s_dim_br.value)
                self.s_diff_in_iwi_acc_x        = sym_t("s_diff_in_iwi_acc_x"       , self.s_dim_mp.value)
                self.s_diff_in_iwi_ovf_x        = sym_t("s_diff_in_iwi_ovf_x"       , self.s_dim_np.value)
                self.s_diff_in_ihi_acc_y        = sym_t("s_diff_in_ihi_acc_y"       , self.s_pad_w.value)
                self.s_y_x_c                    = sym_t("s_y_x_c"                   , self.s_pad_h.value)

            self.s_kitr                     = sym_t("s_kitr"                    , 1)
            if outer.tunable.precision == 'int8':
                self.s_0xff                 = sym_t("s_0xff"                    , sseq(1))
            if outer.tunable.tensor_a_pass_through:
                # need s precache
                # in_npc = ((ta_ce1 // k_pack) - 2) if ((ta_ce1 // k_pack) - 2 > 0 ) else 0
                self.s_in_c_itr             = sym_t("s_in_c_itr"                , 2)
            else:
                self.s_in_offset            = sym_t("s_in_offset"               , sseq(1))
            if outer.tunable.precache_soffset:
                m_wei_2d_global_load, m_in_2d_global_load         = outer.get_macro_global_load()
                wei_npc = m_wei_2d_global_load.get_num_precache_soffset()
                self.s_wei_offset          = sym_t("s_wei_offset"             ,sseq(wei_npc))

            # TODO: this sgpr allocation is a mess
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                # allocate several sgpr to hold magic/shift value.
                self.s_magic_0             = sym_t("s_magic_0"                ,self.s_p_in.value + 2)
                self.s_magic_1             = sym_t("s_magic_1"                ,self.s_p_in.value + 3)
                self.s_magic_2             = sym_t("s_magic_2"                ,self.s_p_out.value + 2)
                self.s_magic_3             = sym_t("s_magic_3"                ,self.s_p_out.value + 3)

                self.s_magic_4             = sym_t("s_magic_4"                ,self.s_p_wei.value + 2)
                self.s_magic_5             = sym_t("s_magic_5"                ,self.s_p_wei.value + 3)
                self.s_shift_pack_0        = sym_t("s_shift_pack_0"           ,sseq(1))
                self.s_shift_pack_1        = sym_t("s_shift_pack_1"           ,sseq(1))

            if outer.tunable.gemm_k_global_split:
                self.s_block_gtc_ic        = sym_t("s_block_gtc_ic"           ,sseq(1)) # add c split
                self.s_gemmk_split         = sym_t("s_gemmk_split"            ,sseq(1))
                self.s_sub_c               = sym_t("s_sub_c"                  ,sseq(1))
            self.s_tmp                     = sym_t("s_tmp"                    ,sseq(6, 2))

            if IGEMM_FWD_GTC_NCHWC_DEBUG == 1:
                self.s_dbg                 = sym_t("s_dbg"                    ,sseq(4, 2))
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                # allocate several sgpr to hold magic/shift value.
                self.s_magic_0             = sym_t("s_magic_0"                ,sseq(1))
                self.s_magic_1             = sym_t("s_magic_1"                ,sseq(1))
            self.s_x_dilation_w            = sym_t("s_x_dilation_w"           ,sseq(1))
            self.s_y_dilation_h            = sym_t("s_y_dilation_h"           ,sseq(1))

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
            ta_k_vec_c, tb_nb0, tb_nb_vec_c = outer.get_thread_lengths()
            ca_k, ca_ce, cb_ce, cb_nb1 = outer.get_cluster_lengths()
            vector_c = self.outer.tunable.vector_c

            nb_per_thread = tb_nb0
            nk_per_thread = ta_k_vec_c
            assert nb_per_thread <= 16, "we pack flag into single vgpr"

            k_pack = outer.get_k_pack()
            share_load_packed  = k_pack

            is_vgpr_acc_c = outer.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS
            vseq = gpr_sequencer_t()
            data_byte                   = amdgpu_precision_data_byte(outer.tunable.precision)
            num_vgpr_global_load_a      = outer.get_num_vgpr_global_load_a()
            num_vgpr_global_load_b      = outer.get_num_vgpr_global_load_b()

            share_load_packed_vgpr      = share_load_packed // (4 // data_byte)

            num_vgpr_acc_a              = share_load_packed_vgpr * outer.tunable.num_vgpr_accumulate_a if not outer.tunable.tensor_a_pass_through else 0
            num_vgpr_acc_b              = share_load_packed_vgpr * outer.tunable.num_vgpr_accumulate_b if not outer.tunable.tensor_b_pass_through else 0

            # print(f"share_load_packed_vgpr:{share_load_packed_vgpr}, tunable.num_vgpr_accumulate_b:{outer.tunable.num_vgpr_accumulate_b}, num_vgpr_acc_b:{num_vgpr_acc_b}")
            if is_vgpr_acc_c:
                self.v_c                = sym_t("v_c"            ,vseq(outer.tunable.num_vgpr_accumulate_c))
                v_c_num                 = vseq()
            else:
                v_c_resuable_num        = num_vgpr_acc_a + num_vgpr_acc_b + \
                                            num_vgpr_global_load_a + num_vgpr_global_load_b + \
                                            3 * nb_per_thread + 6      # from v_sst_a_os to v_co_sst
                #v_c_coalescing_num      = outer.tunable.num_agpr_accumulate_c // outer.coalescing_store_groups
                v_c_coalescing_num      = outer.coalescing_store.get_vgpr_usage()
                v_c_needed              = (v_c_coalescing_num - v_c_resuable_num) if (v_c_coalescing_num - v_c_resuable_num) > 0 else 0

                v_c_needed              = v_c_needed if v_c_needed > 0 else 0  # let at least 0
                self.v_c                = sym_t("v_c"            ,vseq(v_c_needed), f"coalescing:{v_c_coalescing_num}, needed:{v_c_needed}, resuable:{v_c_resuable_num}")

            if not outer.tunable.tensor_a_pass_through:
                self.v_a                = sym_t("v_a"               ,vseq(num_vgpr_acc_a))
            if not outer.tunable.tensor_b_pass_through:
                self.v_b                = sym_t("v_b"               ,vseq(num_vgpr_acc_b))
            self.v_gld_a                = sym_t("v_gld_a"           ,vseq(num_vgpr_global_load_a))
            if outer.tunable.global_prefetch_a_num == 2:
                self.v_gld_a_gpf        = sym_t("v_gld_a_gpf"       ,vseq(num_vgpr_global_load_a))
            self.v_gld_b                = sym_t("v_gld_b"           ,vseq(num_vgpr_global_load_b))
            if outer.tunable.global_prefetch_b_num == 2:
                self.v_gld_b_gpf        = sym_t("v_gld_b_gpf"       ,vseq(num_vgpr_global_load_b))
            if not outer.tunable.tensor_a_pass_through:
                self.v_sst_a_os         = sym_t("v_sst_a_os"        ,vseq(1))
                self.v_sld_a_os         = sym_t("v_sld_a_os"        ,vseq(1))
            if not outer.tunable.tensor_b_pass_through:
                self.v_sst_b_os         = sym_t("v_sst_b_os"        ,vseq(1))
                self.v_sld_b_os         = sym_t("v_sld_b_os"        ,vseq(1))

            self.v_in_os                = sym_t("v_in_os"           ,vseq(nb_per_thread))
            self.v_in_ihi_list          = sym_t("v_in_ihi_list"     ,vseq(nb_per_thread))
            self.v_in_iwi_list          = sym_t("v_in_iwi_list"     ,vseq(nb_per_thread))
            if IGEMM_FWD_GTC_NCHW_PACK_IN_FLAG:
                self.v_in_flag          = sym_t("v_in_flag"         ,vseq(1))   # bfe this!, hi 16bit is flag_n, lo 16 bit is pure flag
            else:
                self.v_in_flag          = sym_t("v_in_flag"         ,vseq(nb_per_thread))
                self.v_in_flag_n        = sym_t("v_in_flag_n"       ,vseq(1))      # bfe this!, lo 16bit is flag_n

            self.v_wei_os               = sym_t("v_wei_os"          ,vseq(1))
            self.v_out_os               = sym_t("v_out_os"          ,vseq(1))

            if outer.tunable.tensor_a_pass_through:
                self.v_gtc_ic_a         = sym_t("v_gtc_ic_a"        ,self.v_gld_a.value)
            if outer.tunable.tensor_b_pass_through:
                self.v_gtc_ic_b         = sym_t("v_gtc_ic_b"        ,self.v_gld_b.value)
            if not (outer.tunable.tensor_a_pass_through and outer.tunable.tensor_b_pass_through):
                self.v_gtc_ic           = sym_t("v_gtc_ic"          ,vseq(1))

            assert not outer.tunable.tensor_b_pass_through
            self.v_gtc_iec          = sym_t("v_gtc_iec"         ,vseq(1))
            self.v_gtc_iy           = sym_t("v_gtc_iy"          ,vseq(1))
            self.v_gtc_ix           = sym_t("v_gtc_ix"          ,vseq(1))
            self.v_in_inb               = sym_t("v_in_inb"          ,vseq(1))
            self.v_in_in                = sym_t("v_in_in"           ,vseq(1))
            self.v_wei_ik               = sym_t("v_wei_ik"          ,vseq(1))

            self.v_co_sst               = sym_t("v_co_sst"          ,self.v_in_in.value)
            self.v_co_sld               = sym_t("v_co_sld"          ,vseq(1))

            self.v_out_flag             = sym_t("v_out_flag"        ,self.v_wei_ik.value)
            self.v_out_inb              = sym_t("v_out_inb"         ,self.v_in_inb.value)
            self.v_out_ik               = sym_t("v_out_ik"          ,self.v_in_inb.value)

            self.v_gemm_in              = sym_t("v_gemm_in"         ,vseq(1))
            self.v_gemm_im              = sym_t("v_gemm_im"         ,vseq(1))
            self.v_co_sub_m_index       = sym_t("v_co_sub_m_index"  ,self.v_gemm_im.value)
            self.v_co_sub_n_index       = sym_t("v_co_sub_n_index"  ,self.v_gemm_in.value)
            self.v_out_in               = sym_t("v_out_in"          ,self.v_gemm_in.value)
            
            self.v_coalescing_store_index = sym_t("v_coalescing_store_index" ,self.v_gemm_in.value)
            
            self.v_tmp                  = sym_t("v_tmp"             ,vseq(6, 2))
            #self.v_wei_tmp_pack         = sym_t("v_wei_tmp_pack"    , (self.v_gld_a.value - 1 if self.v_gld_a.value > 1 else vseq(1)))
            #if nk_per_thread <= 4 and IGEMM_FWD_GTC_NCHW_PACK_IN_FLAG == 0:
            #    self.v_wei_flag         = sym_t("v_wei_flag"        ,self.v_tmp.value)
            #else:
            #    self.v_wei_flag         = sym_t("v_wei_flag"        ,vseq(nk_per_thread))
                
            if IGEMM_FWD_GTC_NCHWC_DEBUG == 1:
                self.v_dbg                = sym_t("v_dbg"            ,vseq(2, 2))
            total_vgpr                  = vseq()
            self.accum_start            = 0
            if outer.tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
                if self.mc.arch_config.arch == AMDGPU_ARCH_GFX90A:
                    total_vgpr          = (total_vgpr + 3) // 4 * 4 # round to multiply of 4
                    self.accum_start    = total_vgpr
                    total_vgpr          = total_vgpr + outer.tunable.num_agpr_accumulate_c
                else:
                    # if xdlops agpr is larger than vgpr usage, must change vgpr count to agpr
                    total_vgpr          = max(total_vgpr, outer.tunable.num_agpr_accumulate_c)
            self.v_end                  = sym_t("v_end"          ,total_vgpr)

        def get_count(self):
            return self.v_end.value
        
        def get_accum_start(self):
            return self.accum_start

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('v_'):
                    self._emit(v.declare())

    class kernel_agpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            assert outer.tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS, 'only xdlops can use agpr'
            self.outer         = outer
            if IGEMM_FWD_GTC_NCHW_ACCVGPR_UNIFIED and self.mc.arch_config.arch == AMDGPU_ARCH_GFX90A:
                vgpr = outer.kernel_vgpr_t(mc, outer)
                aseq = gpr_sequencer_t(vgpr.get_accum_start())
            else:
                aseq = gpr_sequencer_t()
            self.a_c           = sym_t("a_c",          aseq(outer.tunable.num_agpr_accumulate_c))
            self.a_end         = sym_t("a_end",        aseq())

        def get_count(self):
            return self.a_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('a_'):
                    self._emit(v.declare())

    def get_num_vgpr_global_load_a(self):
        ta_k_vec_c, tb_nb0, tb_nb_vec_c = self.get_thread_lengths()
        pack_factor = (4 // amdgpu_precision_data_byte(self.tunable.precision)) if ta_k_vec_c != 1 else 1
        return self.tunable.num_global_load_a // pack_factor
    
    def get_num_vgpr_global_load_b(self):
        ta_k_vec_c, tb_nb0, tb_nb_vec_c = self.get_thread_lengths()
        pack_factor = (4 // amdgpu_precision_data_byte(self.tunable.precision)) if tb_nb_vec_c != 1 else 1
        return self.tunable.num_global_load_b // pack_factor

    def get_thread_lengths(self):
        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(t_ta) == 4 and len(t_tb) == 4
        assert t_ta[0] == 1 and t_ta[1] == 1 and t_ta[2] == 1, "no need to do a thread load in ce dimension. No need to unmerge k dimension"
        assert t_ta[0] == 1 and t_ta[1] == 1 

        ta_k_vec_c  = t_ta[3]
        tb_nb0, tb_nb_vec_c = t_tb[2], t_tb[3]

        #assert self.tunable.nxb == tb_nb0

        return ta_k_vec_c, tb_nb0, tb_nb_vec_c # M, N

    def get_cluster_lengths(self):
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4
        assert c_ta[0] == 1 and c_ta[2] == 1 and c_tb[0] == 1 and c_tb[2] == 1

        ca_ce, ca_k   = c_ta[1], c_ta[3]
        cb_ce, cb_nb1 = c_tb[1], c_tb[3]

        return ca_k, ca_ce, cb_ce, cb_nb1  # M, K, N

    def get_dims_lengths(self):
        ta_k_vec_c, tb_nb0, tb_nb_vec_c = self.get_thread_lengths()
        ca_k, ca_ce, cb_ce, cb_nb1      = self.get_cluster_lengths()

        na_ce, na_k_vec_c           = ca_ce, ta_k_vec_c * ca_k
        nb_ce, nb_nb0, nb_nb1_vec_c = cb_ce, tb_nb0, tb_nb_vec_c * cb_nb1

        return na_k_vec_c, na_ce, nb_ce, nb_nb0, nb_nb1_vec_c  # M, K, N

    def get_thread_copy_dims(self):
        ta_k_vec_c, tb_nb0, tb_nb_vec_c = self.get_thread_lengths()
        in_thread_copy_dims  = [tb_nb0, tb_nb_vec_c]
        wei_thread_copy_dims = [ta_k_vec_c]     # always reordered
        return in_thread_copy_dims, wei_thread_copy_dims

    def get_thread_copy_index(self):
        in_thread_copy_dims, wei_thread_copy_dims = self.get_thread_copy_dims()
        in_thread_copy_index  = _find_non_1_index_in_list(in_thread_copy_dims)
        wei_thread_copy_index = _find_non_1_index_in_list(wei_thread_copy_dims)

        '''
        if thread lengths both dimension is 1, means every thread only copy one pixel.
        we need support this also
        '''
        return in_thread_copy_index, wei_thread_copy_index

    def get_k_pack(self):
        _, _, tb_nb_vec_c = self.get_thread_lengths()
        return tb_nb_vec_c

    def get_macro_global_load(self):
        inline = True if self.tunable.fma_interleave else False
        vector_c = self.tunable.vector_c
        ta_k_vec_c, tb_nb0, tb_nb_vec_c                 = self.get_thread_lengths()
        ca_k, ca_ce, cb_ce, cb_nb1                      = self.get_cluster_lengths()
        na_k_vec_c, na_ce,  nb_ce, nb_nb0, nb_nb1_vec_c = self.get_dims_lengths()

        in_thread_copy_dims, wei_thread_copy_dims = self.get_thread_copy_dims()
        in_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()
        ctrl_wei_gld = ctrl_2d_global_load_t()
        ctrl_in_gld = ctrl_2d_global_load_t()

        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        ctrl_wei_gld.precision = self.tunable.precision
        ctrl_in_gld.precision  = self.tunable.precision
        ctrl_wei_gld.vector_d1 = utility_gcd(vector_c, 4 * (4 // data_byte))
        ctrl_in_gld.vector_d1  = utility_gcd(vector_c, 4 * (4 // data_byte))

        ctrl_in_gld.use_flag = 1
        ctrl_wei_gld.use_flag = 0

        ctrl_in_gld.arch_name = AMDGPU_ARCH_GFX1030
        ctrl_wei_gld.arch_name = AMDGPU_ARCH_GFX1030

        # ctrl_in_gld.vector_d1 = self.get_k_pack()
        if self.in_thread_copy_ndim == 2:
            ctrl_in_gld.flag_on_d0 = 1
            ctrl_in_gld.precache_ptn = GLOBAL_PTN_D0_V | GLOBAL_PTN_D1_K
            ctrl_in_gld.length_d0 = in_thread_copy_dims[in_thread_copy_index[0]]
            ctrl_in_gld.length_d1 = in_thread_copy_dims[in_thread_copy_index[1]]
        elif self.in_thread_copy_ndim == 1:
            if tb_nb0 * tb_nb_vec_c != 1:
                ctrl_in_gld.precache_ptn = GLOBAL_PTN_D0_K | GLOBAL_PTN_D1_V
                ctrl_in_gld.flag_on_d1 = 1
            else:
                ctrl_in_gld.precache_ptn = GLOBAL_PTN_D0_V | GLOBAL_PTN_D1_K
                ctrl_in_gld.flag_on_d0 = 1
            ctrl_in_gld.length_d0 = 1
            ctrl_in_gld.length_d1 = in_thread_copy_dims[in_thread_copy_index[0]]
        else:
            ctrl_in_gld.length_d0 = 1
            ctrl_in_gld.length_d1 = in_thread_copy_dims[-1]

        ctrl_wei_gld.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]
        ctrl_wei_gld.precache_ptn = GLOBAL_PTN_D0_S | GLOBAL_PTN_D1_S

        ctrl_wei_gld.dim_conti_flag = 1
        ctrl_wei_gld.workgroup_length = ca_k

        if self.tunable.precache_soffset:
            return macro_igemm_2d_global_load_precache_soffset_t(self.mc, ctrl_wei_gld, inline), \
                    macro_igemm_2d_global_load_precache_offset_t(self.mc, ctrl_in_gld, inline)
        else:
            return macro_igemm_2d_global_load_t(self.mc, ctrl_wei_gld, inline),  macro_igemm_2d_global_load_precache_voffset_t(self.mc, ctrl_in_gld, inline)

    def get_macro_shared_store(self):
        #in_thread_copy_dims, wei_thread_copy_dims = self.get_thread_copy_dims()
        #in_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()
        na_k_vec_c, na_ce, nb_ce, nb_nb0, nb_nb1_vec_c = self.get_dims_lengths()
        ta_k_vec_c, tb_nb0, tb_nb_vec_c = self.get_thread_lengths()
        ca_k, ca_ce, cb_ce, cb_nb1 = self.get_cluster_lengths()
        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        k_pack = self.get_k_pack()
        m_wei_2d_global_load, m_in_2d_global_load = self.get_macro_global_load()
        k_pack_gld_a = m_in_2d_global_load.ctrl.vector_d1
        k_pack_gld_b = m_wei_2d_global_load.ctrl.vector_d1

        if not self.tunable.tensor_a_pass_through:
            # input is gemm_k * gemm_m * k_pack
            wei_sst_ctrl = ctrl_3d_shared_store_t()
            wei_sst_ctrl.precision = self.tunable.precision
            wei_sst_ctrl.length_d0 = 1
            wei_sst_ctrl.length_d1 = ta_k_vec_c // k_pack_gld_a
            wei_sst_ctrl.length_dp = k_pack_gld_a
            wei_sst_ctrl.vector_dp = k_pack_gld_a
            wei_sst_ctrl.stride_d0 = 1
            wei_sst_ctrl.stride_d1 = (ca_k * self.tunable.vector_c) * data_byte

        if not self.tunable.tensor_b_pass_through:
            # wei is gemm_k * gemm_n * k_pack
            in_sst_ctrl = ctrl_3d_shared_store_t()
            in_sst_ctrl.precision = self.tunable.precision
            in_sst_ctrl.length_d0 = tb_nb0
            in_sst_ctrl.length_d1 = tb_nb_vec_c // k_pack_gld_b
            in_sst_ctrl.length_dp = k_pack_gld_b
            in_sst_ctrl.vector_dp = k_pack_gld_b
            in_sst_ctrl.stride_d0 = nb_nb1_vec_c * data_byte
            in_sst_ctrl.stride_d1 = k_pack_gld_b * data_byte

        inline = True if self.tunable.fma_interleave else False 
        return macro_igemm_3d_shared_store_t(self.mc, in_sst_ctrl, inline) if not self.tunable.tensor_a_pass_through else None, \
            macro_igemm_3d_shared_store_t(self.mc, wei_sst_ctrl, inline) if not self.tunable.tensor_b_pass_through else None

    def get_macro_move_slice_window(self):
        inline = True if self.tunable.fma_interleave else False
        ta_k_vec_c, tb_nb0, tb_nb_vec_c = self.get_thread_lengths()
        nb_per_thread = tb_nb0
        nk_per_thread = ta_k_vec_c
        unroll_k = self.tunable.gemm_k_per_block
        if self.tunable.nxe != 0:
            move_slice_window = self.macro_move_slice_window_block_wise_t(self.mc, self.tunable, inline,
                                        unroll_k=unroll_k, nb_per_thread=nb_per_thread, nk_per_thread=nk_per_thread, m_set_flag_nhw = self.get_macro_set_flag_nhw())
        else:
            move_slice_window = self.macro_move_slice_window_block_wise_1x1_t(self.mc, self.tunable, inline,
                                        unroll_k=unroll_k, nb_per_thread=nb_per_thread, nk_per_thread=nk_per_thread)

        # return single functor !
        return move_slice_window

    def get_macro_set_flag_nhw(self):
        inline = True if self.tunable.fma_interleave else False
        return self.macro_set_flag_nhw(self.mc, inline)

    def get_symbol_global_load_s_stride_d0_d1(self):
        ta_k_vec_c, tb_nb0, tb_nb_vec_c = self.get_thread_lengths()
        # get the symbol object that load 2d may use
        s = self.sgpr
        s_dummy = sym_t("s_dummy")
        in_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()

        # [tb_nb0, tb_nb_vec_c]
        in_stride_gprs = [s_dummy,
                          s_dummy,
                          s_dummy,
                          s_dummy]

        # [ta_k_vec_c]]
        wei_stride_gprs = [s_dummy,
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

        if self.wei_thread_copy_ndim == 2:
            # print(f" ____ wei_thread_copy_index:{len(wei_thread_copy_index)}, {wei_thread_copy_index}")
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
        kernel_code_dict = {
                'enable_sgpr_kernarg_segment_ptr'   :   1,
                'enable_sgpr_workgroup_id_x'        :   1,
                'enable_sgpr_workgroup_id_y'        :   1,
                'enable_sgpr_workgroup_id_z'        :   1,
                'enable_vgpr_workitem_id'           :   0,
                'workgroup_group_segment_byte_size' :   self.tunable.lds_total,
                'kernarg_segment_byte_size'         :   self.karg.get_count(),
                'wavefront_sgpr_count'              :   self.sgpr.get_count() + 2*3,
                'workitem_vgpr_count'               :   self.vgpr.get_count()}
        if self.mc.arch_config.arch == AMDGPU_ARCH_GFX90A:
            assert self.vgpr.get_accum_start() % 4 == 0
            kernel_code_dict['accum_offset']        =   self.vgpr.get_accum_start()
        kernel_code = amdgpu_kernel_code_t(kernel_code_dict)
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
            kas.append(amdgpu_kernel_arg_t('shift_pack_0'   , 4, 112, 'by_value','i32'))
            kas.append(amdgpu_kernel_arg_t('shift_pack_1'   , 4, 116, 'by_value','i32'))
            kas.append(amdgpu_kernel_arg_t('gemm_k_split'   , 4, 120, 'by_value','i32'))
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
                        if e is not None and not e.is_inline():
                            #continue
                            kernel_macros.extend([m for m in rtn])
                else:
                    #if hasattr(rtn, 'emit'):
                    if rtn is not None and rtn.is_inline():
                        #continue
                        kernel_macros.append(rtn)
        return kernel_macros

    def emit_kernel_prologue(self):
        s = self.sgpr
        v = self.vgpr
        k = self.karg

        ta_k_vec_c, tb_nb0, tb_nb_vec_c = self.get_thread_lengths()
        ca_k, ca_ce, cb_ce, cb_nb1 = self.get_cluster_lengths()
        na_k_vec_c, na_ce, nb_ce, nb_nb0, nb_nb1_vec_c = self.get_dims_lengths()

        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        m_set_flag_nhw      = self.get_macro_set_flag_nhw()
        s_in_stride_d0, s_in_stride_d1, s_wei_stride_d0, s_wei_stride_d1 = self.get_symbol_global_load_s_stride_d0_d1()

        m_wei_2d_global_load, m_in_2d_global_load = self.get_macro_global_load()

        tc_index_dispatcher = igemm_thread_cluster_index_dispatcher_t(self.mc)
        tc_index_accumulator = igemm_thread_cluster_index_accumulator_t(self.mc)

        nb_per_thread = tb_nb0
        nk_per_thread = ta_k_vec_c // self.tunable.vector_c

        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            m_mdiv_u32_vs = macro_mdiv_u32_rem_vs_t(self.mc)
            m_mdiv_u32_ss = macro_mdiv_u32_rem_ss_t(self.mc)
        else:
            m_int_div_rem_vv = macro_int_div_rem_vv_t(self.mc)
            m_int_div_rem_vs = macro_int_div_rem_vs_t(self.mc)
            m_int_div_rem_ss = macro_int_div_rem_ss_t(self.mc)

        s_dummy = sym_t("s_dummy")

        m_div_u32_si = div_u32_si_t(self.mc)
        m_mul_u32_si = mul_u32_si_t(self.mc)

        k_pack = self.get_k_pack()
        k_pack_src_mat = k_pack
        k_pack_gld_b = m_in_2d_global_load.ctrl.vector_d1
        k_pack_gld_a = m_wei_2d_global_load.ctrl.vector_d1

        # start emit
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
            self._emit(f"s_load_dwordx2 s[{s.s_magic_2((0, 1))}],  s[{s.s_ka((0, 1))}],  0+{k.k_magic_2()}")
            self._emit(f"s_load_dwordx2 s[{s.s_magic_4((0, 1))}], s[{s.s_ka((0, 1))}],  0+{k.k_magic_4()}")
            self._emit(f"s_load_dword s[{s.s_shift_pack_0()}], s[{s.s_ka((0, 1))}],  0+{k.k_shift_pack_0()}")
            self._emit(f"s_load_dword s[{s.s_shift_pack_1()}], s[{s.s_ka((0, 1))}],  0+{k.k_shift_pack_1()}")
            if self.tunable.gemm_k_global_split:
                self._emit(f"s_load_dword s[{s.s_gemmk_split()}], s[{s.s_ka((0, 1))}],  0+{k.k_gemm_k_global_split()}")

        if IGEMM_FWD_GTC_NCHWC_DEBUG == 1:
            self._emit_empty_line()
            self._emit("; debug vgpr")
            self._emit("v_mov_b32 v1, 0")
            self._emit(f"v_add_lshl_u32 v[{v.v_dbg()}], v0, v1, 2")
            self._emit(f"s_load_dwordx2 s[{s.s_dbg((0,1))}], s[s_ka:s_ka+1], k_p_out")
            self._emit(f"s_mov_b32 s[{s.s_dbg(2)}], s[{s.s_bx()}]")
            self._emit(f"s_mov_b32 s[{s.s_dbg(3)}], s[{s.s_by()}]")
            self._emit_empty_line()

        self._emit(f"; wei(1, ce, 1, k-vec-c) thread_lengths: {1}x{1}x{1}x{ta_k_vec_c}, cluster_length: {1}x{ca_ce}x{1}x{ca_k}, k_pack:{self.tunable.vector_c}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        self._emit(tc_index_dispatcher(v.v_wei_ik(), v.v_tmp(), ca_k, 1))
        self._emit(tc_index_dispatcher(v.v_gtc_iec(), v.v_tmp(), ca_ce, 1, True))
        
        self._emit_empty_line()

        self._emit(f"; inp(1, ce, nb0, nb1) thread_length: {1}x{1}x{tb_nb0}x{tb_nb_vec_c}, cluster_length: {1}x{cb_ce}x{1}x{cb_nb1}, k_pack:{self.tunable.vector_c}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        if self.tunable.tensor_b_pass_through:
            #
            assert False
        else:
            self._emit(tc_index_dispatcher(v.v_in_inb(), v.v_tmp(),  cb_nb1, 1, True))

        # if self.tunable.precision == 'int8':
        #     self._emit(f"s_mov_b32 s[{s.s_0xff()}], 0xff")
        self._emit(f"s_mov_b32 s[{s.s_tmp()}], {0x00ffffff}")
        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit_empty_line()
        self._emit(f"; calculate index")

        if self.tunable.nxe != 0:
            self._emit(f"s_lshr_b32 s[{s.s_move_slice_k_y()}], s[{s.s_y()}], 24")
            self._emit(f"s_lshr_b32 s[{s.s_move_slice_k_x()}], s[{s.s_x()}], 24")
            self._emit(f"s_lshr_b32 s[{s.s_move_slice_k_c()}], s[{s.s_c()}], 24")
            self._emit(f"s_and_b32 s[{s.s_y()}], s[{s.s_tmp()}], s[{s.s_y()}]")
            self._emit(f"s_and_b32 s[{s.s_x()}], s[{s.s_tmp()}], s[{s.s_x()}]")
        self._emit(f"s_and_b32 s[{s.s_c()}], s[{s.s_tmp()}], s[{s.s_c()}]")
        # calculate stride, not shift data byte yet
        # input
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_lshr_b32 s[{s.s_sub_c()}], s[{s.s_c()}], s[{s.s_gemmk_split()}] ;add gkgs for c")
        self._emit(f"s_mul_i32 s[{s.s_in_stride_c()}], s[{s.s_hi()}], s[{s.s_wi()}]")
        self._emit(f"s_mul_i32 s[{s.s_in_stride_c()}], s[{s.s_in_stride_c()}], {self.tunable.vector_c}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_in_stride_c()}], s[{s.s_c()}]")
        self._emit(f"s_mul_i32 s[{s.s_in_stride_n()}], s[{s.s_tmp()}], s[{s.s_group()}]")

        # weight
        self._emit(f"s_mul_i32 s[{s.s_wei_stride_x()}], s[{s.s_k()}], {self.tunable.vector_c}")

        # output
        self._emit(f"s_mul_i32 s[{s.s_out_stride_k()}], s[{s.s_wo() if self.tunable.nxe != 0 else s.s_wi()}], s[{s.s_ho() if self.tunable.nxe != 0 else s.s_hi()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_k()}], s[{s.s_out_stride_k()}]")
        self._emit(f"s_mul_i32 s[{s.s_out_stride_k()}], {self.tunable.vector_c}, s[{s.s_out_stride_k()}]")
        self._emit(f"s_mul_i32 s[{s.s_out_stride_n()}], s[{s.s_tmp()}], s[{s.s_group()}]")

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
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_lshr_b32 s[{s.s_knum()}], s[{s.s_wei_stride_k()}], s[{s.s_gemmk_split()}]")
        else:
            if self.tunable.nxe != 0:
                self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_x()}], s[{s.s_y()}]")
                self._emit(f"s_mul_i32 s[{s.s_knum()}], s[{s.s_tmp()}], s[{s.s_c()}]")
            else:
                self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_c()}]")

        # pad gemm_m, gemm_n
        self._emit(f"s_mul_i32 s[{s.s_dim_br()}], s[{s.s_ho() if self.tunable.nxe != 0 else s.s_hi()}], s[{s.s_wo() if self.tunable.nxe != 0 else s.s_wi()}]")
        self._emit(f"s_mul_i32 s[{s.s_dim_nr()}], s[{s.s_n()}], s[{s.s_dim_br()}]")
        self._emit(f"s_add_u32 s[{s.s_tmp(2)}], {self.tunable.gemm_n_per_block - 1}, s[{s.s_dim_nr()}]")
        
        self._emit(m_div_u32_si(s.s_tmp(1), s.s_tmp(2), self.tunable.gemm_n_per_block, s.s_tmp()))
        self._emit(m_mul_u32_si(s.s_dim_np(), s.s_tmp(1), self.tunable.gemm_n_per_block))
            
        self._emit(f"s_add_u32 s[{s.s_tmp()}], {self.tunable.gemm_m_per_block - 1}, s[{s.s_k()}]")
        self._emit(m_div_u32_si(s.s_tmp(1), s.s_tmp(), self.tunable.gemm_m_per_block, s.s_tmp(1)))
        self._emit(m_mul_u32_si(s.s_dim_mp(), s.s_tmp(1), self.tunable.gemm_m_per_block))

        self._emit_empty_line()
        self._emit(f"; gemm_m_per_block:{self.tunable.gemm_m_per_block}, gemm_n_per_block:{self.tunable.gemm_n_per_block}, source_access_order:{self.tunable.source_access_order}")

        # calculate group index
        self._emit(m_div_u32_si(s.s_tmp(), s.s_dim_mp(), self.tunable.gemm_m_per_block, s.s_tmp()))
        self._emit(m_div_u32_si(s.s_tmp(1), s.s_dim_np(), self.tunable.gemm_n_per_block, s.s_tmp(1)))
        self._emit(f"s_mul_i32 s[0], s[{s.s_tmp(1)}], s[{s.s_tmp()}]")
        if self.tunable.gemm_k_global_split:
            # calculate block ic
            self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], 1, s[{s.s_gemmk_split()}]")
            self._emit(f"s_sub_u32 s[{s.s_tmp(3)}], s[{s.s_tmp(3)}], 1")
            self._emit(f"s_and_b32 s[{s.s_block_gtc_ic()}], s[{s.s_bx()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_lshr_b32 s[{s.s_bx()}], s[{s.s_bx()}], s[{s.s_gemmk_split()}]")
            self._emit(f"s_mul_i32 s[{s.s_block_gtc_ic()}], s[{s.s_block_gtc_ic()}], s[{s.s_sub_c()}]")
        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080018 ; offset:24, width:8")
            self._emit(m_mdiv_u32_ss(s.s_tmp(4), s.s_block_gtc_ig(), s.s_bx(), s.s_magic_3(), s.s_tmp(3), '0', s.s_tmp()))
        else:
            self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_block_gtc_ig(), s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))

        # s.s_tmp(4)=> rem, gemm_m, gemm_n, s.s_block_gtc_ig()=> quo, group
        self._emit(f"s_mov_b32 s[{s.s_bx()}], s[{s.s_tmp(4)}]")

        if self.tunable.source_access_order == IGEMM_GTC_TUNABLE_SOURCE_ACCESS_ORDER_GEMM_M_GEMM_N:
            self._emit(m_div_u32_si('0', s.s_dim_np(), self.tunable.gemm_n_per_block, s.s_tmp()))
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080000 ; offset:0, width:8")
                self._emit(m_mdiv_u32_ss(s.s_tmp(4), s.s_tmp(5), s.s_bx(), s.s_magic_0(), s.s_tmp(3), '0', s.s_tmp()))
            else:
                self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_tmp(5), s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))

        else:
            self._emit(m_div_u32_si('0', s.s_dim_mp(), self.tunable.gemm_m_per_block, s.s_tmp()))
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080000 ; offset:0, width:8")
                self._emit(m_mdiv_u32_ss(s.s_tmp(5), s.s_tmp(4), s.s_bx(), s.s_magic_0(), s.s_tmp(3), '0', s.s_tmp()))
            else:
                self._emit(m_int_div_rem_ss(s.s_tmp(5), s.s_tmp(4), s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))

        self._emit(f"; s_tmp+4:block_gtc_in, s_tmp+5:block_gtc_im")
        if igemm_is_pow2(self.tunable.gemm_n_per_block):
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_inb()}], s[{s.s_tmp(4)}], {igemm_log2(self.tunable.gemm_n_per_block)}")
        else:
            self._emit(f"s_mul_i32 s[{s.s_block_gtc_inb()}], s[{s.s_tmp(4)}], {self.tunable.gemm_n_per_block}")
        if igemm_is_pow2(self.tunable.gemm_m_per_block):
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ik()}], s[{s.s_tmp(5)}], {igemm_log2(self.tunable.gemm_m_per_block)}")
        else:
            self._emit(f"s_mul_i32 s[{s.s_block_gtc_ik()}], s[{s.s_tmp(5)}], {self.tunable.gemm_m_per_block}")

        # transform nb
        self._emit(f"v_add_nc_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_inb()}], v[{v.v_in_inb()}]")
        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080008 ; offset:8, width:8")
            self._emit(m_mdiv_u32_vs(v.v_tmp(4), v.v_in_in(), v.v_tmp(5), s.s_magic_1(), s.s_tmp(3), s.s_dim_br(), v.v_tmp()))
            self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080010 ; offset:16, width:8")
            self._emit(m_mdiv_u32_vs(v.v_in_iwi_list(0), v.v_in_ihi_list(0), v.v_tmp(4), s.s_magic_2(), s.s_tmp(3), s.s_wo() if self.tunable.nxe != 0 else s.s_wi(), v.v_tmp()))
        else:
            self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_in_in(), v.v_tmp(5), s.s_dim_br(), v.v_tmp(), s.s_tmp()))
            self._emit(m_int_div_rem_vs(v.v_in_iwi_list(0), v.v_in_ihi_list(0), v.v_tmp(4), s.s_wo() if self.tunable.nxe != 0 else s.s_wi(), v.v_tmp(), s.s_tmp()))

        if self.tunable.nxe != 0:
            # ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
            # iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w
            self._emit(f"v_mul_lo_u32 v[{v.v_in_ihi_list(0)}], s[{s.s_stride_h()}], v[{v.v_in_ihi_list(0)}]")
            self._emit(f"v_sub_nc_i32 v[{v.v_in_ihi_list(0)}], v[{v.v_in_ihi_list(0)}], s[{s.s_pad_h()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_in_iwi_list(0)}], s[{s.s_stride_w()}], v[{v.v_in_iwi_list(0)}]")
            self._emit(f"v_sub_nc_i32 v[{v.v_in_iwi_list(0)}], v[{v.v_in_iwi_list(0)}], s[{s.s_pad_w()}]")
            
            self._emit_empty_line()

        # transform ce
        if self.tunable.nxe != 0:
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_y()}], s[{s.s_x()}]")
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_1()}], 0x00080000 ; offset:0, width:8")
                self._emit(m_mdiv_u32_vs(v.v_tmp(4), v.v_gtc_ic(), v.v_gtc_iec(), s.s_magic_4(), s.s_tmp(3), s.s_tmp(), v.v_tmp()))
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_1()}], 0x00080008 ; offset:8, width:8")
                self._emit(m_mdiv_u32_vs(v.v_gtc_ix(), v.v_gtc_iy(), v.v_tmp(4), s.s_magic_5(), s.s_tmp(3), s.s_x(), v.v_tmp()))
            else:
                assert False

            self._emit(f"v_mul_lo_u32 v[{v.v_sst_a_os()}], s[{s.s_dilation_h()}], v[{v.v_gtc_iy()}]")
            self._emit(f"v_add_nc_u32 v[{v.v_in_ihi_list(0)}], v[{v.v_in_ihi_list(0)}], v[{v.v_sst_a_os()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_sld_a_os()}], s[{s.s_dilation_w()}], v[{v.v_gtc_ix()}]")
            self._emit(f"v_add_nc_u32 v[{v.v_in_iwi_list(0)}], v[{v.v_in_iwi_list(0)}], v[{v.v_sld_a_os()}]")
        else:
            self._emit(f"v_mov_b32 v[{v.v_gtc_ic()}], v[{v.v_gtc_iec()}]")

        self._emit(f"v_cmp_gt_u32  s[{s.s_n()}], v[{v.v_in_in()}]")
        self._emit(f"v_cndmask_b32 v[{v.v_tmp()}], 0, 1")
        self._emit(f"v_lshlrev_b32 v[{v.v_in_flag_n()}], 0, v[{v.v_tmp()}]")

        self._emit_empty_line()

        self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ig()}], s[{s.s_block_gtc_ig()}], {igemm_log2(data_byte)}")

        def calculate_and_load_input():
            self._emit(f"; calculate in offset")
            if self.tunable.tensor_a_pass_through:
                self._emit(f"s_mov_b32 s[{s.s_in_c_itr()}], 0")
                # self._emit(f"s_mov_b32 s[{s.s_in_stride_k_pack()}], {ca_ce0 * k_pack * data_byte}")
                # for i in range(2, ta_ce1 // k_pack):
                #     self._emit(f"s_mul_i32 s[{s.s_in_offset(i - 2)}], s[{s.s_in_stride_k_pack()}], {i}")
            else:
                self._emit(f"s_mov_b32 s[{s.s_in_offset()}], 0")
                
            # input range set
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_n()}], s[{s.s_in_stride_n()}]")
            self._emit(f"s_lshl_b32 s[{s.s_tmp()}], s[{s.s_tmp()}], {igemm_log2(data_byte)}")
            self._emit(f"s_mov_b32 s[{s.s_p_in(2)}], s[{s.s_tmp()}]")
            #self._emit(f"s_mov_b32 s[{s.s_p_in(2)}], 0xffffffff")
            self._emit(f"s_mov_b32 s[{s.s_p_in(3)}], 0x31014000")
            # compute group distance
            self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_c()}], s[{s.s_in_stride_c()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
            self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
            self._emit(f"s_sub_u32 s[{s.s_p_in(2)}], s[{s.s_p_in(2)}], s[{s.s_tmp()}]")
            self._emit(f"s_add_u32 s[{s.s_p_in(0)}], s[{s.s_p_in(0)}], s[{s.s_tmp()}]")
            self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], s[{s.s_tmp(1)}]")
            self._emit_empty_line()

            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_in_stride_n()}], v[{v.v_in_in()}]")
            # s_in_stride_wi need shift before!
            # self._emit(self.try_shift_stride(s.s_in_stride_c, igemm_log2(data_byte)))

            if self.tunable.gemm_k_global_split:
                self._emit(f"v_add_nc_u32 v[{v.v_tmp(1)}], v[{v.v_tmp(1)}], s[{s.s_block_gtc_ic()}]")

            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(4)}], s[{s.s_in_stride_c()}], v[{v.v_gtc_ic()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wi()}], v[{v.v_in_ihi_list(0)}]")
            self._emit(f"v_add_nc_u32 v[{v.v_tmp()}], v[{v.v_in_iwi_list(0)}], v[{v.v_tmp()}]")
            self._emit(f"v_add_nc_u32 v[{v.v_tmp(4)}], v[{v.v_tmp(4)}], v[{v.v_tmp(1)}]")
            self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(self.tunable.vector_c)}, v[{v.v_tmp()}]")
            self._emit(f"v_add_lshl_u32 v[{v.v_in_os()}], v[{v.v_tmp(4)}], v[{v.v_tmp()}], {igemm_log2(data_byte)}")

            if IGEMM_FWD_GTC_NCHW_PACK_IN_FLAG:
                self._emit(f"v_bfe_u32 v[{v.v_tmp(1)}], v[{v.v_in_flag()}],  16, 1")
                self._emit(m_set_flag_nhw(v.v_tmp(), v.v_tmp(1), v.v_in_ihi_list(0), v.v_in_iwi_list(0), s.s_hi(), s.s_wi()))
                self._emit(f"v_lshl_or_b32 v[{v.v_in_flag()}], v[{v.v_tmp()}], 0,  v[{v.v_in_flag()}]")
            else:
                self._emit(f"v_bfe_u32 v[{v.v_tmp(1)}], v[{v.v_in_flag_n()}],  0, 1")
                self._emit(m_set_flag_nhw(v.v_in_flag(0), v.v_tmp(1), v.v_in_ihi_list(0), v.v_in_iwi_list(0), s.s_hi(), s.s_wi()))
            self._emit_empty_line()

            # voffset, for [1, nb_per_thread) pixels
            if self.tunable.tensor_a_pass_through:
                assert False
            else:
                thread_stride = nb_nb1_vec_c // self.tunable.vector_c

            #print(f"nb_per_thread={nb_per_thread}")

            for i in range(1, nb_per_thread):
                self._emit(f"s_mov_b32 s1, {thread_stride * i}")
                self._emit(f"v_add_nc_u32 v[{v.v_tmp()}], s1, v[{v.v_in_inb()}]")
                self._emit(f"v_add_nc_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_inb()}], v[{v.v_tmp()}]")
                if self.tunable.nxe != 0:
                    if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                        self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080008 ; offset:8, width:8")
                        self._emit(m_mdiv_u32_vs(v.v_tmp(4), v.v_in_in(), v.v_tmp(5), s.s_magic_1(), s.s_tmp(3), s.s_dim_br(), v.v_tmp()))
                        self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080010 ; offset:16, width:8")
                        self._emit(m_mdiv_u32_vs(v.v_in_iwi_list(i), v.v_in_ihi_list(i), v.v_tmp(4), s.s_magic_2(), s.s_tmp(3), s.s_wo(), v.v_tmp()))
                    else:
                        self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_in_in(), v.v_tmp(5), s.s_dim_br(), v.v_tmp(), s.s_tmp()))
                        self._emit(m_int_div_rem_vs(v.v_in_iwi_list(i), v.v_in_ihi_list(i), v.v_tmp(4), s.s_wo(), v.v_tmp(), s.s_tmp()))

                    # ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
                    # iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w
                    self._emit(f"v_mul_lo_u32 v[{v.v_in_ihi_list(i)}], s[{s.s_stride_h()}], v[{v.v_in_ihi_list(i)}]")

                    self._emit(f"v_sub_nc_i32 v[{v.v_in_ihi_list(i)}], v[{v.v_in_ihi_list(i)}], s[{s.s_pad_h()}]")
                    self._emit(f"v_add_nc_u32 v[{v.v_in_ihi_list(i)}], v[{v.v_in_ihi_list(i)}], v[{v.v_sst_a_os()}]")
                    self._emit(f"v_mul_lo_u32 v[{v.v_in_iwi_list(i)}], s[{s.s_stride_w()}], v[{v.v_in_iwi_list(i)}]")

                    self._emit(f"v_sub_nc_i32 v[{v.v_in_iwi_list(i)}], v[{v.v_in_iwi_list(i)}], s[{s.s_pad_w()}]")
                    self._emit(f"v_add_nc_u32 v[{v.v_in_iwi_list(i)}], v[{v.v_in_iwi_list(i)}], v[{v.v_sld_a_os()}]")
                    self._emit_empty_line()

                else:
                    if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                        self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080008 ; offset:8, width:8")
                        self._emit(m_mdiv_u32_vs(v.v_tmp(4), v.v_in_in(), v.v_tmp(5), s.s_magic_1(), s.s_tmp(3), s.s_dim_br(), v.v_tmp()))
                        self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080010 ; offset:16, width:8")
                        self._emit(m_mdiv_u32_vs(v.v_in_iwi_list(i), v.v_in_ihi_list(i), v.v_tmp(4), s.s_magic_2(), s.s_tmp(3), s.s_wi(), v.v_tmp()))
                    else:
                        self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_in_in(), v.v_tmp(5), s.s_dim_br(), v.v_tmp(), s.s_tmp()))
                        self._emit(m_int_div_rem_vs(v.v_in_iwi_list(i), v.v_in_ihi_list(i),  v.v_tmp(4), s.s_wi(), v.v_tmp(), s.s_tmp()))

                self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_in_stride_n()}], v[{v.v_in_in()}]")
                if self.tunable.gemm_k_global_split:
                    self._emit(f"v_add_nc_u32 v[{v.v_tmp(1)}], v[{v.v_tmp(1)}], s[{s.s_block_gtc_ic()}]")
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp(4)}], s[{s.s_in_stride_c()}], v[{v.v_gtc_ic()}]")
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wi()}], v[{v.v_in_ihi_list(i)}]")
                self._emit(f"v_add_nc_u32 v[{v.v_tmp()}], v[{v.v_in_iwi_list(i)}], v[{v.v_tmp()}]")
                self._emit(f"v_add_nc_u32 v[{v.v_tmp(4)}], v[{v.v_tmp(4)}], v[{v.v_tmp(1)}]")
                self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(self.tunable.vector_c)}, v[{v.v_tmp()}]")
                self._emit(f"v_add_lshl_u32 v[{v.v_in_os(i)}], v[{v.v_tmp(4)}], v[{v.v_tmp()}], {igemm_log2(data_byte)}")

                if IGEMM_FWD_GTC_NCHW_PACK_IN_FLAG:
                    # update flag for batch size
                    self._emit(f"v_cmp_gt_u32  s[{s.s_n()}], v[{v.v_in_in()}]")
                    self._emit(f"v_cndmask_b32 v[{v.v_tmp()}], 0, 1")

                    self._emit(f"v_lshl_or_b32 v[{v.v_in_flag()}], v[{v.v_tmp()}], {16 + i}, v[{v.v_in_flag(0)}]")
                    self._emit(m_set_flag_nhw(v.v_tmp(1), v.v_tmp(), v.v_in_ihi_list(i), v.v_in_iwi_list(i), s.s_hi(), s.s_wi()))
                    self._emit(f"v_lshl_or_b32 v[{v.v_in_flag()}], v[{v.v_tmp(1)}], {i}, v[{v.v_in_flag()}]")
                else:
                    self._emit(f"v_cmp_gt_u32  s[{s.s_n()}], v[{v.v_in_in()}]")
                    self._emit(f"v_cndmask_b32 v[{v.v_tmp()}], 0, 1")

                    self._emit(f"v_lshl_or_b32 v[{v.v_in_flag_n()}], v[{v.v_tmp()}], {i}, v[{v.v_in_flag_n()}]")
                    self._emit(m_set_flag_nhw(v.v_in_flag(i), v.v_tmp(), v.v_in_ihi_list(i), v.v_in_iwi_list(i), s.s_hi(), s.s_wi()))

            # load in
            if self.tunable.tensor_a_pass_through and self.tunable.tensor_a_pass_through_interleave_gld:
                mbb_gld_in = create_machine_basic_block(self.global_load_in())
                gld_per_k = self.tunable.wave_repeat_m * self.tunable.wave_step_m
                for i_mbb in mbb_gld_in[0:(-1 * gld_per_k)]:
                    # TODO: need multiple load of pass through side
                    self._emit(machine_basic_block_call(self, i_mbb))
            else:
                self._emit(self.global_load_in())
            self._emit_empty_line()

        def calculate_and_load_weight():
            self._emit(f"; calculate wei offset")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_wei_stride_x()}], s[{s.s_knum()}]")
            self._emit(f"s_lshl_b32 s[{s.s_tmp()}], s[{s.s_tmp()}], {igemm_log2(data_byte)}")
            self._emit(f"s_mov_b32 s[{s.s_p_wei(2)}], s[{s.s_tmp()}]")
            self._emit(f"s_mov_b32 s[{s.s_p_wei(3)}], 0x31014000")
            if self.tunable.nxe != 0:
                self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_x()}], s[{s.s_y()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_k()}], {self.tunable.vector_c}")
            self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], s[{s.s_c()}]")
            if self.tunable.nxe != 0:
                self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], s[{s.s_tmp()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
            self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
            self._emit(f"s_sub_u32 s[{s.s_p_wei(2)}], s[{s.s_p_wei(2)}], s[{s.s_tmp()}]")
            self._emit(f"s_add_u32 s[{s.s_p_wei()}], s[{s.s_p_wei()}], s[{s.s_tmp()}]")
            self._emit(f"s_addc_u32 s[{s.s_p_wei(1)}], s[{s.s_p_wei(1)}], s[{s.s_tmp(1)}]")

            self._emit(f"v_add_nc_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_ik()}], v[{v.v_wei_ik()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_x()}], v[{v.v_gtc_iec()}]")
            if self.tunable.gemm_k_global_split:
                self._emit(f"v_add_nc_u32 v[{v.v_tmp()}], v[{v.v_tmp()}], s[{s.s_block_gtc_ic()}]")
            self._emit(f"v_lshlrev_b32 v[{v.v_tmp(4)}], {igemm_log2(self.tunable.vector_c)}, v[{v.v_tmp(5)}]")
            self._emit(f"v_add_lshl_u32 v[{v.v_wei_os()}], v[{v.v_tmp()}], v[{v.v_tmp(4)}], {igemm_log2(data_byte)}")

            # wei flag
            #self._emit(f"v_cmp_gt_u32  s[{s.s_k()}], v[{v.v_tmp(5)}]")
            #self._emit(f"v_cndmask_b32 v[{v.v_wei_flag()}], 0, 1")
            #self._emit(f"v_mov_b32 v[{v.v_wei_tmp_pack()}], v[{v.v_wei_flag()}]")

            for i in range(1, nk_per_thread):
                if i == 1:
                    k_thread_stride = ca_k
                    self._emit(f"s_mov_b32 s[{s.s_tmp()}], {k_thread_stride}")
                self._emit(f"v_add_nc_u32 v[{v.v_tmp(5)}], s[{s.s_tmp()}], v[{v.v_tmp(5)}]")
                self._emit(f"v_cmp_gt_u32  s[{s.s_k()}], v[{v.v_tmp(5)}]")
                #self._emit(f"v_cndmask_b32 v[{v.v_wei_flag(i)}], 0, 1")
                #self._emit(f"v_lshl_or_b32 v[{v.v_wei_tmp_pack()}], v[{v.v_wei_flag(i)}], {i}, v[{v.v_wei_tmp_pack()}]")

            self._emit_empty_line()
            if self.wei_thread_copy_ndim != 1:
                if s_wei_stride_d0 != s_dummy:
                    self._emit(self.try_shift_stride(s_wei_stride_d0, igemm_log2(data_byte)))
            if s_wei_stride_d1 != s_dummy:
                self._emit(self.try_shift_stride(s_wei_stride_d1, igemm_log2(data_byte)))
            self._emit_empty_line()

            # cyxkc layout do not need s_wei_offset
            # if self.tunable.precache_soffset:
            #     self._emit(m_wei_2d_global_load.init_precache_soffset(s_wei_stride_d0(), s_wei_stride_d1(), s.s_wei_offset(), s.s_tmp()))

            self._emit(f".v_clear_nc {v.v_gld_a()}, {self.get_num_vgpr_global_load_a()}")
            
            if self.tunable.tensor_b_pass_through and self.tunable.tensor_b_pass_through_interleave_gld:
                mbb_gld_wei = create_machine_basic_block(self.global_load_wei())
                gld_per_k = self.tunable.wave_repeat_n * self.tunable.wave_step_n
                for i_mbb in mbb_gld_wei[0:(-1 * gld_per_k)]:
                    # TODO: need multiple load of pass through side
                    self._emit(machine_basic_block_call(self, i_mbb))
            else:
                self._emit(self.global_load_wei())
            self._emit_empty_line()

        # do load
        calculate_and_load_weight()
        calculate_and_load_input()

        if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            v_pack = k_pack if self.tunable.tensor_a_pass_through or self.tunable.tensor_b_pass_through else 1
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], v0")
            self._emit(self.dotx_mapping.get_gemm_index_for_src_matrix(v.v_gemm_in(), v.v_gemm_im(), v.v_tmp(5), v.v_tmp(),
                                    k_pack=k_pack_src_mat, v_pack=v_pack))
            #self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], v0")
            #self._emit(self.dotx_mapping.get_gemm_index_for_dst_matrix(v.v_co_sst(), v.v_co_sld(), v.v_tmp(5), v.v_tmp()))
        else:
            v_pack = k_pack if self.tunable.tensor_a_pass_through or self.tunable.tensor_b_pass_through else 1
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], v0")
            self._emit(self.xdlops_mapping.get_gemm_index_for_src_matrix(v.v_gemm_in(), v.v_gemm_im(), v.v_tmp(5), v.v_tmp(),
                                    k_pack=k_pack_src_mat, v_pack=v_pack))
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], v0")
            self._emit(self.xdlops_mapping.get_gemm_index_for_dst_matrix(v.v_co_sst(), v.v_co_sld(), v.v_tmp(5), v.v_tmp()))

        '''
        gemm_k * gemm_m * k_pack
        '''
        v_igemm_k = v.v_gtc_iec
        if not self.tunable.tensor_a_pass_through:
            self._emit(f"; LDS store, wei: 1,ce,1,k: {1}x{1}x{1}x{ta_k_vec_c}, {1}x{ca_ce}x{1}x{ca_k}, k_pack:{k_pack}, k_pack_gld_a:{k_pack_gld_a}, {self.tunable.precision}")
            if k_pack_src_mat != 1:
                self._emit(f"v_lshlrev_b32 v[{v.v_tmp(2)}], {igemm_log2(k_pack_src_mat)}, v[{v.v_wei_ik()}]")
                #self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v_igemm_k()}], {igemm_log2(na_k_vec_c)}, v[{v.v_tmp(2)}]")
                self._emit(f"v_mad_u32_u24 v[{v.v_tmp()}], v[{v_igemm_k()}], {na_k_vec_c}, v[{v.v_tmp(2)}]")
                if k_pack_src_mat != k_pack_gld_a:
                    assert k_pack_src_mat % k_pack_gld_a == 0
                    self._emit(f"v_and_b32 v[{v.v_tmp(2)}], {k_pack_src_mat - 1}, v[{v_igemm_k()}]")   # gld_a k_pack_src_mat smaller than k_pack_src_mat
                    self._emit(f"v_or_b32 v[{v.v_tmp()}], v[{v.v_tmp()}], v[{v.v_tmp(2)}]")
            else:
                self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v_igemm_k()}], {igemm_log2(na_k0*na_k1 * k_pack_src_mat)}, v[{v.v_in_inb()}]")
            self._emit(f"v_lshlrev_b32 v[{v.v_sst_a_os()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")
            self._emit_empty_line()
            self._emit(f"v_lshlrev_b32 v[{v.v_sld_a_os()}], {igemm_log2(data_byte * self.tunable.vector_c)}, v[{v.v_gemm_im()}] ; LDS load in")

        if not self.tunable.tensor_b_pass_through:
            self._emit(f"; LDS store, input: 1,ce,nb_vec_c: {1}x{1}x{tb_nb0}x{tb_nb_vec_c}, {1}x{cb_ce}x{1}x{cb_nb1}, k_pack:{k_pack_src_mat}, k_pack_gld_b:{k_pack_gld_b}, {self.tunable.precision}")
            if k_pack_src_mat != 1:
                self._emit(f"v_lshlrev_b32 v[{v.v_tmp(2)}], {igemm_log2(k_pack_src_mat)},  v[{v.v_in_inb()}]")
                if igemm_is_pow2(nb_nb0*nb_nb1_vec_c):
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v_igemm_k()}], {igemm_log2(nb_nb0*nb_nb1_vec_c)}, v[{v.v_tmp(2)}]")
                else:
                    self._emit(f"v_mad_u32_u24 v[{v.v_tmp()}], v[{v_igemm_k()}], {nb_nb0*nb_nb1_vec_c}, v[{v.v_tmp(2)}]")
                if k_pack_src_mat != k_pack_gld_b:
                    assert k_pack_src_mat % k_pack_gld_b == 0
                    self._emit(f"v_and_b32 v[{v.v_tmp(2)}], {k_pack_src_mat - 1}, v[{v_igemm_k()}]")   # gld_b k_pack_src_mat smaller than k_pack_src_mat
                    self._emit(f"v_or_b32 v[{v.v_tmp()}], v[{v.v_tmp()}], v[{v.v_tmp(2)}]")
            else:
                if igemm_is_pow2(nb_nb0*nb_nb1_vec_c):
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v_igemm_k()}], {igemm_log2(nb_nb0*nb_nb1_vec_c)}, v[{v.v_in_inb()}]")
                else:
                    self._emit(f"v_mad_u32_u24 v[{v.v_tmp()}], v[{v_igemm_k()}], {nb_nb0*nb_nb1_vec_c}, v[{v.v_in_inb()}]")
            self._emit(f"v_lshlrev_b32 v[{v.v_sst_b_os()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")
            if not self.tunable.tensor_a_pass_through:
                self._emit(f"v_add_nc_u32 v[{v.v_sst_b_os()}], {self.tunable.lds_a_np2}, v[{v.v_sst_b_os()}]")
            self._emit_empty_line()
            self._emit(f"v_lshlrev_b32 v[{v.v_sld_b_os()}], {igemm_log2(data_byte * self.tunable.vector_c)}, v[{v.v_gemm_in()}] ; LDS load wei")
            if not self.tunable.tensor_a_pass_through:
                self._emit(f"v_add_nc_u32 v[{v.v_sld_b_os()}], {self.tunable.lds_a_np2}, v[{v.v_sld_b_os()}]")

        self._emit(self.coalescing_store.init_co_lds_offset(v.v_co_sst(), v.v_co_sld(), v.v_gemm_im(), v.v_gemm_in(), '0', v.v_tmp()))
        self._emit(self.coalescing_store.init_co_sub_m_index(v.v_co_sub_m_index(), '0', v.v_tmp()))
        self._emit(self.coalescing_store.init_co_sub_n_index(v.v_co_sub_n_index(), '0', v.v_tmp()))
        self._emit_empty_line()

        '''
        a good news for nchw and coalescing output is that, we can treat gemm_m (n*ho*wo) as a single dimension,
        and use sgpr to stride along this dimension. this is much easier
        '''
        self._emit(f"; output offset")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(3)}], s[{s.s_k()}], {igemm_log2(self.tunable.vector_c)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp(3)}], s[{s.s_block_gtc_ig()}],s[{s.s_tmp(3)}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_tmp(3)}], s[{s.s_out_stride_k()}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_tmp(3)}], s[{s.s_out_stride_k()}]")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], s[{s.s_tmp(1)}]")

        self._emit_empty_line()
        self._emit(f"s_lshr_b32 s[{s.s_tmp(1)}], s[{s.s_block_gtc_ik()}], {igemm_log2(self.tunable.vector_c)}")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], s[{s.s_tmp(1)}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_tmp(3)}], s[{s.s_out_stride_k()}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_tmp(3)}], s[{s.s_out_stride_k()}]")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out()}+1], s[{s.s_tmp(1)}]")
        self._emit_empty_line()

        #self._emit(self.try_shift_stride(s.s_out_stride_wo, igemm_log2(data_byte)))
        self._emit(f"v_add_nc_u32 v[{v.v_out_inb()}], s[{s.s_block_gtc_inb()}], v[{v.v_co_sub_n_index()}]   ; total n*ho*wo")
        
        #set output flag
        self._emit(f"v_cmp_gt_u32  s[{s.s_dim_nr()}], v[{v.v_out_inb()}]")
        self._emit(f"v_cndmask_b32 v[{v.v_out_flag()}], 0, 1")
        
        self._emit(f";   compute from n1b")
        self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080008 ; offset:8, width:8")
        self._emit(m_mdiv_u32_vs(v.v_tmp(4), v.v_out_in(), v.v_out_inb(), s.s_magic_1(), s.s_tmp(3), s.s_dim_br(), v.v_tmp()))

        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], v[{v.v_out_in()}], s[{s.s_out_stride_n()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], 1, v[{v.v_tmp()}]")
        self._emit(f"v_lshl_add_u32 v[{v.v_out_os()}], v[{v.v_tmp(4)}], {igemm_log2(data_byte * self.tunable.vector_c)}, v[{v.v_tmp()}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_out_stride_k()}], v[{v.v_co_sub_m_index()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")
        self._emit(f"v_add_nc_u32 v[{v.v_out_os()}], v[{v.v_out_os()}], v[{v.v_tmp()}]")
        
        self._emit(f";    mask for coaleascing store")
        self._emit(f"v_mov_b32 v[{v.v_coalescing_store_index()}], v[0]")

        self._emit(f"; move slice stride")

        w_flag_cnt = 0
        #self._emit(f"v_bfe_u32 v[{v.v_wei_flag(0)}], v[{v.v_wei_tmp_pack()}], {0}, 1")
        w_flag_cnt = w_flag_cnt + 1

        if self.tunable.nxe != 0:
            pass
        else:
            self._emit(f"s_mul_i32 s[{s.s_move_slice_k_stride_c()}], s[{s.s_in_stride_c()}], {data_byte * na_ce}")
            
        self._emit(f"s_lshl_b32 s[{s.s_move_slice_k_stride_gemm_k()}], s[{s.s_k()}], {igemm_log2(self.tunable.gemm_k_per_block * data_byte)}")

        #if w_flag_cnt < nk_per_thread:
        #    self._emit(f"v_bfe_u32 v[{v.v_wei_flag(w_flag_cnt)}], v[{v.v_wei_tmp_pack()}], {w_flag_cnt}, 1")
        #    w_flag_cnt = w_flag_cnt + 1

        if self.tunable.nxe != 0:
            # s_diff_in_os_acc_c_y_x   : s_move_slice_k_c * data_byte + s_move_slice_k_x * s_dilation_w * in_stride_wi + s_move_slice_k_y * s_dilation_h * in_stride_hi
            # s_diff_in_os_ovf_y_acc_c : -s_c * data_byte + s_dilation_w * in_stride_wi
            # s_diff_in_os_ovf_x_acc_y : -s_x * s_dilation_w * in_stride_wi + s_dilation_h * in_stride_hi
            # s_diff_in_iwi_acc_x      : s_move_slice_k_x * s_dilation_w
            # s_diff_in_iwi_ovf_x      : s_x * s_dilation_w
            # s_diff_in_ihi_acc_y      : s_move_slice_k_y * s_dilation_h
            self._emit(f"s_mul_i32 s[{s.s_x_dilation_w()}], s[{s.s_x()}], s[{s.s_dilation_w()}]")
            self._emit(f"s_mul_i32 s[{s.s_y_dilation_h()}], s[{s.s_y()}], s[{s.s_dilation_h()}]")
            self._emit(f"s_mul_i32 s[{s.s_x_dilation_w()}], -1, s[{s.s_x_dilation_w()}]")
            self._emit(f"s_mul_i32 s[{s.s_y_dilation_h()}], -1, s[{s.s_y_dilation_h()}]")
            self._emit(f"s_lshl_b32 s[{s.s_tmp(5)}], s[{s.s_wi()}], {igemm_log2(self.tunable.vector_c * data_byte)}")                # in_stride_hi
            self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_in_stride_c()}], {igemm_log2(data_byte)}")        # s_dilation_w * in_stride_wi
            self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_y()}], s[{s.s_tmp(5)}]")
            self._emit(f"s_sub_i32 s[{s.s_diff_in_os_ovf_y_acc_c()}], s[{s.s_tmp(0)}], s[{s.s_tmp(2)}]")
            self._emit(f"s_mul_i32 s[{s.s_diff_in_iwi_acc_x()}], s[{s.s_move_slice_k_x()}], s[{s.s_dilation_w()}]")
            self._emit(f"s_mul_i32 s[{s.s_diff_in_iwi_ovf_x()}], s[{s.s_x()}], s[{s.s_dilation_w()}]")
            self._emit(f"s_mul_i32 s[{s.s_diff_in_ihi_acc_y()}], s[{s.s_move_slice_k_y()}], s[{s.s_dilation_h()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp(5)}], s[{s.s_tmp(5)}], s[{s.s_dilation_h()}]")                # s_dilation_h * in_stride_hi
            self._emit(f"s_lshl_b32 s[{s.s_tmp(2)}], s[{s.s_move_slice_k_x()}], {igemm_log2(data_byte * self.tunable.vector_c)}")            # s_move_slice_k_x * s_dilation_w * in_stride_wi
            self._emit(f"s_mul_i32 s[{s.s_tmp(3)}], s[{s.s_move_slice_k_c()}], s[{s.s_in_stride_c()}]")   # s_move_slice_k_c * data_byte 
            self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_diff_in_ihi_acc_y()}], s[{s.s_tmp(5)}]")         # s_move_slice_k_y * s_dilation_h * in_stride_hi
            self._emit(f"s_lshl_b32 s[{s.s_tmp(1)}], s[{s.s_tmp(3)}], {igemm_log2(data_byte)}")
            self._emit(f"s_add_u32 s[{s.s_diff_in_os_acc_c_y_x()}], s[{s.s_tmp(0)}], s[{s.s_tmp(1)}]")
            self._emit(f"s_add_u32 s[{s.s_diff_in_os_acc_c_y_x()}], s[{s.s_diff_in_os_acc_c_y_x()}], s[{s.s_tmp(2)}]")
            self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_diff_in_iwi_ovf_x()}], {igemm_log2(self.tunable.vector_c * data_byte)}") # s_x * s_dilation_w * in_stride_wi
            self._emit(f"s_sub_i32 s[{s.s_diff_in_os_ovf_x_acc_y()}], s[{s.s_tmp(5)}], s[{s.s_tmp(0)}]")
            
            self._emit(f"s_mul_i32 s[{s.s_y_x_c()}], s[{s.s_x()}], s[{s.s_y()}]")
            self._emit(f"s_mul_i32 s[{s.s_y_x_c()}], s[{s.s_y_x_c()}], s[{s.s_c()}]")
            self._emit(f"s_lshl_b32 s[{s.s_y_x_c()}], s[{s.s_y_x_c()}], {igemm_log2(data_byte)}")

            self._emit(f"v_add_nc_u32 v[{v.v_gtc_ix()}], v[{v.v_gtc_ix()}], s[{s.s_x_dilation_w()}]")
            self._emit(f"v_add_nc_u32 v[{v.v_gtc_iy()}], v[{v.v_gtc_iy()}], s[{s.s_y_dilation_h()}]")

        self._emit_empty_line()

        self._emit(f"s_mov_b32 s[{s.s_p_out(2)}], 0xffffffff")
        #if w_flag_cnt < nk_per_thread:
        #    self._emit(f"v_bfe_u32 v[{v.v_wei_flag(w_flag_cnt)}], v[{v.v_wei_tmp_pack()}], {w_flag_cnt}, 1")
        #    w_flag_cnt = w_flag_cnt + 1
        self._emit(f"s_mov_b32 s[{s.s_p_out(3)}], 0x31014000")
        #for i_w in range(w_flag_cnt, nk_per_thread):
        #    self._emit(f"v_bfe_u32 v[{v.v_wei_flag(i_w)}], v[{v.v_wei_tmp_pack()}], {i_w}, 1")
                
        # pad gemmk
        self._emit(f"s_add_i32 s[{s.s_knum()}], s[{s.s_knum()}], {self.tunable.gemm_k_per_block - 1}")
        self._emit(f"s_lshr_b32 s[{s.s_knum()}], s[{s.s_knum()}], {igemm_log2(self.tunable.gemm_k_per_block)}")
        self._emit(f"s_lshl_b32 s[{s.s_knum()}], s[{s.s_knum()}], {igemm_log2(self.tunable.gemm_k_per_block)}")
        self._emit_empty_line()

    def emit_kernel_fma_main_loop(self):
        s = self.sgpr
        v = self.vgpr
        k = self.karg

        data_byte = amdgpu_precision_data_byte(self.tunable.precision)
        k_pack = self.get_k_pack()
        #k_pack_lanegroup = self.xdlops_mapping.ctrl.lanegroup_k_per_thread()
        k_pack_src_mat = k_pack #if k_pack != 1 else k_pack_lanegroup

        m_move_slice_window             = self.get_macro_move_slice_window()

        def move_slice_window_b():
            '''
            in nchw we only need call one move slice window
            '''
            if self.tunable.nxe != 0:
                with self._deferred_context():
                    self._emit(m_move_slice_window(
                                v.v_gtc_iy(), v.v_gtc_ix(), v.v_gtc_ic(), v.v_tmp(), v.v_tmp(1), v.v_tmp(2), v.v_tmp(3), 
                                v.v_in_os(), v.v_in_ihi_list(), v.v_in_iwi_list(), v.v_in_flag(), v.v_in_flag_n(),
                                v.v_wei_os(),
                                s.s_x_dilation_w(), s.s_y_dilation_h(), s.s_move_slice_k_y(), s.s_move_slice_k_x(), s.s_move_slice_k_c(),
                                s.s_move_slice_k_stride_gemm_k(),
                                s.s_diff_in_os_acc_c_y_x(), s.s_diff_in_os_ovf_y_acc_c(),
                                s.s_diff_in_os_ovf_x_acc_y(), s.s_diff_in_iwi_acc_x(),
                                s.s_dilation_w(),
                                s.s_diff_in_iwi_ovf_x(), s.s_diff_in_ihi_acc_y(),
                                s.s_dilation_h(),
                                v.v_tmp(4), v.v_tmp(5), v.v_gtc_iy(),       # v.v_in_os_diff(), v.v_in_ihi_diff(), v.v_in_iwi_diff(),
                                s.s_x(), s.s_c(), s.s_hi(), s.s_wi()))
                return self._get_deferred()
            else:
                with self._deferred_context():
                    self._emit(m_move_slice_window(
                                s.s_p_in() if self.tunable.tensor_a_pass_through else s.s_in_offset(),
                                v.v_wei_os(),
                                s.s_move_slice_k_stride_c(),
                                s.s_move_slice_k_stride_gemm_k()))
                return self._get_deferred()

        def move_slice_window_a():
            return ''

        def move_slice_window_acc():
            return ''

        if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            # TODO: reopen legacy fma instruction
            if hasattr(self.tunable, 'thread_tile_m'):
                fctrl                             = ctrl_fma_main_loop_t()
                fctrl.thread_m                    = self.tunable.thread_tile_m
                fctrl.thread_n                    = self.tunable.thread_tile_n
                fctrl.unroll_k                    = self.tunable.gemm_k_per_block // self.tunable.vector_c
                fctrl.label_prefix                = self.name()
                fctrl.gemm_m_repeat               = self.tunable.gemm_m_repeat
                fctrl.gemm_m_level0_cluster       = self.tunable.gemm_m_level0_cluster
                fctrl.gemm_m_level1_cluster       = self.tunable.gemm_m_level1_cluster
                fctrl.gemm_n_repeat               = self.tunable.gemm_n_repeat
                fctrl.gemm_n_level0_cluster       = self.tunable.gemm_n_level0_cluster
                fctrl.gemm_n_level1_cluster       = self.tunable.gemm_n_level1_cluster
                fctrl.lds_single_size             = self.tunable.lds_single            # in byte, should be power of 2
                fctrl.lds_buffer_num              = self.tunable.lds_buffer_num
                fctrl.precision                   = self.tunable.precision

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
                fctrl                             = ctrl_dotx_main_loop_t()
                ctrl_dotx_mapping                 = get_ctrl_dotx_mapping_from_wave_tile(self.tunable.gemm_m_per_block, self.tunable.gemm_n_per_block,
                                                                        self.tunable.lanegroup_tile_m, self.tunable.lanegroup_tile_n,
                                                                        self.tunable.lanegroup_wave_m, self.tunable.lanegroup_wave_n,
                                                                        self.tunable.block_size // (self.tunable.lanegroup_wave_m * self.tunable.lanegroup_wave_n * LANEGROUP_SIZE),
                                                                        self.tunable.lanegroup_repeat_m, self.tunable.lanegroup_repeat_n,
                                                                        self.tunable.precision)
                fctrl.dotx_m                      = ctrl_dotx_mapping
                fctrl.unroll_k                    = self.tunable.gemm_k_per_block
                fctrl.label_prefix                = self.name()
                fctrl.lds_single_size             = self.tunable.lds_single            # in byte, should be power of 2
                fctrl.lds_buffer_num              = self.tunable.lds_buffer_num
                fctrl.precision                   = self.tunable.precision
                fctrl.local_prefetch_num          = self.tunable.local_prefetch_num
                fctrl.local_prefetch_num_m        = self.tunable.local_prefetch_num_m

                fctrl.lds_k_pack                  = self.tunable.vector_c

                # functor
                fctrl.global_load_a_functor       = self.global_load_wei
                fctrl.global_load_b_functor       = self.global_load_in
                fctrl.shared_store_a_functor      = self.shared_store_wei
                fctrl.shared_store_b_functor      = self.shared_store_in
                fctrl.shared_load_a_functor       = inst_ds_read_t(data_byte * self.tunable.vector_c)
                fctrl.shared_load_b_functor       = inst_ds_read_t(data_byte * self.tunable.vector_c)
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

                fma_main_loop = dotx_main_loop_t(self.mc, fctrl)
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
            fctrl.accvgpr_unified             = IGEMM_FWD_GTC_NCHW_ACCVGPR_UNIFIED and self.mc.arch_config.arch == AMDGPU_ARCH_GFX90A

            # functor
            # fctrl.global_load_a_functor       = self.global_load_wei
            # fctrl.global_load_b_functor       = self.global_load_in
            # fctrl.shared_store_a_functor      = self.shared_store_wei
            # fctrl.shared_store_b_functor      = self.shared_store_in
            fctrl.global_load_a_functor       = self.global_load_in
            fctrl.global_load_b_functor       = self.global_load_wei
            fctrl.shared_store_a_functor      = self.shared_store_in
            fctrl.shared_store_b_functor      = self.shared_store_wei

            # ta_k0, ta_k1, ta_ce0, ta_ce1, tb_ce0, tb_ce1, tb_nb0, tb_nb1 = self.get_thread_lengths()
            fctrl.lds_k_pack                  = k_pack_src_mat

            share_load_packed                 = k_pack if self.tunable.tensor_a_pass_through or self.tunable.tensor_b_pass_through else ctrl_xdlops_mapping.lanegroup_k_per_thread()

            if ctrl_xdlops_mapping.wave_step_m == 1:
                fctrl.shared_load_a_functor   = inst_ds_read_t(data_byte * share_load_packed)   # xdlops load from LDS always single load
            else:
                assert ctrl_xdlops_mapping.wave_step_m == 2, "currently only support wave_step_m is 2"
                fctrl.shared_load_a_functor   = inst_ds_read2_likely_accumulate_offset_t(self.mc, 2, data_byte * share_load_packed, k_pack*ctrl_xdlops_mapping.wave_tile_m * data_byte, sym_t(self.vgpr.v_tmp(4)))

            if ctrl_xdlops_mapping.wave_step_n == 1:
                fctrl.shared_load_b_functor   = inst_ds_read_t(data_byte * share_load_packed)   # xdlops load from LDS always single load
            else:
                assert ctrl_xdlops_mapping.wave_step_n == 2, "currently only support wave_step_n is 2"
                fctrl.shared_load_b_functor   = inst_ds_read2_likely_accumulate_offset_t(self.mc, 2, data_byte * share_load_packed, k_pack*ctrl_xdlops_mapping.wave_tile_n * data_byte, sym_t(self.vgpr.v_tmp(5)))
            fctrl.move_slice_window_a_functor = move_slice_window_a
            fctrl.move_slice_window_b_functor = move_slice_window_b
            fctrl.move_slice_window_accumule_functor  = None

            # sympol type
            fctrl.v_a                         = v.v_a   if not self.tunable.tensor_a_pass_through else None
            fctrl.v_b                         = v.v_b   if not self.tunable.tensor_b_pass_through else None
            fctrl.a_c                         = a.a_c
            fctrl.v_gld_a                     = v.v_gld_a
            fctrl.v_gld_b                     = v.v_gld_b
            fctrl.v_gld_a_gpf                 = v.v_gld_a_gpf if self.tunable.global_prefetch_a_num == 2 else None
            fctrl.v_gld_b_gpf                 = v.v_gld_b_gpf if self.tunable.global_prefetch_b_num == 2 else None
            fctrl.v_gld_a_num                 = self.get_num_vgpr_global_load_a()
            fctrl.v_gld_b_num                 = self.get_num_vgpr_global_load_b()
            fctrl.v_sld_a_os                  = v.v_sld_a_os  if not self.tunable.tensor_a_pass_through else None
            fctrl.v_sld_b_os                  = v.v_sld_b_os  if not self.tunable.tensor_b_pass_through else None
            fctrl.v_sst_a_os                  = v.v_sst_a_os  if not self.tunable.tensor_a_pass_through else None
            fctrl.v_sst_b_os                  = v.v_sst_b_os  if not self.tunable.tensor_b_pass_through else None
            fctrl.s_kitr                      = s.s_kitr
            fctrl.s_knum                      = s.s_knum
            fctrl.pass_through_a              = self.tunable.tensor_a_pass_through
            fctrl.pass_through_b              = self.tunable.tensor_b_pass_through
            fctrl.pass_through_a_v_pack       = self.get_k_pack()
            fctrl.pass_through_b_v_pack       = self.get_k_pack()

            fctrl.pass_through_a_interleave_gld = 1 if self.tunable.tensor_a_pass_through_interleave_gld else 0
            fctrl.pass_through_b_interleave_gld = 1 if self.tunable.tensor_b_pass_through_interleave_gld else 0

            fctrl.precision                   = self.tunable.precision

            mfma_main_loop = mfma_main_loop_t(self.mc, fctrl)
            mfma_main_loop.emit()


    def emit_kernel_epilogue(self):
        s = self.sgpr
        v = self.vgpr
        #label_out = f"L_{self.name()}_out"

        ta_k_vec_c, tb_nb0, tb_nb_vec_c = self.get_thread_lengths()
        ca_k, ca_ce, cb_ce, cb_nb1 = self.get_cluster_lengths()

        if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            self._emit(self.coalescing_store(v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_out(), v.v_out_os(), None,
                None, s.s_out_stride_k(), s.s_tmp(), v.v_out_flag(), s.s_k(), v.v_out_ik(), s.s_block_gtc_ik(), v.v_co_sub_m_index(), v.v_tmp()))

        else:
            a = self.agpr
            self._emit(self.coalescing_store(a.a_c(), v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_out(), v.v_out_os(), None,
                     None, s.s_out_stride_wo(),
                     s.s_tmp(), v.v_out_flag() if self.tunable.nxe != 0 else v.v_out_flag(), s.s_dim_mr(), v.v_out_inb(), s.s_block_gtc_inb(), v.v_co_sub_m_index(), v.v_tmp()))

        if IGEMM_FWD_GTC_NCHWC_DEBUG == 1:
            self._emit_empty_line()
            self._emit(f"s_branch {self.label_out}")
            self._emit("; debug code to cpy vgpr to host")
            if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
                self._emit(f"L_debug_{self.label_out}:")
            else: 
                self._emit(f"L_debug_{self.label_out}_1:")
            self._emit("s_nop 256")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_waitcnt vmcnt(0)")
            self._emit("s_barrier")
            self._emit(f"s_cmp_lg_u32 s[{s.s_dbg(2)}], 0")
            if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
                self._emit(f"s_cbranch_scc1 L_program_end_{self.label_out}")
            else: 
                self._emit(f"s_cbranch_scc1 L_program_end_{self.label_out}_1")
            self._emit(f"s_cmp_lg_u32 s[{s.s_dbg(3)}], 0")
            if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
                self._emit(f"s_cbranch_scc1 L_program_end_{self.label_out}")
            else: 
                self._emit(f"s_cbranch_scc1 L_program_end_{self.label_out}_1")
            self._emit("v_mov_b32 v[v_tmp], s[s_in_offset]")
            self._emit(f"s_mov_b32 s[{s.s_tmp()}], 0")
            self._emit(f"s_mov_b32 s[{s.s_p_out()}], s[{s.s_dbg()}]")
            self._emit(f"s_mov_b32 s[{s.s_p_out(1)}], s[{s.s_dbg(1)}]")
            self._emit(f"s_mov_b32 s[{s.s_p_out(2)}], 0x80000000")
            self._emit(f"s_mov_b32 s[{s.s_p_out(3)}], 0x31014000")
            self._emit_empty_line()

            self._emit(f"buffer_store_dword v[v_in_os], v[{v.v_dbg()}], s[{s.s_p_out((0,3))}], s[{s.s_tmp()}] offen")

            self._emit("s_waitcnt vmcnt(0)")
            self._emit("s_barrier")
            self._emit_empty_line()

            if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
                self._emit(f"L_program_end_{self.label_out}:")
            else: 
                self._emit(f"L_program_end_{self.label_out}_1:")
            self._emit("s_nop 2")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_waitcnt vmcnt(0)")
            self._emit("s_barrier")

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
