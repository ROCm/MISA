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


IGEMM_FWD_GTC_NHWC_PACK_IN_FLAG = 0
# IGEMM_FWD_GTC_NHWC_P_INTERLEAVE_GLD = False     # p tensor interleave
IGEMM_FWD_GTC_NHWC_ACCVGPR_UNIFIED = True   # used in gfx90a
IGEMM_FWD_GTC_NHWC_USE_BF16_1K_IN_FP16  = True   # used in gfx90a

def _find_non_1_index_in_list(list_object):
    result_list = list()
    for idx, item in enumerate(list_object):
        assert type(item) is int
        if item != 1:
            result_list.append(idx)
    return result_list

class igemm_fwd_gtc_nhwc_t(mc_base_t):
    '''
                      tensor a (input)                   tensor b (wei)
    thread_lengths  : ta_e, ta_c, ta_nb0, ta_nb1,     tb_e, tb_c, tb_k0, tb_k1
    cluster_lengths : ca_e, ca_c, ca_nb0, ca_nb1,     cb_e, cb_c, cb_k0, cb_k1

    for a/b tensor, always load gemm_k dimension first.

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

        if tunable.merge_e == 1:
            assert tunable.nxe != 0, f"there is no meaning if we merge c*y*x{tunable.merge_e} but have a special 1x1 case"
            assert not tunable.tensor_a_pass_through, "currently not support pass through a while do merge e"

        self.coalescing_store_groups = igemm_next_pow2(self.tunable.coalescing_store_groups)
        if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS and not igemm_use_lanegroup_thread_mapping(self.tunable):
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
            na_c0, na_c1e, na_k0, na_k1, nb_c0, nb_c1e, nb_n0, nb_n1b = self.get_dims_lengths()
            ctrl_coalescing_store.gemm_m_m0_m1 = [na_k0, na_k1]
            if gemm_m_order == IGEMM_FWD_GTC_LDS_STORE_ORDER_GEMM_M_K1_K0:
                ctrl_coalescing_store.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M1_M0

            ctrl_coalescing_store.adjust_optimal_coalescing_groups()        # in m1_m0 order, must adjust 
            self.coalescing_store = igemm_coalescing_store_t(mc, ctrl_coalescing_store)

        elif self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS and igemm_use_lanegroup_thread_mapping(self.tunable):
            ctrl_dotx_mapping = get_ctrl_dotx_mapping_from_wave_tile(self.tunable.gemm_m_per_block, self.tunable.gemm_n_per_block,
                                    self.tunable.lanegroup_tile_m, self.tunable.lanegroup_tile_n, self.tunable.lanegroup_wave_m, self.tunable.lanegroup_wave_n,
                                    self.tunable.block_size // self.tunable.wave_size, self.tunable.lanegroup_repeat_m, self.tunable.lanegroup_repeat_n, self.tunable.precision)

        else:
            def flatten(x):
                from functools import reduce
                return reduce(lambda a, b: a*b, x, 1)
            ctrl_xdlops_mapping = get_ctrl_xdlops_mapping_from_wave_tile(self.tunable.gemm_m_per_block, self.tunable.gemm_n_per_block, self.tunable.wave_tile_m, self.tunable.wave_tile_n, self.tunable.wave_tile_k,
                    self.tunable.wave_repeat_m, self.tunable.wave_repeat_n, self.tunable.wave_step_m, self.tunable.wave_step_n, self.tunable.block_size // AMDGPU_WAVE_SIZE, self.tunable.precision, bf16_1k_in_fp16 = self.use_bf16_1k_in_fp16())
            self.xdlops_mapping = igemm_xdlops_mapping_t(self.mc, ctrl_xdlops_mapping)
            assert flatten(ctrl_xdlops_mapping.acc_c_per_thread_m()) % self.coalescing_store_groups == 0, \
                f"coalescing store groups should be divided by agpr per thread in m direction {ctrl_xdlops_mapping.acc_c_per_thread_m()}"

            ctrl_coalescing_store_xdlops = ctrl_coalescing_store_xdlops_t()
            ctrl_coalescing_store_xdlops.cxm = ctrl_xdlops_mapping
            ctrl_coalescing_store_xdlops.gemm_k_global_split = self.tunable.gemm_k_global_split
            ctrl_coalescing_store_xdlops.coalescing_groups = self.coalescing_store_groups
            ctrl_coalescing_store_xdlops.precision = self.tunable.precision
            ctrl_coalescing_store_xdlops.block_size = self.tunable.block_size
            # gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()
            na_nb0, na_nb1, na_e, na_c, nb_e, nb_c, nb_k0, nb_k1 = self.get_dims_lengths()
            ctrl_coalescing_store_xdlops.gemm_m_m0_m1 = [na_nb0, na_nb1]
            ctrl_coalescing_store_xdlops.accvgpr_unified = self.is_accvgpr_unified()

            def get_vector_write_out():
                vector_write = 1
                config_vs = self.tunable.vector_store
                if self.tunable.precision == 'fp32':
                    assert config_vs == 0
                    vector_write = 1                        # fp32 vector seems not much perf improvement
                elif self.tunable.precision == 'fp16':
                    if self.tunable.gemm_k_global_split:
                        if config_vs == 0:
                            vector_write = 2                    # prefer use buffer_atomic_pk_add_f16
                        else:
                            vector_write = utility_gcd(2, config_vs)
                    else:
                        if self.is_pad_c():
                            vector_write = 1
                        else:
                            vector_write = utility_gcd(self.tunable.gemm_n_per_block, config_vs if config_vs != 0 else 8)
                            #return 2
                elif self.tunable.precision == 'bf16':
                    if self.tunable.gemm_k_global_split:
                        vector_write = 1
                    else:
                        if self.is_pad_c():
                            vector_write = 1
                        else:
                            vector_write = utility_gcd(self.tunable.gemm_n_per_block, config_vs if config_vs != 0 else 8)
                elif self.tunable.precision == 'int8':
                    if self.is_pad_c():
                        vector_write = 1
                    else:
                        vector_write = utility_gcd(self.tunable.gemm_n_per_block, config_vs if config_vs != 0 else 16)
                else:
                    assert False

                num_dword_per_group = ctrl_coalescing_store_xdlops.get_num_dword_per_group()
                if vector_write > num_dword_per_group:
                    '''
                    each coalescing group dword can't smaller than vector write size. currently only int8 may going here
                    '''
                    # print(f'adjusted vector_write({vector_write}) out by num_dword_per_group({num_dword_per_group})')
                    vector_write = num_dword_per_group
                return vector_write

            ctrl_coalescing_store_xdlops.vector_write_out = get_vector_write_out()

            if ctrl_coalescing_store_xdlops.vector_write_out == 1 and self.tunable.gemm_k_global_split == 1 and self.tunable.precision == 'fp16':
                ctrl_coalescing_store_xdlops.precision = 'fp32'

            elif self.tunable.gemm_k_global_split == 1 and self.tunable.precision == 'bf16':
                ctrl_coalescing_store_xdlops.precision = 'fp32'

            #if gemm_m_order == IGEMM_FWD_GTC_NHWC_LDS_STORE_ORDER_GEMM_M_N1B_N0:
            #    # we may consider not suppor this mode
            #    ctrl_coalescing_store_xdlops.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M1_M0
            ctrl_coalescing_store_xdlops.adjust_optimal_coalescing_groups()        # in m1_m0 order, must adjust 
            self.coalescing_store = igemm_coalescing_store_xdlops_t(mc, ctrl_coalescing_store_xdlops)

        self.label_out = f"L_{self.name()}_out"
        self.dict_shifted_stride = dict()

        self.karg = self.kernel_karg_t(mc, self)
        self.sgpr = self.kernel_sgpr_t(mc, self)
        self.vgpr = self.kernel_vgpr_t(mc, self)
        if self.tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            self.agpr = self.kernel_agpr_t(mc, self)

    def use_bf16_1k_in_fp16(self):
        if self.tunable.precision == 'fp16' and self.mc.arch_config.arch == AMDGPU_ARCH_GFX90A and IGEMM_FWD_GTC_NHWC_USE_BF16_1K_IN_FP16:
            return True
        else:
            return False

    def get_predefine_for_bf16_1k_in_fp16(self):
        return 'igemm_fwd_fp16_alt_impl'

    def get_predefine_for_bf16_1k_in_fp16_default_value(self):
        return 0

    def name(self):
        return igemm_gtc_encode_kernel_name(self.tunable, self.mc.arch_config.arch)

    def try_shift_stride(self, gpr, shifter):
        assert type(gpr) is sym_t
        with self._deferred_context():
            if gpr.label not in self.dict_shifted_stride:
                self.dict_shifted_stride[gpr.label] = gpr
                self._emit(f"s_lshl_b32 s[{gpr()}], s[{gpr()}], {shifter}")
        return self._get_deferred()

    def is_accvgpr_unified(self):
        return IGEMM_FWD_GTC_NHWC_ACCVGPR_UNIFIED and self.mc.arch_config.arch == AMDGPU_ARCH_GFX90A \
                and not (self.tunable.gemm_m_per_block == 256 and self.tunable.gemm_n_per_block == 256)

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
            return '.v_fwd_gtc_nhwc_set_flag_nhw'

        def expr(self):
            self._emit(f"v_cmp_gt_u32 vcc, s[{self.s_h()}], v[{self.v_ih()}]")
            self._emit(f"v_cndmask_b32 v[{self.v_flag()}], 0, v[{self.v_flag_n()}], vcc")
            self._emit(f"v_cmp_gt_u32 vcc, s[{self.s_w()}], v[{self.v_iw()}]")
            self._emit(f"v_cndmask_b32 v[{self.v_flag()}], 0, v[{self.v_flag()}], vcc")

    class macro_move_slice_window_block_wise_1x1_t(macro_base_t):
        def __init__(self, mc, tunable, inline, **options):
            macro_base_t.__init__(self, mc, True)
            is_pad_c = False if 'is_pad_c' not in options else options['is_pad_c']
            self.tunable = tunable
            if tunable.tensor_a_pass_through:
                self.declare_arg("s_in_base")       # 64bit acc
            else:
                self.declare_arg("s_in_offset")     # use this as c itr, since other dimension of input is voffset
            if is_pad_c:
                if tunable.tensor_a_pass_through:
                    self.declare_arg("v_gtc_ic_a_itr")
                self.declare_arg("v_gtc_ic_itr")
                self.declare_arg("v_in_flag")
                self.declare_arg("v_wei_flag")
                self.declare_arg("v_tmp")           # 2 needed
                self.declare_arg("s_c")
            self.declare_arg("v_wei_os")
            self.declare_arg("s_move_slice_k_stride_c")                  # this is indeed gemm_k * data_byte, same for input/weight
            self.options = options

        def name(self):
            return '.v_fwd_gtc_nhwc_move_slice_window_block_wise_1x1_{self.tunable.tensor_a_pass_through}_{self.tunable.tensor_b_pass_through}'

        def expr(self):
            is_pad_c = False if 'is_pad_c' not in self.options else self.options['is_pad_c']
            if is_pad_c:
                unroll_k = self.options['unroll_k']              # must have value
                nb_per_thread = self.options['nb_per_thread']
                nk_per_thread = self.options['nk_per_thread']
            if self.tunable.tensor_a_pass_through:
                self._emit(f"s_add_u32 s[{self.s_in_base()}], s[{self.s_move_slice_k_stride_c()}], s[{self.s_in_base()}]")
                self._emit(f"s_addc_u32 s[{self.s_in_base(1)}], 0, s[{self.s_in_base(1)}]")
            else:
                self._emit(f"s_add_u32 s[{self.s_in_offset()}],  s[{self.s_move_slice_k_stride_c()}], s[{self.s_in_offset()}]")
            self._emit(f"v_add_u32 v[{self.v_wei_os()}], s[{self.s_move_slice_k_stride_c()}], v[{self.v_wei_os()}]")
            if is_pad_c:
                if self.tunable.tensor_a_pass_through:
                    self._emit(f"v_add_u32 v[{self.v_gtc_ic_a_itr()}], {unroll_k}, v[{self.v_gtc_ic_a_itr()}]")
                self._emit(f"v_add_u32 v[{self.v_gtc_ic_itr()}], {unroll_k}, v[{self.v_gtc_ic_itr()}]")
                self._emit(f"v_cmp_gt_u32 vcc, s[{self.s_c()}], v[{self.v_gtc_ic_itr()}]")
                self._emit(f"v_cndmask_b32 v[{self.v_tmp()}], 0, 1, vcc")
                for i in range(nk_per_thread):
                    self._emit(f"v_and_b32 v[{self.v_wei_flag(i)}], v[{self.v_tmp()}], v[{self.v_wei_flag(i)}]")
                if self.tunable.tensor_a_pass_through:
                    self._emit(f"v_cmp_gt_u32 vcc, s[{self.s_c()}], v[{self.v_gtc_ic_a_itr()}]")
                    self._emit(f"v_cndmask_b32 v[{self.v_tmp()}], 0, 1, vcc")
                for i in range(nb_per_thread):
                    self._emit(f"v_and_b32 v[{self.v_in_flag(i)}], v[{self.v_tmp()}], v[{self.v_in_flag(i)}]")
            self._emit_empty_line()

    class macro_move_slice_window_block_wise_t(macro_base_t):
        '''
        nhwc gemm_k = e*c, and thread/cluster length for e is always 1
        hence always move along c and accumulate into e

        this macro is for input and weight together.
        block-wise move slice window, means we increase y*x*c using sgpr.
        Indeed this is always true, since gemm_k % k_per_block == 0 always true.
        Beside, we always increase along c dimension, this means y, x, c using sgpr is enough

        '''
        def __init__(self, mc, tunable, inline, **options):
            macro_base_t.__init__(self, mc, True)
            self.tunable = tunable
            is_pad_c = False if 'is_pad_c' not in options else options['is_pad_c']

            if tunable.tensor_a_pass_through:
                self.declare_arg("s_in_base")       # 64bit acc
                self.declare_arg("s_in_c_itr")
            else:
                self.declare_arg("s_in_offset")     # use this as c itr, since other dimension of input is voffset
            if is_pad_c:
                if tunable.tensor_a_pass_through:
                    self.declare_arg("v_gtc_ic_a_itr")
                self.declare_arg("v_gtc_ic_itr")
                self.declare_arg("v_in_flag")
                self.declare_arg("v_wei_flag")
                self.declare_arg("v_tmp")           # 2 needed
                self.declare_arg("s_c")
            self.declare_arg("v_wei_os")
            self.declare_arg("s_move_slice_k_stride_c")                  # this is indeed gemm_k * data_byte, same for input/weight
            self.declare_arg("s_gemm_k_num_c")  # c * data_byte
            self.declare_arg("s_flag_need_acc_yx")
            self.options = options

        def name(self):
            return f'.v_fwd_gtc_nhwc_move_slice_window_block_wise_{self.tunable.tensor_a_pass_through}_{self.tunable.tensor_b_pass_through}'

        def expr(self):
            is_pad_c = False if 'is_pad_c' not in self.options else self.options['is_pad_c']
            if is_pad_c:
                unroll_k = self.options['unroll_k']              # must have value
                nb_per_thread = self.options['nb_per_thread']
                nk_per_thread = self.options['nk_per_thread']
            if self.tunable.tensor_a_pass_through:
                self._emit(f"s_add_u32 s[{self.s_in_base()}], s[{self.s_move_slice_k_stride_c()}], s[{self.s_in_base()}]")
                self._emit(f"s_addc_u32 s[{self.s_in_base(1)}], 0, s[{self.s_in_base(1)}]")
            else:
                self._emit(f"s_add_u32 s[{self.s_in_offset()}],  s[{self.s_move_slice_k_stride_c()}], s[{self.s_in_offset()}]")
            self._emit(f"v_add_u32 v[{self.v_wei_os()}], s[{self.s_move_slice_k_stride_c()}], v[{self.v_wei_os()}]")
            if is_pad_c:
                if self.tunable.tensor_a_pass_through:
                    self._emit(f"v_add_u32 v[{self.v_gtc_ic_a_itr()}], {unroll_k}, v[{self.v_gtc_ic_a_itr()}]")
                self._emit(f"v_add_u32 v[{self.v_gtc_ic_itr()}], {unroll_k}, v[{self.v_gtc_ic_itr()}]")
                self._emit(f"v_cmp_gt_u32 vcc, s[{self.s_c()}], v[{self.v_gtc_ic_itr()}]")
                self._emit(f"v_cndmask_b32 v[{self.v_tmp()}], 0, 1, vcc")
                for i in range(nk_per_thread):
                    self._emit(f"v_and_b32 v[{self.v_wei_flag(i)}], v[{self.v_tmp()}], v[{self.v_wei_flag(i)}]")
                if self.tunable.tensor_a_pass_through:
                    self._emit(f"v_cmp_gt_u32 vcc, s[{self.s_c()}], v[{self.v_gtc_ic_a_itr()}]")
                    self._emit(f"v_cndmask_b32 v[{self.v_tmp()}], 0, 1, vcc")
                for i in range(nb_per_thread):
                    self._emit(f"v_and_b32 v[{self.v_in_flag(i)}], v[{self.v_tmp()}], v[{self.v_in_flag(i)}]")
            if self.tunable.tensor_a_pass_through:
                self._emit(f"s_add_u32 s[{self.s_in_c_itr()}],  s[{self.s_move_slice_k_stride_c()}], s[{self.s_in_c_itr()}]")
                self._emit(f"s_cmp_le_u32 s[{self.s_gemm_k_num_c()}], s[{self.s_in_c_itr()}]")
            else:
                self._emit(f"s_cmp_le_u32 s[{self.s_gemm_k_num_c()}], s[{self.s_in_offset()}]")
            if not self.tunable.tensor_a_pass_through and not self.tunable.tensor_b_pass_through:
                self._emit(f"s_cselect_b32 s[{self.s_flag_need_acc_yx()}], 1, 0")
            self._emit_empty_line()

    class macro_move_slice_window_block_wise_acc_yx_t(macro_base_t):
        '''
        can not inline
        prefer to put this before global load wait. And for simplicity, no auto schedule.
        '''
        def __init__(self, mc, tunable, inline, **options):
            macro_base_t.__init__(self, mc, True)
            self.tunable = tunable
            is_pad_c = False if 'is_pad_c' not in options else options['is_pad_c']
            if tunable.tensor_a_pass_through:
                self.declare_arg("s_in_base")
                self.declare_arg("s_in_c_itr")     #
                self.declare_arg("s_gemm_k_num_c") # used to U64 sub s_in_base, can be None
            else:
                self.declare_arg("s_in_offset")     # use this as c itr, since other dimension of input is voffset
            if tunable.gemm_k_global_split or is_pad_c:
                self.declare_arg("v_wei_os")
                self.declare_arg("s_gemm_k_diff_c")
            self.declare_arg("v_in_os")
            self.declare_arg("v_in_ihi_list")
            self.declare_arg("v_in_iwi_list")
            self.declare_arg("v_in_flag")
            if not IGEMM_FWD_GTC_NHWC_PACK_IN_FLAG:
                self.declare_arg("v_in_flag_n")
            if is_pad_c:
                if tunable.tensor_a_pass_through:
                    self.declare_arg("v_gtc_ic_a_itr")
                    self.declare_arg("v_gtc_ic_a")
                self.declare_arg("v_gtc_ic_itr")
                self.declare_arg("v_gtc_ic")
                self.declare_arg("v_wei_flag")
                self.declare_arg("v_wei_tmp_pack")
            self.declare_arg("s_flag_need_acc_yx")
            self.declare_arg("s_move_slice_k_ix")
            self.declare_arg("s_x")
            self.declare_arg("s_in_diff_hi")    # this is s_dilation_h * s_in_stride_hi  -  (x - 1) * s_dilation_w * s_in_stride_wi, always possitive
            self.declare_arg("s_in_diff_wi")    # this is s_dilation_w * s_in_stride_wi
            self.declare_arg("s_dilation_h")
            self.declare_arg("s_dilation_w")
            self.declare_arg("s_dilation_w_x") # this is -1* (x - 1) * s_dilation_w
            self.declare_arg("s_hi")
            self.declare_arg("s_wi")
            self.declare_arg("v_tmp")   # 2 needed
            self.declare_arg("s_tmp")
            self.options = options
        def name(self):
            return '.v_fwd_gtc_nhwc_move_slice_window_block_wise_acc_yx'

        def expr(self):
            assert "label_acc_yx" in self.options
            label_acc_yx = self.options["label_acc_yx"] + '_{}'.format(self.expr_cnt)
            label_acc_yx_end = self.options["label_acc_yx"] + '_end' + '_{}'.format(self.expr_cnt)
            label_acc_yx_x_end = self.options["label_acc_yx"] + '_x_end' + '_{}'.format(self.expr_cnt)

            assert "nb_per_thread" in self.options
            nb_per_thread = self.options["nb_per_thread"]

            assert 'm_set_flag_nhw' in self.options
            m_set_flag_nhw = self.options['m_set_flag_nhw']

            is_pad_c = False if 'is_pad_c' not in self.options else self.options['is_pad_c']
            if is_pad_c:
                nk_per_thread = self.options["nk_per_thread"]

            if not self.tunable.tensor_a_pass_through and not self.tunable.tensor_b_pass_through:
                self._emit(f"s_cmp_eq_u32 1, s[{self.s_flag_need_acc_yx()}]")
            self._emit(f"s_cbranch_scc0 {label_acc_yx_end}  ; no need do accumulate yx")
            self._emit_front(f"{label_acc_yx}:")
            if self.tunable.gemm_k_global_split or is_pad_c:
                # wei os need to add a whole c when yx is changing
                self._emit(f"v_add_u32 v[{self.v_wei_os()}], v[{self.v_wei_os()}], s[{self.s_gemm_k_diff_c()}]")
            if self.tunable.tensor_a_pass_through:
                self._emit(f"s_sub_u32 s[{self.s_in_base()}], s[{self.s_in_base()}], s[{self.s_gemm_k_num_c()}]")
                self._emit(f"s_subb_u32 s[{self.s_in_base(1)}], s[{self.s_in_base(1)}], 0")
                self._emit(f"s_mov_b32 s[{self.s_in_c_itr()}], 0")    # reset input offset. wei, no care
            else:
                self._emit(f"s_mov_b32 s[{self.s_in_offset()}], 0")    # reset input offset. wei, no care
            if is_pad_c:
                if self.tunable.tensor_a_pass_through:
                    self._emit(f"v_mov_b32 v[{self.v_gtc_ic_a_itr()}], v[{self.v_gtc_ic_a()}]")
                self._emit(f"v_mov_b32 v[{self.v_gtc_ic_itr()}], v[{self.v_gtc_ic()}]")
            '''
            ix accumulate, will only accumulate in width, and will never carry on to height
            iy accumulate, will only accumulate in height, and will never carry on to batch
            this makes life easier
            '''
            # ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
            # iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w
            self._emit(f"s_add_u32 s[{self.s_move_slice_k_ix()}], 1, s[{self.s_move_slice_k_ix()}]")
            self._emit(f"s_cmp_le_u32 s[{self.s_x()}], s[{self.s_move_slice_k_ix()}]")

            # update iwi
            self._emit(f"s_cselect_b32 s[{self.s_tmp()}], s[{self.s_dilation_w_x()}], s[{self.s_dilation_w()}]")
            for i in range(nb_per_thread):
                self._emit(f"v_add_u32 v[{self.v_in_iwi_list(i)}], s[{self.s_tmp()}], v[{self.v_in_iwi_list(i)}]")

            # update in_os
            self._emit(f"s_cselect_b32 s[{self.s_tmp()}], s[{self.s_in_diff_hi()}], s[{self.s_in_diff_wi()}]")
            for i in range(nb_per_thread):
                self._emit(f"v_add_u32 v[{self.v_in_os(i)}], s[{self.s_tmp()}], v[{self.v_in_os(i)}]")

            # update ihi, accumulate
            self._emit(f"s_cbranch_scc0 {label_acc_yx_x_end}")
            self._emit(f"s_mov_b32 s[{self.s_move_slice_k_ix()}], 0")
            for i in range(nb_per_thread):
                self._emit(f"v_add_i32 v[{self.v_in_ihi_list(i)}], s[{self.s_dilation_h()}], v[{self.v_in_ihi_list(i)}]")
            self._emit_front(f"{label_acc_yx_x_end}:")

            # now set flags
            for i in range(nb_per_thread):
                if IGEMM_FWD_GTC_NHWC_PACK_IN_FLAG:
                    self._emit(f"v_bfe_u32 v[{self.v_tmp(1)}], v[{self.v_in_flag()}], {16 + i}, 1   ; extract flag_n")
                    self._emit(f"v_and_b32 v[{self.v_in_flag()}], {0xffffffff ^ (1<<i)}, v[{self.v_in_flag()}]")   # reset current flag
                    self._emit(m_set_flag_nhw(self.v_tmp(0), self.v_tmp(1), self.v_in_ihi_list(i), self.v_in_iwi_list(i), self.s_hi(), self.s_wi()))
                    self._emit(f"v_lshl_or_b32 v[{self.v_in_flag()}], v[{self.v_tmp(3)}],  {i}, v[{self.v_in_flag()}] ; reset flag")
                else:
                    self._emit(f"v_bfe_u32 v[{self.v_tmp(5)}], v[{self.v_in_flag_n()}], {i}, 1   ; extract flag_n")
                    self._emit(m_set_flag_nhw(self.v_in_flag(i), self.v_tmp(5), self.v_in_ihi_list(i), self.v_in_iwi_list(i), self.s_hi(), self.s_wi()))

            if is_pad_c:
                for i in range(nk_per_thread):
                    self._emit(f"v_bfe_u32 v[{self.v_wei_flag(i)}], v[{self.v_wei_tmp_pack()}], {i}, 1")

            self._emit_front(f"{label_acc_yx_end}:")
            self._emit_empty_line()

    class macro_move_slice_window_block_wise_merge_e_t(macro_base_t):
        def __init__(self, mc, tunable, inline, **options):
            macro_base_t.__init__(self, mc, True)
            self.tunable = tunable
            self.options = options
            is_pad_c = False if 'is_pad_c' not in self.options else self.options['is_pad_c']
            # ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
            # iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w

            # self.declare_arg("v_gtc_iy_itr")  # will never use this last dimension
            self.declare_arg("v_gtc_ix_itr")
            self.declare_arg("v_gtc_ic_itr")

            if is_pad_c:
                self.declare_arg("v_gtc_iec_itr")   # iterator of y*x*c
                self.declare_arg("s_y_x_c")         # y * x * c, used for range check
                self.declare_arg("v_wei_flag")

            self.declare_arg("v_in_os")
            self.declare_arg("v_in_ihi_list")
            self.declare_arg("v_in_iwi_list")
            self.declare_arg("v_in_flag")
            self.declare_arg("v_in_flag_n")

            self.declare_arg("v_wei_os")

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

            self.declare_arg("s_diff_in_os_acc_y_x_c")          # due to x, y, c increment,
                                                                #    s_move_slice_k_c * data_byte + s_move_slice_k_x * s_dilation_w * in_stride_wi + s_move_slice_k_y * s_dilation_h * in_stride_hi
            self.declare_arg("s_diff_in_os_ovf_c_acc_x")        # due to c increment and overflow
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
            return '.v_fwd_gtc_nhwc_move_slice_window_me'

        def expr(self):
            assert "nb_per_thread" in self.options
            nb_per_thread = self.options["nb_per_thread"]
            is_pad_c = False if 'is_pad_c' not in self.options else self.options['is_pad_c']
            if is_pad_c:
                unroll_k = self.options['unroll_k']
                nk_per_thread = self.options["nk_per_thread"]
            assert 'm_set_flag_nhw' in self.options
            m_set_flag_nhw = self.options['m_set_flag_nhw']

            self._emit(f"v_mov_b32 v[{self.v_in_iwi_diff()}], s[{self.s_diff_in_iwi_acc_x()}]")         # this is indeed in later stage ++x
            self._emit(f"v_mov_b32 v[{self.v_in_ihi_diff()}], s[{self.s_diff_in_ihi_acc_y()}]")         # this is indeed in later stage ++y
            self._emit(f"v_mov_b32 v[{self.v_in_os_diff()}], s[{self.s_diff_in_os_acc_y_x_c()}]")
            if is_pad_c:
                self._emit(f"v_add_u32 v[{self.v_gtc_iec_itr()}], {unroll_k}, v[{self.v_gtc_iec_itr()}]")

            self._emit(f"v_add_u32 v[{self.v_gtc_ic_itr()}], s[{self.s_move_slice_k_c()}], v[{self.v_gtc_ic_itr()}]")
            self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_c()}], v[{self.v_gtc_ic_itr()}]")
            self._emit(f"v_subrev_u32 v[{self.v_gtc_ic_itr()}], s[{self.s_c()}], v[{self.v_gtc_ic_itr()}]")
            self._emit(f"v_add_u32 v[{self.v_gtc_ix_itr()}], 1, v[{self.v_gtc_ix_itr()}]")
            self._emit(f"v_add_u32 v[{self.v_in_iwi_diff()}], s[{self.s_dilation_w()}], v[{self.v_in_iwi_diff()}]")
            self._emit(f"v_add_u32 v[{self.v_in_os_diff()}], s[{self.s_diff_in_os_ovf_c_acc_x()}], v[{self.v_in_os_diff()}]")
            self._emit(f"s_mov_b64 exec, -1")

            self._emit(f"v_add_u32 v[{self.v_gtc_ix_itr()}], s[{self.s_move_slice_k_x()}], v[{self.v_gtc_ix_itr()}]")
            self._emit(f"v_cmpx_le_u32 vcc, s[{self.s_x()}], v[{self.v_gtc_ix_itr()}]")
            self._emit(f"v_subrev_u32 v[{self.v_gtc_ix_itr()}], s[{self.s_x()}], v[{self.v_gtc_ix_itr()}]")
            self._emit(f"v_subrev_u32 v[{self.v_in_iwi_diff()}], s[{self.s_diff_in_iwi_ovf_x()}], v[{self.v_in_iwi_diff()}]")
            #self._emit(f"v_add_u32 v[{self.v_gtc_iy_itr()}], 1, v[{self.v_gtc_iy_itr()}]")
            self._emit(f"v_add_u32 v[{self.v_in_ihi_diff()}], s[{self.s_dilation_h()}], v[{self.v_in_ihi_diff()}]")
            self._emit(f"v_add_u32 v[{self.v_in_os_diff()}], s[{self.s_diff_in_os_ovf_x_acc_y()}], v[{self.v_in_os_diff()}]")
            self._emit(f"s_mov_b64 exec, -1")

            #self._emit(f"v_add_u32 v[{self.v_gtc_iy_itr()}], s[{self.s_move_slice_k_y()}], v[{self.v_gtc_iy_itr()}]")

            # now start to update flags and offsets
            for i in range(nb_per_thread):
                self._emit(f"v_add_u32 v[{self.v_in_iwi_list(i)}], v[{self.v_in_iwi_diff()}], v[{self.v_in_iwi_list(i)}]")
            for i in range(nb_per_thread):
                self._emit(f"v_add_u32 v[{self.v_in_ihi_list(i)}], v[{self.v_in_ihi_diff()}], v[{self.v_in_ihi_list(i)}]")
            
            self._emit(f"v_add_u32 v[{self.v_wei_os()}], s[{self.s_move_slice_k_stride_gemm_k()}], v[{self.v_wei_os()}]")
            if is_pad_c:
                self._emit(f"v_cmp_gt_u32 vcc, s[{self.s_y_x_c()}], v[{self.v_gtc_iec_itr()}]")
                self._emit(f"v_cndmask_b32 v[{self.v_in_iwi_diff()}], 0, 1, vcc")
                for i in range(nk_per_thread):
                    self._emit(f"v_and_b32 v[{self.v_wei_flag(i)}], v[{self.v_in_iwi_diff()}], v[{self.v_wei_flag(i)}]")

            for i in range(nb_per_thread):
                self._emit(f"v_add_u32 v[{self.v_in_os(i)}], v[{self.v_in_os_diff()}], v[{self.v_in_os(i)}]")
                if IGEMM_FWD_GTC_NHWC_PACK_IN_FLAG:
                    assert False
                else:
                    self._emit(f"v_bfe_u32 v[{self.v_in_ihi_diff()}], v[{self.v_in_flag_n()}], {i}, 1   ; extract flag_n")
                    self._emit(f"v_and_b32 v[{self.v_in_ihi_diff()}], v[{self.v_in_iwi_diff()}], v[{self.v_in_ihi_diff()}]")
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
                self._emit(f".v_clear_nc {v.v_gld_a() if tunable.global_prefetch_a_num == 1 else v.v_gld_a_gpf()}, {self.outer.get_num_vgpr_global_load_a()}")
                self._emit(m_in_2d_global_load(v.v_gld_a() if tunable.global_prefetch_a_num == 1 else v.v_gld_a_gpf(),
                    s.s_p_in(), v.v_in_os(),
                    *(None, None, None, None) if tunable.tensor_a_pass_through else (s.s_in_offset(), None, None, None),
                    v.v_in_flag(), v.v_tmp(), None, k.k_gload_in_c_stride))

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
                self._emit(m_wei_2d_global_load(v.v_gld_b(), s.s_p_wei(), v.v_wei_os(), None, s_wei_stride_d0(), s_wei_stride_d1(), s.s_wei_offset(), 
                                v.v_wei_flag(), None, None, None))
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
                if self.outer.use_bf16_1k_in_fp16():
                    m_packed_fp16_to_bf16 = macro_packed_fp16_to_bf16_t(self.mc, num_vgpr = self.outer.get_num_vgpr_global_load_a())
                    fp16_alt_impl_pds = self.outer.get_predefine_for_bf16_1k_in_fp16()
                    self._emit(f'.if {fp16_alt_impl_pds} == 1')
                    self._emit(m_packed_fp16_to_bf16(v.v_gld_a(), v.v_tmp(5)))
                    self._emit(f'.endif')
                self._emit(m_in_2d_shared_store(v.v_gld_a(), v.v_sst_a_os()))
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
                if self.outer.use_bf16_1k_in_fp16():
                    m_packed_fp16_to_bf16 = macro_packed_fp16_to_bf16_t(self.mc, num_vgpr = self.outer.get_num_vgpr_global_load_b())
                    fp16_alt_impl_pds = self.outer.get_predefine_for_bf16_1k_in_fp16()
                    self._emit(f'.if {fp16_alt_impl_pds} == 1')
                    self._emit(m_packed_fp16_to_bf16(v.v_gld_b(), v.v_tmp(5)))
                    self._emit(f'.endif')
                self._emit(m_wei_2d_shared_store(v.v_gld_b(), v.v_sst_b_os()))
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

            ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = outer.get_thread_lengths()
            ca_nb0, ca_nb1, ca_e, ca_c, cb_e, cb_c, cb_k0, cb_k1 = outer.get_cluster_lengths()
            data_byte = amdgpu_precision_data_byte(outer.tunable.precision)

            self.k_gload_in_c_stride    = sym_t('k_gload_in_c_stride', \
                        data_byte * utility_gcd(ta_c, 4 * (4 // data_byte)) * (ca_c if outer.tunable.tensor_a_pass_through else 1))

        def get_count(self):
            return self.k_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('k_'):
                    self._emit(v.declare())

    class kernel_sgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = outer.get_thread_lengths()
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

            self.s_in_stride_wi             = sym_t('s_in_stride_wi'            , sseq(1))
            self.s_in_stride_n              = sym_t('s_in_stride_n'             , sseq(1))

            if tb_k0 != 1:
                self.s_wei_stride_k0        = sym_t('s_wei_stride_k0'           , sseq(1))
            self.s_wei_stride_k             = sym_t('s_wei_stride_k'            , sseq(1))

            self.s_out_stride_wo            = sym_t('s_out_stride_wo'           , sseq(1))
            self.s_out_stride_n             = sym_t('s_out_stride_n'            , sseq(1))

            self.s_block_gtc_ig             = sym_t("s_block_gtc_ig"            , sseq(1))
            self.s_block_gtc_ik             = sym_t("s_block_gtc_ik"            , sseq(1))
            self.s_block_gtc_inb            = sym_t("s_block_gtc_inb"           , sseq(1))

            if outer.tunable.merge_e == 0:
                self.s_move_slice_k_stride_c        = sym_t("s_move_slice_k_stride_c"       , sseq(1))
            else:
                self.s_move_slice_k_stride_gemm_k   = sym_t("s_move_slice_k_stride_gemm_k"  , sseq(1))

            if outer.is_pad_c() and outer.tunable.merge_e == 0:
                self.s_c_padded             = sym_t("s_c_padded"                , sseq(1))
            self.s_knum                     = sym_t("s_knum"                    , 3)

            #if outer.tunable.nxe != 0:
            self.s_dim_br                   = sym_t("s_dim_br"                  , sseq(1))
            self.s_dim_mp                   = sym_t("s_dim_mp"                  , sseq(1))
            self.s_dim_mr                   = sym_t("s_dim_mr"                  , sseq(1))
            self.s_dim_np                   = sym_t("s_dim_np"                  , sseq(1))

            if outer.tunable.merge_e == 0:
                self.s_gemm_k_num_c         = sym_t("s_gemm_k_num_c"            , self.s_dim_np.value)

            if outer.is_pad_c() or outer.tunable.gemm_k_global_split:
                self.s_gemm_k_diff_c        = sym_t("s_gemm_k_diff_c"           , self.s_group.value)

            if outer.tunable.merge_e == 0:
                self.s_in_diff_hi           = sym_t("s_in_diff_hi"              , self.s_block_gtc_ik.value)
                self.s_in_diff_wi           = sym_t("s_in_diff_wi"              , self.s_block_gtc_ig.value)
                self.s_dilation_w_x         = sym_t("s_dilation_w_x"            , self.s_y.value if outer.tunable.nxe != 0 else sseq(1))
                self.s_move_slice_k_ix      = sym_t("s_move_slice_k_ix"         , self.s_dim_br.value)
                self.s_flag_need_acc_yx     = sym_t("s_flag_need_acc_yx"        , self.s_dim_mp.value)
            else:
                self.s_move_slice_k_y           = sym_t("s_move_slice_k_y"          , sseq(1))
                self.s_move_slice_k_x           = sym_t("s_move_slice_k_x"          , sseq(1))
                self.s_move_slice_k_c           = sym_t("s_move_slice_k_c"          , sseq(1))
                self.s_diff_in_os_acc_y_x_c     = sym_t("s_diff_in_os_acc_y_x_c"    , self.s_block_gtc_ig.value)
                self.s_diff_in_os_ovf_c_acc_x   = sym_t("s_diff_in_os_ovf_c_acc_x"  , self.s_y.value)
                self.s_diff_in_os_ovf_x_acc_y   = sym_t("s_diff_in_os_ovf_x_acc_y"  , self.s_dim_br.value)
                self.s_diff_in_iwi_acc_x        = sym_t("s_diff_in_iwi_acc_x"       , self.s_dim_mp.value)
                self.s_diff_in_iwi_ovf_x        = sym_t("s_diff_in_iwi_ovf_x"       , self.s_dim_np.value)
                self.s_diff_in_ihi_acc_y        = sym_t("s_diff_in_ihi_acc_y"       , self.s_pad_w.value)
                self.s_y_x_c                    = sym_t("s_y_x_c"                   , sseq(1))

            self.s_kitr                     = sym_t("s_kitr"                    , 1)
            if outer.tunable.precision == 'int8':
                self.s_0xff                 = sym_t("s_0xff"                    , sseq(1))
            if outer.tunable.tensor_a_pass_through:
                # need s precache
                # in_npc = ((ta_c // k_pack) - 2) if ((ta_c // k_pack) - 2 > 0 ) else 0
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
                if outer.tunable.merge_e == 1:
                    self.s_magic_4         = sym_t("s_magic_4"                ,self.s_p_wei.value + 2)
                    self.s_magic_5         = sym_t("s_magic_5"                ,self.s_p_wei.value + 3)
                    self.s_shift_pack_0    = sym_t("s_shift_pack_0"           ,sseq(1))
                    self.s_shift_pack_1    = sym_t("s_shift_pack_1"           ,sseq(1))
                else:
                    self.s_shift_pack_0    = sym_t("s_shift_pack_0"           ,sseq(1))

            if outer.tunable.gemm_k_global_split:
                self.s_block_gtc_ic        = sym_t("s_block_gtc_ic"           ,sseq(1)) # add c split
                self.s_gemmk_split         = sym_t("s_gemmk_split"            ,sseq(1))
                self.s_sub_c               = sym_t("s_sub_c"                  ,sseq(1))
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
            self.outer = outer
            ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = outer.get_thread_lengths()
            ca_nb0, ca_nb1, ca_e, ca_c, cb_e, cb_c, cb_k0, cb_k1 = outer.get_cluster_lengths()

            nb_per_thread = ta_nb0 if ta_nb0 != 1 else ta_nb1
            nk_per_thread = tb_k0 if tb_k0 != 1 else tb_k1
            assert nb_per_thread <= 16, "we pack flag into single vgpr"

            k_pack = outer.get_k_pack()
            share_load_packed  = k_pack if outer.tunable.tensor_a_pass_through or outer.tunable.tensor_b_pass_through else 1

            is_vgpr_acc_c = outer.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS
            vseq = gpr_sequencer_t()
            data_byte                   = amdgpu_precision_data_byte(outer.tunable.precision)
            num_vgpr_global_load_a      = outer.get_num_vgpr_global_load_a()
            num_vgpr_global_load_b      = outer.get_num_vgpr_global_load_b()

            share_load_packed_vgpr      = share_load_packed // (4 // data_byte) //  outer.xdlops_mapping.ctrl.inst_mfma.num_v_a \
                                            if outer.tunable.tensor_a_pass_through or outer.tunable.tensor_b_pass_through else 1

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
                v_c_coalescing_num      = outer.coalescing_store.ctrl.get_vgpr_usage()
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
            if IGEMM_FWD_GTC_NHWC_PACK_IN_FLAG:
                self.v_in_flag          = sym_t("v_in_flag"         ,vseq(1))   # bfe this!, hi 16bit is flag_n, lo 16 bit is pure flag
            else:
                self.v_in_flag          = sym_t("v_in_flag"         ,vseq(nb_per_thread))
                self.v_in_flag_n        = sym_t("v_in_flag_n"       ,vseq(1))      # bfe this!, lo 16bit is flag_n

            self.v_wei_os               = sym_t("v_wei_os"          ,vseq(1))
            self.v_out_os               = sym_t("v_out_os"          ,vseq(1))

            if outer.tunable.tensor_a_pass_through:
                self.v_gtc_ic_a         = sym_t("v_gtc_ic_a"        ,self.v_gld_a.value if not outer.is_pad_c() else vseq(1))
                if outer.is_pad_c() and outer.tunable.merge_e == 0:
                    self.v_gtc_ic_a_itr = sym_t("v_gtc_ic_a_itr"    ,vseq(1))
            if outer.tunable.tensor_b_pass_through:
                self.v_gtc_ic_b         = sym_t("v_gtc_ic_b"        ,self.v_gld_b.value if not outer.is_pad_c() else vseq(1))
                if outer.is_pad_c() and outer.tunable.merge_e == 0:
                    self.v_gtc_ic_b_itr = sym_t("v_gtc_ic_b_itr"    ,vseq(1))
            if not (outer.tunable.tensor_a_pass_through and outer.tunable.tensor_b_pass_through):
                self.v_gtc_ic           = sym_t("v_gtc_ic"          ,vseq(1))
                if outer.is_pad_c() and outer.tunable.merge_e == 0:
                    self.v_gtc_ic_itr   = sym_t("v_gtc_ic_itr"      ,vseq(1))
            if outer.tunable.merge_e == 1:
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

            self.v_gemm_in              = sym_t("v_gemm_in"         ,vseq(1))
            self.v_gemm_im              = sym_t("v_gemm_im"         ,vseq(1))
            self.v_co_sub_m_index       = sym_t("v_co_sub_m_index"  ,self.v_gemm_im.value)
            self.v_co_sub_n_index       = sym_t("v_co_sub_n_index"  ,self.v_gemm_in.value)

            self.v_tmp                  = sym_t("v_tmp"             ,vseq(6, 2))
            self.v_wei_tmp_pack         = sym_t("v_wei_tmp_pack"    ,vseq(1) if outer.is_pad_c() else \
                                                        (self.v_gld_a.value - 1 if self.v_gld_a.value > 1 else vseq(1)))
            if nk_per_thread <= 4 and IGEMM_FWD_GTC_NHWC_PACK_IN_FLAG == 0:
                self.v_wei_flag         = sym_t("v_wei_flag"        ,self.v_tmp.value)
            else:
                self.v_wei_flag         = sym_t("v_wei_flag"        ,vseq(nk_per_thread))

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
            if outer.is_accvgpr_unified():
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
        ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
        pack_factor = (4 // amdgpu_precision_data_byte(self.tunable.precision)) if ta_c != 1 else 1
        return self.tunable.num_global_load_a // pack_factor
    
    def get_num_vgpr_global_load_b(self):
        ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
        pack_factor = (4 // amdgpu_precision_data_byte(self.tunable.precision)) if tb_c != 1 else 1
        return self.tunable.num_global_load_b // pack_factor

    def get_thread_lengths(self):
        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(t_ta) == 4 and len(t_tb) == 4

        ta_e, ta_c, ta_nb0, ta_nb1 = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        tb_e, tb_c, tb_k0,  tb_k1  = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        if self.tunable.tensor_a_pass_through or self.tunable.tensor_b_pass_through:
            pass
        else:
            assert ta_e == tb_e and ta_c == tb_c
            if self.tunable.precision == 'fp32':
                assert ta_c in (1, 2, 4), "currently c will be used as LDS store/load vector size, now only support this"
            elif self.tunable.precision in ('fp16', 'bf16'):
                assert ta_c in (1, 2, 4, 8, 16, 32)
            elif self.tunable.precision == 'int8':
                assert ta_c in (1, 2, 4, 8, 16, 32, 64)

        assert ta_e == 1, "currently not support >1 in e dimension"

        # it's no point to have both x0, x1 have copy value
        if not self.tunable.tensor_a_pass_through:
            assert not (ta_nb0 != 1 and ta_nb1 != 1)
        if not self.tunable.tensor_b_pass_through:
            assert not (tb_k0 != 1 and tb_k1 != 1)

        return ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1  # M, K, N

    def get_cluster_lengths(self):
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4

        ca_e, ca_c, ca_nb0, ca_nb1 = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        cb_e, cb_c, cb_k0,  cb_k1  = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        if not self.tunable.tensor_a_pass_through:
            assert ca_nb1 != 1
            assert ca_e == cb_e and ca_c == cb_c
            assert ca_nb0 == 1
        if not self.tunable.tensor_b_pass_through:
            assert cb_k0 == 1

        assert ca_e == 1

        return ca_nb0, ca_nb1, ca_e, ca_c, cb_e, cb_c, cb_k0, cb_k1  # M, K, N

    def get_dims_lengths(self):
        ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
        ca_nb0, ca_nb1, ca_e, ca_c, cb_e, cb_c, cb_k0, cb_k1 = self.get_cluster_lengths()

        na_nb0, na_nb1, na_e, na_c = ta_nb0 * ca_nb0, ta_nb1 * ca_nb1,  ta_e * ca_e,   ta_c * ca_c
        nb_k0,  nb_k1 , nb_e, nb_c = tb_k0  * cb_k0,  tb_k1  * cb_k1,   tb_e * cb_e,   tb_c * cb_c

        return na_nb0, na_nb1, na_e, na_c, nb_e, nb_c, nb_k0, nb_k1  # M, K, N

    def get_thread_copy_dims(self):
        ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
        in_thread_copy_dims  = [ta_nb0, ta_nb1, ta_e, ta_c]
        wei_thread_copy_dims = [tb_k0,  tb_k1,  tb_e, tb_c]     # always reordered!
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
        ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
        data_byte = amdgpu_precision_data_byte(self.tunable.precision)
        if (not self.tunable.tensor_a_pass_through and not self.tunable.tensor_b_pass_through) or \
            (self.tunable.tensor_a_pass_through and self.tunable.tensor_b_pass_through):
            assert ta_c == tb_c
            return tb_c
        else:
            if self.tunable.tensor_a_pass_through:
                assert ta_c % tb_c == 0
                return utility_gcd(ta_c, 4 * (4 // data_byte)) if ta_c != 1 else 1
            else:
                assert tb_c % ta_c == 0
                return utility_gcd(tb_c, 4 * (4 // data_byte)) if tb_c != 1 else 1

    def is_pad_c(self):
        '''
        NHWC implementation always want to vector load c, but we can still pad c(like 3) to a good number
        another assumption would be write out. in fp32 we prefer non vector store, so no problem
        but in fp16 we prefer vector store. hence another assumption would be, if this function is true
        then fp16 no longer use vector store.
        this is also true for int8

        update:
        in merge_e config, is_pad_c actually means is_pad_gemm_k
        '''
        ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
        if ta_c == 1 and tb_c == 1:
            assert self.tunable.vector_store == 0
            return True
        return False

    def get_macro_global_load(self):
        '''
        NOTICE: input/wei always load gemm_k (e*c) first. indeed always load c, and do vector load if possible
        '''
        inline = True if self.tunable.fma_interleave else False
        ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
        na_nb0, na_nb1, na_e, na_c, nb_e, nb_c, nb_k0, nb_k1 = self.get_dims_lengths()

        in_thread_copy_dims, wei_thread_copy_dims = self.get_thread_copy_dims()
        in_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()
        ctrl_wei_gld = ctrl_2d_global_load_t()
        ctrl_in_gld = ctrl_2d_global_load_t()

        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        ctrl_wei_gld.precision = self.tunable.precision
        ctrl_in_gld.precision  = self.tunable.precision
        ctrl_wei_gld.vector_d1 = utility_gcd(tb_c, 4 * (4 // data_byte)) if tb_c != 1 else 1
        ctrl_in_gld.vector_d1  = utility_gcd(ta_c, 4 * (4 // data_byte)) if ta_c != 1 else 1

        if self.tunable.tensor_b_pass_through:
            ctrl_wei_gld.precache_ptn = GLOBAL_PTN_D0_S | GLOBAL_PTN_D1_S
            ctrl_wei_gld.flag_on_d0 = 1
            ctrl_wei_gld.length_d0 = tb_k0 if tb_k0 != 1 else tb_k1
            ctrl_wei_gld.length_d1 = tb_c
            ctrl_wei_gld.vector_d1 = self.get_k_pack()
            # ctrl_wei_gld.flag_merge_v = 0 if self.tunable.tensor_b_pass_through_interleave_gld else 1
        else:
            ctrl_wei_gld.precache_ptn = GLOBAL_PTN_D0_S | GLOBAL_PTN_D1_S
            if self.wei_thread_copy_ndim == 2:
                ctrl_wei_gld.flag_on_d0 = 1
                ctrl_wei_gld.length_d0 = wei_thread_copy_dims[wei_thread_copy_index[0]]
                ctrl_wei_gld.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[1]]
            elif self.wei_thread_copy_ndim == 1:
                if tb_k0 * tb_k1 != 1:
                    ctrl_wei_gld.flag_on_d0 = 0
                    ctrl_wei_gld.flag_on_d1 = 1
                else:
                    ctrl_wei_gld.flag_on_d0 = 1
                    ctrl_wei_gld.flag_on_d1 = 0
                ctrl_wei_gld.length_d0 = 1
                ctrl_wei_gld.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]
            else:
                ctrl_wei_gld.length_d0 = 1
                ctrl_wei_gld.flag_on_d1 = 1
                ctrl_wei_gld.length_d1 = wei_thread_copy_dims[-1]

        if self.tunable.tensor_a_pass_through:
            #ctrl_in_gld.length_d0 = ta_c // self.get_k_pack()
            #ctrl_in_gld.length_d1 = (ta_nb0 if ta_nb0 != 1 else ta_nb1) * self.get_k_pack()
            #ctrl_in_gld.vector_d1 = self.get_k_pack()
            #ctrl_in_gld.flag_merge_v = 0 if self.tunable.tensor_a_pass_through_interleave_gld else 1
            ctrl_in_gld.length_d0 = ta_nb0 if ta_nb0 != 1 else ta_nb1
            ctrl_in_gld.length_d1 = ta_c
            ctrl_in_gld.vector_d1 = self.get_k_pack()
            assert not self.tunable.tensor_a_pass_through_interleave_gld, "NHWC always not interleave, this may reduce performance"
            # ctrl_in_gld.flag_merge_v = 1
            ctrl_in_gld.precache_ptn = GLOBAL_PTN_D0_V | GLOBAL_PTN_D1_K
            ctrl_in_gld.flag_on_d0 = 1
        else:
            # ctrl_in_gld.vector_d1 = self.get_k_pack()
            if self.in_thread_copy_ndim == 2:
                ctrl_in_gld.flag_on_d0 = 1
                ctrl_in_gld.precache_ptn = GLOBAL_PTN_D0_V | GLOBAL_PTN_D1_K
                ctrl_in_gld.length_d0 = in_thread_copy_dims[in_thread_copy_index[0]]
                ctrl_in_gld.length_d1 = in_thread_copy_dims[in_thread_copy_index[1]]
            elif self.in_thread_copy_ndim == 1:
                if ta_nb0 * ta_nb1 != 1:
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

        ctrl_in_gld.use_flag = 1
        ctrl_wei_gld.use_flag = 1

        if self.tunable.nxe != 0:
            if IGEMM_FWD_GTC_NHWC_PACK_IN_FLAG:
                ctrl_wei_gld.bfe_flag = 1
                ctrl_in_gld.bfe_flag = 1

        if self.tunable.precache_soffset:
            return macro_igemm_2d_global_load_precache_offset_t(self.mc, ctrl_wei_gld, inline), \
                    macro_igemm_2d_global_load_precache_offset_t(self.mc, ctrl_in_gld, inline)
        else:
            return macro_igemm_2d_global_load_t(self.mc, ctrl_wei_gld, inline),  macro_igemm_2d_global_load_precache_voffset_t(self.mc, ctrl_in_gld, inline)

    def get_macro_shared_store(self):
        #in_thread_copy_dims, wei_thread_copy_dims = self.get_thread_copy_dims()
        #in_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()
        na_nb0, na_nb1, na_e, na_c, nb_e, nb_c, nb_k0, nb_k1 = self.get_dims_lengths()
        ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        k_pack = self.get_k_pack()
        k_pack_lanegroup = self.xdlops_mapping.ctrl.lanegroup_k_per_thread()
        k_pack_src_mat = k_pack if k_pack != 1 else k_pack_lanegroup
        m_wei_2d_global_load, m_in_2d_global_load = self.get_macro_global_load()
        k_pack_gld_a = m_in_2d_global_load.ctrl.vector_d1
        k_pack_gld_b = m_wei_2d_global_load.ctrl.vector_d1

        if not self.tunable.tensor_a_pass_through:
            # input is gemm_k * gemm_m * k_pack
            in_sst_ctrl = ctrl_3d_shared_store_t()
            in_sst_ctrl.precision = self.tunable.precision
            in_sst_ctrl.length_d0 = ta_nb0
            in_sst_ctrl.length_d1 = ta_nb1
            in_sst_ctrl.length_dp = ta_c
            in_sst_ctrl.vector_dp = k_pack_gld_a
            in_sst_ctrl.stride_d0 = na_nb1 * k_pack_src_mat * data_byte
            in_sst_ctrl.stride_d1 = k_pack_src_mat * data_byte

        if not self.tunable.tensor_b_pass_through:
            # wei is gemm_k * gemm_n * k_pack
            wei_sst_ctrl = ctrl_3d_shared_store_t()
            wei_sst_ctrl.precision = self.tunable.precision
            wei_sst_ctrl.length_d0 = tb_k0
            wei_sst_ctrl.length_d1 = tb_k1
            wei_sst_ctrl.length_dp = tb_c
            wei_sst_ctrl.vector_dp = k_pack_gld_b
            wei_sst_ctrl.stride_d0 = nb_k1 * k_pack_src_mat * data_byte
            wei_sst_ctrl.stride_d1 = k_pack_src_mat * data_byte

        inline = True if self.tunable.fma_interleave else False 
        return macro_igemm_3d_shared_store_t(self.mc, in_sst_ctrl, inline) if not self.tunable.tensor_a_pass_through else None, \
            macro_igemm_3d_shared_store_t(self.mc, wei_sst_ctrl, inline) if not self.tunable.tensor_b_pass_through else None

    def get_macro_move_slice_window(self):
        inline = True if self.tunable.fma_interleave else False
        ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
        nb_per_thread = ta_nb0 if ta_nb0 != 1 else ta_nb1
        nk_per_thread = tb_k0 if tb_k0 != 1 else tb_k1
        unroll_k = self.tunable.gemm_k_per_block
        is_pad_c = self.is_pad_c()
        if self.tunable.merge_e == 1:
            move_slice_window = self.macro_move_slice_window_block_wise_merge_e_t(self.mc, self.tunable, inline,
                                        is_pad_c=is_pad_c, unroll_k=unroll_k, nb_per_thread=nb_per_thread, nk_per_thread=nk_per_thread, m_set_flag_nhw = self.get_macro_set_flag_nhw())
        elif self.tunable.nxe != 0:
            move_slice_window = self.macro_move_slice_window_block_wise_t(self.mc, self.tunable, inline,
                                        is_pad_c=is_pad_c, unroll_k=unroll_k, nb_per_thread=nb_per_thread, nk_per_thread=nk_per_thread)
        else:
            move_slice_window = self.macro_move_slice_window_block_wise_1x1_t(self.mc, self.tunable, inline,
                                        is_pad_c=is_pad_c, unroll_k=unroll_k, nb_per_thread=nb_per_thread, nk_per_thread=nk_per_thread)

        # return single functor !
        return move_slice_window

    def get_macro_move_slice_window_accumulate(self):
        inline = True if self.tunable.fma_interleave else False
        if self.tunable.nxe != 0:
            ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
            nb_per_thread = ta_nb0 if ta_nb0 != 1 else ta_nb1
            nk_per_thread = tb_k0 if tb_k0 != 1 else tb_k1
            is_pad_c = self.is_pad_c()
            return self.macro_move_slice_window_block_wise_acc_yx_t(self.mc, self.tunable, inline,
                label_acc_yx = self.name() + "_acc_yx",
                nb_per_thread = nb_per_thread,
                m_set_flag_nhw = self.get_macro_set_flag_nhw(),
                is_pad_c=is_pad_c, nk_per_thread=nk_per_thread)
        else:
            return None

    def get_macro_set_flag_nhw(self):
        inline = True if self.tunable.fma_interleave else False
        return self.macro_set_flag_nhw(self.mc, inline)

    def get_symbol_global_load_s_stride_d0_d1(self):
        ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
        # get the symbol object that load 2d may use
        s = self.sgpr
        s_dummy = sym_t("s_dummy")
        in_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()

        # input is ignored
        # [ta_nb0, ta_nb1, ta_e, ta_c]
        in_stride_gprs = [s_dummy,
                            s_dummy,
                            s_dummy,
                            s_dummy]

        # [tb_k0, tb_k1, ta_e, ta_c]
        wei_stride_gprs = [s.s_wei_stride_k0 if tb_k0 != 1 else s_dummy,
                            s.s_wei_stride_k if tb_k1 != 1 else s_dummy,
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

        ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
        ca_nb0, ca_nb1, ca_e, ca_c, cb_e, cb_c, cb_k0, cb_k1 = self.get_cluster_lengths()
        na_nb0, na_nb1, na_e, na_c, nb_e, nb_c, nb_k0, nb_k1 = self.get_dims_lengths()

        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        m_set_flag_nhw      = self.get_macro_set_flag_nhw()
        s_in_stride_d0, s_in_stride_d1, s_wei_stride_d0, s_wei_stride_d1 = self.get_symbol_global_load_s_stride_d0_d1()

        m_wei_2d_global_load, m_in_2d_global_load = self.get_macro_global_load()

        tc_index_dispatcher = igemm_thread_cluster_index_dispatcher_t(self.mc)
        tc_index_accumulator = igemm_thread_cluster_index_accumulator_t(self.mc)

        nb_per_thread = ta_nb0 if ta_nb0 != 1 else ta_nb1
        nk_per_thread = tb_k0 if tb_k0 != 1 else tb_k1

        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            m_mdiv_u32_vs = macro_mdiv_u32_rem_vs_t(self.mc)
            m_mdiv_u32_ss = macro_mdiv_u32_rem_ss_t(self.mc)
        else:
            m_int_div_rem_vv = macro_int_div_rem_vv_t(self.mc)
            m_int_div_rem_vs = macro_int_div_rem_vs_t(self.mc)
            m_int_div_rem_ss = macro_int_div_rem_ss_t(self.mc)

        s_dummy = sym_t("s_dummy")

        k_pack = self.get_k_pack()
        k_pack_lanegroup = self.xdlops_mapping.ctrl.lanegroup_k_per_thread()
        k_pack_src_mat = k_pack if k_pack != 1 else k_pack_lanegroup
        k_pack_gld_a = m_in_2d_global_load.ctrl.vector_d1
        k_pack_gld_b = m_wei_2d_global_load.ctrl.vector_d1

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
            if self.tunable.merge_e == 1:
                self._emit(f"s_load_dwordx2 s[{s.s_magic_4((0, 1))}], s[{s.s_ka((0, 1))}],  0+{k.k_magic_4()}")
            self._emit(f"s_load_dword s[{s.s_shift_pack_0()}], s[{s.s_ka((0, 1))}],  0+{k.k_shift_pack_0()}")
            if self.tunable.merge_e == 1:
                self._emit(f"s_load_dword s[{s.s_shift_pack_1()}], s[{s.s_ka((0, 1))}],  0+{k.k_shift_pack_1()}")
            if self.tunable.gemm_k_global_split:
                self._emit(f"s_load_dword s[{s.s_gemmk_split()}], s[{s.s_ka((0, 1))}],  0+{k.k_gemm_k_global_split()}")

        # NOTE: if merge_e, ta_c, tb_c indeed indicate merge dimension of e*c. for simplicity, we reuse the naming in various places
        self._emit(f"; in(e, c, nb0, nb1) thread_lengths: {ta_e}x{ta_c}x{ta_nb0}x{ta_nb1}, cluster_length: {ca_e}x{ca_c}x{ca_nb0}x{ca_nb1}, k_pack:{k_pack}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        if self.tunable.tensor_a_pass_through:
            self._emit(tc_index_dispatcher(v.v_in_inb(),  v.v_tmp(),  ca_nb1, ta_nb1))
            self._emit(tc_index_dispatcher(v.v_gtc_ic_a(),  v.v_tmp(),  ca_c, k_pack))    # <= note here, thread length is further reduced!
            self._emit(tc_index_dispatcher(v.v_tmp(1),  v.v_tmp(),  ca_nb0, ta_nb0, True))
            self._emit(tc_index_accumulator(v.v_in_inb(), v.v_tmp(1),  v.v_in_inb(), ca_nb0, ca_nb1, na_nb0, na_nb1))
        else:
            self._emit(tc_index_dispatcher(v.v_gtc_ic() if self.tunable.merge_e == 0 else v.v_gtc_iec(),  v.v_tmp(),  ca_c, ta_c))
            self._emit(tc_index_dispatcher(v.v_in_inb(), v.v_tmp(),  ca_nb1, ta_nb1, True))

        self._emit(f"; wei(e, c, k0, k1) thread_length: {tb_e}x{tb_c}x{tb_k0}x{tb_k1}, cluster_length: {cb_e}x{cb_c}x{cb_k0}x{cb_k1}, k_pack:{k_pack}")
        # weight ic same as input
        if (not self.tunable.tensor_a_pass_through) and (not self.tunable.tensor_b_pass_through):
            self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], {igemm_log2(ca_c)}, v0")
            self._emit(tc_index_dispatcher(v.v_wei_ik(), v.v_tmp(), cb_k1, tb_k1, True))
        elif self.tunable.tensor_a_pass_through:
            self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
            self._emit(tc_index_dispatcher(v.v_gtc_ic() if self.tunable.merge_e == 0 else v.v_gtc_iec(),  v.v_tmp(),  cb_c, tb_c))
            self._emit(tc_index_dispatcher(v.v_wei_ik(), v.v_tmp(), cb_k1, tb_k1, True))
        else:
            assert False, "unimplemented"
        self._emit_empty_line()

        if self.tunable.precision == 'int8':
            self._emit(f"s_mov_b32 s[{s.s_0xff()}], 0xff")
        if self.tunable.merge_e == 1:
            self._emit(f"s_mov_b32 s[{s.s_tmp()}], {0x00ffffff}")

        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit_empty_line()
        self._emit(f"; calculate index")
        if self.tunable.merge_e == 1:
            self._emit(f"s_lshr_b32 s[{s.s_move_slice_k_y()}], s[{s.s_y()}], 24")
            self._emit(f"s_lshr_b32 s[{s.s_move_slice_k_x()}], s[{s.s_x()}], 24")
            self._emit(f"s_lshr_b32 s[{s.s_move_slice_k_c()}], s[{s.s_c()}], 24")
            self._emit(f"s_and_b32 s[{s.s_y()}], s[{s.s_tmp()}], s[{s.s_y()}]")
            self._emit(f"s_and_b32 s[{s.s_x()}], s[{s.s_tmp()}], s[{s.s_x()}]")
            self._emit(f"s_and_b32 s[{s.s_c()}], s[{s.s_tmp()}], s[{s.s_c()}]")

            # re-dispatch gtc_ic as gtc_iec
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_c()}], s[{s.s_x()}]")
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_1()}], 0x00080000 ; offset:0, width:8")
                self._emit(m_mdiv_u32_vs(v.v_tmp(4), v.v_gtc_iy(), v.v_gtc_iec(), s.s_magic_4(), s.s_tmp(3), s.s_tmp(), v.v_tmp()))
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_1()}], 0x00080008 ; offset:8, width:8")
                self._emit(m_mdiv_u32_vs(v.v_gtc_ic(), v.v_gtc_ix(), v.v_tmp(4), s.s_magic_5(), s.s_tmp(3), s.s_c(), v.v_tmp()))

        # calculate stride, not shift data byte yet
        # input
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_lshr_b32 s[{s.s_sub_c()}], s[{s.s_c()}], s[{s.s_gemmk_split()}] ;add gkgs for c")
        self._emit(f"s_mul_i32 s[{s.s_in_stride_wi()}], s[{s.s_c()}], s[{s.s_group()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_wi()}], s[{s.s_in_stride_wi()}]")
        self._emit(f"s_mul_i32 s[{s.s_in_stride_n()}], s[{s.s_hi()}], s[{s.s_tmp(2)}]")

        # weight
        if self.tunable.nxe != 0:
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_x()}], s[{s.s_c()}]")
            self._emit(f"s_mul_i32 s[{s.s_wei_stride_k()}], s[{s.s_tmp()}], s[{s.s_y()}]")
        else:
            self._emit(f"s_mov_b32 s[{s.s_wei_stride_k()}], s[{s.s_c()}]")
        
        if tb_k0 != 1:
            self._emit(f"s_lshl_b32 s[{s.s_wei_stride_k0()}], s[{s.s_wei_stride_k()}], {igemm_log2(nb_k1)}")

        if self.tunable.merge_e == 1:
            self._emit(f"s_mov_b32 s[{s.s_y_x_c()}], s[{s.s_wei_stride_k()}]")

        # output
        self._emit(f"s_mul_i32 s[{s.s_out_stride_wo()}], s[{s.s_k()}], s[{s.s_group()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_wo() if self.tunable.nxe != 0 else s.s_wi()}], s[{s.s_out_stride_wo()}]")
        self._emit(f"s_mul_i32 s[{s.s_out_stride_n()}], s[{s.s_ho() if self.tunable.nxe != 0 else s.s_hi()}], s[{s.s_tmp(1)}]")

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
        if self.is_pad_c() and self.tunable.merge_e == 0:
            self._emit(f"s_add_u32 s[{s.s_tmp(2)}], {self.tunable.gemm_k_per_block - 1}, s[{s.s_c()}]")
            self._emit(f"s_lshr_b32 s[{s.s_c_padded()}], s[{s.s_tmp(2)}], {igemm_log2(self.tunable.gemm_k_per_block)}")
            self._emit(f"s_lshl_b32 s[{s.s_c_padded()}], s[{s.s_c_padded()}], {igemm_log2(self.tunable.gemm_k_per_block)}")
            if self.tunable.nxe != 0:
                self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_x()}], s[{s.s_y()}]")
                self._emit(f"s_mul_i32 s[{s.s_knum()}], s[{s.s_tmp()}], s[{s.s_c_padded()}]")
            else:
                self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_c_padded()}]")
            if self.tunable.gemm_k_global_split:
                self._emit(f"s_lshr_b32 s[{s.s_knum()}], s[{s.s_knum()}], s[{s.s_gemmk_split()}]")
        elif self.tunable.merge_e == 1:
            self._emit(f"s_add_u32 s[{s.s_tmp()}], {self.tunable.gemm_k_per_block - 1}, s[{s.s_wei_stride_k()}]")
            self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_k_per_block)}")
            self._emit(f"s_lshl_b32 s[{s.s_knum()}], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_k_per_block)}")
        else:
            if self.tunable.gemm_k_global_split:
                self._emit(f"s_lshr_b32 s[{s.s_knum()}], s[{s.s_wei_stride_k()}], s[{s.s_gemmk_split()}]")
            else:
                self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_wei_stride_k()}]")

        # pad gemm_m, gemm_n
        if self.tunable.nxe != 0:
            self._emit(f"s_mul_i32 s[{s.s_dim_br()}], s[{s.s_ho()}], s[{s.s_wo()}]")
        else:
            self._emit(f"s_mul_i32 s[{s.s_dim_br()}], s[{s.s_hi()}], s[{s.s_wi()}]")

        self._emit(f"s_mul_i32 s[{s.s_dim_mr()}], s[{s.s_n()}], s[{s.s_dim_br()}]")
        if self.tunable.merge_e == 1:
            self._emit(f"v_mul_u32_u24 v[{v.v_sst_a_os()}], s[{s.s_dilation_h()}], v[{v.v_gtc_iy()}]")            # iy * s_dilation_h
        self._emit(f"s_add_u32 s[{s.s_tmp()}], {self.tunable.gemm_m_per_block - 1}, s[{s.s_dim_mr()}]")
        if self.tunable.merge_e == 1:
            self._emit(f"v_subrev_u32 v[{v.v_sst_a_os()}], s[{s.s_pad_h()}], v[{v.v_sst_a_os()}]")                # iy * s_dilation_h - s_pad_h
        self._emit(f"s_lshr_b32 s[{s.s_tmp(1)}], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_m_per_block)}")
        if self.tunable.merge_e == 1:
            self._emit(f"v_mul_u32_u24 v[{v.v_sld_a_os()}], s[{s.s_dilation_w()}], v[{v.v_gtc_ix()}]")            # ix * s_dilation_w
        self._emit(f"s_lshl_b32 s[{s.s_dim_mp()}], s[{s.s_tmp(1)}], {igemm_log2(self.tunable.gemm_m_per_block)}")
        if self.tunable.merge_e == 1:
            self._emit(f"v_subrev_u32 v[{v.v_sld_a_os()}], s[{s.s_pad_w()}], v[{v.v_sld_a_os()}]")                # ix * s_dilation_w - s_pad_w

        self._emit(f"s_add_u32 s[{s.s_tmp()}], {self.tunable.gemm_n_per_block - 1}, s[{s.s_k()}]")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(1)}], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_n_per_block)}")
        self._emit(f"s_lshl_b32 s[{s.s_dim_np()}], s[{s.s_tmp(1)}], {igemm_log2(self.tunable.gemm_n_per_block)}")

        self._emit_empty_line()
        self._emit(f"; gemm_m_per_block:{self.tunable.gemm_m_per_block}, gemm_n_per_block:{self.tunable.gemm_n_per_block}, source_access_order:{self.tunable.source_access_order}")

        # calculate group index
        self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_dim_mp()}], {igemm_log2(self.tunable.gemm_m_per_block)}")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(1)}], s[{s.s_dim_np()}], {igemm_log2(self.tunable.gemm_n_per_block)}")
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
            self._emit(f"s_lshr_b32 s[0], s[{s.s_dim_np()}], {igemm_log2(self.tunable.gemm_n_per_block)}")
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080000 ; offset:0, width:8")
                self._emit(m_mdiv_u32_ss(s.s_tmp(4), s.s_tmp(5), s.s_bx(), s.s_magic_0(), s.s_tmp(3), '0', s.s_tmp()))
            else:
                self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_tmp(5), s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))

        else:
            self._emit(f"s_lshr_b32 s[0], s[{s.s_dim_mp()}], {igemm_log2(self.tunable.gemm_m_per_block)}")
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self._emit(f"s_bfe_u32 s[{s.s_tmp(3)}], s[{s.s_shift_pack_0()}], 0x00080000 ; offset:0, width:8")
                self._emit(m_mdiv_u32_ss(s.s_tmp(5), s.s_tmp(4), s.s_bx(), s.s_magic_0(), s.s_tmp(3), '0', s.s_tmp()))
            else:
                self._emit(m_int_div_rem_ss(s.s_tmp(5), s.s_tmp(4), s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))

        self._emit(f"; s_tmp+4:block_gtc_in, s_tmp+5:block_gtc_im")
        self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ik()}], s[{s.s_tmp(4)}], {igemm_log2(self.tunable.gemm_n_per_block)}")
        self._emit(f"s_lshl_b32 s[{s.s_block_gtc_inb()}], s[{s.s_tmp(5)}], {igemm_log2(self.tunable.gemm_m_per_block)}")

        # transform nb
        self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_inb()}], v[{v.v_in_inb()}]")
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
            if self.tunable.merge_e == 0:
                self._emit(f"v_sub_i32 v[{v.v_in_ihi_list(0)}], v[{v.v_in_ihi_list(0)}], s[{s.s_pad_h()}]")
            else:
                self._emit(f"v_add_u32 v[{v.v_in_ihi_list(0)}], v[{v.v_in_ihi_list(0)}], v[{v.v_sst_a_os()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_in_iwi_list(0)}], s[{s.s_stride_w()}], v[{v.v_in_iwi_list(0)}]")
            if self.tunable.merge_e == 0:
                self._emit(f"v_sub_i32 v[{v.v_in_iwi_list(0)}], v[{v.v_in_iwi_list(0)}], s[{s.s_pad_w()}]")
            else:
                self._emit(f"v_add_u32 v[{v.v_in_iwi_list(0)}], v[{v.v_in_iwi_list(0)}], v[{v.v_sld_a_os()}]")
            self._emit_empty_line()

        if IGEMM_FWD_GTC_NHWC_PACK_IN_FLAG:
            # update flag for batch size
            self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_n()}], v[{v.v_in_in()}]")
            self._emit(f"v_cndmask_b32 v[{v.v_tmp()}], 0, 1, vcc")
            if self.is_pad_c():
                if self.tunable.merge_e == 0:
                    self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_c()}], v[{v.v_gtc_ic_a() if self.tunable.tensor_a_pass_through else v.v_gtc_ic()}]")
                else:
                    self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_y_x_c()}], v[{v.v_gtc_iec()}]")
                self._emit(f"v_cndmask_b32 v[{v.v_tmp(1)}], 0, 1, vcc")
                self._emit(f"v_and_b32 v[{v.v_tmp()}], v[{v.v_tmp(1)}], v[{v.v_tmp()}]")
            self._emit(f"v_lshlrev_b32 v[{v.v_in_flag(0)}], 16, v[{v.v_tmp()}]")
        else:
            self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_n()}], v[{v.v_in_in()}]")
            self._emit(f"v_cndmask_b32 v[{v.v_tmp()}], 0, 1, vcc")
            if self.is_pad_c():
                if self.tunable.merge_e == 0:
                    self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_c()}], v[{v.v_gtc_ic_a() if self.tunable.tensor_a_pass_through else v.v_gtc_ic()}]")
                else:
                    self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_y_x_c()}], v[{v.v_gtc_iec()}]")
                self._emit(f"v_cndmask_b32 v[{v.v_tmp(1)}], 0, 1, vcc")
                self._emit(f"v_and_b32 v[{v.v_tmp()}], v[{v.v_tmp(1)}], v[{v.v_tmp()}]")
            self._emit(f"v_lshlrev_b32 v[{v.v_in_flag_n()}], 0, v[{v.v_tmp()}]")

        self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ig()}], s[{s.s_block_gtc_ig()}], {igemm_log2(data_byte)}")

        def calculate_and_load_input():
            self._emit(f"; calculate in offset")
            if self.tunable.tensor_a_pass_through:
                self._emit(f"s_mov_b32 s[{s.s_in_c_itr()}], 0")
                # self._emit(f"s_mov_b32 s[{s.s_in_stride_k_pack()}], {ca_c * k_pack * data_byte}")
                # for i in range(2, ta_c // k_pack):
                #     self._emit(f"s_mul_i32 s[{s.s_in_offset(i - 2)}], s[{s.s_in_stride_k_pack()}], {i}")
            else:
                self._emit(f"s_mov_b32 s[{s.s_in_offset()}], 0")
            # compute group distance
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_block_gtc_ig()}], s[{s.s_c()}]")
            self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_block_gtc_ig()}], s[{s.s_c()}]")
            self._emit(f"s_add_u32 s[{s.s_p_in(0)}], s[{s.s_p_in(0)}], s[{s.s_tmp()}]")
            self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], s[{s.s_tmp(1)}]")
            self._emit_empty_line()

            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_in_stride_n()}], v[{v.v_in_in()}]")
            # s_in_stride_wi need shift before!
            self._emit(self.try_shift_stride(s.s_in_stride_wi, igemm_log2(data_byte)))

            if self.tunable.gemm_k_global_split:
                self._emit(f"v_add_u32 v[{v.v_tmp(1)}], v[{v.v_tmp(1)}], s[{s.s_block_gtc_ic()}]")

            self._emit(f"v_add_lshl_u32 v[{v.v_tmp(4)}], v[{v.v_gtc_ic_a() if self.tunable.tensor_a_pass_through else v.v_gtc_ic()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wi()}], v[{v.v_in_ihi_list(0)}]")
            self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_in_iwi_list(0)}], v[{v.v_tmp()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_in_stride_wi()}], v[{v.v_tmp()}]")
            self._emit(f"v_add_u32 v[{v.v_in_os()}], v[{v.v_tmp(4)}], v[{v.v_tmp()}]")

            if True:
                if IGEMM_FWD_GTC_NHWC_PACK_IN_FLAG:
                    self._emit(f"v_bfe_u32 v[{v.v_tmp(1)}], v[{v.v_in_flag()}],  16, 1")
                    self._emit(m_set_flag_nhw(v.v_tmp(), v.v_tmp(1), v.v_in_ihi_list(0), v.v_in_iwi_list(0), s.s_hi(), s.s_wi()))
                    self._emit(f"v_lshl_or_b32 v[{v.v_in_flag()}], v[{v.v_tmp()}], 0,  v[{v.v_in_flag()}]")
                else:
                    self._emit(f"v_bfe_u32 v[{v.v_tmp(1)}], v[{v.v_in_flag_n()}],  0, 1")
                    self._emit(m_set_flag_nhw(v.v_in_flag(0), v.v_tmp(1), v.v_in_ihi_list(0), v.v_in_iwi_list(0), s.s_hi(), s.s_wi()))
            self._emit_empty_line()

            # voffset, for [1, nb_per_thread) pixels
            if self.tunable.tensor_a_pass_through:
                thread_stride = ca_nb0 * ca_nb1
            else:
                thread_stride = na_nb1 if ta_nb0 != 1 else 1

            for i in range(1, nb_per_thread):
                self._emit(f"s_mov_b32 s1, {thread_stride * i}")
                self._emit(f"v_add_u32 v[{v.v_tmp()}], s1, v[{v.v_in_inb()}]")
                self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_inb()}], v[{v.v_tmp()}]")
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
                    if self.tunable.merge_e == 0:
                        self._emit(f"v_sub_i32 v[{v.v_in_ihi_list(i)}], v[{v.v_in_ihi_list(i)}], s[{s.s_pad_h()}]")
                    else:
                        self._emit(f"v_add_u32 v[{v.v_in_ihi_list(i)}], v[{v.v_in_ihi_list(i)}], v[{v.v_sst_a_os()}]")
                    self._emit(f"v_mul_lo_u32 v[{v.v_in_iwi_list(i)}], s[{s.s_stride_w()}], v[{v.v_in_iwi_list(i)}]")
                    if self.tunable.merge_e == 0:
                        self._emit(f"v_sub_i32 v[{v.v_in_iwi_list(i)}], v[{v.v_in_iwi_list(i)}], s[{s.s_pad_w()}]")
                    else:
                        self._emit(f"v_add_u32 v[{v.v_in_iwi_list(i)}], v[{v.v_in_iwi_list(i)}], v[{v.v_sld_a_os()}]")
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
                    self._emit(f"v_add_u32 v[{v.v_tmp(1)}], v[{v.v_tmp(1)}], s[{s.s_block_gtc_ic()}]")
                self._emit(f"v_add_lshl_u32 v[{v.v_tmp(4)}], v[{v.v_gtc_ic_a() if self.tunable.tensor_a_pass_through else v.v_gtc_ic()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wi()}], v[{v.v_in_ihi_list(i)}]")
                self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_in_iwi_list(i)}], v[{v.v_tmp()}]")
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_in_stride_wi()}], v[{v.v_tmp()}]")
                self._emit(f"v_add_u32 v[{v.v_in_os(i)}], v[{v.v_tmp(4)}], v[{v.v_tmp()}]")

                if IGEMM_FWD_GTC_NHWC_PACK_IN_FLAG:
                    # update flag for batch size
                    self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_n()}], v[{v.v_in_in()}]")
                    self._emit(f"v_cndmask_b32 v[{v.v_tmp()}], 0, 1, vcc")
                    if self.is_pad_c():
                        if self.tunable.merge_e == 0:
                            self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_c()}], v[{v.v_gtc_ic_a() if self.tunable.tensor_a_pass_through else v.v_gtc_ic()}]")
                        else:
                            self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_y_x_c()}], v[{v.v_gtc_iec()}]")
                        self._emit(f"v_cndmask_b32 v[{v.v_tmp(1)}], 0, 1, vcc")
                        self._emit(f"v_and_b32 v[{v.v_tmp()}], v[{v.v_tmp(1)}], v[{v.v_tmp()}]")
                    self._emit(f"v_lshl_or_b32 v[{v.v_in_flag()}], v[{v.v_tmp()}], {16 + i}, v[{v.v_in_flag(0)}]")
                    self._emit(m_set_flag_nhw(v.v_tmp(1), v.v_tmp(), v.v_in_ihi_list(i), v.v_in_iwi_list(i), s.s_hi(), s.s_wi()))
                    self._emit(f"v_lshl_or_b32 v[{v.v_in_flag()}], v[{v.v_tmp(1)}], {i}, v[{v.v_in_flag()}]")
                else:
                    self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_n()}], v[{v.v_in_in()}]")
                    self._emit(f"v_cndmask_b32 v[{v.v_tmp()}], 0, 1, vcc")
                    if self.is_pad_c():
                        if self.tunable.merge_e == 0:
                            self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_c()}], v[{v.v_gtc_ic_a() if self.tunable.tensor_a_pass_through else v.v_gtc_ic()}]")
                        else:
                            self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_y_x_c()}], v[{v.v_gtc_iec()}]")
                        self._emit(f"v_cndmask_b32 v[{v.v_tmp(1)}], 0, 1, vcc")
                        self._emit(f"v_and_b32 v[{v.v_tmp()}], v[{v.v_tmp(1)}], v[{v.v_tmp()}]")
                    self._emit(f"v_lshl_or_b32 v[{v.v_in_flag_n()}], v[{v.v_tmp()}], {i}, v[{v.v_in_flag_n()}]")
                    self._emit(m_set_flag_nhw(v.v_in_flag(i), v.v_tmp(), v.v_in_ihi_list(i), v.v_in_iwi_list(i), s.s_hi(), s.s_wi()))

            # load in
            self._emit(f"s_mov_b32 s[{s.s_p_in(2)}], 0xffffffff")
            self._emit(f"s_mov_b32 s[{s.s_p_in(3)}], 0x27000")
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
            self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_k()}], s[{s.s_wei_stride_k()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
            self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_block_gtc_ig()}], s[{s.s_tmp(2)}]")
            self._emit(f"s_add_u32 s[{s.s_p_wei()}], s[{s.s_p_wei()}], s[{s.s_tmp()}]")
            self._emit(f"s_addc_u32 s[{s.s_p_wei(1)}], s[{s.s_p_wei(1)}], s[{s.s_tmp(1)}]")

            self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_ik()}], v[{v.v_wei_ik()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_k()}], v[{v.v_tmp(5)}]")
            if self.tunable.gemm_k_global_split:
                self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_tmp()}], s[{s.s_block_gtc_ic()}]")
            self._emit(f"v_add_lshl_u32 v[{v.v_wei_os()}], v[{v.v_tmp()}], v[{v.v_gtc_ic() if self.tunable.merge_e == 0 else v.v_gtc_iec()}], {igemm_log2(data_byte)}")

            # wei flag
            self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_k()}], v[{v.v_tmp(5)}]")
            self._emit(f"v_cndmask_b32 v[{v.v_wei_flag()}], 0, 1, vcc")
            if self.is_pad_c():
                if self.tunable.merge_e == 0:
                    self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_c()}], v[{v.v_gtc_ic_b() if self.tunable.tensor_b_pass_through else v.v_gtc_ic()}]")
                else:
                    self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_y_x_c()}], v[{v.v_gtc_iec()}]")
                self._emit(f"v_cndmask_b32 v[{v.v_wei_flag()}], 0, v[{v.v_wei_flag()}], vcc")
            self._emit(f"v_mov_b32 v[{v.v_wei_tmp_pack()}], v[{v.v_wei_flag()}]")

            for i in range(1, nk_per_thread):
                if i == 1:
                    k_thread_stride = nb_k1 if tb_k0 != 1 else 1
                    self._emit(f"s_mov_b32 s[{s.s_tmp()}], {k_thread_stride}")
                self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_tmp()}], v[{v.v_tmp(5)}]")
                self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_k()}], v[{v.v_tmp(5)}]")
                self._emit(f"v_cndmask_b32 v[{v.v_wei_flag(i)}], 0, 1, vcc")
                if self.is_pad_c():
                    if self.tunable.merge_e == 0:
                        self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_c()}], v[{v.v_gtc_ic_b() if self.tunable.tensor_b_pass_through else v.v_gtc_ic()}]")
                    else:
                        self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_y_x_c()}], v[{v.v_gtc_iec()}]")
                    self._emit(f"v_cndmask_b32 v[{v.v_wei_flag(i)}], 0, v[{v.v_wei_flag(i)}], vcc")
                self._emit(f"v_lshl_or_b32 v[{v.v_wei_tmp_pack()}], v[{v.v_wei_flag(i)}], {i}, v[{v.v_wei_tmp_pack()}]")

            self._emit_empty_line()
            if self.wei_thread_copy_ndim != 1:
                if s_wei_stride_d0 != s_dummy:
                    self._emit(self.try_shift_stride(s_wei_stride_d0, igemm_log2(data_byte)))
            if s_wei_stride_d1 != s_dummy:
                self._emit(self.try_shift_stride(s_wei_stride_d1, igemm_log2(data_byte)))
            self._emit_empty_line()

            if self.tunable.precache_soffset:
                self._emit(m_wei_2d_global_load.init_precache_soffset(s_wei_stride_d0(), s_wei_stride_d1(), s.s_wei_offset(), s.s_tmp()))

            self._emit(f".v_clear_nc {v.v_gld_b()}, {self.get_num_vgpr_global_load_b()}")
            self._emit(f"s_mov_b32 s[{s.s_p_wei(2)}], 0xffffffff")
            self._emit(f"s_mov_b32 s[{s.s_p_wei(3)}], 0x27000")
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
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], v0")
            self._emit(self.thread_mapping(v.v_gemm_in(), v.v_gemm_im(), v.v_tmp(5), v.v_tmp()))
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
        v_igemm_k = v.v_gtc_ic if self.tunable.merge_e == 0 else v.v_gtc_iec
        if not self.tunable.tensor_a_pass_through:
            self._emit(f"; LDS store, in: e,c,nb0,nb1: {ta_e}x{ta_c}x{ta_nb0}x{ta_nb1}, {ca_e}x{ca_c}x{ca_nb0}x{ca_nb1}, k_pack:{k_pack}, k_pack_gld_a:{k_pack_gld_a}, {self.tunable.precision}")
            if k_pack_src_mat != 1:
                self._emit(f"v_lshlrev_b32 v[{v.v_tmp(2)}], {igemm_log2(k_pack_src_mat)},  v[{v.v_in_inb()}]")
                self._emit(f"v_lshrrev_b32 v[{v.v_tmp(1)}], {igemm_log2(k_pack_src_mat)},  v[{v_igemm_k()}]")
                self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(na_nb0*na_nb1 * k_pack_src_mat)}, v[{v.v_tmp(2)}]")
                if k_pack_src_mat != k_pack_gld_a:
                    assert k_pack_src_mat % k_pack_gld_a == 0
                    self._emit(f"v_and_b32 v[{v.v_tmp(2)}], {k_pack_src_mat - 1}, v[{v_igemm_k()}]")   # gld_a k_pack_src_mat smaller than k_pack_src_mat
                    self._emit(f"v_or_b32 v[{v.v_tmp()}], v[{v.v_tmp()}], v[{v.v_tmp(2)}]")
            else:
                self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v_igemm_k()}], {igemm_log2(na_nb0*na_nb1 * k_pack_src_mat)}, v[{v.v_in_inb()}]")
            self._emit(f"v_lshlrev_b32 v[{v.v_sst_a_os()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")
            self._emit_empty_line()
            self._emit(f"v_lshlrev_b32 v[{v.v_sld_a_os()}], {igemm_log2(data_byte)}, v[{v.v_gemm_im()}] ; LDS load in")

        if not self.tunable.tensor_b_pass_through:
            self._emit(f"; LDS store, wei: e,c,k: {tb_e}x{tb_c}x{tb_k0}x{tb_k1}, {cb_e}x{cb_c}x{cb_k0}x{cb_k1}, k_pack:{k_pack}, k_pack_gld_b:{k_pack_gld_b}, {self.tunable.precision}")
            if k_pack_src_mat != 1:
                self._emit(f"v_lshlrev_b32 v[{v.v_tmp(2)}], {igemm_log2(k_pack_src_mat)},  v[{v.v_wei_ik()}]")
                self._emit(f"v_lshrrev_b32 v[{v.v_tmp(1)}], {igemm_log2(k_pack_src_mat)},  v[{v_igemm_k()}]")
                self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(nb_k0*nb_k1 * k_pack_src_mat)}, v[{v.v_tmp(2)}]")
                if k_pack_src_mat != k_pack_gld_b:
                    assert k_pack_src_mat % k_pack_gld_b == 0
                    self._emit(f"v_and_b32 v[{v.v_tmp(2)}], {k_pack_src_mat - 1}, v[{v_igemm_k()}]")   # gld_b k_pack_src_mat smaller than k_pack_src_mat
                    self._emit(f"v_or_b32 v[{v.v_tmp()}], v[{v.v_tmp()}], v[{v.v_tmp(2)}]")
            else:
                self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v_igemm_k()}], {igemm_log2(nb_k0*nb_k1 * k_pack_src_mat)}, v[{v.v_wei_ik()}]")
            self._emit(f"v_lshlrev_b32 v[{v.v_sst_b_os()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")
            if not self.tunable.tensor_a_pass_through:
                self._emit(f"v_add_u32 v[{v.v_sst_b_os()}], {self.tunable.lds_a_np2}, v[{v.v_sst_b_os()}]")
            self._emit_empty_line()
            self._emit(f"v_lshlrev_b32 v[{v.v_sld_b_os()}], {igemm_log2(data_byte)}, v[{v.v_gemm_in()}] ; LDS load wei")
            if not self.tunable.tensor_a_pass_through:
                self._emit(f"v_add_u32 v[{v.v_sld_b_os()}], {self.tunable.lds_a_np2}, v[{v.v_sld_b_os()}]")

        # self._emit(f"; LDS load")
        #if not self.tunable.tensor_a_pass_through:
        #self._emit(f"v_lshlrev_b32 v[{v.v_sld_b_os()}], {igemm_log2(data_byte)}, v[{v.v_gemm_in()}]")
        #self._emit(f"v_lshlrev_b32 v[{v.v_sld_a_os()}], {igemm_log2(data_byte)}, v[{v.v_gemm_im()}]")
        #self._emit(f"v_add_u32 v[{v.v_sld_b_os()}], {self.tunable.lds_a_np2}, v[{v.v_sld_b_os()}]")
        #self._emit_empty_line()

        if self.tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            self._emit(f"v_mov_b32 v[{v.v_gemm_in()}], v[{v.v_co_sst()}]")
            self._emit(f"v_mov_b32 v[{v.v_gemm_im()}], v[{v.v_co_sld()}]")
        self._emit(self.coalescing_store.init_co_lds_offset(v.v_co_sst(), v.v_co_sld(), v.v_gemm_im(), v.v_gemm_in(), '0', v.v_tmp()))
        self._emit(self.coalescing_store.init_co_sub_m_index(v.v_co_sub_m_index(), '0', v.v_tmp()))
        self._emit(self.coalescing_store.init_co_sub_n_index(v.v_co_sub_n_index(), '0', v.v_tmp()))
        self._emit_empty_line()

        #if self.tunable.nxe != 0:
        if True:
            self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_block_gtc_ik()}], v[{v.v_co_sub_n_index()}]")
            self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_k()}], v[{v.v_tmp()}]")
            self._emit(f"v_cndmask_b32 v[{v.v_out_flag()}], 0, 1, vcc")

        '''
        a good news for nhwc and coalescing output is that, we can treat gemm_m (n*ho*wo) as a single dimension,
        and use sgpr to stride along this dimension. this is much easier
        '''
        self._emit(f"; output offset")
        if self.tunable.use_fp32_atomic_add_for_fp16_data:
            # s_block_gtc_ig = ig*2, but for wei workspace, s_block_gtc_ig need to be ig*4, so here we give it a (*2)
            self._emit(f"s_mul_i32 s[{s.s_block_gtc_ig()}], s[{s.s_block_gtc_ig()}], 2")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_block_gtc_ig()}], s[{s.s_k()}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_block_gtc_ig()}], s[{s.s_k()}]")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], s[{s.s_tmp(1)}]")

        self._emit_empty_line()
        if self.tunable.use_fp32_atomic_add_for_fp16_data:
            self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], s[{s.s_block_gtc_ik()}], 2")
        else:
            self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], s[{s.s_block_gtc_ik()}], {igemm_log2(data_byte)}")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp(3)}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out()}+1], 0")
        self._emit_empty_line()

        if self.tunable.use_fp32_atomic_add_for_fp16_data:
            self._emit(self.try_shift_stride(s.s_out_stride_wo, 2))
        else:
            self._emit(self.try_shift_stride(s.s_out_stride_wo, igemm_log2(data_byte)))
        self._emit(f"v_add_u32 v[{v.v_out_inb()}], s[{s.s_block_gtc_inb()}], v[{v.v_co_sub_m_index()}]   ; total n*ho*wo")
        self._emit(f"v_mul_lo_u32 v[{v.v_out_os()}], s[{s.s_out_stride_wo()}], v[{v.v_out_inb()}]")
        if self.tunable.use_fp32_atomic_add_for_fp16_data:
            self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], 2, v[{v.v_co_sub_n_index()}]")
        else:
            self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(data_byte)}, v[{v.v_co_sub_n_index()}]")
        self._emit(f"v_add_u32 v[{v.v_out_os()}], v[{v.v_out_os()}], v[{v.v_tmp()}]")

        self._emit(f"; move slice stride")
        if self.tunable.merge_e == 0:
            if self.tunable.gemm_k_global_split:
                if self.is_pad_c():
                    self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_c_padded()}], s[{s.s_gemmk_split()}] ;add gkgs for c")
                    self._emit(f"s_lshl_b32 s[{s.s_gemm_k_num_c()}], s[{s.s_tmp()}], {igemm_log2(data_byte)}")
                else:
                    self._emit(f"s_lshl_b32 s[{s.s_gemm_k_num_c()}], s[{s.s_sub_c()}], {igemm_log2(data_byte)}")
                self._emit(f"s_lshl_b32 s[{s.s_tmp()}], s[{s.s_c()}], {igemm_log2(data_byte)}")
                self._emit(f"s_sub_u32  s[{s.s_gemm_k_diff_c()}],  s[{s.s_tmp()}], s[{s.s_gemm_k_num_c()}]")
            else:
                self._emit(f"s_lshl_b32 s[{s.s_gemm_k_num_c()}], s[{s.s_c_padded() if self.is_pad_c() else s.s_c()}], {igemm_log2(data_byte)}")
                if self.is_pad_c():
                    self._emit(f"s_lshl_b32 s[{s.s_tmp()}], s[{s.s_c()}], {igemm_log2(data_byte)}")
                    self._emit(f"s_sub_u32  s[{s.s_gemm_k_diff_c()}],  s[{s.s_tmp()}], s[{s.s_gemm_k_num_c()}]")

        w_flag_cnt = 0
        self._emit(f"v_bfe_u32 v[{v.v_wei_flag(0)}], v[{v.v_wei_tmp_pack()}], {0}, 1")
        w_flag_cnt = w_flag_cnt + 1

        # if self.tunable.nxe != 0:
        #     self._emit(f"s_mov_b32 s[{s.s_tmp()}], {na_c}")
        #     self._emit(f"s_mul_i32 s[{s.s_move_slice_k_stride_c()}], s[{s.s_tmp()}], {igemm_log2(data_byte)}")
        # else:
        if self.tunable.merge_e == 0:
            self._emit(f"s_mov_b32 s[{s.s_move_slice_k_stride_c()}], {na_c * data_byte}")
        else:
            self._emit(f"s_mov_b32 s[{s.s_move_slice_k_stride_gemm_k()}], {self.tunable.gemm_k_per_block * data_byte}")

        if w_flag_cnt < nk_per_thread:
            self._emit(f"v_bfe_u32 v[{v.v_wei_flag(w_flag_cnt)}], v[{v.v_wei_tmp_pack()}], {w_flag_cnt}, 1")
            w_flag_cnt = w_flag_cnt + 1

        if self.tunable.nxe != 0 and self.tunable.merge_e == 0:
            # s_in_diff_wi   : s_dilation_w * s_in_stride_wi
            # s_in_diff_hi   : s_dilation_h * s_in_stride_hi - (x - 1) * s_dilation_w * s_in_stride_wi, always possitive
            # s_dilation_w_x : -1* (x - 1) * s_dilation_w
            self._emit(f"s_mov_b32 s[{s.s_move_slice_k_ix()}], 0")
            self._emit(f"s_mul_i32 s[{s.s_in_diff_wi()}], s[{s.s_dilation_w()}], s[{s.s_in_stride_wi()}]") # shifted
            self._emit(f"s_sub_i32 s[{s.s_tmp(3)}], s[{s.s_x()}], 1")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_in_diff_wi()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_in_stride_wi()}], s[{s.s_wi()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_tmp(1)}], s[{s.s_dilation_h()}]")
            self._emit(f"s_sub_i32 s[{s.s_in_diff_hi()}], s[{s.s_tmp(1)}], s[{s.s_tmp()}]")
            self._emit(f"s_mul_i32 s[{s.s_dilation_w_x()}], s[{s.s_dilation_w()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_mul_i32 s[{s.s_dilation_w_x()}], s[{s.s_dilation_w_x()}], -1")

        if self.tunable.nxe != 0 and self.tunable.merge_e == 1:
            # s_diff_in_os_acc_y_x_c   : s_move_slice_k_c * data_byte + s_move_slice_k_x * s_dilation_w * in_stride_wi + s_move_slice_k_y * s_dilation_h * in_stride_hi
            # s_diff_in_os_ovf_c_acc_x : -s_c * data_byte + s_dilation_w * in_stride_wi
            # s_diff_in_os_ovf_x_acc_y : -s_x * s_dilation_w * in_stride_wi + s_dilation_h * in_stride_hi
            # s_diff_in_iwi_acc_x      : s_move_slice_k_x * s_dilation_w
            # s_diff_in_iwi_ovf_x      : s_x * s_dilation_w
            # s_diff_in_ihi_acc_y      : s_move_slice_k_y * s_dilation_h
            self._emit(self.try_shift_stride(s.s_in_stride_wi, igemm_log2(data_byte)))
            self._emit(f"s_mul_i32 s[{s.s_tmp(5)}], s[{s.s_wi()}], s[{s.s_in_stride_wi()}]")                # in_stride_hi
            self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_dilation_w()}], s[{s.s_in_stride_wi()}]")        # s_dilation_w * in_stride_wi
            self._emit(f"s_lshl_b32 s[{s.s_tmp(1)}], s[{s.s_c()}], {igemm_log2(data_byte)}")
            self._emit(f"s_sub_i32 s[{s.s_diff_in_os_ovf_c_acc_x()}], s[{s.s_tmp(0)}], s[{s.s_tmp(1)}]")
            self._emit(f"s_mul_i32 s[{s.s_diff_in_iwi_acc_x()}], s[{s.s_move_slice_k_x()}], s[{s.s_dilation_w()}]")
            self._emit(f"s_mul_i32 s[{s.s_diff_in_iwi_ovf_x()}], s[{s.s_x()}], s[{s.s_dilation_w()}]")
            self._emit(f"s_mul_i32 s[{s.s_diff_in_ihi_acc_y()}], s[{s.s_move_slice_k_y()}], s[{s.s_dilation_h()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp(5)}], s[{s.s_tmp(5)}], s[{s.s_dilation_h()}]")                # s_dilation_h * in_stride_hi
            self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(0)}], s[{s.s_move_slice_k_x()}]")            # s_move_slice_k_x * s_dilation_w * in_stride_wi
            self._emit(f"s_lshl_b32 s[{s.s_tmp(1)}], s[{s.s_move_slice_k_c()}], {igemm_log2(data_byte)}")   # s_move_slice_k_c * data_byte 
            self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_diff_in_ihi_acc_y()}], s[{s.s_tmp(5)}]")         # s_move_slice_k_y * s_dilation_h * in_stride_hi
            self._emit(f"s_add_u32 s[{s.s_diff_in_os_acc_y_x_c()}], s[{s.s_tmp(0)}], s[{s.s_tmp(1)}]")
            self._emit(f"s_add_u32 s[{s.s_diff_in_os_acc_y_x_c()}], s[{s.s_diff_in_os_acc_y_x_c()}], s[{s.s_tmp(2)}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_diff_in_iwi_ovf_x()}], s[{s.s_in_stride_wi()}]") # s_x * s_dilation_w * in_stride_wi
            self._emit(f"s_sub_i32 s[{s.s_diff_in_os_ovf_x_acc_y()}], s[{s.s_tmp(5)}], s[{s.s_tmp(0)}]")
            # if s.s_wei_stride_k.label in self.dict_shifted_stride:
            #     self._emit(f"s_lshr_b32 s[{s.s_y_x_c()}], s[{s.s_wei_stride_k()}], {igemm_log2(data_byte)}")
            # else:
            #     self._emit(f"s_mov_b32 s[{s.s_y_x_c()}], s[{s.s_wei_stride_k()}]")

        self._emit_empty_line()

        self._emit(f"s_mov_b32 s[{s.s_p_out(2)}], 0xffffffff")
        if w_flag_cnt < nk_per_thread:
            self._emit(f"v_bfe_u32 v[{v.v_wei_flag(w_flag_cnt)}], v[{v.v_wei_tmp_pack()}], {w_flag_cnt}, 1")
            w_flag_cnt = w_flag_cnt + 1
        self._emit(f"s_mov_b32 s[{s.s_p_out(3)}], 0x27000")
        for i_w in range(w_flag_cnt, nk_per_thread):
            self._emit(f"v_bfe_u32 v[{v.v_wei_flag(i_w)}], v[{v.v_wei_tmp_pack()}], {i_w}, 1")

        if self.is_pad_c() and self.tunable.merge_e == 0:
            self._emit(f"v_mov_b32 v[{v.v_gtc_ic_itr()}], v[{v.v_gtc_ic()}]")
            if self.tunable.tensor_a_pass_through:
                self._emit(f"v_mov_b32 v[{v.v_gtc_ic_a_itr()}], v[{v.v_gtc_ic_a()}]")

    def emit_kernel_fma_main_loop(self):
        s = self.sgpr
        v = self.vgpr
        k = self.karg

        data_byte = amdgpu_precision_data_byte(self.tunable.precision)
        k_pack = self.get_k_pack()
        k_pack_lanegroup = self.xdlops_mapping.ctrl.lanegroup_k_per_thread()
        k_pack_src_mat = k_pack if k_pack != 1 else k_pack_lanegroup

        m_move_slice_window             = self.get_macro_move_slice_window()
        m_move_slice_window_accumulate  = self.get_macro_move_slice_window_accumulate()

        def move_slice_window_b():
            '''
            in nhwc we only need call one move slice window
            '''
            if self.tunable.merge_e == 1:
                with self._deferred_context():
                    self._emit(m_move_slice_window(
                                v.v_gtc_ix(), v.v_gtc_ic(),
                                *(v.v_gtc_iec(), s.s_y_x_c(), v.v_wei_flag()) if self.is_pad_c() else (),
                                v.v_in_os(), v.v_in_ihi_list(), v.v_in_iwi_list(), v.v_in_flag(), v.v_in_flag_n(),
                                v.v_wei_os(),
                                s.s_move_slice_k_y(), s.s_move_slice_k_x(), s.s_move_slice_k_c(),
                                s.s_move_slice_k_stride_gemm_k(),
                                s.s_diff_in_os_acc_y_x_c(), s.s_diff_in_os_ovf_c_acc_x(),
                                s.s_diff_in_os_ovf_x_acc_y(), s.s_diff_in_iwi_acc_x(),
                                s.s_dilation_w(),
                                s.s_diff_in_iwi_ovf_x(), s.s_diff_in_ihi_acc_y(),
                                s.s_dilation_h(),
                                v.v_tmp(4), v.v_tmp(5), v.v_gtc_iy(),       # v.v_in_os_diff(), v.v_in_ihi_diff(), v.v_in_iwi_diff(),
                                s.s_x(), s.s_c(), s.s_hi(), s.s_wi()))
                return self._get_deferred()
            elif self.tunable.nxe != 0:
                with self._deferred_context():
                    self._emit(m_move_slice_window(
                                *(s.s_p_in(), s.s_in_c_itr()) if self.tunable.tensor_a_pass_through else (s.s_in_offset(),),
                                *(v.v_gtc_ic_a_itr(),) if self.is_pad_c() and self.tunable.tensor_a_pass_through else (),
                                *(v.v_gtc_ic_itr(), v.v_in_flag(), v.v_wei_flag(), v.v_tmp(), s.s_c()) if self.is_pad_c() else (),
                                v.v_wei_os(),
                                s.s_move_slice_k_stride_c(),
                                s.s_gemm_k_num_c(),
                                s.s_flag_need_acc_yx()))
                return self._get_deferred()
            else:
                with self._deferred_context():
                    self._emit(m_move_slice_window(
                                s.s_p_in() if self.tunable.tensor_a_pass_through else s.s_in_offset(),
                                *(v.v_gtc_ic_a_itr(),) if self.is_pad_c() and self.tunable.tensor_a_pass_through else (),
                                *(v.v_gtc_ic_itr(), v.v_in_flag(), v.v_wei_flag(), v.v_tmp(), s.s_c()) if self.is_pad_c() else (),
                                v.v_wei_os(),
                                s.s_move_slice_k_stride_c()))
                return self._get_deferred()

        def move_slice_window_a():
            return ''

        def move_slice_window_acc():
            if self.tunable.nxe == 0:
                return ''
            elif self.tunable.merge_e == 1:
                return ''
            else:
                with self._deferred_context():
                    if IGEMM_FWD_GTC_NHWC_PACK_IN_FLAG:
                        self._emit(m_move_slice_window_accumulate(
                                    *(s.s_p_in(), s.s_in_c_itr(), s.s_gemm_k_num_c()) if self.tunable.tensor_a_pass_through else (s.s_in_offset(),),
                                    *(v.v_wei_os(), s.s_gemm_k_diff_c(), v.v_in_os()) if self.tunable.gemm_k_global_split or self.is_pad_c() else (v.v_in_os(),),
                                    v.v_in_ihi_list(),
                                    v.v_in_iwi_list(),
                                    v.v_in_flag(),
                                    *(v.v_gtc_ic_a_itr(), v.v_gtc_ic_a()) if self.is_pad_c() and self.tunable.tensor_a_pass_through else (),
                                    *(v.v_gtc_ic_itr(), v.v_gtc_ic(), v.v_wei_flag(), v.v_wei_tmp_pack()) if self.is_pad_c() else (),
                                    s.s_flag_need_acc_yx(),
                                    s.s_move_slice_k_ix(),
                                    s.s_x(),
                                    s.s_in_diff_hi(),
                                    s.s_in_diff_wi(),
                                    s.s_dilation_h(),
                                    s.s_dilation_w(),
                                    s.s_dilation_w_x(),
                                    s.s_hi(),
                                    s.s_wi(),
                                    v.v_tmp(),
                                    s.s_tmp()))
                    else:
                        self._emit(m_move_slice_window_accumulate(
                                    *(s.s_p_in(), s.s_in_c_itr(), s.s_gemm_k_num_c()) if self.tunable.tensor_a_pass_through else (s.s_in_offset(),),
                                    *(v.v_wei_os(), s.s_gemm_k_diff_c(), v.v_in_os()) if self.tunable.gemm_k_global_split or self.is_pad_c() else (v.v_in_os(),),
                                    v.v_in_ihi_list(),
                                    v.v_in_iwi_list(),
                                    v.v_in_flag(),
                                    v.v_in_flag_n(),
                                    *(v.v_gtc_ic_a_itr(), v.v_gtc_ic_a()) if self.is_pad_c() and self.tunable.tensor_a_pass_through else (),
                                    *(v.v_gtc_ic_itr(), v.v_gtc_ic(), v.v_wei_flag(), v.v_wei_tmp_pack()) if self.is_pad_c() else (),
                                    s.s_flag_need_acc_yx(),
                                    s.s_move_slice_k_ix(),
                                    s.s_x(),
                                    s.s_in_diff_hi(),
                                    s.s_in_diff_wi(),
                                    s.s_dilation_h(),
                                    s.s_dilation_w(),
                                    s.s_dilation_w_x(),
                                    s.s_hi(),
                                    s.s_wi(),
                                    v.v_tmp(),
                                    s.s_tmp()))
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
                                                                        self.tunable.precision, bf16_1k_in_fp16 = self.use_bf16_1k_in_fp16())
            fctrl.cxm                         = ctrl_xdlops_mapping
            fctrl.unroll_k                    = self.tunable.gemm_k_per_block
            fctrl.label_prefix                = self.name()
            fctrl.lds_single_size             = self.tunable.lds_single            # in byte, should be power of 2
            fctrl.lds_buffer_num              = self.tunable.lds_buffer_num
            fctrl.local_prefetch_num          = self.tunable.local_prefetch_num
            fctrl.interleave                  = self.tunable.fma_interleave
            fctrl.accvgpr_unified             = self.is_accvgpr_unified()

            # functor
            # fctrl.global_load_a_functor       = self.global_load_wei
            # fctrl.global_load_b_functor       = self.global_load_in
            # fctrl.shared_store_a_functor      = self.shared_store_wei
            # fctrl.shared_store_b_functor      = self.shared_store_in
            fctrl.global_load_a_functor       = self.global_load_in
            fctrl.global_load_b_functor       = self.global_load_wei
            fctrl.shared_store_a_functor      = self.shared_store_in
            fctrl.shared_store_b_functor      = self.shared_store_wei

            # ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
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
            fctrl.move_slice_window_accumule_functor  = move_slice_window_acc if self.tunable.nxe != 0 and self.tunable.merge_e == 0 else None

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
            fctrl.pass_through_bf16_1k_in_fp16  = self.use_bf16_1k_in_fp16()
            fctrl.pass_through_bf16_1k_in_fp16_predefine = self.get_predefine_for_bf16_1k_in_fp16()

            fctrl.pass_through_a_interleave_gld = 1 if self.tunable.tensor_a_pass_through_interleave_gld else 0
            fctrl.pass_through_b_interleave_gld = 1 if self.tunable.tensor_b_pass_through_interleave_gld else 0

            fctrl.precision                   = self.tunable.precision

            mfma_main_loop = mfma_main_loop_t(self.mc, fctrl)
            mfma_main_loop.emit()


    def emit_kernel_epilogue(self):
        s = self.sgpr
        v = self.vgpr
        #label_out = f"L_{self.name()}_out"

        ta_nb0, ta_nb1, ta_e, ta_c, tb_e, tb_c, tb_k0, tb_k1 = self.get_thread_lengths()
        ca_nb0, ca_nb1, ca_e, ca_c, cb_e, cb_c, cb_k0, cb_k1 = self.get_cluster_lengths()

        if self.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            # if self.tunable.nxe != 0:
            #     self._emit(self.coalescing_store(v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_in(), v.v_in_os(), None,
            #         s.s_in_stride_c0() if self.tunable.gemm_m_unmerge_cluster == 1 else None, s.s_stride_c(), s.s_tmp(), v.v_in_flag()))
            # else:
            #     self._emit(self.coalescing_store(v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_in(), v.v_in_os(), None,
            #         s.s_in_stride_c0() if self.tunable.gemm_m_unmerge_cluster == 1 else None, s.s_stride_c(), s.s_tmp()))
            assert False
        else:
            a = self.agpr
            self._emit(self.coalescing_store(a.a_c(), v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_out(), v.v_out_os(), None,
                     None, s.s_out_stride_wo(),
                     s.s_tmp(), v.v_out_flag() if self.tunable.nxe != 0 else v.v_out_flag(), s.s_dim_mr(), v.v_out_inb(), s.s_block_gtc_inb(), v.v_co_sub_m_index(), v.v_tmp()))

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
