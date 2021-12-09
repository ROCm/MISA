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

import math
from .shared_memory import *
from .global_memory import *
from .dotx_mapping import *
from .generic_tensor_transformation import *
import copy
import itertools

MAX_LGKMCNT = 64    # 0...63

class ctrl_coalescing_store_dotx_t(object):
    '''
    like xdlops, we still assume register within single thread first loop over m direction
    '''
    def __init__(self):
        self.cdm = None                     # ctrl_dotx_mapping_t
        self.coalescing_groups = 1
        self.block_size = 256
        self.vector_store_m = 1             # global vector store in m/n
        self.vector_store_n = 1             # ... m, n can't be non-1 at the same time
        self.vector_fold_m = 1              # due to vector store, we might want to fold m/n
        self.vector_fold_n = 1              # ... while calculating m/n global index
        self.precision = 'fp16'             # dotx only support fp16 & int8
        self.gemm_k_global_split = False
        self.feat_vgpr_collapse = True
        self.co_m_update_os_functor = None  # update offset based on current i_m. otherwise use sgpr to update offset

        self.feat_co_m_flag_check = False   # custom flag check, not using internal check
        self.co_m_flag_check_start_functor = None
        self.co_m_flag_check_reset_functor = None

    def get_m_split_lengths(self):
        l_mg = self.get_length_m_groups()
        l_ng = self.get_length_n_groups()
        num_m_groups = math.gcd(self.coalescing_groups, l_mg)
        num_n_groups = math.gcd(self.coalescing_groups//num_m_groups, l_ng)

        assert num_n_groups == 1, "if have multiple groups along n dimension, coalesing is meaningless"

        m_lengths = [self.cdm.lanegroup_repeat_m, self.cdm.lanegroup_m_per_thread()]
        split_lengths = tensor_util_split_lengths(num_m_groups, m_lengths,
                                            tensor_util_arithmetic_sequence_gen(0, len(m_lengths), 1))

        return split_lengths

    def get_lanegroup_granularity_m(self):
        '''
        dotx granularity is unlike xdlops(4), it has 8x register along m direction
        but we need to further sub divide into a number that can use single ds_write_b128
        '''
        l_mr, l_mt = self.get_m_split_lengths()
        return math.gcd(l_mt, 4)

    def get_length_m_groups(self):
        ''' agpr per thread in m dimension '''
        return self.cdm.lanegroup_repeat_m * self.cdm.lanegroup_m_per_thread()

    def get_length_n_groups(self):
        ''' agpr per thread in n dimension '''
        return self.cdm.lanegroup_repeat_n * self.cdm.lanegroup_n_per_thread()

    # def get_length_m_max_groups(self):
    #     '''
    #     max possible groups due to we want to vector ds_write
    #     '''
    #     return self.get_length_m_groups() // (self.get_lanegroup_granularity_m() if self.vector_store_n == 1 else 1)

    def get_subgroups(self):
        # assert self.get_length_m_max_groups() % self.coalescing_groups == 0

        # assert g_mt == 1, 'we do not want to split inside this granularity within a thread'

        split_lengths = self.get_m_split_lengths()
        m_lengths = [self.cdm.lanegroup_repeat_m, self.cdm.lanegroup_m_per_thread()]

        g_mr, g_mt = m_lengths[0] // split_lengths[0], m_lengths[1] // split_lengths[1]

        return g_mr, g_mt

    def get_subgroup_length(self):
        g_mr, g_mt = self.get_subgroups()

        l_mr = self.cdm.lanegroup_repeat_m // g_mr
        l_mt = self.cdm.lanegroup_m_per_thread() // g_mt
        if self.vector_store_m != 1:
            assert l_mt % self.vector_store_m == 0, 'can not write out vector_m in single coalescing group'
        return l_mr, l_mt

    def get_num_dword_per_group(self):
        '''
        devide the total register used with the number of groups
        '''
        assert self.cdm.total_acc_c() % self.coalescing_groups == 0
        return self.cdm.total_acc_c() // self.coalescing_groups

    def get_m_index_per_group(self):
        '''
        return a 3d list to describe the index in m after LDS transpose
            num_groups * cluster_len_m * thread_len_m

        num_groups    : coalescing groups
        cluster_len_m : how many cluster (different threads) along m to collaborately store m
        thread_len_m  : how many pixel within a thread to issue out
                        if have vector_store_m, will not count every index within vector_store_m
                        (because they will be issued by a single global store instruction)
                        also, vector_fold_m will not be considered here

        NOTE:
            although this function may not necessarily require all its element
            it is still a good practice to implement this function, to help understand the whole process
            We'll have some sanity check inside

        '''
        def flatten(x):
            from functools import reduce
            return reduce(lambda a, b: a*b, x, 1)
        num_dword_per_group = self.get_num_dword_per_group()
        g_mr, g_mt = self.get_subgroups()
        l_mr, l_mt = self.get_subgroup_length()

        assert num_dword_per_group == (l_mr * l_mt) * flatten(self.cdm.acc_c_per_thread_n()), \
                f"num_dword_per_group:{num_dword_per_group}, l_mr:{l_mr}, l_mt:{l_mt} x acc_c_per_thread_n:{self.cdm.acc_c_per_thread_n()}"

        # print(f"mr:{g_mr}x{l_mr}x{self.cdm.wave_repeat_m}, ms:{g_ms}x{l_ms}x{self.cdm.wave_step_m}, mw:{g_mw}x{l_mw}x{self.cdm.lanegroup_m_per_wave()}, mb:{g_mb}x{l_mb}x{self.cdm.lanegroup_m_per_block()}, mt:{g_mt}x{l_mt}x{self.cdm.lanegroup_m_per_thread()}")

        # ttm = self.get_transposed_thread_mapping()
        m_index_per_group = list()
        m_index_total = list()

        # NOTE: this for loop is based on index before LDS transpose
        #       this is designed on purpose, hope we can get some 
        for i_g_mr, i_g_mt in itertools.product(range(g_mr), range(g_mt)):
            m_idx_start_per_group = i_g_mr * l_mr * self.cdm.waves_per_m() * self.cdm.lanegroup_m_per_wave() * self.cdm.lanegroup_m_per_thread() * self.cdm.lanegroup_m_per_cluster() + \
                                                i_g_mt * l_mt
            m_index = []
            # iterate between thread
            for t_along_waves_per_m, t_along_lanegroup_m_per_wave, t_along_lanegroup_m_per_cluster in itertools.product(
                                    range(self.cdm.waves_per_m()), range(self.cdm.lanegroup_m_per_wave()), range(self.cdm.lanegroup_m_per_cluster())):
                m_idx_start = m_idx_start_per_group + t_along_waves_per_m * self.cdm.lanegroup_m_per_wave() * self.cdm.lanegroup_m_per_thread() * self.cdm.lanegroup_m_per_cluster() + \
                                t_along_lanegroup_m_per_wave * self.cdm.lanegroup_m_per_thread() * self.cdm.lanegroup_m_per_cluster() +\
                                t_along_lanegroup_m_per_cluster * self.cdm.lanegroup_m_per_thread()
                # iterate within thread
                for i_mr, i_mt in itertools.product(range(l_mr), range(l_mt)):
                    m_idx_current = m_idx_start + i_mr * self.cdm.waves_per_m() * self.cdm.lanegroup_m_per_wave() * self.cdm.lanegroup_m_per_thread() * self.cdm.lanegroup_m_per_cluster() + \
                                                i_mt
                    m_index.append(m_idx_current)
            m_index.sort()
            m_index_total.extend(m_index)
            pixel_m_index = [None] * ttm.c_m0()
            for i, m in enumerate(m_index):
                _icm0 = (i // ttm.t_m0()) % ttm.c_m0()  # NOTE! here is m granularity take place
                if not pixel_m_index[_icm0]: pixel_m_index[_icm0] = list()
                pixel_m_index[_icm0].append(m)
            m_index_per_group.append(pixel_m_index)

        m_index_total.sort()
        # print(m_index_total)
        assert m_index_total == [xx for xx in range(self.cdm.macro_tile_m)], f"len:{len(m_index_total)}, {m_index_total}"

        # do some validation
        for ig in range(len(m_index_per_group)):
            if len(m_index_per_group[ig]) == 1:
                continue
            for ic in range(len(m_index_per_group[ig]) - 1):
                c_curr_list = m_index_per_group[ig][ic]
                c_next_list = m_index_per_group[ig][ic + 1]
                assert len(c_curr_list) == len(c_next_list)
                diff = c_next_list[0] - c_curr_list[0]
                for i in range(1, len(c_curr_list)):
                    diff_i = c_next_list[i] - c_curr_list[i]
                    assert diff == diff_i, "stride between different transpose m0 not the same, should not happen"

        return m_index_per_group

    def get_co_sub_m_index(self):
        pass

    def get_vgpr_usage(self):
        '''
        return the number of vgpr needed for coalescing store process
        '''
        agpr_per_store_group = self.cdm.total_acc_c() // self.coalescing_groups
        if self.feat_vgpr_collapse:
            data_byte = amdgpu_precision_data_byte(self.precision)
            inst_sld_byte = (self.get_lanegroup_granularity_m() if self.vector_store_n == 1 else self.vector_store_n) * data_byte
            issues_per_ssgroup = 4 if inst_sld_byte == 16 or inst_sld_byte == 8 else 8

            num_sld_total_dword = self.get_num_dword_per_group() // (self.get_lanegroup_granularity_m() if self.vector_store_n == 1 else self.vector_store_n)

            total_lgkmcnt = num_sld_total_dword     # TODO: assume sld is single issue

            assert MAX_LGKMCNT % issues_per_ssgroup == 0
            # print(f"issues_per_ssgroup:{issues_per_ssgroup}, total_lgkmcnt:{total_lgkmcnt}, get_num_dword_per_group:{self.get_num_dword_per_group()}, vector_store_n:{self.vector_store_n}")

            # we need further split based on issues_per_ssgroup
            split_sld_groups = (total_lgkmcnt + issues_per_ssgroup - 1) // issues_per_ssgroup

            agpr_per_store_split_sld_group = (agpr_per_store_group + split_sld_groups - 1) // split_sld_groups
            assert agpr_per_store_split_sld_group >= 4

            return agpr_per_store_split_sld_group
        else:
            return agpr_per_store_group

    def can_skip_coalescing(self):
        '''
        currently, this api define CAN skip, but indeed this is MUST skip
        for coalescing write out, we assume thread coalescing along N, and it is easy to divide block size
        (256) along a power-of-2 number, but non-power-of-2 is very hard to do so.
        '''
        if not utility_is_pow2(self.cdm.macro_tile_n):
            return True
        return False

class igemm_coalescing_store_dotx_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_coalescing_store_dotx_t
        self.ctrl = ctrl

    def name(self):
        return ''

    def init_co_lds_offset(self, v_co_sst, v_co_sld, v_gemm_im, v_gemm_in, v_tid, v_tmp4):
        ctrl = self.ctrl
        data_byte = amdgpu_precision_data_byte(ctrl.precision)
        g_mr, g_mt = ctrl.get_subgroups()
        l_mr, l_mt = ctrl.get_subgroup_length()

        with self._deferred_context():
            self._emit(f"; init_co_lds_offset for dotx")
            '''
            gemm_m_shrink is in multi-dimension.
            then, consider that introduced by granularity
            '''
            self._emit(f"v_lshrrev_b32 v[{v_tmp4}], {utility_log2(ctrl.cdm.lanegroup_m_per_thread())}, v[{v_gemm_im}]")
            self._emit(f"v_and_b32 v[{v_tmp4}],  {ctrl.cdm.lanegroup_m_per_cluster() - 1} v[{v_tmp4}]   ; thread id of lanegroup_m_per_cluster")
            self._emit(f"v_lshlrev_b32 v[{v_co_sst}], {utility_log2(ctrl.cdm.lanegroup_m_per_thread())}, v[{v_tmp4}]")

            if ctrl.cdm.lanegroup_m_per_wave() != 1:
                length_above_lanegroup_m_per_wave = ctrl.cdm.lanegroup_m_per_cluster() * ctrl.cdm.lanegroup_m_per_thread()
                self._emit(f"v_lshrrev_b32 v[{v_tmp4}+1], {utility_log2(length_above_lanegroup_m_per_wave)}, v[{v_gemm_im}]")
                self._emit(f"v_and_b32 v[{v_tmp4}+1], {ctrl.cdm.lanegroup_m_per_wave() - 1}  , v[{v_tmp4}+1]   ; thread id of lanegroup_m_per_wave")
                assert length_above_lanegroup_m_per_wave % g_mt == 0, f"length_above_lanegroup_m_per_wave:{length_above_lanegroup_m_per_wave}, g_mt:{g_mt}"
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp4}+1], {utility_log2(length_above_lanegroup_m_per_wave // g_mt)}, v[{v_co_sst}]")

            if ctrl.cdm.waves_per_m() != 1:
                length_above_waves_per_m = ctrl.cdm.lanegroup_m_per_wave() * ctrl.cdm.lanegroup_m_per_cluster() * ctrl.cdm.lanegroup_m_per_thread()
                self._emit(f"v_lshrrev_b32 v[{v_tmp4}+2], {utility_log2(length_above_waves_per_m)}, v[{v_gemm_im}]  ; thread id of waves_per_m")
                assert length_above_waves_per_m % (g_mt) == 0, f"length_above_waves_per_m:{length_above_waves_per_m}, g_mt:{g_mt}"
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp4}+2], {utility_log2(length_above_waves_per_m // (g_mt))}, v[{v_co_sst}]")

            if ctrl.vector_store_n == 1:
                self._emit(f"v_lshrrev_b32 v[{v_tmp4}], {utility_log2(ctrl.get_lanegroup_granularity_m())}, v[{v_co_sst}]")
                self._emit(f"v_lshlrev_b32 v[{v_tmp4}+1], {utility_log2(ctrl.get_lanegroup_granularity_m())}, v[{v_gemm_in}]   ; implicit transpose with m granularity:{ctrl.get_lanegroup_granularity_m()} while store")
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp4}], {utility_log2(ctrl.cdm.macro_tile_n * ctrl.get_lanegroup_granularity_m())}, v[{v_tmp4}+1]")
            else:
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_co_sst}], {utility_log2(ctrl.cdm.macro_tile_n)}, v[{v_gemm_in}]")

            self._emit(f"v_lshlrev_b32 v[{v_co_sst}], {utility_log2(data_byte)}, v[{v_co_sst}]")
            self._emit(f"v_lshlrev_b32 v[{v_co_sld}], {utility_log2(data_byte * (ctrl.get_lanegroup_granularity_m() if ctrl.vector_store_n == 1 else ctrl.vector_store_n))}, v[{v_tid}]")

        return self._get_deferred()

    def init_co_sub_m_index(self, v_co_sub_m_index, v_tid, v_tmp6):
        def flatten(x):
            from functools import reduce
            return reduce(lambda a, b: a*b, x, 1)
        ctrl = self.ctrl
        # need use v_co_sub_m_index to calculate v offset in m direction
        g_mr, g_mt = ctrl.get_subgroups()
        l_mr, l_mt = ctrl.get_subgroup_length()
        n_mc = ctrl.cdm.lanegroup_m_per_cluster()       # this iteration is among different thread
        n_ml = ctrl.cdm.lanegroup_m_per_wave()          # this iteration is among different thread
        n_mv = ctrl.cdm.waves_per_m()                   # this iteration is among different thread

        nd_stride = [l_mt, n_mc, n_ml, n_mv, 1 ]

        with self._deferred_context():
            self._emit(f"; init_co_sub_m_index dotx, block_size:{ctrl.cdm.block_size()}, macro-tile:{ctrl.cdm.macro_tile_m}x{ctrl.cdm.macro_tile_n} sub_m_index:{ctrl.get_co_sub_m_index()}")
            self._emit(f"; g_mr:{g_mr}, g_mt:{g_mt} | l_mr:{l_mr}, l_mt:{l_mt} | n_mc:{n_mc}, n_ml:{n_ml}, n_mv:{n_mv}")
            self._emit(f"; nd_stride:{nd_stride}")
            c_m0 = ctrl.block_size // (ctrl.cdm.macro_tile_n // ctrl.vector_store_n)
            if c_m0 == 1:
                # give a chance to early exit, only let the co_sub_m to be zero
                self._emit(f"v_mov_b32 v[{v_co_sub_m_index}], 0")
            else:
                if ctrl.vector_store_n == 1:
                    self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {utility_log2(ctrl.cdm.macro_tile_n)}, v[{v_tid}]   ; get tid along m")
                else:
                    self._emit(f"v_lshlrev_b32 v[{v_tmp6}], {utility_log2(ctrl.vector_store_n)}, v[{v_tid}]")
                    self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {utility_log2(ctrl.cdm.macro_tile_n)}, v[{v_tmp6}]  ; get tid along m")

                v_idx = 0
                # iterate all dimensions
                if ctrl.vector_store_n != 1:
                    # if have vector store, we no longer have granularity, hence every dimension participate in m divide
                    if l_mt != 1 and c_m0 not in (0, 1):
                        self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {l_mt - 1}, v[{v_co_sub_m_index}]                   ; => x_mt")
                        v_idx = v_idx + 1
                        c_m0  = c_m0 // l_mt
                        if c_m0 not in (0, 1):
                            self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {utility_log2(l_mt)}  ,v[{v_co_sub_m_index}]")
                if n_mc != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {n_mc - 1}, v[{v_co_sub_m_index}]                   ; => x_mc")
                    v_idx = v_idx + 1
                    c_m0  = c_m0 // n_mc
                    if c_m0 not in (0, 1):
                        self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {utility_log2(n_mc)}  ,v[{v_co_sub_m_index}]")
                if n_ml != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {n_ml - 1}, v[{v_co_sub_m_index}]                   ; => x_ml")
                    v_idx = v_idx + 1
                    c_m0  = c_m0 // n_ml
                    if c_m0 not in (0, 1):
                        self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {utility_log2(n_ml)}  ,v[{v_co_sub_m_index}]")
                if n_mv != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {n_mv - 1}, v[{v_co_sub_m_index}]                   ; => x_mv")
                    v_idx = v_idx + 1
                    c_m0  = c_m0 // n_mv
                    if c_m0 not in (0, 1):
                        self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {utility_log2(n_mv)}  ,v[{v_co_sub_m_index}]")
                if l_mr != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {l_mr - 1}, v[{v_co_sub_m_index}]                   ; => x_mr")

                # indeed accoroding to current implementation, at most 2 tmp vgpr is used.
                # we keep 6 temp here in case in the future there might be different mapping
                assert v_idx <= 5, "since current we only assign 6 vgpr to do this nd split-merge, so larger than 6 vgpr currently not supported"

                class pretty_accumulate_co_sub_m_t(object):
                    def __init__(self):
                        self.first = 1
                    def __call__(self, v_co_sub_m_index, v_idx, k_multiplier):
                        '''
                        return a string to calculate v_co_sub_m_index = v_idx * k_multiplier + v_co_sub_m_index
                        '''
                        assert utility_is_pow2(k_multiplier)
                        if self.first == 1:
                            self.first = 0
                            if k_multiplier == 1:
                                return f"v_mov_b32 v[{v_co_sub_m_index}], v[{v_idx}]"
                            else:
                                return f"v_lshlrev_b32 v[{v_co_sub_m_index}], {utility_log2(k_multiplier)}, v[{v_idx}]"
                        else:
                            if k_multiplier == 1:
                                return  f"v_add_u32 v[{v_co_sub_m_index}], v[{v_idx}], v[{v_co_sub_m_index}]"
                            else:
                                return f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_idx}], {utility_log2(k_multiplier)}, v[{v_co_sub_m_index}]"

                c_m0 = ctrl.block_size // (ctrl.cdm.macro_tile_n // ctrl.vector_store_n)
                v_idx_r = 0
                accumulate_co_sub_m = pretty_accumulate_co_sub_m_t()
                if ctrl.vector_store_n != 1:
                    # if have vector store, we no longer have granularity, hence every dimension participate in m divide
                    if l_mt != 1 and c_m0 not in (0, 1):
                        self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", 1) + "      ; => accumulate x_mt")
                        v_idx_r = v_idx_r + 1
                        c_m0    = c_m0 // l_mt
                if n_mc != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:1])) + "      ; => accumulate x_mc")
                    v_idx_r = v_idx_r + 1
                    c_m0    = c_m0 // n_mc
                if n_ml != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:2])) + "      ; => accumulate x_ml")
                    v_idx_r = v_idx_r + 1
                    c_m0    = c_m0 // n_ml

                if n_mv != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:3])) + "      ; => accumulate x_mv")
                    v_idx_r = v_idx_r + 1
                    c_m0    = c_m0 // n_mv
                if l_mr != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:4])) + "      ; => accumulate x_mr")

                assert v_idx == v_idx_r, "please check!"

        return self._get_deferred()

    def init_co_sub_n_index(self, v_co_sub_n_index, v_tid, v_tmp2):
        '''
        in n dimension, always have one thread per column
        '''
        ctrl = self.ctrl
        # need use v_co_sub_n_index to calculate v offset in n direction

        with self._deferred_context():
            self._emit(f"; init_co_sub_n_index dotx")
            if ctrl.vector_store_n == 1:
                self._emit(f"v_and_b32 v[{v_co_sub_n_index}], {ctrl.cdm.macro_tile_n - 1}, v[{v_tid}]")
            else:
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {utility_log2(ctrl.vector_store_n)}, v[{v_tid}]")
                self._emit(f"v_and_b32 v[{v_co_sub_n_index}], {ctrl.cdm.macro_tile_n - 1}, v[{v_tmp2}]")
        return self._get_deferred()

    '''
    def get_vgpr_usage(self):
        ctrl = self.ctrl
        agpr_per_store_group = ctrl.cdm.total_acc_c() // ctrl.coalescing_groups
        if ctrl.feat_vgpr_collapse:
            data_byte = amdgpu_precision_data_byte(ctrl.precision)
            inst_sld_byte = (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_store_n == 1 else ctrl.vector_store_n) * data_byte
            issues_per_ssgroup = 4 if inst_sld_byte == 16 or inst_sld_byte == 8 else 8

            num_sld_total_dword = ctrl.get_num_dword_per_group() // (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_store_n == 1 else ctrl.vector_store_n)

            total_lgkmcnt = num_sld_total_dword     # TODO: assume sld is single issue

            assert MAX_LGKMCNT % issues_per_ssgroup == 0

            # we need further split based on issues_per_ssgroup
            split_sld_groups = (total_lgkmcnt + issues_per_ssgroup - 1) // issues_per_ssgroup

            agpr_per_store_split_sld_group = (agpr_per_store_group + split_sld_groups - 1) // split_sld_groups
            assert agpr_per_store_split_sld_group >= 4

            return agpr_per_store_split_sld_group
        else:
            return agpr_per_store_group
    '''

    def __call__(self, v_c_tmp, v_c, v_co_sst, v_co_sld, s_p_out, v_out_offset, s_out_offset, s_gemm_m0_stride, s_gemm_m1_stride, s_tmp6, v_store_flag = None, s_k = None, v_cur_k = None, s_block_gtc_ik = None, v_co_sub_m_index = None, v_tmp0 = None):

        # if no need s_out_offset, set to integer 0
        # if no need flag to dicide store, set v_store_flag to 0
        def flatten(x):
            from functools import reduce
            return reduce(lambda a, b: a*b, x, 1)
        ctrl = self.ctrl
        assert not (ctrl.vector_store_m != 1 and ctrl.vector_store_n != 1), "currently not support vector store both in m and n direction"
        assert not (ctrl.vector_fold_m != 1 and ctrl.vector_fold_n != 1), "... fold in both m and n direction have no meaning"

        data_byte = amdgpu_precision_data_byte(ctrl.precision)
        v_c = sym_t(v_c)
        v_c_tmp = sym_t(v_c_tmp)
        v_co_sst = sym_t(v_co_sst)
        v_co_sld = sym_t(v_co_sld)
        s_tmp6 = sym_t(s_tmp6)

        if s_k is not None:
            s_k = sym_t(s_k)
            v_cur_k = sym_t(v_cur_k)
            s_block_gtc_ik = sym_t(s_block_gtc_ik)
            v_co_sub_m_index = sym_t(v_co_sub_m_index)
            v_tmp0 = sym_t(v_tmp0)

        g_mr, g_mt = ctrl.get_subgroups()
        l_mr, l_mt = ctrl.get_subgroup_length()
        n_mc = ctrl.cdm.lanegroup_m_per_cluster()       # this is among different thread
        n_ml = ctrl.cdm.lanegroup_m_per_wave()          # this is among different thread
        n_mv = ctrl.cdm.waves_per_m()                   # this is among different thread

        n_nc = ctrl.cdm.lanegroup_n_per_cluster()       # this is among different thread
        n_nl = ctrl.cdm.lanegroup_n_per_wave()          # this is among different thread
        n_nv = ctrl.cdm.waves_per_n()                   # this is among different thread

        no_s_out_offset = s_out_offset is None
        # inst_sst_byte = ctrl.vector_store_n * data_byte
        # inst_sld_byte = (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_store_n == 1 else ctrl.vector_store_n) * data_byte

        '''
            1. if vector_store_m == 1 and vector_store_n == 1
                LDS store & read both using lanegroup_granularity_m as vector size, and global write out as single pixel
                in this case, vector_fold_m, vector_fold_n must be 1

            2. if vector_store_m != 1 and vector_store_n == 1
                only valid when vector_fold_m != 1
                vector_store_m != 1 means we want to do vector store along m direction.
                If m is fast changing dimension, actually we dont want this be happen for now. this coalescing store class by default
                assume thread is conitnuout along n direction, so if m is fast changing, it is better thread is continuous along m,
                which may violate the assumption. TODO: implement thread continous along m in the future.

                we allow in case that vector_fold_m != 1, which is useful in case like NCHWvect_c, where n is still fast changing dim, 
                only with vect_c=vector_fold_m in m dim.

                LDS read/write in vector_store_m, and vector_store_m <= lanegroup_granularity_m

            3. if vector_store_m == 1 and vector_store_n != 1
                LDS write in 1 pixel unit, LDS read in vector_store_n
                vector_fold_n seems not needed?
        '''
        def get_smem_co_vector_size():
            sst_vec, sld_vec, smem_trans = 1, 1, False
            if ctrl.vector_store_m == 1 and ctrl.vector_store_n == 1:
                assert ctrl.vector_fold_m == 1 and ctrl.vector_fold_n == 1
                sst_vec, sld_vec, smem_trans = ctrl.get_lanegroup_granularity_m(), ctrl.get_lanegroup_granularity_m(), False
            elif ctrl.vector_store_m != 1 and ctrl.vector_store_n == 1:
                assert ctrl.vector_fold_m != 1 and ctrl.vector_fold_n == 1
                assert ctrl.get_lanegroup_granularity_m() % ctrl.vector_fold_m == 0
                sst_vec, sld_vec, smem_trans = ctrl.vector_fold_m, ctrl.vector_fold_m, False
            elif ctrl.vector_store_m == 1 and ctrl.vector_store_n != 1:
                assert ctrl.vector_fold_m == 1
                sst_vec, sld_vec, smem_trans = 1, ctrl.vector_store_n, True
            else:
                assert False, f'not supported vector_store_m:{ctrl.vector_store_m}, vector_store_n:{ctrl.vector_store_n}'
            return sst_vec, sld_vec, smem_trans
        sst_vec, sld_vec, smem_trans = get_smem_co_vector_size()
        assert l_mt % sst_vec == 0

        inst_sst = inst_ds_write_t(sst_vec * data_byte)
        inst_sld = inst_ds_read_t(sld_vec * data_byte)
        if ctrl.gemm_k_global_split:
            v_pack = 2 if ctrl.vector_store_n == 2 and data_byte == 2 else 1
            inst_gst = inst_buffer_atomic_add_dword_t(ctrl.vector_store_n * data_byte, v_pack) 
        else:
            inst_gst = inst_buffer_store_t(ctrl.vector_store_n * data_byte)

        s_out_offset_itr = sym_t(s_tmp6(0))
        # s_thread_m_stride = sym_t(s_tmp4(1))

        #m_index_per_group = ctrl.get_m_index_per_group()
        # thread_m_stride = ctrl.get_thread_m_stride()

        #assert len(m_index_per_group) == ctrl.coalescing_groups

        t_mr, t_nr, t_nt, t_mt = ctrl.cdm.acc_c_lengths()
        # 1. vgpr desc
        vgpr_lengths = list(ctrl.cdm.acc_c_lengths())
        vgpr_desc = make_naive_tensor_descriptor_packed(vgpr_lengths)

        # only split along gemm m
        vgpr_split_lengths = tensor_util_split_lengths(ctrl.coalescing_groups, vgpr_lengths, [0, 1, 2, 3], [1, 0, 0, 1])
        vgpr_split_desc = make_transform_tensor_descriptor(vgpr_desc, 
                                    make_tuple(trans_grouped_slice(vgpr_desc.get_lengths(),
                                                                [0, 0, 0, 0],
                                                                vgpr_split_lengths)),
                                    make_tuple([0, 1, 2, 3]),
                                    make_tuple([0, 1, 2, 3]))

        assert vgpr_split_desc.get_lengths() == [l_mr, t_nr, t_nt, l_mt]

        vgpr_last_dim_issues = l_mt // sst_vec

        vgpr_co_desc = make_transform_tensor_descriptor(vgpr_split_desc, 
                                    make_tuple(
                                            trans_passthrough(vgpr_split_desc.get_length(0)),
                                            trans_passthrough(vgpr_split_desc.get_length(1)),
                                            trans_passthrough(vgpr_split_desc.get_length(2)),
                                            trans_vectorize(vgpr_split_desc.get_length(3), vgpr_last_dim_issues),
                                        ),
                                    make_tuple(0, 1, 2, 3),
                                    make_tuple(0, 1, 2, 3))

        # 2. gemm_m desc
        gemm_m_lengths = [ctrl.cdm.lanegroup_repeat_m,          # thread lengths
                        ctrl.cdm.waves_per_m(),
                        ctrl.cdm.lanegroup_m_per_wave(),
                        ctrl.cdm.lanegroup_m_per_cluster(),
                        ctrl.cdm.lanegroup_m_per_thread()]      # thread lengths

        gemm_m_desc = make_naive_tensor_descriptor_packed(gemm_m_lengths)
        gemm_m_split_lengths = tensor_util_split_lengths(ctrl.coalescing_groups, gemm_m_lengths, [0, 1, 2, 3, 4], [1, 0, 0, 0, 1])

        gemm_m_split_desc = make_transform_tensor_descriptor(gemm_m_desc, 
                                    make_tuple(trans_grouped_slice(gemm_m_desc.get_lengths(),
                                                                [0, 0, 0, 0, 0],
                                                                gemm_m_split_lengths)),
                                    make_tuple([0, 1, 2, 3, 4]),
                                    make_tuple([0, 1, 2, 3, 4]))

        gemm_m_co_desc = gemm_m_split_desc

        # 3. lds store desc
        sst_co_lengths = [  l_mr,                   # m, within thread
                            n_mv * n_ml * n_mc,     # m, among different thread
                            l_mt // sst_vec,        # m, within thread, consider vector fold
                            t_nr,                   # n, within thread
                            n_nv * n_nl * n_nc,     # n, among different thread
                            t_nt,                   # n, within thread
                            sst_vec * data_byte]    #    store vector  size

        sst_co_desc = make_naive_tensor_descriptor_packed(sst_co_lengths)

        def vgpr_coord_2_sst_coord(v_coord):
            # TODO: coordinate remapping
            assert len(v_coord) == 4
            s_coord = [0] * 7
            s_coord[0], s_coord[2], s_coord[3], s_coord[5] = v_coord[0], v_coord[3], v_coord[1], v_coord[2]
            return s_coord

        # 4. gemm desc before/after transpose
        gemm_m_lengths = [ctrl.cdm.lanegroup_repeat_m,              # thread lengths
                            ctrl.cdm.waves_per_m(),
                            ctrl.cdm.lanegroup_m_per_wave(),
                            ctrl.cdm.lanegroup_m_per_cluster(),
                            ctrl.cdm.lanegroup_m_per_thread()]      # thread lengths
        gemm_n_lengths = [ctrl.cdm.lanegroup_repeat_n,              # thread lengths
                            ctrl.cdm.waves_per_n(),
                            ctrl.cdm.lanegroup_n_per_wave(),
                            ctrl.cdm.lanegroup_n_per_cluster(),
                            ctrl.cdm.lanegroup_n_per_thread()]      # thread lengths

        gemm_m_size = tensor_util_reduce(gemm_m_lengths, lambda a, b: a*b, 1)
        gemm_n_size = tensor_util_reduce(gemm_n_lengths, lambda a, b: a*b, 1)

        gemm_m_split_lengths = tensor_util_split_lengths(
                                            ctrl.coalescing_groups, gemm_m_lengths,
                                            [0, 1, 2, 3, 4], [1, 0, 0, 0, 1])

        gemm_lengths = [*gemm_m_lengths, *gemm_n_lengths]           # m*n

        gemm_desc = make_naive_tensor_descriptor_packed(gemm_lengths)
        gemm_co_split_lengths = tensor_util_split_lengths(
                                    ctrl.coalescing_groups, gemm_lengths,
                                    [0, 1, 2, 3, 4,   5, 6, 7, 8, 9],
                                    [1, 0, 0, 0, 1,   0, 0, 0, 0, 0])

        assert gemm_m_split_lengths == gemm_co_split_lengths[:5]

        gemm_m_slice_size = tensor_util_reduce(gemm_m_split_lengths, lambda a, b: a*b, 1)

        gemm_split_desc = make_transform_tensor_descriptor(gemm_desc, 
                                    make_tuple(trans_grouped_slice(gemm_desc.get_lengths(),
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                gemm_co_split_lengths)),
                                    make_tuple([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                    make_tuple([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

        # desc before transpose
        gemm_co_prev_desc = make_transform_tensor_descriptor(gemm_split_desc,
                                    make_tuple(
                                        trans_passthrough(gemm_split_desc.get_length(0)),
                                        trans_passthrough(gemm_split_desc.get_length(1)),
                                        trans_passthrough(gemm_split_desc.get_length(2)),
                                        trans_passthrough(gemm_split_desc.get_length(3)),
                                        trans_passthrough(gemm_split_desc.get_length(4)),
                                        trans_merge(gemm_n_lengths)),
                                    make_tuple(0, 1, 2, 3, 4, [5, 6, 7, 8, 9]),
                                    make_tuple(0, 1, 2, 3, 4, 5))

        gemm_co_2d_desc = make_transform_tensor_descriptor(gemm_co_prev_desc,
                                    make_tuple( trans_merge(gemm_m_split_lengths),
                                                trans_passthrough(gemm_n_size)),
                                    make_tuple([0, 1, 2, 3, 4], 5),
                                    make_tuple(0, 1))
        # print(f'gemm_split_desc:{gemm_split_desc.get_lengths()}, xxxx {gemm_co_2d_desc.get_lengths()}')

        gemm_n_post_thread_length = sld_vec
        gemm_n_post_cluster_length = gemm_n_size // gemm_n_post_thread_length
        gemm_m_post_cluster_length = ctrl.cdm.block_size() // gemm_n_post_cluster_length
        gemm_m_post_thread_length = gemm_m_slice_size // gemm_m_post_cluster_length

        # print(f'm:{gemm_m_post_thread_length}x{gemm_m_post_cluster_length}, n:{gemm_n_post_cluster_length}x{gemm_n_post_thread_length}')

        # further more, we need sub-divide m_post_thread_length, because ds_read can not be larger than lgkmcnt max(15)
        # actually, we want to sub divide ds_read by its vector size.
        max_issues_per_ssgroup = 4 if sld_vec in (2, 4) else 8
        num_issues_per_ssgroup = utility_gcd(gemm_m_post_thread_length, max_issues_per_ssgroup)
        split_sld_groups = gemm_m_post_thread_length // num_issues_per_ssgroup


        assert num_issues_per_ssgroup * split_sld_groups * gemm_m_post_cluster_length == gemm_co_2d_desc.get_length(0), f'{num_issues_per_ssgroup} * {split_sld_groups} * {gemm_m_post_cluster_length} == {gemm_co_2d_desc.get_length(0)}'
        assert gemm_n_post_thread_length * gemm_n_post_cluster_length == gemm_co_2d_desc.get_length(1)

        # desc after transpose
        gemm_co_post_desc = make_transform_tensor_descriptor(gemm_co_2d_desc,
                                    make_tuple(trans_unmerge([split_sld_groups, num_issues_per_ssgroup, gemm_m_post_cluster_length]),
                                                trans_unmerge([gemm_n_post_cluster_length, gemm_n_post_thread_length])),
                                    make_tuple(0, 1),
                                    make_tuple([0, 1, 2], [3, 4]))

        with self._deferred_context():
            self._emit(f"; coalescing store, mapping:{ctrl.cdm.serialize()}")
            self._emit(f"; coalescing_groups:{ctrl.coalescing_groups}, num_dword_per_group:{ctrl.get_num_dword_per_group()}, block_size:{ctrl.cdm.block_size()}")
            self._emit(f'; gemm_desc:{gemm_desc.get_lengths()}, gemm_co_prev_desc:{gemm_co_prev_desc.get_lengths()}, gemm_co_split_lengths:{gemm_co_split_lengths}, gemm_co_post_desc:{gemm_co_post_desc.get_lengths()}')
            self._emit(f"s_mul_i32 s[{s_gemm_m1_stride}], {(ctrl.cdm.block_size() // ctrl.cdm.macro_tile_n) * data_byte}, s[{s_gemm_m1_stride}]")
            self._emit(f"s_barrier")

            gemm_m_co_start_coord = [0, 0, 0, 0, 0]
            vgpr_co_start_coord = [0, 0, 0, 0]

            accvgpr_consume_list = list()   # record the list for vgpr used to store C matrix, for debug

            for i_group in range(ctrl.coalescing_groups):
                m_index_start_per_group = gemm_co_prev_desc.calculate_offset([0, 0, 0, 0, 0, 0]) // gemm_n_size

                for vgpr_coord in itertools.product(*[range(d) for d in vgpr_co_desc.get_lengths()]):
                    # vgpr_coord should be a list with length 4
                    vgpr_index = vgpr_co_desc.calculate_offset(list(vgpr_coord))
                    sst_coord = vgpr_coord_2_sst_coord(list(vgpr_coord))
                    sst_offset = sst_co_desc.calculate_offset(sst_coord)
                    self._emit(f"; vgpr_coord:{vgpr_coord}, lengths:{vgpr_co_desc.get_lengths()}, sst_coord:{sst_coord}, lengths:{sst_co_desc.get_lengths()}")

                    if ctrl.precision == 'fp16':
                        for i in range(vgpr_last_dim_issues):
                            self._emit(f"v_cvt_f16_f32_e32 v[{v_c(vgpr_index + i)}], v[{v_c(vgpr_index + i)}]")
                            accvgpr_consume_list.append(vgpr_index + i)

                        if not smem_trans:
                            for i in range(sst_vec // 2):
                                self._emit(f"v_pack_b32_f16 v[{v_c(vgpr_index + i)}], v[{v_c(vgpr_index + 2 * i)}], v[{v_c(vgpr_index + 2 * i + 1)}]")
                            #self._emit(f"v_pack_b32_f16 v[{v_c(vgpr_index + 1)}], v[{v_c(vgpr_index + 2)}], v[{v_c(vgpr_index + 3)}]")

                    elif ctrl.precision == 'int8':
                        # CAUSION: must have a symbol s_0xff and pre inited with 0xff
                        if not smem_trans:
                            for i in range(sst // 4):
                                vi = vgpr_index + 4 * i
                                self._emit(f"v_and_b32 v[{v_c(vi + 0)}], s[s_0xff], v[{v_c(vi + 0)}]")
                                self._emit(f"v_and_b32 v[{v_c(vi + 1)}], s[s_0xff], v[{v_c(vi + 1)}]")
                                self._emit(f"v_and_b32 v[{v_c(vi + 2)}], s[s_0xff], v[{v_c(vi + 2)}]")
                                self._emit(f"v_lshlrev_b32 v[{v_c(vi + 3)}], 24, v[{v_c(vi + 3)}]")

                                self._emit(f"v_lshlrev_b32 v[{v_c(vi + 1)}],  8, v[{v_c(vi + 1)}]")
                                self._emit(f"v_lshlrev_b32 v[{v_c(vi + 2)}], 16, v[{v_c(vi + 2)}]")
                                self._emit(f"v_or_b32 v[{v_c(vi + 0)}], v[{v_c(vi + 0)}], v[{v_c(vi + 3)}]")
                                self._emit(f"v_or3_b32 v[{v_c(vi + 0)}], v[{v_c(vi + 0)}], v[{v_c(vi + 1)}], v[{v_c(vi + 2)}]")
                                for j in range(4):
                                    accvgpr_consume_list.append(vi + j)
                        else:
                            # CAUSION: ds_write_b8 already clamp the value for us. if need other clamp methor, need further consideration
                            pass

                    if not smem_trans:
                        self._emit(inst_sst(v_co_sst(), v_c(vgpr_index), sst_offset))
                    else:
                        for i in range(vgpr_last_dim_issues):
                            self._emit(inst_sst(v_co_sst(), v_c(vgpr_index + i), sst_offset + i * ctrl.cdm.macro_tile_n * data_byte))

                def emit_calculate_out_offset_itr_m(i_m, i_m0, i_m1):
                    comments = f"   ; i_m:{i_m}(i_m0:{i_m0},i_m1:{i_m1})"
                    if ctrl.co_m_update_os_functor:
                        self._emit(ctrl.co_m_update_os_functor(i_m, i_m0, i_m1))        # TODO: better sigture
                    else:
                        if s_gemm_m0_stride is not None:
                            assert False, "not supported"
                        else:
                            '''
                            no m0_stride, which indicate m0, m1 is continuous, no need to deal with m0, m1 seperately
                            '''
                            if i_m == 0:
                                if no_s_out_offset:
                                    self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], 0" + comments)
                                    if s_k is not None:
                                        self._emit(f"v_add_u32 v[{v_cur_k()}], s[{s_block_gtc_ik()}], v[{v_co_sub_m_index()}]")
                                        self._emit(f"v_mov_b32 v[{v_tmp0()}], v[{v_cur_k()}]")
                                else:
                                    self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], s[{s_out_offset}]" + comments)
                            elif i_m == 1:
                                if no_s_out_offset:
                                    self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], s[{s_gemm_m1_stride}]" + comments)
                                    if s_k is not None:
                                        self._emit(f"v_add_u32 v[{v_tmp0()}], 1, v[{v_cur_k()}]")
                                else:
                                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_gemm_m1_stride}], s[{s_out_offset}]" + comments)
                            else:
                                if no_s_out_offset:
                                    self._emit(f"s_mul_i32 s[{s_out_offset_itr()}], {i_m}, s[{s_gemm_m1_stride}]" + comments)
                                    if s_k is not None:
                                        self._emit(f"v_add_u32 v[{v_tmp0()}], {i_m}, v[{v_cur_k()}]")
                                else:
                                    self._emit(f"s_mul_i32 s[{s_tmp6(3)}], {i_m}, s[{s_gemm_m1_stride}]")
                                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_tmp6(3)}], s[{s_out_offset}]" + comments)

                # emit first calculation before wait for store
                emit_calculate_out_offset_itr_m(m_index_start_per_group, 0, 0)

                issue_list = []
                num_sld_total_dword = ctrl.get_num_dword_per_group() // sld_vec
                
                self._emit(f"s_waitcnt lgkmcnt(0)")
                self._emit(f"s_barrier")
                for i_d in range(num_sld_total_dword):
                    vgpr_index = i_d * sld_vec * data_byte // 4 # when data byte is 2, only cost 2 vgpr per time
                    sld_offset = i_d * sld_vec * ctrl.block_size  * data_byte
                    # self._emit(inst_sld(v_c(vgpr_index), v_co_sld(), sld_offset))
                    issue_list.append(inst_sld.get_issues(sld_offset))

                total_lgkmcnt = utility_flatten_list_accumulate(issue_list)
                # issues_per_ssgroup = 8 if inst_sld_byte == 16 or inst_sld_byte == 8 else 8
                #issues_per_ssgroup = 8

                #assert MAX_LGKMCNT % issues_per_ssgroup == 0

                # we need further split based on issues_per_ssgroup
                #split_sld_groups = (total_lgkmcnt + issues_per_ssgroup - 1) // issues_per_ssgroup
                #num_issues_per_ssgroup = len(issue_list) // split_sld_groups
                assert (ctrl.get_num_dword_per_group() // sld_vec) % split_sld_groups == 0, "TODO: need adjust ssgroup value based on dword per group"
                num_gst_per_ssgroup = ctrl.get_num_dword_per_group() // sld_vec // split_sld_groups

                assert num_sld_total_dword % split_sld_groups == 0, "TODO: need adjust"
                num_sld_per_ssgroup = num_sld_total_dword // split_sld_groups

                assert num_issues_per_ssgroup == num_gst_per_ssgroup and num_gst_per_ssgroup == num_sld_per_ssgroup

                for i_ssgroup in range(split_sld_groups):
                    self._emit(f";   load from lds, i_ssgroup:{i_ssgroup}, num_sld_per_ssgroup:{num_sld_per_ssgroup}")
                    for i_d in range(num_sld_per_ssgroup):
                        vgpr_index = (i_d + (i_ssgroup if not ctrl.feat_vgpr_collapse else 0) * num_sld_per_ssgroup) * sld_vec * data_byte // 4 # when data byte is 2, only cost 2 vgpr per time
                        sld_offset = (i_d + i_ssgroup * num_sld_per_ssgroup) * sld_vec * ctrl.block_size  * data_byte
                        self._emit(inst_sld(v_c(vgpr_index), v_co_sld(), sld_offset))
                    current_issue_list = issue_list[i_ssgroup * num_issues_per_ssgroup : (i_ssgroup+1) * num_issues_per_ssgroup]
                    if not ctrl.feat_co_m_flag_check and (v_store_flag is not None and type(v_store_flag) is str):
                        self._emit(v_cmpx_eq_u32("vcc", 1, v_store_flag))
                    self._emit(f";   store to global, m index start:{m_index_start_per_group}")

                    for i_gst in range(num_gst_per_ssgroup):
                        i_gst_flat = i_gst + i_ssgroup * num_gst_per_ssgroup
                        if len(current_issue_list) != 0:
                            if i_gst % (ctrl.get_lanegroup_granularity_m() if sld_vec == 1 else 1) == 0:
                                i_issues =  (i_gst // (ctrl.get_lanegroup_granularity_m() if sld_vec == 1 else 1)) + 1
                                i_issue_list = current_issue_list[i_issues:]
                                i_issue_cnt = utility_flatten_list_accumulate(i_issue_list) if len(i_issue_list) != 0 else 0
                                self._emit(f"s_waitcnt lgkmcnt({i_issue_cnt})")
                        # vdata, vaddr, srsrc, soffset, offset
                        if not ctrl.feat_co_m_flag_check and (s_k is not None):
                            self._emit(f"v_cmp_gt_u32 vcc, s[{s_k()}], v[{v_tmp0()}]")
                            self._emit(f"s_and_saveexec_b64 s[{s_tmp6(4)}:{s_tmp6(5)}], vcc")
                        elif ctrl.feat_co_m_flag_check:
                            self._emit(ctrl.co_m_flag_check_start_functor())
                        cur_vgpr_gst = (i_gst_flat if not ctrl.feat_vgpr_collapse else i_gst) * sld_vec//(4 // data_byte)
                        lo_hi = i_gst_flat % 2 if ctrl.precision == 'fp16' and sld_vec == 1 else 0
                        self._emit(inst_gst(v_c(cur_vgpr_gst), v_out_offset, s_p_out, s_out_offset_itr(), 0, lo_hi))
                        if not ctrl.feat_co_m_flag_check and (s_k is not None):
                            self._emit(f"s_or_b64 exec, exec, s[{s_tmp6(4)}:{s_tmp6(5)}]")
                        elif ctrl.feat_co_m_flag_check:
                            self._emit(ctrl.co_m_flag_check_reset_functor())
                        if ctrl.precision == 'int8' and sld_vec == 1:
                            if i_gst_flat % 4 != 3:
                                self._emit(f"v_lshrrev_b32 v[{v_c(cur_vgpr_gst)}], 8, v[{v_c(cur_vgpr_gst)}]")

                        if i_gst_flat != (ctrl.get_num_dword_per_group() // sld_vec) - 1:
                            i_m = i_gst + 1
                            # self._emit(f"; >>>>>> i_m :{i_m}, i_gst:{i_gst}, m_index_per_group[i_group][0]:{m_index_per_group[i_group][0]}")

                            emit_calculate_out_offset_itr_m(i_m, 0, 0)
                    if not ctrl.feat_co_m_flag_check and (v_store_flag is not None and type(v_store_flag) is str):
                        self._emit(f"s_mov_b64 exec, -1")

                if ctrl.feat_vgpr_collapse:
                    agpr_per_store_group = ctrl.cdm.total_acc_c() // ctrl.coalescing_groups
                    assert ctrl.get_vgpr_usage() == ((agpr_per_store_group + split_sld_groups - 1) // split_sld_groups), f"vgpr_usage:{ctrl.get_vgpr_usage()}, agpr_per_store_group:{agpr_per_store_group}, split_sld_groups:{split_sld_groups}"
            
            
                # move slice window for next loop
                move_grouped_slice_start_coord(gemm_co_prev_desc, gemm_co_split_lengths)
                move_grouped_slice_start_coord(gemm_co_post_desc, gemm_co_split_lengths)
                move_grouped_slice_start_coord(vgpr_co_desc, vgpr_split_lengths)
            
            # do some assert
            accvgpr_consume_list.sort()
            assert accvgpr_consume_list == [x for x in range(ctrl.cdm.total_acc_c())], f"accvgpr_consume_list:{accvgpr_consume_list}"

        return self._get_deferred()
