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
import copy
import itertools

MAX_LGKMCNT = 64    # 0...63

class ctrl_coalescing_store_dotx_t(object):
    def __init__(self):
        self.dotx_m = None # ctrl_dotx_mapping_t
        self.coalescing_groups = 1
        self.block_size = 256
        self.vector_write_out = 1
        self.precision = 'fp32'
        self.arch_name = AMDGPU_ARCH_GFX1030
        self.gemm_k_global_split = False
        self.feat_vgpr_collapse = True
        self.co_m_update_os_functor = None  # update offset based on current i_m. otherwise use sgpr to update offset

        self.feat_co_m_flag_check = False   # custom flag check, not using internal check
        self.co_m_flag_check_start_functor = None
        self.co_m_flag_check_reset_functor = None

    def get_length_m_groups(self):
        ''' agpr per thread in m dimension '''
        return self.dotx_m.lanegroup_repeat_m * self.dotx_m.lanegroup_tile_m

    def get_length_n_groups(self):
        ''' agpr per thread in n dimension '''
        return self.dotx_m.lanegroup_repeat_n

    def get_length_m_max_groups(self):
        ''' maximum number of agpr along m dimension per thread can be divided into groups.
            but consider LDS load/store, we want to utilize the fact that every lanegroup is a 4x64 matrix,
            which might be distributed along multiple blocks (like 4x4x1), or every block contains multiple lanegroups(like 16x16x1, 32x32x1)
            henve in m dimension there always have a granularity of 4 per thread, that is continuous along m dimension.
            we want to use a single ds_write_b128/ds_read_b128 to deal with this granularity (which implies a transpose)
            hence we do not want split inside the "4" granularity within a thread
        '''
        return self.get_length_m_groups() // (LANEGROUP_SIZE if self.vector_write_out == 1 else 1)

    def get_subgroups(self):
        def split_ndim_length(total_length, dim_lengths):
            assert type(dim_lengths) in (list, tuple)
            length = total_length
            split_length = list()
            for d in dim_lengths:
                s = math.gcd(d, length)
                length = length // s
                split_length.append(s)
            return tuple(split_length)

        # assert self.get_length_m_groups() % self.coalescing_groups == 0, \
        #     f"if coalescing groups:{self.coalescing_groups} larger than single xdlops agpr number along m:{self.get_length_m_groups()}, can not do this split"
        assert self.get_length_m_max_groups() % self.coalescing_groups == 0, \
            f"if coalescing groups:{self.coalescing_groups} larger than maximum single xdlops agpr(divided by 4) number along m:{self.get_length_m_max_groups()}, can not do this split"
        l_mg = self.get_length_m_groups()
        l_ng = self.get_length_n_groups()
        num_m_groups = math.gcd(self.coalescing_groups, l_mg)
        num_n_groups = math.gcd(self.coalescing_groups//num_m_groups, l_ng)

        assert num_n_groups == 1, "if have multiple groups along n dimension, coalesing is meaningless"

        # for lanegroup_per_cluster, since within thread will not contain this, so must specify this to 1
        split_m_lengths = (self.dotx_m.lanegroup_repeat_m, self.dotx_m.lanegroup_tile_m)
        g_mr, g_mt = split_ndim_length(num_m_groups, split_m_lengths)
        assert g_mt == 1, 'we do not want to split inside this granularity within a thread'

        return g_mr, g_mt # groups in m_repeat, m_step, lanegroup_m_per_wave, lanegroup_m_per_block, lanegroup_m_per_thread

    def get_subgroup_length(self):
        g_mr, g_mt = self.get_subgroups()
        # self.dotx_m.wave_repeat_m, self.dotx_m.wave_step_m, self.dotx_m.lanegroup_m_per_wave(), self.dotx_m.lanegroup_m_per_block(), self.dotx_m.lanegroup_m_per_thread()

        l_mr = self.dotx_m.lanegroup_repeat_m // g_mr
        l_mt = self.dotx_m.lanegroup_tile_m // g_mt
        return l_mr, l_mt

    def get_num_dword_per_group(self):
        assert self.dotx_m.lanegroup_repeat_m * self.dotx_m.lanegroup_tile_m * self.dotx_m.lanegroup_repeat_n % self.coalescing_groups == 0, \
                f"total_acc_c:{self.dotx_m.lanegroup_repeat_m * self.dotx_m.lanegroup_tile_m * self.dotx_m.lanegroup_repeat_n}, coalescing_groups:{self.coalescing_groups}, m_groups:{self.get_length_m_groups()}, inst:{self.dotx_m.inst_mfma.m}x{self.dotx_m.inst_mfma.n}x{self.dotx_m.inst_mfma.k}"
        return self.dotx_m.lanegroup_repeat_m * self.dotx_m.lanegroup_tile_m * self.dotx_m.lanegroup_repeat_n // self.coalescing_groups

    def get_m_index_per_group(self):
        pass

    def get_co_sub_m_index(self):
        pass

    def get_vgpr_usage(self):
        '''
        return the number of vgpr needed for coalescing store process
        '''
        agpr_per_store_group = self.dotx_m.total_acc_c() // self.coalescing_groups
        if self.feat_vgpr_collapse:
            data_byte = amdgpu_precision_data_byte(self.precision)
            inst_sld_byte = (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if self.vector_write_out == 1 else self.vector_write_out) * data_byte
            issues_per_ssgroup = 8 if inst_sld_byte == 16 or inst_sld_byte == 8 else 8

            #print(f"self.get_num_dword_per_group() = {self.get_num_dword_per_group()}")

            num_sld_total_dword = self.get_num_dword_per_group() // (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if self.vector_write_out == 1 else self.vector_write_out)

            #print(f"num_sld_total_dword={num_sld_total_dword}")
            total_lgkmcnt = num_sld_total_dword     # TODO: assume sld is single issue

            assert MAX_LGKMCNT % issues_per_ssgroup == 0
            # print(f"issues_per_ssgroup:{issues_per_ssgroup}, total_lgkmcnt:{total_lgkmcnt}, get_num_dword_per_group:{self.get_num_dword_per_group()}, vector_write_out:{self.vector_write_out}")

            # we need further split based on issues_per_ssgroup
            split_sld_groups = (total_lgkmcnt + issues_per_ssgroup - 1) // issues_per_ssgroup
            #print(f"agpr_per_store_group={agpr_per_store_group}")

            agpr_per_store_split_sld_group = (agpr_per_store_group + split_sld_groups - 1) // split_sld_groups
            assert agpr_per_store_split_sld_group >= 4

            #print(f"agpr_per_store_split_sld_group={agpr_per_store_split_sld_group}")

            return agpr_per_store_split_sld_group
        else:
            return agpr_per_store_group

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
            self._emit(f"; init_co_lds_offset for xdlops")
            '''
            gemm_m_shrink is in multi-dimension.
            then, consider that introduced by granularity
            '''
            self._emit(f"v_lshrrev_b32 v[{v_tmp4}], {utility_log2(ctrl.dotx_m.lanegroup_m_per_thread())}, v[{v_gemm_im}]")
            self._emit(f"v_and_b32 v[{v_tmp4}],  {ctrl.dotx_m.lanegroup_m_per_cluster() - 1} v[{v_tmp4}]   ; thread id of lanegroup_m_per_cluster")
            self._emit(f"v_lshlrev_b32 v[{v_co_sst}], {utility_log2(ctrl.dotx_m.lanegroup_m_per_thread())}, v[{v_tmp4}]")

            if ctrl.dotx_m.block_m_per_lanegroup() != 1:
                length_above_block_m_per_lanegroup = ctrl.dotx_m.lanegroup_m_per_block() * ctrl.dotx_m.lanegroup_m_per_cluster() * \
                                                    ctrl.dotx_m.lanegroup_m_per_thread()
                self._emit(f"v_lshrrev_b32 v[{v_tmp4}+1], {utility_log2(length_above_block_m_per_lanegroup)}, v[{v_gemm_im}]")
                self._emit(f"v_and_b32 v[{v_tmp4}+1], {ctrl.dotx_m.block_m_per_lanegroup() - 1}  , v[{v_tmp4}+1]   ; thread id of block_m_per_lanegroup")
                assert length_above_block_m_per_lanegroup % g_mb == 0, f"length_above_block_m_per_lanegroup:{length_above_block_m_per_lanegroup}, g_mb:{g_mb}"
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp4}+1], {utility_log2(length_above_block_m_per_lanegroup // g_mb)}, v[{v_co_sst}]")

            if ctrl.dotx_m.waves_per_m() != 1:
                length_above_waves_per_m = ctrl.dotx_m.wave_step_m * ctrl.dotx_m.lanegroup_m_per_wave() * \
                                                    ctrl.dotx_m.lanegroup_m_per_block() * ctrl.dotx_m.block_m_per_lanegroup() * \
                                                    ctrl.dotx_m.lanegroup_m_per_thread() * ctrl.dotx_m.lanegroup_m_per_cluster()
                self._emit(f"v_lshrrev_b32 v[{v_tmp4}+2], {utility_log2(length_above_waves_per_m)}, v[{v_gemm_im}]  ; thread id of waves_per_m")
                assert length_above_waves_per_m % (g_ms * g_mw * g_mb) == 0, f"length_above_waves_per_m:{length_above_waves_per_m}, g_ms:{g_ms} g_mw:{g_mw} g_mb:{g_mb}"
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp4}+2], {utility_log2(length_above_waves_per_m // (g_ms * g_mw * g_mb))}, v[{v_co_sst}]")

            if ctrl.vector_write_out == 1:
                self._emit(f"v_lshrrev_b32 v[{v_tmp4}], {utility_log2(AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M)}, v[{v_co_sst}]")
                self._emit(f"v_lshlrev_b32 v[{v_tmp4}+1], {utility_log2(AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M)}, v[{v_gemm_in}]   ; implicit transpose with m granularity:{AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M} while store")
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp4}], {utility_log2(ctrl.dotx_m.macro_tile_n * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M)}, v[{v_tmp4}+1]")
            else:
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_co_sst}], {utility_log2(ctrl.dotx_m.macro_tile_n)}, v[{v_gemm_in}]")

            # gemm_m_shrink = g_mw * g_mb * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M     # => granularity shrink
            # gemm_m_shrink = AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M
            # if gemm_m_shrink != 1:
            #     self._emit(f"v_lshrrev_b32 v[{v_tmp4}], {utility_log2(gemm_m_shrink)}, v[{v_gemm_im}]")
            #     self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp4}], {utility_log2(ctrl.dotx_m.macro_tile_n * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M)}, v[{v_tmp4}+1]")
            # else:
            #     assert False, "impossible"
            self._emit(f"v_lshlrev_b32 v[{v_co_sst}], {utility_log2(data_byte)}, v[{v_co_sst}]")
            self._emit(f"v_lshlrev_b32 v[{v_co_sld}], {utility_log2(data_byte * (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else ctrl.vector_write_out))}, v[{v_tid}]")

        return self._get_deferred()

    def init_co_sub_m_index(self, v_co_sub_m_index, v_tid, v_tmp6):
        def flatten(x):
            from functools import reduce
            return reduce(lambda a, b: a*b, x, 1)
        ctrl = self.ctrl
        # need use v_co_sub_m_index to calculate v offset in m direction
        g_mr, g_ms, g_mw, g_mb, g_mt = ctrl.get_subgroups()
        l_mr, l_ms, l_mw, l_mb, l_mt = ctrl.get_subgroup_length()
        n_mc = ctrl.dotx_m.lanegroup_m_per_cluster()       # this iteration is among different thread
        n_ml = ctrl.dotx_m.block_m_per_lanegroup()         # this iteration is among different thread
        n_mv = ctrl.dotx_m.waves_per_m()                   # this iteration is among different thread
        ttm = ctrl.get_transposed_thread_mapping()
        nd_stride = [l_mt, n_mc, n_ml, g_mb * l_mb, l_mw * g_mw, g_ms * l_ms, n_mv, 1 ]

        with self._deferred_context():
            self._emit(f"; init_co_sub_m_index xdlops, block_size:{ctrl.dotx_m.block_size()}, macro-tile:{ctrl.dotx_m.macro_tile_m}x{ctrl.dotx_m.macro_tile_n} sub_m_index:{ctrl.get_co_sub_m_index()}")
            self._emit(f"; g_mr:{g_mr}, g_ms:{g_ms}, g_mw:{g_mw}, g_mb:{g_mb}, g_mt:{g_mt} | l_mr:{l_mr}, l_ms:{l_ms}, l_mw:{l_mw}, l_mb:{l_mb}, l_mt:{l_mt} | n_mc:{n_mc}, n_ml:{n_ml}, n_mv:{n_mv}")
            self._emit(f"; nd_stride:{nd_stride}")
            c_m0 = ttm.c_m0()
            if c_m0 == 1:
                # give a chance to early exit, only let the co_sub_m to be zero
                self._emit(f"v_mov_b32 v[{v_co_sub_m_index}], 0")
            else:
                if ctrl.vector_write_out == 1:
                    self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {utility_log2(ctrl.dotx_m.macro_tile_n)}, v[{v_tid}]   ; get tid along m")
                else:
                    self._emit(f"v_lshlrev_b32 v[{v_tmp6}], {utility_log2(ctrl.vector_write_out)}, v[{v_tid}]")
                    self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {utility_log2(ctrl.dotx_m.macro_tile_n)}, v[{v_tmp6}]  ; get tid along m")

                v_idx = 0
                # iterate all dimensions
                if ctrl.vector_write_out != 1:
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
                if l_mb != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {l_mb - 1}, v[{v_co_sub_m_index}]                   ; => x_mb")
                    v_idx = v_idx + 1
                    c_m0  = c_m0 // l_mb
                    if c_m0 not in (0, 1):
                        self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {utility_log2(l_mb)}  ,v[{v_co_sub_m_index}]")
                if l_mw != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {l_mw - 1}, v[{v_co_sub_m_index}]                   ; => x_mw")
                    v_idx = v_idx + 1
                    c_m0  = c_m0 // l_mw
                    if c_m0 not in (0, 1):
                        self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {utility_log2(l_mw)}  ,v[{v_co_sub_m_index}]")
                if l_ms != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {l_ms - 1}, v[{v_co_sub_m_index}]                   ; => x_ms")
                    v_idx = v_idx + 1
                    c_m0  = c_m0 // l_ms
                    if c_m0 not in (0, 1):
                        self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {utility_log2(l_ms)}  ,v[{v_co_sub_m_index}]")
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

                c_m0 = ttm.c_m0()
                v_idx_r = 0
                # self._emit(f"v_mov_b32 v[{v_co_sub_m_index}], 0")
                # if n_mc != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_add_u32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], v[{v_co_sub_m_index}]  ; => accumulate x_mc")
                #     v_idx_r = v_idx_r + 1
                #     c_m0    = c_m0 // n_mc
                # if n_ml != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], {utility_log2(flatten(nd_stride[0:1]))}, v[{v_co_sub_m_index}]  ; => accumulate x_ml")
                #     v_idx_r = v_idx_r + 1
                #     c_m0    = c_m0 // n_ml
                # if l_mb != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], {utility_log2(flatten(nd_stride[0:2]))}, v[{v_co_sub_m_index}]  ; => accumulate x_mb")
                #     v_idx_r = v_idx_r + 1
                #     c_m0    = c_m0 // l_mb
                # if l_mw != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], {utility_log2(flatten(nd_stride[0:3]))}, v[{v_co_sub_m_index}]  ; => accumulate x_mw")
                #     v_idx_r = v_idx_r + 1
                #     c_m0    = c_m0 // l_mw
                # if l_ms != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], {utility_log2(flatten(nd_stride[0:4]))}, v[{v_co_sub_m_index}]  ; => accumulate x_ms")
                #     v_idx_r = v_idx_r + 1
                #     c_m0    = c_m0 // l_ms
                # if n_mv != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], {utility_log2(flatten(nd_stride[0:5]))}, v[{v_co_sub_m_index}]  ; => accumulate x_mv")
                #     v_idx_r = v_idx_r + 1
                #     c_m0    = c_m0 // n_mv
                # if l_mr != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], {utility_log2(flatten(nd_stride[0:6]))}, v[{v_co_sub_m_index}]  ; => accumulate x_mr")
                accumulate_co_sub_m = pretty_accumulate_co_sub_m_t()
                if ctrl.vector_write_out != 1:
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
                if l_mb != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:3])) + "      ; => accumulate x_mb")
                    v_idx_r = v_idx_r + 1
                    c_m0    = c_m0 // l_mb
                if l_mw != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:4])) + "      ; => accumulate x_mw")
                    v_idx_r = v_idx_r + 1
                    c_m0    = c_m0 // l_mw
                if l_ms != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:5])) + "      ; => accumulate x_ms")
                    v_idx_r = v_idx_r + 1
                    c_m0    = c_m0 // l_ms
                if n_mv != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:6])) + "      ; => accumulate x_mv")
                    v_idx_r = v_idx_r + 1
                    c_m0    = c_m0 // n_mv
                if l_mr != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:7])) + "      ; => accumulate x_mr")

                # if ctrl.vector_write_out == 1:
                #     self._emit(f"v_lshlrev_b32 v[{v_co_sub_m_index}], {utility_log2(AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M)}, v[{v_co_sub_m_index}]")

                assert v_idx == v_idx_r, "please check!"

        return self._get_deferred()

    def init_co_sub_n_index(self, v_co_sub_n_index, v_tid, v_tmp2):
        '''
        in n dimension, always have one thread per column
        '''
        ctrl = self.ctrl
        # need use v_co_sub_n_index to calculate v offset in n direction

        with self._deferred_context():
            self._emit(f"; init_co_sub_n_index xdlops")
            if ctrl.vector_write_out == 1:
                self._emit(f"v_and_b32 v[{v_co_sub_n_index}], {ctrl.dotx_m.macro_tile_n - 1}, v[{v_tid}]")
            else:
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {utility_log2(ctrl.vector_write_out)}, v[{v_tid}]")
                self._emit(f"v_and_b32 v[{v_co_sub_n_index}], {ctrl.dotx_m.macro_tile_n - 1}, v[{v_tmp2}]")
        return self._get_deferred()

    '''
    def get_vgpr_usage(self):
        ctrl = self.ctrl
        agpr_per_store_group = ctrl.dotx_m.total_acc_c() // ctrl.coalescing_groups
        if ctrl.feat_vgpr_collapse:
            data_byte = amdgpu_precision_data_byte(ctrl.precision)
            inst_sld_byte = (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else ctrl.vector_write_out) * data_byte
            issues_per_ssgroup = 4 if inst_sld_byte == 16 or inst_sld_byte == 8 else 8

            num_sld_total_dword = ctrl.get_num_dword_per_group() // (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else ctrl.vector_write_out)

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
        n_mc = ctrl.dotx_m.lanegroup_m_per_cluster()       # this iteration is among different thread
        n_ml = 1
        n_mv = ctrl.dotx_m.waves_per_m()                   # this iteration is among different thread

        n_nc = ctrl.dotx_m.lanegroup_n_per_cluster()       # this iteration is among different thread
        n_nl = 1
        n_nv = ctrl.dotx_m.waves_per_n()                   # this iteration is among different thread

        nd_stride = [n_mc, n_ml, n_mv, 1 ]

        no_s_out_offset = s_out_offset is None
        inst_sst_byte = ctrl.vector_write_out * data_byte
        inst_sld_byte = (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else ctrl.vector_write_out) * data_byte

        # for xdlops, always consider granularity in column, hence here is always ds_write_b128/ds_read_b128
        inst_sst = inst_ds_write_t(inst_sst_byte)
        inst_sld = inst_ds_read_t(inst_sld_byte)
        if ctrl.gemm_k_global_split:
            v_pack = 2 if ctrl.vector_write_out == 2 and data_byte == 2 else 1
            inst_gst = inst_buffer_atomic_add_dword_t(ctrl.vector_write_out * data_byte, v_pack) 
        else:
            inst_gst = inst_buffer_store_t(ctrl.vector_write_out * data_byte)
       

        s_out_offset_itr = sym_t(s_tmp6(0))
        # s_thread_m_stride = sym_t(s_tmp4(1))

        #m_index_per_group = ctrl.get_m_index_per_group()
        # thread_m_stride = ctrl.get_thread_m_stride()

        #assert len(m_index_per_group) == ctrl.coalescing_groups

        t_mr, t_nr, t_nt, t_mt = ctrl.dotx_m.acc_c_lengths()
        s_mr, s_nr, s_nt, s_mt = t_nr * t_nt * t_mt, \
                                 t_nt * t_mt, \
                                 t_mt, \
                                 1

        i_g_list = list()
        for i_g_mr, i_g_mt in itertools.product(range(g_mr), range(g_mt)):
            i_g_list.append((i_g_mr, i_g_mt))

        with self._deferred_context():
            self._emit(f"; coalescing store, mapping:{ctrl.dotx_m.serialize()}")
            self._emit(f"; coalescing_groups:{ctrl.coalescing_groups}, num_dword_per_group:{ctrl.get_num_dword_per_group()}")
            self._emit(f"; init_co_sub_m_index xdlops, block_size:{ctrl.dotx_m.block_size()}, macro-tile:{ctrl.dotx_m.macro_tile_m}x{ctrl.dotx_m.macro_tile_n} sub_m_index:{ctrl.get_co_sub_m_index()}")
            self._emit(f"; g_mr:{g_mr}, g_mt:{g_mt} | l_mr:{l_mr}, l_mt:{l_mt} | n_mc:{n_mc}, n_ml:{n_ml}, n_mv:{n_mv}")
            self._emit(f"; nd_stride:{nd_stride}")

            self._emit(f"s_mul_i32 s[{s_gemm_m1_stride}], {(ctrl.dotx_m.block_size() // ctrl.dotx_m.macro_tile_n) * data_byte}, s[{s_gemm_m1_stride}]")
            self._emit(f"s_barrier")

            for i_group in range(ctrl.coalescing_groups):
                #m_index_start_per_group = m_index_per_group[i_group][0][0]
                #m0_index_start_per_group, m1_index_start_per_group = ctrl.get_m0_m1_index(m_index_start_per_group)

                i_g_mr, i_g_mt = i_g_list[i_group]

                self._emit(f"; start group {i_group}, i_g_mr:{i_g_mr}, i_g_mt:{i_g_mt}")
                #if not ctrl.can_skip_coalescing():
                #    self._emit(f"s_barrier")

                vgpr_index_acc = 0
                for i_mr in range(l_mr):
                    gpr_m_offset = i_mr * s_mr
                    sst_m_offset = i_mr * (ctrl.dotx_m.macro_tile_m // ctrl.dotx_m.lanegroup_repeat_m // ctrl.vector_write_out)
                    #print(f"sst_m_offset={sst_m_offset}")
                    
                    #if ctrl.vector_write_out != 1:
                    #    sst_m_offset = sst_m_offset * l_mt          # ATTENTION! if vector write out, we no longer have shrink granularity, hence need multiply back
                    # iterate through all m within current group
                    for i_nr in range(t_nr):
                        gpr_n_offset = i_nr * s_nr
                        sst_n_offset = i_nr * n_nv * ctrl.dotx_m.lanegroup_n_per_wave() * n_nl * n_nc
                        # self._emit(f" => ctrl.dotx_m.wave_step_n:{ctrl.dotx_m.wave_step_n}, ctrl.dotx_m.lanegroup_n_per_wave():{ctrl.dotx_m.lanegroup_n_per_wave()}, n_nl:{n_nl}, n_nc:{n_nc}")
                        # agpr_index = a_group_start_index + gpr_m_offset + gpr_n_offset
                        # vgpr_index = gpr_m_offset + gpr_n_offset
                        vgpr_index = vgpr_index_acc
                        sst_offset = (sst_m_offset * ctrl.dotx_m.macro_tile_n * ctrl.vector_write_out + \
                                     sst_n_offset * ctrl.vector_write_out) * data_byte
                        #assert sst_offset < lds_size_per_group and sst_offset + (m_index_per_group[i_group][-1][0] -  m_index_per_group[i_group][0][0]) * data_byte < lds_size_per_group

                        # for fp16 and bf16, vgpr need to be cast to 16 bits
                        if ctrl.precision == 'fp16':
                            
                            for i in range(ctrl.vector_write_out):
                                if i % 2 == 0:
                                    self._emit(f"v_cvt_f16_f32_sdwa v[{v_c_tmp(i // 2)}], v[{v_c(vgpr_index + i)}]")
                                else:
                                    self._emit(f"v_cvt_f16_f32_sdwa v[{v_c_tmp(i // 2)}], v[{v_c(vgpr_index + i)}] dst_sel:WORD_1")
                        elif ctrl.precision == 'int8':
                            # CAUSION: must have a symbol s_0xff and pre inited with 0xff
                            pass

                        
                        idword = sst_offset // (data_byte * (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else 1))
                        if ctrl.vector_write_out == 1:
                            self._emit(inst_sst(v_co_sst(), v_c(vgpr_index), sst_offset) + \
                                f"   ; idword:{idword}({idword // ctrl.dotx_m.macro_tile_n},{idword % ctrl.dotx_m.macro_tile_n}),  {sst_m_offset}x{sst_n_offset} |" + \
                                f" /{(AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else 1)}, i_mr:{i_mr}, x  i_nr:{i_nr}")
                        else:
                            self._emit(inst_sst(v_co_sst(), v_c_tmp(), sst_offset) + \
                                f' ; idword:{idword}({idword // ctrl.dotx_m.macro_tile_n},{idword % ctrl.dotx_m.macro_tile_n}), {sst_m_offset}x{sst_n_offset}, i_mr:{i_mr}, x  i_nr:{i_nr}')
                        #vgpr_index_acc += (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else ctrl.vector_write_out)

                        if ctrl.feat_vgpr_collapse:
                            if vgpr_index_acc + ctrl.vector_write_out >= ctrl.get_vgpr_usage():
                                vgpr_index_acc = 0
                            else:
                                vgpr_index_acc += ctrl.vector_write_out # to have a balanced vgpr reusage
                        else:
                            vgpr_index_acc += AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M     # can always use granularity to increase acc vgpr index

                def emit_calculate_out_offset_itr_m(i_m, i_m0, i_m1):
                    # self._emit(f"; i_m:{i_m},  i_m0:{i_m0}xi_m1:{i_m1}")
                    comments = f"   ; i_m:{i_m}(i_m0:{i_m0},i_m1:{i_m1})"
                    if ctrl.co_m_update_os_functor:
                        self._emit(ctrl.co_m_update_os_functor(i_m, i_m0, i_m1))        # TODO: better sigture
                    else:
                        if s_gemm_m0_stride is not None:
                            self._emit(f"s_mul_i32 s[{s_tmp6(2)}], {i_m0}, s[{s_gemm_m0_stride}]")
                            self._emit(f"s_mul_i32 s[{s_tmp6(3)}], {i_m1}, s[{s_gemm_m1_stride}]")
                            self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_tmp6(2)}], s[{s_tmp6(3)}]" + comments)
                            if not no_s_out_offset:
                                self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_out_offset}], s[{s_out_offset_itr()}] ")
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
                emit_calculate_out_offset_itr_m(0, 0, 0)

                issue_list = []
                num_sld_total_dword = ctrl.get_num_dword_per_group() // (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else ctrl.vector_write_out)
                
                self._emit(f"s_waitcnt lgkmcnt(0)")
                self._emit(f"s_barrier")
                for i_d in range(num_sld_total_dword):
                    vgpr_index = i_d * (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else ctrl.vector_write_out) * data_byte // 4 # when data byte is 2, only cost 2 vgpr per time
                    sld_offset = i_d * (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else ctrl.vector_write_out) * ctrl.block_size  * data_byte
                    # self._emit(inst_sld(v_c(vgpr_index), v_co_sld(), sld_offset))
                    issue_list.append(inst_sld.get_issues(sld_offset))

                total_lgkmcnt = utility_flatten_list_accumulate(issue_list)
                issues_per_ssgroup = 8 if inst_sld_byte == 16 or inst_sld_byte == 8 else 8

                assert MAX_LGKMCNT % issues_per_ssgroup == 0

                # we need further split based on issues_per_ssgroup
                split_sld_groups = (total_lgkmcnt + issues_per_ssgroup - 1) // issues_per_ssgroup
                num_issues_per_ssgroup = len(issue_list) // split_sld_groups
                assert (ctrl.get_num_dword_per_group() // ctrl.vector_write_out) % split_sld_groups == 0, "TODO: need adjust ssgroup value based on dword per group"
                num_gst_per_ssgroup = ctrl.get_num_dword_per_group() // ctrl.vector_write_out // split_sld_groups

                assert num_sld_total_dword % split_sld_groups == 0, "TODO: need adjust"
                num_sld_per_ssgroup = num_sld_total_dword // split_sld_groups

                for i_ssgroup in range(split_sld_groups):
                    
                    self._emit(f";   load from lds, i_ssgroup:{i_ssgroup}, num_sld_per_ssgroup:{num_sld_per_ssgroup}")
                    for i_d in range(num_sld_per_ssgroup):
                        vgpr_index = (i_d + (i_ssgroup if not ctrl.feat_vgpr_collapse else 0) * num_sld_per_ssgroup) * (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else ctrl.vector_write_out) * data_byte // 4 # when data byte is 2, only cost 2 vgpr per time
                        sld_offset = (i_d + i_ssgroup * num_sld_per_ssgroup) * (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else ctrl.vector_write_out) * ctrl.block_size  * data_byte
                        self._emit(inst_sld(v_c(vgpr_index), v_co_sld(), sld_offset))
                    current_issue_list = issue_list[i_ssgroup * num_issues_per_ssgroup : (i_ssgroup+1) * num_issues_per_ssgroup]
                    if not ctrl.feat_co_m_flag_check and (v_store_flag is not None and type(v_store_flag) is str):
                        self._emit(v_cmpx_eq_u32("vcc", 1, v_store_flag))
                    self._emit(f";   store to global, m index start")

                    for i_gst in range(num_gst_per_ssgroup):
                        i_gst_flat = i_gst + i_ssgroup * num_gst_per_ssgroup
                        if len(current_issue_list) != 0:
                            if i_gst % (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else 1) == 0:
                                i_issues =  (i_gst // (AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M if ctrl.vector_write_out == 1 else 1)) + 1
                                i_issue_list = current_issue_list[i_issues:]
                                i_issue_cnt = utility_flatten_list_accumulate(i_issue_list) if len(i_issue_list) != 0 else 0
                                self._emit(f"s_waitcnt lgkmcnt({i_issue_cnt})")
                        # vdata, vaddr, srsrc, soffset, offset
                        if not ctrl.feat_co_m_flag_check and (s_k is not None):
                            self._emit(f"v_cmp_gt_u32 vcc, s[{s_k()}], v[{v_tmp0()}]")
                            self._emit(f"s_and_saveexec_b64 s[{s_tmp6(4)}:{s_tmp6(5)}], vcc")
                        elif ctrl.feat_co_m_flag_check:
                            self._emit(ctrl.co_m_flag_check_start_functor())
                        cur_vgpr_gst = (i_gst_flat if not ctrl.feat_vgpr_collapse else i_gst) * ctrl.vector_write_out//(4 // data_byte)
                        lo_hi = i_gst_flat % 2 if ctrl.precision == 'fp16' and ctrl.vector_write_out == 1 else 0
                        self._emit(inst_gst(v_c(cur_vgpr_gst), v_out_offset, s_p_out, s_out_offset_itr(), 0, lo_hi))
                        if not ctrl.feat_co_m_flag_check and (s_k is not None):
                            self._emit(f"s_or_b64 exec, exec, s[{s_tmp6(4)}:{s_tmp6(5)}]")
                        elif ctrl.feat_co_m_flag_check:
                            self._emit(ctrl.co_m_flag_check_reset_functor())
                        if ctrl.precision == 'int8' and ctrl.vector_write_out == 1:
                            if i_gst_flat % 4 != 3:
                                self._emit(f"v_lshrrev_b32 v[{v_c(cur_vgpr_gst)}], 8, v[{v_c(cur_vgpr_gst)}]")

                        if i_gst_flat != (ctrl.get_num_dword_per_group() // ctrl.vector_write_out) - 1:
                            i_m = i_gst + 1
                            # self._emit(f"; >>>>>> i_m :{i_m}, i_gst:{i_gst}, m_index_per_group[i_group][0]:{m_index_per_group[i_group][0]}")

                            emit_calculate_out_offset_itr_m(i_m, 0, 0)
                    if not ctrl.feat_co_m_flag_check and (v_store_flag is not None and type(v_store_flag) is str):
                        self._emit(f"s_mov_b64 exec, -1")

                if ctrl.feat_vgpr_collapse:
                    agpr_per_store_group = ctrl.dotx_m.total_acc_c() // ctrl.coalescing_groups
                    assert ctrl.get_vgpr_usage() == ((agpr_per_store_group + split_sld_groups - 1) // split_sld_groups), f"vgpr_usage:{ctrl.get_vgpr_usage()}, agpr_per_store_group:{agpr_per_store_group}, split_sld_groups:{split_sld_groups}"
            # do some assert
            #agpr_consume_list.sort()
            #assert agpr_consume_list == [x for x in range(ctrl.dotx_m.total_acc_c())], f"agpr_consume_list:{agpr_consume_list}"
            #if agpr_consume_list != [x for x in range(ctrl.dotx_m.total_acc_c())]:
            #    self._emit(f"; [WRONG!] agpr ~~~~~~~~, {agpr_consume_list}")
        return self._get_deferred()
