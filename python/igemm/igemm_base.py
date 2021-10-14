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
from __future__ import print_function
import sys
import math
from ..codegen import *
from ..operations import *


IGEMM_GTC_FEAT_ALLOW_LDS_REORDER = 0
IGEMM_GTC_FEAT_PRECACHE_SOFFSET = 1
IGEMM_GTC_FEAT_LOCAL_PREFETCH = 1
IGEMM_GTC_FEAT_FMA_INTERLEAVE = 1
IGEMM_GTC_FEAT_MAGIC_DIVISION = 1
IGEMM_GTC_FEAT_SOURCE_ACCESS_ENCODING_KERNEL_NAME = 0

# IGEMM_GTC_TENSOR_LAYOUT_NCHW = ((1 << 4) | 0)
# IGEMM_GTC_TENSOR_LAYOUT_NHWC = ((1 << 4) | 1)
# IGEMM_GTC_TENSOR_LAYOUT_CNHW = ((1 << 4) | 2)


IGEMM_GTC_TUNABLE_FMA_TYPE_MAC               = 'mac'
IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS             = 'dlops'
IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS            = 'xdlops'


IGEMM_GTC_TUNABLE_SOURCE_ACCESS_ORDER_GEMM_M_GEMM_N       = 0    # m*n, load gemm_n first
IGEMM_GTC_TUNABLE_SOURCE_ACCESS_ORDER_GEMM_N_GEMM_M       = 1    # n*m, load gemm_m first

def igemm_get_vector_size(v):
    vec_size = 1
    if v % 4 == 0:
        vec_size = 4
    elif v % 2 == 0:
        vec_size = 2
    else:
        pass
    return vec_size

# compute next power of 2
def igemm_next_pow2(n):
    if n == 0:
        return 1
    if n & (n - 1) == 0:
        return n
    while n & (n - 1) > 0:
        n &= (n - 1)
    return n << 1

def igemm_next_mul(n, mul):
    d = n // mul
    d = d + (1 if (n % mul != 0) else 0)
    return d * mul

def igemm_is_pow2(v):
    return v and (not(v & (v - 1)))

def igemm_log2(v):
    assert (v and (not(v & (v - 1)))), 'v:{} must be power of 2'.format(v)
    return int(math.log2(v))

def igemm_get_epack_length(precision):
        # GetEPackLength
        epack = 1
        if precision == AMDGPU_PRECISION_FP16:
            # todo: xdlops check
            epack = 2
        elif precision == AMDGPU_PRECISION_BF16:
            epack = 2
        return epack

def igemm_gcd(a, b):
    # math.gcd new in python 3.5
    return math.gcd(a, b)

def igemm_lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

def igemm_flatten_list_product(x):
    assert type(x) is list
    from functools import reduce
    return reduce(lambda a, b: a*b, x)

def igemm_flatten_list_accumulate(x):
    assert type(x) is list
    from functools import reduce
    return reduce(lambda a, b: a+b, x)

def get_igemm_gtc_fma_type(tunable_dict):
    assert type(tunable_dict) is dict
    if 'gemm_m_per_thread' in tunable_dict and 'gemm_n_per_thread' in tunable_dict:
        if tunable_dict['arch'] == 'gfx900':
            return IGEMM_GTC_TUNABLE_FMA_TYPE_MAC
        if tunable_dict['arch'] == 'gfx906':
            return IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS
        if tunable_dict['arch'] in ('gfx908', 'gfx90a'):
            return IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS
    if 'wave_tile_m' in tunable_dict and 'wave_tile_n' in tunable_dict:
        assert tunable_dict['arch'] in ('gfx908', 'gfx90a')
        return IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS
    assert False

def get_igemm_gtc_gemm_k_global_split(tunable_dict):
    assert type(tunable_dict) is dict
    if tunable_dict['arch'] in ('gfx908', 'gfx90a'):
        gemm_k_global_split = utility_dict_with_default_t(tunable_dict)('gemm_k_global_split', 0)
        if gemm_k_global_split > 0:
            return 1
        else:
            return 0
    else:
        return 0

def igemm_get_fma_type_from_arch_config(arch_config):
    assert type(arch_config) is amdgpu_arch_config_t
    if arch_config.use_xdlops:
        return IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS
    if arch_config.use_dlops:
        return IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS
    return IGEMM_GTC_TUNABLE_FMA_TYPE_MAC

class igemm_gtc_tunable_parameter_t(object):
    '''
    generic tensor contraction
    '''
    def __init__(self, tunable_dict):
        self.tensor_layout                      = utility_dict_with_default_t(tunable_dict)('tensor_layout', 'nchw')
        self.gemm_m_per_block                   = tunable_dict['gemm_m_per_block']
        self.gemm_n_per_block                   = tunable_dict['gemm_n_per_block']
        self.gemm_k_per_block                   = tunable_dict['gemm_k_per_block']
        self.fma_type                           = get_igemm_gtc_fma_type(tunable_dict)
        if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
            self.gemm_m_per_thread              = tunable_dict['gemm_m_per_thread']
            self.gemm_m_level0_cluster          = tunable_dict['gemm_m_level0_cluster']
            self.gemm_m_level1_cluster          = tunable_dict['gemm_m_level1_cluster']
            self.gemm_n_per_thread              = tunable_dict['gemm_n_per_thread']
            self.gemm_n_level0_cluster          = tunable_dict['gemm_n_level0_cluster']
            self.gemm_n_level1_cluster          = tunable_dict['gemm_n_level1_cluster']
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            self.wave_tile_m                    = tunable_dict['wave_tile_m']
            self.wave_step_m                    = tunable_dict['wave_step_m']
            self.wave_repeat_m                  = tunable_dict['wave_repeat_m']
            self.wave_tile_n                    = tunable_dict['wave_tile_n']
            self.wave_step_n                    = tunable_dict['wave_step_n']
            self.wave_repeat_n                  = tunable_dict['wave_repeat_n']
            self.wave_tile_k                    = utility_dict_with_default_t(tunable_dict)('wave_tile_k', 1)
        else:
            assert False

        self.tensor_a_pass_through              = utility_dict_with_default_t(tunable_dict)('tensor_a_pass_through', 0)
        self.tensor_b_pass_through              = utility_dict_with_default_t(tunable_dict)('tensor_b_pass_through', 0)
        self.tensor_a_thread_lengths            = tunable_dict['tensor_a_thread_lengths']     # list!
        self.tensor_a_cluster_lengths           = tunable_dict['tensor_a_cluster_lengths']    # list!
        self.tensor_b_thread_lengths            = tunable_dict['tensor_b_thread_lengths']     # list!
        self.tensor_b_cluster_lengths           = tunable_dict['tensor_b_cluster_lengths']    # list!
        self.direction                          = tunable_dict['direction']
        self.precision                          = tunable_dict['precision']
        self.nxb                                = tunable_dict['nxb']           # multiplier of b
        self.nxe                                = tunable_dict['nxe']           # muptiplier of e. here if 0, means x=y=1
        default_mh                              = 1 if (self.direction == 'bwd' and self.tensor_layout == "nhwc" and self.nxe != 0) else 0
        self.multihead                          = utility_dict_with_default_t(tunable_dict)('multihead', default_mh)
        self.gemm_k_global_split                = get_igemm_gtc_gemm_k_global_split(tunable_dict)
        self.allow_lds_reorder                  = utility_dict_with_default_t(tunable_dict)('allow_lds_reorder', IGEMM_GTC_FEAT_ALLOW_LDS_REORDER)
        self.precache_soffset                   = utility_dict_with_default_t(tunable_dict)('precache_soffset', IGEMM_GTC_FEAT_PRECACHE_SOFFSET)

        default_source_access_order             = IGEMM_GTC_TUNABLE_SOURCE_ACCESS_ORDER_GEMM_N_GEMM_M if (self.direction == 'fwd' and self.tensor_layout == 'nchw') \
                                                        else IGEMM_GTC_TUNABLE_SOURCE_ACCESS_ORDER_GEMM_M_GEMM_N
        self.source_access_order                = utility_dict_with_default_t(tunable_dict)('source_access_order', default_source_access_order)

        self.gemm_m_unmerge_cluster             = utility_dict_with_default_t(tunable_dict)('gemm_m_unmerge_cluster', 0)
        self.gemm_n_unmerge_cluster             = utility_dict_with_default_t(tunable_dict)('gemm_n_unmerge_cluster', 0)
        self.gemm_k_unmerge_cluster             = utility_dict_with_default_t(tunable_dict)('gemm_k_unmerge_cluster', 0)     # maybe no need support for 1
        self.vector_store                       = utility_dict_with_default_t(tunable_dict)('vector_store', 0)
        self.gemm_k_global_split                = utility_dict_with_default_t(tunable_dict)('gemm_k_global_split', 0)
        self.merge_e                            = utility_dict_with_default_t(tunable_dict)('merge_e', 0)   # indicate if merge c*y*x for gemm_k (fwd), useful in nhwc fwd
        #  x -(unmerge)-> x0*x1, if set to 1, means cluster first iterate all x1
        # hence stride of x0 should not be x1, but be total number of x divide by x0

        assert type(self.tensor_a_thread_lengths) is list and type(self.tensor_a_cluster_lengths) is list
        assert type(self.tensor_b_thread_lengths) is list and type(self.tensor_b_cluster_lengths) is list
        # assert type(self.opt_1x1) is bool
        assert self.direction in ('fwd', 'bwd', 'wrw')
        assert self.precision in ('fp32', 'fp16', 'bf16', 'int8')
        if self.tensor_layout == "nchw":
            assert self.nxb in (1,4,8,16,32,64,128,256)
        elif self.tensor_layout == "nhwc":
            assert self.nxb == 0, 'nhwc now no need have different nxb value'
        else:
            assert False
        assert self.nxe in (0,1)

        # TODO: better specify
        if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
            self.block_size                     = self.gemm_m_level0_cluster * self.gemm_n_level0_cluster * self.gemm_m_level1_cluster * self.gemm_n_level1_cluster

        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            assert self.gemm_m_per_block % (self.wave_tile_m * self.wave_step_m * self.wave_repeat_m) == 0
            assert self.gemm_n_per_block % (self.wave_tile_n * self.wave_step_n * self.wave_repeat_n) == 0
            waves_per_m = self.gemm_m_per_block // (self.wave_tile_m * self.wave_step_m * self.wave_repeat_m)
            waves_per_n = self.gemm_n_per_block // (self.wave_tile_n * self.wave_step_n * self.wave_repeat_n)
            self.block_size                     = waves_per_m * waves_per_n * amdgpu_wave_size(tunable_dict['arch'])

        assert self.block_size == igemm_flatten_list_product(self.tensor_a_cluster_lengths), f"block_size:{self.block_size}, a_cluster_lengths:{self.tensor_a_cluster_lengths}, {self.gemm_m_per_block} - {self.wave_tile_m}x{self.wave_step_m}x{self.wave_repeat_m}, {self.gemm_n_per_block} - {self.wave_tile_n}x{self.wave_step_n}x{self.wave_repeat_n}"
        assert self.block_size == igemm_flatten_list_product(self.tensor_b_cluster_lengths), f"block_size:{self.block_size}, b_cluster_lengths:{self.tensor_b_cluster_lengths}"

        def _unmerge_x1_from_e(unroll_k, nxe):
            if nxe == 0:
                return unroll_k # not used, 1x1 special
            if unroll_k % nxe == 0:
                return unroll_k // nxe
            return unroll_k     # not used

        if self.direction == 'fwd':
            if self.tensor_layout == 'nchw':
                assert self.gemm_n_per_block % self.nxb == 0
                self.unmerge_sub_n = self.gemm_n_per_block // self.nxb
                self.unmerge_sub_k = 1                          # not used
                self.unmerge_sub_c = _unmerge_x1_from_e(self.gemm_k_per_block, self.nxe)
            elif self.tensor_layout == 'nhwc':
                self.unmerge_sub_n = 1                          # not used
                self.unmerge_sub_k = 1                          # not used
                self.unmerge_sub_c = 1                          # not used
            else:
                assert False
        elif self.direction == 'bwd':
            if self.tensor_layout == 'nchw':
                assert self.gemm_n_per_block % self.nxb == 0
                self.unmerge_sub_n = self.gemm_n_per_block // self.nxb
                self.unmerge_sub_k = _unmerge_x1_from_e(self.gemm_k_per_block, self.nxe)
                self.unmerge_sub_c = 1                             # not used
            elif self.tensor_layout == 'nhwc':
                self.unmerge_sub_n = 1                          # not used
                self.unmerge_sub_k = 1                          # not used
                self.unmerge_sub_c = 1                          # not used
            else:
                assert False
        else:
            if self.tensor_layout == 'nchw':
                assert self.gemm_k_per_block % self.nxb == 0
                self.unmerge_sub_n = _unmerge_x1_from_e(self.gemm_k_per_block, self.nxb)
                self.unmerge_sub_k = 1
                self.unmerge_sub_c = self.gemm_n_per_block
            elif self.tensor_layout == 'nhwc':
                self.unmerge_sub_n = 1                          # not used
                self.unmerge_sub_k = 1                          # not used
                self.unmerge_sub_c = 1                          # not used

        self.tensor_a_pass_through_interleave_gld = 0 if self.tensor_layout == 'nhwc' else 1
        self.tensor_b_pass_through_interleave_gld = 0 if self.tensor_layout == 'nhwc' else 1

        self.fma_interleave = IGEMM_GTC_FEAT_FMA_INTERLEAVE
        self.local_prefetch_num = 1
        # vector global/lds implicit here
        if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
            self.gemm_m_repeat                  = self.gemm_m_per_block // (self.gemm_m_per_thread * self.gemm_m_level0_cluster * self.gemm_m_level1_cluster)
            self.gemm_n_repeat                  = self.gemm_n_per_block // (self.gemm_n_per_thread * self.gemm_n_level0_cluster * self.gemm_n_level1_cluster)
            # register for a,b,c buffer
            self.num_vgpr_accumulate_c          = (self.gemm_m_repeat * self.gemm_m_per_thread * self.gemm_n_repeat * self.gemm_n_per_thread)
            self.num_vgpr_accumulate_a          = (self.gemm_m_repeat * self.gemm_m_per_thread)
            self.num_vgpr_accumulate_b          = (self.gemm_n_repeat * self.gemm_n_per_thread)

        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            self.local_prefetch_num             = 2 if IGEMM_GTC_FEAT_LOCAL_PREFETCH else 1
            if (self.tensor_a_pass_through and self.wave_repeat_n == 2) or (self.tensor_b_pass_through and self.wave_repeat_m == 2):
                self.local_prefetch_num         = 1
            # register for a,b,c buffer
            xdlops_mapping = get_ctrl_xdlops_mapping_from_wave_tile(self.gemm_m_per_block, self.gemm_n_per_block, self.wave_tile_m, self.wave_tile_n, self.wave_tile_k, 
                    self.wave_repeat_m, self.wave_repeat_n, self.wave_step_m, self.wave_step_n, self.block_size // amdgpu_wave_size(tunable_dict['arch']), self.precision)
            self.num_agpr_accumulate_c          = xdlops_mapping.total_acc_c()
            assert self.num_agpr_accumulate_c == self.gemm_m_per_block * self.gemm_n_per_block // self.block_size, f"block_size:{self.block_size}, {self.gemm_m_per_block}x{self.gemm_n_per_block}x{self.gemm_k_per_block}"
            self.num_vgpr_accumulate_a          = self.wave_step_m * self.wave_repeat_m * xdlops_mapping.inst_mfma.num_v_a * self.local_prefetch_num
            self.num_vgpr_accumulate_b          = self.wave_step_n * self.wave_repeat_n * xdlops_mapping.inst_mfma.num_v_b * self.local_prefetch_num

        self.global_prefetch_a_num              = 2 if self.tensor_a_pass_through and not self.tensor_a_pass_through_interleave_gld else 1
        self.global_prefetch_b_num              = 2 if self.tensor_b_pass_through and not self.tensor_b_pass_through_interleave_gld else 1

        self.num_global_load_a                  = igemm_flatten_list_product(self.tensor_a_thread_lengths)
        self.num_global_load_b                  = igemm_flatten_list_product(self.tensor_b_thread_lengths)

        assert self.num_global_load_a * self.block_size == self.gemm_m_per_block * self.gemm_k_per_block, f"gemm_m_per_block:{self.gemm_m_per_block} - {self.wave_tile_m}x{self.wave_step_m}x{self.wave_repeat_m}, gemm_n_per_block:{self.gemm_n_per_block} - {self.wave_tile_n}x{self.wave_step_n}x{self.wave_repeat_n}, gemm_k_per_block:{self.gemm_k_per_block}"
        assert self.num_global_load_b * self.block_size == self.gemm_n_per_block * self.gemm_k_per_block, f"gemm_m_per_block:{self.gemm_m_per_block} - {self.wave_tile_m}x{self.wave_step_m}x{self.wave_repeat_m}, gemm_n_per_block:{self.gemm_n_per_block} - {self.wave_tile_n}x{self.wave_step_n}x{self.wave_repeat_n}, gemm_k_per_block:{self.gemm_k_per_block}"

        # LDS size
        self.lds_pad_m, self.lds_pad_n = self.get_lds_pad() # LDS pad
        self.lds_a                     = amdgpu_precision_data_byte(self.precision) * self.gemm_k_per_block * self.gemm_m_per_block if not self.tensor_a_pass_through else 0
        self.lds_b                     = amdgpu_precision_data_byte(self.precision) * self.gemm_k_per_block * self.gemm_n_per_block if not self.tensor_b_pass_through else 0
        self.lds_a_np2                 = igemm_next_pow2( self.lds_a) if self.lds_a != 0 else 0
        self.lds_b_np2                 = igemm_next_pow2( self.lds_b) if self.lds_b != 0 else 0
        lds_a_pad                      = self.lds_a_np2 // 32 * (32 + self.lds_pad_m)
        lds_b_pad                      = self.lds_b_np2 // 32 * (32 + self.lds_pad_n)
        self.lds_single                = igemm_next_pow2( self.lds_a_np2 + self.lds_b_np2) if (self.lds_a_np2 + self.lds_b_np2 != 0) else 0
        self.lds_buffer_num            = 2
        self.lds_total                 = self.lds_buffer_num * self.lds_single

        # for case whose tile size is like 128x128x32, the top priority is to keep the occupancy bigger than 2
        # TODO: need to make some compromise in occupancy and lds double buffer
        if self.is_occupancy_decreased():
            self.lds_buffer_num                 = 1 if self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS else 2
            self.lds_total                      = self.lds_buffer_num * self.lds_single
        if self.lds_total > 32 * 1024:
            self.lds_buffer_num                 = 1
            self.lds_total                      = self.lds_buffer_num * self.lds_single
        # print(f"lds_a:{self.lds_a}, lds_b:{self.lds_b}, lds_a_np2:{self.lds_a_np2}, lds_b_np2:{self.lds_b_np2}, lds_single:{self.lds_single}, lds_total:{self.lds_total}")
        # TODO: LDS size check

        # some parameter not in modular_conv
        if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
            self.thread_tile_m                      = self.gemm_m_repeat * self.gemm_m_per_thread
            self.thread_tile_n                      = self.gemm_n_repeat * self.gemm_n_per_thread
            self.thread_sub_tile_m                  = self.gemm_m_per_thread
            self.thread_sub_tile_n                  = self.gemm_n_per_thread

        # number of loops at least needed for final coalescing store, dicided by LDS size
        # self.coalescing_store_groups            = (self.gemm_m_per_block * self.gemm_n_per_block) // \
        #         (self.lds_buffer_num * igemm_next_pow2(igemm_next_pow2(self.gemm_k_per_block * self.gemm_m_per_block) + igemm_next_pow2(self.gemm_k_per_block * self.gemm_n_per_block) ))
        if self.direction == "wrw" and (self.tensor_b_thread_lengths[3] == 1 or self.vector_store == 1) and self.gemm_k_global_split == 1 and self.precision == 'fp16':
            self.use_fp32_atomic_add_for_fp16_data = 1
        else:
            self.use_fp32_atomic_add_for_fp16_data = 0

        if (self.direction == "fwd" or self.direction == "bwd") and self.vector_store == 1 and self.gemm_k_global_split == 1 and self.precision == 'fp16':
            self.use_fp32_atomic_add_for_fp16_data = 1

        if self.gemm_k_global_split == 1 and self.precision == 'bf16':
            self.use_fp32_atomic_add_for_fp16_data = 1

        self.coalescing_store_groups = (self.gemm_m_per_block * self.gemm_n_per_block) // (self.lds_total // (amdgpu_precision_data_byte(self.precision) if self.use_fp32_atomic_add_for_fp16_data == 0 else 4))

        if self.coalescing_store_groups == 0:
            self.coalescing_store_groups = 1        # this means LDS size is already bigger than c matrix all pixel. just use one group is ok
        #if self.coalescing_store_groups < 2:
        #    self.coalescing_store_groups = 2
        shrinked_lds_buffer_num = self.lds_buffer_num
        if self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            # check on grouping
            xdlops_mapping = get_ctrl_xdlops_mapping_from_wave_tile(self.gemm_m_per_block, self.gemm_n_per_block, self.wave_tile_m, self.wave_tile_n, self.wave_tile_k, 
                    self.wave_repeat_m, self.wave_repeat_n, self.wave_step_m, self.wave_step_n, self.block_size // amdgpu_wave_size(tunable_dict['arch']), self.precision)
            length_in_m =  xdlops_mapping.wave_repeat_m * xdlops_mapping.wave_step_m * xdlops_mapping.lanegroup_m_per_wave() * xdlops_mapping.lanegroup_m_per_block() # no need xdlops_mapping.lanegroup_m_per_thread()
            if length_in_m % self.coalescing_store_groups != 0:
                # we still asume both value are power of 2
                assert self.coalescing_store_groups % length_in_m == 0
                shrink_in_co_group = self.coalescing_store_groups // length_in_m

                # TODO: this may affect occupancy!
                shrinked_lds_buffer_num = shrinked_lds_buffer_num * shrink_in_co_group
                self.lds_total = shrinked_lds_buffer_num * self.lds_single
                self.coalescing_store_groups = self.coalescing_store_groups // shrink_in_co_group
                assert length_in_m % self.coalescing_store_groups == 0

        if self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            if self.lds_total >= lds_a_pad + lds_b_pad:
                pass
            else:
                self.lds_total += (lds_a_pad - self.lds_a_np2 + lds_b_pad - self.lds_b_np2)

    def get_lds_pad(self):
        if self.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            return 0, 0
        if self.direction == 'wrw' and self.precision in ('fp16', 'bf16'):
            if self.gemm_k_per_block == 32 and self.gemm_m_per_block >= 128 and self.gemm_n_per_block >= 128 and self.tensor_b_thread_lengths[1] >= 4:
                return 4, 4
            else:
                return 0, 0
        if self.direction == 'bwd' and self.precision in ('fp16', 'bf16'):
            if self.gemm_k_per_block == 32 and self.gemm_m_per_block >= 128 and self.gemm_n_per_block >= 128 and self.tensor_b_thread_lengths[1] >= 8:
                if self.gemm_m_per_block == 128 and self.gemm_n_per_block == 128:
                    return 0, 0
                else:
                    return 0, 4
            else:
                return 0, 0
        else:
            return 0, 0

    def is_occupancy_decreased(self):
        is_decreased = False
        is_lds_decreased = False
        is_agpr_decreased = False
        is_vgpr_decreased = False
        if self.lds_single <= 16 * 1024 and self.lds_single > 8 * 1024:
            is_lds_decreased = True

        if self.num_agpr_accumulate_c < 128:
            is_agpr_decreased = True

        a_data_per_vgpr = 1 

        # for fwd and bwd pass, return true directly, because they do not use lds double buffer
        if self.direction == 'fwd':
            return True

        elif self.direction == "wrw":
            if self.precision == "fp32":
                return True
            elif self.tensor_a_thread_lengths[3] > 1:
                a_data_per_vgpr = 2
            else:
                a_data_per_vgpr = 1
        else:
            return True

        if self.num_global_load_a // a_data_per_vgpr <= 8:
            is_vgpr_decreased = True

        is_decreased = is_lds_decreased and is_agpr_decreased and is_vgpr_decreased
        return is_decreased

    def output(self):
        def to_miopen_prec(precision):
            if precision == 'fp32':
                return 'miopenFloat'
            if precision == 'fp16':
                return 'miopenHalf'
            if precision == 'bf16':
                return 'miopenBFloat16'
            else:
                assert False, "unkown data type"

        if False:
            brace_left='   {'
            brace_right='}'
            direction = "\"" + self.direction + "\""
            precision = "\"" + self.precision + "\""
            out_str = (f"\t\t{'{':2}{direction}{',':2}{precision},{self.nxb:4},{self.nxe:4},{self.gemm_m_per_block:4},{self.gemm_n_per_block:4},{self.gemm_k_per_block:4},")
            out_str += (f"{self.wave_tile_m:4},{self.wave_tile_n:4},{self.wave_tile_k:4},{self.wave_step_m:4},{self.wave_step_n:4},{self.wave_repeat_m:4},{self.wave_repeat_n:4},")
            out_str += (f"{brace_left}{self.tensor_a_thread_lengths[0]},{self.tensor_a_thread_lengths[1]:4},{self.tensor_a_thread_lengths[2]:4},{self.tensor_a_thread_lengths[3]:4}{brace_right},")
            out_str += (f"{brace_left}{self.tensor_a_cluster_lengths[0]},{self.tensor_a_cluster_lengths[1]:4},{self.tensor_a_cluster_lengths[2]:4},{self.tensor_a_cluster_lengths[3]:4}{brace_right},")
            out_str += (f"{brace_left}{self.tensor_b_thread_lengths[0]},{self.tensor_b_thread_lengths[1]:4},{self.tensor_b_thread_lengths[2]:4},{self.tensor_b_thread_lengths[3]:4}{brace_right},")
            out_str += (f"{brace_left}{self.tensor_b_cluster_lengths[0]},{self.tensor_b_cluster_lengths[1]:4},{self.tensor_b_cluster_lengths[2]:4},{self.tensor_b_cluster_lengths[3]:4}{brace_right},")
            out_str += (f"{self.gemm_k_global_split:4}{brace_right},")
        else:
            brace_left='{'
            brace_right='}'
            direction = "\"" + self.direction + "\""
            tensor_layout = "\"" + self.tensor_layout + "\""
            precision = to_miopen_prec(self.precision)
            out_str = (f"        {'{'}{direction}, {tensor_layout}, {precision}, {self.nxb:2},{self.nxe:2},{self.gemm_m_per_block:4},{self.gemm_n_per_block:4},{self.gemm_k_per_block:4},")
            out_str += (f"{self.wave_tile_m:3},{self.wave_tile_n:3},{self.wave_tile_k:3},{self.wave_step_m:2},{self.wave_step_n:2},{self.wave_repeat_m:2},{self.wave_repeat_n:2},")
            out_str += (f"{self.multihead:2},{self.vector_store:2},{self.gemm_k_global_split:2},{self.merge_e:2},{self.tensor_a_pass_through:2},")
            out_str += (f" {brace_left}{self.tensor_a_thread_lengths[0]:2},{self.tensor_a_thread_lengths[1]:2},{self.tensor_a_thread_lengths[2]:2},{self.tensor_a_thread_lengths[3]:2}{brace_right},")
            out_str += (f" {brace_left}{self.tensor_a_cluster_lengths[0]:3},{self.tensor_a_cluster_lengths[1]:3},{self.tensor_a_cluster_lengths[2]:3},{self.tensor_a_cluster_lengths[3]:3}{brace_right},")
            out_str += (f" {brace_left}{self.tensor_b_thread_lengths[0]:2},{self.tensor_b_thread_lengths[1]:2},{self.tensor_b_thread_lengths[2]:2},{self.tensor_b_thread_lengths[3]:2}{brace_right},")
            out_str += (f" {brace_left}{self.tensor_b_cluster_lengths[0]:3},{self.tensor_b_cluster_lengths[1]:3},{self.tensor_b_cluster_lengths[2]:3},{self.tensor_b_cluster_lengths[3]:3}{brace_right}")
            out_str += f"{brace_right},"
            
        return out_str

    def to_dict(self):
        tunable_dict = {}
        tunable_dict['tensor_layout']                   = self.tensor_layout
        tunable_dict['fma_type']                        = self.fma_type
        tunable_dict['gemm_m_per_block']                = self.gemm_m_per_block
        tunable_dict['gemm_n_per_block']                = self.gemm_n_per_block
        tunable_dict['gemm_k_per_block']                = self.gemm_k_per_block
        if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
            tunable_dict['gemm_m_per_thread']           = self.gemm_m_per_thread
            tunable_dict['gemm_m_level0_cluster']       = self.gemm_m_level0_cluster
            tunable_dict['gemm_m_level1_cluster']       = self.gemm_m_level1_cluster
            tunable_dict['gemm_n_per_thread']           = self.gemm_n_per_thread
            tunable_dict['gemm_n_level0_cluster']       = self.gemm_n_level0_cluster
            tunable_dict['gemm_n_level1_cluster']       = self.gemm_n_level1_cluster
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            tunable_dict['wave_tile_m']                 = self.wave_tile_m
            tunable_dict['wave_step_m']                 = self.wave_step_m
            tunable_dict['wave_repeat_m']               = self.wave_repeat_m
            tunable_dict['wave_tile_n']                 = self.wave_tile_n
            tunable_dict['wave_step_n']                 = self.wave_step_n
            tunable_dict['wave_repeat_n']               = self.wave_repeat_n
            tunable_dict['wave_tile_k']                 = self.wave_tile_k
        else:
            assert False
        tunable_dict['tensor_a_pass_through']           = self.tensor_a_pass_through
        tunable_dict['tensor_b_pass_through']           = self.tensor_b_pass_through
        tunable_dict['tensor_a_thread_lengths']         = self.tensor_a_thread_lengths
        tunable_dict['tensor_a_cluster_lengths']        = self.tensor_a_cluster_lengths
        tunable_dict['tensor_b_thread_lengths']         = self.tensor_b_thread_lengths
        tunable_dict['tensor_b_cluster_lengths']        = self.tensor_b_cluster_lengths
        tunable_dict['direction']                       = self.direction
        tunable_dict['precision']                       = self.precision
        tunable_dict['nxb']                             = self.nxb
        tunable_dict['nxe']                             = self.nxe
        tunable_dict['source_access_order']             = self.source_access_order
        tunable_dict['gemm_k_global_split']             = self.gemm_k_global_split
        tunable_dict['merge_e']                         = self.merge_e
        tunable_dict['multihead']                       = self.multihead
        tunable_dict['allow_lds_reorder']               = self.allow_lds_reorder
        tunable_dict['precache_soffset']                = self.precache_soffset

        tunable_dict['local_prefetch_num']              = self.local_prefetch_num
        tunable_dict['global_prefetch_a_num']           = self.global_prefetch_a_num
        tunable_dict['global_prefetch_b_num']           = self.global_prefetch_b_num
        tunable_dict['fma_interleave']                  = self.fma_interleave

        tunable_dict['gemm_m_unmerge_cluster']          = self.gemm_m_unmerge_cluster
        tunable_dict['gemm_n_unmerge_cluster']          = self.gemm_n_unmerge_cluster
        tunable_dict['gemm_k_unmerge_cluster']          = self.gemm_k_unmerge_cluster
        tunable_dict['vector_store']                    = self.vector_store

        return tunable_dict

    def serialize(self, **options):
        def get_dict_with_default(some_dict, key, default_value):
            if key in some_dict:
                return some_dict[key]
            return default_value

        section_name = get_dict_with_default(options, 'section_name', False)
        line_start = get_dict_with_default(options, 'line_start', '; ')
        new_line = get_dict_with_default(options, 'new_line', '\n')
        equal = get_dict_with_default(options, 'equal', ':')
        extra_info = get_dict_with_default(options, 'extra_info', True)
        sstr = ''

        if section_name:
            sstr += \
                line_start + f'[igemm_{self.direction}_gtc]' + new_line

        sstr += line_start + 'tensor_layout              {} {}'.format(equal, '\'' + self.tensor_layout + '\'') + new_line + \
                line_start + 'gemm_m_per_block           {} {}'.format(equal, self.gemm_m_per_block) + new_line + \
                line_start + 'gemm_n_per_block           {} {}'.format(equal, self.gemm_n_per_block) + new_line + \
                line_start + 'gemm_k_per_block           {} {}'.format(equal, self.gemm_k_per_block) + new_line
        if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
            sstr += \
                line_start + 'gemm_m_per_thread          {} {}'.format(equal, self.gemm_m_per_thread) + new_line + \
                line_start + 'gemm_m_level0_cluster      {} {}'.format(equal, self.gemm_m_level0_cluster) + new_line + \
                line_start + 'gemm_m_level1_cluster      {} {}'.format(equal, self.gemm_m_level1_cluster) + new_line + \
                line_start + 'gemm_n_per_thread          {} {}'.format(equal, self.gemm_n_per_thread) + new_line + \
                line_start + 'gemm_n_level0_cluster      {} {}'.format(equal, self.gemm_n_level0_cluster) + new_line + \
                line_start + 'gemm_n_level1_cluster      {} {}'.format(equal, self.gemm_n_level1_cluster) + new_line
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            sstr += \
                line_start + 'wave_tile_m                {} {}'.format(equal, self.wave_tile_m) + new_line + \
                line_start + 'wave_step_m                {} {}'.format(equal, self.wave_step_m) + new_line + \
                line_start + 'wave_repeat_m              {} {}'.format(equal, self.wave_repeat_m) + new_line + \
                line_start + 'wave_tile_n                {} {}'.format(equal, self.wave_tile_n) + new_line + \
                line_start + 'wave_step_n                {} {}'.format(equal, self.wave_step_n) + new_line + \
                line_start + 'wave_repeat_n              {} {}'.format(equal, self.wave_repeat_n) + new_line + \
                line_start + 'wave_tile_k                {} {}'.format(equal, self.wave_tile_k) + new_line
        if self.tensor_a_pass_through:
            sstr += \
                line_start + 'tensor_a_pass_through      {} {}'.format(equal, self.tensor_a_pass_through) + new_line
        if self.tensor_b_pass_through:
            sstr += \
                line_start + 'tensor_b_pass_through      {} {}'.format(equal, self.tensor_b_pass_through) + new_line
        sstr += \
                line_start + 'tensor_a_thread_lengths    {} {}'.format(equal, self.tensor_a_thread_lengths) + new_line + \
                line_start + 'tensor_a_cluster_lengths   {} {}'.format(equal, self.tensor_a_cluster_lengths) + new_line + \
                line_start + 'tensor_b_thread_lengths    {} {}'.format(equal, self.tensor_b_thread_lengths) + new_line + \
                line_start + 'tensor_b_cluster_lengths   {} {}'.format(equal, self.tensor_b_cluster_lengths) + new_line + \
                line_start + 'direction                  {} {}'.format(equal, '\'' + self.direction + '\'') + new_line + \
                line_start + 'precision                  {} {}'.format(equal, '\'' + self.precision + '\'') + new_line + \
                line_start + 'nxb                        {} {}'.format(equal, self.nxb) + new_line + \
                line_start + 'nxe                        {} {}'.format(equal, self.nxe) + new_line
        if self.gemm_k_global_split:
            sstr += \
                line_start + 'gemm_k_global_split        {} {}'.format(equal, self.gemm_k_global_split) + new_line
        if self.merge_e:
            sstr += \
                line_start + 'merge_e                    {} {}'.format(equal, self.merge_e) + new_line
        if self.vector_store:
            sstr += \
                line_start + 'vector_store               {} {}'.format(equal, self.vector_store) + new_line
        if extra_info:
            sstr += \
                line_start + new_line + \
                line_start + 'block_size                 {} {}'.format(equal, self.block_size) + new_line
            if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
                sstr += \
                line_start + 'thread_tile                {} {}x{}'.format(equal, self.thread_tile_m, self.thread_tile_n) + new_line
            sstr += \
                line_start + 'lds_total                  {} {}'.format(equal, self.lds_total) + new_line + \
                line_start + 'lds_buffer_num             {} {}'.format(equal, self.lds_buffer_num) + new_line + \
                line_start
        return sstr

    def serialize_as_section(self):
        return self.serialize(section_name=True, line_start='', equal='=', extra_info=False)

def igemm_gtc_encode_kernel_base_name(tunable, arch):
    assert type(tunable) is igemm_gtc_tunable_parameter_t

    kernel_name = f"igemm_{tunable.direction}_"
    if type(arch) is not str:
        arch_str = amdgpu_arch_to_string(arch)
    else:
        arch_str = arch

    if tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC:
        kernel_name += 'gtcm_'                                  # generic tensor contraction with mac
    elif tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS:
        kernel_name += 'gtc_'                                   # generic tensor contraction with dlops
    elif tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
        if arch_str == 'gfx908':
            kernel_name += 'gtcx_'                              # generic tensor contraction with xdlops
        elif arch_str == 'gfx90a':
            kernel_name += 'gtcx2_'
        else:
            assert False

    kernel_name += f"{tunable.tensor_layout}_{tunable.precision}"

    return kernel_name

def igemm_gtc_encode_kernel_name(tunable, arch):
    def lengths_str(lengths):
        assert type(lengths) is list
        return "x".join( [f"{x}" for x in lengths] )

    assert type(tunable) is igemm_gtc_tunable_parameter_t

    kernel_name = igemm_gtc_encode_kernel_base_name(tunable, arch) + '_'

    kernel_name += f"bx{tunable.nxb}_ex{tunable.nxe}_"
    if IGEMM_GTC_FEAT_SOURCE_ACCESS_ENCODING_KERNEL_NAME:
        kernel_name += f"sa{tunable.source_access_order}_"
    kernel_name += f"bt{tunable.gemm_m_per_block}x{tunable.gemm_n_per_block}x{tunable.gemm_k_per_block}_"
    if tunable.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
        kernel_name +=   f"tt{tunable.thread_tile_m}x{tunable.thread_tile_n}_" +\
                         f"gm{tunable.gemm_m_repeat}x{tunable.gemm_m_level0_cluster}x{tunable.gemm_m_level1_cluster}_" +\
                         f"gn{tunable.gemm_n_repeat}x{tunable.gemm_n_level0_cluster}x{tunable.gemm_n_level1_cluster}_"
    elif tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
        kernel_name +=   f'wt{tunable.wave_tile_m}x{tunable.wave_tile_n}x{tunable.wave_tile_k}_' +\
                         f'ws{tunable.wave_step_m}x{tunable.wave_step_n}_' +\
                         f'wr{tunable.wave_repeat_m}x{tunable.wave_repeat_n}_'

    kernel_name +=       "ta" + lengths_str(tunable.tensor_a_thread_lengths) + "_" + lengths_str(tunable.tensor_a_cluster_lengths) + "_" +\
                         "tb" + lengths_str(tunable.tensor_b_thread_lengths) + "_" + lengths_str(tunable.tensor_b_cluster_lengths)

    if tunable.tensor_a_pass_through:
        kernel_name += "_pta"

    if tunable.tensor_b_pass_through:
        kernel_name += "_ptb"

    if tunable.gemm_m_unmerge_cluster:
        kernel_name += "_mc"

    if tunable.gemm_n_unmerge_cluster:
        kernel_name += "_nc"

    if tunable.gemm_k_unmerge_cluster:
        kernel_name += "_kc"

    if tunable.multihead:
        kernel_name += "_mh"

    if tunable.merge_e:
        kernel_name += "_me"

    if tunable.vector_store:
        kernel_name += f"_vs{tunable.vector_store}"

    if tunable.gemm_k_global_split:
        kernel_name += "_gkgs"

    return kernel_name


class igemm_kernel_detail_base_t(object):
    # gemm problem details
    def __init__(self):
        self.vgpr_total = 0
        self.sgpr_total = 0

        self.thread_m = 0
        self.thread_n = 0
        self.block_m = 0
        self.block_n = 0
        self.unroll_k = 0
        self.block_size = 0

        self.vgpr_c_accumulate = 0
        self.vgpr_a_accumulate = 0
        self.vgpr_b_accumulate = 0
        self.vgpr_a_global_fetch = 0
        self.vgpr_b_global_fetch = 0
        # if local fetch to accumulate directly, no extra local fetch gpr is needed
        self.vgpr_a_local_fetch = 0
        self.vgpr_b_local_fetch = 0
        self.vgpr_other = 0

        self.lds_total = 0
        self.lds_buffers = 1        # single buffer, double buffer...
        self.occupancy = 1

        
        # now hard code v4r1 tiling stratagy. in the future, this should be more flex
        # wei->tensor_a, input->tensor_b
        # wei: e_k, e is unroll_k

        self.msg = ''

    def getattrs(self):
        attrs = [i for i in dir(self) if not callable(getattr(self,i)) and not i.startswith("__") and not i == 'msg']
        return attrs

    def key(self):
        attrs = self.getattrs()
        return '-'.join( [ str(getattr(self, attr)) for attr in attrs] ) 

    def serialize(self):
        return  'thread_mxn          : {}x{}'.format(self.thread_m, self.thread_n) + '\n' + \
                'block_mxn           : {}x{}'.format(self.block_m, self.block_n) + '\n' + \
                'unroll_k            : {}'.format(self.unroll_k) + '\n' + \
                'block_size          : {}'.format(self.block_size) + '\n' + \
                'vgpr_total          : {}'.format(self.vgpr_total) + '\n' + \
                'sgpr_total          : {}'.format(self.sgpr_total) + '\n' + \
                'lds_total           : {}'.format(self.lds_total) + '\n' + \
                'lds_buffers         : {}'.format(self.lds_buffers) + '\n' + \
                'occupancy           : {}'.format(self.occupancy) + '\n' + \
                'vgpr_c_accumulate   : {}'.format(self.vgpr_c_accumulate) + '\n' + \
                'vgpr_a_accumulate   : {}'.format(self.vgpr_a_accumulate) + '\n' + \
                'vgpr_b_accumulate   : {}'.format(self.vgpr_b_accumulate) + '\n' + \
                'vgpr_a_global_fetch : {}'.format(self.vgpr_a_global_fetch) + '\n' + \
                'vgpr_b_global_fetch : {}'.format(self.vgpr_b_global_fetch) + '\n' + \
                'vgpr_a_local_fetch  : {}'.format(self.vgpr_a_local_fetch) + '\n' + \
                'vgpr_b_local_fetch  : {}'.format(self.vgpr_b_local_fetch) + '\n' + \
                'vgpr_other          : {}'.format(self.vgpr_other) + '\n'

class igemm_thread_cluster_index_dispatcher_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    
    def __call__(self, v_x, v_tid_shifter, c_x, t_x, is_last = False):
        with self._deferred_context():
            if c_x == 1:
                self._emit(f"v_mov_b32 v[{v_x}], 0")
            else:
                self._emit(f"v_and_b32 v[{v_x}], {c_x - 1}, v[{v_tid_shifter}]")
                if t_x != 1:
                    self._emit(f"v_lshlrev_b32 v[{v_x}], {igemm_log2(t_x)}, v[{v_x}]")
                if not is_last:
                    self._emit(f"v_lshrrev_b32 v[{v_tid_shifter}], {igemm_log2(c_x)}, v[{v_tid_shifter}]")
        return self._get_deferred()

class igemm_thread_cluster_index_accumulator_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)

    def __call__(self, v_dst, v_x0, v_x1, c_x0, c_x1, n_x0, n_x1):
        assert not (c_x0 == 1 and c_x1 == 1)
        with self._deferred_context():
            if c_x0 != 1 and c_x1 != 1:
                self._emit(f"v_lshl_or_b32 v[{v_dst}], v[{v_x0}], {igemm_log2(n_x1)}, v[{v_x1}]")
            elif c_x0 == 1 and c_x1 != 1:
                self._emit(f"v_mov_b32 v[{v_dst}], v[{v_x1}]")
            elif c_x0 != 1 and c_x1 == 1:
                if n_x1 == 1:
                    self._emit(f"v_mov_b32 v[{v_dst}], v[{v_x0}]")
                else:
                    self._emit(f"v_lshlrev_b32 v[{v_dst}], {igemm_log2(n_x1)}, v[{v_x0}]")
        return self._get_deferred()
