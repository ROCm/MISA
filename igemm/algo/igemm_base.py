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
from __future__ import print_function
import sys
import math
from ..codegen import *
from .utility import *
from .conv import *


IGEMM_GTC_FEAT_ALLOW_LDS_REORDER = 0
IGEMM_GTC_FEAT_PRECACHE_SOFFSET = 1


# IGEMM_GTC_TENSOR_LAYOUT_NCHW = ((1 << 4) | 0)
# IGEMM_GTC_TENSOR_LAYOUT_NHWC = ((1 << 4) | 1)
# IGEMM_GTC_TENSOR_LAYOUT_CNHW = ((1 << 4) | 2)


IGEMM_GTC_TUNABLE_TYPE_THREAD_WISE_GEMM = ((1<<16)|0)
IGEMM_GTC_TUNABLE_TYPE_WAVE_WISE_GEMM   = ((1<<16)|1)

IGEMM_GTC_TUNABLE_FMA_TYPE_MAC               = ((1<<16)|4)
IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS             = ((1<<16)|5)
IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS            = ((1<<16)|6)

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


def get_igemm_gtc_tunable_type(tunable_dict):
    assert type(tunable_dict) is dict
    if 'gemm_m_per_thread' in tunable_dict and 'gemm_n_per_thread' in tunable_dict:
        return IGEMM_GTC_TUNABLE_TYPE_THREAD_WISE_GEMM
    if 'gemm_m_per_wave' in tunable_dict and 'gemm_n_per_wave' in tunable_dict:
        return IGEMM_GTC_TUNABLE_TYPE_WAVE_WISE_GEMM
    assert False

class igemm_gtc_tunable_parameter_t(object):
    '''
    generic tensor contraction
    '''
    def __init__(self, tunable_dict):
        self.tensor_layout                      = utility_dict_with_default_t(tunable_dict)('tensor_layout', 'nchw')
        self.fma_type                           = utility_dict_with_default_t(tunable_dict)('fma_type', IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS)
        self.gemm_m_per_block                   = tunable_dict['gemm_m_per_block']
        self.gemm_n_per_block                   = tunable_dict['gemm_n_per_block']
        self.gemm_k_per_block                   = tunable_dict['gemm_k_per_block']
        self.tunable_type                       = get_igemm_gtc_tunable_type(tunable_dict)
        if self.tunable_type == IGEMM_GTC_TUNABLE_TYPE_THREAD_WISE_GEMM
            self.gemm_m_per_thread              = tunable_dict['gemm_m_per_thread']
            self.gemm_m_level0_cluster          = tunable_dict['gemm_m_level0_cluster']
            self.gemm_m_level1_cluster          = tunable_dict['gemm_m_level1_cluster']
            self.gemm_n_per_thread              = tunable_dict['gemm_n_per_thread']
            self.gemm_n_level0_cluster          = tunable_dict['gemm_n_level0_cluster']
            self.gemm_n_level1_cluster          = tunable_dict['gemm_n_level1_cluster']
        elif self.tunable_type == IGEMM_GTC_TUNABLE_TYPE_WAVE_WISE_GEMM:
            self.gemm_m_per_wave                = tunable_dict['gemm_m_per_wave']
            self.gemm_m_wave_step               = tunable_dict['gemm_m_wave_step']
            self.gemm_m_wave_repeat             = tunable_dict['gemm_m_wave_repeat']
            self.gemm_n_per_wave                = tunable_dict['gemm_n_per_wave']
            self.gemm_n_wave_step               = tunable_dict['gemm_n_wave_step']
            self.gemm_n_wave_repeat             = tunable_dict['gemm_n_wave_repeat']
        else:
            assert False

        self.tensor_a_thread_lengths            = tunable_dict['tensor_a_thread_lengths']     # list!
        self.tensor_a_cluster_lengths           = tunable_dict['tensor_a_cluster_lengths']    # list!
        self.tensor_b_thread_lengths            = tunable_dict['tensor_b_thread_lengths']     # list!
        self.tensor_b_cluster_lengths           = tunable_dict['tensor_b_cluster_lengths']    # list!
        self.direction                          = tunable_dict['direction']
        self.precision                          = tunable_dict['precision']
        self.nxb                                = tunable_dict['nxb']           # multiplier of b
        self.nxe                                = tunable_dict['nxe']           # muptiplier of e. here if 0, means x=y=1
        self.multihead                          = utility_dict_with_default_t(tunable_dict)('multihead', 0)
        self.allow_lds_reorder                  = utility_dict_with_default_t(tunable_dict)('allow_lds_reorder', IGEMM_GTC_FEAT_ALLOW_LDS_REORDER)
        self.precache_soffset                   = utility_dict_with_default_t(tunable_dict)('precache_soffset', IGEMM_GTC_FEAT_PRECACHE_SOFFSET)

        self.gemm_m_unmerge_cluster             = utility_dict_with_default_t(tunable_dict)('gemm_m_unmerge_cluster', 0)
        self.gemm_n_unmerge_cluster             = utility_dict_with_default_t(tunable_dict)('gemm_n_unmerge_cluster', 0)
        self.gemm_k_unmerge_cluster             = utility_dict_with_default_t(tunable_dict)('gemm_k_unmerge_cluster', 0)     # maybe no need support for 1
        #  x -(unmerge)-> x0*x1, if set to 1, means cluster first iterate all x1
        # hence stride of x0 should not be x1, but be total number of x divide by x0

        assert type(self.tensor_a_thread_lengths) is list and type(self.tensor_a_cluster_lengths) is list
        assert type(self.tensor_b_thread_lengths) is list and type(self.tensor_b_cluster_lengths) is list
        # assert type(self.opt_1x1) is bool
        assert self.direction in ('fwd', 'bwd', 'wrw')
        assert self.precision in ('fp32', 'fp16', 'bf16')
        assert self.nxb in (1,4,16,64,256)
        assert self.nxe in (0,1)

        # TODO: better specify
        if self.tunable_type == IGEMM_GTC_TUNABLE_TYPE_THREAD_WISE_GEMM:
            self.block_size                     = self.gemm_m_level0_cluster * self.gemm_n_level0_cluster * self.gemm_m_level1_cluster * self.gemm_n_level1_cluster

        elif self.tunable_type == IGEMM_GTC_TUNABLE_TYPE_WAVE_WISE_GEMM:
            assert self.gemm_m_per_block % (self.gemm_m_per_wave * self.gemm_m_wave_step * self.gemm_m_wave_repeat)
            assert self.gemm_n_per_block % (self.gemm_n_per_wave * self.gemm_n_wave_step * self.gemm_n_wave_repeat)
            waves_per_m = self.gemm_m_per_block // (self.gemm_m_per_wave * self.gemm_m_wave_step * self.gemm_m_wave_repeat)
            waves_per_n = self.gemm_n_per_block // (self.gemm_n_per_wave * self.gemm_n_wave_step * self.gemm_n_wave_repeat)
            self.block_size                     = waves_per_m * waves_per_n * AMDGPU_WAVE_SIZE

        assert self.block_size == igemm_flatten_list_product(self.tensor_a_cluster_lengths)
        assert self.block_size == igemm_flatten_list_product(self.tensor_b_cluster_lengths)

        def _unmerge_x1_from_e(unroll_k, nxe):
            if nxe == 0:
                return unroll_k # not used, 1x1 special
            if unroll_k % nxe == 0:
                return unroll_k // nxe
            return unroll_k     # not used

        if self.direction == 'fwd':
            assert self.gemm_n_per_block % self.nxb == 0
            self.unmerge_sub_n = self.gemm_n_per_block // self.nxb
            self.unmerge_sub_k = 1                             # not used
            self.unmerge_sub_c = _unmerge_x1_from_e(self.gemm_k_per_block, self.nxe)
        elif self.direction == 'bwd':
            assert self.gemm_n_per_block % self.nxb == 0
            self.unmerge_sub_n = self.gemm_n_per_block // self.nxb
            self.unmerge_sub_k = _unmerge_x1_from_e(self.gemm_k_per_block, self.nxe)
            self.unmerge_sub_c = 1                             # not used
        else:
            # TODO: wrw maybe different
            self.unmerge_sub_n = 1
            self.unmerge_sub_k = 1
            self.unmerge_sub_c = 1

        # vector global/lds implicit here
        if self.tunable_type == IGEMM_GTC_TUNABLE_TYPE_THREAD_WISE_GEMM:
            self.gemm_m_repeat                  = self.gemm_m_per_block // (self.gemm_m_per_thread * self.gemm_m_level0_cluster * self.gemm_m_level1_cluster)
            self.gemm_n_repeat                  = self.gemm_n_per_block // (self.gemm_n_per_thread * self.gemm_n_level0_cluster * self.gemm_n_level1_cluster)
            # register for a,b,c buffer
            self.num_vgpr_accumulate_c              = (self.gemm_m_repeat*self.gemm_m_per_thread*self.gemm_n_repeat*self.gemm_n_per_thread)
            self.num_vgpr_accumulate_a              = (self.gemm_m_repeat*self.gemm_m_per_thread)
            self.num_vgpr_accumulate_b              = (self.gemm_n_repeat*self.gemm_n_per_thread)
        
        elif self.tunable_type == IGEMM_GTC_TUNABLE_TYPE_WAVE_WISE_GEMM:
            # register for a,b,c buffer
            # self.num_vgpr_accumulate_c              = 0
            self.num_vgpr_accumulate_a              = self.gemm_m_wave_step * self.gemm_m_wave_repeat
            self.num_vgpr_accumulate_b              = self.gemm_n_wave_step * self.gemm_n_wave_repeat

        self.num_vgpr_global_load_a             = igemm_flatten_list_product(self.tensor_a_thread_lengths)
        self.num_vgpr_global_load_b             = igemm_flatten_list_product(self.tensor_b_thread_lengths)

        assert self.num_vgpr_global_load_a * self.block_size == self.gemm_m_per_block * self.gemm_k_per_block
        assert self.num_vgpr_global_load_b * self.block_size == self.gemm_n_per_block * self.gemm_k_per_block

        # LDS size
        self.lds_a                              = amdgpu_precision_data_byte(self.precision) * self.gemm_k_per_block * self.gemm_m_per_block
        self.lds_b                              = amdgpu_precision_data_byte(self.precision) * self.gemm_k_per_block * self.gemm_n_per_block
        self.lds_a_np2                          = igemm_next_pow2( self.lds_a)
        self.lds_b_np2                          = igemm_next_pow2( self.lds_b)
        self.lds_single                         = igemm_next_pow2( self.lds_a_np2 + self.lds_b_np2)
        self.lds_total                          = (2*self.lds_single)
        # print(f"lds_a:{self.lds_a}, lds_b:{self.lds_b}, lds_a_np2:{self.lds_a_np2}, lds_b_np2:{self.lds_b_np2}, lds_single:{self.lds_single}, lds_total:{self.lds_total}")
        # TODO: LDS size check

        # some parameter not in modular_conv
        if self.tunable_type == IGEMM_GTC_TUNABLE_TYPE_THREAD_WISE_GEMM:
            self.thread_tile_m                      = self.gemm_m_repeat * self.gemm_m_per_thread
            self.thread_tile_n                      = self.gemm_n_repeat * self.gemm_n_per_thread
            self.thread_sub_tile_m                  = self.gemm_m_per_thread
            self.thread_sub_tile_n                  = self.gemm_n_per_thread

        # number of loops at least needed for final coalescing store, dicided by LDS size
        self.coalescing_store_groups            = (self.gemm_m_per_block * self.gemm_n_per_block) // \
                (2 * igemm_next_pow2(igemm_next_pow2(self.gemm_k_per_block * self.gemm_m_per_block) + igemm_next_pow2(self.gemm_k_per_block * self.gemm_n_per_block) ))
        if self.coalescing_store_groups < 2:
            self.coalescing_store_groups = 2

    def to_dict(self):
        tunable_dict = {}
        tunable_dict['tensor_layout']                   = self.tensor_layout
        tunable_dict['fma_type']                        = self.fma_type
        tunable_dict['gemm_m_per_block']                = self.gemm_m_per_block
        tunable_dict['gemm_n_per_block']                = self.gemm_n_per_block
        tunable_dict['gemm_k_per_block']                = self.gemm_k_per_block
        if self.tunable_type == IGEMM_GTC_TUNABLE_TYPE_THREAD_WISE_GEMM:
            tunable_dict['gemm_m_per_thread']           = self.gemm_m_per_thread
            tunable_dict['gemm_m_level0_cluster']       = self.gemm_m_level0_cluster
            tunable_dict['gemm_m_level1_cluster']       = self.gemm_m_level1_cluster
            tunable_dict['gemm_n_per_thread']           = self.gemm_n_per_thread
            tunable_dict['gemm_n_level0_cluster']       = self.gemm_n_level0_cluster
            tunable_dict['gemm_n_level1_cluster']       = self.gemm_n_level1_cluster
        elif self.tunable_type == IGEMM_GTC_TUNABLE_TYPE_WAVE_WISE_GEMM:
            tunable_dict['gemm_m_per_wave']             = self.gemm_m_per_wave
            tunable_dict['gemm_m_wave_step']            = self.gemm_m_wave_step
            tunable_dict['gemm_m_wave_repeat']          = self.gemm_m_wave_repeat
            tunable_dict['gemm_n_per_wave']             = self.gemm_n_per_wave
            tunable_dict['gemm_n_wave_step']            = self.gemm_n_wave_step
            tunable_dict['gemm_n_wave_repeat']          = self.gemm_n_wave_repeat
        else:
            assert False
        tunable_dict['tensor_a_thread_lengths']         = self.tensor_a_thread_lengths
        tunable_dict['tensor_a_cluster_lengths']        = self.tensor_a_cluster_lengths
        tunable_dict['tensor_b_thread_lengths']         = self.tensor_b_thread_lengths
        tunable_dict['tensor_b_cluster_lengths']        = self.tensor_b_cluster_lengths
        tunable_dict['direction']                       = self.direction
        tunable_dict['precision']                       = self.precision
        tunable_dict['nxb']                             = self.nxb
        tunable_dict['nxe']                             = self.nxe

        tunable_dict['multihead']                       = self.multihead
        tunable_dict['allow_lds_reorder']               = self.allow_lds_reorder
        tunable_dict['precache_soffset']                = self.precache_soffset

        tunable_dict['gemm_m_unmerge_cluster']          = self.gemm_m_unmerge_cluster
        tunable_dict['gemm_n_unmerge_cluster']          = self.gemm_n_unmerge_cluster
        tunable_dict['gemm_k_unmerge_cluster']          = self.gemm_k_unmerge_cluster

        return tunable_dict

    def serialize(self, line_starter = '; '):
        sstr =  line_starter + 'tensor_layout              : {}'.format(self.tensor_layout) + '\n' + \
                line_starter + 'gemm_m_per_block           : {}'.format(self.gemm_m_per_block) + '\n' + \
                line_starter + 'gemm_n_per_block           : {}'.format(self.gemm_n_per_block) + '\n' + \
                line_starter + 'gemm_k_per_block           : {}'.format(self.gemm_k_per_block) + '\n'
        if self.tunable_type == IGEMM_GTC_TUNABLE_TYPE_THREAD_WISE_GEMM:
            sstr += \
                line_starter + 'gemm_m_per_thread          : {}'.format(self.gemm_m_per_thread) + '\n' + \
                line_starter + 'gemm_m_level0_cluster      : {}'.format(self.gemm_m_level0_cluster) + '\n' + \
                line_starter + 'gemm_m_level1_cluster      : {}'.format(self.gemm_m_level1_cluster) + '\n' + \
                line_starter + 'gemm_n_per_thread          : {}'.format(self.gemm_n_per_thread) + '\n' + \
                line_starter + 'gemm_n_level0_cluster      : {}'.format(self.gemm_n_level0_cluster) + '\n' + \
                line_starter + 'gemm_n_level1_cluster      : {}'.format(self.gemm_n_level1_cluster) + '\n'
        elif self.tunable_type == IGEMM_GTC_TUNABLE_TYPE_WAVE_WISE_GEMM:
            sstr += \
                line_starter + 'gemm_m_per_wave            : {}'.format(self.gemm_m_per_wave) + '\n' + \
                line_starter + 'gemm_m_wave_step           : {}'.format(self.gemm_m_wave_step) + '\n' + \
                line_starter + 'gemm_m_wave_repeat         : {}'.format(self.gemm_m_wave_repeat) + '\n' + \
                line_starter + 'gemm_n_per_wave            : {}'.format(self.gemm_n_per_wave) + '\n' + \
                line_starter + 'gemm_n_wave_step           : {}'.format(self.gemm_n_wave_step) + '\n' + \
                line_starter + 'gemm_n_wave_repeat         : {}'.format(self.gemm_n_wave_repeat) + '\n'
        sstr += \
                line_starter + 'tensor_a_thread_lengths    : {}'.format(self.tensor_a_thread_lengths) + '\n' + \
                line_starter + 'tensor_a_cluster_lengths   : {}'.format(self.tensor_a_cluster_lengths) + '\n' + \
                line_starter + 'tensor_b_thread_lengths    : {}'.format(self.tensor_b_thread_lengths) + '\n' + \
                line_starter + 'tensor_b_cluster_lengths   : {}'.format(self.tensor_b_cluster_lengths) + '\n' + \
                line_starter + 'direction                  : {}'.format(self.direction) + '\n' + \
                line_starter + 'precision                  : {}'.format(self.precision) + '\n' + \
                line_starter + 'nxb                        : {}'.format(self.nxb) + '\n' + \
                line_starter + 'nxe                        : {}'.format(self.nxe) + '\n' + \
                line_starter + '\n' + \
                line_starter + 'block_size                 : {}'.format(self.block_size) + '\n' + \
                line_starter + 'thread_tile                : {}x{}'.format(self.thread_tile_m, self.thread_tile_n) + '\n' + \
                line_starter + 'lds_total                  : {}'.format(self.lds_total) + '\n' + \
                line_starter
        return sstr

def igemm_gtc_encode_kernel_name(tunable):
    def lengths_str(lengths):
        assert type(lengths) is list
        return "x".join( [f"{x}" for x in lengths] )

    assert type(tunable) is igemm_gtc_tunable_parameter_t

    kernel_name = f"igemm_{tunable.direction}_"

    if tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC:
        kernel_name += 'gtcm_'                                  # generic tensor contraction with mac
    elif tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS:
        kernel_name += 'gtc_'                                   # generic tensor contraction with dlops
    elif tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
        kernel_name += 'gtcx_'                                  # generic tensor contraction with xdlops

    kernel_name += f"{tunable.tensor_layout}_{tunable.precision}_bx{tunable.nxb}_ex{tunable.nxe}_"
    kernel_name += f"bt{tunable.gemm_m_per_block}x{tunable.gemm_n_per_block}x{tunable.gemm_k_per_block}_"
    if tunebla.tunable_type == IGEMM_GTC_TUNABLE_TYPE_THREAD_WISE_GEMM:
        kernel_name +=   f"tt{tunable.thread_tile_m}x{tunable.thread_tile_n}_" +\
                         f"gm{tunable.gemm_m_repeat}x{tunable.gemm_m_level0_cluster}x{tunable.gemm_m_level1_cluster}_" +\
                         f"gn{tunable.gemm_n_repeat}x{tunable.gemm_n_level0_cluster}x{tunable.gemm_n_level1_cluster}_"
    elif tunebla.tunable_type == IGEMM_GTC_TUNABLE_TYPE_WAVE_WISE_GEMM:
        kernel_name +=   f'wt{tunable.gemm_m_per_wave}x{tunable.gemm_n_per_wave}' +\
                         f'gm{tunable.gemm_m_wave_step}x{tunable.gemm_m_wave_repeat}' +\
                         f'gn{tunable.gemm_m_wave_repeat}x{tunable.gemm_n_wave_repeat}'

    kernel_name +=       "ta" + lengths_str(tunable.tensor_a_thread_lengths) + "_" + lengths_str(tunable.tensor_a_cluster_lengths) + "_" +\
                         "tb" + lengths_str(tunable.tensor_b_thread_lengths) + "_" + lengths_str(tunable.tensor_b_cluster_lengths)

    if tunable.gemm_m_unmerge_cluster:
        kernel_name += "_mc"

    if tunable.gemm_n_unmerge_cluster:
        kernel_name += "_nc"

    if tunable.gemm_k_unmerge_cluster:
        kernel_name += "_kc"

    if tunable.multihead:
        kernel_name += "_mh"

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
                self._emit(f"v_lshl_or_b32 v[{v_dst}], v[{v_x0}], {igemm_log2(n_c1)}, v[{v_x1}]")
            elif c_x0 == 1 and c_x1 != 1:
                self._emit(f"v_mov_b32 v[{v_dst}], v[{v_x1}]")
            elif c_x0 != 1 and c_x1 == 1:
                if n_c1 == 1:
                    self._emit(f"v_mov_b32 v[{v_dst}], v[{v_x0}]")
                else:
                    self._emit(f"v_lshlrev_b32 v[{v_dst}], {igemm_log2(n_c1)}, v[{v_x0}]")
        return self._get_deferred()
