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
import numpy as np
from .amdgpu import *

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
    return int(np.log2(v))

def igemm_get_epack_length(precision):
        # GetEPackLength
        epack = 1
        if precision == AMDGPU_PRECISION_FP16:
            # todo: xdlops check
            epack = 2
        elif precision == AMDGPU_PRECISION_BF16:
            epack = 2
        return epack

class igemm_tunable_parameter_t(object):
    def __init__(self, tunable_dict):
        self.b_per_block                         = tunable_dict['b_per_block']
        self.k_per_block                         = tunable_dict['k_per_block']
        self.e_per_block                         = tunable_dict['e_per_block']
        self.gemm_n_repeat                       = tunable_dict['gemm_n_repeat']
        self.gemm_m_per_thread_subc              = tunable_dict['gemm_m_per_thread_subc']
        self.gemm_n_per_thread_subc              = tunable_dict['gemm_n_per_thread_subc']
        self.gemm_m_level1_cluster               = tunable_dict['gemm_m_level1_cluster']
        self.gemm_n_level1_cluster               = tunable_dict['gemm_n_level1_cluster']
        self.gemm_m_level0_cluster               = tunable_dict['gemm_m_level0_cluster']
        self.gemm_n_level0_cluster               = tunable_dict['gemm_n_level0_cluster']
        self.in_block_copy_cluster_lengths_e     = tunable_dict['in_block_copy_cluster_lengths_e']
        self.in_block_copy_cluster_lengths_n1    = tunable_dict['in_block_copy_cluster_lengths_n1']
        self.in_block_copy_cluster_lengths_b     = tunable_dict['in_block_copy_cluster_lengths_b']
        self.in_block_copy_cluster_lengths_n2    = tunable_dict['in_block_copy_cluster_lengths_n2']
        self.wei_block_copy_cluster_lengths_e    = tunable_dict['wei_block_copy_cluster_lengths_e']
        self.wei_block_copy_cluster_lengths_k    = tunable_dict['wei_block_copy_cluster_lengths_k']
        self.name                                = tunable_dict['name']

        self.gemm_m_repeat = self.k_per_block // (self.gemm_m_per_thread_subc * self.gemm_m_level0_cluster * self.gemm_m_level1_cluster)

        self.in_block_copy_sub_lengths_e         = self.e_per_block // self.in_block_copy_cluster_lengths_e
        self.in_block_copy_sub_lengths_n1        = self.gemm_n_repeat // self.in_block_copy_cluster_lengths_n1
        self.in_block_copy_sub_lengths_b         = self.b_per_block // self.in_block_copy_cluster_lengths_b
        self.in_block_copy_sub_lengths_n2        = self.gemm_n_per_thread_subc // self.in_block_copy_cluster_lengths_n2
        self.wei_block_copy_sub_lengths_e        = self.e_per_block // self.wei_block_copy_cluster_lengths_e
        self.wei_block_copy_sub_lengths_k        = self.k_per_block // self.wei_block_copy_cluster_lengths_k

        self.block_size                          = self.gemm_m_level0_cluster * self.gemm_n_level0_cluster * self.gemm_m_level1_cluster * self.gemm_n_level1_cluster

        assert self.in_block_copy_sub_lengths_e == 1 and self.in_block_copy_sub_lengths_b == 1, \
                'in_sub_e:{}, in_sub_b:{}'.format(self.in_block_copy_sub_lengths_e, self.in_block_copy_sub_lengths_b)

        self.in_block_copy_src_data_per_read_b   = igemm_get_vector_size(self.in_block_copy_sub_lengths_b)
        self.in_block_copy_dst_data_per_write_n2 = igemm_get_vector_size(self.in_block_copy_sub_lengths_n2)
        self.wei_block_copy_src_data_per_read_e  = igemm_get_vector_size(self.wei_block_copy_sub_lengths_e)
        self.wei_block_copy_src_data_per_write_k = igemm_get_vector_size(self.wei_block_copy_sub_lengths_k)

        # register for a,b,c buffer
        self.num_accumulate_c_vgpr              = (self.gemm_m_repeat*self.gemm_m_per_thread_subc*self.gemm_n_repeat*self.gemm_n_per_thread_subc)
        self.num_accumulate_a_vgpr              = (self.gemm_m_repeat*self.gemm_m_per_thread_subc)
        self.num_accumulate_b_vgpr              = (self.gemm_n_repeat*self.gemm_n_per_thread_subc)

        self.num_global_load_a_vgpr             = (self.wei_block_copy_sub_lengths_e*self.wei_block_copy_sub_lengths_k)
        self.num_global_load_b_vgpr             = (self.in_block_copy_sub_lengths_e*self.in_block_copy_sub_lengths_n1*self.in_block_copy_sub_lengths_b*self.in_block_copy_sub_lengths_n2)

        # LDS size
        self.byte_lds_a                         = (4*self.e_per_block*self.k_per_block)
        self.byte_lds_b                         = (4*self.e_per_block*(self.gemm_n_repeat*self.b_per_block*self.gemm_n_per_thread_subc))
        self.byte_lds_a_np2                     = igemm_next_pow2( self.byte_lds_a)
        self.byte_lds_b_np2                     = igemm_next_pow2( self.byte_lds_b)
        self.byte_lds_single                    = igemm_next_pow2( self.byte_lds_a_np2 + self.byte_lds_b_np2)
        self.byte_lds_total                     = (2*self.byte_lds_single)
        # TODO: LDS size check

        # some parameter not in modular_conv
        self.thread_tile_m                      = self.gemm_m_repeat * self.gemm_m_per_thread_subc
        self.thread_tile_n                      = self.gemm_n_repeat * self.gemm_n_per_thread_subc
        self.thread_sub_tile_m                  = self.gemm_m_per_thread_subc
        self.thread_sub_tile_n                  = self.gemm_n_per_thread_subc

    def is_1x1(self):
        return self.name.endswith('1x1_dynamic_kernel')

    def to_dict(self):
        tunable_dict = {}
        tunable_dict['b_per_block']                       = self.b_per_block
        tunable_dict['k_per_block']                       = self.k_per_block 
        tunable_dict['e_per_block']                       = self.e_per_block
        tunable_dict['gemm_n_repeat']                     = self.gemm_n_repeat
        tunable_dict['gemm_m_per_thread_subc']            = self.gemm_m_per_thread_subc
        tunable_dict['gemm_n_per_thread_subc']            = self.gemm_n_per_thread_subc
        tunable_dict['gemm_m_level1_cluster']             = self.gemm_m_level1_cluster
        tunable_dict['gemm_n_level1_cluster']             = self.gemm_n_level1_cluster
        tunable_dict['gemm_m_level0_cluster']             = self.gemm_m_level0_cluster
        tunable_dict['gemm_n_level0_cluster']             = self.gemm_n_level0_cluster
        tunable_dict['in_block_copy_cluster_lengths_e']   = self.in_block_copy_cluster_lengths_e
        tunable_dict['in_block_copy_cluster_lengths_n1']  = self.in_block_copy_cluster_lengths_n1
        tunable_dict['in_block_copy_cluster_lengths_b']   = self.in_block_copy_cluster_lengths_b
        tunable_dict['in_block_copy_cluster_lengths_n2']  = self.in_block_copy_cluster_lengths_n2
        tunable_dict['wei_block_copy_cluster_lengths_e']  = self.wei_block_copy_cluster_lengths_e
        tunable_dict['wei_block_copy_cluster_lengths_k']  = self.wei_block_copy_cluster_lengths_k
        tunable_dict['name']                              = self.name
        return tunable_dict

    def serialize(self, line_starter = '; '):
        return  line_starter + 'b_per_block                      : {}'.format(self.b_per_block) + '\n' + \
                line_starter + 'k_per_block                      : {}'.format(self.k_per_block) + '\n' + \
                line_starter + 'e_per_block                      : {}'.format(self.e_per_block) + '\n' + \
                line_starter + 'gemm_n_repeat                    : {}'.format(self.gemm_n_repeat) + '\n' + \
                line_starter + 'gemm_m_per_thread_subc           : {}'.format(self.gemm_m_per_thread_subc) + '\n' + \
                line_starter + 'gemm_m_level0_cluster            : {}'.format(self.gemm_m_level0_cluster) + '\n' + \
                line_starter + 'gemm_m_level1_cluster            : {}'.format(self.gemm_m_level1_cluster) + '\n' + \
                line_starter + 'gemm_n_per_thread_subc           : {}'.format(self.gemm_n_per_thread_subc) + '\n' + \
                line_starter + 'gemm_n_level0_cluster            : {}'.format(self.gemm_n_level0_cluster) + '\n' + \
                line_starter + 'gemm_n_level1_cluster            : {}'.format(self.gemm_n_level1_cluster) + '\n' + \
                line_starter + 'in_block_copy_cluster_lengths_e  : {}'.format(self.in_block_copy_cluster_lengths_e) + '\n' + \
                line_starter + 'in_block_copy_cluster_lengths_n1 : {}'.format(self.in_block_copy_cluster_lengths_n1) + '\n' + \
                line_starter + 'in_block_copy_cluster_lengths_b  : {}'.format(self.in_block_copy_cluster_lengths_b) + '\n' + \
                line_starter + 'in_block_copy_cluster_lengths_n2 : {}'.format(self.in_block_copy_cluster_lengths_n2) + '\n' + \
                line_starter + 'wei_block_copy_cluster_lengths_e : {}'.format(self.wei_block_copy_cluster_lengths_e) + '\n' + \
                line_starter + 'wei_block_copy_cluster_lengths_k : {}'.format(self.wei_block_copy_cluster_lengths_k) + '\n' + \
                line_starter + '\n' + \
                line_starter + 'in_block_copy_sub_lengths_e      : {}'.format(self.in_block_copy_sub_lengths_e) + '\n' + \
                line_starter + 'in_block_copy_sub_lengths_n1     : {}'.format(self.in_block_copy_sub_lengths_n1) + '\n' + \
                line_starter + 'in_block_copy_sub_lengths_b      : {}'.format(self.in_block_copy_sub_lengths_b) + '\n' + \
                line_starter + 'in_block_copy_sub_lengths_n2     : {}'.format(self.in_block_copy_sub_lengths_n2) + '\n' + \
                line_starter + 'wei_block_copy_sub_lengths_e     : {}'.format(self.wei_block_copy_sub_lengths_e) + '\n' + \
                line_starter + 'wei_block_copy_sub_lengths_k     : {}'.format(self.wei_block_copy_sub_lengths_k) + '\n' + \
                line_starter + 'block_size                       : {}'.format(self.block_size) + '\n' + \
                line_starter + 'thread_tile                      : {}x{}'.format(self.thread_tile_m, self.thread_tile_n) + '\n' + \
                line_starter

    def serialize_as_init_list(self):
        return '{{{:>4},{:>4},{:>4},{:>4},{:>4},{:>4},{:>4},{:>4},{:>4},{:>4},{:>4},{:>4},{:>4},{:>4},{:>4},{:>4}}}'.format(
                            self.b_per_block,
                            self.k_per_block,
                            self.e_per_block,
                            self.gemm_n_repeat,
                            self.gemm_m_per_thread_subc,
                            self.gemm_n_per_thread_subc,
                            self.gemm_m_level0_cluster,
                            self.gemm_n_level0_cluster,
                            self.gemm_m_level1_cluster,
                            self.gemm_n_level1_cluster,
                            self.in_block_copy_cluster_lengths_e,
                            self.in_block_copy_cluster_lengths_n1,
                            self.in_block_copy_cluster_lengths_b,
                            self.in_block_copy_cluster_lengths_n2,
                            self.wei_block_copy_cluster_lengths_e,
                            self.wei_block_copy_cluster_lengths_k)

def igemm_encode_v4r1_kernel_name(tunable):
    if type(tunable) is igemm_tunable_parameter_t:
        tunable_dict = tunable.to_dict()
    else:
        tunable_dict = tunable
    b_per_block                       = tunable_dict['b_per_block']
    k_per_block                       = tunable_dict['k_per_block']
    e_per_block                       = tunable_dict['e_per_block']
    gemm_n_repeat                     = tunable_dict['gemm_n_repeat']
    gemm_m_per_thread_subc            = tunable_dict['gemm_m_per_thread_subc']
    gemm_n_per_thread_subc            = tunable_dict['gemm_n_per_thread_subc']
    gemm_m_level1_cluster             = tunable_dict['gemm_m_level1_cluster']
    gemm_n_level1_cluster             = tunable_dict['gemm_n_level1_cluster']
    gemm_m_level0_cluster             = tunable_dict['gemm_m_level0_cluster']
    gemm_n_level0_cluster             = tunable_dict['gemm_n_level0_cluster']
    in_block_copy_cluster_lengths_e   = tunable_dict['in_block_copy_cluster_lengths_e']
    in_block_copy_cluster_lengths_n1  = tunable_dict['in_block_copy_cluster_lengths_n1']
    in_block_copy_cluster_lengths_b   = tunable_dict['in_block_copy_cluster_lengths_b']
    in_block_copy_cluster_lengths_n2  = tunable_dict['in_block_copy_cluster_lengths_n2']
    wei_block_copy_cluster_lengths_e  = tunable_dict['wei_block_copy_cluster_lengths_e']
    wei_block_copy_cluster_lengths_k  = tunable_dict['wei_block_copy_cluster_lengths_k']

    # above is from config
    assert k_per_block % (gemm_m_per_thread_subc * gemm_m_level0_cluster * gemm_m_level1_cluster) == 0
    gemm_m_repeat = k_per_block // (gemm_m_per_thread_subc * gemm_m_level0_cluster * gemm_m_level1_cluster)

    thread_tile_m                     = gemm_m_repeat * gemm_m_per_thread_subc
    thread_tile_n                     = gemm_n_repeat * gemm_n_per_thread_subc

    if tunable.is_1x1():
        name_prefix = 'igemm_v4r1_1x1_dynamic_'
    else:
        name_prefix = 'igemm_v4r1_dynamic_'

    return name_prefix + '{}x{}x{}_{}x{}_{}x{}x{}x{}x{}x{}_{}x{}x{}x{}_{}x{}'.format(
                k_per_block, b_per_block*gemm_n_repeat*gemm_n_per_thread_subc, e_per_block, 
                thread_tile_m, thread_tile_n,
                gemm_m_per_thread_subc,gemm_m_level0_cluster,gemm_m_level1_cluster,gemm_n_per_thread_subc,gemm_n_level0_cluster,gemm_n_level1_cluster,
                in_block_copy_cluster_lengths_e,in_block_copy_cluster_lengths_n1,in_block_copy_cluster_lengths_b,in_block_copy_cluster_lengths_n2,
                wei_block_copy_cluster_lengths_e,wei_block_copy_cluster_lengths_k)

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
        # wei->matrix_a, input->matrix_b
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
