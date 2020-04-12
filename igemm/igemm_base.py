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

def igemm_log2(v):
    assert (v and (not(v & (v - 1)))), 'v must be power of 2'
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
        self.b_per_block                         = tunable_dict['BPerBlock']
        self.k_per_block                         = tunable_dict['KPerBlock']
        self.e_per_block                         = tunable_dict['EPerBlock']
        self.gemm_n_repeat                       = tunable_dict['GemmNRepeat']
        self.gemm_m_per_thread_subc              = tunable_dict['GemmMPerThreadSubC']
        self.gemm_n_per_thread_subc              = tunable_dict['GemmNPerThreadSubC']
        self.gemm_m_level1_cluster               = tunable_dict['GemmMLevel1Cluster']
        self.gemm_n_level1_cluster               = tunable_dict['GemmNLevel1Cluster']
        self.gemm_m_level0_cluster               = tunable_dict['GemmMLevel0Cluster']
        self.gemm_n_level0_cluster               = tunable_dict['GemmNLevel0Cluster']
        self.in_block_copy_cluster_lengths_e     = tunable_dict['InBlockCopyClusterLengths_E']
        self.in_block_copy_cluster_lengths_n1    = tunable_dict['InBlockCopyClusterLengths_N1']
        self.in_block_copy_cluster_lengths_b     = tunable_dict['InBlockCopyClusterLengths_B']
        self.in_block_copy_cluster_lengths_n2    = tunable_dict['InBlockCopyClusterLengths_N2']
        self.wei_block_copy_cluster_lengths_e    = tunable_dict['WeiBlockCopyClusterLengths_E']
        self.wei_block_copy_cluster_lengths_k    = tunable_dict['WeiBlockCopyClusterLengths_K']

        self.gemm_m_repeat = self.k_per_block // (self.gemm_m_per_thread_subc * self.gemm_m_level0_cluster * self.gemm_m_level1_cluster)

        self.in_block_copy_sub_lengths_e         = self.e_per_block // self.in_block_copy_cluster_lengths_e
        self.in_block_copy_sub_lengths_n1        = self.gemm_n_repeat // self.in_block_copy_cluster_lengths_n1
        self.in_block_copy_sub_lengths_b         = self.b_per_block // self.in_block_copy_cluster_lengths_b
        self.in_block_copy_sub_lengths_n2        = self.gemm_n_per_thread_subc // self.in_block_copy_cluster_lengths_n2
        self.wei_block_copy_sub_lengths_e        = self.e_per_block // self.wei_block_copy_cluster_lengths_e
        self.wei_block_copy_sub_lengths_k        = self.k_per_block // self.wei_block_copy_cluster_lengths_k

        self.block_size                          = self.gemm_m_level0_cluster * self.gemm_n_level0_cluster * self.gemm_m_level1_cluster * self.gemm_n_level1_cluster

        assert self.in_block_copy_sub_lengths_e == 1 and self.in_block_copy_sub_lengths_b == 1

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

    def to_dict(self):
        tunable_dict = {}
        tunable_dict['BPerBlock']                     = self.b_per_block
        tunable_dict['KPerBlock']                     = self.k_per_block 
        tunable_dict['EPerBlock']                     = self.e_per_block
        tunable_dict['GemmNRepeat']                   = self.gemm_n_repeat
        tunable_dict['GemmMPerThreadSubC']            = self.gemm_m_per_thread_subc
        tunable_dict['GemmNPerThreadSubC']            = self.gemm_n_per_thread_subc
        tunable_dict['GemmMLevel1Cluster']            = self.gemm_m_level1_cluster
        tunable_dict['GemmNLevel1Cluster']            = self.gemm_n_level1_cluster
        tunable_dict['GemmMLevel0Cluster']            = self.gemm_m_level0_cluster
        tunable_dict['GemmNLevel0Cluster']            = self.gemm_n_level0_cluster
        tunable_dict['InBlockCopyClusterLengths_E']   = self.in_block_copy_cluster_lengths_e
        tunable_dict['InBlockCopyClusterLengths_N1']  = self.in_block_copy_cluster_lengths_n1
        tunable_dict['InBlockCopyClusterLengths_B']   = self.in_block_copy_cluster_lengths_b
        tunable_dict['InBlockCopyClusterLengths_N2']  = self.in_block_copy_cluster_lengths_n2
        tunable_dict['WeiBlockCopyClusterLengths_E']  = self.wei_block_copy_cluster_lengths_e
        tunable_dict['WeiBlockCopyClusterLengths_K']  = self.wei_block_copy_cluster_lengths_k
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


def igemm_encode_v4r1_kernel_name(tunable):
    if type(tunable) is igemm_tunable_parameter_t:
        tunable_dict = tunable.to_dict()
    else:
        tunable_dict = tunable
    b_per_block                       = tunable_dict['BPerBlock']
    k_per_block                       = tunable_dict['KPerBlock']
    e_per_block                       = tunable_dict['EPerBlock']
    gemm_n_repeat                     = tunable_dict['GemmNRepeat']
    gemm_m_per_thread_subc            = tunable_dict['GemmMPerThreadSubC']
    gemm_n_per_thread_subc            = tunable_dict['GemmNPerThreadSubC']
    gemm_m_level1_cluster             = tunable_dict['GemmMLevel1Cluster']
    gemm_n_level1_cluster             = tunable_dict['GemmNLevel1Cluster']
    gemm_m_level0_cluster             = tunable_dict['GemmMLevel0Cluster']
    gemm_n_level0_cluster             = tunable_dict['GemmNLevel0Cluster']
    in_block_copy_cluster_lengths_e   = tunable_dict['InBlockCopyClusterLengths_E']
    in_block_copy_cluster_lengths_n1  = tunable_dict['InBlockCopyClusterLengths_N1']
    in_block_copy_cluster_lengths_b   = tunable_dict['InBlockCopyClusterLengths_B']
    in_block_copy_cluster_lengths_n2  = tunable_dict['InBlockCopyClusterLengths_N2']
    wei_block_copy_cluster_lengths_e  = tunable_dict['WeiBlockCopyClusterLengths_E']
    wei_block_copy_cluster_lengths_k  = tunable_dict['WeiBlockCopyClusterLengths_K']

    # above is from config
    assert k_per_block % (gemm_m_per_thread_subc * gemm_m_level0_cluster * gemm_m_level1_cluster) == 0
    gemm_m_repeat = k_per_block // (gemm_m_per_thread_subc * gemm_m_level0_cluster * gemm_m_level1_cluster)

    thread_tile_m                     = gemm_m_repeat * gemm_m_per_thread_subc
    thread_tile_n                     = gemm_n_repeat * gemm_n_per_thread_subc

    return 'igemm_v4r1_dynamic_' + '{}x{}x{}_{}x{}_{}x{}x{}x{}x{}x{}_{}x{}x{}x{}_{}x{}'.format(
                k_per_block, b_per_block*gemm_n_repeat*gemm_n_per_thread_subc, e_per_block, 
                thread_tile_m, thread_tile_n,
                gemm_m_per_thread_subc,gemm_m_level0_cluster,gemm_m_level1_cluster,gemm_n_per_thread_subc,gemm_n_level0_cluster,gemm_n_level1_cluster,
                in_block_copy_cluster_lengths_e,in_block_copy_cluster_lengths_n1,in_block_copy_cluster_lengths_b,in_block_copy_cluster_lengths_n2,
                wei_block_copy_cluster_lengths_e,wei_block_copy_cluster_lengths_k)

def igemm_get_v4r1_tunable_dict_array():
    dict_array = []

    dict = {}
    dict['BPerBlock']                       = 16
    dict['KPerBlock']                       = 128
    dict['EPerBlock']                       = 16

    dict['GemmNRepeat']                     = 2
    dict['GemmMPerThreadSubC']              = 4
    dict['GemmNPerThreadSubC']              = 4

    dict['GemmMLevel0Cluster']              = 4
    dict['GemmNLevel0Cluster']              = 4
    dict['GemmMLevel1Cluster']              = 4
    dict['GemmNLevel1Cluster']              = 4

    dict['InBlockCopyClusterLengths_E']     = 16
    dict['InBlockCopyClusterLengths_N1']    = 1
    dict['InBlockCopyClusterLengths_B']     = 16
    dict['InBlockCopyClusterLengths_N2']    = 1

    dict['WeiBlockCopyClusterLengths_E']    = 4
    dict['WeiBlockCopyClusterLengths_K']    = 64
    dict_array.append(dict)

    # 0 
    dict = {}
    dict['BPerBlock']                       = 16
    dict['KPerBlock']                       = 128
    dict['EPerBlock']                       = 8

    dict['GemmNRepeat']                     = 2
    dict['GemmMPerThreadSubC']              = 4
    dict['GemmNPerThreadSubC']              = 4

    dict['GemmMLevel0Cluster']              = 4
    dict['GemmNLevel0Cluster']              = 4
    dict['GemmMLevel1Cluster']              = 4
    dict['GemmNLevel1Cluster']              = 4

    dict['InBlockCopyClusterLengths_E']     = 8
    dict['InBlockCopyClusterLengths_N1']    = 2
    dict['InBlockCopyClusterLengths_B']     = 16
    dict['InBlockCopyClusterLengths_N2']    = 1

    dict['WeiBlockCopyClusterLengths_E']    = 2
    dict['WeiBlockCopyClusterLengths_K']    = 128
    dict_array.append(dict)

    # 1
    dict = {}
    dict['BPerBlock']                       = 8
    dict['KPerBlock']                       = 128
    dict['EPerBlock']                       = 8

    dict['GemmNRepeat']                     = 2
    dict['GemmMPerThreadSubC']              = 4
    dict['GemmNPerThreadSubC']              = 4

    dict['GemmMLevel0Cluster']              = 4
    dict['GemmNLevel0Cluster']              = 4
    dict['GemmMLevel1Cluster']              = 4
    dict['GemmNLevel1Cluster']              = 2

    dict['InBlockCopyClusterLengths_E']     = 8
    dict['InBlockCopyClusterLengths_N1']    = 1
    dict['InBlockCopyClusterLengths_B']     = 8
    dict['InBlockCopyClusterLengths_N2']    = 2

    dict['WeiBlockCopyClusterLengths_E']    = 2
    dict['WeiBlockCopyClusterLengths_K']    = 64
    dict_array.append(dict)

    # 2
    dict = {}
    dict['BPerBlock']                       = 8
    dict['KPerBlock']                       = 64
    dict['EPerBlock']                       = 8

    dict['GemmNRepeat']                     = 2
    dict['GemmMPerThreadSubC']              = 4
    dict['GemmNPerThreadSubC']              = 4

    dict['GemmMLevel0Cluster']              = 4
    dict['GemmNLevel0Cluster']              = 2
    dict['GemmMLevel1Cluster']              = 2
    dict['GemmNLevel1Cluster']              = 4

    dict['InBlockCopyClusterLengths_E']     = 8
    dict['InBlockCopyClusterLengths_N1']    = 1
    dict['InBlockCopyClusterLengths_B']     = 8
    dict['InBlockCopyClusterLengths_N2']    = 1

    dict['WeiBlockCopyClusterLengths_E']    = 4
    dict['WeiBlockCopyClusterLengths_K']    = 16
    dict_array.append(dict)

    # 3
    dict = {}
    dict['BPerBlock']                       = 16
    dict['KPerBlock']                       = 32
    dict['EPerBlock']                       = 4

    dict['GemmNRepeat']                     = 2
    dict['GemmMPerThreadSubC']              = 4
    dict['GemmNPerThreadSubC']              = 4

    dict['GemmMLevel0Cluster']              = 1
    dict['GemmNLevel0Cluster']              = 4
    dict['GemmMLevel1Cluster']              = 4
    dict['GemmNLevel1Cluster']              = 4

    dict['InBlockCopyClusterLengths_E']     = 4
    dict['InBlockCopyClusterLengths_N1']    = 1
    dict['InBlockCopyClusterLengths_B']     = 16
    dict['InBlockCopyClusterLengths_N2']    = 1

    dict['WeiBlockCopyClusterLengths_E']    = 4
    dict['WeiBlockCopyClusterLengths_K']    = 16
    dict_array.append(dict)

    # 3
    dict = {}
    dict['BPerBlock']                       = 16
    dict['KPerBlock']                       = 32
    dict['EPerBlock']                       = 4

    dict['GemmNRepeat']                     = 2
    dict['GemmMPerThreadSubC']              = 4
    dict['GemmNPerThreadSubC']              = 4

    dict['GemmMLevel0Cluster']              = 1
    dict['GemmNLevel0Cluster']              = 4
    dict['GemmMLevel1Cluster']              = 4
    dict['GemmNLevel1Cluster']              = 4

    dict['InBlockCopyClusterLengths_E']     = 4
    dict['InBlockCopyClusterLengths_N1']    = 1
    dict['InBlockCopyClusterLengths_B']     = 16
    dict['InBlockCopyClusterLengths_N2']    = 1

    dict['WeiBlockCopyClusterLengths_E']    = 4
    dict['WeiBlockCopyClusterLengths_K']    = 16
    dict_array.append(dict)

    # 4
    dict = {}
    dict['BPerBlock']                       = 16
    dict['KPerBlock']                       = 16
    dict['EPerBlock']                       = 4

    dict['GemmNRepeat']                     = 2
    dict['GemmMPerThreadSubC']              = 2
    dict['GemmNPerThreadSubC']              = 2

    dict['GemmMLevel0Cluster']              = 2
    dict['GemmNLevel0Cluster']              = 4
    dict['GemmMLevel1Cluster']              = 2
    dict['GemmNLevel1Cluster']              = 4

    dict['InBlockCopyClusterLengths_E']     = 4
    dict['InBlockCopyClusterLengths_N1']    = 1
    dict['InBlockCopyClusterLengths_B']     = 16
    dict['InBlockCopyClusterLengths_N2']    = 1

    dict['WeiBlockCopyClusterLengths_E']    = 4
    dict['WeiBlockCopyClusterLengths_K']    = 16
    dict_array.append(dict)

    # 5
    dict = {}
    dict['BPerBlock']                       = 8
    dict['KPerBlock']                       = 32
    dict['EPerBlock']                       = 4

    dict['GemmNRepeat']                     = 2
    dict['GemmMPerThreadSubC']              = 2
    dict['GemmNPerThreadSubC']              = 2

    dict['GemmMLevel0Cluster']              = 2
    dict['GemmNLevel0Cluster']              = 4
    dict['GemmMLevel1Cluster']              = 4
    dict['GemmNLevel1Cluster']              = 2

    dict['InBlockCopyClusterLengths_E']     = 4
    dict['InBlockCopyClusterLengths_N1']    = 2
    dict['InBlockCopyClusterLengths_B']     = 8
    dict['InBlockCopyClusterLengths_N2']    = 1

    dict['WeiBlockCopyClusterLengths_E']    = 4
    dict['WeiBlockCopyClusterLengths_K']    = 16
    dict_array.append(dict)
    return dict_array