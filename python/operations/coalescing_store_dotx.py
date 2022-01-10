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
        self.vector_fold_n = 1              # ... while calculating m/n global index    -> ignore now
        self.precision = 'fp16'             # dotx only support fp16 & int8
        self.gemm_k_global_split = False
        self.feat_vgpr_collapse = True
        self.co_m_update_os_functor = None  # update offset based on current i_m. otherwise use sgpr to update offset

        self.feat_co_m_flag_check = False   # custom flag check, not using internal check
        self.co_m_flag_check_start_functor = None
        self.co_m_flag_check_reset_functor = None
        self.div_v_const_func = None
        self.div_rem_v_const_func = None
        
    def get_gemmn_ratio(self):
        return self.cdm.block_size() / ((self.cdm.block_size() // self.cdm.macro_tile_n) * self.cdm.macro_tile_n)

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

    def can_skip_coalescing(self):
        '''
        currently, this api define CAN skip, but indeed this is MUST skip
        for coalescing write out, we assume thread coalescing along N, and it is easy to divide block size
        (256) along a power-of-2 number, but non-power-of-2 is very hard to do so.
        '''
        # if not utility_is_pow2(self.cdm.macro_tile_n):
        #     return True
        return False

class igemm_coalescing_store_dotx_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_coalescing_store_dotx_t
        self.ctrl = ctrl

    def name(self):
        return ''

    def get_smem_co_vector_size(self):
        '''
        1. if vector_store_m == 1 and vector_store_n == 1
            LDS store & read both using lanegroup_granularity_m as vector size, and global write out as single pixel

        2. if vector_store_m != 1 and vector_store_n == 1
            vector_store_m != 1 means we want to do vector store along m direction.
            If m is fast changing dimension, actually we dont want this be happen for now. this coalescing store class by default
            assume thread is conitnuout along n direction, so if m is fast changing, it is better thread is continuous along m,
            which may violate the assumption. TODO: implement thread continous along m in the future.

            we allow in case that vector_store_m != 1, which is useful in case like NCHWvect_c, where H*W (gemm_n) is still fast changing dim, 
            only with vect_c=vector_store_m in m dim.

            LDS read/write in vector_store_m, and vector_store_m <= lanegroup_granularity_m

        3. if vector_store_m == 1 and vector_store_n != 1
            LDS write in 1 pixel unit, LDS read in vector_store_n
        '''
        ctrl = self.ctrl
        l_mr, l_mt = ctrl.get_subgroup_length()
        data_byte = amdgpu_precision_data_byte(ctrl.precision)

        sst_vec, sld_vec, smem_trans = 1, 1, False
        if ctrl.vector_store_m == 1 and ctrl.vector_store_n == 1:
            assert ctrl.vector_fold_m == 1
            vector_size = min(l_mt * data_byte // 4, 1)
            sst_vec, sld_vec, smem_trans = vector_size, vector_size, False
        elif ctrl.vector_store_m != 1 and ctrl.vector_store_n == 1:
            # assert ctrl.get_lanegroup_granularity_m() % ctrl.vector_store_m == 0
            vector_size = min(l_mt * data_byte // 4, 1)
            # assert vector_size % ctrl.vector_store_m == 0
            assert ctrl.vector_store_m % ctrl.vector_fold_m == 0
            sst_vec, sld_vec, smem_trans = ctrl.vector_store_m, ctrl.vector_store_m, False
        elif ctrl.vector_store_m == 1 and ctrl.vector_store_n != 1:
            assert ctrl.vector_fold_m == 1
            sst_vec, sld_vec, smem_trans = 1, ctrl.vector_store_n, True
        else:
            assert False, f'not supported vector_store_m:{ctrl.vector_store_m}, vector_store_n:{ctrl.vector_store_n}'
        return sst_vec, sld_vec, smem_trans

    def get_gst_vector_size(self):
        ctrl = self.ctrl
        if ctrl.vector_store_m == 1 and ctrl.vector_store_n == 1:
            return 1
        elif ctrl.vector_store_m != 1 and ctrl.vector_store_n == 1:
            return ctrl.vector_store_m
        elif ctrl.vector_store_m == 1 and ctrl.vector_store_n != 1:
            return ctrl.vector_store_n
        else:
            assert False, f'not supported vector_store_m:{ctrl.vector_store_m}, vector_store_n:{ctrl.vector_store_n}'

    def get_co_desc(self):
        ctrl = self.ctrl
        assert not (ctrl.vector_store_m != 1 and ctrl.vector_store_n != 1), "currently not support vector store both in m and n direction"

        data_byte = amdgpu_precision_data_byte(ctrl.precision)

        g_mr, g_mt = ctrl.get_subgroups()
        l_mr, l_mt = ctrl.get_subgroup_length()
        n_mc = ctrl.cdm.lanegroup_m_per_cluster()       # this is among different thread
        n_ml = ctrl.cdm.lanegroup_m_per_wave()          # this is among different thread
        n_mv = ctrl.cdm.waves_per_m()                   # this is among different thread

        n_nc = ctrl.cdm.lanegroup_n_per_cluster()       # this is among different thread
        n_nl = ctrl.cdm.lanegroup_n_per_wave()          # this is among different thread
        n_nv = ctrl.cdm.waves_per_n()                   # this is among different thread

        sst_vec, sld_vec, smem_trans = self.get_smem_co_vector_size()
        assert l_mt % sst_vec == 0

        gst_vec = self.get_gst_vector_size()
        t_mr, t_nr, t_nt, t_mt = ctrl.cdm.acc_c_lengths()

        # 1. vgpr desc
        vgpr_lengths = list(ctrl.cdm.acc_c_lengths())
        vgpr_desc = make_naive_tensor_descriptor_packed(vgpr_lengths)

        # only split along gemm m
        vgpr_co_split_lengths = tensor_util_split_lengths(ctrl.coalescing_groups, vgpr_lengths, [0, 1, 2, 3], [1, 0, 0, 1])
        vgpr_split_desc = make_transform_tensor_descriptor(vgpr_desc, 
                                    make_tuple(trans_grouped_slice(vgpr_desc.get_lengths(),
                                                                [0, 0, 0, 0],
                                                                vgpr_co_split_lengths)),
                                    make_tuple([0, 1, 2, 3]),
                                    make_tuple([0, 1, 2, 3]))

        assert vgpr_split_desc.get_lengths() == [l_mr, t_nr, t_nt, l_mt]

        vgpr_last_dim_num = l_mt if sst_vec == 1 else sst_vec

        vgpr_co_desc = make_transform_tensor_descriptor(vgpr_split_desc, 
                                    make_tuple(
                                            trans_passthrough(vgpr_split_desc.get_length(0)),
                                            trans_passthrough(vgpr_split_desc.get_length(1)),
                                            trans_passthrough(vgpr_split_desc.get_length(2)),
                                            trans_vectorize(vgpr_split_desc.get_length(3), vgpr_last_dim_num),
                                        ),
                                    make_tuple(0, 1, 2, 3),
                                    make_tuple(0, 1, 2, 3))

        # 2. lds store desc
        sst_co_lengths = [  l_mr,                   # m, within thread
                            n_mv * n_ml * n_mc,     # m, among different thread
                            l_mt // sst_vec,        # m, within thread, consider vector fold
                            t_nr,                   # n, within thread
                            n_nv * n_nl * n_nc,     # n, among different thread
                            t_nt,                   # n, within thread
                            sst_vec * data_byte]    #    store vector  size

        sst_co_desc = make_naive_tensor_descriptor_packed(sst_co_lengths)

        # print(f'vgpr_lengths:{vgpr_lengths}, vgpr_co_split_lengths:{vgpr_co_split_lengths}, vgpr_co_desc:{vgpr_co_desc.get_lengths()}, vgpr_last_dim_num:{vgpr_last_dim_num}, sst_co_lengths:{sst_co_lengths}')

        # 3. gemm desc before/after transpose
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

        assert gemm_m_size == ctrl.cdm.macro_tile_m and gemm_n_size == ctrl.cdm.macro_tile_n

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

        assert gemm_co_prev_desc.get_length(4) % sst_vec == 0
        gemm_co_prev_sst_desc = make_transform_tensor_descriptor(gemm_co_prev_desc,
                                    make_tuple(
                                        trans_passthrough(gemm_co_prev_desc.get_length(0)),
                                        trans_passthrough(gemm_co_prev_desc.get_length(1)),
                                        trans_passthrough(gemm_co_prev_desc.get_length(2)),
                                        trans_passthrough(gemm_co_prev_desc.get_length(3)),
                                        trans_unmerge([gemm_co_prev_desc.get_length(4) // sst_vec, sst_vec]),
                                        trans_passthrough(gemm_co_prev_desc.get_length(5))),
                                    make_tuple(0, 1, 2, 3, 4, 5),
                                    make_tuple(0, 1, 2, 3, [4, 6], 5))

        gemm_co_3d_prev_desc = make_transform_tensor_descriptor(gemm_co_prev_sst_desc,
                                    make_tuple( trans_merge(gemm_co_prev_sst_desc.get_lengths()[:5]),
                                                trans_passthrough(gemm_n_size),
                                                trans_passthrough(sst_vec)),
                                    make_tuple([0, 1, 2, 3, 4], 5, 6),
                                    make_tuple(0, 1, 2))

        gemm_co_2d_desc = make_transform_tensor_descriptor(gemm_co_3d_prev_desc,
                                    make_tuple( trans_passthrough(gemm_co_3d_prev_desc.get_length(0)),
                                                trans_merge([gemm_n_size, sst_vec])),
                                    make_tuple(0, [1, 2]),
                                    make_tuple(0, 1))
        # print(f'gemm_split_desc:{gemm_split_desc.get_lengths()}, xxxx {gemm_co_2d_desc.get_lengths()}')

        # further more, we need sub-divide m_post_thread_length, because ds_read can not be larger than lgkmcnt max(15)
        # actually, we want to sub divide ds_read by its vector size.
        # note, number of gst issue maybe larger than sld issue, but sld issue never be larger than gst issue.
        # that is, we never want do to multiple ds_read to construct a data, then single global_store_dwordx...
        max_sld_issues_per_ssgroup = 4 if sld_vec in (2, 4) else 8

        if sst_vec != 1 and sld_vec != 1:
            # last dim of 3d_prev_desc is gemm_m
            assert sst_vec == sld_vec
            gemm_n_post_thread_length = 1
            gemm_n_post_cluster_length = gemm_n_size // gemm_n_post_thread_length
            gemm_m_post_cluster_length = ctrl.cdm.block_size() // gemm_n_post_cluster_length
            gemm_m_post_thread_length = gemm_m_slice_size // gemm_m_post_cluster_length

            assert (gemm_m_post_thread_length // sst_vec) * gemm_m_post_cluster_length == gemm_co_3d_prev_desc.get_length(0)

            gemm_co_post_sld_desc = make_transform_tensor_descriptor(gemm_co_3d_prev_desc,
                                    make_tuple( trans_unmerge([gemm_m_post_thread_length // sst_vec, gemm_m_post_cluster_length]),
                                                trans_unmerge([gemm_n_post_cluster_length, gemm_n_post_thread_length]),
                                                trans_passthrough(sst_vec)),
                                    make_tuple(0, 1, 2),
                                    make_tuple([0, 2], [3, 4], 1))

            # split due to max lgkmcnt
            num_sld_issues_per_ssgroup = utility_gcd(gemm_co_post_sld_desc.get_length(0), max_sld_issues_per_ssgroup)
            split_sld_groups = gemm_co_post_sld_desc.get_length(0) // num_sld_issues_per_ssgroup
            assert (ctrl.get_num_dword_per_group() // gst_vec) % split_sld_groups == 0, "TODO: need adjust ssgroup value based on dword per group"
            num_gst_per_ssgroup = ctrl.get_num_dword_per_group() // gst_vec // split_sld_groups
            
            # recompute num sst when ctrl.cdm.block_size() % gemm_n_size != 0
            gemmn_ratio = ctrl.get_gemmn_ratio()
            num_gst_per_ssgroup = int(num_gst_per_ssgroup * gemmn_ratio)

            gemm_co_post_sld_2_desc = make_transform_tensor_descriptor(gemm_co_post_sld_desc,
                                    make_tuple( trans_unmerge([split_sld_groups, num_sld_issues_per_ssgroup]),
                                                trans_passthrough(gemm_co_post_sld_desc.get_length(1)),     # sst_vec
                                                trans_passthrough(gemm_co_post_sld_desc.get_length(2)),     # gemm_m_post_cluster_length
                                                trans_passthrough(gemm_co_post_sld_desc.get_length(3)),     # gemm_n_post_cluster_length
                                                trans_passthrough(gemm_co_post_sld_desc.get_length(4)),     # gemm_n_post_thread_length
                                                ),
                                    make_tuple(0, 1, 2, 3, 4),
                                    make_tuple([0, 1], 2, 3, 4, 5))

            assert num_gst_per_ssgroup * gst_vec == num_sld_issues_per_ssgroup * sst_vec, f'num_gst_per_ssgroup:{num_gst_per_ssgroup}, gst_vec:{gst_vec}, num_sld_issues_per_ssgroup:{num_sld_issues_per_ssgroup}, sst_vec:{sst_vec}'
            gemm_co_post_desc = make_transform_tensor_descriptor(gemm_co_post_sld_2_desc,
                                    make_tuple( trans_passthrough(gemm_co_post_sld_2_desc.get_length(0)),
                                                trans_merge([num_sld_issues_per_ssgroup, sst_vec]),
                                                trans_passthrough(gemm_co_post_sld_2_desc.get_length(3)),
                                                trans_passthrough(gemm_co_post_sld_2_desc.get_length(4)),
                                                trans_passthrough(gemm_co_post_sld_2_desc.get_length(5))),
                                    make_tuple(0, [1, 2], 3, 4, 5),
                                    make_tuple(0, 1, 2, 3, 4))

            # 4 lds read desc TODO: better specify
            sld_co_lengths = [split_sld_groups, num_sld_issues_per_ssgroup, gemm_m_post_cluster_length * sld_vec,
                                gemm_n_post_cluster_length, gemm_n_post_thread_length * data_byte]
            sld_co_desc = make_naive_tensor_descriptor_packed(sld_co_lengths)

        else:
            assert sst_vec == 1 and sld_vec != 1 and sld_vec == gst_vec
            gemm_n_post_thread_length = sld_vec
            gemm_n_post_cluster_length = gemm_n_size // gemm_n_post_thread_length
            gemm_m_post_cluster_length = ctrl.cdm.block_size() // gemm_n_post_cluster_length
            gemm_m_post_thread_length = gemm_co_2d_desc.get_length(0) // gemm_m_post_cluster_length

            # print(f'm:{gemm_m_post_thread_length}x{gemm_m_post_cluster_length}, n:{gemm_n_post_cluster_length}x{gemm_n_post_thread_length}')

            num_sld_issues_per_ssgroup = utility_gcd(gemm_m_post_thread_length, max_sld_issues_per_ssgroup)
            split_sld_groups = gemm_m_post_thread_length // num_sld_issues_per_ssgroup
            assert (ctrl.get_num_dword_per_group() // gst_vec) % split_sld_groups == 0, "TODO: need adjust ssgroup value based on dword per group"
            num_gst_per_ssgroup = ctrl.get_num_dword_per_group() // gst_vec // split_sld_groups
            
            # recompute num sst when ctrl.cdm.block_size() % gemm_n_size != 0
            gemmn_ratio = ctrl.cdm.block_size() / ((ctrl.cdm.block_size() // gemm_n_size) * gemm_n_size)
            num_gst_per_ssgroup = int(num_gst_per_ssgroup * gemmn_ratio)

            #assert num_sld_issues_per_ssgroup * split_sld_groups * gemm_m_post_cluster_length == gemm_co_2d_desc.get_length(0), f'{num_sld_issues_per_ssgroup} * {split_sld_groups} * {gemm_m_post_cluster_length} == {gemm_co_2d_desc.get_length(0)}'
            assert gemm_n_post_thread_length * gemm_n_post_cluster_length == gemm_co_2d_desc.get_length(1)

            # desc after transpose
            gemm_co_post_desc = make_transform_tensor_descriptor(gemm_co_2d_desc,
                                        make_tuple(trans_unmerge([split_sld_groups, num_sld_issues_per_ssgroup, gemm_m_post_cluster_length]),
                                                    trans_unmerge([gemm_n_post_cluster_length, gemm_n_post_thread_length])),
                                        make_tuple(0, 1),
                                        make_tuple([0, 1, 2], [3, 4]))

            # 4 lds read desc
            sld_co_lengths = [split_sld_groups, num_sld_issues_per_ssgroup, gemm_m_post_cluster_length,
                                gemm_n_post_cluster_length, gemm_n_post_thread_length * data_byte]
            sld_co_desc = make_naive_tensor_descriptor_packed(sld_co_lengths)

        return vgpr_last_dim_num, split_sld_groups, num_sld_issues_per_ssgroup, num_gst_per_ssgroup, \
                vgpr_co_split_lengths, gemm_co_split_lengths, \
                vgpr_co_desc, sst_co_desc, gemm_co_prev_desc, gemm_co_post_desc, sld_co_desc

    def init_co_lds_offset(self, v_co_sst, v_co_sld, v_gemm_im, v_gemm_in, v_tid, v_tmp4):
        ctrl = self.ctrl
        data_byte = amdgpu_precision_data_byte(ctrl.precision)
        t_mr, t_nr, t_nt, t_mt = ctrl.cdm.acc_c_lengths()
        l_mr, l_mt = ctrl.get_subgroup_length()
        sst_vec, sld_vec, smem_trans = self.get_smem_co_vector_size()
        gst_vec = self.get_gst_vector_size()

        with self._deferred_context():
            self._emit(f"; init_co_lds_offset for dotx")
            if sst_vec != 1 and sld_vec != 1:
                assert sst_vec == sld_vec
                assert t_mt % l_mt == 0
                self._emit(f"v_lshrrev_b32 v[{v_tmp4}], {utility_log2(sst_vec * (t_mt // l_mt))}, v[{v_gemm_im}]    ; shink m by {sst_vec * (t_mt // l_mt)}")
                self._emit(f"v_lshlrev_b32 v[{v_tmp4} + 1],  {utility_log2(sst_vec)}, v[{v_gemm_in}]    ; expand n by {sst_vec}")
                #self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp4}], {utility_log2(ctrl.cdm.macro_tile_n * sst_vec)}, v[{v_tmp4} + 1]    ; macro_tile_n:{ctrl.cdm.macro_tile_n}, sst_vec:{sst_vec}")
                self._emit(f"v_mad_u32_u24 v[{v_co_sst}], v[{v_tmp4}], {ctrl.cdm.macro_tile_n * sst_vec}, v[{v_tmp4} + 1]    ; macro_tile_n:{ctrl.cdm.macro_tile_n}, sst_vec:{sst_vec}")
            else:
                assert sst_vec == 1 and sld_vec != 1 and sld_vec == gst_vec
                assert t_mt % l_mt == 0
                self._emit(f"v_lshrrev_b32 v[{v_tmp4}], {utility_log2(t_mt // l_mt)}, v[{v_gemm_im}]    ; shink m by {sst_vec * (t_mt // l_mt)}")
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp4}], {utility_log2(ctrl.cdm.macro_tile_n)}, v[{v_gemm_in}]")

            self._emit(f"v_lshlrev_b32 v[{v_co_sld}], {utility_log2(data_byte * sld_vec)}, v[{v_tid}]   ; sld vec:{sld_vec} * byte:{data_byte}")
            self._emit(f"v_lshlrev_b32 v[{v_co_sst}], {utility_log2(data_byte)}, v[{v_co_sst}] ; byte:{data_byte}")

        return self._get_deferred()

    def init_co_sub_m_index(self, v_co_sub_m_index, v_tid, v_tmp4):
        ctrl = self.ctrl

        l_mr, l_mt = ctrl.get_subgroup_length()
        t_mr, t_nr, t_nt, t_mt = ctrl.cdm.acc_c_lengths()

        #        gemm_m_lengths = [ctrl.cdm.lanegroup_repeat_m,              # thread lengths
        #                            ctrl.cdm.waves_per_m(),
        #                            ctrl.cdm.lanegroup_m_per_wave(),
        #                            ctrl.cdm.lanegroup_m_per_cluster(),
        #                            ctrl.cdm.lanegroup_m_per_thread()]      # thread lengths
        #
        #        gemm_m_split_lengths = tensor_util_split_lengths(
        #                                            ctrl.coalescing_groups, gemm_m_lengths,
        #                                            [0, 1, 2, 3, 4], [1, 0, 0, 0, 1])
        sst_vec, sld_vec, smem_trans = self.get_smem_co_vector_size()
        gst_vec = self.get_gst_vector_size()

        cluster_m_len = ctrl.cdm.waves_per_m() * ctrl.cdm.lanegroup_m_per_wave() * ctrl.cdm.lanegroup_m_per_cluster()

        # def pretty_div(n, d):
        #     assert n % d == 0
        #     return n // d

        with self._deferred_context():
            self._emit(f"; init_co_sub_m_index for dotx")
            if sst_vec != 1 and sld_vec != 1:
                # last dim of 3d_prev_desc is gemm_m
                assert sst_vec == sld_vec
                gemm_n_post_thread_length = 1
                gemm_n_post_cluster_length = ctrl.cdm.macro_tile_n // gemm_n_post_thread_length
            else:
                assert sst_vec == 1 and sld_vec != 1 and sld_vec == gst_vec
                gemm_n_post_thread_length = sld_vec
                gemm_n_post_cluster_length = ctrl.cdm.macro_tile_n // gemm_n_post_thread_length

            def emit_co_m():
                gemm_m_cluster_length = ctrl.cdm.block_size() // gemm_n_post_cluster_length
                with self._deferred_context():
                    while True:
                        if gemm_m_cluster_length > 1:
                            #self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {utility_log2(gemm_n_post_cluster_length)}, v[{v_tid}]  ; {gemm_n_post_cluster_length} cluster per gemm_n")
                            self._emit(ctrl.div_v_const_func(v_co_sub_m_index, v_tid, gemm_n_post_cluster_length, v_tmp4))
                            if sst_vec != 1:
                                self._emit(f"v_lshlrev_b32 v[{v_co_sub_m_index}], {utility_log2(sst_vec)}, v[{v_co_sub_m_index}] ; expand m by sst_vec:{sst_vec}")
                        else:
                            self._emit(f"v_mov_b32 v[{v_co_sub_m_index}], 0")   # early exist
                            break
                        # this is to compute in gemm_m, by 3 d, [l_mr, cluster_m_len, l_mt], so we want to calculate from right to left

                        # 1. dim l_mt
                        if gemm_m_cluster_length > l_mt:
                            gemm_m_cluster_length = gemm_m_cluster_length // l_mt
                            if t_mt != l_mt:
                                self._emit(f"v_and_b32 v[{v_tmp4} + 1], {l_mt - 1},  v[{v_co_sub_m_index}]  ; length distributed within l_mt:{l_mt}")
                                self._emit(f"v_lshrrev_b32 v[{v_tmp4} + 2], {utility_log2(l_mt)}, v[{v_co_sub_m_index}]")
                                self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp4} + 2], {utility_log2(t_mt)}, v[{v_tmp4} + 1]")
                        else:
                            gemm_m_cluster_length = gemm_m_cluster_length // l_mt
                            break

                        # 2. dim cluster_m_len
                        # if gemm_m_cluster_length > cluster_m_len:
                        #     gemm_m_cluster_length = pretty_div(gemm_m_cluster_length, cluster_m_len)
                        #     # nonthing to emit
                        # else:
                        #     break
                        gemm_m_cluster_length = gemm_m_cluster_length // cluster_m_len

                        # 3. dim l_mr
                        if gemm_m_cluster_length > l_mr:
                            gemm_m_cluster_length = gemm_m_cluster_length // l_mr
                            if t_mr != l_mr:
                                self._emit(f"v_and_b32 v[{v_tmp4} + 1], {l_mr - 1},  v[{v_co_sub_m_index}]  ; length distributed within l_mr:{l_mr}")
                                self._emit(f"v_lshrrev_b32 v[{v_tmp4} + 2], {utility_log2(l_mr)}, v[{v_co_sub_m_index}]")
                                self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp4} + 2], {utility_log2(t_mr)}, v[{v_tmp4} + 1]")
                        else:
                            gemm_m_cluster_length = gemm_m_cluster_length // l_mr
                            break

                        break

                assert gemm_m_cluster_length in (0, 1), f'gemm_m_cluster_length:{gemm_m_cluster_length}'
                return self._get_deferred()

            self._emit(emit_co_m())
            if ctrl.vector_fold_m != 1:
                self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {utility_log2(ctrl.vector_fold_m)}, v[{v_co_sub_m_index}] ; fold sub_m by {ctrl.vector_fold_m}")
        return self._get_deferred()

    def init_co_sub_n_index(self, v_co_sub_n_index, v_tid, v_tmp2):
        '''
        in n dimension, always have one thread per column
        '''
        ctrl = self.ctrl
        sst_vec, sld_vec, smem_trans = self.get_smem_co_vector_size()

        with self._deferred_context():
            self._emit(f"; init_co_sub_n_index dotx")
            if sst_vec != 1 and sld_vec != 1:
                assert sst_vec == sld_vec
                self._emit(ctrl.div_rem_v_const_func(v_co_sub_n_index, None, v_tid, ctrl.cdm.macro_tile_n, v_tmp2))
                #self._emit(f"v_and_b32 v[{v_co_sub_n_index}], {ctrl.cdm.macro_tile_n - 1}, v[{v_tid}]")
            else:
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {utility_log2(sld_vec)}, v[{v_tid}]")
                self._emit(f"v_and_b32 v[{v_co_sub_n_index}], {ctrl.cdm.macro_tile_n - 1}, v[{v_tmp2}]")
        return self._get_deferred()

    def get_vgpr_usage(self):
        ctrl = self.ctrl
        if ctrl.feat_vgpr_collapse:
            vgpr_last_dim_num, split_sld_groups, num_sld_issues_per_ssgroup, num_gst_per_ssgroup, \
            vgpr_co_split_lengths, gemm_co_split_lengths, \
            vgpr_co_desc, sst_co_desc, gemm_co_prev_desc, gemm_co_post_desc, sld_co_desc = \
                    self.get_co_desc()
            vgpr_lengths = vgpr_co_desc.get_lengths()
            return tensor_util_reduce(vgpr_lengths, lambda a, b: a*b, 1)
        else:
            return ctrl.cxm.total_acc_c() // ctrl.coalescing_groups

    def __call__(self, v_c, v_co_sst, v_co_sld, s_p_out, v_out_offset, s_out_offset, s_gemm_m0_stride, s_gemm_m1_stride, s_tmp6, v_store_flag = None, s_k = None, v_cur_k = None, s_block_gtc_ik = None, v_co_sub_m_index = None, v_tmp0 = None):

        # if no need s_out_offset, set to integer 0
        # if no need flag to dicide store, set v_store_flag to 0
        def flatten(x):
            from functools import reduce
            return reduce(lambda a, b: a*b, x, 1)
        ctrl = self.ctrl
        assert not (ctrl.vector_store_m != 1 and ctrl.vector_store_n != 1), "currently not support vector store both in m and n direction"

        data_byte = amdgpu_precision_data_byte(ctrl.precision)
        v_c = sym_t(v_c)
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

        sst_vec, sld_vec, smem_trans = self.get_smem_co_vector_size()
        assert l_mt % sst_vec == 0, f'l_mt:{l_mt}, sst_vec:{sst_vec}'

        gst_vec = self.get_gst_vector_size()

        inst_sst = inst_ds_write_t(sst_vec * data_byte)
        inst_sld = inst_ds_read_t(sld_vec * data_byte)
        if ctrl.gemm_k_global_split:
            v_pack = 2 if gst_vec == 2 and data_byte == 2 else 1
            inst_gst = inst_buffer_atomic_add_dword_t(gst_vec * data_byte, v_pack) 
        else:
            inst_gst = inst_buffer_store_t(gst_vec * data_byte)

        s_out_offset_itr = sym_t(s_tmp6(0))

        t_mr, t_nr, t_nt, t_mt = ctrl.cdm.acc_c_lengths()

        vgpr_last_dim_num, split_sld_groups, num_sld_issues_per_ssgroup, num_gst_per_ssgroup, \
        vgpr_co_split_lengths, gemm_co_split_lengths, \
        vgpr_co_desc, sst_co_desc, gemm_co_prev_desc, gemm_co_post_desc, sld_co_desc = \
                self.get_co_desc()

        def vgpr_coord_2_sst_coord(v_coord):
            # TODO: coordinate remapping
            assert len(v_coord) == 4
            s_coord = [0] * 7
            s_coord[0], s_coord[2], s_coord[3], s_coord[5] = v_coord[0], v_coord[3], v_coord[1], v_coord[2]
            return s_coord

        with self._deferred_context():
            self._emit(f"; coalescing store, mapping:{ctrl.cdm.serialize()}")
            self._emit(f"; coalescing_groups:{ctrl.coalescing_groups}, num_dword_per_group:{ctrl.get_num_dword_per_group()}, block_size:{ctrl.cdm.block_size()}")
            self._emit(f'; gemm_co_prev_desc:{gemm_co_prev_desc.get_lengths()}, gemm_co_split_lengths:{gemm_co_split_lengths}, gemm_co_post_desc:{gemm_co_post_desc.get_lengths()}')
            self._emit(f"s_mul_i32 s[{s_gemm_m1_stride}], {data_byte}, s[{s_gemm_m1_stride}] ; data_byte:{data_byte}")
            self._emit(f"s_barrier")

            gemm_m_co_start_coord = [0, 0, 0, 0, 0]
            vgpr_co_start_coord = [0, 0, 0, 0]

            accvgpr_consume_list = list()   # record the list for vgpr used to store C matrix, for debug

            for i_group in range(ctrl.coalescing_groups):
                m_index_start_per_group = gemm_co_prev_desc.calculate_offset([0, 0, 0, 0, 0, 0]) // ctrl.cdm.macro_tile_n

                for vgpr_coord in itertools.product(*[range(d) for d in vgpr_co_desc.get_lengths()]):
                    # vgpr_coord should be a list with length 4
                    vgpr_index = vgpr_co_desc.calculate_offset(list(vgpr_coord))
                    sst_coord = vgpr_coord_2_sst_coord(list(vgpr_coord))
                    sst_offset = sst_co_desc.calculate_offset(sst_coord)
                    #self._emit(f"; vgpr_coord:{vgpr_coord}, lengths:{vgpr_co_desc.get_lengths()}, sst_coord:{sst_coord}, lengths:{sst_co_desc.get_lengths()}")

                    if ctrl.precision == 'fp16':
                        for i in range(vgpr_last_dim_num):
                            # self._emit(f"v_cvt_f16_f32 v[{v_c(vgpr_index + i)}], v[{v_c(vgpr_index + i)}]")
                            if i % 2 == 0:
                                self._emit(f"v_cvt_f16_f32 v[{v_c(vgpr_index + (i // 2))}], v[{v_c(vgpr_index + i)}]")
                            else:
                                self._emit(f"v_cvt_f16_f32_sdwa v[{v_c(vgpr_index + (i // 2))}], v[{v_c(vgpr_index + i)}]  dst_sel:WORD_1")
                            accvgpr_consume_list.append(vgpr_index + i)

                        #if not smem_trans:
                        #    for i in range(sst_vec // 2):
                        #        self._emit(f"v_pack_b32_f16 v[{v_c(vgpr_index + i)}], v[{v_c(vgpr_index + 2 * i)}], v[{v_c(vgpr_index + 2 * i + 1)}]")

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
                        for i in range(vgpr_last_dim_num):
                            self._emit(inst_sst(v_co_sst(), v_c(vgpr_index + i), sst_offset + i * ctrl.cdm.macro_tile_n * data_byte))

                def emit_calculate_out_offset_itr_m(i_m, i_m0, i_m1):
                    comments = f"   ; i_m:{i_m}(i_m0:{i_m0},i_m1:{i_m1}, fold_m:{ctrl.vector_fold_m})"
                    if ctrl.co_m_update_os_functor:
                        self._emit(ctrl.co_m_update_os_functor(i_m, i_m0, i_m1))        # TODO: better sigture
                    else:
                        if s_gemm_m0_stride is not None:
                            assert False, "not supported"
                        else:
                            '''
                            no m0_stride, which indicate m0, m1 is continuous, no need to deal with m0, m1 seperately
                            '''
                            i_m_fold = i_m // ctrl.vector_fold_m

                            if i_m_fold == 0:
                                if no_s_out_offset:
                                    self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], 0" + comments)
                                    if s_k is not None:
                                        if ctrl.vector_fold_m != 1:
                                            # attention!, we lshrrev v_co_sub_m_index before. here we need to shift back
                                            self._emit(f"v_lshlrev_b32 v[{v_co_sub_m_index()}], {utility_log2(ctrl.vector_fold_m)}, v[{v_co_sub_m_index()}]")
                                        self._emit(v_add_nc_u32(v_cur_k(), s_block_gtc_ik(), v_co_sub_m_index()))
                                        self._emit(f"v_mov_b32 v[{v_tmp0()}], v[{v_cur_k()}]")
                                else:
                                    self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], s[{s_out_offset}]" + comments)
                            elif i_m_fold == 1:
                                if no_s_out_offset:
                                    self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], s[{s_gemm_m1_stride}]" + comments)
                                    if s_k is not None:
                                        self._emit(v_add_nc_u32(v_tmp0(), i_m, v_cur_k()))
                                else:
                                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_gemm_m1_stride}], s[{s_out_offset}]" + comments)
                            else:
                                if no_s_out_offset:
                                    self._emit(f"s_mul_i32 s[{s_out_offset_itr()}], {i_m_fold}, s[{s_gemm_m1_stride}]" + comments)
                                    if s_k is not None:
                                        self._emit(v_add_nc_u32(v_tmp0(), i_m, v_cur_k()))
                                else:
                                    self._emit(f"s_mul_i32 s[{s_tmp6(3)}], {i_m_fold}, s[{s_gemm_m1_stride}]")
                                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_tmp6(3)}], s[{s_out_offset}]" + comments)

                # emit first calculation before wait for store
                emit_calculate_out_offset_itr_m(m_index_start_per_group, 0, 0)

                issue_list = []
                gemmn_ratio = ctrl.get_gemmn_ratio()
                num_sld_total_dword = int(ctrl.get_num_dword_per_group() // sld_vec * gemmn_ratio)

                self._emit(f"s_waitcnt lgkmcnt(0)")
                self._emit(f"s_barrier")
                valid_threads = int(ctrl.cdm.block_size() / gemmn_ratio)
                
                for i_d in range(num_sld_total_dword):
                    issue_list.append(1)    # always issue 1

                gst_desc = make_naive_tensor_descriptor_packed([split_sld_groups, num_gst_per_ssgroup])
                gst_coord = [0, 0]

                for i_ssgroup in range(split_sld_groups):
                    self._emit(f";   load from lds, i_ssgroup:{i_ssgroup}, num_sld_issues_per_ssgroup:{num_sld_issues_per_ssgroup}")
                    self._emit(f"v_cmpx_gt_u32 {valid_threads}, v[v_coalescing_store_index]")
                    for i_d in range(num_sld_issues_per_ssgroup):
                        vgpr_index = (i_d + (i_ssgroup if not ctrl.feat_vgpr_collapse else 0) * num_sld_issues_per_ssgroup) * sld_vec * data_byte // 4 # when data byte is 2, only cost 2 vgpr per time
                        sld_coord = [i_ssgroup, i_d, 0, 0, 0]
                        sld_offset = sld_co_desc.calculate_offset(sld_coord)
                        self._emit(inst_sld(v_c(vgpr_index), v_co_sld(), sld_offset))
                    current_issue_list = issue_list[i_ssgroup * num_sld_issues_per_ssgroup : (i_ssgroup+1) * num_sld_issues_per_ssgroup]
                    if not ctrl.feat_co_m_flag_check and (v_store_flag is not None and type(v_store_flag) is str):
                        self._emit(v_cmp_eq_i32("vcc", 1, v_store_flag))
                        self._emit(f"s_and_saveexec_b64 s[{s_tmp6(4)}:{s_tmp6(5)}], vcc")
                    self._emit(f";   store to global, m index start:{m_index_start_per_group}")

                    for i_gst in range(num_gst_per_ssgroup):
                        # i_gst_flat = i_gst + i_ssgroup * num_gst_per_ssgroup
                        i_gst_flat = gst_desc.calculate_offset(gst_coord)
                        if len(current_issue_list) != 0:
                            if i_gst % (sld_vec if gst_vec == 1 else 1) == 0:
                                i_issues =  (i_gst // (sld_vec if gst_vec == 1 else 1)) + 1
                                i_issue_list = current_issue_list[i_issues:]
                                i_issue_cnt = utility_flatten_list_accumulate(i_issue_list) if len(i_issue_list) != 0 else 0
                                self._emit(f"s_waitcnt lgkmcnt({i_issue_cnt})")
                        # vdata, vaddr, srsrc, soffset, offset
                        if not ctrl.feat_co_m_flag_check and (s_k is not None):
                            #self._emit(f"v_cmp_gt_u32 vcc, s[{s_k()}], v[{v_tmp0()}]")
                            self._emit(v_cmp_gt_u32('vcc', s_k(), v_tmp0()))
                            self._emit(f"s_and_saveexec_b64 s[{s_tmp6(4)}:{s_tmp6(5)}], vcc")
                        elif ctrl.feat_co_m_flag_check:
                            self._emit(ctrl.co_m_flag_check_start_functor())
                        cur_vgpr_gst = (i_gst_flat if not ctrl.feat_vgpr_collapse else i_gst) * gst_vec//(4 // data_byte)
                        lo_hi = i_gst_flat % 2 if ctrl.precision == 'fp16' and gst_vec == 1 else 0
                        self._emit(inst_gst(v_c(cur_vgpr_gst), v_out_offset, s_p_out, s_out_offset_itr(), 0, lo_hi))
                        if not ctrl.feat_co_m_flag_check and (s_k is not None):
                            self._emit(f"s_or_b64 exec, exec, s[{s_tmp6(4)}:{s_tmp6(5)}]")
                        elif ctrl.feat_co_m_flag_check:
                            self._emit(ctrl.co_m_flag_check_reset_functor())
                        if ctrl.precision == 'int8' and gst_vec == 1:
                            if i_gst_flat % 4 != 3:
                                self._emit(f"v_lshrrev_b32 v[{v_c(cur_vgpr_gst)}], 8, v[{v_c(cur_vgpr_gst)}]")

                        if i_gst_flat != (int(ctrl.get_num_dword_per_group() * gemmn_ratio) // gst_vec) - 1:
                            gst_coord = move_tensor_coordinate(gst_desc, gst_coord, [1, 1])
                            # TODO: this is ugly. better unify this by further transpose of gemm_co_post_desc
                            next_i_ssgroup = gst_coord[0]
                            if sst_vec != 1 and sld_vec != 1:
                                next_i_gst = gst_coord[1] * gst_vec
                            else:
                                next_i_gst = gst_coord[1]            # in this case, gst_vec actually is the same as last dim of gemm_co_post_desc
                            gemm_coord_next = [next_i_ssgroup, next_i_gst, 0, 0, 0]

                            # print(f'gemm_coord_next:{gemm_coord_next}, gemm_co_post_desc:{gemm_co_post_desc.get_lengths()}, gst_vec:{gst_vec}, num_gst_per_ssgroup:{num_gst_per_ssgroup}, split_sld_groups:{split_sld_groups}, num_sld_issues_per_ssgroup:{num_sld_issues_per_ssgroup}, i_gst_flat:{i_gst_flat}')
                            im_next = gemm_co_post_desc.calculate_offset(gemm_coord_next) // ctrl.cdm.macro_tile_n
                            # print(f'gemm_coord_next:{gemm_coord_next},{gemm_co_post_desc.calculate_offset(gemm_coord_next)}, gemm_co_post_desc:{gemm_co_post_desc.get_lengths()}, num_gst_per_ssgroup:{num_gst_per_ssgroup}, gst_vec:{gst_vec}, gemm_co_post_desc:{gemm_co_post_desc.get_lengths()}')

                            emit_calculate_out_offset_itr_m(im_next, 0, 0)
                    if not ctrl.feat_co_m_flag_check and (v_store_flag is not None and type(v_store_flag) is str):
                        self._emit(f"s_mov_b64 exec, -1")

                # move slice window for next loop
                move_grouped_slice_start_coord(gemm_co_prev_desc, gemm_co_split_lengths)
                move_grouped_slice_start_coord(gemm_co_post_desc, gemm_co_split_lengths)
                move_grouped_slice_start_coord(vgpr_co_desc, vgpr_co_split_lengths)
            
            # do some assert
            accvgpr_consume_list.sort()
            assert accvgpr_consume_list == [x for x in range(ctrl.cdm.total_acc_c())], f"accvgpr_consume_list:{accvgpr_consume_list}"

        return self._get_deferred()
