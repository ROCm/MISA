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
from .utility import *
from .dotx import *


# assuming dpp8. dpp16 is less efficient so ignore that
LANEGROUP_SIZE   = 8

class ctrl_dotx_mapping_t(object):
    '''
    dlops dotx mapping
    following dpp8's term, each lanegroup is a 8x8 matrix, and not support interleave inside lanegroup
    C matrix:

    m_dim   | thread_length                | cluster_length
    --------+------------------------------+------------------------------+
    level_0 | lanegroup_m_per_thread(), 8x | lanegroup_m_per_cluster(), 1 | -> lanegroup_tile_m
    level_1 | 1                            | lanegroup_m_per_wave()       |
    level_2 | 1                            | waves_per_m()                |
    level_3 | lanegroup_repeat_m           | 1                            |

    n_dim   | thread_length                | cluster_length
    --------+------------------------------+------------------------------+
    level_0 | lanegroup_n_per_thread()     | lanegroup_n_per_cluster(), 8 | -> lanegroup_tile_n
    level_1 | 1                            | lanegroup_n_per_wave()       |
    level_2 | 1                            | waves_per_n()                |
    level_3 | lanegroup_repeat_n           | 1                            |

    -----------------------------------------------------------------------------------

    A/B matrix:

    m_dim   | thread_length                | cluster_length
    --------+------------------------------+------------------------------+
    level_0 | thread_m()                   | lanegroup_size_m(), 8        |
    level_1 | 1                            | lanegroup_m_per_wave()       | same as C
    level_2 | 1                            | waves_per_m()                | same as C
    level_3 | lanegroup_repeat_m           | 1                            | same as C

    n_dim   | thread_length                | cluster_length
    --------+------------------------------+------------------------------+
    level_0 | thread_n()                   | lanegroup_size_n(), 8        |
    level_1 | 1                            | lanegroup_n_per_wave()       | same as C
    level_2 | 1                            | waves_per_n()                | same as C
    level_3 | lanegroup_repeat_n           | 1                            | same as C

    k_dim   | thread_length                | cluster_length
    --------+------------------------------+------------------------------+
    level_0 | lanegroup_k_per_thread()     | 1                            |

    NOTE:
    we can still pack k inside LDS
    e.g.
    v_dot2c_f32_f16, lanegroup_k_per_thread()=2
    but we may prefer locate 8 continus k in LDS, to form a single ds_read_b128
    we may still have thread_m() = 1, which means we only have single m, but intead 8 of k

    '''
    def __init__(self, macro_tile_m, macro_tile_n, lanegroup_tile_m, lanegroup_tile_n, 
                        lanegroup_wave_m, lanegroup_wave_n, waves, lanegroup_repeat_m, lanegroup_repeat_n, inst_dotx):
        assert lanegroup_tile_m % LANEGROUP_SIZE == 0 and lanegroup_tile_n % LANEGROUP_SIZE == 0
        wave_size = lanegroup_wave_m * lanegroup_wave_n * LANEGROUP_SIZE
        assert wave_size in (32, 64), f'unsupported wave size config:{wave_size}, wl_m:{lanegroup_wave_m}, wl_n:{lanegroup_wave_n}'
        assert macro_tile_m % (lanegroup_repeat_m * lanegroup_tile_m * lanegroup_wave_m) == 0
        assert macro_tile_n % (lanegroup_repeat_n * lanegroup_tile_n * lanegroup_wave_n) == 0

        self.macro_tile_m = macro_tile_m
        self.macro_tile_n = macro_tile_n
        self.lanegroup_tile_m = lanegroup_tile_m
        self.lanegroup_tile_n = lanegroup_tile_n
        self.lanegroup_wave_m = lanegroup_wave_m
        self.lanegroup_wave_n = lanegroup_wave_n
        self.waves = waves
        self.lanegroup_repeat_m = lanegroup_repeat_m
        self.lanegroup_repeat_n = lanegroup_repeat_n
        self.inst_dotx = inst_dotx

        self.macro_tile_validate()
        self.wave_tile_validate()

    def wave_size(self):
        return self.lanegroup_wave_m * self.lanegroup_wave_n * LANEGROUP_SIZE

    def acc_c_lengths(self):
        '''
        return agpr lengths for each dimension. from right to left is agpr index increase
        '''      
        t_mr, t_nr, t_nt, t_mt = self.lanegroup_repeat_m, self.lanegroup_repeat_n, \
                                    self.lanegroup_n_per_thread(), self.lanegroup_m_per_thread()
        return t_mr, t_nr, t_nt, t_mt

    def acc_c_per_thread_n(self):
        t_nr, t_nt = self.lanegroup_repeat_n, self.lanegroup_n_per_thread()
        return t_nr, t_nt

    def acc_c_per_thread_m(self):
        t_mr, t_mt = self.lanegroup_repeat_m, self.lanegroup_m_per_thread()
        return t_mr, t_mt

    def total_acc_c(self):
        def flatten(x):
            from functools import reduce
            return reduce(lambda a, b: a*b, x, 1)
        assert flatten(self.acc_c_per_thread_m()) * flatten(self.acc_c_per_thread_n()) == flatten(self.acc_c_lengths())
        return flatten(self.acc_c_lengths())

    def lanegroup_m_per_wave(self):
        '''
        how many lanegroups to form a single wave, in m direction
        '''
        return self.lanegroup_wave_m

    def lanegroup_n_per_wave(self):
        '''
        how many lanegroups to form a single wave, in n direction
        '''
        return self.lanegroup_wave_n

    def waves_per_m(self):
        ''' attention! not count repeat'''
        return self.macro_tile_m // (self.lanegroup_repeat_m * self.lanegroup_tile_m * self.lanegroup_wave_m)

    def waves_per_n(self):
        ''' attention! not count repeat'''
        return self.macro_tile_n // (self.lanegroup_repeat_n * self.lanegroup_tile_n * self.lanegroup_wave_n)

    def block_size(self):
        return self.waves * self.wave_size()

    def lanegroup_m_per_thread(self):
        ''' [within thread] need be 8x '''
        return self.lanegroup_tile_m

    def lanegroup_n_per_thread(self):
        ''' [within thread] '''
        return self.lanegroup_tile_n // LANEGROUP_SIZE

    def lanegroup_m_per_cluster(self):
        return 1

    def lanegroup_n_per_cluster(self):
        return LANEGROUP_SIZE

    def lanegroup_k_per_thread(self):
        return self.inst_dotx.k

    def lanegroup_size_m(self):
        ''' '''
        return LANEGROUP_SIZE
    
    def lanegroup_size_n(self):
        return LANEGROUP_SIZE

    def thread_m(self):
        return self.lanegroup_tile_m // self.lanegroup_size_m()

    def thread_n(self):
        return self.lanegroup_tile_n // self.lanegroup_size_n()

    def macro_tile_validate(self):
        assert self.macro_tile_m == self.lanegroup_m_per_thread() * self.lanegroup_m_per_cluster() * self.lanegroup_m_per_wave() * self.waves_per_m() * self.lanegroup_repeat_m
        assert self.macro_tile_n == self.lanegroup_n_per_thread() * self.lanegroup_n_per_cluster() * self.lanegroup_n_per_wave() * self.waves_per_n() * self.lanegroup_repeat_n
        assert self.macro_tile_m * self.macro_tile_n == self.block_size() * self.lanegroup_m_per_thread() * self.lanegroup_repeat_m * self.lanegroup_n_per_thread() * self.lanegroup_repeat_n
        assert self.macro_tile_m == self.thread_m() * self.lanegroup_size_m() * self.lanegroup_m_per_wave() * self.waves_per_m() * self.lanegroup_repeat_m
        assert self.macro_tile_n == self.thread_n() * self.lanegroup_size_n() * self.lanegroup_n_per_wave() * self.waves_per_n() * self.lanegroup_repeat_n

    def wave_tile_validate(self):
        wave_tile_m = self.lanegroup_m_per_thread() * self.lanegroup_m_per_cluster() * self.lanegroup_m_per_wave()
        wave_tile_n = self.lanegroup_n_per_thread() * self.lanegroup_n_per_cluster() * self.lanegroup_n_per_wave()
        assert wave_tile_m == self.macro_tile_m // (self.waves_per_m() * self.lanegroup_repeat_m)
        assert wave_tile_n == self.macro_tile_n // (self.waves_per_n() * self.lanegroup_repeat_n)
        assert wave_tile_m == self.thread_m() * self.lanegroup_size_m() * self.lanegroup_m_per_wave()
        assert wave_tile_n == self.thread_n() * self.lanegroup_size_n() * self.lanegroup_n_per_wave()

    def serialize(self):
        self.macro_tile_validate()
        self.wave_tile_validate()
        s  = f"c_m:{self.lanegroup_m_per_thread()}x{self.lanegroup_m_per_cluster()}-1x{self.lanegroup_m_per_wave()}-1x{self.waves_per_m()}-{self.lanegroup_repeat_m}x1, "
        s += f"c_n:{self.lanegroup_n_per_thread()}x{self.lanegroup_n_per_cluster()}-1x{self.lanegroup_n_per_wave()}-1x{self.waves_per_n()}-{self.lanegroup_repeat_n}x1, "
        s += f"a_m:{self.thread_m()}x{self.lanegroup_size_m()}, a_n:{self.thread_n()}x{self.lanegroup_size_n()}, a_k:{self.lanegroup_k_per_thread()}x1"
        return s


    # def extra_k_pack_list(self, max_bytes = 4 * 4):
    #     '''
    #     return a list contains all the possible values that can be used as k_pack, based on max_bytes
    #     '''
    #     data_byte = amdgpu_precision_data_byte(self.inst_dotx.data_type)
    #     packed_pixel_byte = self.inst_dotx.k * data_byte
    #     extra_k_pack = 1

                        #  mt_m,mt_n,lt_m,lt_n,lw_m,lw_n,  ws,lr_m,lr_n, inst_mfma
ctrl_dotx_mapping_fp16 = [
        ctrl_dotx_mapping_t(256, 128,   8,   8,   4,   2,   4,   2,   8, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t(128, 256,   8,   8,   2,   4,   4,   2,   8, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        
        ctrl_dotx_mapping_t(128, 192,   8,   8,   2,   4,   4,   2,   6, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t(192, 128,   8,   8,   2,   4,   4,   3,   4, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        
        #ctrl_dotx_mapping_t(128, 144,   8,   8,   2,   2,   4,   2,   9, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        #ctrl_dotx_mapping_t(144, 128,   8,   8,   2,   2,   4,   9,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        
        ctrl_dotx_mapping_t(128, 128,   8,   8,   2,   4,   4,   2,   4, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4      
        ctrl_dotx_mapping_t(128, 128,   8,   8,   4,   2,   4,   2,   4, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        # ctrl_dotx_mapping_t(128, 128,   8,   8,   2,   4,   4,   4,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4 # wrong case
        # ctrl_dotx_mapping_t(128, 128,  16,  16,   2,   4,   4,   2,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2
        # ctrl_dotx_mapping_t(128, 128,  16,  16,   4,   2,   4,   1,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2

        ctrl_dotx_mapping_t(128,  96,   8,   8,   4,   2,   4,   2,   3, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t( 96, 128,   8,   8,   4,   2,   4,   3,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4

        ctrl_dotx_mapping_t(128,  64,   8,   8,   4,   2,   4,   2,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        #ctrl_dotx_mapping_t(128,  64,  16,  16,   4,   2,   4,   1,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2
        ctrl_dotx_mapping_t( 64, 128,   8,   8,   2,   4,   4,   2,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        #ctrl_dotx_mapping_t( 64, 128,  16,  16,   2,   4,   4,   1,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2

        ctrl_dotx_mapping_t(128,  32,   8,   8,   4,   1,   4,   2,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t( 32, 128,   8,   8,   1,   4,   4,   2,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        #ctrl_dotx_mapping_t(128,  32,   8,   8,   4,   2,   4,   2,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        #ctrl_dotx_mapping_t( 32, 128,   8,   8,   2,   4,   4,   1,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4

        ctrl_dotx_mapping_t( 96,  96,   8,   8,   2,   2,   4,   3,   3, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4

        ctrl_dotx_mapping_t( 64,  64,   8,   8,   2,   2,   4,   2,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        #ctrl_dotx_mapping_t( 64,  64,   8,   8,   4,   2,   4,   1,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        #ctrl_dotx_mapping_t( 64,  64,   8,   8,   2,   4,   4,   2,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4

        ctrl_dotx_mapping_t( 64,  32,   8,   8,   4,   2,   1,   2,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        #ctrl_dotx_mapping_t( 64,  32,   8,   8,   2,   2,   4,   2,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        #ctrl_dotx_mapping_t( 64,  32,   8,   8,   4,   2,   4,   1,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t( 32,  64,   8,   8,   2,   4,   1,   2,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        #ctrl_dotx_mapping_t( 32,  64,   8,   8,   2,   2,   4,   1,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        #ctrl_dotx_mapping_t( 32,  64,   8,   8,   2,   4,   4,   1,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        
        ctrl_dotx_mapping_t( 32,  32,   8,   8,   2,   2,   1,   2,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
    ]

def get_ctrl_dotx_mapping_from_wave_tile(macro_tile_m, macro_tile_n, lanegroup_tile_m, lanegroup_tile_n, lanegroup_wave_m, lanegroup_wave_n, waves, lanegroup_repeat_m, lanegroup_repeat_n, precision):
    if type(precision) is str:
        precision = amdgpu_string_to_precision(precision)
    
    if precision == AMDGPU_PRECISION_FP16:
        ctrl_dotx_mapping = ctrl_dotx_mapping_fp16
    else:
        assert False, f"wrong data type"

    target_mfma_tiling = list()
    for t in ctrl_dotx_mapping:
        if t.macro_tile_m == macro_tile_m and \
            t.macro_tile_n == macro_tile_n and \
            t.lanegroup_tile_m == lanegroup_tile_m and \
            t.lanegroup_tile_n == lanegroup_tile_n and \
            t.lanegroup_wave_m == lanegroup_wave_m and \
            t.lanegroup_wave_n == lanegroup_wave_n and \
            t.waves == waves and \
            t.lanegroup_repeat_m == lanegroup_repeat_m and \
            t.lanegroup_repeat_n == lanegroup_repeat_n:
            target_mfma_tiling.append(t)

    assert len(target_mfma_tiling) != 0, f"unsupported macro_tile_m:{macro_tile_m}, macro_tile_n:{macro_tile_n}, lanegroup_tile_m:{lanegroup_tile_m}, lanegroup_tile_n:{lanegroup_tile_n}, lanegroup_repeat_m:{lanegroup_repeat_m}, lanegroup_repeat_n:{lanegroup_repeat_n}, "
    # TODO: we may have multiple match, aka multipl wave mapping/dotx for single 
    return target_mfma_tiling[0]

class igemm_dotx_mapping_t(mc_base_t):
    def name(self):
        return ''
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_dotx_mapping_t
        self.ctrl = ctrl
    def get_gemm_index_for_src_matrix(self, v_gemm_in, v_gemm_im, v_thread_id, v_tmp4, **options):
        '''
        notice! this is to calculate LDS offset for A/B matrix input, it is not the same as C matrix output layout, due to dotx
        C matrix output describe is in coalescint_storec
        '''
        def get_dict_with_default(some_dict, key, default_value):
            if key in some_dict:
                return some_dict[key]
            return default_value
        ctrl = self.ctrl

        assert ctrl.lanegroup_size_n() == ctrl.lanegroup_size_m()
        k_pack = get_dict_with_default(options, "k_pack", 1)
        v_pack = get_dict_with_default(options, "v_pack", 1)
        assert v_pack in (1, k_pack),  'currently only support v_pack is 1 or k_pack'
        if k_pack != 1:
            assert k_pack % ctrl.lanegroup_k_per_thread() == 0, f'inst:{ctrl.inst_mfma.name()} require k_pack:{k_pack} since lanegroup_k_per_thread:{ctrl.lanegroup_k_per_thread()}'
            # assert k_pack % ctrl.inst_mfma.num_v_a == 0 and k_pack % ctrl.inst_mfma.num_v_b == 0, f'inst:{ctrl.inst_mfma.name()} require k_pack since num_v_a:{ctrl.inst_mfma.num_v_a}, num_v_b:{ctrl.inst_mfma.num_v_b}'
            k_pack_per_thread = k_pack // ctrl.lanegroup_k_per_thread()
        else:
            k_pack_per_thread = 1
        with self._deferred_context():
            self._emit(f"; dotx mapping, get source matrix gemm index, k_pack:{k_pack}, v_pack:{v_pack}, k_pack_per_thread:{k_pack_per_thread}")
            self._emit(f"v_and_b32 v[{v_gemm_in}], {ctrl.lanegroup_size_n() - 1}, v[{v_thread_id}]           ; lanegroup_n index ")
            self._emit(f"v_and_b32 v[{v_gemm_im}], {ctrl.lanegroup_size_m() - 1}, v[{v_thread_id}]           ; lanegroup_m index ")

            self._emit(f"v_lshrrev_b32 v[{v_thread_id}], {utility_log2(ctrl.lanegroup_size_n())}, v[{v_thread_id}]")
            if ctrl.lanegroup_n_per_wave() != 1:
                self._emit(f"v_and_b32 v[{v_tmp4} + 0], {ctrl.lanegroup_n_per_wave() - 1}, v[{v_thread_id}]          ; lanegroup_n_per_wave index")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_in}], v[{v_tmp4} + 0], {utility_log2(ctrl.lanegroup_size_n())}, v[{v_gemm_in}]")
                self._emit(f"v_lshrrev_b32 v[{v_thread_id}], {utility_log2(ctrl.lanegroup_n_per_wave())}, v[{v_thread_id}]")
            if ctrl.lanegroup_m_per_wave() != 1:
                self._emit(f"v_and_b32 v[{v_tmp4} + 1], {ctrl.lanegroup_m_per_wave() - 1}, v[{v_thread_id}]          ; lanegroup_m_per_wave index")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_im}], v[{v_tmp4} + 1], {utility_log2(ctrl.lanegroup_size_m())}, v[{v_gemm_im}]")
                self._emit(f"v_lshrrev_b32 v[{v_thread_id}], {utility_log2(ctrl.lanegroup_m_per_wave())}, v[{v_thread_id}]")
            if ctrl.waves_per_n() != 1:
                self._emit(f"v_and_b32 v[{v_tmp4} + 2], {ctrl.waves_per_n() - 1}, v[{v_thread_id}]  ; waves_per_n index")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_in}], v[{v_tmp4} + 2], {utility_log2(ctrl.lanegroup_size_n() * ctrl.lanegroup_n_per_wave())}, v[{v_gemm_in}]")
                self._emit(f"v_lshrrev_b32 v[{v_thread_id}], {utility_log2(ctrl.waves_per_n())}, v[{v_thread_id}]")
            if ctrl.waves_per_m() != 1:
                self._emit(f"v_and_b32 v[{v_tmp4} + 3], {ctrl.waves_per_m() - 1}, v[{v_thread_id}]  ; waves_per_m index")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_im}], v[{v_tmp4} + 3], {utility_log2(ctrl.lanegroup_size_m() * ctrl.lanegroup_m_per_wave())}, v[{v_gemm_im}]")
                # self._emit(f"v_lshrrev_b32 v[{v_thread_id}], {utility_log2(ctrl.waves_per_n())}, v[{v_thread_id}]")
            self._emit_empty_line()
        return self._get_deferred()

    def get_gemm_index_for_dst_matrix(self, v_gemm_in, v_gemm_im, v_thread_id, v_tmp4):
        class emit_pretty_split_accumulator_t(mc_base_t):
            def __init__(self, mc):
                mc_base_t.__init__(self, mc)
                self.first = 1
            def __call__(self, v_accumulator, v_idx, k_len, k_multiplier):
                assert utility_is_pow2(k_multiplier)
                if k_len == 1:
                    pass
                else:
                    if self.first == 1:
                        self.first = 0
                        if k_multiplier == 1:
                            self._emit(f"v_mov_b32 v[{v_accumulator}], v[{v_idx}]")
                        else:
                            self._emit(f"v_lshlrev_b32 v[{v_accumulator}], {utility_log2(k_multiplier)}, v[{v_idx}]")
                    else:
                        if k_multiplier == 1:
                            self._emit( f"v_add_u32 v[{v_accumulator}], v[{v_idx}], v[{v_accumulator}]")
                        else:
                            self._emit(f"v_lshl_or_b32 v[{v_accumulator}], v[{v_idx}], {utility_log2(k_multiplier)}, v[{v_accumulator}]")
        class emit_pretty_split_shift_t(mc_base_t):
            def __init__(self, mc):
                mc_base_t.__init__(self, mc)
            def __call__(self, v_idx, v_shifter, k_len, is_last = False):
                if k_len != 1:
                    self._emit(f"v_and_b32 v[{v_idx}], {k_len - 1}, v[{v_shifter}]")
                    if not is_last:
                        self._emit(f"v_lshrrev_b32 v[{v_shifter}], {utility_log2(k_len)}, v[{v_shifter}]")
                else:
                    pass
        ctrl = self.ctrl
        p_sac_m = emit_pretty_split_accumulator_t(self.mc)
        p_sac_n = emit_pretty_split_accumulator_t(self.mc)
        p_ssh = emit_pretty_split_shift_t(self.mc)

        with self._deferred_context():
            self._emit(f"; dotx mapping, get dst matrix gemm index")

            # first, calculate within each wave tile
            p_ssh(f"{v_tmp4}+0",  v_thread_id, ctrl.lanegroup_n_per_cluster())
            p_ssh(f"{v_tmp4}+1",  v_thread_id, ctrl.lanegroup_m_per_cluster())  # should always be 1
            p_ssh(f"{v_tmp4}+2",  v_thread_id, ctrl.lanegroup_n_per_wave())
            p_ssh(f"{v_tmp4}+3",  v_thread_id, ctrl.lanegroup_m_per_wave())

            p_sac_n(v_gemm_in, f"{v_tmp4}+0",  ctrl.lanegroup_n_per_cluster(), ctrl.lanegroup_n_per_thread())
            p_sac_n(v_gemm_in, f"{v_tmp4}+2",  ctrl.lanegroup_n_per_wave(), ctrl.lanegroup_n_per_cluster() * ctrl.lanegroup_n_per_thread())

            p_sac_m(v_gemm_im, f"{v_tmp4}+1",  ctrl.lanegroup_m_per_cluster(), ctrl.lanegroup_m_per_thread())
            p_sac_m(v_gemm_im, f"{v_tmp4}+3",  ctrl.lanegroup_m_per_wave(), ctrl.lanegroup_m_per_cluster() * ctrl.lanegroup_m_per_thread())

            # second, calculate among waves
            p_ssh(f"{v_tmp4}+0",  v_thread_id, ctrl.waves_per_n())   
            p_ssh(f"{v_tmp4}+1",  v_thread_id, ctrl.waves_per_m(), True)

            p_sac_n(v_gemm_in, f"{v_tmp4}+0",  ctrl.waves_per_n(),  ctrl.lanegroup_n_per_thread() * ctrl.lanegroup_n_per_cluster() * ctrl.lanegroup_n_per_wave())
            p_sac_m(v_gemm_im, f"{v_tmp4}+1",  ctrl.waves_per_m(),  ctrl.lanegroup_m_per_thread() * ctrl.lanegroup_m_per_cluster() * ctrl.lanegroup_m_per_wave())

            if p_sac_n.first == 1:
                self._emit(f"v_mov_b32 v[{v_gemm_in}], 0")
            if p_sac_m.first == 1:
                self._emit(f"v_mov_b32 v[{v_gemm_im}], 0")

            self._emit_empty_line()

        return self._get_deferred()

    def emit(self):
        assert False
