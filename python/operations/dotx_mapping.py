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
    level_3 | wave_repeat_m                | 1                            |

    n_dim   | thread_length                | cluster_length
    --------+------------------------------+------------------------------+
    level_0 | lanegroup_n_per_thread()     | lanegroup_n_per_cluster(), 8 | -> lanegroup_tile_n
    level_1 | 1                            | lanegroup_n_per_wave()       |
    level_2 | 1                            | waves_per_n()                |
    level_3 | wave_repeat_n                | 1                            |

    -----------------------------------------------------------------------------------

    A/B matrix:

    m_dim   | thread_length                | cluster_length
    --------+------------------------------+------------------------------+
    level_0 | thread_m()                   | lanegroup_size_m(), 8        |
    level_1 | 1                            | lanegroup_m_per_wave()       | same as C
    level_2 | 1                            | waves_per_m()                | same as C
    level_3 | wave_repeat_m                | 1                            | same as C

    n_dim   | thread_length                | cluster_length
    --------+------------------------------+------------------------------+
    level_0 | thread_n()                   | lanegroup_size_n(), 8        |
    level_1 | 1                            | lanegroup_n_per_wave()       | same as C
    level_2 | 1                            | waves_per_n()                | same as C
    level_3 | wave_repeat_n                | 1                            | same as C

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
                        lanegroup_wave_m, lanegroup_wave_n, waves, wave_repeat_m, wave_repeat_n, inst_dotx):
        assert lanegroup_tile_m % LANEGROUP_SIZE == 0 and lanegroup_tile_n % LANEGROUP_SIZE == 0
        wave_size = lanegroup_wave_m * lanegroup_wave_n * LANEGROUP_SIZE
        assert wave_size in (32, 64), f'unsupported wave size config:{wave_size}, wl_m:{lanegroup_wave_m}, wl_n:{lanegroup_wave_n}'
        assert macro_tile_m % (wave_repeat_m * lanegroup_tile_m * lanegroup_wave_m) == 0
        assert macro_tile_n % (wave_repeat_n * lanegroup_tile_n * lanegroup_wave_n) == 0

        self.macro_tile_m = macro_tile_m
        self.macro_tile_n = macro_tile_n
        self.lanegroup_tile_m = lanegroup_tile_m
        self.lanegroup_tile_n = lanegroup_tile_n
        self.lanegroup_wave_m = lanegroup_wave_m
        self.lanegroup_wave_n = lanegroup_wave_n
        self.waves = waves
        self.wave_repeat_m = wave_repeat_m
        self.wave_repeat_n = wave_repeat_n
        self.inst_dotx = inst_dotx

        self.macro_tile_validate()

    def acc_c_lengths(self):
        '''
        return agpr lengths for each dimension. from right to left is agpr index increase
        '''      
        t_mr, t_nr, t_nt, t_mt = self.wave_repeat_m, self.wave_repeat_n, \
                                    self.lanegroup_n_per_thread(), self.lanegroup_m_per_thread()
        return t_mr, t_nr, t_nt, t_mt

    def acc_c_per_thread_n(self):
        t_nr, t_nt = self.wave_repeat_n, self.lanegroup_n_per_thread()
        return t_nr, t_nt

    def acc_c_per_thread_m(self):
        t_mr, t_mt = self.wave_repeat_m, self.lanegroup_m_per_thread()
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
        return self.macro_tile_m // (self.wave_repeat_m * self.lanegroup_tile_m * self.lanegroup_wave_m)

    def waves_per_n(self):
        ''' attention! not count repeat'''
        return self.macro_tile_n // (self.wave_repeat_n * self.lanegroup_tile_n * self.lanegroup_wave_n)

    def block_size(self):
        wave_size = self.lanegroup_wave_m * self.lanegroup_wave_n * LANEGROUP_SIZE
        return self.waves * wave_size

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
        return self.lanegroup_m_per_thread() // self.lanegroup_size_m()

    def thread_n(self):
        return self.lanegroup_n_per_thread() // self.lanegroup_size_n()

    def macro_tile_validate(self):
        assert self.macro_tile_m == self.lanegroup_m_per_thread() * self.lanegroup_m_per_cluster() * self.lanegroup_m_per_wave() * self.waves_per_m() * self.wave_repeat_m
        assert self.macro_tile_n == self.lanegroup_n_per_thread() * self.lanegroup_n_per_cluster() * self.lanegroup_n_per_wave() * self.waves_per_n() * self.wave_repeat_n
        assert self.macro_tile_m * self.macro_tile_n == self.block_size() * self.lanegroup_m_per_thread() * self.wave_repeat_m * self.lanegroup_n_per_thread() * self.wave_repeat_n

    def serialize(self):
        self.macro_tile_validate()
        s  = f"c_m:{self.lanegroup_m_per_thread()}x{self.lanegroup_m_per_cluster()}-1x{self.lanegroup_m_per_wave()}-1x{self.waves_per_m()}-{self.wave_repeat_m}x1, "
        s += f"c_n:{self.lanegroup_n_per_thread()}x{self.lanegroup_n_per_cluster()}-1x{self.lanegroup_n_per_wave()}-1x{self.waves_per_n()}-{self.wave_repeat_n}x1, "
        s += f"a_m:{self.thread_m()}x{self.lanegroup_size_m()}, a_n:{self.thread_n()}x{self.lanegroup_size_n()}, a_k:{self.lanegroup_k_per_thread()}x1"
        return s


    # def extra_k_pack_list(self, max_bytes = 4 * 4):
    #     '''
    #     return a list contains all the possible values that can be used as k_pack, based on max_bytes
    #     '''
    #     data_byte = amdgpu_precision_data_byte(self.inst_dotx.data_type)
    #     packed_pixel_byte = self.inst_dotx.k * data_byte
    #     extra_k_pack = 1

                        #  mt_m,mt_n,lt_m,lt_n,lw_m,lw_n,  ws, r_m, r_n, inst_mfma
ctrl_dotx_mapping_fp16 = [
        ctrl_dotx_mapping_t(128, 128,   8,   8,   4,   2,   4,   2,   4, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t(128, 128,   8,   8,   2,   4,   4,   4,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t(128, 128,  16,  16,   2,   4,   4,   2,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2
        ctrl_dotx_mapping_t(128, 128,  16,  16,   4,   2,   4,   1,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2

        ctrl_dotx_mapping_t(128,  96,   8,   8,   4,   2,   4,   2,   3, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t( 96, 128,   8,   8,   2,   4,   4,   3,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4

        ctrl_dotx_mapping_t(128,  64,   8,   8,   4,   2,   4,   2,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t(128,  64,  16,  16,   4,   2,   4,   1,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2
        ctrl_dotx_mapping_t( 64, 128,   8,   8,   2,   4,   4,   2,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t( 64, 128,  16,  16,   2,   4,   4,   1,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2

        ctrl_dotx_mapping_t(128,  32,   8,   8,   4,   2,   4,   2,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t( 32, 128,   8,   8,   2,   4,   4,   1,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4

        ctrl_dotx_mapping_t( 96,  96,   8,   8,   2,   2,   4,   3,   3, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4

        ctrl_dotx_mapping_t( 64,  64,   8,   8,   2,   2,   4,   2,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t( 64,  64,   8,   8,   4,   2,   4,   1,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t( 64,  64,   8,   8,   2,   4,   4,   2,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4

        ctrl_dotx_mapping_t( 64,  32,   8,   8,   2,   2,   4,   2,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t( 64,  32,   8,   8,   4,   2,   4,   1,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t( 32,  64,   8,   8,   2,   2,   4,   1,   2, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
        ctrl_dotx_mapping_t( 32,  64,   8,   8,   2,   4,   4,   1,   1, v_dot2c_f32_f16),  # extra k pack can be 1, 2, 4
    ]
