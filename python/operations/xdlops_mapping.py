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
from .mfma import *

class ctrl_xdlops_mapping_t(object):
    '''
    xdlops mapping for output C matrix, forms a 6-d mapping of thread-lanegroup-block-wave-macro_tile
    each dimension is composed within one thread(thread_length), or composed by several thread(cluster_length)

    m_dim   | thread_length                | cluster_length
    --------+------------------------------+-----------------------------+
    level_0 | lanegroup_m_per_thread(), 4  | lanegroup_m_per_cluster()   |
    level_1 | lanegroup_m_per_block()      | block_m_per_lanegroup()     |
    level_2 | lanegroup_m_per_wave()       | 1                           |
    level_3 | wave_step_m                  | 1                           |
    level_4 | 1                            | waves_per_m()               |
    level_5 | wave_repeat_m                | 1                           |

    n_dim   | thread_length                | cluster_length
    --------+------------------------------+-----------------------------+
    level_0 | lanegroup_n_per_thread(), 1  | lanegroup_n_per_cluster(), n|
    level_1 | lanegroup_n_per_block(), 1   | block_n_per_lanegroup()     |
    level_2 | lanegroup_n_per_wave()       | 1                           |
    level_3 | wave_step_n                  | 1                           |
    level_4 | 1                            | waves_per_n()               |
    level_5 | wave_repeat_n                | 1                           |

    so consider per_thread order:

    V   |           m               |         n
        +---------------------------+---------------------------+
        | wave_repeat_m             | _                         | => level_5, m
        | _                         | wave_repeat_n             | => level_5, n
        | wave_step_m               | _                         | => level_3, m
        | _                         | wave_step_n               | => level_3, n
        | lanegroup_m_per_wave()    | _                         | => level_2, m
        | _                         | lanegroup_n_per_wave()    | => level_2, n
        | lanegroup_m_per_block()   | _                         | => level_1, m
        | lanegroup_m_per_thread()  | _                         | => level_0, m

    level_0     describe lanegroup level
    level_0/1/2 describe a single xdlops, that form a wave-tile
    level_3/4/5 describe macro-tile, formed by wave-tile

    -----------------------------------------------------------------------------------

    xdlops for input A/B matrix, is simpler

    m_dim   | thread_length                | cluster_length
    --------+------------------------------+-----------------------------+
    level_0 | 1                            | block_m()                   |
    level_1 | 1                            | block_m_per_wave()          |
    level_2 | wave_step_m                  | 1                           |
    level_3 | 1                            | waves_per_m()               |
    level_4 | wave_repeat_m                | 1                           |


    n_dim   | thread_length                | cluster_length
    --------+------------------------------+-----------------------------+
    level_0 | 1                            | block_n()                   |
    level_1 | 1                            | block_n_per_wave()          |
    level_2 | wave_step_n                  | 1                           |
    level_3 | 1                            | waves_per_n()               |
    level_4 | wave_repeat_n                | 1                           |

    level_0/1   describe wave-tile level
    level_2/3/4 describte macro-tile level, same as matric C layout above

    k_dim   | thread_length                | cluster_length
    --------+------------------------------+-----------------------------+
    level_0 | lanegroup_k_per_thread()     | 1                           |
    level_1 | 1                            | block_k()                   |

    '''
    def __init__(self, macro_tile_m, macro_tile_n, wave_tile_m, wave_tile_n, wave_tile_k, waves, wave_repeat_m, wave_repeat_n, wave_step_m, wave_step_n, inst_mfma):
        self.macro_tile_m = macro_tile_m
        self.macro_tile_n = macro_tile_n
        self.wave_tile_m = wave_tile_m
        self.wave_tile_n = wave_tile_n
        self.wave_tile_k = wave_tile_k
        self.waves = waves
        self.wave_repeat_m = wave_repeat_m
        self.wave_repeat_n = wave_repeat_n
        self.wave_step_m = wave_step_m
        self.wave_step_n = wave_step_n
        self.inst_mfma = inst_mfma

    def acc_c_lengths(self):
        '''
        return agpr lengths for each dimension. from right to left is agpr index increase
        '''      
        t_mr, t_nr, t_ms, t_ns, t_mw, t_nw, t_mb, t_mt = self.wave_repeat_m, self.wave_repeat_n, self.wave_step_m, self.wave_step_n,\
                                            self.lanegroup_m_per_wave(), self.lanegroup_n_per_wave(),\
                                            self.lanegroup_m_per_block(), self.lanegroup_m_per_thread()
        return  t_mr, t_nr, t_ms, t_ns, t_mw, t_nw, t_mb, t_mt

    def composed_wave_tile_m(self):
        return self.wave_tile_m * self.wave_step_m

    def composed_wave_tile_n(self):
        return self.wave_tile_n * self.wave_step_n

    def waves_per_m(self):
        ''' attention! not count repeat'''
        return self.macro_tile_m // (self.wave_repeat_m * self.wave_tile_m * self.wave_step_m)

    def waves_per_n(self):
        ''' attention! not count repeat'''
        return self.macro_tile_n // (self.wave_repeat_n * self.wave_tile_n * self.wave_step_n)

    def total_acc_c(self):
        def flatten(x):
            from functools import reduce
            return reduce(lambda a, b: a*b, x, 1)
        total_c = self.wave_repeat_m * self.wave_repeat_n * self.wave_step_m * self.wave_step_n * self.inst_mfma.num_a_c
        assert total_c == flatten(self.acc_c_per_thread_m()) * flatten(self.acc_c_per_thread_n())
        assert total_c == flatten(self.acc_c_lengths())
        return total_c

    def acc_c_per_thread_n(self):
        t_nr, t_ns, t_nw, t_nb, t_nt = self.wave_repeat_n, self.wave_step_n, self.lanegroup_n_per_wave(), self.lanegroup_n_per_block(), self.lanegroup_n_per_thread()
        return t_nr, t_ns, t_nw, t_nb, t_nt

    def acc_c_per_thread_m(self):
        t_mr, t_ms, t_mw, t_mb, t_mt = self.wave_repeat_m, self.wave_step_m, self.lanegroup_m_per_wave(), self.lanegroup_m_per_block(), self.lanegroup_m_per_thread()
        return t_mr, t_ms, t_mw, t_mb, t_mt

    def block_size(self):
        return self.waves * 64 # wave size 64

    def wave_tile_validate(self):
        assert self.wave_tile_m == self.lanegroup_m_per_thread() * self.lanegroup_m_per_cluster() * self.lanegroup_m_per_block() * self.block_m_per_lanegroup() * self.lanegroup_m_per_wave()
        assert self.wave_tile_n == self.lanegroup_n_per_thread() * self.lanegroup_n_per_cluster() * self.lanegroup_n_per_block() * self.block_n_per_lanegroup() * self.lanegroup_n_per_wave()
        assert self.lanegroup_m_per_cluster() * self.block_m_per_lanegroup() * self.lanegroup_n_per_cluster() * self.block_n_per_lanegroup() == AMDGPU_WAVE_SIZE
        assert self.block_m() * self.block_m_per_wave() == self.wave_tile_m and self.block_n() * self.block_n_per_wave() == self.wave_tile_n
        assert self.block_n() == self.block_m() and self.block_k_per_wave() * self.block_n() * self.block_n_per_wave() * self.block_m_per_wave() == AMDGPU_WAVE_SIZE

    def macro_tile_validate(self):
        assert self.macro_tile_m == self.wave_tile_m * self.wave_step_m * self.wave_repeat_m * self.waves_per_m()
        assert self.macro_tile_n == self.wave_tile_n * self.wave_step_n * self.wave_repeat_n * self.waves_per_n()

    def lanegroup_validate(self):
        assert self.lanegroup_n_per_thread() * self.lanegroup_n_per_block() * self.lanegroup_n_per_wave() * \
                self.lanegroup_m_per_thread() * self.lanegroup_m_per_block() * self.lanegroup_m_per_wave() \
                == self.inst_mfma.num_a_c

    def block_m(self):
        return self.inst_mfma.m

    def block_n(self):
        return self.inst_mfma.n

    def block_k(self):
        return self.inst_mfma.k

    # lanegroup layout for a single xdlops issue
    # each lanegroup is a 4x64 matrix, and expand into whole block, then wave
    # hence we group xdlops layout into 4 levels: per_thread->per_cluster->per_block->per_wave
    #
    def block_m_per_wave(self):
        ''' [among different thread] '''
        assert self.wave_tile_m % (self.lanegroup_m_per_thread() * self.lanegroup_m_per_cluster() * self.lanegroup_m_per_block()) == 0
        assert self.inst_mfma.m == self.lanegroup_m_per_thread() * self.lanegroup_m_per_cluster() * self.lanegroup_m_per_block()
        #print(f"wave_tile_m={self.wave_tile_m}, waves={self.waves}") 
        #print(f"lanegroup_m_per_cluster={self.lanegroup_m_per_cluster()}, lanegroup_n_per_cluster={self.lanegroup_n_per_cluster()}")
        #print(f"lanegroup_m_per_block={self.lanegroup_m_per_block()}, lanegroup_n_per_block={self.lanegroup_n_per_block()}")
        #print(f"lanegroup_m_per_wave={self.lanegroup_m_per_wave()}, lanegroup_n_per_wave={self.lanegroup_n_per_wave()}")
        return self.wave_tile_m // (self.lanegroup_m_per_thread() * self.lanegroup_m_per_cluster() * self.lanegroup_m_per_block()) 

    def block_n_per_wave(self):
        ''' [among different thread] '''
        assert self.wave_tile_n % (self.lanegroup_n_per_thread() * self.lanegroup_n_per_cluster() * self.lanegroup_n_per_block()) == 0
        assert self.inst_mfma.n == self.lanegroup_n_per_thread() * self.lanegroup_n_per_cluster() * self.lanegroup_n_per_block()
        return self.wave_tile_n // (self.lanegroup_n_per_thread() * self.lanegroup_n_per_cluster() * self.lanegroup_n_per_block())

    def block_k_per_wave(self):
        assert self.block_k() % self.lanegroup_k_per_thread() == 0
        return self.block_k() // self.lanegroup_k_per_thread()

    def block_m_per_lanegroup(self):
        ''' [among different thread] '''
        assert self.block_m_per_wave() % self.lanegroup_m_per_wave() == 0
        return self.block_m_per_wave() // self.lanegroup_m_per_wave()

    def block_n_per_lanegroup(self):
        ''' [among different thread] '''
        assert self.block_n_per_wave() % self.lanegroup_n_per_wave() == 0
        return self.block_n_per_wave() // self.lanegroup_n_per_wave()

    def lanegroup_m_per_thread(self):
        ''' [within thread] for xdlops, always continuous 4 agpr for a c matrix, then interleave with other lanegroup '''
        return 4

    def lanegroup_n_per_thread(self):
        ''' [within thread] for xdlops, always 1 column per lanegroup'''
        return 1

    def lanegroup_k_per_thread(self):
        ''' [within thread] for xdlops, 
            fp32 1/
            fp16 4/
            bf16 2
            columns per lanegroup'''
        if self.inst_mfma.data_type == AMDGPU_PRECISION_FP32:
            return 1
        if self.inst_mfma.data_type == AMDGPU_PRECISION_FP16:
            return 4
        if self.inst_mfma.data_type == AMDGPU_PRECISION_BF16:
            if 'bf16_1k' in self.inst_mfma.options and self.inst_mfma.options['bf16_1k']:
                return 4
            else:
                return 2
        if self.inst_mfma.data_type == AMDGPU_PRECISION_INT8:
            return 4
        assert False

    def lanegroup_m_per_cluster(self):
        ''' [among different thread] for xdlops, always m per block as clusters. perthread agpr do not contain this'''
        return utility_gcd(self.inst_mfma.m//self.lanegroup_m_per_thread(), AMDGPU_WAVE_SIZE // self.inst_mfma.n )

    def lanegroup_n_per_cluster(self):
        ''' [among different thread] for xdlops, always n per block as clusters. perthread agpr do not contain this'''
        return self.inst_mfma.n

    def lanegroup_m_per_block(self):
        ''' [within thread]  '''
        assert self.inst_mfma.m % (self.lanegroup_m_per_thread() * self.lanegroup_m_per_cluster()) == 0
        return self.inst_mfma.m // (self.lanegroup_m_per_thread() * self.lanegroup_m_per_cluster())

    def lanegroup_n_per_block(self):
        ''' [within thread]  '''
        return 1

    def lanegroup_m_per_wave(self):
        ''' [within thread] indeed descipbe agpr per thread within different blocks to form a wave tile '''
        assert self.inst_mfma.num_a_c % (self.lanegroup_n_per_thread() * self.lanegroup_n_per_block() * self.lanegroup_n_per_wave() * self.lanegroup_m_per_thread() * self.lanegroup_m_per_block()) == 0
        return self.inst_mfma.num_a_c // (self.lanegroup_n_per_thread() * self.lanegroup_n_per_block() * self.lanegroup_n_per_wave() * self.lanegroup_m_per_thread() * self.lanegroup_m_per_block())

    def lanegroup_n_per_wave(self):
        ''' [within thread] indeed descipbe agpr per thread within different blocks to form a wave tile '''
        assert self.inst_mfma.num_a_c % (self.lanegroup_m_per_thread() * self.lanegroup_m_per_block()) == 0
        return utility_gcd(self.block_n_per_wave(), self.inst_mfma.num_a_c // (self.lanegroup_m_per_thread() * self.lanegroup_m_per_block()))

    def serialize(self):
        self.lanegroup_validate()
        self.wave_tile_validate()
        self.macro_tile_validate()
        s = f"mt_m:{self.macro_tile_m}, mt_n:{self.macro_tile_n}, wt_m:{self.wave_tile_m}, wt_n:{self.wave_tile_n}, ws:{self.waves}, r_m:{self.wave_repeat_m}, r_n:{self.wave_repeat_n}, s_m:{self.wave_step_m}, s_n:{self.wave_step_n} | "
        s += f"{self.inst_mfma.m}x{self.inst_mfma.n}x{self.inst_mfma.k}, " + \
                f"lanegroup_m_tcbw:{self.lanegroup_m_per_thread()}x{self.lanegroup_m_per_cluster()}x{self.lanegroup_m_per_block()}x{self.lanegroup_m_per_wave()}, " + \
                f"lanegroup_n_tcbw:{self.lanegroup_n_per_thread()}x{self.lanegroup_n_per_cluster()}x{self.lanegroup_n_per_block()}x{self.lanegroup_n_per_wave()}"
        # s += "\n" + f"   lanegroup_m_per_thread:{self.lanegroup_m_per_thread()}, lanegroup_m_per_cluster:{self.lanegroup_m_per_cluster()}, lanegroup_m_per_block:{self.lanegroup_m_per_block()}, block_m_per_lanegroup:{self.block_m_per_lanegroup()}, lanegroup_m_per_wave:{self.lanegroup_m_per_wave()}"
        # s += "\n" + f"   lanegroup_n_per_thread:{self.lanegroup_n_per_thread()}, lanegroup_n_per_cluster:{self.lanegroup_n_per_cluster()}, lanegroup_n_per_block:{self.lanegroup_n_per_block()}, block_n_per_lanegroup:{self.block_n_per_lanegroup()}, lanegroup_n_per_wave:{self.lanegroup_n_per_wave()}"
        return s

#                             mt_m,mt_n,wt_m,wt_n,wt_k,ws,r_m,r_n,s_m,s_n, inst_mfma
ctrl_xdlops_mapping_fp32 = [
        ctrl_xdlops_mapping_t( 256, 256,  32,  32,  2, 4,  2,  2,  2,  2,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 256, 128,  64,  32,  1, 4,  2,  2,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 256, 128,  32,  32,  2, 4,  2,  2,  2,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 128, 256,  32,  64,  1, 4,  2,  2,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 128, 256,  32,  32,  2, 4,  2,  2,  1,  2,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 256, 64 ,  64,  16,  1, 4,  2,  2,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 256, 64 ,  32,  32,  2, 4,  2,  2,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 64 , 256,  16,  64,  1, 4,  2,  2,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 64 , 256,  32,  32,  2, 4,  2,  2,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 256, 32 ,  64,  4 ,  1, 4,  2,  2,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 256, 32 ,  32,  32,  2, 4,  2,  1,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 256, 32 ,  64,  32,  1, 2,  2,  1,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 32 , 256,  4 ,  64,  1, 4,  2,  2,  2,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 256,  32,  32,  2, 4,  1,  2,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 256, 16 ,  64,  4 ,  1, 4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 256,  4 ,  64,  1, 4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),

        #ctrl_xdlops_mapping_t( 256, 16 ,  64,  16,  2,  1,  1,  2,  1,  v_mfma_f32_16x16x1f32),     # TODO: this will fail in coalescing
        #ctrl_xdlops_mapping_t( 16 , 256,  16,  64,  2,  1,  1,  1,  1,  v_mfma_f32_16x16x1f32),     # TODO: this will fail in coalescing

        ctrl_xdlops_mapping_t( 128, 128,  32,  32,  1, 4,  2,  2,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 128, 128,  32,  32,  2, 4,  2,  2,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 128, 128,  32,  32,  2, 4,  1,  2,  1,  2,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 128, 128,  32,  64,  1, 4,  1,  1,  2,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 128, 128,  64,  32,  1, 4,  1,  1,  1,  2,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 128, 64 ,  32,  8 ,  1, 4,  2,  2,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 128, 64 ,  32,  32,  2, 4,  2,  1,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 128, 64 ,  32,  32,  2, 4,  1,  2,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 64 , 128,  8 ,  32,  1, 4,  2,  2,  2,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 128,  32,  64,  1, 4,  1,  1,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 64 , 128,  64,  32,  1, 4,  1,  1,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 64 , 128,  32,  32,  2, 4,  1,  2,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 128, 64 ,  32,  32,  2, 2,  2,  2,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 128, 64 ,  64,  32,  1, 2,  1,  2,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 128, 64 ,  64,  32,  1, 4,  1,  1,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 64 , 128,  32,  32,  2, 2,  2,  2,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 128, 64 ,  32,  32,  2, 1,  2,  2,  2,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 128, 32 ,  32,  8 ,  1, 4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 128, 32 ,  16,  16,  4, 4,  2,  2,  1,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 128, 32 ,  32,  32,  2, 4,  1,  1,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 128, 32 ,  32,  32,  2, 2,  2,  1,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 128, 32 ,  64,  16,  1, 4,  2,  2,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 128, 32 ,  64,  32,  1, 2,  1,  1,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 128, 32 ,  32,  32,  2, 4,  1,  1,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 32 , 128,  8 ,  32,  1, 4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 128,  16,  64,  1, 4,  1,  1,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 32 , 128,  16,  16,  4, 4,  2,  2,  1,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 32 , 128,  32,  32,  2, 2,  1,  2,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 32 , 128,  32,  32,  2, 4,  1,  1,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 64 , 64 ,  16,  16,  1, 4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 64 ,  16,  16,  4, 4,  2,  2,  1,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 64 , 64 ,  32,  32,  2, 4,  1,  1,  1,  1,  v_mfma_f32_32x32x2f32),  # this is not as good as 16x16x4
        #ctrl_xdlops_mapping_t( 128, 16 ,  64,  4 ,  4,  1,  1,  2,  1,  v_mfma_f32_4x4x1f32),
        #ctrl_xdlops_mapping_t( 16 , 128,  4 ,  64,  4,  1,  1,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 128, 16 ,  64,  16,  1, 2,  1,  1,  1,  1,  v_mfma_f32_16x16x1f32),  # need re-design coalescing. or do irregular gemm
        ctrl_xdlops_mapping_t( 128, 16 ,  16,  16,  4, 4,  2,  1,  1,  1,  v_mfma_f32_16x16x4f32),  # need re-design coalescing. or do irregular gemm
        ctrl_xdlops_mapping_t( 16 , 128,  16,  64,  1, 2,  1,  1,  1,  1,  v_mfma_f32_16x16x1f32),  # need re-design coalescing. or do irregular gemm
        ctrl_xdlops_mapping_t( 16 , 128,  16,  16,  4, 4,  1,  2,  1,  1,  v_mfma_f32_16x16x4f32),  # need re-design coalescing. or do irregular gemm
        ctrl_xdlops_mapping_t( 64 , 32 ,  32,  8 ,  1, 4,  1,  1,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 32 ,  16,  16,  4, 4,  2,  1,  1,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 64 , 32 ,  16,  16,  4, 4,  1,  2,  1,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 64 , 48 ,  16,  16,  4, 4,  1,  3,  1,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 64 , 32 ,  16,  16,  4, 4,  1,  1,  2,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 64 , 32 ,  32,  32,  2, 2,  1,  1,  1,  1,  v_mfma_f32_32x32x2f32),
        ctrl_xdlops_mapping_t( 32 , 64 ,  8 ,  32,  1, 4,  1,  1,  2,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 64 ,  16,  16,  4, 4,  1,  2,  1,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 32 , 32 ,  16,  16,  1, 4,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 32 ,  16,  16,  4, 4,  1,  1,  1,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 32 , 32 ,  16,  16,  4, 2,  1,  2,  1,  1,  v_mfma_f32_16x16x4f32),
        #ctrl_xdlops_mapping_t( 256, 4  ,  64,  4 ,  4,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),      # TODO: small/skinny gemm
        #ctrl_xdlops_mapping_t( 4  , 256,  4 ,  64,  4,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),      # TODO: small/skinny gemm
        ctrl_xdlops_mapping_t( 64 , 16 ,  64,  4 ,  1, 4,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 16 ,  16,  16,  4, 4,  1,  1,  1,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 64 , 16 ,  16,  16,  4, 2,  2,  1,  1,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 64 , 16 ,  64,  16,  1, 1,  1,  1,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 16 , 64 ,  4 ,  64,  1, 4,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 64 ,  16,  16,  4, 4,  1,  1,  1,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 16 , 64 ,  16,  16,  4, 2,  1,  2,  1,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 64 , 16 ,  64,  4 ,  1, 2,  1,  1,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 64 ,  4 ,  64,  1, 2,  1,  1,  2,  1,  v_mfma_f32_4x4x1f32),
        # 2waves, block_size=128
        #ctrl_xdlops_mapping_t( 128, 4  ,  64,  4 ,  2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),      # TODO: small/skinny gemm
        #ctrl_xdlops_mapping_t( 4  , 128,  4 ,  64,  2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),      # TODO: small/skinny gemm
        ctrl_xdlops_mapping_t( 64 , 8  ,  64,  4 ,  1, 2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 8  , 64 ,  4 ,  64,  1, 2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 16 ,  32,  8 ,  1, 2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 16 ,  16,  16,  4, 2,  1,  1,  1,  1,  v_mfma_f32_16x16x4f32),
        ctrl_xdlops_mapping_t( 16 , 32 ,  8 ,  32,  1, 2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 32 ,  16,  16,  4, 2,  1,  1,  1,  1,  v_mfma_f32_16x16x4f32),
        # 1 wave
        ctrl_xdlops_mapping_t( 32 , 16 ,  32,  8 ,  1, 1,  1,  1,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 32 ,  8 ,  32,  1, 1,  1,  1,  2,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 4  ,  64,  4 ,  1, 1,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 4  , 64 ,  4 ,  64,  1, 1,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 16 ,  16,  16,  1, 1,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 16 ,  16,  16,  4, 1,  1,  1,  1,  1,  v_mfma_f32_16x16x4f32)]

#                             mt_m,mt_n,wt_m,wt_n,wt_k,ws,r_m,r_n,s_m,s_n, inst_mfma
ctrl_xdlops_mapping_fp16 = [
        ctrl_xdlops_mapping_t( 256, 256,  64,  32,  4, 4,  2,  2,  1,  2,  v_mfma_f32_32x32x4f16),
        ctrl_xdlops_mapping_t( 256, 256,  32,  32,  8, 4,  2,  2,  2,  2,  v_mfma_f32_32x32x8f16),
        ctrl_xdlops_mapping_t( 256, 128,  64,  32,  4, 4,  2,  2,  1,  1,  v_mfma_f32_32x32x4f16),
        ctrl_xdlops_mapping_t( 256, 128,  32,  32,  8, 4,  2,  2,  2,  1,  v_mfma_f32_32x32x8f16),
        ctrl_xdlops_mapping_t( 128, 256,  32,  64,  4, 4,  2,  2,  1,  1,  v_mfma_f32_32x32x4f16),
        ctrl_xdlops_mapping_t( 128, 256,  32,  32,  8, 4,  2,  2,  1,  2,  v_mfma_f32_32x32x8f16),
        ctrl_xdlops_mapping_t( 256, 64 ,  64,  16,  4, 4,  2,  2,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 256, 64 ,  64,  32,  4, 4,  2,  1,  1,  1,  v_mfma_f32_32x32x4f16),
        ctrl_xdlops_mapping_t( 256, 64 ,  64,  32,  4, 4,  1,  2,  1,  1,  v_mfma_f32_32x32x4f16),
        ctrl_xdlops_mapping_t( 256, 64 ,  32,  32,  8, 4,  2,  2,  1,  1,  v_mfma_f32_32x32x8f16),
        ctrl_xdlops_mapping_t( 64 , 256,  16,  64,  4, 4,  2,  2,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 64 , 256,  32,  64,  4, 4,  1,  1,  1,  2,  v_mfma_f32_32x32x4f16),
        ctrl_xdlops_mapping_t( 64 , 256,  32,  32,  8, 4,  2,  2,  1,  1,  v_mfma_f32_32x32x8f16),
        ctrl_xdlops_mapping_t( 256, 32 ,  64,  16,  4, 4,  2,  1,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 256, 32 ,  32,  32,  8, 4,  2,  1,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 256, 32 ,  64,  4 ,  4, 4,  2,  2,  1,  2,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 32 , 256,  16,  64,  4, 4,  1,  2,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 32 , 256,  4 ,  64,  4, 4,  2,  2,  2,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 256, 16 ,  64,  4 ,  4, 4,  2,  2,  1,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 16 , 256,  4 ,  64,  4, 4,  2,  2,  1,  1,  v_mfma_f32_4x4x4f16),

        ctrl_xdlops_mapping_t( 128, 128,  32,  32,  4, 4,  2,  2,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 128, 128,  32,  32,  8, 4,  2,  2,  1,  1,  v_mfma_f32_32x32x8f16),
        ctrl_xdlops_mapping_t( 128, 128,  32,  32,  8, 4,  1,  1,  2,  2,  v_mfma_f32_32x32x8f16),
        ctrl_xdlops_mapping_t( 128, 128,  16,  16, 16, 4,  2,  2,  2,  2,  v_mfma_f32_16x16x16f16),
        ctrl_xdlops_mapping_t( 128,  64,  16,  16, 16, 4,  2,  2,  2,  1,  v_mfma_f32_16x16x16f16),
        ctrl_xdlops_mapping_t( 128,  64,  32,  32,  8, 4,  1,  2,  1,  1,  v_mfma_f32_32x32x8f16),
        ctrl_xdlops_mapping_t( 128,  64,  32,  32,  4, 4,  2,  1,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 128, 128,  32,  64,  4, 4,  1,  1,  2,  1,  v_mfma_f32_32x32x4f16),
        ctrl_xdlops_mapping_t( 128, 64 ,  32,  8 ,  4, 4,  2,  2,  1,  2,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 64 , 128,  8 ,  32,  4, 4,  2,  2,  2,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 64 , 128,  32,  64,  4, 4,  1,  1,  1,  1,  v_mfma_f32_32x32x4f16),
        ctrl_xdlops_mapping_t( 64 , 128,  64,  32,  4, 4,  1,  1,  1,  1,  v_mfma_f32_32x32x4f16),
        ctrl_xdlops_mapping_t( 64 , 128,  32,  32,  8, 4,  2,  1,  1,  1,  v_mfma_f32_32x32x8f16),
        ctrl_xdlops_mapping_t( 128, 32 ,  32,  8 ,  4, 4,  2,  2,  1,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 128, 32 ,  64,  16,  4, 4,  1,  1,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 32 , 128,  8 ,  32,  4, 4,  2,  2,  1,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 32 , 128,  16,  64,  4, 4,  1,  1,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 64 , 64 ,  16,  16,  4, 4,  2,  2,  1,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 64 , 64 ,  16,  16, 16, 4,  2,  2,  1,  1,  v_mfma_f32_16x16x16f16),
        ctrl_xdlops_mapping_t( 64 , 64 ,  16,  16, 16, 4,  1,  1,  2,  2,  v_mfma_f32_16x16x16f16),
        ctrl_xdlops_mapping_t( 64 , 64 ,  32,  32,  8, 4,  1,  1,  1,  1,  v_mfma_f32_32x32x8f16),
        ctrl_xdlops_mapping_t( 64 , 64 ,  32,  32,  4, 4,  1,  1,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 128, 16 ,  64,  16,  4, 2,  1,  1,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 16 , 128,  16,  64,  4, 2,  1,  1,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 64 , 32 ,  32,  8 ,  4, 4,  1,  1,  1,  2,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 64 , 32 ,  16,  16, 16, 4,  2,  1,  1,  1,  v_mfma_f32_16x16x16f16),
        ctrl_xdlops_mapping_t( 64 , 32 ,  16,  16,  4, 4,  2,  1,  1,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 64 , 32 ,  16,  16, 16, 4,  1,  1,  2,  1,  v_mfma_f32_16x16x16f16),
        ctrl_xdlops_mapping_t( 64 , 32 ,  64,  16,  4, 2,  1,  1,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 32 , 64 ,  8 ,  32,  4, 4,  1,  1,  2,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 32 , 64 ,  16,  64,  4, 2,  1,  1,  1,  1,  v_mfma_f32_16x16x4f16),
        ctrl_xdlops_mapping_t( 32 , 32 ,  16,  16,  4, 4,  1,  1,  1,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 32 , 32 ,  16,  16, 16, 4,  1,  1,  1,  1,  v_mfma_f32_16x16x16f16),
        ctrl_xdlops_mapping_t( 64 , 16 ,  64,  4 ,  4, 4,  1,  1,  1,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 16 , 64 ,  4 ,  64,  4, 4,  1,  1,  1,  1,  v_mfma_f32_4x4x4f16),
        # 2 waves
        ctrl_xdlops_mapping_t( 64 , 16 ,  64,  4 ,  4, 2,  1,  1,  1,  2,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 16 , 64 ,  4 ,  64,  4, 2,  1,  1,  2,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 64 , 8  ,  64,  4 ,  4, 2,  1,  1,  1,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 8  , 64 ,  4 ,  64,  4, 2,  1,  1,  1,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 32 , 16 ,  32,  8 ,  4, 2,  1,  1,  1,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 16 , 32 ,  8 ,  32,  4, 2,  1,  1,  1,  1,  v_mfma_f32_4x4x4f16),
        # 1 wave
        ctrl_xdlops_mapping_t( 32 , 16 ,  32,  8 ,  4, 1,  1,  1,  1,  2,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 16 , 32 ,  8 ,  32,  4, 1,  1,  1,  2,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 64 , 4 ,  64,  4 ,   4, 1,  1,  1,  1,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 4  , 64,  4 ,  64,   4, 1,  1,  1,  1,  1,  v_mfma_f32_4x4x4f16),
        ctrl_xdlops_mapping_t( 16 , 16,  16,  16,   4, 1,  1,  1,  1,  1,  v_mfma_f32_4x4x4f16)]

def fp16_mfma_to_bf16_1k(fp16_mfma):
    if fp16_mfma.name() == 'v_mfma_f32_4x4x4f16':
        return v_mfma_f32_4x4x4bf16_1k
    if fp16_mfma.name() == 'v_mfma_f32_16x16x4f16':
        return v_mfma_f32_16x16x4bf16_1k
    if fp16_mfma.name() == 'v_mfma_f32_16x16x16f16':
        return v_mfma_f32_16x16x16bf16_1k
    if fp16_mfma.name() == 'v_mfma_f32_32x32x4f16':
        return v_mfma_f32_32x32x4bf16_1k
    if fp16_mfma.name() == 'v_mfma_f32_32x32x8f16':
        return v_mfma_f32_32x32x8bf16_1k
    assert False, 'no such fp16 inst ' + fp16_mfma.name()
    return None

ctrl_xdlops_mapping_bf16_1k = [ctrl_xdlops_mapping_t(item.macro_tile_m, item.macro_tile_n,
                                    item.wave_tile_m, item.wave_tile_n, item.wave_tile_k, item.waves,
                                    item.wave_repeat_m, item.wave_repeat_n, item.wave_step_m, item.wave_step_n,
                                    fp16_mfma_to_bf16_1k(item.inst_mfma))  for item in ctrl_xdlops_mapping_fp16 ]

def fp16_mfma_to_16f(fp16_mfma):
    if fp16_mfma.name() == 'v_mfma_f32_4x4x4f16':
        return v_mfma_f32_4x4x4_16f_m
    if fp16_mfma.name() == 'v_mfma_f32_16x16x4f16':
        return v_mfma_f32_16x16x4_16f_m
    if fp16_mfma.name() == 'v_mfma_f32_16x16x16f16':
        return v_mfma_f32_16x16x16_16f_m
    if fp16_mfma.name() == 'v_mfma_f32_32x32x4f16':
        return v_mfma_f32_32x32x4_16f_m
    if fp16_mfma.name() == 'v_mfma_f32_32x32x8f16':
        return v_mfma_f32_32x32x8_16f_m
    assert False, 'no such fp16 inst ' + fp16_mfma.name()
    return None

ctrl_xdlops_mapping_16f = [ctrl_xdlops_mapping_t(item.macro_tile_m, item.macro_tile_n,
                                    item.wave_tile_m, item.wave_tile_n, item.wave_tile_k, item.waves,
                                    item.wave_repeat_m, item.wave_repeat_n, item.wave_step_m, item.wave_step_n,
                                    fp16_mfma_to_16f(item.inst_mfma))  for item in ctrl_xdlops_mapping_fp16 ]

ctrl_xdlops_mapping_int8 = [
        ctrl_xdlops_mapping_t( 256, 256,  64,  32,  4, 4,  2,  2,  1,  2,  v_mfma_i32_32x32x4i8),
        ctrl_xdlops_mapping_t( 256, 256,  32,  32,  8, 4,  2,  2,  2,  2,  v_mfma_i32_32x32x8i8),
        ctrl_xdlops_mapping_t( 256, 128,  64,  32,  4, 4,  2,  2,  1,  1,  v_mfma_i32_32x32x4i8),
        ctrl_xdlops_mapping_t( 256, 128,  32,  32,  8, 4,  2,  2,  2,  1,  v_mfma_i32_32x32x8i8),
        ctrl_xdlops_mapping_t( 128, 256,  32,  64,  4, 4,  2,  2,  1,  1,  v_mfma_i32_32x32x4i8),
        ctrl_xdlops_mapping_t( 128, 256,  32,  32,  8, 4,  2,  2,  1,  2,  v_mfma_i32_32x32x8i8),
        ctrl_xdlops_mapping_t( 256, 64 ,  64,  16,  4, 4,  2,  2,  1,  1,  v_mfma_i32_16x16x4i8),
        ctrl_xdlops_mapping_t( 256, 64 ,  64,  32,  4, 4,  2,  1,  1,  1,  v_mfma_i32_32x32x4i8),
        ctrl_xdlops_mapping_t( 256, 64 ,  64,  32,  4, 4,  1,  2,  1,  1,  v_mfma_i32_32x32x4i8),
        ctrl_xdlops_mapping_t( 256, 64 ,  32,  32,  8, 4,  2,  2,  1,  1,  v_mfma_i32_32x32x8i8),
        ctrl_xdlops_mapping_t( 64 , 256,  16,  64,  4, 4,  2,  2,  1,  1,  v_mfma_i32_16x16x4i8),
        ctrl_xdlops_mapping_t( 64 , 256,  32,  64,  4, 4,  1,  1,  1,  2,  v_mfma_i32_32x32x4i8),
        ctrl_xdlops_mapping_t( 64 , 256,  32,  32,  8, 4,  2,  2,  1,  1,  v_mfma_i32_32x32x8i8),
        ctrl_xdlops_mapping_t( 256, 32 ,  64,  16,  4, 4,  2,  1,  1,  1,  v_mfma_i32_16x16x4i8),
        ctrl_xdlops_mapping_t( 256, 32 ,  64,  4 ,  4, 4,  2,  2,  1,  2,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 32 , 256,  16,  64,  4, 4,  1,  2,  1,  1,  v_mfma_i32_16x16x4i8),
        ctrl_xdlops_mapping_t( 32 , 256,  4 ,  64,  4, 4,  2,  2,  2,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 256, 16 ,  64,  4 ,  4, 4,  2,  2,  1,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 16 , 256,  4 ,  64,  4, 4,  2,  2,  1,  1,  v_mfma_i32_4x4x4i8),

        ctrl_xdlops_mapping_t( 128, 128,  32,  32,  4, 4,  2,  2,  1,  1,  v_mfma_i32_16x16x4i8),
        ctrl_xdlops_mapping_t( 128, 128,  32,  32,  8, 4,  2,  2,  1,  1,  v_mfma_i32_32x32x8i8),
        ctrl_xdlops_mapping_t( 128, 128,  32,  32,  8, 4,  1,  1,  2,  2,  v_mfma_i32_32x32x8i8),
        ctrl_xdlops_mapping_t( 128, 128,  16,  16, 16, 4,  2,  2,  2,  2,  v_mfma_i32_16x16x16i8),
        ctrl_xdlops_mapping_t( 128,  64,  16,  16, 16, 4,  2,  2,  2,  1,  v_mfma_i32_16x16x16i8),
        ctrl_xdlops_mapping_t( 128,  64,  32,  32,  8, 4,  1,  2,  1,  1,  v_mfma_i32_32x32x8i8),
        ctrl_xdlops_mapping_t( 128,  64,  32,  32,  4, 4,  2,  1,  1,  1,  v_mfma_i32_16x16x4i8),
        ctrl_xdlops_mapping_t( 128, 128,  32,  64,  4, 4,  1,  1,  2,  1,  v_mfma_i32_32x32x4i8),
        ctrl_xdlops_mapping_t( 128, 64 ,  32,  8 ,  4, 4,  2,  2,  1,  2,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 64 , 128,  8 ,  32,  4, 4,  2,  2,  2,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 64 , 128,  32,  64,  4, 4,  1,  1,  1,  1,  v_mfma_i32_32x32x4i8),
        ctrl_xdlops_mapping_t( 64 , 128,  64,  32,  4, 4,  1,  1,  1,  1,  v_mfma_i32_32x32x4i8),
        ctrl_xdlops_mapping_t( 64 , 128,  32,  32,  8, 4,  2,  1,  1,  1,  v_mfma_i32_32x32x8i8),
        ctrl_xdlops_mapping_t( 128, 32 ,  32,  8 ,  4, 4,  2,  2,  1,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 128, 32 ,  64,  16,  4, 4,  1,  1,  1,  1,  v_mfma_i32_16x16x4i8),
        ctrl_xdlops_mapping_t( 32 , 128,  8 ,  32,  4, 4,  2,  2,  1,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 32 , 128,  16,  64,  4, 4,  1,  1,  1,  1,  v_mfma_i32_16x16x4i8),
        ctrl_xdlops_mapping_t( 64 , 64 ,  16,  16,  4, 4,  2,  2,  1,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 64 , 64 ,  16,  16, 16, 4,  2,  2,  1,  1,  v_mfma_i32_16x16x16i8),
        ctrl_xdlops_mapping_t( 64 , 64 ,  16,  16, 16, 4,  1,  1,  2,  2,  v_mfma_i32_16x16x16i8),
        ctrl_xdlops_mapping_t( 128, 16 ,  64,  16,  4, 2,  1,  1,  1,  1,  v_mfma_i32_16x16x4i8),
        ctrl_xdlops_mapping_t( 16 , 128,  16,  64,  4, 2,  1,  1,  1,  1,  v_mfma_i32_16x16x4i8),
        ctrl_xdlops_mapping_t( 64 , 32 ,  32,  8 ,  4, 4,  1,  1,  1,  2,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 64 , 32 ,  64,  16,  4, 2,  1,  1,  1,  1,  v_mfma_i32_16x16x4i8),
        ctrl_xdlops_mapping_t( 32 , 64 ,  8 ,  32,  4, 4,  1,  1,  2,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 32 , 64 ,  16,  64,  4, 2,  1,  1,  1,  1,  v_mfma_i32_16x16x4i8),
        ctrl_xdlops_mapping_t( 32 , 32 ,  16,  16,  4, 4,  1,  1,  1,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 32 , 32 ,  16,  16, 16, 4,  1,  1,  1,  1,  v_mfma_i32_16x16x16i8),
        ctrl_xdlops_mapping_t( 64 , 16 ,  64,  4 ,  4, 4,  1,  1,  1,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 16 , 64 ,  4 ,  64,  4, 4,  1,  1,  1,  1,  v_mfma_i32_4x4x4i8),
        # 2 waves
        ctrl_xdlops_mapping_t( 64 , 16 ,  64,  4 ,  4, 2,  1,  1,  1,  2,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 16 , 64 ,  4 ,  64,  4, 2,  1,  1,  2,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 64 , 8  ,  64,  4 ,  4, 2,  1,  1,  1,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 8  , 64 ,  4 ,  64,  4, 2,  1,  1,  1,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 32 , 16 ,  32,  8 ,  4, 2,  1,  1,  1,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 16 , 32 ,  8 ,  32,  4, 2,  1,  1,  1,  1,  v_mfma_i32_4x4x4i8),
        # 1 wave
        ctrl_xdlops_mapping_t( 32 , 16 ,  32,  8 ,  4, 1,  1,  1,  1,  2,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 16 , 32 ,  8 ,  32,  4, 1,  1,  1,  2,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 64 , 4 ,  64,  4 ,   4, 1,  1,  1,  1,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 4  , 64,  4 ,  64,   4, 1,  1,  1,  1,  1,  v_mfma_i32_4x4x4i8),
        ctrl_xdlops_mapping_t( 16 , 16,  16,  16,   4, 1,  1,  1,  1,  1,  v_mfma_i32_4x4x4i8)]



def get_ctrl_xdlops_mapping_from_wave_tile(macro_tile_m, macro_tile_n, wave_tile_m, wave_tile_n, wave_tile_k,  wave_repeat_m, wave_repeat_n, wave_step_m, wave_step_n, waves, precision, **options):
    if type(precision) is str:
        precision = amdgpu_string_to_precision(precision)
    ctrl_xdlops_mapping = ctrl_xdlops_mapping_fp32
    if precision == AMDGPU_PRECISION_FP32:
        ctrl_xdlops_mapping = ctrl_xdlops_mapping_fp32
    elif precision == AMDGPU_PRECISION_FP16:
        if 'bf16_1k_in_fp16' in options and options['bf16_1k_in_fp16']:
            ctrl_xdlops_mapping = ctrl_xdlops_mapping_16f
        else:
            ctrl_xdlops_mapping = ctrl_xdlops_mapping_fp16
    elif precision == AMDGPU_PRECISION_INT8:
        ctrl_xdlops_mapping = ctrl_xdlops_mapping_int8
    elif precision == AMDGPU_PRECISION_BF16:
        ctrl_xdlops_mapping = ctrl_xdlops_mapping_bf16_1k   # TODO: this is limited to gpu arch
    else:
        assert False, f"wrong data type"
    target_mfma_tiling = list()
    for t in ctrl_xdlops_mapping:
        if t.macro_tile_m == macro_tile_m and t.macro_tile_n == macro_tile_n and\
                t.wave_tile_m == wave_tile_m and t.wave_tile_n == wave_tile_n and t.wave_tile_k == wave_tile_k and \
                t.wave_repeat_m == wave_repeat_m and t.wave_repeat_n == wave_repeat_n and \
                t.wave_step_m == wave_step_m and t.wave_step_n == wave_step_n and \
                t.waves == waves:
            target_mfma_tiling.append(t)

    assert len(target_mfma_tiling) != 0, f"unsupported macro_tile_m:{macro_tile_m}, macro_tile_n:{macro_tile_n}, wave_tile_m:{wave_tile_m}, wave_tile_n:{wave_tile_n}, wave_repeat_m:{wave_repeat_m}, wave_repeat_n:{wave_repeat_n}, "
    # TODO: we may have multiple match, aka multipl wave mapping/mfma for single 
    return target_mfma_tiling[0]

def set_ctrl_xdlops_mapping_accvgpr_unified(accvgpr_unified):
    '''
    be very careful, we override the mfma accvgpr_unified field
    '''
    if not hasattr(set_ctrl_xdlops_mapping_accvgpr_unified, "cached_accvgpr_unified"):
        set_ctrl_xdlops_mapping_accvgpr_unified.cached_accvgpr_unified = False
    if set_ctrl_xdlops_mapping_accvgpr_unified.cached_accvgpr_unified == accvgpr_unified:
        return
    set_ctrl_xdlops_mapping_accvgpr_unified.cached_accvgpr_unified = accvgpr_unified
    for ctrl in (ctrl_xdlops_mapping_fp32, ctrl_xdlops_mapping_fp16, ctrl_xdlops_mapping_int8, ctrl_xdlops_mapping_bf16_1k, ctrl_xdlops_mapping_16f):
        for x in ctrl:
            x.inst_mfma.accvgpr_unified = accvgpr_unified

class igemm_xdlops_mapping_t(mc_base_t):
    '''
    used for mfma wave tile mapping
    most macro tile can be descripbed by this pattern. some small tiles can be achieved by w_n1=w_m1=1.
    other very small tiles can be achieved by pack different wave into different k
    '''
    def name(self):
        return ''
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_xdlops_mapping_t
        self.ctrl = ctrl
    def get_gemm_index_for_src_matrix(self, v_gemm_in, v_gemm_im, v_thread_id, v_tmp4, **options):
        '''
        notice! this is to calculate LDS offset for A/B matrix input, it is not the same as C matrix output layout, due to xdlops
        C matrix output describe is in coalescint_store
        '''
        def get_dict_with_default(some_dict, key, default_value):
            if key in some_dict:
                return some_dict[key]
            return default_value
        ctrl = self.ctrl
        #print(f"ctrl.block_n()={ctrl.block_n()}, ctrl.block_m()={ctrl.block_m()}")
        #print(f"ctrl.block_n_per_wave()={ctrl.block_n_per_wave()}, ctrl.block_m_per_wave()={ctrl.block_m_per_wave()}")
        assert ctrl.block_n() == ctrl.block_m() and ctrl.block_k_per_wave() * ctrl.block_n() * ctrl.block_n_per_wave() * ctrl.block_m_per_wave() == AMDGPU_WAVE_SIZE
        k_pack = get_dict_with_default(options, "k_pack", 1)
        v_pack = get_dict_with_default(options, "v_pack", 1)
        assert v_pack in (1, k_pack),  'currently only support v_pack is 1 or k_pack'
        if k_pack != 1:
            assert k_pack % ctrl.lanegroup_k_per_thread() == 0, f'inst:{ctrl.inst_mfma.name()} require k_pack:{k_pack} since lanegroup_k_per_thread:{ctrl.lanegroup_k_per_thread()}'
            # assert k_pack % ctrl.inst_mfma.num_v_a == 0 and k_pack % ctrl.inst_mfma.num_v_b == 0, f'inst:{ctrl.inst_mfma.name()} require k_pack since num_v_a:{ctrl.inst_mfma.num_v_a}, num_v_b:{ctrl.inst_mfma.num_v_b}'
            k_pack_per_thread = k_pack // ctrl.lanegroup_k_per_thread()
        with self._deferred_context():
            self._emit(f"; xdlops mapping, get source matrix gemm index, k_pack:{k_pack}, v_pack:{v_pack}, k_pack_per_thread:{k_pack_per_thread if k_pack != 1 else 1}")
            self._emit(f"v_and_b32 v[{v_gemm_in}], {ctrl.block_n() - 1}, v[{v_thread_id}]           ; block_n index ")
            self._emit(f"v_and_b32 v[{v_gemm_im}], {ctrl.block_m() - 1}, v[{v_thread_id}]           ; block_m index ")
            if k_pack != 1:
                self._emit(f"v_lshlrev_b32 v[{v_gemm_in}], {utility_log2(k_pack)}, v[{v_gemm_in}]   ; shift left k_pack:{k_pack}")
                self._emit(f"v_lshlrev_b32 v[{v_gemm_im}], {utility_log2(k_pack)}, v[{v_gemm_im}]   ; shift left k_pack:{k_pack}")

            self._emit(f"v_lshrrev_b32 v[{v_thread_id}], {utility_log2(ctrl.block_n())}, v[{v_thread_id}]")
            if ctrl.block_k_per_wave() != 1:
                self._emit(f"v_and_b32 v[{v_tmp4} + 0], {ctrl.block_k_per_wave() - 1}, v[{v_thread_id}]          ; block_k_per_wave index")
                if k_pack != 1:
                    if v_pack == 1:
                        if k_pack_per_thread >= ctrl.block_k_per_wave():
                            #self._emit(f"v_or_b32 v[{v_gemm_in}],  v[{v_tmp4} + 0], v[{v_gemm_in}]  ; or k_pack_per_thread:{k_pack_per_thread}")
                            #self._emit(f"v_or_b32 v[{v_gemm_im}],  v[{v_tmp4} + 0], v[{v_gemm_im}]  ; or k_pack_per_thread:{k_pack_per_thread}")
                            self._emit(f"v_lshl_or_b32 v[{v_gemm_in}],  v[{v_tmp4} + 0], {utility_log2(ctrl.lanegroup_k_per_thread())}, v[{v_gemm_in}]  ; or lanegroup_k_per_thread:{ctrl.lanegroup_k_per_thread()}")
                            self._emit(f"v_lshl_or_b32 v[{v_gemm_im}],  v[{v_tmp4} + 0], {utility_log2(ctrl.lanegroup_k_per_thread())}, v[{v_gemm_im}]  ; or lanegroup_k_per_thread:{ctrl.lanegroup_k_per_thread()}")
                        else:
                            self._emit(f"v_and_b32 v[{v_tmp4} + 1], {k_pack_per_thread - 1}, v[{v_tmp4} + 0]   ; and k_pack_per_thread:{k_pack_per_thread}")
                            self._emit(f"v_lshrrev_b32 v[{v_tmp4} + 0], {utility_log2(k_pack_per_thread)}, v[{v_tmp4} + 0] ; shift right k_pack_per_thread:{k_pack_per_thread}")
                            #self._emit(f"v_or_b32 v[{v_gemm_in}],  v[{v_tmp4} + 1], v[{v_gemm_in}]  ; or k_pack_per_thread:{k_pack_per_thread}")
                            #self._emit(f"v_or_b32 v[{v_gemm_im}],  v[{v_tmp4} + 1], v[{v_gemm_im}]  ; or k_pack_per_thread:{k_pack_per_thread}")
                            self._emit(f"v_lshl_or_b32 v[{v_gemm_in}],  v[{v_tmp4} + 1], {utility_log2(ctrl.lanegroup_k_per_thread())}, v[{v_gemm_in}]  ; or lanegroup_k_per_thread:{ctrl.lanegroup_k_per_thread()}")
                            self._emit(f"v_lshl_or_b32 v[{v_gemm_im}],  v[{v_tmp4} + 1], {utility_log2(ctrl.lanegroup_k_per_thread())}, v[{v_gemm_im}]  ; or lanegroup_k_per_thread:{ctrl.lanegroup_k_per_thread()}")
                            self._emit(f"v_lshl_or_b32 v[{v_gemm_in}], v[{v_tmp4} + 0], {utility_log2(ctrl.macro_tile_n * k_pack)}, v[{v_gemm_in}]")
                            self._emit(f"v_lshl_or_b32 v[{v_gemm_im}], v[{v_tmp4} + 0], {utility_log2(ctrl.macro_tile_m * k_pack)}, v[{v_gemm_im}]")
                    else:
                        self._emit(f"v_lshl_or_b32 v[{v_gemm_in}], v[{v_tmp4} + 0], {utility_log2(ctrl.macro_tile_n * k_pack)}, v[{v_gemm_in}]")
                        self._emit(f"v_lshl_or_b32 v[{v_gemm_im}], v[{v_tmp4} + 0], {utility_log2(ctrl.macro_tile_m * k_pack)}, v[{v_gemm_im}]")
                else:
                    self._emit(f"v_lshl_or_b32 v[{v_gemm_in}], v[{v_tmp4} + 0], {utility_log2(ctrl.macro_tile_n)}, v[{v_gemm_in}]")
                    self._emit(f"v_lshl_or_b32 v[{v_gemm_im}], v[{v_tmp4} + 0], {utility_log2(ctrl.macro_tile_m)}, v[{v_gemm_im}]")
                self._emit(f"v_lshrrev_b32 v[{v_thread_id}], {utility_log2(ctrl.block_k_per_wave())}, v[{v_thread_id}]")
                pass

            if ctrl.block_n_per_wave() != 1:
                self._emit(f"v_and_b32 v[{v_tmp4} + 0], {ctrl.block_n_per_wave() - 1}, v[{v_thread_id}]          ; block_n_per_wave index")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_in}], v[{v_tmp4} + 0], {utility_log2(ctrl.block_n() * k_pack)}, v[{v_gemm_in}]")
                self._emit(f"v_lshrrev_b32 v[{v_thread_id}], {utility_log2(ctrl.block_n_per_wave())}, v[{v_thread_id}]")
            if ctrl.block_m_per_wave() != 1:
                self._emit(f"v_and_b32 v[{v_tmp4} + 1], {ctrl.block_m_per_wave() - 1}, v[{v_thread_id}]          ; block_m_per_wave index")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_im}], v[{v_tmp4} + 1], {utility_log2(ctrl.block_m() * k_pack)}, v[{v_gemm_im}]")
                self._emit(f"v_lshrrev_b32 v[{v_thread_id}], {utility_log2(ctrl.block_m_per_wave())}, v[{v_thread_id}]")
            if ctrl.waves_per_n() != 1:
                self._emit(f"v_and_b32 v[{v_tmp4} + 2], {ctrl.waves_per_n() - 1}, v[{v_thread_id}]  ; waves_per_n index")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_in}], v[{v_tmp4} + 2], {utility_log2(ctrl.wave_tile_n * ctrl.wave_step_n * k_pack)}, v[{v_gemm_in}]")
                self._emit(f"v_lshrrev_b32 v[{v_thread_id}], {utility_log2(ctrl.waves_per_n())}, v[{v_thread_id}]")
            if ctrl.waves_per_m() != 1:
                self._emit(f"v_and_b32 v[{v_tmp4} + 3], {ctrl.waves_per_m() - 1}, v[{v_thread_id}]  ; waves_per_m index")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_im}], v[{v_tmp4} + 3], {utility_log2(ctrl.wave_tile_m * ctrl.wave_step_m * k_pack)}, v[{v_gemm_im}]")
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
            self._emit(f"; xdlops mapping, get dst matrix gemm index")

            assert ctrl.lanegroup_m_per_cluster() * ctrl.block_m_per_lanegroup() * ctrl.lanegroup_n_per_cluster() * ctrl.block_n_per_lanegroup() == AMDGPU_WAVE_SIZE
            # first, calculate within each wave tile
            p_ssh(f"{v_tmp4}+0",  v_thread_id, ctrl.lanegroup_n_per_cluster())
            p_ssh(f"{v_tmp4}+1",  v_thread_id, ctrl.lanegroup_m_per_cluster())
            p_ssh(f"{v_tmp4}+2",  v_thread_id, ctrl.block_n_per_lanegroup())
            p_ssh(f"{v_tmp4}+3",  v_thread_id, ctrl.block_m_per_lanegroup())

            p_sac_n(v_gemm_in, f"{v_tmp4}+0",  ctrl.lanegroup_n_per_cluster(),  1)
            p_sac_n(v_gemm_in, f"{v_tmp4}+2",  ctrl.block_n_per_lanegroup(),   ctrl.lanegroup_n_per_cluster())

            p_sac_m(v_gemm_im, f"{v_tmp4}+1",  ctrl.lanegroup_m_per_cluster(),  ctrl.lanegroup_m_per_thread())
            p_sac_m(v_gemm_im, f"{v_tmp4}+3",  ctrl.block_m_per_lanegroup(),  ctrl.lanegroup_m_per_block() * ctrl.lanegroup_m_per_cluster() * ctrl.lanegroup_m_per_thread())

            # second, calculate among waves
            p_ssh(f"{v_tmp4}+0",  v_thread_id, ctrl.waves_per_n())   
            p_ssh(f"{v_tmp4}+1",  v_thread_id, ctrl.waves_per_m(), True)

            p_sac_n(v_gemm_in, f"{v_tmp4}+0",  ctrl.waves_per_n(),  ctrl.wave_tile_n * ctrl.wave_step_n)
            p_sac_m(v_gemm_im, f"{v_tmp4}+1",  ctrl.waves_per_m(),  ctrl.wave_tile_m * ctrl.wave_step_m)

            if p_sac_n.first == 1:
                self._emit(f"v_mov_b32 v[{v_gemm_in}], 0")
            if p_sac_m.first == 1:
                self._emit(f"v_mov_b32 v[{v_gemm_im}], 0")

            self._emit_empty_line()

        return self._get_deferred()

    def emit(self):
        assert False
