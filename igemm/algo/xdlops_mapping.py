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
# pylint: disable=maybe-no-member

from ..codegen import *
from .igemm_base import *
from .mfma_main_loop import *
import math

class ctrl_xdlops_mapping_t(object):
    def __init__(self, macro_tile_m, macro_tile_n, wave_tile_m, wave_tile_n, waves, wave_repeat_m, wave_repeat_n, wave_step_m, wave_step_n, inst_mfma):
        self.macro_tile_m = macro_tile_m
        self.macro_tile_n = macro_tile_n
        self.wave_tile_m = wave_tile_m
        self.wave_tile_n = wave_tile_n
        self.waves = waves
        self.wave_repeat_m = wave_repeat_m
        self.wave_repeat_n = wave_repeat_n
        self.wave_step_m = wave_step_m
        self.wave_step_n = wave_step_n
        self.inst_mfma = inst_mfma

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
        return total_c

    def acc_c_per_thread_n(self):
        tn_r, tn_s, tn_w, tn_b, tn_t = self.wave_repeat_n, self.wave_step_n, self.lanegroup_n_per_wave(), self.lanegroup_n_per_block(), self.lanegroup_n_per_thread()
        return tn_r, tn_s, tn_w, tn_b, tn_t

    def acc_c_per_thread_m(self):
        tm_r, tm_s, tm_w, tm_b, tm_t = self.wave_repeat_m, self.wave_step_m, self.lanegroup_m_per_wave(), self.lanegroup_m_per_block(), self.lanegroup_m_per_thread()
        return tm_r, tm_s, tm_w, tm_b, tm_t


    def block_size(self):
        return self.waves * 64 # wave size 64

    def lanegroup_validate(self):
        assert self.lanegroup_n_per_thread() * self.lanegroup_n_per_block() * self.lanegroup_n_per_wave() * \
                self.lanegroup_m_per_thread() * self.lanegroup_m_per_block() * self.lanegroup_m_per_wave() \
                == self.inst_mfma.num_a_c
    # lanegroup layout for a single xdlops issue
    # each lanegroup is a 4x64 matrix, and expand into whole block, then wave
    # hence we group xdlops layout into 4 levels: per_thread->per_cluster->per_block->per_wave
    #
    def block_m_per_wave(self):
        ''' [among different thread] '''
        assert self.wave_tile_m % (self.lanegroup_m_per_thread() * self.lanegroup_m_per_cluster() * self.lanegroup_m_per_block()) == 0
        assert self.inst_mfma.m == self.lanegroup_m_per_thread() * self.lanegroup_m_per_cluster() * self.lanegroup_m_per_block()
        return self.wave_tile_m // (self.lanegroup_m_per_thread() * self.lanegroup_m_per_cluster() * self.lanegroup_m_per_block()) 

    def block_n_per_wave(self):
        ''' [among different thread] '''
        assert self.wave_tile_n % (self.lanegroup_n_per_thread() * self.lanegroup_n_per_cluster() * self.lanegroup_n_per_block()) == 0
        assert self.inst_mfma.n == self.lanegroup_n_per_thread() * self.lanegroup_n_per_cluster() * self.lanegroup_n_per_block()
        return self.wave_tile_n // (self.lanegroup_n_per_thread() * self.lanegroup_n_per_cluster() * self.lanegroup_n_per_block())

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

    def lanegroup_m_per_cluster(self):
        ''' [among different thread] for xdlops, always m per block as clusters. perthread agpr do not contain this'''
        return math.gcd(self.inst_mfma.m//self.lanegroup_m_per_thread(), AMDGPU_WAVE_SIZE // self.inst_mfma.n )

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
        return math.gcd(self.block_n_per_wave(), self.inst_mfma.num_a_c // (self.lanegroup_m_per_thread() * self.lanegroup_m_per_block()))

    def serialize(self):
        self.lanegroup_validate()
        s = f"mt_m:{self.macro_tile_m}, mt_n:{self.macro_tile_n}, wt_m:{self.wave_tile_m}, wt_n:{self.wave_tile_n}, ws:{self.waves}, r_m:{self.wave_repeat_m}, r_n:{self.wave_repeat_n}, s_m:{self.wave_step_m}, s_n:{self.wave_step_n} | "
        s += f"{self.inst_mfma.m}x{self.inst_mfma.n}x{self.inst_mfma.k}, " + \
                f"lanegroup_m_tcbw:{self.lanegroup_m_per_thread()}x{self.lanegroup_m_per_cluster()}x{self.lanegroup_m_per_block()}x{self.lanegroup_m_per_wave()}, " + \
                f"lanegroup_n_tcbw:{self.lanegroup_n_per_thread()}x{self.lanegroup_n_per_cluster()}x{self.lanegroup_n_per_block()}x{self.lanegroup_n_per_wave()}"
        #print(s)
        return s

#                             mt_m,mt_n,wt_m,wt_n, ws,r_m,r_n,s_m,s_n, inst_mfma
ctrl_xdlops_mapping_fp32 = [
        # ctrl_xdlops_mapping_t( 256, 256,  32,  64,  4,  2,  2,  2,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 256, 128,  64,  32,  4,  2,  2,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 128, 256,  32,  64,  4,  2,  2,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 256, 64 ,  64,  16,  4,  2,  2,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 64 , 256,  16,  64,  4,  2,  2,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 256, 32 ,  64,  4 ,  4,  2,  2,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 256,  4 ,  64,  4,  2,  2,  2,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 256, 16 ,  64,  4 ,  4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 256,  4 ,  64,  4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 128, 128,  32,  32,  4,  2,  2,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 128, 64 ,  32,  8 ,  4,  2,  2,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 128,  8 ,  32,  4,  2,  2,  2,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 128, 32 ,  32,  8 ,  4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 128,  8 ,  32,  4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 64 ,  16,  16,  4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 128, 16 ,  64,  4 ,  4,  1,  1,  2,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 128,  4 ,  64,  4,  1,  1,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 32 ,  32,  8 ,  4,  1,  1,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 64 ,  8 ,  32,  4,  1,  1,  2,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 32 ,  16,  16,  4,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 256, 4  ,  64,  4 ,  4,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 4  , 256,  4 ,  64,  4,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 16 ,  64,  4 ,  4,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 64 ,  4 ,  64,  4,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        # 2waves, block_size=128
        ctrl_xdlops_mapping_t( 128, 4  ,  64,  4 ,  2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 4  , 128,  4 ,  64,  2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 8  ,  64,  4 ,  2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 8  , 64 ,  4 ,  64,  2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 16 ,  32,  8 ,  2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 32 ,  8 ,  32,  2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32)]

def get_ctrl_xdlops_mapping_fp32(macro_tile_m, macro_tile_n, waves = 4):
    target_mfma_tiling_fp32 = list()
    for t in ctrl_xdlops_mapping_fp32:
        if t.macro_tile_m == macro_tile_m and t.macro_tile_n == macro_tile_n and t.waves == waves:
            target_mfma_tiling_fp32.append(t)

    assert len(target_mfma_tiling_fp32) != 0, f"unsupported macro_tile_m:{macro_tile_m}, macro_tile_n:{macro_tile_n}, waves:{waves}"
    # TODO: we may have multiple match, aka multipl wave mapping/mfma for single 
    return target_mfma_tiling_fp32[0]

def get_ctrl_xdlops_mapping_from_wave_fp32(wave_tile_m, wave_tile_n, wave_repeat_m, wave_repeat_n, wave_step_m, wave_step_n):
    target_mfma_tiling_fp32 = list()
    for t in ctrl_xdlops_mapping_fp32:
        if t.wave_tile_m == wave_tile_m and t.wave_tile_n == wave_tile_n and \
                t.wave_repeat_m == wave_repeat_m and t.wave_repeat_n == wave_repeat_n and \
                t.wave_step_m == wave_step_m and t.wave_step_n == wave_step_n:
            target_mfma_tiling_fp32.append(t)

    assert len(target_mfma_tiling_fp32) != 0, f"unsupported wave_tile_m:{wave_tile_m}, wave_tile_n:{wave_tile_n}, wave_repeat_m:{wave_repeat_m},  wave_repeat_n:{wave_repeat_n}"
    # TODO: we may have multiple match, aka multipl wave mapping/mfma for single 
    return target_mfma_tiling_fp32[0]

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
    def __call__(self, v_gemm_in, v_gemm_im, v_wave_id, v_lane_id, v_tmp2):
        ctrl = self.ctrl
        with self._deferred_context():
            if ctrl.waves_per_n() != 1:
                self._emit(f"v_and_b32 v[{v_tmp2}], {ctrl.wave_tile_n - 1}, v[{v_lane_id}]")
                self._emit(f"v_and_b32 v[{v_tmp2}+1], {igemm_log2(ctrl.waves_per_n())}, v[{v_wave_id}]")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_in}],  v[{v_tmp2}+1], {igemm_log2(ctrl.composed_wave_tile_n())}, v[{v_tmp2}]")
            else:
                self._emit(f"v_and_b32 v[{v_gemm_in}], {ctrl.wave_tile_n - 1}, v[{v_lane_id}]")

            if ctrl.waves_per_m() != 1:
                self._emit(f"v_and_b32 v[{v_tmp2}], {ctrl.wave_tile_m - 1}, v[{v_lane_id}]")
                self._emit(f"v_lshrrev_b32 v[{v_tmp2}+1], {igemm_log2(ctrl.waves_per_n())}, v[{v_wave_id}] ")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_im}],  v[{v_tmp2}+1], {igemm_log2(ctrl.composed_wave_tile_m())}, v[{v_tmp2}]")
            else:
                self._emit(f"v_and_b32 v[{v_gemm_im}], {ctrl.wave_tile_m - 1}, v[{v_lane_id}]")

        return self._get_deferred()

    def emit(self):
        assert False
