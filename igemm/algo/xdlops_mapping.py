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

    def composed_waves_per_m(self):
        ''' attention! not count repeat'''
        return self.macro_tile_m // (self.wave_repeat_m * self.wave_tile_m * self.wave_step_m)

    def composed_waves_per_n(self):
        ''' attention! not count repeat'''
        return self.macro_tile_n // (self.wave_repeat_n * self.wave_tile_n * self.wave_step_n)

    def total_acc_c(self):
        return self.wave_repeat_m * self.wave_repeat_n * self.wave_step_m * self.wave_step_n * self.inst_mfma.num_a_c

    def block_size(self):
        return self.waves * 64 # wave size 64

#                             mt_m,mt_n,wt_m,wt_n, ws,r_m,r_n,s_m,s_n, inst_mfma
ctrl_xdlops_mapping_fp32 = [
        ctrl_xdlops_mapping_t( 256, 256,  32,  64,  4,  2,  2,  2,  1,  v_mfma_f32_32x32x1f32),
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
            if ctrl.composed_waves_per_n() != 1:
                self._emit(f"v_and_b32 v[{v_tmp2}], {ctrl.wave_tile_n - 1}, v[{v_lane_id}]")
                self._emit(f"v_and_b32 v[{v_tmp2}+1], {igemm_log2(ctrl.composed_waves_per_n())}, v[{v_wave_id}]")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_in}],  v[{v_tmp2}+1], {igemm_log2(ctrl.composed_wave_tile_n())}, v[{v_tmp2}]")
            else:
                self._emit(f"v_and_b32 v[{v_gemm_in}], {ctrl.wave_tile_n - 1}, v[{v_lane_id}]")

            if ctrl.composed_waves_per_m() != 1:
                self._emit(f"v_and_b32 v[{v_tmp2}], {ctrl.wave_tile_m - 1}, v[{v_lane_id}]")
                self._emit(f"v_lshrrev_b32 v[{v_tmp2}+1], {igemm_log2(ctrl.composed_waves_per_n())}, v[{v_wave_id}] ")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_im}],  v[{v_tmp2}+1], {igemm_log2(ctrl.composed_wave_tile_m())}, v[{v_tmp2}]")
            else:
                self._emit(f"v_and_b32 v[{v_gemm_im}], {ctrl.wave_tile_m - 1}, v[{v_lane_id}]")

        return self._get_deferred()

    def emit(self):
        assert False
