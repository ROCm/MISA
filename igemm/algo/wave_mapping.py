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

class ctrl_wave_mapping_t(object):
    def __init__(self):
        self.wave_lengths = list
        self.cluster_lengths = list

    def valid_wave_lengths(self):
        assert type(self.wave_lengths) is list and len(self.wave_lengths) == 4
    def valid_cluster_lengths(self):
        assert type(self.cluster_lengths) is list and len(self.cluster_lengths) == 4

    def w_n0(self):
        self.valid_wave_lengths()
        return self.wave_lengths[-1]
    def w_m0(self):
        self.valid_wave_lengths()
        return self.wave_lengths[-2]
    def w_n1(self):
        self.valid_wave_lengths()
        return self.wave_lengths[-3]
    def w_m1(self):
        self.valid_wave_lengths()
        return self.wave_lengths[-4]

    def c_n0(self):
        self.valid_cluster_lengths()
        return self.cluster_lengths[-1]
    def c_m0(self):
        self.valid_cluster_lengths()
        return self.cluster_lengths[-2]
    def c_n1(self):
        self.valid_cluster_lengths()
        return self.cluster_lengths[-3]
    def c_m1(self):
        self.valid_cluster_lengths()
        return self.cluster_lengths[-4]

    def n_n0(self):
        return self.w_n0()*self.c_n0()
    def n_m0(self):
        return self.w_m0()*self.c_m0()
    def n_n1(self):
        return self.w_n1()*self.c_n1()
    def n_m1(self):
        return self.w_m1()*self.c_m1()

class igemm_wave_mapping_t(mc_base_t):
    '''

            +< c_n0>+
            |w_n0   |
    +----+--#---+---#---+---+
    ^  w_m0 |w0 |w1 |w0 |w1 |
   c_m0  +--+---+---+---+---+
    v       |w2 |w3 |w2 |w3 |
    +----+--#---+---#---+---+

    '''
    def name(self):
        return ''
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_wave_mapping_t
        self.ctrl = ctrl
    def __call__(self, v_gemm_in, v_gemm_im, v_tid_shifter, v_tmp4):
        ctrl = self.ctrl
        w_m1, w_n1, w_m0, w_n0 = ctrl.w_m1(), ctrl.w_n1(), ctrl.w_m0(), ctrl.w_n0()
        c_m1, c_n1, c_m0, c_n0 = ctrl.c_m1(), ctrl.c_n1(), ctrl.c_m0(), ctrl.c_n0()

        assert c_m1 == 1 and c_n1 == 1
        with self._deferred_context():
            self._emit(f"; c wave mapping ")
            self._emit(f"; ->           ML1 x NL1 x ML0 x NL0")
            self._emit(f";  cluster       1 x   1 x  {c_m0}  x  {c_n0}")
            self._emit(f";  perwave       {w_m1} x   {w_n1} x {w_m0}  x  {w_n0}")

            self._emit(f"v_and_b32 v[{v_tmp4}], {w_n0 - 1}, v[{v_tid_shifter}]")
            self._emit(f"v_and_b32 v[{v_tmp4}+1], {w_m0 - 1}, v[{v_tid_shifter}]")

            if c_n0 != 1:
                self._emit(f"v_lshrrev_b32 v[{v_tmp4}+2], {igemm_log2(w_n0)}, v[{v_tid_shifter}]")
                self._emit(f"v_and_b32 v[{v_tmp4}+2], {c_n0 - 1}, v[{v_tmp4}+2]     ; in0")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_in}],  v[{v_tmp4}+2], {igemm_log2(w_n0)}, v[{v_tmp4}]")
            else:
                self._emit(f"v_mov_b32 v[{v_gemm_in}],  v[{v_tmp4}]")
            
            if c_m0 != 1:
                self._emit(f"v_lshrrev_b32 v[{v_tmp4}+3], {igemm_log2(w_m0 * c_m0)}, v[{v_tid_shifter}]")
                self._emit(f"v_and_b32 v[{v_tmp4}+3], {c_m0 - 1}, v[{v_tmp4}+3] ")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_im}],  v[{v_tmp4}+3], {igemm_log2(w_m0)}, v[{v_tmp4}+1]")
            else:
                self._emit(f"v_mov_b32 v[{v_gemm_im}],  v[{v_tmp4}+1]")
        return self._get_deferred()

    def emit(self):
        assert False
