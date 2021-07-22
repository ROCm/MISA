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

class ctrl_thread_mapping_t(object):
    def __init__(self):
        self.thread_lengths = list
        self.cluster_lengths = list

    def valid_thread_lengths(self):
        assert type(self.thread_lengths) is list and len(self.thread_lengths) == 6
    def valid_cluster_lengths(self):
        assert type(self.cluster_lengths) is list and len(self.cluster_lengths) == 6

    def t_n0(self):
        self.valid_thread_lengths()
        return self.thread_lengths[5]
    def t_m0(self):
        self.valid_thread_lengths()
        return self.thread_lengths[4]
    def t_n1(self):
        self.valid_thread_lengths()
        return self.thread_lengths[3]
    def t_m1(self):
        self.valid_thread_lengths()
        return self.thread_lengths[2]
    def t_nr(self):
        self.valid_thread_lengths()
        return self.thread_lengths[1]
    def t_mr(self):
        self.valid_thread_lengths()
        return self.thread_lengths[0]

    def c_n0(self):
        self.valid_cluster_lengths()
        return self.cluster_lengths[5]
    def c_m0(self):
        self.valid_cluster_lengths()
        return self.cluster_lengths[4]
    def c_n1(self):
        self.valid_cluster_lengths()
        return self.cluster_lengths[3]
    def c_m1(self):
        self.valid_cluster_lengths()
        return self.cluster_lengths[2]
    def c_nr(self):
        self.valid_cluster_lengths()
        return self.cluster_lengths[1]
    def c_mr(self):
        self.valid_cluster_lengths()
        return self.cluster_lengths[0]

    def n_n0(self):
        return self.t_n0()*self.c_n0()
    def n_m0(self):
        return self.t_m0()*self.c_m0()
    def n_n1(self):
        return self.t_n1()*self.c_n1()
    def n_m1(self):
        return self.t_m1()*self.c_m1()
    def n_nr(self):
        return self.t_nr()*self.c_nr()
    def n_mr(self):
        return self.t_mr()*self.c_mr()

    def n_n_total(self):
        return self.n_n0() * self.n_n1() * self.n_nr()
    def n_m_total(self):
        return self.n_m0() * self.n_m1() * self.n_mr()

    def dump(self):
        print(f"thread:{utility_list_to_string(self.thread_lengths)}, cluster:{utility_list_to_string(self.cluster_lengths)}")

class igemm_thread_mapping_t(mc_base_t):
    def name(self):
        return ''
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        self.ctrl = ctrl
    def __call__(self, v_gemm_in, v_gemm_im, v_tid_shifter, v_tmp4):
        ctrl = self.ctrl
        t_mr, t_nr, t_m1, t_n1, t_m0, t_n0 = ctrl.t_mr(), ctrl.t_nr(), ctrl.t_m1(), ctrl.t_n1(), ctrl.t_m0(), ctrl.t_n0()
        c_mr, c_nr, c_m1, c_n1, c_m0, c_n0 = ctrl.c_mr(), ctrl.c_nr(), ctrl.c_m1(), ctrl.c_n1(), ctrl.c_m0(), ctrl.c_n0()

        assert c_mr == 1 and c_nr == 1
        assert t_m1 == 1 and t_n1 == 1
        with self._deferred_context():
            self._emit(f"; c thread mapping ")
            self._emit(f"; ->            MR x  NR x ML1 x NL1 x ML0 x NL0")
            self._emit(f";  cluster       1 x   1 x  {c_m1}  x  {c_n1}  x  {c_m0}  x  {c_n0}")
            self._emit(f";  perthrd       {t_mr} x   {t_nr} x  1  x  1  x  {t_m0}  x  {t_n0}")

            self._emit(f"v_and_b32 v[{v_tmp4}], {c_n0 - 1}, v[{v_tid_shifter}]")
            self._emit(f"v_lshlrev_b32 v[{v_tmp4}], {utility_log2(t_n0)}, v[{v_tmp4}]         ; => iNL0")
            self._emit(f"v_lshrrev_b32 v[{v_tid_shifter}], {utility_log2(c_n0)}, v[{v_tid_shifter}]")

            self._emit(f"v_and_b32 v[{v_tmp4}+1], {c_m0 - 1}, v[{v_tid_shifter}]")
            self._emit(f"v_lshlrev_b32 v[{v_tmp4}+1], {utility_log2(t_m0)}, v[{v_tmp4}+1]     ; => iML0")
            self._emit(f"v_lshrrev_b32 v[{v_tid_shifter}], {utility_log2(c_m0)}, v[{v_tid_shifter}]")

            self._emit(f"v_and_b32 v[{v_tmp4}+2],   {c_n1 - 1}, v[{v_tid_shifter}]       ; => iNL1")
            self._emit(f"v_lshrrev_b32 v[{v_tid_shifter}], {utility_log2(c_n1)}, v[{v_tid_shifter}]")
            self._emit(f"v_and_b32 v[{v_tmp4}+3],   {c_m1 - 1}, v[{v_tid_shifter}]       ; => iML1")

            self._emit(f"v_lshl_or_b32 v[{v_gemm_in}], v[{v_tmp4}+2], {utility_log2(t_n0 * c_n0)}, v[{v_tmp4}]               ; in  (without repeat)")
            self._emit(f"v_lshl_or_b32 v[{v_gemm_im}], v[{v_tmp4}+3], {utility_log2(t_m0 * c_m0)}, v[{v_tmp4}+1]              ; im  (without repeat)")
        return self._get_deferred()

    def emit(self):
        assert False
