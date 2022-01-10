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

from ..codegen import *
import math

class macro_int_div_vv_t(macro_base_t):
    '''
    integer divide to compute `v_q = v_n / v_d`, v_q, v_n, v_d all vgpr
    '''
    def name(self):
        return '.v_u32_div'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, v_q, v_n, v_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}'.format(self.name(), v_q, v_n, v_d, v_tmp4, s_tmp4)
    def emit(self):
        with self._emit_macro_indented(".macro {} v_q, v_n, v_d, v_tmp4, s_tmp4".format(self.name())):
            self._emit("v_cvt_f32_u32     v[\\v_tmp4+0],   v[\\v_d]")
            self._emit("v_rcp_f32         v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_f32         v[\\v_tmp4+0],   0x4f800000, v[\\v_tmp4+0]")
            self._emit("v_cvt_u32_f32     v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   v[\\v_d],      v[\\v_tmp4+0]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+2],   v[\\v_d],      v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+3],   vcc, 0,     v[\\v_tmp4+1]")
            self._emit("v_cmp_ne_i32      s[\\s_tmp4:\\s_tmp4+1], 0,          v[\\v_tmp4+2]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+1],   v[\\v_tmp4+3],   v[\\v_tmp4+1],   s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+1],   v[\\v_tmp4+1],   v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_add_co_u32      v[\\v_tmp4+0],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_tmp4+2],   s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_n]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   v[\\v_tmp4+0],   v[\\v_d]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_ge_u32      s[\\s_tmp4:\\s_tmp4+1], v[\\v_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_ge_u32      s[\\s_tmp4+2:\\s_tmp4+3], v[\\v_tmp4+2],   v[\\v_d]")
            self._emit("v_add_co_u32      v[\\v_tmp4+2],   vcc, 1, v[\\v_tmp4+0]")
            self._emit("s_and_b64         s[\\s_tmp4+2:\\s_tmp4+3], s[\\s_tmp4:\\s_tmp4+1], s[\\s_tmp4+2:\\s_tmp4+3]")
            self._emit("v_add_co_u32      v[\\v_tmp4+1],   vcc, -1,    v[\\v_tmp4+0]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+0],   v[\\v_tmp4+2],      s[\\s_tmp4+2:\\s_tmp4+3]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+1],   v[\\v_tmp4+2],      s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_cmp_ne_i32      vcc,          0,          v[\\v_d]")
            self._emit("v_cndmask_b32     v[\\v_q],      -1,         v[\\v_tmp4+2],      vcc")

class macro_int_div_rem_vv_t(macro_base_t):
    '''
    integer divide to compute `v_q = v_n / v_d, v_r = v_n % v_d`, v_r, v_q, v_n, v_d all vgpr
    '''
    def name(self):
        return '.v_u32_div_rem'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, v_r, v_q, v_n, v_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}, {}'.format(self.name(), v_r, v_q, v_n, v_d, v_tmp4, s_tmp4)
    def emit(self):
        int_div_vv = macro_int_div_vv_t(self.mc)
        with self._emit_macro_indented(".macro {} v_r, v_q, v_n, v_d, v_tmp4, s_tmp4".format(self.name())):
            self._emit(int_div_vv("\\v_q", "\\v_n", "\\v_d", "\\v_tmp4", "\\s_tmp4"))
            self._emit(f"v_mul_lo_u32 v[\\v_tmp4], v[\\v_d], v[\\v_q]")
            self._emit(f"v_sub_u32 v[\\v_r], v[\\v_n], v[\\v_tmp4]")

class macro_int_div_vs_t(macro_base_t):
    '''
    integer divide to compute `v_q = v_n / s_d`, v_q, v_n are vgpr, s_d is sgpr
    '''
    def name(self):
        return '.v_u32_div_vs'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, v_q, v_n, s_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}'.format(self.name(), v_q, v_n, s_d, v_tmp4, s_tmp4)
    def emit(self):
        with self._emit_macro_indented(".macro {} v_q, v_n, s_d, v_tmp4, s_tmp4".format(self.name())):
            self._emit("v_cvt_f32_u32     v[\\v_tmp4+0],   s[\\s_d]")
            self._emit("v_rcp_f32         v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_f32         v[\\v_tmp4+0],   0x4f800000, v[\\v_tmp4+0]")
            self._emit("v_cvt_u32_f32     v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],      v[\\v_tmp4+0]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+2],   s[\\s_d],      v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+3],   vcc, 0,     v[\\v_tmp4+1]")
            self._emit("v_cmp_ne_i32      s[\\s_tmp4:\\s_tmp4+1], 0,          v[\\v_tmp4+2]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+1],   v[\\v_tmp4+3],   v[\\v_tmp4+1],   s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+1],   v[\\v_tmp4+1],   v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_add_co_u32      v[\\v_tmp4+0],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_tmp4+2],   s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_n]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],     v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_ge_u32      s[\\s_tmp4:\\s_tmp4+1], v[\\v_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_le_u32      s[\\s_tmp4+2:\\s_tmp4+3],  s[\\s_d],    v[\\v_tmp4+2]")
            self._emit("v_add_co_u32      v[\\v_tmp4+2],   vcc, 1, v[\\v_tmp4+0]")
            self._emit("s_and_b64         s[\\s_tmp4+2:\\s_tmp4+3], s[\\s_tmp4:\\s_tmp4+1], s[\\s_tmp4+2:\\s_tmp4+3]")
            self._emit("v_add_co_u32      v[\\v_tmp4+1],   vcc, -1,    v[\\v_tmp4+0]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+0],   v[\\v_tmp4+2],      s[\\s_tmp4+2:\\s_tmp4+3]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+1],   v[\\v_tmp4+2],      s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_cmp_ne_i32      vcc,          s[\\s_d],   0")
            self._emit("v_cndmask_b32     v[\\v_q],      -1,         v[\\v_tmp4+2],      vcc")

class macro_int_div_rem_vs_t(macro_base_t):
    '''
    integer divide to compute `v_q = v_n / s_d, v_r = v_n % s_d`, v_r, v_q, v_n are vgpr, s_d is sgpr
    '''
    def name(self):
        return '.v_u32_div_rem_vs'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, v_r, v_q, v_n, s_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}, {}'.format(self.name(), v_r, v_q, v_n, s_d, v_tmp4, s_tmp4)
    def emit(self):
        int_div_vs = macro_int_div_vs_t(self.mc)
        with self._emit_macro_indented(".macro {} v_r, v_q, v_n, s_d, v_tmp4, s_tmp4".format(self.name())):
            self._emit(int_div_vs("\\v_q", "\\v_n", "\\s_d", "\\v_tmp4", "\\s_tmp4"))
            self._emit(f"v_mul_lo_u32 v[\\v_tmp4], s[\\s_d], v[\\v_q]")
            self._emit(f"v_sub_u32 v[\\v_r], v[\\v_n], v[\\v_tmp4]")

class macro_int_div_ss_t(macro_base_t):
    '''
    integer divide to compute `s_q = s_n / s_d`, s_q, s_n, s_d all sgpr
    '''
    def name(self):
        return '.v_u32_div_ss'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, v_q, s_n, s_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}'.format(self.name(), v_q, s_n, s_d, v_tmp4, s_tmp4)
    def emit(self):
        with self._emit_macro_indented(".macro .v_u32_div_ss v_q, s_n, s_d, v_tmp4, s_tmp4"):
            self._emit("v_cvt_f32_u32     v[\\v_tmp4+0],   s[\\s_d]")
            self._emit("v_rcp_f32         v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_f32         v[\\v_tmp4+0],   0x4f800000, v[\\v_tmp4+0]")
            self._emit("v_cvt_u32_f32     v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],      v[\\v_tmp4+0]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+2],   s[\\s_d],      v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+3],   vcc, 0,     v[\\v_tmp4+1]")
            self._emit("v_cmp_ne_i32      s[\\s_tmp4:\\s_tmp4+1], 0,          v[\\v_tmp4+2]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+1],   v[\\v_tmp4+3],   v[\\v_tmp4+1],   s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+1],   v[\\v_tmp4+1],   v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_add_co_u32      v[\\v_tmp4+0],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_tmp4+2],   s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+0],   s[\\s_n],   v[\\v_tmp4+0]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],     v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        s[\\s_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_ge_u32      s[\\s_tmp4:\\s_tmp4+1], s[\\s_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_le_u32      s[\\s_tmp4+2:\\s_tmp4+3],  s[\\s_d],    v[\\v_tmp4+2]")
            self._emit("v_add_co_u32      v[\\v_tmp4+2],   vcc, 1, v[\\v_tmp4+0]")
            self._emit("s_and_b64         s[\\s_tmp4+2:\\s_tmp4+3], s[\\s_tmp4:\\s_tmp4+1], s[\\s_tmp4+2:\\s_tmp4+3]")
            self._emit("v_add_co_u32      v[\\v_tmp4+1],   vcc, -1,    v[\\v_tmp4+0]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+0],   v[\\v_tmp4+2],      s[\\s_tmp4+2:\\s_tmp4+3]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+1],   v[\\v_tmp4+2],      s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_cmp_ne_i32      vcc,          s[\\s_d],   0")
            self._emit("v_cndmask_b32     v[\\v_q],      -1,         v[\\v_tmp4+2],      vcc")

class macro_int_div_rem_ss_t(macro_base_t):
    '''
    integer divide to compute `s_q = s_n / s_d, s_r = s_n % s_d`, s_r, s_q, s_n, s_d all sgpr
    '''
    def name(self):
        return '.v_u32_div_rem_ss'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)

    def __call__(self, s_r, s_q, s_n, s_d, v_q, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}, {}, {}'.format(self.name(), s_r, s_q, s_n, s_d, v_q, v_tmp4, s_tmp4) 

    def emit(self):
        int_div_ss = macro_int_div_ss_t(self.mc)
        with self._emit_macro_indented(".macro {} s_r, s_q, s_n, s_d, v_q, v_tmp4, s_tmp4".format(self.name())):
            self._emit(int_div_ss("\\v_q", "\\s_n", "\\s_d", "\\v_tmp4", "\\s_tmp4"))
            self._emit(f"v_readfirstlane_b32 s[\\s_q], v[\\v_q]")
            self._emit(f"s_mul_i32 s[\\s_tmp4], s[\\s_d], s[\\s_q]")
            self._emit(f"s_sub_i32 s[\\s_r], s[\\s_n], s[\\s_tmp4]")


class macro_mdiv_u32_si_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_si'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("s_quot")
        self.declare_arg("s_numer")
        self.declare_arg("magic")
        self.declare_arg("shift")
        self.declare_arg("s_tmp")   # can be the same as s_quot
    def expr(self):
        self._emit(f"s_mul_hi_u32 s[{self.s_tmp()}], {self.magic()}, s[{self.s_numer()}]")
        self._emit(f"s_add_u32 s[{self.s_tmp()}], s[{self.s_tmp()}], s[{self.s_numer()}]")
        self._emit(f"s_lshr_b32 s[{self.s_quot()}], s[{self.s_tmp()}], {self.shift()}")
        

class macro_mdiv_u32_vi_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_vi'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_quot")
        self.declare_arg("v_numer")
        self.declare_arg("magic")
        self.declare_arg("shift")
        self.declare_arg("v_tmp")
        self.mc = mc
    def expr(self):
        self._emit(f"v_mul_hi_u32 v[{self.v_tmp()}], {self.magic()}, v[{self.v_numer()}]")
        self._emit(v_add_nc_u32(f"{self.v_tmp()}", f"{self.v_tmp()}", f"{self.v_numer()}"))
        self._emit(f"v_lshrrev_b32 v[{self.v_quot()}], {self.shift()}, v[{self.v_tmp()}]")
        
class div_u32_vi_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    
    def __call__(self, v_quot, v_numer, denom, v_tmp):
        assert isinstance(denom, int)
        with self._deferred_context():
            if utility_is_pow2(denom):
                self._emit(f"v_lshrrev_b32 v[{v_quot}], {utility_log2(denom)}, v[{v_numer}]")
            else:
                d_magic, d_shift = utility_division_magic(denom)
                mdiv_u32_vi = macro_mdiv_u32_vi_t(self.mc)
                self._emit(mdiv_u32_vi(v_quot, v_numer, str(d_magic), str(d_shift), v_tmp))
        return self._get_deferred()

class div_u32_si_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, s_quot, s_numer, denom, s_tmp):
        assert isinstance(denom, int)
        with self._deferred_context():
            if utility_is_pow2(denom):
                self._emit(f"s_lshr_b32 s[{s_quot}], s[{s_numer}], {utility_log2(denom)}")
            else:
                d_magic, d_shift = utility_division_magic(denom)
                mdiv_u32_si = macro_mdiv_u32_si_t(self.mc)
                self._emit(mdiv_u32_si(s_quot, s_numer, str(d_magic), str(d_shift), s_tmp))
        return self._get_deferred()

class mul_u32_si_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, s_o, s_i, multiplier):
        assert isinstance(multiplier, int)
        with self._deferred_context():
            if utility_is_pow2(multiplier):
                self._emit(f"s_lshl_b32 s[{s_o}], s[{s_i}], {utility_log2(multiplier)}")
            else:
                self._emit(f"s_mul_i32 s[{s_o}], s[{s_i}], {multiplier}")
        return self._get_deferred()

class macro_mdiv_u32_rem_vi_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_rem_vi'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_rem")
        self.declare_arg("v_quot")
        self.declare_arg("v_numer")
        self.declare_arg("magic")
        self.declare_arg("shift")
        self.declare_arg("denom")
        self.declare_arg("v_tmp")
    def expr(self):
        mdiv_u32_vi = macro_mdiv_u32_vi_t(self.mc, self.inline)
        self._emit(mdiv_u32_vi( self.v_quot(), self.v_numer(), self.magic(), self.shift(), self.v_tmp()  ))
        self._emit(f"v_mul_lo_u32 v[{self.v_tmp()}], {self.denom()}, v[{self.v_quot()}]")
        self._emit(v_sub_nc_u32(f"{self.v_rem()}", f"{self.v_numer()}", f"{self.v_tmp()}"))

class div_rem_u32_vi_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    
    def __call__(self, v_rem, v_quot, v_numer, denom, v_tmp):
        assert isinstance(denom, int)
        with self._deferred_context():
            if utility_is_pow2(denom):
                if v_quot != None:
                    self._emit(f"v_lshrrev_b32 v[{v_quot}], {utility_log2(denom)}, v[{v_numer}]")
                self._emit(f"v_and_b32 v[{v_rem}], {denom - 1}, v[{v_numer}]")
            else:
                if v_quot == None:
                    v_quot = v_tmp + "+1"
                d_magic, d_shift = utility_division_magic(denom)
                mdiv_rem_u32_vi = macro_mdiv_u32_rem_vi_t(self.mc)
                self._emit(mdiv_rem_u32_vi(v_rem, v_quot, v_numer, str(d_magic), str(d_shift), str(denom), v_tmp))
        return self._get_deferred()

class macro_mdiv_u32_ss_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_ss'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("s_quot")
        self.declare_arg("s_numer")
        self.declare_arg("s_magic")
        self.declare_arg("s_shift")
        self.declare_arg("s_tmp")
    def expr(self):
        self._emit(f"s_mul_hi_u32 s[{self.s_tmp()}], s[{self.s_magic()}], s[{self.s_numer()}]")
        self._emit(f"s_add_u32 s[{self.s_tmp()}], s[{self.s_tmp()}], s[{self.s_numer()}]")
        self._emit(f"s_lshr_b32 s[{self.s_quot()}], s[{self.s_tmp()}], s[{self.s_shift()}]")


class macro_mdiv_u32_rem_ss_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_rem_ss'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("s_rem")
        self.declare_arg("s_quot")
        self.declare_arg("s_numer")
        self.declare_arg("s_magic")
        self.declare_arg("s_shift")
        self.declare_arg("s_denom")
        self.declare_arg("s_tmp")
    def expr(self):
        mdiv_u32_ss = macro_mdiv_u32_ss_t(self.mc, self.inline)
        self._emit(mdiv_u32_ss(self.s_quot(), self.s_numer(), self.s_magic(), self.s_shift(), self.s_tmp()))
        self._emit(f"s_mul_i32 s[{self.s_tmp()}], s[{self.s_denom()}], s[{self.s_quot()}]")
        self._emit(f"s_sub_u32 s[{self.s_rem()}], s[{self.s_numer()}], s[{self.s_tmp()}]")


class macro_mdiv_u32_vs_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_vs'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_quot")
        self.declare_arg("v_numer")
        self.declare_arg("s_magic")
        self.declare_arg("s_shift")
        self.declare_arg("v_tmp")
        self.mc = mc
    def expr(self):
        self._emit(f"v_mul_hi_u32 v[{self.v_tmp()}], s[{self.s_magic()}], v[{self.v_numer()}]")
        if self.mc.arch_config.arch == AMDGPU_ARCH_GFX1030:
            self._emit(f"v_add_nc_u32 v[{self.v_tmp()}], v[{self.v_tmp()}], v[{self.v_numer()}]")
        else:
            self._emit(f"v_add_u32 v[{self.v_tmp()}], v[{self.v_tmp()}], v[{self.v_numer()}]")
        self._emit(f"v_lshrrev_b32 v[{self.v_quot()}], s[{self.s_shift()}], v[{self.v_tmp()}]")

class macro_mdiv_u32_rem_vs_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_rem_vs'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_rem")
        self.declare_arg("v_quot")
        self.declare_arg("v_numer")
        self.declare_arg("s_magic")
        self.declare_arg("s_shift")
        self.declare_arg("s_denom")
        self.declare_arg("v_tmp")
    def expr(self):
        mdiv_u32_vs = macro_mdiv_u32_vs_t(self.mc, self.inline)
        self._emit(mdiv_u32_vs( self.v_quot(), self.v_numer(), self.s_magic(), self.s_shift(), self.v_tmp()  ))
        self._emit(f"v_mul_lo_u32 v[{self.v_tmp()}], s[{self.s_denom()}], v[{self.v_quot()}]")
        if self.mc.arch_config.arch == AMDGPU_ARCH_GFX1030:
            self._emit(f"v_sub_nc_u32 v[{self.v_rem()}], v[{self.v_numer()}], v[{self.v_tmp()}]")
        else:
            self._emit(f"v_sub_u32 v[{self.v_rem()}], v[{self.v_numer()}], v[{self.v_tmp()}]")


class macro_c_clear_t(macro_base_t):
    def name(self):
        return '.v_clear_nc'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, vid, num):
        return '{} {}, {}'.format(self.name(), vid, num)
    def emit(self):
        with self._emit_macro_indented(".macro {} vid, num".format(self.name())):
            self._emit("_v = \\vid")
            self._emit(".rept \\num")
            with self._indent_context():
                self._emit("v_mov_b32 v[_v], 0")
                self._emit("_v = _v + 1")
            self._emit(".endr")

class macro_acc_c_clear_t(macro_base_t):
    '''
    gfx908 RAW harzard attention!
    '''
    def name(self):
        return '.v_clear_acc_c'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, a, num):
        return '{} {}, {}'.format(self.name(), a, num)
    def emit(self):
        with self._emit_macro_indented(".macro {} a, num".format(self.name())):
            self._emit("_a = \\a")
            self._emit(".rept \\num")
            with self._indent_context():
                self._emit("v_accvgpr_write_b32 a[_a], 0")
                self._emit("_a = _a + 1")
            self._emit(".endr")

class gpr_sequencer_t(object):
    def __init__(self, cnt = 0):
        self.cnt = cnt
    def __call__(self, step = 0, alignment = 0):
        previous_cnt = self.cnt
        if alignment:
            aligned_cnt = ((previous_cnt + alignment - 1) // alignment) * alignment
            self.cnt = aligned_cnt
            previous_cnt = aligned_cnt
        self.cnt += step
        return previous_cnt
    def get(self):
        return self.cnt

class macro_packed_fp16_to_bf16_t(macro_base_t):
    def __init__(self, mc, **options):
        macro_base_t.__init__(self, mc, True)
        self.options = options
        self.declare_arg("v_packed_f16")
        self.declare_arg("v_tmp")
        assert 'num_vgpr' in options

    def name(self):
        return '.v_packed_fp16_to_bf16'

    def expr(self):
        num_vgpr = self.options["num_vgpr"]
        for i in range(num_vgpr):
            self._emit(f"v_cvt_f32_f16 v[{self.v_tmp()}], v[{self.v_packed_f16(i)}]")
            self._emit(f"v_cvt_f32_f16 v[{self.v_packed_f16(i)}], v[{self.v_packed_f16(i)}] src0_sel:WORD_1")
            self._emit(f"v_pack_b32_f16 v[{self.v_packed_f16(i)}], v[{self.v_tmp()}], v[{self.v_packed_f16(i)}] op_sel:[1,1]")

def utility_list_to_string(arr):
    assert type(arr) is list
    return 'x'.join(f'{itm}' for itm in arr)

class utility_dict_with_default_t(object):
    def __init__(self, d):
        self.d = d
    def __call__(self, key, default_value):
        if self.d is None:
            return default_value
        if key in self.d:
            return self.d[key]
        return default_value

# compute next power of 2
def utility_next_pow2(n):
    if n == 0:
        return 1
    if n & (n - 1) == 0:
        return n
    while n & (n - 1) > 0:
        n &= (n - 1)
    return n << 1

def utility_next_mul(n, mul):
    d = n // mul
    d = d + (1 if (n % mul != 0) else 0)
    return d * mul

def utility_is_pow2(v):
    return v and (not(v & (v - 1)))

def utility_log2(v):
    assert (v and (not(v & (v - 1)))), 'v:{} must be power of 2'.format(v)
    return int(math.log2(v))

def utility_get_epack_length(precision):
        # GetEPackLength
        epack = 1
        if precision == AMDGPU_PRECISION_FP16:
            # todo: xdlops check
            epack = 2
        elif precision == AMDGPU_PRECISION_BF16:
            epack = 2
        return epack

def utility_gcd(a, b):
    # math.gcd new in python 3.5
    return math.gcd(a, b)

def utility_lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

def utility_flatten_list_product(x):
    assert type(x) is list
    from functools import reduce
    return reduce(lambda a, b: a*b, x, 1)

def utility_flatten_list_accumulate(x):
    assert type(x) is list
    from functools import reduce
    return reduce(lambda a, b: a+b, x, 0)

def utility_division_magic(divisor):
    '''
    compute magic num for fast int divison
    divisor <= pow(2, 31)
    '''
    assert(divisor <= pow(2, 31))
    magic_shift = 0
    for i in range(31):
        if pow(2, i) >= divisor:
            magic_shift = i
            break
    magic_num = int(pow(2, 32) * (pow(2, magic_shift) - divisor) / divisor) + 1
    return magic_num, magic_shift