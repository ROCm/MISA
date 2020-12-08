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

from ..codegen import *

MODIFIER_TYPE_NEG = 0
MODIFIER_TYPE_ABS = 1


class modifier_t(object):
    def __init__(self, operand, mtype):
        self.operand = operand
        self.mtype = mtype

def m_neg(operand):
    return modifier_t(operand, MODIFIER_TYPE_NEG)

def m_abs(operand):
    return modifier_t(operand, MODIFIER_TYPE_ABS)


class inst_mt_operand_t(object):
    '''
    multi-typed operand warpper.
    '''
    def __init__(self, operand):
        self.operand = operand
    def __call__(self):
        def expr_operand(opr):
            if type(self.operand) is str:
                if self.operand[0] == 's':
                    return f's[{self.operand}]'
                elif self.operand[0] == 'v':
                    if self.operand == 'vcc':
                        return self.operand
                    else:
                        return f'v[{self.operand}]'
                elif self.operand[0] == 'k':
                    return f'{self.operand}'
                else:
                    assert False, f"unknown operand:{self.operand}"
            else:
                return f'{self.operand}'

        if type(self.operand) is modifier_t:
            if self.operand.mtype == MODIFIER_TYPE_NEG:
                return f'neg({expr_operand(self.operand.operand)})'
            elif self.operand.mtype == MODIFIER_TYPE_ABS:
                return f'abs({expr_operand(self.operand.operand)})'
            else:
                assert False

        else:
            return expr_operand(self.operand)

def mt_opr(operand):
    return inst_mt_operand_t(operand)()

class inst_v_madmk_t(mc_base_t):
    '''
    v_madmk_f32, v_fmamk_f32
    '''
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, vdst, src0, imm32, vsrc1):
        if self.mc.arch_config.arch < 1000:
            return 'v_madmk_f32 v[{}], {}, {} v[{}]'.format(vdst, mt_opr(src0), imm32, vsrc1)
        else:
            # assume gfx10+
            return 'v_fmamk_f32 v[{}], {}, {} v[{}]'.format(vdst, mt_opr(src0), imm32, vsrc1)

class inst_v_add_nc_u32_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, vdst, src0, src1):
        if self.mc.arch_config.arch < 1000:
            return 'v_add_u32 v[{}], {}, {}'.format(vdst, mt_opr(src0), mt_opr(src1))
        else:
            return 'v_add_nc_u32 v[{}], {}, {}'.format(vdst, mt_opr(src0), mt_opr(src1))

class inst_v_sub_nc_u32_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, vdst, src0, src1):
        if self.mc.arch_config.arch < 1000:
            return 'v_sub_u32 v[{}], {}, {}'.format(vdst, mt_opr(src0), mt_opr(src1))
        else:
            return 'v_sub_nc_u32 v[{}], {}, {}'.format(vdst, mt_opr(src0), mt_opr(src1))

class inst_v_subrev_nc_u32_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, vdst, src0, src1):
        if self.mc.arch_config.arch < 1000:
            return 'v_subrev_u32 v[{}], {}, {}'.format(vdst, mt_opr(src0), mt_opr(src1))
        else:
            return 'v_subrev_nc_u32 v[{}], {}, {}'.format(vdst, mt_opr(src0), mt_opr(src1))

class inst_v_cmpx_eq_u32_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, dst, src0, src1):
        if self.mc.arch_config.arch < 1000:
            return 'v_cmpx_eq_u32 {}, {}, {}'.format(mt_opr(dst), mt_opr(src0), mt_opr(src1))
        else:
            return 'v_cmpx_eq_u32 {}, {}'.format(mt_opr(src0), mt_opr(src1))

class inst_v_cmpx_ne_u32_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, dst, src0, src1):
        if self.mc.arch_config.arch < 1000:
            return 'v_cmpx_ne_u32 {}, {}, {}'.format(mt_opr(dst), mt_opr(src0), mt_opr(src1))
        else:
            return 'v_cmpx_ne_u32 {}, {}'.format(mt_opr(src0), mt_opr(src1))

class inst_v_cmpx_gt_u32_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, dst, src0, src1):
        if self.mc.arch_config.arch < 1000:
            return 'v_cmpx_gt_u32 {}, {}, {}'.format(mt_opr(dst), mt_opr(src0), mt_opr(src1))
        else:
            return 'v_cmpx_gt_u32 {}, {}'.format(mt_opr(src0), mt_opr(src1))

class inst_v_cmpx_ge_u32_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, dst, src0, src1):
        if self.mc.arch_config.arch < 1000:
            return 'v_cmpx_ge_u32 {}, {}, {}'.format(mt_opr(dst), mt_opr(src0), mt_opr(src1))
        else:
            return 'v_cmpx_ge_u32 {}, {}'.format(mt_opr(src0), mt_opr(src1))

class inst_v_cmpx_lt_u32_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, dst, src0, src1):
        if self.mc.arch_config.arch < 1000:
            return 'v_cmpx_lt_u32 {}, {}, {}'.format(mt_opr(dst), mt_opr(src0), mt_opr(src1))
        else:
            return 'v_cmpx_lt_u32 {}, {}'.format(mt_opr(src0), mt_opr(src1))

class inst_v_cmpx_le_u32_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, dst, src0, src1):
        if self.mc.arch_config.arch < 1000:
            return 'v_cmpx_le_u32 {}, {}, {}'.format(mt_opr(dst), mt_opr(src0), mt_opr(src1))
        else:
            return 'v_cmpx_le_u32 {}, {}'.format(mt_opr(src0), mt_opr(src1))

class inst_v_cmpx_eq_i32_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, dst, src0, src1):
        if self.mc.arch_config.arch < 1000:
            return 'v_cmpx_eq_i32 {}, {}, {}'.format(mt_opr(dst), mt_opr(src0), mt_opr(src1))
        else:
            return 'v_cmpx_eq_i32 {}, {}'.format(mt_opr(src0), mt_opr(src1))
