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

LANEGROUP_SIZE   = 8

class inst_dotx_vop2_t(inst_base_t):
    def __init__(self, name, k, data_type):
        inst_base_t.__init__(self, INST_ENCODING_VOP2)
        self.name = name
        self.k = k
        self.data_type = data_type
    
    def __call__(self, vdst, vsrc0, vsrc1, dpp8 = list(), fi = 0):
        modifier = ''
        if len(dpp8) != 0:
            modifier += ' dpp8:[{}]'.format(','.join(str(i) for i in dpp8))
        if fi != 0:
            modifier += ' fi:{}'.format(fi)
        return f'{self.name} v[{vdst}], v[{vsrc0}], v[{vsrc1}]' + modifier

class inst_dotx_vop3p_t(inst_base_t):
    def __init__(self, name, k, data_type):
        inst_base_t.__init__(self, INST_ENCODING_VOP3P)
        self.name = name
        self.k = k
        self.data_type = data_type
    
    def __call__(self, vdst, src0, src1, src2):
        return f'{self.name} v[{vdst}], v[{src0}], v[{src1}], v[{src2}]'

v_dot2c_f32_f16 = inst_dotx_vop2_t('v_dot2c_f32_f16',  2, AMDGPU_PRECISION_FP16)
v_dot4c_i32_i8  = inst_dotx_vop2_t('v_dot4c_i32_i8' ,  4, AMDGPU_PRECISION_INT8)
v_dot2_f32_f16  = inst_dotx_vop3p_t('v_dot2_f32_f16',  2, AMDGPU_PRECISION_FP16)
v_dot2_i32_i16  = inst_dotx_vop3p_t('v_dot2_i32_i16',  2, AMDGPU_PRECISION_INT16)
v_dot2_u32_u16  = inst_dotx_vop3p_t('v_dot2_u32_u16',  2, AMDGPU_PRECISION_UINT16)
v_dot4_i32_i8   = inst_dotx_vop3p_t('v_dot4_i32_i8' ,  4, AMDGPU_PRECISION_INT8)
v_dot4_u32_u8   = inst_dotx_vop3p_t('v_dot4_u32_u8' ,  4, AMDGPU_PRECISION_UINT8)
v_dot8_i32_i4   = inst_dotx_vop3p_t('v_dot8_i32_i4' ,  8, AMDGPU_PRECISION_INT4)
v_dot8_u32_u4   = inst_dotx_vop3p_t('v_dot8_u32_u4' ,  8, AMDGPU_PRECISION_UINT4)
v_fmac_f32      = inst_dotx_vop2_t('v_fmac_f32' ,  1, AMDGPU_PRECISION_FP32)   # TODO: maybe not call it dotx?

def dotx_support_dpp8(inst):
    if isinstance(inst, inst_dotx_vop2_t):
        if inst.k == 1:
            return False
        else:
            return True
    elif isinstance(inst, inst_dotx_vop3p_t):
        return False
    else:
        # assert False
        return False

class macro_dotx_mxn_t(macro_base_t):
    '''
    the size is always in unit of lanegroup
    '''
    def name(self):
        return f".v_lanegroup_dotx_{self.precision}_{self.lanegroup_tile_m}x{self.lanegroup_tile_n}x{self.lanegroup_tile_k}" + \
                ("" if self.stride == 1 else f"_s{self.stride}")

    def __init__(self, mc, lanegroup_tile_m, lanegroup_tile_n, stride, precision):
        macro_base_t.__init__(self, mc)
        assert lanegroup_tile_m % LANEGROUP_SIZE == 0 and lanegroup_tile_n % LANEGROUP_SIZE == 0
        self.lanegroup_tile_m = lanegroup_tile_m
        self.lanegroup_tile_n = lanegroup_tile_n
        self.stride = stride
        self.precision = precision
        #assert stride >= lanegroup_tile_n and stride % lanegroup_tile_n == 0
        self.lanegroup_tile_k = self.get_dotx_instruction().k
    def __call__(self, c, a, b):
        return '{} {},{},{}'.format(self.name(), c, a, b)
    
    def get_dotx_instruction(self):
        if self.precision == 'fp16':
            return v_dot2c_f32_f16
        elif self.precision == 'int8':
            return v_dot4c_i32_i8
        else:
            assert False

    def emit(self):
        reg_a = msym_t(sym_t('a'))
        reg_b = msym_t(sym_t('b'))
        reg_c = msym_t(sym_t('c'))
        with self._emit_macro_indented('.macro {} c, a, b'.format(self.name())):
            for idx_m in range(self.lanegroup_tile_m // LANEGROUP_SIZE):
                for idx_n in range(self.lanegroup_tile_n // LANEGROUP_SIZE):
                    for idx_dpp in range(LANEGROUP_SIZE):
                        if self.precision == 'fp16':
                            self._emit(v_dot2c_f32_f16(reg_c(idx_m * self.stride + idx_n + idx_dpp), reg_a(idx_m), reg_b(idx_n), [idx_dpp] * 8))
                        elif self.precision == 'int8':
                            self._emit(v_dot4c_i32_i8(reg_c(idx_m * self.stride + idx_n + idx_dpp), reg_a(idx_m), reg_b(idx_n), [idx_dpp] * 8))
                        else:
                            assert False

class macro_dotx_mxnxk_t(macro_base_t):
    '''
    continuous fma, or strided fma
    TODO: implement any index-ed fma (for rdna)
    '''
    def name(self):
        return f".v_lanegroup_dotx_{self.precision}_{self.lanegroup_tile_m}x{self.lanegroup_tile_n}x{self.lanegroup_tile_k}" + \
                ("" if self.stride == 1 else f"_s{self.stride}")

    def __init__(self, mc, lanegroup_tile_m, lanegroup_tile_n, lanegroup_tile_k, stride, precision):
        macro_base_t.__init__(self, mc)
        self.lanegroup_tile_m = lanegroup_tile_m
        self.lanegroup_tile_n = lanegroup_tile_n
        self.lanegroup_tile_k = lanegroup_tile_k
        self.stride = stride
        self.precision = precision
        #assert stride >= lanegroup_tile_n and stride % lanegroup_tile_n == 0
    def __call__(self, c, a, b):
        return '{} {},{},{}'.format(self.name(), c, a, b)
    def emit(self):
        reg_a = msym_t(sym_t('a'))
        reg_b = msym_t(sym_t('b'))
        reg_c = msym_t(sym_t('c'))
        v_dotx_mxn = macro_dotx_mxn_t(self.mc, self.lanegroup_tile_m, self.lanegroup_tile_n, self.stride, self.precision)
        with self._emit_macro_indented('.macro {} c, a, b'.format(self.name())):
            for idx_k in range(self.lanegroup_tile_k // v_dotx_mxn.lanegroup_tile_k):
                self._emit(v_dotx_mxn(reg_c(), reg_a(idx_k), reg_b(idx_k)))
                
class macro_dotx_mxnxk_non_dpp_t(macro_base_t):
    '''
    the size is always in unit of thread
    '''
    def name(self):
        return f".v_dotx_{self.precision}_{self.thread_tile_m}x{self.thread_tile_n}x{self.thread_tile_k}" + \
                ("" if self.stride == 1 else f"_s{self.stride}")

    def __init__(self, mc, thread_tile_m, thread_tile_n, thread_tile_k, stride, precision):
        macro_base_t.__init__(self, mc)
        self.thread_tile_m = thread_tile_m
        self.thread_tile_n = thread_tile_n
        self.stride = stride
        self.precision = precision
        self.thread_tile_k = thread_tile_k 
        self.loop_k = thread_tile_k // self.get_dotx_instruction().k
    def __call__(self, c, a, b):
        return '{} {},{},{}'.format(self.name(), c, a, b)
    
    def get_dotx_instruction(self):
        if self.precision == 'int4':
            return v_dot8_i32_i4
        elif self.precision == 'int8':
            return v_dot4_i32_i8
        elif self.precision == 'fp16':
            return v_dot2_f32_f16
        else:
            assert False

    def emit(self):
        reg_a = msym_t(sym_t('a'))
        reg_b = msym_t(sym_t('b'))
        reg_c = msym_t(sym_t('c'))
        v_dot = self.get_dotx_instruction()
        with self._emit_macro_indented('.macro {} c, a, b'.format(self.name())):
            for idx_loop_k in range(self.loop_k):              
                for idx_n in range(self.thread_tile_n):
                    for idx_m in range(self.thread_tile_m):
                        idx_c = idx_m * self.stride + idx_n * self.thread_tile_m
                        idx_a = idx_m * self.loop_k + idx_loop_k
                        idx_b = idx_n * self.loop_k + idx_loop_k
                        self._emit(v_dot(reg_c(idx_c), reg_a(idx_a), reg_b(idx_b), reg_c(idx_c)))

