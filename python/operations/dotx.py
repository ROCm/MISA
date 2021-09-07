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
