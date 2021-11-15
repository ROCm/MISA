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
import copy

def inst_mfma_data_type_to_string(data_type):
    if data_type == AMDGPU_PRECISION_FP32:
        return 'fp32'
    if data_type == AMDGPU_PRECISION_FP16:
        return 'fp16'
    if data_type == AMDGPU_PRECISION_BF16:
        return 'bf16'
    if data_type == AMDGPU_PRECISION_INT8:
        return 'int8'
    assert False

class inst_mfma_t(object):
    '''
    http://llvm.org/docs/AMDGPU/AMDGPUAsmGFX908.html
    '''
    def __init__(self, m, n, k, data_type, cycle, num_v_a, num_v_b, num_a_c, num_blocks, **options):
        #self.arch_config = arch_config
        self.m = m
        self.n = n
        self.k = k
        self.data_type = data_type
        self.cycle = cycle
        self.num_v_a = num_v_a
        self.num_v_b = num_v_b
        self.num_a_c = num_a_c
        self.num_blocks = num_blocks
        self.accvgpr_unified = False
        self.options = options
        # self.num_a_c_per_lanegroup = 4      # all xdlops instruction output agpr is 4 agpr per lanegroup.
        #assert arch_config.arch == AMDGPU_ARCH_GFX908 and arch_config.use_xdlops

    def name(self):
        if 'name' in self.options and self.options['name'] != None:
            return self.options['name']
        def src_datatype_string(data_type_string):
            if data_type_string == 'fp32':
                return 'f32'
            if data_type_string == 'fp16':
                return 'f16'
            if data_type_string == 'bf16':
                return 'bf16'
            if data_type_string == 'int8':
                return 'i8'
            assert False, f"unknow type :{data_type_string}"
        mfma_acc_type = 'i32' if self.data_type == AMDGPU_PRECISION_INT8 else 'f32' # TODO: int8 mfma accumulate type is i32
        mfma_trait = f'{self.m}x{self.n}x{self.k}' + src_datatype_string(inst_mfma_data_type_to_string(self.data_type))
        mfma_inst = f'v_mfma_{mfma_acc_type}_{mfma_trait}'
        if 'bf16_1k' in self.options and self.options['bf16_1k'] and self.data_type == AMDGPU_PRECISION_BF16:
            mfma_inst += '_1k'
        return mfma_inst

    def __call__(self, reg_d, reg_a, reg_b, reg_c, cbsz=0, abid=0, blgp=0):
        mfma_inst = self.name()
        cbsz_str = f"cbsz:{cbsz}" if cbsz != 0 else ""
        abid_str = f"abid:{abid}" if abid != 0 else ""
        blgp_str = f"blgp:{blgp}" if blgp != 0 else ""
        if self.accvgpr_unified:
            return  f"{mfma_inst} v[{reg_d}], v[{reg_a}], v[{reg_b}], v[{reg_c}] {cbsz_str} {abid_str} {blgp_str}"
        else:
            return  f"{mfma_inst} a[{reg_d}], v[{reg_a}], v[{reg_b}], a[{reg_c}] {cbsz_str} {abid_str} {blgp_str}"

    def get_nop_count_mfma_acc_raw(self):
        # in unit of passes, aka 4 cycle
        return (self.cycle // 4) + 2


#                                     m,  n,  k,  precision,           cycle, v_a, v_b, a_c, #block
v_mfma_f32_4x4x1f32     = inst_mfma_t(4,  4,  1,  AMDGPU_PRECISION_FP32,   8,   1,   1,  4,    16)
v_mfma_f32_16x16x1f32   = inst_mfma_t(16, 16, 1,  AMDGPU_PRECISION_FP32,  32,   1,   1,  16,   4 )
v_mfma_f32_16x16x4f32   = inst_mfma_t(16, 16, 4,  AMDGPU_PRECISION_FP32,  32,   1,   1,  4,    1 )
v_mfma_f32_32x32x1f32   = inst_mfma_t(32, 32, 1,  AMDGPU_PRECISION_FP32,  64,   1,   1,  32,   2 )
v_mfma_f32_32x32x2f32   = inst_mfma_t(32, 32, 2,  AMDGPU_PRECISION_FP32,  64,   1,   1,  16,   1 )

v_mfma_f32_4x4x4f16     = inst_mfma_t(4,  4,  4,  AMDGPU_PRECISION_FP16,   8,   2,   2,  4,    16)
v_mfma_f32_16x16x4f16   = inst_mfma_t(16, 16, 4,  AMDGPU_PRECISION_FP16,  32,   2,   2,  16,   4 )
v_mfma_f32_16x16x16f16  = inst_mfma_t(16, 16, 16, AMDGPU_PRECISION_FP16,  32,   2,   2,  4,    1 )
v_mfma_f32_32x32x4f16   = inst_mfma_t(32, 32, 4,  AMDGPU_PRECISION_FP16,  64,   2,   2,  32,   2 )
v_mfma_f32_32x32x8f16   = inst_mfma_t(32, 32, 8,  AMDGPU_PRECISION_FP16,  64,   2,   2,  16,   1 )

v_mfma_f32_4x4x2bf16    = inst_mfma_t(4,  4,  2,  AMDGPU_PRECISION_BF16,   8,   1,   1,  4,    16)
v_mfma_f32_16x16x2bf16  = inst_mfma_t(16, 16, 2,  AMDGPU_PRECISION_BF16,  32,   1,   1,  16,   4 )
v_mfma_f32_16x16x8bf16  = inst_mfma_t(16, 16, 8,  AMDGPU_PRECISION_BF16,  32,   1,   1,  4,    1 )
v_mfma_f32_32x32x2bf16  = inst_mfma_t(32, 32, 2,  AMDGPU_PRECISION_BF16,  64,   1,   1,  32,   2 )
v_mfma_f32_32x32x4bf16  = inst_mfma_t(32, 32, 4,  AMDGPU_PRECISION_BF16,  64,   1,   1,  16,   1 )

v_mfma_i32_4x4x4i8      = inst_mfma_t(4,  4,  4,  AMDGPU_PRECISION_INT8,   8,   1,   1,  4,    16)
v_mfma_i32_16x16x4i8    = inst_mfma_t(16, 16, 4,  AMDGPU_PRECISION_INT8,  32,   1,   1,  16,   4 )
v_mfma_i32_16x16x16i8   = inst_mfma_t(16, 16, 16, AMDGPU_PRECISION_INT8,  32,   1,   1,  4,    1 )
v_mfma_i32_32x32x4i8    = inst_mfma_t(32, 32, 4,  AMDGPU_PRECISION_INT8,  64,   1,   1,  32,   2 )
v_mfma_i32_32x32x8i8    = inst_mfma_t(32, 32, 8,  AMDGPU_PRECISION_INT8,  64,   1,   1,  16,   1 )

v_mfma_f32_4x4x4bf16_1k     = inst_mfma_t(4,  4,  4,  AMDGPU_PRECISION_BF16,   8,   2,   2,  4,    16, bf16_1k=True)
v_mfma_f32_16x16x4bf16_1k   = inst_mfma_t(16, 16, 4,  AMDGPU_PRECISION_BF16,  32,   2,   2,  16,   4 , bf16_1k=True)
v_mfma_f32_16x16x16bf16_1k  = inst_mfma_t(16, 16, 16, AMDGPU_PRECISION_BF16,  32,   2,   2,  4,    1 , bf16_1k=True)
v_mfma_f32_32x32x4bf16_1k   = inst_mfma_t(32, 32, 4,  AMDGPU_PRECISION_BF16,  64,   2,   2,  32,   2 , bf16_1k=True)
v_mfma_f32_32x32x8bf16_1k   = inst_mfma_t(32, 32, 8,  AMDGPU_PRECISION_BF16,  64,   2,   2,  16,   1 , bf16_1k=True)

v_mfma_f32_4x4x4_16f_m      = inst_mfma_t(4,  4,  4,  AMDGPU_PRECISION_BF16,   8,   2,   2,  4,    16, bf16_1k=True, name='v_mfma_f32_4x4x4_16f_m')
v_mfma_f32_16x16x4_16f_m    = inst_mfma_t(16, 16, 4,  AMDGPU_PRECISION_BF16,  32,   2,   2,  16,   4 , bf16_1k=True, name='v_mfma_f32_16x16x4_16f_m')
v_mfma_f32_16x16x16_16f_m   = inst_mfma_t(16, 16, 16, AMDGPU_PRECISION_BF16,  32,   2,   2,  4,    1 , bf16_1k=True, name='v_mfma_f32_16x16x16_16f_m')
v_mfma_f32_32x32x4_16f_m    = inst_mfma_t(32, 32, 4,  AMDGPU_PRECISION_BF16,  64,   2,   2,  32,   2 , bf16_1k=True, name='v_mfma_f32_32x32x4_16f_m')
v_mfma_f32_32x32x8_16f_m    = inst_mfma_t(32, 32, 8,  AMDGPU_PRECISION_BF16,  64,   2,   2,  16,   1 , bf16_1k=True, name='v_mfma_f32_32x32x8_16f_m')

def inst_mfma_emit_macro_mfma_16f(mc, predefined_symbol_bf16_enable, default_value):
    mc.emit(f'.ifndef {predefined_symbol_bf16_enable}')
    mc.emit(f'.set {predefined_symbol_bf16_enable}, {default_value}')
    mc.emit(f'.endif')
    mc.emit_empty_line()

    the_list = [v_mfma_f32_4x4x4_16f_m, v_mfma_f32_16x16x4_16f_m, v_mfma_f32_16x16x16_16f_m, v_mfma_f32_32x32x4_16f_m, v_mfma_f32_32x32x8_16f_m]

    for inst in the_list:
        inst_16f = copy.deepcopy(inst)
        inst_16f.options['name'] = None
        # print(f'{inst.options}')
        inst_16f.data_type = AMDGPU_PRECISION_BF16
        macro_name = inst.options['name']
        mc.emit(f'.macro {macro_name} d, a, b, c')
        mc.emit(f'.if {predefined_symbol_bf16_enable} == 1')
        mc.emit(f'    {inst_16f.name()} \\d, \\a, \\b, \\c')
        mc.emit(f'.else')
        inst_16f.data_type = AMDGPU_PRECISION_FP16
        mc.emit(f'    {inst_16f.name()} \\d, \\a, \\b, \\c')
        mc.emit(f'.endif')
        mc.emit(f'.endm')
        mc.emit_empty_line()

# class inst_composed_mfma_t(object):
#     '''
#     handy class to issue several mfma to form a wave wise mxn
#     '''
#     pass
# 
# class inst_composed_mfma_f32_64x64x1f32_t(object):
#     def __init__(self):
#         self.m = 64
#         self.n = 64
#         self.k = 1
#         self.data_type = AMDGPU_PRECISION_FP32
#         self.mfma = v_mfma_f32_32x32x1f32
#     def issues(self):
#         return 2
#     def issue0(self, reg_d, reg_a, reg_b, reg_c):
#         return self.mfma(reg_d, reg_a, reg_b, reg_c, 1, 0, 0)
#     def issue1(self, reg_d, reg_a, reg_b, reg_c):
#         return self.mfma(reg_d, reg_a, reg_b, reg_c, 1, 1, 0)
#     def __call__(self, reg_d, reg_a, reg_b, reg_c):
#         with self._deferred_context():
#             self._emit(self.issue0(reg_d, reg_a, reg_b, reg_c))
#             self._emit(self.issue1(reg_d + '+32', reg_a, reg_b, reg_c))
#         return self._get_deferred()
# 
# class inst_composed_mfma_f32_32x64x1f32_t(object):
#     def __init__(self):
#         self.m = 32
#         self.n = 64
#         self.k = 1
#         self.data_type = AMDGPU_PRECISION_FP32
#         self.mfma = v_mfma_f32_32x32x1f32
#     def issues(self):
#         return 1
#     def issue0(self, reg_d, reg_a, reg_b, reg_c):
#         return self.mfma(reg_d, reg_a, reg_b, reg_c, 1, 0, 0)
#     def __call__(self, reg_d, reg_a, reg_b, reg_c):
#         return self.issue0(reg_d, reg_a, reg_b, reg_c)
# 
# class inst_composed_mfma_f32_64x32x1f32_t(object):
#     def __init__(self):
#         self.m = 64
#         self.n = 32
#         self.k = 1
#         self.data_type = AMDGPU_PRECISION_FP32
#         self.mfma = v_mfma_f32_32x32x1f32
#     def issues(self):
#         return 1
#     def issue0(self, reg_d, reg_a, reg_b, reg_c):
#         return self.mfma(reg_d, reg_a, reg_b, reg_c, 0, 0, 1)
#     def __call__(self, reg_d, reg_a, reg_b, reg_c):
#         return self.issue0(reg_d, reg_a, reg_b, reg_c)
# 
# class inst_composed_mfma_f32_16x64x1f32_t(object):
#     def __init__(self):
#         self.m = 16
#         self.n = 64
#         self.k = 1
#         self.data_type = AMDGPU_PRECISION_FP32
#         self.mfma = v_mfma_f32_16x16x1f32
#     def issues(self):
#         return 1
#     def issue0(self, reg_d, reg_a, reg_b, reg_c):
#         return self.mfma(reg_d, reg_a, reg_b, reg_c, 2, 0, 0)
#     def __call__(self, reg_d, reg_a, reg_b, reg_c):
#         return self.issue0(reg_d, reg_a, reg_b, reg_c)
# 
# class inst_composed_mfma_f32_64x16x1f32_t(object):
#     def __init__(self):
#         self.m = 64
#         self.n = 16
#         self.k = 1
#         self.data_type = AMDGPU_PRECISION_FP32
#         self.mfma = v_mfma_f32_16x16x1f32
#     def issues(self):
#         return 1
#     def issue0(self, reg_d, reg_a, reg_b, reg_c):
#         return self.mfma(reg_d, reg_a, reg_b, reg_c, 0, 0, 4)
#     def __call__(self, reg_d, reg_a, reg_b, reg_c):
#         return self.issue0(reg_d, reg_a, reg_b, reg_c)
# 
# v_mfma_f32_64x64x1f32   = inst_composed_mfma_f32_64x64x1f32_t()
# v_mfma_f32_32x64x1f32   = inst_composed_mfma_f32_32x64x1f32_t()
# v_mfma_f32_64x32x1f32   = inst_composed_mfma_f32_64x32x1f32_t()
# v_mfma_f32_16x64x1f32   = inst_composed_mfma_f32_16x64x1f32_t()
# v_mfma_f32_64x16x1f32   = inst_composed_mfma_f32_64x16x1f32_t()
