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

class inst_dlops_t(object):
    def __init__(self, precision):
        self.precision = precision
    def __call__(self, reg_c, reg_a, reg_b):
        if self.precision == 'fp16':
            return 'v_dot2c_f32_f16 v[{}], v[{}], v[{}]'.format(reg_c, reg_a, reg_b)
        # xdlops
        else:
            assert False, 'unimplemented fma type'

class macro_dlops_mxn_t(macro_base_t):
    '''
    continuous fma, or strided fma
    TODO: implement any index-ed fma (for rdna)
    '''
    def name(self):
        return f".v_dlops_{self.precision}_{self.m}x{self.n}" + \
                ("" if self.stride == 1 else f"_s{self.stride}")

    def __init__(self, mc, m, n, stride, precision):
        macro_base_t.__init__(self, mc)
        self.m = m
        self.n = n
        self.stride = stride
        self.precision = precision
        assert stride >= n and stride % n == 0
    def __call__(self, c, a, b):
        return '{} {},{},{}'.format(self.name(), c, a, b)
    def emit(self):
        fma = inst_dlops_t(self.precision)
        reg_a = msym_t(sym_t('a'))
        reg_b = msym_t(sym_t('b'))
        reg_c = msym_t(sym_t('c'))
        with self._emit_macro_indented('.macro {} c, a, b'.format(self.name())):
            for idx_m in range(self.m):
                for idx_n in range(self.n):
                    self._emit(fma(reg_c(idx_m * self.stride + idx_n), reg_a(idx_m), reg_b(idx_n)))