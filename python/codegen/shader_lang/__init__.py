################################################################################
# 
#  MIT License
# 
#  Copyright (c) 2022 Advanced Micro Devices, Inc.
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

import string
from .lang_api import *

__extension_available = False
try:
    __import__('imp').find_module('extension.lang')
    __extension_available = True
    #import extension.lang
except ImportError:
    pass


def get_kernel_lang_class(self, kernel_info, **kwargs) -> base_lang_api:
    lang = kwargs.get('lang', None)
    if(lang == 'llvm-asm' or lang == None):
        from ..shader_lang.llvm_asm import llvm_kernel
        return llvm_kernel(self.mc, kernel_info, self.instr_ctrl._emmit_created_code)
    elif(__extension_available):
        import extension.lang
        ret = extension.lang.get_kernel_lang_class(self, kernel_info, **kwargs)
        if (ret != None):
            return  ret
    pass


