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

from .symbol import *
from .mc import *

class macro_base_t(mc_base_t):
    '''
    base class of a macro
    a macro can be inline-ed, which means no need to generate .macro ..... .endm and then call this macro
    '''
    def __init__(self, mc, inline = False):
        mc_base_t.__init__(self, mc)
        self.arg_list = list()
        self.inline = inline
        self.expr_cnt = 0
    def name(self):
        return 'n/a macro'
    def is_inline(self):
        return self.inline

    def _declare_arg(self, arg_name):
        assert type(arg_name) is str
        if self.is_inline():
            setattr(self, arg_name, sym_t(arg_name))
        else:
            setattr(self, arg_name, msym_t(sym_t(arg_name)))

    def declare_arg(self, arg_name):
        '''
        we want to call this function when child class __init__(), that means all args are statically created
        can't suport both inline and non-inline
        '''
        self._declare_arg(arg_name)
        self.arg_list.append(arg_name)

    def expr(self):
        '''
        child class should overload this function to type kernel body
        child class overload this function must not have deferred_context
        '''
        pass
    def __call__(self, *args):
        assert len(args) == len(self.arg_list), f'parse in args:{args} not equal to arg_list:{self.arg_list}'
        if self.is_inline():
            # 1st, overwrite original declared arguments
            for i in range(len(args)):
                if type(args[i]) is str:
                    setattr(self, self.arg_list[i], sym_t(args[i]))
                elif type(args[i]) is sym_t:
                    setattr(self, self.arg_list[i], args[i])
                elif args[i] is None:
                    setattr(self, self.arg_list[i], None)

            # 2nd, do the emit
            with self._deferred_context():
                self.expr()
            self.expr_cnt += 1

            # last, restore arg to default value.
            for a in self.arg_list:
                self._declare_arg(a)

            return self._get_deferred()
        else:
            for x in args:
                assert type(x) is str, f'call this macro need parse in string!, {x}:{type(x)}'
            return '{} {}'.format(self.name(), ','.join(args))

    def emit(self):
        if not self.is_inline():
            with self._emit_macro_indented(".macro {} {}".format(self.name(), ' '.join(self.arg_list))):
                self.expr()
                self.expr_cnt += 1
