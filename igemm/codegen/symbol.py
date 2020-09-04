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


class sym_t(object):
    '''
    symbol used in asm source, can use '.set <label>,   <value>'
    '''
    def __init__(self, label, value = 0, comments = ''):
        assert type(label) is str
        assert type(value) is int
        self.label = label
        self.value = value
        self.comments = comments
    def declare(self):
        comments_str = '' if self.comments == '' else f'  ; {self.comments}'
        return f'.set {self.label}, {self.value}{comments_str}'
    @staticmethod
    def expr(label, index = 0):
        if type(index) is int:
            if index == 0:
                return label
            return f'{label}+{index}'
        elif type(index) is tuple:
            assert len(index) == 2, 'expect tuple (start-index, end-index), inclusive'
            return f'{label}+{index[0]}:{label}+{index[1]}'
        else:
            assert False

    def __call__(self, index = 0):
        return self.expr(self.label, index)
    
    def __eq__(self, other):
        return self.label == other.label and self.value == other.value
    def __ne__(self, other):
        return not self == other

class msym_t(object):
    """ reference a symbol inside macro """
    def __init__(self, sym):
        assert type(sym) is sym_t
        self.sym = sym
        self.label_in_macro = f'\\{sym.label}'

    def __call__(self, index = 0):
        return self.sym.expr(self.label_in_macro, index)
