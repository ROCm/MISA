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

from re import S
from ..codegen import *
from .utility import *
from .dotx_mapping import *
from .dotx import *
from .main_loop_graph import *
    
class dotx_main_loop_t(mc_base_t):
    '''
    TODO: 
    '''
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        self.ctrl = ctrl
        self.graph = dotx_core_loop_graph(ctrl, mc)
        
    def print_ir(self, node):
        if isinstance(node, dotx_core_loop_expr):
            print(node.expr_ir())
            return
        else:
            print_ir(node.first)
            print_ir(node.second)
            return
    
    def emit_graph(self, node):
        if isinstance(node, dotx_core_loop_expr):
            if "label" in node.name:
                self._emit_front(node.expr_asm_codes())
            else:
                self._emit(node.expr_asm_codes())
            return
        elif isinstance(node, dotx_core_loop_node):
            #print(node.name)
            self.emit_graph(node.first)
            self.emit_graph(node.second)
            return
        else:
            assert False, f"wrong node"
        
    def emit(self):
        self.graph.creat_base_graph()
        #self.print_ir(self.graph.base_node)
        self.emit_graph(self.graph.base_node)
