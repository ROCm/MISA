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

class dotx_core_loop_expr(mc_base_t):
    def __init__(self, mc, name, func=None) -> None:
        mc_base_t.__init__(self, mc)
        self.func = func
        self.name = name
        
    def expr_ir(self):
        return self.name
    
    def expr_asm_codes(self):
        return self.func()
    
    def emit_expr_asm_codes(self):
        self._emit(self.func())

class dotx_core_loop_node(mc_base_t):
    def __init__(self, mc, name) -> None:
        mc_base_t.__init__(self, mc)
        self.name = name
        self.first = None
        self.second = None
        
    def get_first_node(self):
        return self.first
    
    def get_second_node(self):
        return self.second
    
    def set_next_node(self, first):
        self.first = first
        
    def set_second_node(self, second):
        self.second = second
    
def print_ir(node):
    if isinstance(node, dotx_core_loop_expr):
        print(node.expr_ir())
        return
    else:
        print_ir(node.first)
        print_ir(node.second)
        return
    
def print_asm(node):
    if isinstance(node, dotx_core_loop_expr):
        print(node.expr_asm_codes())
        return
    else:
        print_asm(node.first)
        print_asm(node.second)
        return
    
def emit_asm(node):
    if isinstance(node, dotx_core_loop_expr):
        node.emit_expr_asm_codes()
        return
    else:
        emit_asm(node.first)
        emit_asm(node.second)
        return
        
class dotx_core_loop_for_loop(dotx_core_loop_node):
    def __init__(self, mc, name, loop_var="", loop_begin=None, jump_expr=None, loop_end=None, loop_check=None, loop_stmt=None) -> None:
        super().__init__(mc, name)
        self.loop_var = loop_var
        self.loop_begin = loop_begin
        self.jump_expr = jump_expr
        self.loop_end = loop_end
        self.loop_check = loop_check
        self.loop_stmt = loop_stmt
        self.name = name

class dotx_core_loop_graph(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        self.ctrl = ctrl
        
    def creat_base_graph(self):
        label_fma_body = 'L_{}_fma_body'.format(self.ctrl.label_prefix)
        label_fma_finishing = 'L_{}_fma_finishing'.format(self.ctrl.label_prefix)
        label_fma_end = 'L_{}_end'.format(self.ctrl.label_prefix)
        
        f_gld_a = self.ctrl.global_load_a_functor
        f_gld_b = self.ctrl.global_load_b_functor
        f_sst_a = self.ctrl.shared_store_a_functor
        f_sst_b = self.ctrl.shared_store_b_functor

        f_sld_a = self.ctrl.shared_load_a_functor
        f_sld_b = self.ctrl.shared_load_b_functor

        f_move_slice_window_a = self.ctrl.move_slice_window_a_functor
        f_move_slice_window_b = self.ctrl.move_slice_window_b_functor

        v_a = self.ctrl.v_a
        v_b = self.ctrl.v_b
        v_c = self.ctrl.v_c

        v_gld_a = self.ctrl.v_gld_a
        v_gld_b = self.ctrl.v_gld_b

        v_sst_a_os = self.ctrl.v_sst_a_os
        v_sld_a_os = self.ctrl.v_sld_a_os
        v_sst_b_os = self.ctrl.v_sst_b_os
        v_sld_b_os = self.ctrl.v_sld_b_os

        s_kitr = self.ctrl.s_kitr
        s_knum = self.ctrl.s_knum
        dotx_m = self.ctrl.dotx_m
        
        base_node = dotx_core_loop_node(mc, "core_loop")
        node_clear_c = dotx_core_loop_expr(mc, None, ".clear_c")
        
        base_for_loop = dotx_core_loop_for_loop(mc, "core_loop")
        
        sld_a = dotx_core_loop_expr(mc, None, "sld_a")
        sld_b = dotx_core_loop_expr(mc, None, "sld_b")
        sst_a = dotx_core_loop_expr(mc, None, "sst_a")
        sst_b = dotx_core_loop_expr(mc, None, "sst_b")
        
    
         
if __name__ == "__main__":
    asm_target = os.path.join('out', 'core_loop_test.s')
    emitter = mc_emit_to_file_t(asm_target)
    arch = amdgpu_arch_config_t({
        'arch'          :   'gfx1030',
        'data_type'     :   AMDGPU_PRECISION_FP32,
        'code_object'   :   'cov3'})

    # create mc
    mc = mc_asm_printer_t(emitter, arch)
    mc_set_current(mc)
    sld_a = dotx_core_loop_expr(mc, None, "sld_a")
    sld_b = dotx_core_loop_expr(mc, None, "sld_b")
    sst_a = dotx_core_loop_expr(mc, None, "sst_a")
    sst_b = dotx_core_loop_expr(mc, None, "sst_b")
    core_loop = dotx_core_loop_node(mc, "core_loop")
    core_loop_0 = dotx_core_loop_node(mc, "core_loop_0")
    core_loop_1 = dotx_core_loop_node(mc, "core_loop_1")
    core_loop.first = sld_a
    core_loop.second = core_loop_0
    core_loop_0.first = sld_b
    core_loop_0.second = core_loop_1
    core_loop_1.first = sst_a
    core_loop_1.second = sst_b
    print_ir(core_loop)
    
        