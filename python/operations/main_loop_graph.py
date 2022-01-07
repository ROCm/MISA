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

class ctrl_dotx_main_loop_t(object):
    def __init__(self):
        self.dotx_m                      = None
        self.unroll_k                    = 0
        self.label_prefix                = ''                      # usually be kernel name of caller kernel
        self.data_type                   = AMDGPU_PRECISION_FP32
        
        self.lds_single_size             = 0                    # in byte, should be power of 2
        self.lds_buffer_num              = 2
        self.local_prefetch_num          = 1
        self.local_prefetch_num_m        = 2

        # functor
        self.global_load_a_functor       = None
        self.global_load_b_functor       = None
        self.shared_store_a_functor      = None
        self.shared_store_b_functor      = None
        self.shared_load_a_functor       = None
        self.shared_load_b_functor       = None
        self.move_slice_window_a_functor = None
        self.move_slice_window_b_functor = None

        # sympol type
        self.v_a                         = None
        self.v_b                         = None
        self.v_c                         = None
        self.v_gld_a                     = None
        self.v_gld_b                     = None
        self.v_sld_a_os                  = None
        self.v_sld_b_os                  = None
        self.v_sst_a_os                  = None
        self.v_sst_b_os                  = None
        self.s_kitr                      = None
        self.s_knum                      = None

        # arch and fma type
        self.arch_name                   = AMDGPU_ARCH_GFX1030
        self.precision                   = 'fp16'

        # below is in unit of pixel, not considered data bytes
        self.lds_k_pack                  = 1
        self.lds_pad_m                   = 0        # pad how many pixels per m row
        self.lds_pad_n                   = 0        # pad how many pixels per n row

class ds_waitcnt_t(object):
    '''
    TODO: compute lds wait count num
    '''
    def __init__(self) -> None:
        super().__init__()
        self.max_num = 0
        self.vpgr_num_dict = dict()
        self.waited_vgprs = set()

    def push_new_vgpr(self, vgpr):
        self.vpgr_num_dict[vgpr] = self.max_num
        self.max_num = self.max_num + 1
        
        self.waited_vgprs.discard(vgpr)

    def get_max_num(self):
        return max(self.vpgr_num_dict.values())

    def compute_waitcnt(self, vgpr_list):
        assert isinstance(vgpr_list, list)
        do_not_need_swait = True
        for vgpr in vgpr_list:
            do_not_need_swait = do_not_need_swait and (vgpr in self.waited_vgprs)
        if do_not_need_swait:
            return -1
        self.waited_vgprs.update(vgpr_list)
        max_index = 0
        for vgpr in vgpr_list:
            max_index = max(max_index, self.vpgr_num_dict[vgpr])
        return self.get_max_num() - max_index

class dotx_core_loop_expr(mc_base_t):
    def __init__(self, mc, name, func=None) -> None:
        mc_base_t.__init__(self, mc)
        self.func = func
        self.name = name
        self.args = ()
        
    def expr_set_args(self, *args):
        self.args = args
        
    def expr_ir(self):
        return self.name
    
    def expr_asm_codes(self):
        if isinstance(self.func, str):
            return self.func
        else:
            return self.func(*self.args)
    
    def emit_expr_asm_codes(self):
        self._emit(self.expr_asm_codes())

class dotx_core_loop_node():
    def __init__(self, name, first=None, second=None) -> None:
        self.name = name
        self.first = first
        self.second = second
        
    def get_first_node(self):
        return self.first
    
    def get_second_node(self):
        return self.second
    
    def set_first_node(self, first):
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
      
class dotx_core_loop_for_loop(dotx_core_loop_node):
    def __init__(self, mc, name, loop_var="", loop_begin=None, loop_step=None, loop_end=None, loop_check=None, 
                 loop_label=None, loop_label_finish=None, loop_label_end=None) -> None:
        super().__init__(name)
        self.loop_var = loop_var
        self.loop_begin = loop_begin
        self.loop_step = loop_step
        self.loop_end = loop_end
        self.loop_check = loop_check
        self.loop_label = loop_label
        self.loop_label_finish = loop_label_finish
        self.loop_label_end = loop_label_end
        self.name = name
        self.mc = mc
        
    def form_loop_jump_finish_check(self):
        #if self.second.name != "loop_jump_check":
        #    assert False, "please provide a loop jump check node"
        
        loop_jump_check_node = dotx_core_loop_node("branch jump finish")
        mc = self.mc
        if self.loop_check == "gt":
            expr = dotx_core_loop_expr(mc, "loop expr", f"s_sub_i32 s[{self.loop_var()}], s[{self.loop_var()}], {self.loop_step}")
            cond = dotx_core_loop_expr(mc, "condition", f"s_cmp_gt_i32 s[{self.loop_var()}], {self.loop_end}")
        else:
            assert False, "not support condition"
            
        jump_branch = dotx_core_loop_expr(mc, "branch", f"s_cbranch_scc0 {self.loop_label_finish}")
        loop_jump_check_node.first = expr
        loop_jump_check_node.second = dotx_core_loop_node("branch jump", cond, jump_branch)
        return loop_jump_check_node
        
    def form_loop_jump_end_check(self):
        mc = self.mc
        if self.loop_check == "gt":
            expr = dotx_core_loop_expr(mc, "loop expr", f"s_sub_i32 s[{self.loop_var()}], s[{self.loop_begin()}], {self.loop_step}")
            cond = dotx_core_loop_expr(mc, "condition", f"s_cmp_gt_i32 s[{self.loop_var()}], {self.loop_end}")
        else:
            assert False, "not support condition"
            
        jump_branch = dotx_core_loop_expr(mc, "branch", f"s_cbranch_scc0 {self.loop_label_end}")
        jump_end_check = dotx_core_loop_node("jump to end")
        jump_end_check.first = expr
        jump_end_check.second = dotx_core_loop_node("branch jump", cond, jump_branch)
        return jump_end_check
    
    def form_loop_jump_to_begin(self):
        jump_branch = dotx_core_loop_expr(self.mc, "branch", f"s_branch {self.loop_label}")
        return jump_branch
    
    def append_new_node(self, new_node, node_stack, next_node_name):
        assert isinstance(node_stack, list) and isinstance(node_stack[-1], dotx_core_loop_node), f"wrong stack type"
        assert isinstance(new_node, dotx_core_loop_node) or isinstance(new_node, dotx_core_loop_expr), f"wrong node to append"
        assert isinstance(next_node_name, str)
        
        tmp_node = node_stack.pop()
        tmp_node.first = new_node
        tmp_node.second = dotx_core_loop_node(next_node_name)
        node_stack.append(tmp_node.second)
        
    def finish_stack(self, node_stack):
        tmp_node = node_stack.pop()
        expr_empty_line = dotx_core_loop_expr(self.mc, "empty line", "")
        tmp_node.first = expr_empty_line
        tmp_node.second = expr_empty_line
        
    def form_loop_body(self, ctrl):
        assert isinstance(ctrl, ctrl_dotx_main_loop_t), "wrong ctrl type"
        f_gld_a = ctrl.global_load_a_functor
        f_gld_b = ctrl.global_load_b_functor
        f_sst_a = ctrl.shared_store_a_functor
        f_sst_b = ctrl.shared_store_b_functor

        f_sld_a = ctrl.shared_load_a_functor
        f_sld_b = ctrl.shared_load_b_functor

        f_move_slice_window_a = ctrl.move_slice_window_a_functor
        f_move_slice_window_b = ctrl.move_slice_window_b_functor

        v_a = ctrl.v_a
        v_b = ctrl.v_b
        v_c = ctrl.v_c

        v_sst_a_os = ctrl.v_sst_a_os
        v_sld_a_os = ctrl.v_sld_a_os
        v_sst_b_os = ctrl.v_sst_b_os
        v_sld_b_os = ctrl.v_sld_b_os

        dotx_m = ctrl.dotx_m

        lds_single_size = ctrl.lds_single_size
        local_prefetch_num = ctrl.local_prefetch_num
        local_prefetch_num_m = ctrl.local_prefetch_num_m
        
        assert local_prefetch_num_m <= dotx_m.lanegroup_repeat_m or local_prefetch_num <= dotx_m.lanegroup_repeat_n, "prefetch too much"
        assert not(local_prefetch_num_m < dotx_m.lanegroup_repeat_m and local_prefetch_num < dotx_m.lanegroup_repeat_n), "can not re-use prefetch for both side"

        # used as offset:x number. may some 
        lds_base_m = 0
        lds_base_n = 0
        unroll_k = ctrl.unroll_k // ctrl.lds_k_pack
        k_per_inst = dotx_m.lanegroup_k_per_thread()

        pad_m = ctrl.lds_pad_m
        pad_n = ctrl.lds_pad_n

        #thread_m = dotx_m.lanegroup_repeat_m
        thread_n = dotx_m.lanegroup_repeat_n * 8
        local_buffer_m = ctrl.lds_k_pack // dotx_m.inst_dotx.k
        local_buffer_n = ctrl.lds_k_pack // dotx_m.inst_dotx.k
        #thread_sub_n = local_buffer_n
        #thread_sub_m = local_buffer_m
        
        v_dotx_k = macro_dotx_mxnxk_t(self.mc, 1, 1, ctrl.lds_k_pack, 1, ctrl.precision)
        
        gld_a = dotx_core_loop_expr(self.mc, "gld_a", f_gld_a)
        gld_b = dotx_core_loop_expr(self.mc, "gld_b", f_gld_b)
        gld_a_b = dotx_core_loop_node("global load a/b", gld_a, gld_b)
        sld_a = dotx_core_loop_expr(self.mc, "sld_a", f_sld_a)
        sld_a.expr_set_args(v_a(), v_sld_a_os(), lds_base_m)
        sld_b = dotx_core_loop_expr(self.mc, "sld_b", f_sld_b)
        sld_b.expr_set_args(v_b(), v_sld_b_os(), lds_base_n)
        sst_a = dotx_core_loop_node("sst a node", 
                                    dotx_core_loop_expr(self.mc, "wait a global load", f"s_waitcnt vmcnt({f_gld_b.get_issues()})"), 
                                    dotx_core_loop_expr(self.mc, "sst_a", f_sst_a))
        sst_b = dotx_core_loop_node("sst b node", 
                                    dotx_core_loop_expr(self.mc, "wait b global load", f"s_waitcnt vmcnt(0)"), 
                                    dotx_core_loop_expr(self.mc, "sst_b", f_sst_b))
        
        sst_a_b = dotx_core_loop_node("sst a/b before core loop", sst_a, sst_b)
        
        msw_a_b = dotx_core_loop_node("msw a/b node", 
                                      dotx_core_loop_expr(self.mc, "msw a", f_move_slice_window_a), 
                                      dotx_core_loop_expr(self.mc, "msw b", f_move_slice_window_b))
        
        wait_all_lgkm = dotx_core_loop_expr(self.mc, "wait all lds", f"s_waitcnt lgkmcnt(0)")
        barrier = dotx_core_loop_expr(self.mc, "barrier", f"s_barrier")
        wait_sst_node = dotx_core_loop_node("wait sst node", wait_all_lgkm, barrier)
        
        
        # sst a/b double buffer switch
        sst_buffer_switch_b = dotx_core_loop_expr(self.mc, "sst a buffer switch", f"v_xor_b32 v[{v_sst_b_os()}], {hex(lds_single_size)}, v[{v_sst_b_os()}] ; switch double buffer b store")
        sst_buffer_switch_a = dotx_core_loop_expr(self.mc, "sst a buffer switch", f"v_xor_b32 v[{v_sst_a_os()}], {hex(lds_single_size)}, v[{v_sst_a_os()}] ; switch double buffer a store")
        sst_buffer_switch_node = dotx_core_loop_node("sst buffer switch node", sst_buffer_switch_b, sst_buffer_switch_a)
        
        
        dotx = dotx_core_loop_expr(self.mc, "dotx", v_dotx_k)
        dotx.expr_set_args(v_c(), v_a(), v_b())
        
        loop_body = dotx_core_loop_node("loop_body")
            
        stack = [loop_body]
        # form repeat k 
        fma_main_loop_node = self.form_loop_fma_body(ctrl, dotx_m.lanegroup_repeat_m, dotx_m.lanegroup_repeat_n - 1)
        self.append_new_node(fma_main_loop_node, stack, "after fma")
        
        if ctrl.lds_buffer_num == 2:
            sld_buffer_switch_b = dotx_core_loop_expr(self.mc, "sst a buffer switch", f"v_xor_b32 v[{v_sld_b_os()}], {hex(lds_single_size)}, v[{v_sld_b_os()}] ; switch double buffer b load")
            sld_buffer_switch_a = dotx_core_loop_expr(self.mc, "sst a buffer switch", f"v_xor_b32 v[{v_sld_a_os()}], {hex(lds_single_size)}, v[{v_sld_a_os()}] ; switch double buffer a load")
            sld_buffer_switch_node = dotx_core_loop_node("sst buffer switch node", sld_buffer_switch_b, sld_buffer_switch_a)
            self.append_new_node(sld_buffer_switch_node, stack, "after sld buffer switch")
        else:
            self.append_new_node(wait_sst_node, stack, "after wait lds op")
            
        # sst node
        self.append_new_node(sst_a_b, stack, "after sst")
        
        # jump to finish branch
        self.append_new_node(self.form_loop_jump_finish_check(), stack, "global load and last repeat n")
        
        # move slice window part
        self.append_new_node(msw_a_b, stack, "after msw")
        
        # last k last n dotx
        wait_lgkmcnt = dotx_core_loop_expr(self.mc, f"wait for all sld", f's_waitcnt lgkmcnt({f_sst_a.get_issues() + f_sst_b.get_issues()})')
        self.append_new_node(wait_lgkmcnt, stack, "last n dotx")
        
        for i_rm in range(dotx_m.lanegroup_repeat_m - 1):
            # compute index for three matrice
            i_rn = dotx_m.lanegroup_repeat_n - 1
            c_index = i_rm * thread_n + i_rn * 8
            a_index = (i_rm % local_prefetch_num_m) * local_buffer_m
            b_index = (((unroll_k - 1) * dotx_m.lanegroup_repeat_n + i_rn) % local_prefetch_num) * local_buffer_n 
            
            dotx = dotx_core_loop_expr(self.mc, "dotx", v_dotx_k)
            dotx.expr_set_args(v_c(c_index), v_a(a_index), v_b(b_index))   
            self.append_new_node(dotx, stack, "last dotx")
        
        # sst double buffer 
        if ctrl.lds_buffer_num == 2:
            self.append_new_node(sst_buffer_switch_node, stack, "after sst switch node")
        
        # wait for sst done
        self.append_new_node(wait_sst_node, stack, "after waiting for sst")
        
        # global load node
        self.append_new_node(gld_a_b, stack, "after global load")
        
        # last n repeat
        i_rn = dotx_m.lanegroup_repeat_n - 1
        i_rm = dotx_m.lanegroup_repeat_m - 1
        c_index = i_rm * thread_n + i_rn * 8
        a_index = (i_rm % local_prefetch_num_m) * local_buffer_m
        b_index = (((unroll_k - 1) * dotx_m.lanegroup_repeat_n + i_rn) % local_prefetch_num) * local_buffer_n 
        dotx = dotx_core_loop_expr(self.mc, "dotx", v_dotx_k)
        dotx.expr_set_args(v_c(c_index), v_a(a_index), v_b(b_index))
        self.append_new_node(dotx, stack, "last dotx")
        
        # jump to begin
        self.append_new_node(self.form_loop_jump_to_begin(), stack, "finishing branch")
        
        # finishing branch 
        loop_finishing_label = dotx_core_loop_expr(self.mc, "loop finish label", self.loop_label_finish+':')
        self.append_new_node(loop_finishing_label, stack, "finishing branch")
        
        wait_lgkmcnt = dotx_core_loop_expr(self.mc, f"wait for all sld", f's_waitcnt lgkmcnt({f_sst_a.get_issues() + f_sst_b.get_issues()})')
        self.append_new_node(wait_lgkmcnt, stack, "last n dotx in finish branch")
        for i_rm in range(dotx_m.lanegroup_repeat_m):
            # compute index for three matrice
            i_rn = dotx_m.lanegroup_repeat_n - 1
            c_index = i_rm * thread_n + i_rn * 8
            a_index = (i_rm % local_prefetch_num_m) * local_buffer_m
            b_index = (((unroll_k - 1) * dotx_m.lanegroup_repeat_n + i_rn) % local_prefetch_num) * local_buffer_n 
            
            dotx = dotx_core_loop_expr(self.mc, "dotx", v_dotx_k)
            dotx.expr_set_args(v_c(c_index), v_a(a_index), v_b(b_index))
            self.append_new_node(dotx, stack, "last dotx")
            
        # loop end branch
        loop_end_label = dotx_core_loop_expr(self.mc, "loop end label", self.loop_label_end+':')
        self.append_new_node(loop_end_label, stack, "node end fma body")
        
        wait_all_lgkm = dotx_core_loop_expr(self.mc, "wait all lds", f"s_waitcnt lgkmcnt(0)")
        barrier = dotx_core_loop_expr(self.mc, "barrier", f"s_barrier")
        wait_sst_node = dotx_core_loop_node("wait sst node", wait_all_lgkm, barrier)
        
        self.append_new_node(wait_sst_node, stack, "end loop dotx")
        loop_end_node = self.form_loop_fma_body(ctrl, dotx_m.lanegroup_repeat_m, dotx_m.lanegroup_repeat_n)
        self.append_new_node(loop_end_node, stack, "finish")
        self.finish_stack(stack)
        
        return loop_body
    
    def form_loop_fma_body(self, ctrl, repeat_m, repeat_n):
        assert isinstance(ctrl, ctrl_dotx_main_loop_t), "wrong ctrl type"
        
        f_sld_a = ctrl.shared_load_a_functor
        f_sld_b = ctrl.shared_load_b_functor

        v_a = ctrl.v_a
        v_b = ctrl.v_b
        v_c = ctrl.v_c
        
        v_sld_a_os = ctrl.v_sld_a_os
        v_sld_b_os = ctrl.v_sld_b_os
        
        dotx_m = ctrl.dotx_m
        
        data_byte = amdgpu_precision_data_byte(amdgpu_string_to_precision(ctrl.precision))

        lds_width_m_per_read = data_byte * (dotx_m.macro_tile_m // dotx_m.lanegroup_repeat_m) * ctrl.lds_k_pack
        lds_width_n_per_read = data_byte * (dotx_m.macro_tile_n // dotx_m.lanegroup_repeat_n) * ctrl.lds_k_pack
        lds_width_m = data_byte * dotx_m.macro_tile_m * ctrl.lds_k_pack
        lds_width_n = data_byte * dotx_m.macro_tile_n * ctrl.lds_k_pack
        local_prefetch_num = ctrl.local_prefetch_num
        local_prefetch_num_m = ctrl.local_prefetch_num_m
        
        assert local_prefetch_num_m <= dotx_m.lanegroup_repeat_m or local_prefetch_num <= dotx_m.lanegroup_repeat_n, "prefetch too much"
        assert not(local_prefetch_num_m < dotx_m.lanegroup_repeat_m and local_prefetch_num < dotx_m.lanegroup_repeat_n), "can not re-use prefetch for both side"

        # used as offset:x number. may some 
        lds_base_m = 0
        lds_base_n = 0
        unroll_k = ctrl.unroll_k // ctrl.lds_k_pack

        thread_m = dotx_m.lanegroup_repeat_m
        thread_n = dotx_m.lanegroup_repeat_n * 8
        local_buffer_m = ctrl.lds_k_pack // dotx_m.inst_dotx.k
        local_buffer_n = ctrl.lds_k_pack // dotx_m.inst_dotx.k
        
        v_dotx_k = macro_dotx_mxnxk_t(self.mc, 1, 1, ctrl.lds_k_pack, 1, ctrl.precision)
        
        sld_a = dotx_core_loop_expr(self.mc, "sld_a", f_sld_a)
        sld_a.expr_set_args(v_a(), v_sld_a_os(), lds_base_m)
        sld_b = dotx_core_loop_expr(self.mc, "sld_b", f_sld_b)
        sld_b.expr_set_args(v_b(), v_sld_b_os(), lds_base_n)
        
        loop_fma_body = dotx_core_loop_node("loop fma body")
        stack = [loop_fma_body]
        #loop_end_label = dotx_core_loop_expr(self.mc, "loop end label", self.loop_label_end+':')
        
        #self.append_new_node(loop_end_label, stack, "node after body")
        
        ds_waitcnt = ds_waitcnt_t()
        
        prefetch_a = []
        prefetch_b = []
        local_prefetch = []
        for i_prefetch in range(local_prefetch_num_m):
            sld_a = dotx_core_loop_expr(self.mc, "sld_a", f_sld_a)
            sld_a.expr_set_args(v_a(i_prefetch * local_buffer_m), v_sld_a_os(), lds_base_m + i_prefetch * lds_width_m_per_read)
            prefetch_a.append(sld_a)
            
        for i_prefetch in range(local_prefetch_num):
            sld_b = dotx_core_loop_expr(self.mc, "sld_b", f_sld_b)
            sld_b.expr_set_args(v_b(i_prefetch * local_buffer_n), v_sld_b_os(), lds_base_n + i_prefetch * lds_width_n_per_read)
            prefetch_b.append(sld_b)
            
        local_prefetch[0:0] = prefetch_b
        local_prefetch[1:1] = prefetch_a
        
        node_local_prefecth = dotx_core_loop_node("local refetch node")
        stack_prefetch = [node_local_prefecth]
        for expr_local_prefetch in local_prefetch:
            self.append_new_node(expr_local_prefetch, stack_prefetch, f"local prefetch")
            ds_waitcnt.push_new_vgpr(expr_local_prefetch.args[0])
            
        self.finish_stack(stack_prefetch)
            
        self.append_new_node(node_local_prefecth, stack, "loop end main loop")
        # form repeat k 
        for i_k in range(unroll_k - 1):
            for i_rn in range(dotx_m.lanegroup_repeat_n):
                if i_rn > 0:
                    sld_b = dotx_core_loop_expr(self.mc, "sld_b", f_sld_b)
                    sld_b.expr_set_args(v_b(((i_k * dotx_m.lanegroup_repeat_n + i_rn - 1) % local_prefetch_num)* local_buffer_n), v_sld_b_os(), f'{lds_base_n}+{i_k}*{lds_width_n}+{(i_rn+1)*lds_width_n_per_read}')
                    ds_waitcnt.push_new_vgpr(v_b(((i_k * dotx_m.lanegroup_repeat_n + i_rn - 1) % local_prefetch_num)* local_buffer_n))
                    self.append_new_node(sld_b, stack, "after prefetch b")
                for i_rm in range(dotx_m.lanegroup_repeat_m):
                    # compute index for three matrice
                    c_index = i_rm * thread_n + i_rn * 8
                    a_index = (i_rm % local_prefetch_num_m) * local_buffer_m
                    b_index = ((i_k * dotx_m.lanegroup_repeat_n + i_rn) % local_prefetch_num) * local_buffer_n 
                    lgkmcnt = ds_waitcnt.compute_waitcnt([v_a(a_index), v_b(b_index)])
                    if lgkmcnt != -1:
                        wait_lgkmcnt = dotx_core_loop_expr(self.mc, f"wait for dotx {i_k, i_rm, i_rn}", f's_waitcnt lgkmcnt({lgkmcnt})')
                        self.append_new_node(wait_lgkmcnt, stack, "after wait cnt")
                    
                    if i_rn == dotx_m.lanegroup_repeat_n - 1 and i_rm > 0:
                        sld_a = dotx_core_loop_expr(self.mc, "sld a", f_sld_a)
                        sld_a.expr_set_args(v_a((i_rm - 1) * local_buffer_m), v_sld_a_os(), f'{lds_base_m}+{i_k+1}*{lds_width_m}+{(i_rm-1)*lds_width_m_per_read}')
                        ds_waitcnt.push_new_vgpr(v_a((i_rm - 1) * local_buffer_m))
                        self.append_new_node(sld_a, stack, "after prefetch a")
                        
                    dotx = dotx_core_loop_expr(self.mc, "dotx", v_dotx_k)
                    dotx.expr_set_args(v_c(c_index), v_a(a_index), v_b(b_index))
                    self.append_new_node(dotx, stack, f"dotx node next {i_k, i_rm, i_rn}")
                    
            sld_a = dotx_core_loop_expr(self.mc, "sld a", f_sld_a)
            sld_a.expr_set_args(v_a((local_prefetch_num_m - 1) * local_buffer_m), v_sld_a_os(), f'{lds_base_m}+{i_k + 1}*{lds_width_m}+{(local_prefetch_num_m - 1)*lds_width_m_per_read}')
            ds_waitcnt.push_new_vgpr(v_a((local_prefetch_num_m - 1) * local_buffer_m))
            sld_b = dotx_core_loop_expr(self.mc, "sld_b", f_sld_b)
            sld_b.expr_set_args(v_b((i_k * dotx_m.lanegroup_repeat_n + i_rn) % local_prefetch_num * local_buffer_n), v_sld_b_os(), f'{lds_base_n}+{i_k + 1}*{lds_width_n}+{(local_prefetch_num - 1)*lds_width_n_per_read}')
            ds_waitcnt.push_new_vgpr(v_b((i_k * dotx_m.lanegroup_repeat_n + i_rn) % local_prefetch_num * local_buffer_n))
            sld_a_b = dotx_core_loop_node("last sld a/b", sld_a, sld_b)
            self.append_new_node(sld_a_b, stack, f"next dotx node {i_k}")
            
        for i_rn in range(repeat_n):
            if i_rn > 0 and dotx_m.lanegroup_repeat_n > local_prefetch_num:
                sld_b = dotx_core_loop_expr(self.mc, "sld_b", f_sld_b)
                sld_b.expr_set_args(v_b((((unroll_k - 1) * dotx_m.lanegroup_repeat_n + i_rn - 1) % local_prefetch_num * local_buffer_n)), v_sld_b_os(), f'{lds_base_n}+{unroll_k - 1}*{lds_width_n}+{(i_rn+1)*lds_width_n_per_read}')
                ds_waitcnt.push_new_vgpr(v_b((((unroll_k - 1) * dotx_m.lanegroup_repeat_n + i_rn - 1) % local_prefetch_num * local_buffer_n)))
                
                self.append_new_node(sld_b, stack, "after prefetch b")
            
            for i_rm in range(repeat_m):
                # compute index for three matrice
                c_index = i_rm * thread_n + i_rn * 8
                a_index = (i_rm % local_prefetch_num_m) * local_buffer_m
                b_index = (((unroll_k - 1) * dotx_m.lanegroup_repeat_n + i_rn) % local_prefetch_num) * local_buffer_n 
                lgkmcnt = ds_waitcnt.compute_waitcnt([v_a(a_index), v_b(b_index)])
                if lgkmcnt != -1:
                    wait_lgkmcnt = dotx_core_loop_expr(self.mc, f"wait for dotx {i_k, i_rm, i_rn}", f's_waitcnt lgkmcnt({lgkmcnt})')
                    self.append_new_node(wait_lgkmcnt, stack, "after wait cnt")

                dotx = dotx_core_loop_expr(self.mc, "dotx", v_dotx_k)
                dotx.expr_set_args(v_c(c_index), v_a(a_index), v_b(b_index))
                self.append_new_node(dotx, stack, f"dotx node next {i_k, i_rm, i_rn}")
                
        self.finish_stack(stack)
        
        return loop_fma_body
        
class dotx_core_loop_graph():
    def __init__(self, ctrl, mc=None):
        self.ctrl = ctrl
        self.base_node = None
        self.mc = mc
        
    def add_node_comment(self, node, str_comment):
        comment_expr = dotx_core_loop_expr(self.mc, "comments", str_comment)
        new_node = dotx_core_loop_node("with_comments: "+node.name)
        new_node.first = comment_expr
        new_node.second = node
        return new_node
        
    def creat_base_graph(self):
        
        label_fma_body = 'L_{}_fma_body'.format(self.ctrl.label_prefix)
        label_fma_finishing = 'L_{}_fma_finishing'.format(self.ctrl.label_prefix)
        label_fma_end = 'L_{}_end'.format(self.ctrl.label_prefix)
        
        v_c = self.ctrl.v_c

        s_kitr = self.ctrl.s_kitr
        s_knum = self.ctrl.s_knum
        dotx_m = self.ctrl.dotx_m
        
        # used as offset:x number. may some 
        unroll_k = self.ctrl.unroll_k // self.ctrl.lds_k_pack

        thread_m = dotx_m.lanegroup_repeat_m
        thread_n = dotx_m.lanegroup_repeat_n * 8
        
        f_gld_a = self.ctrl.global_load_a_functor
        f_gld_b = self.ctrl.global_load_b_functor
        
        f_gld_b = self.ctrl.global_load_b_functor
        f_sst_a = self.ctrl.shared_store_a_functor
        f_sst_b = self.ctrl.shared_store_b_functor
        
        f_move_slice_window_a = self.ctrl.move_slice_window_a_functor
        f_move_slice_window_b = self.ctrl.move_slice_window_b_functor
        
        v_sst_a_os = self.ctrl.v_sst_a_os
        v_sst_b_os = self.ctrl.v_sst_b_os
        
        lds_single_size = self.ctrl.lds_single_size
        
        gld_a = dotx_core_loop_expr(self.mc, "gld_a", f_gld_a)
        gld_b = dotx_core_loop_expr(self.mc, "gld_b", f_gld_b)
        
        sst_a = dotx_core_loop_node("sst a node", 
                                    dotx_core_loop_expr(self.mc, "wait a global load", f"s_waitcnt vmcnt({f_gld_b.get_issues()})"), 
                                    dotx_core_loop_expr(self.mc, "sst_a", f_sst_a))
        sst_b = dotx_core_loop_node("sst b node", 
                                    dotx_core_loop_expr(self.mc, "wait b global load", f"s_waitcnt vmcnt(0)"), 
                                    dotx_core_loop_expr(self.mc, "sst_b", f_sst_b))
        
        msw_a_b = dotx_core_loop_node("msw a/b node", 
                                      dotx_core_loop_expr(self.mc, "msw a", f_move_slice_window_a), 
                                      dotx_core_loop_expr(self.mc, "msw b", f_move_slice_window_b))
        
        base_node = dotx_core_loop_node("core_loop")
        node_clear_c = dotx_core_loop_expr(self.mc, ".clear_c", f".v_clear_nc {v_c()}, {thread_m * thread_n}")
        
        base_for_loop = dotx_core_loop_for_loop(self.mc, "core_loop", s_kitr, s_knum, unroll_k, 0, "gt", label_fma_body, label_fma_finishing, label_fma_end)
        
        loop_begin_check = dotx_core_loop_expr(self.mc, "loop_begin_check")
        loop_body = dotx_core_loop_node("loop_body")
        
        
        loop_body = base_for_loop.form_loop_body(self.ctrl)
        
        base_for_loop.first = dotx_core_loop_node("loop body with label", dotx_core_loop_expr(self.mc, "loop label", label_fma_body+':'), loop_body)
        base_for_loop.second = dotx_core_loop_expr(self.mc, "reserved line", "")
        
        # base_for_loop.form_loop_jump_check()
        
        first_sst = dotx_core_loop_node("sst a/b before core loop", sst_a, sst_b)
        node_before_for_loop = dotx_core_loop_node("sst a/b before core loop0", first_sst, node_clear_c)
        check_loop_end_node = base_for_loop.form_loop_jump_end_check()
        end_check_before_msw = dotx_core_loop_node("end_check_before_msw", node_before_for_loop, check_loop_end_node)
        base_node.first = dotx_core_loop_node("sst a/b before core loop1", end_check_before_msw, msw_a_b)
        
        # sst a/b double buffer switch
        sst_buffer_switch_b = dotx_core_loop_expr(self.mc, "sst a buffer switch", f"v_xor_b32 v[{v_sst_b_os()}], {hex(lds_single_size)}, v[{v_sst_b_os()}] ; switch double buffer b store")
        sst_buffer_switch_a = dotx_core_loop_expr(self.mc, "sst a buffer switch", f"v_xor_b32 v[{v_sst_a_os()}], {hex(lds_single_size)}, v[{v_sst_a_os()}] ; switch double buffer a store")
        sst_buffer_switch_node = dotx_core_loop_node("sst buffer switch node", sst_buffer_switch_b, sst_buffer_switch_a)
        
        # first barrier and waitcnt
        wait_all_lgkm = dotx_core_loop_expr(self.mc, "wait all lds", f"s_waitcnt lgkmcnt(0)")
        barrier = dotx_core_loop_expr(self.mc, "barrier", f"s_barrier")
        wait_sst_node = dotx_core_loop_node("wait sst node", wait_all_lgkm, barrier)
        
        # 
        if self.ctrl.lds_buffer_num == 2:
            wait_node = dotx_core_loop_node("wait node", sst_buffer_switch_node, wait_sst_node)
        else:
            wait_node = wait_sst_node
            
        # global load before loop
        global_load_a_b = dotx_core_loop_node("global load a/b", gld_a, gld_b)
        
        # node with init loop var
        base_node.second = dotx_core_loop_node("node 0")
        base_node.second.first = wait_node
        base_node.second.second = dotx_core_loop_node("node 1")
        base_node.second.second.first = global_load_a_b
        base_node.second.second.second = base_for_loop
        
        # last unroll k
        # dotx_core_loop_node("loop body with label", dotx_core_loop_expr(self.mc, "loop end label", label_fma_end+':'), loop_body)
        
        base_node = self.add_node_comment(base_node, f"; start FMA loop, {thread_m}x{thread_n}")
        
        self.base_node = base_node
         
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
    
    core_loop_graph = dotx_core_loop_graph(mc, None)
    core_loop_graph.creat_base_graph()
    print_ir(core_loop_graph.base_node)
    
        