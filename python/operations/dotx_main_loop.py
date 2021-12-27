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
        

class dotx_main_loop_t_0(mc_base_t):
    '''
    TODO: 
    '''
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        self.ctrl = ctrl
    def emit(self):
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

        # assert type(v_a) is sym_t and type(s_kitr) is sym_t  # other gpr type check ignore

        data_byte = amdgpu_precision_data_byte(amdgpu_string_to_precision(self.ctrl.precision))

        lds_width_m_per_read = data_byte * (dotx_m.macro_tile_m // dotx_m.lanegroup_repeat_m) * self.ctrl.lds_k_pack
        lds_width_n_per_read = data_byte * (dotx_m.macro_tile_n // dotx_m.lanegroup_repeat_n) * self.ctrl.lds_k_pack
        lds_width_m = data_byte * dotx_m.macro_tile_m * self.ctrl.lds_k_pack
        lds_width_n = data_byte * dotx_m.macro_tile_n * self.ctrl.lds_k_pack
        lds_single_size = self.ctrl.lds_single_size
        local_prefetch_num = self.ctrl.local_prefetch_num

        # used as offset:x number. may some 
        lds_base_m = 0
        lds_base_n = 0
        unroll_k = self.ctrl.unroll_k // self.ctrl.lds_k_pack
        k_per_inst = dotx_m.lanegroup_k_per_thread()

        pad_m = self.ctrl.lds_pad_m
        pad_n = self.ctrl.lds_pad_n

        thread_m = dotx_m.lanegroup_repeat_m
        thread_n = dotx_m.lanegroup_repeat_n * 8
        local_buffer_m = self.ctrl.lds_k_pack // dotx_m.inst_dotx.k
        local_buffer_n = self.ctrl.lds_k_pack // dotx_m.inst_dotx.k
        thread_sub_n = local_buffer_n
        thread_sub_m = local_buffer_m

        ds_waitcnt = ds_waitcnt_t()

        def mapped_ioffset(i_k, width_byte, pad_pixel, offset = 0):
            k_pack = self.ctrl.lds_k_pack
            i_k0 = i_k // k_pack
            i_kp = i_k % k_pack
            return i_k0 * (width_byte * k_pack + pad_pixel * data_byte) + i_kp * data_byte + offset * k_pack

        # mi = mapped_ioffset
        def mi_m(i_k, offset = 0):
            if pad_m > 0:
                return mapped_ioffset(i_k, lds_width_m, 0, offset) // 32 * (32 + pad_m)
            else:
                return mapped_ioffset(i_k, lds_width_m, 0, offset)
        
        def mi_n(i_k, offset = 0):
            if pad_n > 0:
                return mapped_ioffset(i_k, lds_width_n, 0, offset) // 32 * (32 + pad_n)
            else:
                return mapped_ioffset(i_k, lds_width_n, 0, offset)

        v_dotx_k = macro_dotx_mxnxk_t(self.mc, 1, 1, self.ctrl.lds_k_pack, 1, self.ctrl.precision)

        # start emit
        self._emit(f"; start FMA loop, {thread_m}x{thread_n}")
        self._emit(f"s_waitcnt vmcnt({f_gld_b.get_issues()})")

        self._emit(f_sst_a())
        self._emit_empty_line()
        self._emit(f"s_waitcnt vmcnt(0)")
        self._emit(f_sst_b())
        self._emit_empty_line()

        self._emit(f".v_clear_nc {v_c()}, {thread_m * thread_n}")

        # decrese k
        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_knum()}], {unroll_k}")
        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
        self._emit(f"s_cbranch_scc0 {label_fma_end}")
        self._emit_empty_line()

        self._emit(f_move_slice_window_b())
        self._emit(f_move_slice_window_a())

        if self.ctrl.lds_buffer_num == 2:
            self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {hex(lds_single_size)}, v[{v_sst_b_os()}] ; switch double buffer b store")
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {hex(lds_single_size)}, v[{v_sst_a_os()}] ; switch double buffer a store")

        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit(f"s_barrier")
        self._emit_empty_line()
        self._emit(f_gld_a())
        self._emit(f_gld_b())
        self._emit_empty_line()

        # Label: start of fma body
        self._emit_front(f"{label_fma_body}:")
        self._emit(f"; do fma accumulate with unroll {unroll_k}")
        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
        ds_waitcnt.push_new_vgpr(v_b())
        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
        ds_waitcnt.push_new_vgpr(v_a())
        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + lds_width_m_per_read))
        ds_waitcnt.push_new_vgpr(v_a(local_buffer_m))
        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + lds_width_n_per_read))
        ds_waitcnt.push_new_vgpr(v_b(local_buffer_n))
        self._emit_empty_line()

        self._emit(f".itr_k = 0")
        self._emit(f".rept {unroll_k-1}")
        with self._indent_context():
            for i_rn in range(dotx_m.lanegroup_repeat_n):
                for i_rm in range(dotx_m.lanegroup_repeat_m):
                    # compute index for three matrice
                    c_index = i_rm * thread_n + i_rn * 8
                    a_index = (i_rm % local_prefetch_num) * local_buffer_m
                    b_index = (i_rn % local_prefetch_num) * local_buffer_n 
                    lgkmcnt = ds_waitcnt.compute_waitcnt([v_a(a_index), v_b(b_index)])
                    if lgkmcnt != -1:
                        self._emit(f's_waitcnt lgkmcnt({lgkmcnt})')
                    if i_rn == dotx_m.lanegroup_repeat_n - 1 and i_rm == dotx_m.lanegroup_repeat_m - 1:
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), f'{lds_base_m} + (.itr_k+1)*{lds_width_m}'))
                        ds_waitcnt.push_new_vgpr(v_a())
                    self._emit(v_dotx_k(v_c(c_index), v_a(a_index), v_b(b_index)))
                if i_rn + local_prefetch_num < dotx_m.lanegroup_repeat_n:
                    self._emit(f_sld_b(v_b((i_rn % local_prefetch_num) * local_buffer_n), v_sld_b_os(), f'{lds_base_n}+.itr_k*{lds_width_n}+{(i_rn + local_prefetch_num) * lds_width_n_per_read}'))
                    ds_waitcnt.push_new_vgpr(v_b((i_rn % local_prefetch_num) * local_buffer_n))
                
                if dotx_m.lanegroup_repeat_n - local_prefetch_num > 0:
                    if i_rn == max(dotx_m.lanegroup_repeat_n - local_prefetch_num, local_prefetch_num):
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}'))
                        ds_waitcnt.push_new_vgpr(v_b())
                else:
                    if i_rn == max(dotx_m.lanegroup_repeat_n - local_prefetch_num, 0):
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}'))
                        ds_waitcnt.push_new_vgpr(v_b())

                if i_rn == dotx_m.lanegroup_repeat_n - 1:
                    self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m}+{lds_width_m_per_read}'))
                    ds_waitcnt.push_new_vgpr(v_a(local_buffer_m))
                    self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}+{lds_width_n_per_read}'))
                    ds_waitcnt.push_new_vgpr(v_b(local_buffer_n))

            self._emit('.itr_k = .itr_k + 1')

        self._emit(f".endr")
        self._emit_empty_line()
        self._emit(f"; last unroll")

        for i_rn in range(dotx_m.lanegroup_repeat_n - 1):
            for i_rm in range(dotx_m.lanegroup_repeat_m):
                # compute index for three matrice
                c_index = i_rm * thread_n + i_rn * 8
                a_index = (i_rm % local_prefetch_num) * local_buffer_m
                b_index = (i_rn % local_prefetch_num) * local_buffer_n 
                lgkmcnt = ds_waitcnt.compute_waitcnt([v_a(a_index), v_b(b_index)])
                if lgkmcnt != -1:
                    self._emit(f's_waitcnt lgkmcnt({lgkmcnt})')
                self._emit(v_dotx_k(v_c(c_index), v_a(a_index), v_b(b_index)))
            if i_rn + local_prefetch_num < dotx_m.lanegroup_repeat_n:
                self._emit(f_sld_b(v_b((i_rn % local_prefetch_num) * local_buffer_n), v_sld_b_os(), f'{lds_base_n}+.itr_k*{lds_width_n}+{(i_rn + local_prefetch_num) * lds_width_n_per_read}'))
                ds_waitcnt.push_new_vgpr(v_b((i_rn % local_prefetch_num) * local_buffer_n))

        if self.ctrl.lds_buffer_num == 2:
            self._emit(f"; switch lds load")
            self._emit(f"v_xor_b32 v[{v_sld_b_os()}], {lds_single_size}, v[{v_sld_b_os()}] ; switch double buffer b load")
            self._emit(f"v_xor_b32 v[{v_sld_a_os()}], {lds_single_size}, v[{v_sld_a_os()}] ; switch double buffer a load")

        if self.ctrl.lds_buffer_num == 1:
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
        #       wait global and store to LDS
        self._emit(f"s_waitcnt vmcnt({f_gld_b.get_issues()})")
        self._emit(f_sst_a())
        self._emit(f"s_waitcnt vmcnt(0)")
        self._emit(f_sst_b())

        #       iteration--
        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
        self._emit(f"s_cbranch_scc0 {label_fma_finishing}")

        self._emit(f_move_slice_window_b())
        self._emit(f_move_slice_window_a())

        # last repeat n dotx loop
        self._emit(f"s_waitcnt lgkmcnt({f_sst_a.get_issues() + f_sst_b.get_issues()})")
        for i_rm in range(dotx_m.lanegroup_repeat_m - 1):
            # compute index for three matrice
            i_rn = dotx_m.lanegroup_repeat_n - 1
            c_index = i_rm * thread_n + i_rn * 8
            a_index = (i_rm % local_prefetch_num) * local_buffer_m
            b_index = (i_rn % local_prefetch_num) * local_buffer_n 
            self._emit(v_dotx_k(v_c(c_index), v_a(a_index), v_b(b_index)))
        
        self._emit_empty_line()
        
        if self.ctrl.lds_buffer_num == 2:
            self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {lds_single_size}, v[{v_sst_b_os()}] ; switch double buffer b store")
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {lds_single_size}, v[{v_sst_a_os()}] ; switch double buffer a store")
        #       barrier here!
        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit(f"s_barrier")

        #       load next from global
        self._emit(f_gld_a())
        self._emit(f_gld_b())

        # last repeat m dotx loop
        i_rn = dotx_m.lanegroup_repeat_n - 1
        i_rm = dotx_m.lanegroup_repeat_m - 1
        c_index = i_rm * thread_n + i_rn * 8
        a_index = (i_rm % local_prefetch_num) * local_buffer_m
        b_index = (i_rn % local_prefetch_num) * local_buffer_n 
        self._emit(v_dotx_k(v_c(c_index), v_a(a_index), v_b(b_index)))
        self._emit_empty_line()
        self._emit(f"s_branch {label_fma_body}")

        assert dotx_m.lanegroup_repeat_m <= local_prefetch_num, f"do not support the cases whose repeat m num is greater than local prefetch num"

        # Label: finishing of fma body
        self._emit_front(f"{label_fma_finishing}:")
        self._emit(f"s_waitcnt lgkmcnt({f_sst_a.get_issues() + f_sst_a.get_issues()})")
        for i_rm in range(dotx_m.lanegroup_repeat_m):
            # compute index for three matrice
            i_rn = dotx_m.lanegroup_repeat_n - 1
            c_index = i_rm * thread_n + i_rn * 8
            a_index = (i_rm % local_prefetch_num) * local_buffer_m
            b_index = (i_rn % local_prefetch_num) * local_buffer_n 
            self._emit(v_dotx_k(v_c(c_index), v_a(a_index), v_b(b_index)))

        # Label: end of fma body
        self._emit_front(f"{label_fma_end}:")
        self._emit("s_waitcnt lgkmcnt(0)")
        self._emit("s_barrier")

        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
        ds_waitcnt.push_new_vgpr(v_b())
        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
        ds_waitcnt.push_new_vgpr(v_a())
        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + lds_width_m_per_read))
        ds_waitcnt.push_new_vgpr(v_a(local_buffer_m))
        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + lds_width_n_per_read))
        ds_waitcnt.push_new_vgpr(v_b(local_buffer_n))
        self._emit_empty_line()

        self._emit(f".itr_k = 0")
        self._emit(f".rept {unroll_k-1}")
        with self._indent_context():
            for i_rn in range(dotx_m.lanegroup_repeat_n):
                for i_rm in range(dotx_m.lanegroup_repeat_m):
                    # compute index for three matrice
                    c_index = i_rm * thread_n + i_rn * 8
                    a_index = (i_rm % local_prefetch_num) * local_buffer_m
                    b_index = (i_rn % local_prefetch_num) * local_buffer_n 
                    lgkmcnt = ds_waitcnt.compute_waitcnt([v_a(a_index), v_b(b_index)])
                    if lgkmcnt != -1:
                        self._emit(f's_waitcnt lgkmcnt({lgkmcnt})')
                    if i_rn == dotx_m.lanegroup_repeat_n - 1 and i_rm == dotx_m.lanegroup_repeat_m - 1:
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), f'{lds_base_m} + (.itr_k+1)*{lds_width_m}'))
                        ds_waitcnt.push_new_vgpr(v_a())
                    self._emit(v_dotx_k(v_c(c_index), v_a(a_index), v_b(b_index)))
                if i_rn + local_prefetch_num < dotx_m.lanegroup_repeat_n:
                    self._emit(f_sld_b(v_b((i_rn % local_prefetch_num) * local_buffer_n), v_sld_b_os(), f'{lds_base_n}+.itr_k*{lds_width_n}+{(i_rn + local_prefetch_num) * lds_width_n_per_read}'))
                    ds_waitcnt.push_new_vgpr(v_b((i_rn % local_prefetch_num) * local_buffer_n))
                    
                if dotx_m.lanegroup_repeat_n - local_prefetch_num > 0:
                    if i_rn == max(dotx_m.lanegroup_repeat_n - local_prefetch_num, local_prefetch_num):
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}'))
                        ds_waitcnt.push_new_vgpr(v_b())
                else:
                    if i_rn == max(dotx_m.lanegroup_repeat_n - local_prefetch_num, 0):
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}'))
                        ds_waitcnt.push_new_vgpr(v_b())

                if i_rn == dotx_m.lanegroup_repeat_n - 1:
                    self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m}+{lds_width_m_per_read}'))
                    ds_waitcnt.push_new_vgpr(v_a(local_buffer_m))
                    self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}+{lds_width_n_per_read}'))
                    ds_waitcnt.push_new_vgpr(v_b(local_buffer_n))

            self._emit('.itr_k = .itr_k + 1')

        self._emit(f".endr")
        self._emit_empty_line()
        self._emit(f"; last unroll")

        for i_rn in range(dotx_m.lanegroup_repeat_n):
            for i_rm in range(dotx_m.lanegroup_repeat_m):
                # compute index for three matrice
                c_index = i_rm * thread_n + i_rn * 8
                a_index = (i_rm % local_prefetch_num) * local_buffer_m
                b_index = (i_rn % local_prefetch_num) * local_buffer_n 
                lgkmcnt = ds_waitcnt.compute_waitcnt([v_a(a_index), v_b(b_index)])
                if lgkmcnt != -1:
                    self._emit(f's_waitcnt lgkmcnt({lgkmcnt})')
                self._emit(v_dotx_k(v_c(c_index), v_a(a_index), v_b(b_index)))
            if i_rn + local_prefetch_num < dotx_m.lanegroup_repeat_n:
                self._emit(f_sld_b(v_b((i_rn % local_prefetch_num) * local_buffer_n), v_sld_b_os(), f'{lds_base_n}+.itr_k*{lds_width_n}+{(i_rn + local_prefetch_num) * lds_width_n_per_read}'))
                ds_waitcnt.push_new_vgpr(v_b((i_rn % local_prefetch_num) * local_buffer_n))