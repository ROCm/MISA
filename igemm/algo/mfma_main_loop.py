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
# pylint: disable=maybe-no-member

from ..codegen import *
from .utility import *
from .mfma import *
from .xdlops_mapping import *

class ctrl_mfma_main_loop_t(object):
    def __init__(self):
        self.wave_tile_m                 = 0
        self.wave_tile_n                 = 0
        self.wave_repeat_m               = 0
        self.wave_repeat_n               = 0
        self.wave_step_m                 = 0
        self.wave_step_n                 = 0

        self.unroll_k                    = 0
        self.label_prefix                = ''                      # usually be kernel name of caller kernel
        self.data_type                   = AMDGPU_PRECISION_FP32

        self.lds_single_size             = 0                    # in byte, should be power of 2
        self.lds_buffer_num              = 2

        # functor
        self.global_load_a_functor       = None
        self.global_load_b_functor       = None
        self.shared_store_a_functor      = None
        self.shared_store_b_functor      = None
        self.shared_load_a_functor       = None
        self.shared_load_b_functor       = None
        self.move_slice_window_a_functor = None
        self.move_slice_window_b_functor = None

        # symbol type
        self.v_a                         = None
        self.v_b                         = None
        self.a_c                         = None
        self.v_gld_a                     = None
        self.v_gld_b                     = None
        self.v_sld_a_os                  = None
        self.v_sld_b_os                  = None
        self.v_sst_a_os                  = None
        self.v_sst_b_os                  = None
        self.s_kitr                      = None
        self.s_knum                      = None

class mfma_main_loop_t(mc_base_t):
    '''
    implement fma main loop with 2x2 sub buffer
    4x4, 4x6, 4x8, 6x4, 6x6, 6x8, 8x4, 8x6, 8x8
    other tile size may also useful, but can't form 2x2 sub buffer

    TODO: currently load b matrix first
    TODO: used for non-xdlops
    '''
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        self.ctrl = ctrl
        assert type(ctrl) is ctrl_mfma_main_loop_t
    def emit(self):
        label_mfma_body = 'L_{}_mfma_body'.format(self.ctrl.label_prefix)
        label_mfma_finishing = 'L_{}_mfma_finishing'.format(self.ctrl.label_prefix)
        label_mfma_end = 'L_{}_mfma_end'.format(self.ctrl.label_prefix)

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
        a_c = self.ctrl.a_c

        v_gld_a = self.ctrl.v_gld_a
        v_gld_b = self.ctrl.v_gld_b

        v_sst_a_os = self.ctrl.v_sst_a_os
        v_sld_a_os = self.ctrl.v_sld_a_os
        v_sst_b_os = self.ctrl.v_sst_b_os
        v_sld_b_os = self.ctrl.v_sld_b_os

        s_kitr = self.ctrl.s_kitr
        s_knum = self.ctrl.s_knum

        wave_tile_m   = self.ctrl.wave_tile_m
        wave_tile_n   = self.ctrl.wave_tile_n
        wave_repeat_m = self.ctrl.wave_repeat_m
        wave_repeat_n = self.ctrl.wave_repeat_n
        wave_step_m   = self.ctrl.wave_step_m
        wave_step_n   = self.ctrl.wave_step_n

        data_byte = amdgpu_precision_data_byte(self.ctrl.data_type)

        lds_width_m = data_byte * wave_tile_m * wave_step_m * wave_repeat_m
        lds_width_n = data_byte * wave_tile_n * wave_step_n * wave_repeat_n
        lds_single_size = self.ctrl.lds_single_size

        # used as offset:x number. may some 
        lds_base_m = 0
        lds_base_n = 0
        unroll_k = self.ctrl.unroll_k

        v_mfma_wave_wise = get_ctrl_xdlops_mapping_from_wave_fp32(wave_tile_m, wave_tile_n, wave_repeat_m, wave_repeat_n, wave_step_m, wave_step_n)

        def mfma_step_mxn(i_repeat_m, i_repeat_n):
            mfma = v_mfma_wave_wise.inst_mfma
            num_agpr_per_issue = mfma.num_a_c
            with self._deferred_context():
                for i_step_n in range(wave_step_n):
                    for i_step_m in range(wave_step_m):
                        #a_index = i_repeat_m * wave_tile_m * wave_step_m + i_repeat_n *wave_tile_n*wave_step_n +  i_step_m
                        c_index = i_repeat_m * wave_step_m * wave_step_n * wave_repeat_n * num_agpr_per_issue + \
                                    i_repeat_n * wave_step_m * wave_step_n * num_agpr_per_issue + \
                                    i_step_m * wave_step_n * num_agpr_per_issue + \
                                    i_step_n * num_agpr_per_issue
                        c_index_end = c_index + num_agpr_per_issue - 1
                        a_index = i_repeat_m * wave_step_m * mfma.num_v_a + \
                                    i_step_m * mfma.num_v_a

                        b_index = i_repeat_n * wave_step_n * mfma.num_v_b + \
                                    i_step_n * mfma.num_v_b
                        self._emit(mfma(a_c((c_index, c_index_end)), v_a(a_index), v_b(b_index), a_c((c_index, c_index_end))) + f"  ; repeat:{i_repeat_m}x{i_repeat_n}, step:{i_step_m}x{i_step_n}, num_a_c:{num_agpr_per_issue}")
            return self._get_deferred()

        def mfma_loop_repeat_2x2():
            mfma = v_mfma_wave_wise.inst_mfma
            repeat_m_thread_offset = wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = wave_step_n * mfma.num_v_b

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())

            self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {hex(lds_single_size)}, v[{v_sst_b_os()}] ; switch double buffer b store")
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {hex(lds_single_size)}, v[{v_sst_a_os()}] ; switch double buffer a store")
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            self._emit_empty_line()
            self._emit(f_gld_b())                                           # global load
            self._emit(f_gld_a())                                           # global load3
            self._emit_empty_line()

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_width_m // 2 ))

            self._emit(f".itr_k = 0")
            self._emit(f".rept {unroll_k-1}")
            with self._indent_context():
                # 1st fma
                self._emit(f's_waitcnt lgkmcnt(2)')
                self._emit(mfma_step_mxn(0, 0))

                # 2nd fma
                self._emit(f's_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(0, 1))

                # 3rd fma
                self._emit(f_sld_a(v_a(), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m}'))
                self._emit(f's_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(1, 0))

                # 4th fma
                self._emit(f_sld_b(v_b(), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}'))
                self._emit(mfma_step_mxn(1, 1))
                self._emit_empty_line()

                # last
                self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}+{lds_width_n//2}'))
                self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m}+{lds_width_m//2}'))
                self._emit('.itr_k = .itr_k + 1')

            self._emit(f".endr")
            self._emit_empty_line()
            self._emit(f"; last unroll")
            self._emit(f"v_xor_b32 v[{v_sld_b_os()}], {lds_single_size}, v[{v_sld_b_os()}] ; switch double buffer b load")
            self._emit(f"v_xor_b32 v[{v_sld_a_os()}], {lds_single_size}, v[{v_sld_a_os()}] ; switch double buffer a load")

            # 1st fma
            self._emit(f"s_waitcnt lgkmcnt(2)")
            self._emit(mfma_step_mxn(0, 0))

            # 2nd fma
            self._emit(f"s_waitcnt lgkmcnt(1)")
            self._emit(mfma_step_mxn(0, 1))

            #       wait global and store to LDS
            self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
            self._emit(f_sst_b())
            self._emit(f"s_waitcnt vmcnt(0)")
            self._emit(f_sst_a())

            #       iteration--
            self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
            self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
            self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())

            # 3rd fma
            self._emit(f"s_waitcnt lgkmcnt({f_sst_a.get_issues() + f_sst_b.get_issues()})")
            self._emit(mfma_step_mxn(1, 0))

            self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {lds_single_size}, v[{v_sst_b_os()}] ; switch double buffer b store")
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {lds_single_size}, v[{v_sst_a_os()}] ; switch double buffer a store")
            #       barrier here!
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")

            #       load next from global
            self._emit(f_gld_b())
            self._emit(f_gld_a())

            # 4th fma
            self._emit(mfma_step_mxn(1, 1))
            self._emit_empty_line()
            self._emit(f"s_branch {label_mfma_body}")

            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            self._emit(f"s_waitcnt lgkmcnt({f_sst_a.get_issues() + f_sst_a.get_issues()})")
            self._emit(mfma_step_mxn(1, 0))
            self._emit(mfma_step_mxn(1, 1))

            # Label: end of fma body
            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_width_m // 2 ))

            self._emit(f".itr_k = 0")
            self._emit(f".rept {unroll_k - 1}")
            with self._indent_context():
                # 1st fma
                self._emit('s_waitcnt lgkmcnt(2)')
                self._emit(mfma_step_mxn(0, 0))

                # 2nd fma
                self._emit('s_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(0, 1))

                # 3rd fma
                self._emit(f_sld_a(v_a(), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m}'))
                self._emit('s_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(1, 0))

                # 4th fma
                self._emit(f_sld_b(v_b(), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}'))
                self._emit(mfma_step_mxn(1, 1))
                self._emit_empty_line()

                # last
                self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}+{lds_width_n//2}'))
                self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m}+{lds_width_m//2}'))
                self._emit('.itr_k = .itr_k + 1')
            self._emit('.endr')
            self._emit_empty_line()
            self._emit('; last unroll')
            # 1st fma
            self._emit('s_waitcnt lgkmcnt(2)')
            self._emit(mfma_step_mxn(0, 0))

            # 2nd fma
            self._emit('s_waitcnt lgkmcnt(1)')
            self._emit(mfma_step_mxn(0, 1))

            # 3rd fma
            self._emit('s_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(1, 0))

            # 4th fma
            self._emit(mfma_step_mxn(1, 1))
            self._emit_empty_line()

        def mfma_loop_repeat_1x1():
            pass

        # start emit
        self._emit(f"; start MFMA loop, {wave_tile_m}x{wave_tile_n} wave tile with {wave_repeat_m}x{wave_repeat_n} repeat, {wave_step_m}x{wave_step_n} step")
        self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")

        self._emit(f_sst_b())
        self._emit_empty_line()
        self._emit(f"s_waitcnt vmcnt(0)")
        self._emit(f_sst_a())
        self._emit_empty_line()

        self._emit(f".v_clear_acc_c {a_c()}, {v_mfma_wave_wise.total_acc_c()}")
        self._emit(f"; make sure acc WAR harzard, at least 1 nop for src_c")

        # decrese k
        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_knum()}], {unroll_k}")
        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
        self._emit(f"s_cbranch_scc0 {label_mfma_end}")
        self._emit_empty_line()

        # diverge based on repeat
        if wave_repeat_m == 2 and wave_repeat_n == 2:
            mfma_loop_repeat_2x2()
        elif wave_repeat_m == 1 and wave_repeat_n == 1:
            mfma_loop_repeat_1x1()
        else:
            assert False, f"un implemented repeat {wave_repeat_m}x{wave_repeat_n}"
