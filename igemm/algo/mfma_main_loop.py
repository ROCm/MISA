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
from .nop import *

class ctrl_mfma_main_loop_t(object):
    def __init__(self):
        self.cxm                         = None

        self.unroll_k                    = 0
        self.lds_gemm_k_pack             = 1
        self.label_prefix                = ''                      # usually be kernel name of caller kernel
        self.data_type                   = AMDGPU_PRECISION_FP32 # c matrix data type
        self.precision                   = 'fp32'                # a/b matrix data type

        self.lds_single_size             = 0                    # in byte, should be power of 2
        self.lds_buffer_num              = 2
        self.local_prefetch_num          = 1
        self.interleave                  = False

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
        cxm = self.ctrl.cxm
        lds_gemm_k_pack = self.ctrl.lds_gemm_k_pack

        data_byte = amdgpu_precision_data_byte(self.ctrl.precision)

        lds_width_m = data_byte * cxm.wave_tile_m * cxm.wave_step_m * cxm.waves_per_m() * cxm.wave_repeat_m
        lds_width_n = data_byte * cxm.wave_tile_n * cxm.wave_step_n * cxm.waves_per_n() * cxm.wave_repeat_n
        lds_single_size = self.ctrl.lds_single_size

        # used as offset:x number. may some 
        lds_base_m = 0
        lds_base_n = 0
        assert self.ctrl.unroll_k % cxm.block_k() == 0
        unroll_k = self.ctrl.unroll_k
        k_per_inst = cxm.block_k()
        #print(f"k_per_inst={k_per_inst}")

        def mfma_step_mxn(i_repeat_m, i_repeat_n, i_local_buffer_m = 0, i_local_buffer_n = 0):
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n
            mfma = cxm.inst_mfma
            num_agpr_per_issue = mfma.num_a_c
            with self._deferred_context():
                for i_step_n in range(cxm.wave_step_n):
                    for i_step_m in range(cxm.wave_step_m):
                        #a_index = i_repeat_m * cxm.wave_tile_m * cxm.wave_step_m + i_repeat_n *cxm.wave_tile_n*cxm.wave_step_n +  i_step_m
                        c_index = i_repeat_m * cxm.wave_step_m * cxm.wave_step_n * cxm.wave_repeat_n * num_agpr_per_issue + \
                                    i_repeat_n * cxm.wave_step_m * cxm.wave_step_n * num_agpr_per_issue + \
                                    i_step_m * cxm.wave_step_n * num_agpr_per_issue + \
                                    i_step_n * num_agpr_per_issue
                        c_index_end = c_index + num_agpr_per_issue - 1
                        a_index = i_local_buffer_m * local_buffer_m + \
                                    i_repeat_m * cxm.wave_step_m * mfma.num_v_a + \
                                    i_step_m * mfma.num_v_a
                        b_index = i_local_buffer_n * local_buffer_n + \
                                    i_repeat_n * cxm.wave_step_n * mfma.num_v_b + \
                                    i_step_n * mfma.num_v_b
                        if mfma.num_v_a == 1 and mfma.num_v_b == 1:
                            self._emit(mfma(a_c((c_index, c_index_end)), v_a(a_index), v_b(b_index), a_c((c_index, c_index_end))) + f"  ; repeat:{i_repeat_m}x{i_repeat_n}, step:{i_step_m}x{i_step_n}, num_a_c:{num_agpr_per_issue}")
                        elif mfma.num_v_a == 2 and mfma.num_v_b == 2:
                            a_index_end = a_index + mfma.num_v_a - 1
                            b_index_end = b_index + mfma.num_v_b - 1
                            self._emit(mfma(a_c((c_index, c_index_end)), v_a((a_index, a_index_end)), v_b((b_index, b_index_end)), a_c((c_index, c_index_end))) + f"  ; repeat:{i_repeat_m}x{i_repeat_n}, step:{i_step_m}x{i_step_n}, num_a_c:{num_agpr_per_issue}")
                        else:
                            assert False, "patern of num v_a or v_b non-valid"
            return self._get_deferred()

        def mfma_loop_repeat_1x1_lp2():
            mfma = cxm.inst_mfma
            #print(f"num_v_a={mfma.num_v_a}")
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            self._emit_empty_line()
            self._emit(f_gld_b())                                           # global load
            self._emit(f_gld_a())                                           # global load
            self._emit_empty_line()

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst))
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst))

            def do_unroll_k_1x1_sub():
                unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                for i_k in range(unroll_k_sub):
                    self._emit(f's_waitcnt lgkmcnt(2)')
                    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m  + (2*i_k+2) * lds_width_m * k_per_inst))
                    self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n  + (2*i_k+2) * lds_width_n * k_per_inst))

                    self._emit(f's_waitcnt lgkmcnt(2)')
                    self._emit(mfma_step_mxn(0, 0, 1, 1))
                    self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m  + (2*i_k+3) * lds_width_m * k_per_inst))
                    self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n  + (2*i_k+3) * lds_width_n * k_per_inst))

            do_unroll_k_1x1_sub()
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))

            self._emit_empty_line()

            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(f"s_barrier")
            self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
            self._emit(f_sst_b())
            self._emit(f"s_waitcnt vmcnt(0)")
            self._emit(f_sst_a())

            self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
            self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
            self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")


            self._emit(mfma_step_mxn(0, 0, 1, 1))
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            #       load next from global
            self._emit(f_gld_b())
            self._emit(f_gld_a())
            self._emit(f"s_branch {label_mfma_body}")
            self._emit_empty_line()

            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            self._emit(mfma_step_mxn(0, 0, 1, 1))

            
            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + k_per_inst * lds_width_m))
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + k_per_inst * lds_width_n))
            do_unroll_k_1x1_sub()
            self._emit(f's_waitcnt lgkmcnt(2)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(0, 0, 1, 1))

        def mfma_loop_repeat_1x1_lp2_with_interleave():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n

            def do_interleave_unroll_k_sub():
                with self._deferred_context():
                    unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                    for i_k in range(unroll_k_sub):
                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m  + (2*i_k+2) * k_per_inst *  lds_width_m))
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n  + (2*i_k+2) * k_per_inst * lds_width_n))

                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m  + (2*i_k+3) * k_per_inst * lds_width_m))
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n  + (2*i_k+3) * k_per_inst * lds_width_n))
                    #if unroll_k_sub == 0:
                    #    self._emit(f's_waitcnt lgkmcnt(2)')
                    #    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    #    self._emit(f's_waitcnt lgkmcnt(0)')
                    #    self._emit(mfma_step_mxn(0, 0, 1, 1))

                return self._get_deferred()

            def do_interleave_gload_and_move_slice_window():
                with self._deferred_context():
                    self._emit(f_gld_b())
                    self._emit(f_gld_a())
                    self._emit(f_move_slice_window_b())
                    self._emit(f_move_slice_window_a())
                return self._get_deferred()

            def do_interleave_unroll_k_last():
                with self._deferred_context():
                    # self._emit(f's_waitcnt lgkmcnt(0)')
                    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    self._emit(mfma_step_mxn(0, 0, 1, 1))

                    self._emit_empty_line()

                    self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                    self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                    self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

                    #self._emit(mfma_step_mxn(0, 0, 1, 1))
                    self._emit(f"s_waitcnt lgkmcnt(0)")
                    self._emit(f"s_barrier")
                    self._emit(f"s_branch {label_mfma_body}")
                    self._emit_empty_line()
                return self._get_deferred()

            def do_interleave_share_store():
                with self._deferred_context():
                    self._emit(f's_waitcnt lgkmcnt(0)')
                    self._emit(f"s_barrier")
                    self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
                    self._emit(f_sst_b())
                    self._emit(f"s_waitcnt vmcnt(0)")
                    self._emit(f_sst_a())
                return self._get_deferred()

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            # self._emit_empty_line()
            # self._emit(f_gld_b())                                           # global load
            # self._emit(f_gld_a())                                           # global load
            # self._emit_empty_line()

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + k_per_inst * lds_width_m))
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + k_per_inst * lds_width_n))


            if ((unroll_k // k_per_inst) // 2 - 1) != 0:
                mbb_list_sub = [create_machine_basic_block(do_interleave_unroll_k_sub(), group_mbb_by_end_of_inst_op="v_mfma"),
                            create_machine_basic_block(do_interleave_gload_and_move_slice_window())]
                se_sub = create_scheduler(self.mc, mbb_list_sub)
                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                             create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)

                self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))
                self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1))
            else:
                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                             create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)
                self._emit(do_interleave_gload_and_move_slice_window())
                self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1))


            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            #self._emit(mfma_step_mxn(0, 0, 1, 1))

            
            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + k_per_inst * lds_width_m))
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + k_per_inst * lds_width_n))
            self._emit(do_interleave_unroll_k_sub())
            self._emit(f's_waitcnt lgkmcnt(2)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(0, 0, 1, 1))

        def mfma_loop_repeat_1x1_lp2_double_buffer_with_interleave():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n

            def do_interleave_unroll_k_sub():
                with self._deferred_context():
                    unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                    for i_k in range(unroll_k_sub):
                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m  + (2*i_k+2) * k_per_inst *  lds_width_m))
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n  + (2*i_k+2) * k_per_inst * lds_width_n))

                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m  + (2*i_k+3) * k_per_inst * lds_width_m))
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n  + (2*i_k+3) * k_per_inst * lds_width_n))
                    #if unroll_k_sub == 0:
                    #    self._emit(f's_waitcnt lgkmcnt(2)')
                    #    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    #    self._emit(f's_waitcnt lgkmcnt(0)')
                    #    self._emit(mfma_step_mxn(0, 0, 1, 1))

                return self._get_deferred()

            def do_interleave_gload_and_move_slice_window():
                with self._deferred_context():
                    #self._emit(f_gld_b())
                    #self._emit(f_gld_a())
                    self._emit(f_move_slice_window_b())
                    self._emit(f_move_slice_window_a())
                return self._get_deferred()

            def do_interleave_unroll_k_last():
                with self._deferred_context():
                    # self._emit(f's_waitcnt lgkmcnt(0)')
                    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    self._emit(mfma_step_mxn(0, 0, 1, 1))

                    self._emit_empty_line()

                    self._emit(f"v_xor_b32 v[{v_sld_b_os()}], {lds_single_size}, v[{v_sld_b_os()}] ; switch double buffer b load")
                    self._emit(f"v_xor_b32 v[{v_sld_a_os()}], {lds_single_size}, v[{v_sld_a_os()}] ; switch double buffer a load")

                    self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                    self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                    self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

                    self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {lds_single_size}, v[{v_sst_b_os()}] ; switch double buffer b store")
                    self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {lds_single_size}, v[{v_sst_a_os()}] ; switch double buffer a store")

                    self._emit(f"s_branch {label_mfma_body}")
                    self._emit_empty_line()
                return self._get_deferred()

            def do_interleave_share_store():
                with self._deferred_context():
                    self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
                    self._emit(f_sst_b())
                    self._emit(f"s_waitcnt vmcnt(0)")
                    self._emit(f_sst_a())
                return self._get_deferred()

            # double buffer switch
            self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {hex(lds_single_size)}, v[{v_sst_b_os()}] ; switch double buffer b store")
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {hex(lds_single_size)}, v[{v_sst_a_os()}] ; switch double buffer a store")

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())

            self._emit_front(f"{label_mfma_body}:")
            self._emit(f_gld_b())                                           # global load
            self._emit(f_gld_a())                                           # global load
            self._emit(f"; do fma accumulate with unroll {unroll_k}")

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            self._emit_empty_line()

            # Label: start of fma body
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + k_per_inst * lds_width_m))
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + k_per_inst * lds_width_n))


            if ((unroll_k // k_per_inst) // 2 - 1) != 0:
                mbb_list_sub = [create_machine_basic_block(do_interleave_unroll_k_sub(), group_mbb_by_end_of_inst_op="v_mfma"),
                            create_machine_basic_block(do_interleave_gload_and_move_slice_window())]
                se_sub = create_scheduler(self.mc, mbb_list_sub)
                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                             create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)

                self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))
                self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1))
            else:
                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                             create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)
                self._emit(do_interleave_gload_and_move_slice_window())
                self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1))


            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            #self._emit(mfma_step_mxn(0, 0, 1, 1))

            
            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + k_per_inst * lds_width_m))
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + k_per_inst * lds_width_n))
            self._emit(do_interleave_unroll_k_sub())
            self._emit(f's_waitcnt lgkmcnt(2)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(0, 0, 1, 1))

        def mfma_loop_repeat_2x2_lp2():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())

            #self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {hex(lds_single_size)}, v[{v_sst_b_os()}] ; switch double buffer b store")
            #self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {hex(lds_single_size)}, v[{v_sst_a_os()}] ; switch double buffer a store")
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            self._emit_empty_line()
            self._emit(f_gld_b())                                           # global load
            self._emit(f_gld_a())                                           # global load
            self._emit_empty_line()

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m // 2 ))

            def do_unroll_k_sub():
                unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                for i_k in range(unroll_k_sub):
                    self._emit(f"; k iteration : {2 * i_k + 0}")
                    # 1st fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k == 0 else 5})')
                    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    if i_k == 0:
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                    else:
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+1) * lds_width_m * k_per_inst + lds_width_m // 2) + f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                    self._emit_empty_line()

                    # 2nd fma
                    self._emit(f's_waitcnt lgkmcnt({3 if i_k == 0 else 5})')
                    self._emit(mfma_step_mxn(0, 1, 0, 0))
                    if i_k == 0:
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n // 2 ) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2 ) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                    else:
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    self._emit_empty_line()

                    # 3rd fma
                    self._emit(f's_waitcnt lgkmcnt({4 if i_k == 0 else 5})')
                    self._emit(mfma_step_mxn(1, 0, 0, 0))
                    if i_k == 0:
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    self._emit_empty_line()

                    # 4th fma
                    self._emit(mfma_step_mxn(1, 1, 0, 0))
                    self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n // 2) + \
                                                f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                    self._emit_empty_line()

                    self._emit(f"; k iteration : {2 * i_k + 1}")
                    # 1st fma
                    self._emit(f's_waitcnt lgkmcnt(5)')
                    self._emit(mfma_step_mxn(0, 0, 1, 1))
                    self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2)+ \
                                                f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                    self._emit_empty_line()

                    # 2nd fma
                    self._emit(f's_waitcnt lgkmcnt(5)')
                    self._emit(mfma_step_mxn(0, 1, 1, 1))
                    self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+3) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                    self._emit_empty_line()

                    # 3rd fma
                    self._emit(f's_waitcnt lgkmcnt(5)')
                    self._emit(mfma_step_mxn(1, 0, 1, 1))
                    self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                    self._emit_empty_line()

                    # 4th fma
                    self._emit(mfma_step_mxn(1, 1, 1, 1))
                    self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst + lds_width_n//2) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}")
                    if i_k == unroll_k_sub - 1:
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (unroll_k // k_per_inst - 1) * lds_width_m * k_per_inst + lds_width_m // 2) + f" ; load i_k:{unroll_k // k_per_inst - 1} into local buffer {1}, repeat {1}")
                    self._emit_empty_line()

            do_unroll_k_sub()

            self._emit(f"; k iteration : {unroll_k - 2}")
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(6)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())
            self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(f"s_barrier")
            self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
            self._emit(f_sst_b())
            self._emit(mfma_step_mxn(0, 1, 0, 0))

            self._emit_empty_line()

            # 3rd fma
            #self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(1, 0, 0, 0))
            self._emit_empty_line()

            # 4th fma
            self._emit(mfma_step_mxn(1, 1, 0, 0))
            self._emit(f"s_waitcnt vmcnt(0)")
            self._emit(f_sst_a())

            self._emit(f"; k iteration : {unroll_k - 1}")
            # 1st fma
            #self._emit(f's_waitcnt lgkmcnt(0)')
            #self._emit(f"s_barrier")
            #       wait global and store to LDS
            

            self._emit(mfma_step_mxn(0, 0, 1, 1))
            

            self._emit_empty_line()

            # 2nd fma
            self._emit(mfma_step_mxn(0, 1, 1, 1))
            self._emit_empty_line()
            #       iteration--
            self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
            self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
            self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

            # 3rd fma
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit_empty_line()

            # 4th fma
            self._emit(mfma_step_mxn(1, 1, 1, 1))
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            #       load next from global
            self._emit(f_gld_b())
            self._emit(f_gld_a())
            self._emit(f"s_branch {label_mfma_body}")
            self._emit_empty_line()

            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            # 3rd fma
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit_empty_line()

            # 4th fma
            self._emit(mfma_step_mxn(1, 1, 1, 1))
            self._emit_empty_line()

            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m // 2 ))
            do_unroll_k_sub()
            self._emit(f"; k iteration : {unroll_k - 2}")
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(6)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))
            self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(5)')
            self._emit(mfma_step_mxn(0, 1, 0, 0))
            self._emit_empty_line()

            # 3rd fma
            self._emit(f's_waitcnt lgkmcnt(4)')
            self._emit(mfma_step_mxn(1, 0, 0, 0))
            self._emit_empty_line()

            # 4th fma
            self._emit(mfma_step_mxn(1, 1, 0, 0))
            #       iteration--

            self._emit(f"; k iteration : {unroll_k - 1}")
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(2)')
            self._emit(mfma_step_mxn(0, 0, 1, 1))
            self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(1)')
            self._emit(mfma_step_mxn(0, 1, 1, 1))
            self._emit_empty_line()

            # 3rd fma
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit_empty_line()

            # 4th fma
            self._emit(mfma_step_mxn(1, 1, 1, 1))
            self._emit_empty_line()

        def mfma_loop_repeat_2x2_lp2_double_buffer_with_interleave():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n

            # right after clear acc
            def do_interleave_move_slice_window():
                with self._deferred_context():
                    #self._emit(f_gld_b())                                           # global load
                    #self._emit(f_gld_a())                                           # global load
                    #self._emit_empty_line()
                    self._emit(f_move_slice_window_b())
                    self._emit(f_move_slice_window_a())
                    self._emit_empty_line()
                return self._get_deferred()

            def do_interleave_global_load():
                with self._deferred_context():
                    self._emit(f_gld_b())                                           # global load
                    self._emit(f_gld_a())                                           # global load
                    self._emit_empty_line()
                return self._get_deferred()

            self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {hex(lds_single_size)}, v[{v_sst_b_os()}] ; switch double buffer b store")
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {hex(lds_single_size)}, v[{v_sst_a_os()}] ; switch double buffer a store")

            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())

            self._emit_front(f"{label_mfma_body}:")
            self._emit(f_gld_b())                                           # global load
            self._emit(f_gld_a()) 
            self._emit(f"; do fma accumulate with unroll {unroll_k}")

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            self._emit_empty_line()

            # Label: start of fma body
            
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m // 2 ))

            #self._emit(do_interleave_move_slice_window())
            #self._emit(do_interleave_global_load())

            def do_interleave_unroll_k_sub_lds_double_buffer():
                with self._deferred_context():
                    unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                    for i_k in range(unroll_k_sub):
                        self._emit(f"; k iteration : {2 * i_k + 0}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k == 0 else 5})')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        else:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+1) * lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2) + f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({3 if i_k == 0 else 5})')
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n // 2 ) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2 ) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        else:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        # 3rd fma
                        self._emit(f's_waitcnt lgkmcnt({4 if i_k == 0 else 5})')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 0, 0))
                        self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n // 2) + \
                                                f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit_empty_line()

                        self._emit(f"; k iteration : {2 * i_k + 1}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2)+ \
                                                f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(0, 1, 1, 1))
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+3) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 3rd fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 1, 1))
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n // 2) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}")
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (unroll_k // k_per_inst - 1) * lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2) + f" ; load i_k:{unroll_k // k_per_inst - 1} into local buffer {1}, repeat {1}")
                        self._emit_empty_line()
                    if unroll_k_sub == 0:
                        self._emit(f"; k iteration : {0}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({2})')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()
                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({3})')
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n // 2 ) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2 ) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        self._emit_empty_line()
                        # 3rd fma
                        self._emit(f's_waitcnt lgkmcnt({4})')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        self._emit_empty_line()
                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 0, 0))
                        self._emit_empty_line()

                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt(1)')
                        self._emit(mfma_step_mxn(0, 1, 1, 1))
                        self._emit_empty_line()
                return self._get_deferred()

            #self._emit(do_interleave_unroll_k_sub_lds_double_buffer())

            mbb_list_sub = [create_machine_basic_block(do_interleave_unroll_k_sub_lds_double_buffer(), group_mbb_by_end_of_inst_op="v_mfma"),
                            create_machine_basic_block(do_interleave_move_slice_window())]

            se_sub = create_scheduler(self.mc, mbb_list_sub)
            self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))

            self._emit(f"; k iteration : {unroll_k - 2 * k_per_inst}")
            
            self._emit(f's_waitcnt lgkmcnt(0)')

            def do_interleave_unroll_k_last_double_buffer():
                unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                with self._deferred_context():
                    if unroll_k_sub > 0:
                        # 1st fma
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        # 2nd fma
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                        self._emit_empty_line()

                        # 3rd fma
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 0, 0))

                        self._emit(f"; k iteration : {unroll_k - (2 + 1) * k_per_inst}")
                        # 1st fma
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(mfma_step_mxn(0, 1, 1, 1))
                        self._emit_empty_line()

                        self._emit(f"v_xor_b32 v[{v_sld_b_os()}], {lds_single_size}, v[{v_sld_b_os()}] ; switch double buffer b load")
                        self._emit(f"v_xor_b32 v[{v_sld_a_os()}], {lds_single_size}, v[{v_sld_a_os()}] ; switch double buffer a load")

                        # 3rd fma
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 1, 1))

                        #       iteration--
                        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                        self._emit(f"s_cbranch_scc0 {label_mfma_end}")

                        self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {lds_single_size}, v[{v_sst_b_os()}] ; switch double buffer b store")
                        self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {lds_single_size}, v[{v_sst_a_os()}] ; switch double buffer a store")

                        
                        #self._emit(do_interleave_global_load())
                        #       load next from global
                        self._emit(f"s_branch {label_mfma_body}")
                        self._emit_empty_line()
                    else:
                        # 3rd fma
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 1, 1))

                        self._emit(f"v_xor_b32 v[{v_sld_b_os()}], {lds_single_size}, v[{v_sld_b_os()}] ; switch double buffer b load")
                        self._emit(f"v_xor_b32 v[{v_sld_a_os()}], {lds_single_size}, v[{v_sld_a_os()}] ; switch double buffer a load")

                        #       iteration--
                        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                        self._emit(f"s_cbranch_scc0 {label_mfma_end}")

                        self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {lds_single_size}, v[{v_sst_b_os()}] ; switch double buffer b store")
                        self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {lds_single_size}, v[{v_sst_a_os()}] ; switch double buffer a store")

                        #self._emit(do_interleave_global_load())
                        #       load next from global
                        self._emit(f"s_branch {label_mfma_body}")
                        self._emit_empty_line()

                return self._get_deferred()
            
            #self._emit(do_interleave_unroll_k_last_double_buffer())

            def do_interleave_share_store_double_buffer():
                with self._deferred_context():
                    self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
                    self._emit(f_sst_b())
           
                    self._emit(f"s_waitcnt vmcnt(0)")
                    self._emit(f_sst_a())
                return self._get_deferred()

            #self._emit(do_interleave_share_store_double_buffer())
            unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1

            mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last_double_buffer(), group_mbb_by_end_of_inst_op="v_mfma"),
                             create_machine_basic_block(do_interleave_share_store_double_buffer(), group_mbb_by_end_of_inst_op="ds_write")]

            #for x in mbb_list_last:
            #    print(f'len x:{len(x)}')
            #    for y in x:
            #        y.dump()
            se_last = create_scheduler(self.mc, mbb_list_last)
            mbb_0_mfma_cnt_after_branch_to_start = 0 # if unroll_k_sub == 0 else 2 * cxm.wave_step_m * cxm.wave_step_n - 1 # number of mfma not count into share store interleave slot, check do_interleave_unroll_k_last for last 2 mfma
            self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1, mbb_0_mfma_cnt_after_branch_to_start=mbb_0_mfma_cnt_after_branch_to_start))

            
            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m // 2 ))

            self._emit(do_interleave_unroll_k_sub_lds_double_buffer())

            if unroll_k_sub > 0:

                self._emit(f"; k iteration : {unroll_k - 2 * k_per_inst}")
                # 1st fma
                self._emit(f's_waitcnt lgkmcnt(6)')
                self._emit(mfma_step_mxn(0, 0, 0, 0))
                self._emit_empty_line()

                # 2nd fma
                self._emit(f's_waitcnt lgkmcnt(5)')
                self._emit(mfma_step_mxn(0, 1, 0, 0))
                self._emit_empty_line()

                # 3rd fma
                self._emit(f's_waitcnt lgkmcnt(4)')
                self._emit(mfma_step_mxn(1, 0, 0, 0))
                self._emit_empty_line()

                # 4th fma
                self._emit(mfma_step_mxn(1, 1, 0, 0))
                #       iteration--

                self._emit(f"; k iteration : {unroll_k - 1 * k_per_inst}")
                # 1st fma
                self._emit(f's_waitcnt lgkmcnt(2)')
                self._emit(mfma_step_mxn(0, 0, 1, 1))
                self._emit_empty_line()

                # 2nd fma
                self._emit(f's_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(0, 1, 1, 1))
                self._emit_empty_line()

            # 3rd fma
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit_empty_line()

            # 4th fma
            self._emit(mfma_step_mxn(1, 1, 1, 1))
            self._emit_empty_line()


        def mfma_loop_repeat_2x2_lp2_double_buffer():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n

            self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {hex(lds_single_size)}, v[{v_sst_b_os()}] ; switch double buffer b store")
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {hex(lds_single_size)}, v[{v_sst_a_os()}] ; switch double buffer a store")

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            self._emit_empty_line()

            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())

            self._emit(f_gld_b())                                           # global load
            self._emit(f_gld_a())                                           # global load
            self._emit_empty_line()

            ## load the a and b data of the first repeat of the first k_per_inst of the first iteration
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            ## load the b and a data of the second repeat of the first k_per_inst of the first iteration
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m // 2 ))

            def do_unroll_k_sub():
                with self._deferred_context():
                    unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                    for i_k in range(unroll_k_sub):
                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))

                        self._emit(f's_waitcnt lgkmcnt(1)')
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                   
                        ## load the a and b data of the first repeat of the second k_per_inst of the current iteration
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+1)*lds_width_m * k_per_inst) +  \
                                                                          f" ; load a i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+1)*lds_width_n * k_per_inst) +  \
                                                                          f" ; load b i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        self._emit(mfma_step_mxn(1, 1, 0, 0))

                        ## load the b and a data of the second repeat of the second k_per_inst of the current iteration
                        self._emit(f_sld_b(v_b(local_buffer_n+repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+1)*lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n // 2) +  \
                                                                          f" ; load b i_k:{1} into local buffer {1}, repeat {1}")
                        self._emit(f_sld_a(v_a(local_buffer_m+repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+1)*lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2) +  \
                                                                          f" ; load a i_k:{1} into local buffer {1}, repeat {1}")
                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))

                        self._emit(f's_waitcnt lgkmcnt(1)')
                        self._emit(mfma_step_mxn(0, 1, 1, 1))

                        ## load the a and b data of the first repeat of the first k_per_inst of the next iteration
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2)*lds_width_m * k_per_inst) +  \
                                                                          f" ; load a i_k:{1} into local buffer {0}, repeat {0}")
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + (2*i_k+2)*lds_width_n * k_per_inst) +  \
                                                                          f" ; load b i_k:{1} into local buffer {0}, repeat {0}")
                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        self._emit(mfma_step_mxn(1, 1, 1, 1))

                        ## load the b and a data of the second repeat of the first k_per_inst of the next iteration
                        self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+2)*lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n // 2) +  \
                                                                          f" ; load b i_k:{2} into local buffer {0}, repeat {1}")
                        self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+2)*lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2) +  \
                                                                          f" ; load a i_k:{2} into local buffer {0}, repeat {1}")
                return self._get_deferred() 

            self._emit(do_unroll_k_sub())

            self._emit(f's_waitcnt lgkmcnt(2)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))

            self._emit(f's_waitcnt lgkmcnt(1)')
            self._emit(mfma_step_mxn(0, 1, 0, 0))

            unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
            i_k_per_inst = unroll_k // k_per_inst -1 if unroll_k_sub > 0 else 1

            ## load the a and b data of the first repeat of the second k_per_inst of the last iteration
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + i_k_per_inst*lds_width_m * k_per_inst) +  \
                                                                          f" ; load a i_k:{1} into local buffer {1}, repeat {0}")
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + i_k_per_inst*lds_width_n * k_per_inst) +  \
                                                                          f" ; load b i_k:{1} into local buffer {1}, repeat {0}")
            self._emit(f's_waitcnt lgkmcnt(2)')
            self._emit(mfma_step_mxn(1, 0, 0, 0))
            self._emit(mfma_step_mxn(1, 1, 0, 0))

            ## load the b and a data of the second repeat of the second k_per_inst of the last iteration
            self._emit(f_sld_b(v_b(local_buffer_n+repeat_n_thread_offset), v_sld_b_os(), lds_base_n + i_k_per_inst*lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n // 2) +  \
                                                                          f" ; load b i_k:{1} into local buffer {1}, repeat {1}")
            self._emit(f_sld_a(v_a(local_buffer_m+repeat_m_thread_offset), v_sld_a_os(), lds_base_m + i_k_per_inst*lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2) +  \
                                                                          f" ; load a i_k:{1} into local buffer {1}, repeat {1}")
            self._emit(f's_waitcnt lgkmcnt(2)')
            self._emit(mfma_step_mxn(0, 0, 1, 1))
            self._emit(f's_waitcnt lgkmcnt(1)')
            self._emit(mfma_step_mxn(0, 1, 1, 1))

            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit(mfma_step_mxn(1, 1, 1, 1))

            #  wait for the global load to be done 
            self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
            self._emit(f_sst_b())
            self._emit(f"s_waitcnt vmcnt(0)")
            self._emit(f_sst_a())

            self._emit(f"v_xor_b32 v[{v_sld_b_os()}], {lds_single_size}, v[{v_sld_b_os()}] ; switch double buffer b load")
            self._emit(f"v_xor_b32 v[{v_sld_a_os()}], {lds_single_size}, v[{v_sld_a_os()}] ; switch double buffer a load")

            #  check the left number of unroll-k 
            self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
            self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
            self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

            self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {lds_single_size}, v[{v_sst_b_os()}] ; switch double buffer b store")
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {lds_single_size}, v[{v_sst_a_os()}] ; switch double buffer a store")

            self._emit(f"s_branch {label_mfma_body}")
            self._emit_empty_line()

            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")

            # do the last unroll_k
            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m // 2 ))

            self._emit(do_unroll_k_sub())

            self._emit(f's_waitcnt lgkmcnt(2)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))

            self._emit(f's_waitcnt lgkmcnt(1)')
            self._emit(mfma_step_mxn(0, 1, 0, 0))

            unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
            i_k_per_inst = unroll_k // k_per_inst -1 if unroll_k_sub > 0 else 1

            ## load the a and b data of the first repeat of the second k_per_inst of the last iteration
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + i_k_per_inst*lds_width_m * k_per_inst) +  \
                                                                          f" ; load a i_k:{1} into local buffer {1}, repeat {0}")
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + i_k_per_inst*lds_width_n * k_per_inst) +  \
                                                                          f" ; load b i_k:{1} into local buffer {1}, repeat {0}")
            self._emit(f's_waitcnt lgkmcnt(2)')
            self._emit(mfma_step_mxn(1, 0, 0, 0))
            self._emit(mfma_step_mxn(1, 1, 0, 0))

            ## load the b and a data of the second repeat of the second k_per_inst of the last iteration
            self._emit(f_sld_b(v_b(local_buffer_n+repeat_n_thread_offset), v_sld_b_os(), lds_base_n + i_k_per_inst*lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n // 2) +  \
                                                                          f" ; load b i_k:{1} into local buffer {1}, repeat {1}")
            self._emit(f_sld_a(v_a(local_buffer_m+repeat_m_thread_offset), v_sld_a_os(), lds_base_m + i_k_per_inst*lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2) +  \
                                                                          f" ; load a i_k:{1} into local buffer {1}, repeat {1}")
            self._emit(f's_waitcnt lgkmcnt(2)')
            self._emit(mfma_step_mxn(0, 0, 1, 1))
            self._emit(f's_waitcnt lgkmcnt(1)')
            self._emit(mfma_step_mxn(0, 1, 1, 1))

            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit(mfma_step_mxn(1, 1, 1, 1))



        def mfma_loop_repeat_2x2_lp2_with_interleave():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n

            def do_interleave_unroll_k_sub():
                with self._deferred_context():
                    unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                    for i_k in range(unroll_k_sub):
                        self._emit(f"; k iteration : {2 * i_k + 0}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k == 0 else 5})')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        else:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+1) * lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2) + f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({3 if i_k == 0 else 5})')
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n // 2 ) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2 ) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        else:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        # 3rd fma
                        self._emit(f's_waitcnt lgkmcnt({4 if i_k == 0 else 5})')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 0, 0))
                        self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n // 2) + \
                                                    f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit_empty_line()

                        self._emit(f"; k iteration : {2 * i_k + 1}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2)+ \
                                                    f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(0, 1, 1, 1))
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+3) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 3rd fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 1, 1))
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n//2) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}")
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (unroll_k // k_per_inst - 1) * lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2) + f" ; load i_k:{unroll_k // k_per_inst - 1} into local buffer {1}, repeat {1}")
                        self._emit_empty_line()
                    if unroll_k_sub == 0:
                        self._emit(f"; k iteration : {0}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({2})')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()
                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({3})')
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst + lds_gemm_k_pack * lds_width_n // 2 ) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst + lds_gemm_k_pack * lds_width_m // 2 ) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        self._emit_empty_line()
                        # 3rd fma
                        self._emit(f's_waitcnt lgkmcnt({4})')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        self._emit_empty_line()
                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 0, 0))
                        self._emit_empty_line()

                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt(1)')
                        self._emit(mfma_step_mxn(0, 1, 1, 1))
                        self._emit_empty_line()

                        # 3rd and 4th will be compute with finish branch
                        # 3rd fma
                        #self._emit(f's_waitcnt lgkmcnt(0)')
                        #self._emit(mfma_step_mxn(1, 0, 1, 1))
                        #self._emit_empty_line()

                        # 4th fma
                        #self._emit(mfma_step_mxn(1, 1, 1, 1))

                return self._get_deferred()

            def do_interleave_gload_and_move_slice_window():
                with self._deferred_context():
                    self._emit(f_gld_b())
                    self._emit(f_gld_a())
                    self._emit(f_move_slice_window_b())
                    self._emit(f_move_slice_window_a())
                return self._get_deferred()

            def do_interleave_unroll_k_last():
                with self._deferred_context():
                    unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                    if unroll_k_sub != 0:
                        self._emit(f"; k iteration : {unroll_k - 2}")
                        # 1st fma
                        # self._emit(f's_waitcnt lgkmcnt(6)')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(mfma_step_mxn(0, 1, 0, 0))

                        self._emit_empty_line()

                        # 3rd fma
                        #self._emit(f's_waitcnt lgkmcnt(0)')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 0, 0))

                        self._emit(f"; k iteration : {unroll_k - 1}")
                        # 1st fma
                        self._emit(mfma_step_mxn(0, 0, 1, 1))

                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(mfma_step_mxn(0, 1, 1, 1))
                        self._emit_empty_line()

                        # 3rd fma
                        #self._emit(mfma_step_mxn(1, 0, 1, 1))
                        self._emit_empty_line()

                        # 4th fma
                        #self._emit(mfma_step_mxn(1, 1, 1, 1))

                        #       iteration--
                        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                        self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

                        # 3rd fma
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        #self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 1, 1))

                        self._emit(f"s_waitcnt lgkmcnt(0)")
                        self._emit(f"s_barrier")
                        self._emit(f"s_branch {label_mfma_body}")
                    else:
                        #       iteration--
                        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                        self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")
                        # 3rd fma
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 1, 1))
                        self._emit(f"s_waitcnt lgkmcnt(0)")
                        self._emit(f"s_barrier")
                        self._emit(f"s_branch {label_mfma_body}")
                return self._get_deferred()

            def do_interleave_share_store():
                with self._deferred_context():
                    self._emit(f's_waitcnt lgkmcnt(0)')
                    self._emit(f"s_barrier")
                    self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
                    self._emit(f_sst_b())
                    self._emit(f"s_waitcnt vmcnt(0)")
                    self._emit(f_sst_a())
                return self._get_deferred()

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m // 2 ))


            mbb_list_sub = [create_machine_basic_block(do_interleave_unroll_k_sub(), group_mbb_by_end_of_inst_op="v_mfma"),
                            create_machine_basic_block(do_interleave_gload_and_move_slice_window())]
            
            #for x in mbb_list_sub:
            #    print(f'len x:{len(x)}')
            #    for y in x:
            #        y.dump()

            se_sub = create_scheduler(self.mc, mbb_list_sub)

            mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                             create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

            #for x in mbb_list_last:
            #    print(f'len x:{len(x)}')
            #    for y in x:
            #        y.dump()
            se_last = create_scheduler(self.mc, mbb_list_last)
            self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))
            mbb_0_mfma_cnt_after_branch_to_start = 2 * cxm.wave_step_m * cxm.wave_step_n - 1 # number of mfma not count into share store interleave slot, check do_interleave_unroll_k_last for last 2 mfma
            self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1, mbb_0_mfma_cnt_after_branch_to_start=mbb_0_mfma_cnt_after_branch_to_start))


            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            # 3rd fma
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            # self._emit_empty_line()

            # 4th fma
            self._emit(mfma_step_mxn(1, 1, 1, 1))
            #self._emit_empty_line()

            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m // 2 ))
            self._emit(do_interleave_unroll_k_sub())

            unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
            if unroll_k_sub > 0:
                self._emit(f"; k iteration : {unroll_k - 2}")
                # 1st fma
                self._emit(f's_waitcnt lgkmcnt(6)')
                self._emit(mfma_step_mxn(0, 0, 0, 0))
                self._emit_empty_line()

                # 2nd fma
                self._emit(f's_waitcnt lgkmcnt(5)')
                self._emit(mfma_step_mxn(0, 1, 0, 0))
                self._emit_empty_line()

                # 3rd fma
                self._emit(f's_waitcnt lgkmcnt(4)')
                self._emit(mfma_step_mxn(1, 0, 0, 0))
                self._emit_empty_line()

                # 4th fma
                self._emit(mfma_step_mxn(1, 1, 0, 0))

                self._emit(f"; k iteration : {unroll_k - 1}")
                # 1st fma
                self._emit(f's_waitcnt lgkmcnt(2)')
                self._emit(mfma_step_mxn(0, 0, 1, 1))
                self._emit_empty_line()

                # 2nd fma
                self._emit(f's_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(0, 1, 1, 1))
                self._emit_empty_line()

            # 3rd fma
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit_empty_line()

            # 4th fma
            self._emit(mfma_step_mxn(1, 1, 1, 1))
            self._emit_empty_line()

        def mfma_loop_repeat_2x2():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b

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
            self._emit(f"; do fma accumulate with unroll {unroll_k // k_per_inst}")
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m // 2 ))

            self._emit(f".itr_k = 0")
            self._emit(f".rept {unroll_k // k_per_inst - 1}")
            with self._indent_context():
                # 1st fma
                self._emit(f's_waitcnt lgkmcnt(2)')
                self._emit(mfma_step_mxn(0, 0))

                # 2nd fma
                self._emit(f's_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(0, 1))

                # 3rd fma
                self._emit(f_sld_a(v_a(), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m * k_per_inst}'))
                self._emit(f's_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(1, 0))

                # 4th fma
                self._emit(f_sld_b(v_b(), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n * k_per_inst}'))
                self._emit(mfma_step_mxn(1, 1))
                self._emit_empty_line()

                # last
                self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n * k_per_inst}+{lds_width_n//2}'))
                self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m * k_per_inst}+{lds_width_m//2}'))
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
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m // 2 ))

            self._emit(f".itr_k = 0")
            self._emit(f".rept {unroll_k // k_per_inst - 1}")
            with self._indent_context():
                # 1st fma
                self._emit('s_waitcnt lgkmcnt(2)')
                self._emit(mfma_step_mxn(0, 0))

                # 2nd fma
                self._emit('s_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(0, 1))

                # 3rd fma
                self._emit(f_sld_a(v_a(), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m * k_per_inst}'))
                self._emit('s_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(1, 0))

                # 4th fma
                self._emit(f_sld_b(v_b(), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n * k_per_inst}'))
                self._emit(mfma_step_mxn(1, 1))
                self._emit_empty_line()

                # last
                self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n * k_per_inst}+{lds_width_n//2}'))
                self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m * k_per_inst}+{lds_width_m//2}'))
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

        def mfma_loop_repeat_2x2_double_buffer():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b

            self._emit_empty_line()
            self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {hex(lds_single_size)}, v[{v_sst_b_os()}] ; switch double buffer b store")
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {hex(lds_single_size)}, v[{v_sst_a_os()}] ; switch double buffer a store")
            self._emit_empty_line()

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with  {unroll_k // k_per_inst} k_per_inst per size {unroll_k} along k-dimension")

            ## wait the shared storing to finish by the whole work-group
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")

            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())
            self._emit_empty_line()
            self._emit(f_gld_b())                                           # global load
            self._emit(f_gld_a())                                           # global load
            self._emit_empty_line()

            ## load the first k_per_inst rows of data from LDS to vgprs
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n// 2))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m// 2))

            self._emit_empty_line()

            def do_unroll_k_2x2_sub():
                with self._deferred_context():
                    unroll_k_sub = unroll_k // k_per_inst - 1
                    for i_k in range(unroll_k_sub):
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(0, 0))

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt(1)')
                        self._emit(mfma_step_mxn(0, 1))

                        # 3rd fma
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m+(i_k+1)*lds_width_m * k_per_inst))
                        self._emit(f's_waitcnt lgkmcnt(1)')
                        self._emit(mfma_step_mxn(1, 0))

                        # 4th fma
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n+(i_k+1)*lds_width_n * k_per_inst))
                        self._emit(mfma_step_mxn(1, 1))
                        self._emit_empty_line()

                        self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (i_k+1)*lds_width_n*k_per_inst + lds_gemm_k_pack * lds_width_n//2))
                        self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (i_k+1)*lds_width_m*k_per_inst + lds_gemm_k_pack * lds_width_m//2))
                return self._get_deferred()

            self._emit(do_unroll_k_2x2_sub())

            self._emit(f"; last k_per_inst of unroll_k")

            # 1st fma
            self._emit(f"s_waitcnt lgkmcnt(2)")
            self._emit(mfma_step_mxn(0, 0))

            # 2nd fma
            self._emit(f"s_waitcnt lgkmcnt(1)")
            self._emit(mfma_step_mxn(0, 1))

            # wait global load to finish and store data to LDS
            self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
            self._emit(f_sst_b())
            self._emit(f"s_waitcnt vmcnt(0)")
            self._emit(f_sst_a())

            self._emit(f"v_xor_b32 v[{v_sld_b_os()}], {lds_single_size}, v[{v_sld_b_os()}] ; switch double buffer b load")
            self._emit(f"v_xor_b32 v[{v_sld_a_os()}], {lds_single_size}, v[{v_sld_a_os()}] ; switch double buffer a load")

            # check the left number of unroll_k to do
            self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
            self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
            self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

            # 3rd and 4th fma
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(mfma_step_mxn(1, 0))
            self._emit(mfma_step_mxn(1, 1))
            self._emit_empty_line()

            self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {lds_single_size}, v[{v_sst_b_os()}] ; switch double buffer b store")
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {lds_single_size}, v[{v_sst_a_os()}] ; switch double buffer a store")

            self._emit_empty_line()
            self._emit(f"s_branch {label_mfma_body}")

            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            self._emit(mfma_step_mxn(1, 0))
            self._emit(mfma_step_mxn(1, 1))

            # Label: end of fma body
            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m // 2 ))

            self._emit(do_unroll_k_2x2_sub())

            self._emit_empty_line()
            self._emit('; last k_per_inst of unroll_k')
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

        def mfma_loop_repeat_2x2_lds_double_buffer_with_interleave():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b

            self._emit_empty_line()
            self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {hex(lds_single_size)}, v[{v_sst_b_os()}] ; switch double buffer b store")
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {hex(lds_single_size)}, v[{v_sst_a_os()}] ; switch double buffer a store")
            self._emit_empty_line()

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with  {unroll_k // k_per_inst} k_per_inst per size {unroll_k} along k-dimension")

            ## wait the shared storing to finish by the whole work-group
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")

            

            ## load the first k_per_inst rows of data from LDS to vgprs
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n// 2))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m// 2))

            self._emit_empty_line()

            def do_interleave_gload_and_move_slice_window():
                with self._deferred_context():
                    self._emit(f_move_slice_window_b())
                    self._emit(f_move_slice_window_a())
                    self._emit(f_gld_b())
                    self._emit(f_gld_a())
                return self._get_deferred()

            #self._emit(do_interleave_gload_and_move_slice_window())

            def do_unroll_k_2x2_sub():
                with self._deferred_context():
                    unroll_k_sub = unroll_k // k_per_inst - 1
                    for i_k in range(unroll_k_sub):
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(0, 0))

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt(1)')
                        self._emit(mfma_step_mxn(0, 1))

                        # 3rd fma
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m+(i_k+1)*lds_width_m * k_per_inst))
                        self._emit(f's_waitcnt lgkmcnt(1)')
                        self._emit(mfma_step_mxn(1, 0))

                        # 4th fma
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n+(i_k+1)*lds_width_n * k_per_inst))
                        self._emit(mfma_step_mxn(1, 1))
                        self._emit_empty_line()

                        self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (i_k+1)*lds_width_n*k_per_inst + lds_gemm_k_pack * lds_width_n//2))
                        self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (i_k+1)*lds_width_m*k_per_inst + lds_gemm_k_pack * lds_width_m//2))
                return self._get_deferred()

            #self._emit(do_unroll_k_2x2_sub())
            mbb_list_sub = [create_machine_basic_block(do_unroll_k_2x2_sub(), group_mbb_by_end_of_inst_op="v_mfma"),
                            create_machine_basic_block(do_interleave_gload_and_move_slice_window())]

            se_sub = create_scheduler(self.mc, mbb_list_sub)

            self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))

            self._emit(f"; last k_per_inst of unroll_k")

            
            def do_unroll_k_last():
                with self._deferred_context():
                    # 1st fma
                    self._emit(f"s_waitcnt lgkmcnt(2)")
                    self._emit(mfma_step_mxn(0, 0))

                    # 2nd fma
                    self._emit(f"s_waitcnt lgkmcnt(1)")
                    self._emit(mfma_step_mxn(0, 1))
                return self._get_deferred()

            

            # wait global load to finish and store data to LDS
            def do_shared_store():
                with self._deferred_context():
                    self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
                    self._emit(f_sst_b())
                    self._emit(f"s_waitcnt vmcnt(0)")
                    self._emit(f_sst_a())
                return self._get_deferred()

            mbb_list_last = [create_machine_basic_block(do_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                             create_machine_basic_block(do_shared_store(), group_mbb_by_end_of_inst_op="ds_write")]

            se_last = create_scheduler(self.mc, mbb_list_last)
            
            mbb_0_mfma_cnt_after_branch_to_start = 0#2 * cxm.wave_step_m * cxm.wave_step_n - 1 # number of mfma not count into share store interleave slot, check do_interleave_unroll_k_last for last 2 mfma
            self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1, mbb_0_mfma_cnt_after_branch_to_start=mbb_0_mfma_cnt_after_branch_to_start))


            self._emit(f"v_xor_b32 v[{v_sld_b_os()}], {lds_single_size}, v[{v_sld_b_os()}] ; switch double buffer b load")
            self._emit(f"v_xor_b32 v[{v_sld_a_os()}], {lds_single_size}, v[{v_sld_a_os()}] ; switch double buffer a load")

            # check the left number of unroll_k to do
            self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
            self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
            self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

            # 3rd and 4th fma
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(mfma_step_mxn(1, 0))
            self._emit(mfma_step_mxn(1, 1))
            self._emit_empty_line()

            self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {lds_single_size}, v[{v_sst_b_os()}] ; switch double buffer b store")
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {lds_single_size}, v[{v_sst_a_os()}] ; switch double buffer a store")

            self._emit_empty_line()
            self._emit(f"s_branch {label_mfma_body}")

            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            self._emit(mfma_step_mxn(1, 0))
            self._emit(mfma_step_mxn(1, 1))

            # Label: end of fma body
            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_gemm_k_pack * lds_width_n // 2 ))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_gemm_k_pack * lds_width_m // 2 ))

            self._emit(do_unroll_k_2x2_sub())

            self._emit_empty_line()
            self._emit('; last k_per_inst of unroll_k')
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

        def mfma_loop_repeat_2x1_lp2():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            # repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            self._emit_empty_line()
            self._emit(f_gld_b())                                           # global load
            self._emit(f_gld_a())                                           # global load
            self._emit_empty_line()

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_width_m // 2 ))

            def do_unroll_k_sub():
                unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                for i_k in range(unroll_k_sub):
                    self._emit(f"; k iteration : {(2 * i_k + 0) * k_per_inst}")
                    # 1st fma
                    self._emit(f's_waitcnt lgkmcnt({1 if i_k == 0 else 2})')
                    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    if i_k == 0:
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        if unroll_k_sub == 1:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst + lds_width_m // 2 ) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                    elif i_k == unroll_k_sub - 1:
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+1) * lds_width_m * k_per_inst) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+1) * lds_width_m * k_per_inst + lds_width_m // 2 ) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                    else:
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+1) * lds_width_m * k_per_inst) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                    self._emit_empty_line()

                    # 2nd fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                    self._emit(mfma_step_mxn(1, 0, 0, 0))
                    if i_k == unroll_k_sub - 1:
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    else:
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+1) * lds_width_m * k_per_inst + lds_width_m // 2 ) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    self._emit_empty_line()

                    self._emit(f"; k iteration : {(2 * i_k + 1) * k_per_inst}")
                    # 1st fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                    self._emit(mfma_step_mxn(0, 0, 1, 1))
                    if i_k == unroll_k_sub - 1:
                        self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst + lds_width_m // 2)+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+3) * lds_width_m * k_per_inst) + f" ; load i_k:{(2*i_k+3)} into local buffer {1}, repeat {0}")

                    else:
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    self._emit_empty_line()

                    # 2nd fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 4})')
                    self._emit(mfma_step_mxn(1, 0, 1, 1))
                    if i_k == unroll_k_sub - 1:
                        # v_b attension!
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+3) * lds_width_m * k_per_inst + lds_width_m // 2) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}")
                    else:
                        self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst + lds_width_m // 2)+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                    self._emit_empty_line()

            do_unroll_k_sub()

            self._emit(f"; k iteration : {unroll_k - 2 * k_per_inst}")
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(4)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())
            self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(f"s_barrier")
            self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
            self._emit(f_sst_b())
            self._emit(mfma_step_mxn(1, 0, 0, 0))
            self._emit_empty_line()

            # 1st fma
            self._emit(f"s_waitcnt vmcnt(0)")
            self._emit(f_sst_a())
            self._emit(f"; k iteration : {unroll_k - 1 * k_per_inst}")
            self._emit(mfma_step_mxn(0, 0, 1, 1))

            self._emit_empty_line()
            #       iteration--
            self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
            self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
            self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

            # 2nd fma
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit_empty_line()
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            #       load next from global
            self._emit(f_gld_b())
            self._emit(f_gld_a())
            self._emit(f"s_branch {label_mfma_body}")
            self._emit_empty_line()

            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit_empty_line()

            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_width_m // 2 ))
            do_unroll_k_sub()
            self._emit(f"; k iteration : {unroll_k - 2 * k_per_inst}")
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(4)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))
            self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(3)')
            self._emit(mfma_step_mxn(1, 0, 0, 0))
            self._emit_empty_line()

            self._emit(f"; k iteration : {unroll_k - 1 * k_per_inst}")
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(1)')
            self._emit(mfma_step_mxn(0, 0, 1, 1))
            self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit_empty_line()


        def mfma_loop_repeat_2x1_lp2_with_interleave():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n

            def do_interleave_unroll_k_sub():
                with self._deferred_context():
                    unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                    for i_k in range(unroll_k_sub):
                        self._emit(f"; k iteration : {(2 * i_k + 0) * k_per_inst}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({1 if i_k == 0 else 2})')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                            if unroll_k_sub == 1:
                                self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst + lds_width_m // 2 ) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        elif i_k == unroll_k_sub - 1:
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+1) * lds_width_m * k_per_inst) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+1) * lds_width_m * k_per_inst + lds_width_m // 2 ) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                        else:
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+1) * lds_width_m * k_per_inst) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        else:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+1) * lds_width_m * k_per_inst + lds_width_m // 2 ) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        self._emit(f"; k iteration : {(2 * i_k + 1) * k_per_inst}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst + lds_width_m // 2)+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+3) * lds_width_m * k_per_inst) + f" ; load i_k:{(2*i_k+3)} into local buffer {1}, repeat {0}")

                        else:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 4})')
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        if i_k == unroll_k_sub - 1:
                            # v_b attension!
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+3) * lds_width_m * k_per_inst + lds_width_m // 2) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}")
                        else:
                            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst + lds_width_m // 2)+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()
                return self._get_deferred()

            def do_interleave_gload_and_move_slice_window():
                with self._deferred_context():
                    self._emit(f_gld_b())
                    self._emit(f_gld_a())
                    self._emit(f_move_slice_window_b())
                    self._emit(f_move_slice_window_a())
                return self._get_deferred()

            def do_interleave_unroll_k_last():
                with self._deferred_context():
                    self._emit(f"; k iteration : {unroll_k - 2 * k_per_inst}")
                    # 1st fma
                    #self._emit(f's_waitcnt lgkmcnt(4)')
                    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    self._emit_empty_line()

                    # 2nd fma
                    #self._emit(f's_waitcnt lgkmcnt(0)')
                    self._emit(f"s_barrier")
                    # self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
                    self._emit(mfma_step_mxn(1, 0, 0, 0))
                    self._emit_empty_line()

                    # 1st fma
                    #self._emit(f"s_waitcnt vmcnt(0)")
                    self._emit(f"; k iteration : {unroll_k - 1 * k_per_inst}")
                    self._emit(mfma_step_mxn(0, 0, 1, 1))

                    self._emit_empty_line()
                    #       iteration--
                    self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                    self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                    self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

                    # 2nd fma
                    self._emit(mfma_step_mxn(1, 0, 1, 1))
                    self._emit(f"s_waitcnt lgkmcnt(0)")
                    self._emit(f"s_barrier")
                    self._emit(f"s_branch {label_mfma_body}")
                return self._get_deferred()

            def do_interleave_share_store():
                with self._deferred_context():
                    self._emit(f's_waitcnt lgkmcnt(0)')
                    self._emit(f"s_barrier")
                    self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
                    self._emit(f_sst_b())
                    self._emit(f"s_waitcnt vmcnt(0)")
                    self._emit(f_sst_a())
                return self._get_deferred()

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_width_m // 2 ))

            mbb_list_sub = [create_machine_basic_block(do_interleave_unroll_k_sub(), group_mbb_by_end_of_inst_op="v_mfma"),
                            create_machine_basic_block(do_interleave_gload_and_move_slice_window())]

            se_sub = create_scheduler(self.mc, mbb_list_sub)

            mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                             create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

            se_last = create_scheduler(self.mc, mbb_list_last)
            self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))
            mbb_0_mfma_cnt_after_branch_to_start = 2 * cxm.wave_step_m * cxm.wave_step_n - 1 # number of mfma not count into share store interleave slot, check do_interleave_unroll_k_last for last 2 mfma
            self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1, mbb_0_mfma_cnt_after_branch_to_start=mbb_0_mfma_cnt_after_branch_to_start))

            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit_empty_line()

            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + lds_width_m // 2 ))
            self._emit(do_interleave_unroll_k_sub())
            self._emit(f"; k iteration : {unroll_k - 2 * k_per_inst}")
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(4)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))
            self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(3)')
            self._emit(mfma_step_mxn(1, 0, 0, 0))
            self._emit_empty_line()

            self._emit(f"; k iteration : {unroll_k - 1 * k_per_inst}")
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(1)')
            self._emit(mfma_step_mxn(0, 0, 1, 1))
            self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit_empty_line()


        def mfma_loop_repeat_1x2_lp2():
            mfma = cxm.inst_mfma
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            self._emit_empty_line()
            self._emit(f_gld_b())                                           # global load
            self._emit(f_gld_a())                                           # global load
            self._emit_empty_line()

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_width_n // 2 ))

            def do_unroll_k_sub():
                unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                for i_k in range(unroll_k_sub):
                    self._emit(f"; k iteration : {(2 * i_k + 0) * k_per_inst}")
                    # 1st fma
                    self._emit(f's_waitcnt lgkmcnt({1 if i_k == 0 else 2})')
                    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    if i_k == 0:
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        if unroll_k_sub == 1:
                            self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst + lds_width_n // 2 ) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                    elif i_k == unroll_k_sub - 1:
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+1) * lds_width_n * k_per_inst) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+1) * lds_width_n * k_per_inst + lds_width_n // 2 ) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                    else:
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+1) * lds_width_n * k_per_inst) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                    self._emit_empty_line()

                    # 2nd fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                    self._emit(mfma_step_mxn(0, 1, 0, 0))
                    if i_k == unroll_k_sub - 1:
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    else:
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+1) * lds_width_n * k_per_inst + lds_width_n // 2 ) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    self._emit_empty_line()

                    self._emit(f"; k iteration : {(2 * i_k + 1) * k_per_inst}")
                    # 1st fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                    self._emit(mfma_step_mxn(0, 0, 1, 1))
                    if i_k == unroll_k_sub - 1:
                        self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst + lds_width_n // 2)+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst) + f" ; load i_k:{(2*i_k+3)} into local buffer {1}, repeat {0}")

                    else:
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    self._emit_empty_line()

                    # 2nd fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 4})')
                    self._emit(mfma_step_mxn(0, 1, 1, 1))
                    if i_k == unroll_k_sub - 1:
                        # v_b attension!
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+3) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst + lds_width_n // 2) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}")
                    else:
                        self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst + lds_width_n // 2)+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+3) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                    self._emit_empty_line()

            do_unroll_k_sub()

            self._emit(f"; k iteration : {unroll_k - 2 * k_per_inst}")
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(4)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())
            self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(f"s_barrier")
            self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
            self._emit(f_sst_b())
            self._emit(mfma_step_mxn(0, 1, 0, 0))
            self._emit_empty_line()

            # 1st fma
            self._emit(f"s_waitcnt vmcnt(0)")
            self._emit(f_sst_a())
            self._emit(f"; k iteration : {unroll_k - 1 * k_per_inst}")
            self._emit(mfma_step_mxn(0, 0, 1, 1))

            self._emit_empty_line()
            #       iteration--
            self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
            self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
            self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

            # 2nd fma
            self._emit(mfma_step_mxn(0, 1, 1, 1))
            self._emit_empty_line()
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            #       load next from global
            self._emit(f_gld_b())
            self._emit(f_gld_a())
            self._emit(f"s_branch {label_mfma_body}")
            self._emit_empty_line()

            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            self._emit(mfma_step_mxn(0, 1, 1, 1))
            self._emit_empty_line()

            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_width_n // 2 ))
            do_unroll_k_sub()
            self._emit(f"; k iteration : {unroll_k - 2 * k_per_inst}")
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(4)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))
            self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(3)')
            self._emit(mfma_step_mxn(0, 1, 0, 0))
            self._emit_empty_line()

            self._emit(f"; k iteration : {unroll_k - 1 * k_per_inst}")
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(1)')
            self._emit(mfma_step_mxn(0, 0, 1, 1))
            self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(0, 1, 1, 1))
            self._emit_empty_line()


        def mfma_loop_repeat_1x2_lp2_with_interleave():
            mfma = cxm.inst_mfma
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n
            def do_interleave_unroll_k_sub():
                with self._deferred_context():
                    unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                    for i_k in range(unroll_k_sub):
                        self._emit(f"; k iteration : {(2 * i_k + 0) * k_per_inst}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({1 if i_k == 0 else 2})')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + lds_width_m * k_per_inst) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                            if unroll_k_sub == 1:
                                self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_width_n * k_per_inst + lds_width_n // 2 ) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        elif i_k == unroll_k_sub - 1:
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+1) * lds_width_n * k_per_inst) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+1) * lds_width_n * k_per_inst + lds_width_n // 2 ) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                        else:
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+1) * lds_width_n * k_per_inst) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        else:
                            self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+1) * lds_width_n * k_per_inst + lds_width_n // 2 ) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + (2*i_k+2) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        self._emit(f"; k iteration : {(2 * i_k + 1) * k_per_inst}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst + lds_width_n // 2)+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst) + f" ; load i_k:{(2*i_k+3)} into local buffer {1}, repeat {0}")

                        else:
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 4})')
                        self._emit(mfma_step_mxn(0, 1, 1, 1))
                        if i_k == unroll_k_sub - 1:
                            # v_b attension!
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+3) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+3) * lds_width_n * k_per_inst + lds_width_n // 2) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}")
                        else:
                            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + (2*i_k+2) * lds_width_n * k_per_inst + lds_width_n // 2)+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + (2*i_k+3) * lds_width_m * k_per_inst) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()
                return self._get_deferred()

            def do_interleave_gload_and_move_slice_window():
                with self._deferred_context():
                    self._emit(f_gld_b())
                    self._emit(f_gld_a())
                    self._emit(f_move_slice_window_b())
                    self._emit(f_move_slice_window_a())
                return self._get_deferred()

            def do_interleave_unroll_k_last():
                with self._deferred_context():
                    self._emit(f"; k iteration : {unroll_k - 2 * k_per_inst}")
                    # 1st fma
                    self._emit(f's_waitcnt lgkmcnt(4)')
                    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    self._emit_empty_line()

                    # 2nd fma
                    self._emit(f"s_barrier")
                    self._emit(mfma_step_mxn(0, 1, 0, 0))
                    self._emit_empty_line()

                    # 1st fma
                    self._emit(f"; k iteration : {unroll_k - 1 * k_per_inst}")
                    self._emit(mfma_step_mxn(0, 0, 1, 1))

                    self._emit_empty_line()
                    #       iteration--
                    self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                    self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                    self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

                    # 2nd fma
                    self._emit(mfma_step_mxn(0, 1, 1, 1))
                    self._emit_empty_line()
                    self._emit(f"s_waitcnt lgkmcnt(0)")
                    self._emit(f"s_barrier")
                    self._emit(f"s_branch {label_mfma_body}")
                return self._get_deferred()

            def do_interleave_share_store():
                with self._deferred_context():
                    self._emit(f's_waitcnt lgkmcnt(0)')
                    self._emit(f"s_barrier")
                    self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
                    self._emit(f_sst_b())
                    self._emit(f"s_waitcnt vmcnt(0)")
                    self._emit(f_sst_a())
                return self._get_deferred()

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_width_n // 2 ))

            mbb_list_sub = [create_machine_basic_block(do_interleave_unroll_k_sub(), group_mbb_by_end_of_inst_op="v_mfma"),
                            create_machine_basic_block(do_interleave_gload_and_move_slice_window())]

            se_sub = create_scheduler(self.mc, mbb_list_sub)

            mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                             create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

            se_last = create_scheduler(self.mc, mbb_list_last)
            self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))
            mbb_0_mfma_cnt_after_branch_to_start = 2 * cxm.wave_step_m * cxm.wave_step_n - 1 # number of mfma not count into share store interleave slot, check do_interleave_unroll_k_last for last 2 mfma
            self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1, mbb_0_mfma_cnt_after_branch_to_start=mbb_0_mfma_cnt_after_branch_to_start))

            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            self._emit(mfma_step_mxn(0, 1, 1, 1))
            self._emit_empty_line()

            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + lds_width_n // 2 ))
            self._emit(do_interleave_unroll_k_sub())
            self._emit(f"; k iteration : {unroll_k - 2 * k_per_inst}")
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(4)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))
            self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(3)')
            self._emit(mfma_step_mxn(0, 1, 0, 0))
            self._emit_empty_line()

            self._emit(f"; k iteration : {unroll_k - 1 * k_per_inst}")
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(1)')
            self._emit(mfma_step_mxn(0, 0, 1, 1))
            self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(0, 1, 1, 1))
            self._emit_empty_line()


        # start emit
        self._emit(f"; start MFMA loop, {cxm.wave_tile_m}x{cxm.wave_tile_n} wave tile with {cxm.wave_repeat_m}x{cxm.wave_repeat_n} repeat, {cxm.wave_step_m}x{cxm.wave_step_n} step")
        self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")

        self._emit(f_sst_b())
        self._emit_empty_line()
        self._emit(f"s_waitcnt vmcnt(0)")
        self._emit(f_sst_a())
        self._emit_empty_line()

        self._emit(f".v_clear_acc_c {a_c()}, {cxm.total_acc_c()}")
        self._emit(f"; make sure acc WAR harzard, at least 1 nop for src_c")

        # decrese k
        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_knum()}], {unroll_k}")
        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
        self._emit(f"s_cbranch_scc0 {label_mfma_end}")
        self._emit_empty_line()

        # diverge based on repeat
        if cxm.wave_repeat_m == 2 and cxm.wave_repeat_n == 2:
            if self.ctrl.local_prefetch_num == 1:
                if self.ctrl.lds_buffer_num == 2:
                    if self.ctrl.interleave:
                        mfma_loop_repeat_2x2_lds_double_buffer_with_interleave()
                    else:
                        mfma_loop_repeat_2x2_double_buffer()
            elif self.ctrl.local_prefetch_num == 2:
                if self.ctrl.lds_buffer_num == 2:
                    if self.ctrl.interleave:
                        mfma_loop_repeat_2x2_lp2_double_buffer_with_interleave()
                    else:
                        mfma_loop_repeat_2x2_lp2_double_buffer()
                else:
                    if self.ctrl.interleave:
                        mfma_loop_repeat_2x2_lp2_with_interleave()
                    else:
                        assert False, "function have bugs to fix"
        elif cxm.wave_repeat_m == 1 and cxm.wave_repeat_n == 1:
            if self.ctrl.local_prefetch_num == 1:
                assert False
            elif self.ctrl.local_prefetch_num == 2:
                if self.ctrl.interleave:
                    if self.ctrl.lds_buffer_num == 2:
                        mfma_loop_repeat_1x1_lp2_double_buffer_with_interleave()
                    else:
                        mfma_loop_repeat_1x1_lp2_with_interleave()
                    #mfma_loop_repeat_1x1_lp2()
                else:
                    mfma_loop_repeat_1x1_lp2()
        elif cxm.wave_repeat_m == 2 and cxm.wave_repeat_n == 1:
            if self.ctrl.local_prefetch_num == 1:
                assert False
            elif self.ctrl.local_prefetch_num == 2:
                if self.ctrl.interleave:
                    mfma_loop_repeat_2x1_lp2_with_interleave()
                else:
                    mfma_loop_repeat_2x1_lp2()
        elif cxm.wave_repeat_m == 1 and cxm.wave_repeat_n == 2:
            if self.ctrl.local_prefetch_num == 1:
                assert False
            elif self.ctrl.local_prefetch_num == 2:
                if self.ctrl.interleave:
                    mfma_loop_repeat_1x2_lp2_with_interleave()
                else:
                    mfma_loop_repeat_1x2_lp2()
        else:
            assert False, f"un implemented repeat {cxm.wave_repeat_m}x{cxm.wave_repeat_n}"

        nop = emit_nop_t(self.mc)
        nop(cxm.inst_mfma.get_nop_count_mfma_acc_raw())   # solve dependency 
