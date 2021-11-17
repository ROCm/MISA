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
from .mfma import *
from .xdlops_mapping import *
from .nop import *
import re

MFMA_FEAT_SINGLE_PASS_THROUGH_EARLY_LAST_DS_WAIT = 1       # last wait for ds_read advance a mfma slot

class ctrl_mfma_main_loop_t(object):
    def __init__(self):
        self.cxm                         = None

        self.unroll_k                    = 0
        self.label_prefix                = ''                      # usually be kernel name of caller kernel
        self.precision                   = 'fp32'

        self.lds_single_size             = 0                    # in byte, should be power of 2
        self.lds_buffer_num              = 2
        self.local_prefetch_num          = 1
        self.interleave                  = False
        self.accvgpr_unified             = False        # if true, means using accvgpr unified mode, will use vgpr instead of agpr

        # functor
        self.global_load_a_functor       = None
        self.global_load_b_functor       = None
        self.shared_store_a_functor      = None
        self.shared_store_b_functor      = None
        self.shared_load_a_functor       = None
        self.shared_load_b_functor       = None
        self.move_slice_window_a_functor = None
        self.move_slice_window_b_functor = None
        self.move_slice_window_accumule_functor  = None

        # symbol type
        self.v_a                         = None
        self.v_b                         = None
        self.a_c                         = None
        self.v_gld_a                     = None
        self.v_gld_a_gpf                 = None     # used for a pass through and not interleaved, as global prefetch register
        self.v_gld_a_num                 = 1
        self.v_gld_b                     = None
        self.v_gld_b_gpf                 = None     # used for b pass through and not interleaved, as global prefetch register
        self.v_gld_b_num                 = 1
        self.v_sld_a_os                  = None
        self.v_sld_b_os                  = None
        self.v_sst_a_os                  = None
        self.v_sst_b_os                  = None
        self.s_kitr                      = None
        self.s_knum                      = None

        # below is in unit of pixel, not considered data bytes
        self.lds_k_pack                  = 1
        self.lds_pad_m                   = 0        # pad how many pixels per m row
        self.lds_pad_n                   = 0        # pad how many pixels per n row

        self.pass_through_a              = 0        # a tensor not using LDS
        self.pass_through_b              = 0        # b tensor not using LDS
        self.pass_through_a_v_pack       = 1        # passthough tensor may have v pack, indicate vector load
        self.pass_through_b_v_pack       = 1
        self.pass_through_a_interleave_gld         = 1
        self.pass_through_b_interleave_gld         = 1
        self.pass_through_bf16_1k_in_fp16          = False   # the pass through side is indeed bf16 1k
        self.pass_through_bf16_1k_in_fp16_predefine = None   # predefine symbol for .if....else
        self.opt_1st_sld                 = True    # optimize 1st ds_read

class mfma_main_loop_t(mc_base_t):
    '''
    '''
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        self.ctrl = ctrl
        assert type(ctrl) is ctrl_mfma_main_loop_t
    
    def emit_single_pass_through(self):
        '''
        one side of A/B tensor not using LDS, used for skinny gemm
        a/b -> p/q, where p side passthrough lds, q side is normal
        '''

        p_idx = 0 if self.ctrl.pass_through_a else 1
        q_idx = p_idx ^ 1
        ctrl = self.ctrl

        label_mfma_body = 'L_{}_mfma_body'.format(self.ctrl.label_prefix)
        label_mfma_finishing = 'L_{}_mfma_finishing'.format(self.ctrl.label_prefix)
        label_mfma_end = 'L_{}_mfma_end'.format(self.ctrl.label_prefix)

        f_gld_p = [ctrl.global_load_a_functor, ctrl.global_load_b_functor][p_idx]
        f_gld_q = [ctrl.global_load_a_functor, ctrl.global_load_b_functor][q_idx]
        f_sst_p = [ctrl.shared_store_a_functor, ctrl.shared_store_b_functor][p_idx]
        f_sst_q = [ctrl.shared_store_a_functor, ctrl.shared_store_b_functor][q_idx]

        f_sld_p = [ctrl.shared_load_a_functor, ctrl.shared_load_b_functor][p_idx]
        f_sld_q = [ctrl.shared_load_a_functor, ctrl.shared_load_b_functor][q_idx]

        f_move_slice_window_p    = [ctrl.move_slice_window_a_functor, ctrl.move_slice_window_b_functor][p_idx]
        f_move_slice_window_q    = [ctrl.move_slice_window_a_functor, ctrl.move_slice_window_b_functor][q_idx]
        f_move_slice_window_acc  = ctrl.move_slice_window_accumule_functor

        v_gld_p     = [ctrl.v_gld_a, ctrl.v_gld_b][p_idx]
        v_gld_q     = [ctrl.v_gld_a, ctrl.v_gld_b][q_idx]

        v_gld_p_gpf = [ctrl.v_gld_a_gpf, ctrl.v_gld_b_gpf][p_idx]
        v_gld_p_num = [ctrl.v_gld_a_num, ctrl.v_gld_b_num][p_idx]

        a_c         = ctrl.a_c
        v_q         = [ctrl.v_a, ctrl.v_b][q_idx]
        v_sld_q_os  = [ctrl.v_sld_a_os, ctrl.v_sld_b_os][q_idx]
        v_sst_q_os  = [ctrl.v_sst_a_os, ctrl.v_sst_b_os][q_idx]

        s_kitr = ctrl.s_kitr
        s_knum = ctrl.s_knum
        cxm = ctrl.cxm

        data_byte = amdgpu_precision_data_byte(ctrl.precision)

        lds_width_m = data_byte * cxm.wave_tile_m * cxm.wave_step_m * cxm.waves_per_m() * cxm.wave_repeat_m
        lds_width_n = data_byte * cxm.wave_tile_n * cxm.wave_step_n * cxm.waves_per_n() * cxm.wave_repeat_n
        lds_single_size = ctrl.lds_single_size

        lds_width_q = [lds_width_m, lds_width_n][q_idx]

        # used as offset:x number. may some 
        lds_base_m = 0
        lds_base_n = 0
        assert ctrl.unroll_k % cxm.block_k() == 0
        unroll_k = ctrl.unroll_k
        k_per_inst = cxm.block_k()

        pad_m = ctrl.lds_pad_m
        pad_n = ctrl.lds_pad_n

        lds_base_q = [lds_base_m, lds_base_n][q_idx]
        pad_q      = [pad_m, pad_n][q_idx]

        num_v_p         = [cxm.inst_mfma.num_v_a, cxm.inst_mfma.num_v_b][p_idx]
        num_v_q         = [cxm.inst_mfma.num_v_a, cxm.inst_mfma.num_v_b][q_idx]
        wave_step_p     = [cxm.wave_step_m, cxm.wave_step_n][p_idx]
        wave_step_q     = [cxm.wave_step_m, cxm.wave_step_n][q_idx]
        wave_repeat_p   = [cxm.wave_repeat_m,  cxm.wave_repeat_n][p_idx]
        wave_repeat_q   = [cxm.wave_repeat_m,  cxm.wave_repeat_n][q_idx]

        p_interleave_gld = [ctrl.pass_through_a_interleave_gld, ctrl.pass_through_b_interleave_gld][p_idx]

        # assert wave_repeat_q == 2, "currently the side need LDS must have repeat 2, following limitation seems have BUG"

        v_pack_p        = [ctrl.pass_through_a_v_pack, ctrl.pass_through_b_v_pack][p_idx]
        v_pack_q        = [ctrl.pass_through_a_v_pack, ctrl.pass_through_b_v_pack][q_idx]
        assert v_pack_p == v_pack_q, "currently only support p, q the same"

        v_pack_per_k    = 4 // data_byte        # how many pack along k dimension
        assert v_pack_per_k * num_v_p == cxm.lanegroup_k_per_thread(), "this should always holds!"
        
        v_pack_p_per_kpt  = v_pack_p // cxm.lanegroup_k_per_thread()  # how many p dimension for each kpt (k_per_thread)
        v_pack_q_per_kpt  = v_pack_q // cxm.lanegroup_k_per_thread()

        # assert v_pack_p % v_pack_per_k == 0
        # v_pack_p_per_k  = v_pack_p // v_pack_per_k  # how many p dimension for each k
        # v_pack_q_per_k  = v_pack_q // v_pack_per_k

        k_per_v_pack    = v_pack_p * cxm.block_k_per_wave()

        assert unroll_k % k_per_v_pack == 0, f'unroll_k:{unroll_k}, k_per_v_pack:{k_per_v_pack}'
        unroll_k_slot = unroll_k // k_per_v_pack

        def global_load_p():
            with self._deferred_context():
                self._emit(f_gld_p())
            return self._get_deferred()

        def global_load_q():
            with self._deferred_context():
                self._emit(f_gld_q())
            return self._get_deferred()

        def move_slice_window_pq():
            with self._deferred_context():
                if f_move_slice_window_p:
                    self._emit(f_move_slice_window_p())
                if f_move_slice_window_q:
                    self._emit(f_move_slice_window_q())
            return self._get_deferred()

        def move_slice_window_acc():
            with self._deferred_context():
                if f_move_slice_window_acc:
                    self._emit(f_move_slice_window_acc())
            return self._get_deferred()

        def call_mbb(mbb):
            return machine_basic_block_call(self, mbb)
        
        # parse global load of p tensor into list of single load
        mbb_gld_p = create_machine_basic_block(global_load_p())
        mbb_gld_q = create_machine_basic_block(global_load_q(), merge_mbb = 1)

        mbb_p_clear = 1 if  mbb_gld_p[0].mc_inst(-1).type() == MC_INST_TYPE_LEGACY_MACRO else 0
        mbb_q_clear = 1 if  mbb_gld_q[0].mc_inst(-1).type() == MC_INST_TYPE_LEGACY_MACRO else 0

        if mbb_p_clear == 1:
            # hack on v_clear_nc
            v_clear_nc_strs = mbb_gld_p[0].mc_inst(-1).inst_str
            v_clear_nc_list = re.split('[,\s]+', v_clear_nc_strs)
            assert len(v_clear_nc_list) == 3 and v_clear_nc_list[0] == '.v_clear_nc'
            num_gld_p = int(v_clear_nc_list[2]) # TODO: check number
            assert num_gld_p % (len(mbb_gld_p) - mbb_p_clear) == 0
            num_gld_p_per_issue = num_gld_p // (len(mbb_gld_p) - mbb_p_clear)
            def emit_v_clear_nc_p(i):
                with self._deferred_context():
                    self._emit(f".v_clear_nc {v_gld_p(i * num_gld_p_per_issue) if p_interleave_gld else v_gld_p_gpf(i * num_gld_p_per_issue)}, {num_gld_p_per_issue}")
                return self._get_deferred()

            mbb_gld_p_wrapper = list()
            for i in range(len(mbb_gld_p) - mbb_p_clear):
                mbb_gld_p_wrapper += create_machine_basic_block(emit_v_clear_nc_p(i) + '\n' + call_mbb(mbb_gld_p[i+1]), merge_mbb = 1)

            mbb_gld_p = mbb_gld_p_wrapper
            mbb_p_clear = 0

        num_p_issue = len(mbb_gld_p) - mbb_p_clear
        num_q_issue = len(mbb_gld_q) - mbb_q_clear

        mbb_msw_pq = create_machine_basic_block(move_slice_window_pq(), merge_mbb = 1) if (f_move_slice_window_p or f_move_slice_window_q) else list()
        mbb_msw_acc = create_machine_basic_block(move_slice_window_acc(), merge_mbb = 1) if f_move_slice_window_acc else list()

        def mapped_ioffset(i_k, width_byte, pad_pixel, offset = 0):
            k_pack = self.ctrl.lds_k_pack
            i_k0 = i_k // k_pack
            i_kp = i_k % k_pack
            return i_k0 * (width_byte * k_pack + pad_pixel * data_byte) + i_kp * data_byte + offset * k_pack
       
        def mi_q(i_k, offset = 0):
            return mapped_ioffset(i_k, lds_width_q, pad_q, offset)

        def mfma_step_pxq_vk(i_k, i_repeat_p, i_repeat_q, i_v, i_local_buffer_q = 0):
            # v_pack is in k direction, hence c_index stay the same across different i_v
            mfma = cxm.inst_mfma
            num_agpr_per_issue = mfma.num_a_c
            with self._deferred_context():
                for i_step_q in range(wave_step_q):
                    for i_step_p in range(wave_step_p):
                        if p_idx == 0:
                            c_index = i_repeat_p * wave_step_p * wave_step_q * wave_repeat_q * num_agpr_per_issue + \
                                            i_repeat_q * wave_step_p * wave_step_q * num_agpr_per_issue + \
                                            i_step_p * wave_step_q * num_agpr_per_issue + \
                                            i_step_q * num_agpr_per_issue
                        else:
                            c_index = i_repeat_q * wave_step_q * wave_step_p * wave_repeat_p * num_agpr_per_issue + \
                                            i_repeat_p * wave_step_q * wave_step_p * num_agpr_per_issue + \
                                            i_step_q * wave_step_p * num_agpr_per_issue + \
                                            i_step_p * num_agpr_per_issue
                        c_index_end = c_index + num_agpr_per_issue - 1

                        p_index = i_k * wave_repeat_p * wave_step_p * v_pack_p_per_kpt *  num_v_p + \
                                    i_repeat_p * wave_step_p * v_pack_p_per_kpt *  num_v_p + \
                                    i_step_p * v_pack_p_per_kpt *  num_v_p + \
                                    i_v * num_v_p

                        q_index = i_local_buffer_q * wave_step_q * wave_repeat_q * v_pack_q_per_kpt *  num_v_q + \
                                    i_repeat_q * wave_step_q * v_pack_q_per_kpt *  num_v_q + \
                                    i_step_q * v_pack_q_per_kpt *  num_v_q + \
                                    i_v * num_v_q
                        if num_v_p == 1 and num_v_q == 1:
                            self._emit(mfma(a_c((c_index, c_index_end)), v_gld_p(p_index), v_q(q_index), a_c((c_index, c_index_end))) + f"  ; repeat:{i_repeat_p}x{i_repeat_q}, step:{i_step_p}x{i_step_q}, k:{i_k}, v:{i_v}, num_a_c:{num_agpr_per_issue}")
                        else:
                            p_index_end = p_index + num_v_p - 1
                            q_index_end = q_index + num_v_q - 1
                            self._emit(mfma(a_c((c_index, c_index_end)), v_gld_p((p_index, p_index_end)), v_q((q_index, q_index_end)), a_c((c_index, c_index_end))) + f"  ; repeat:{i_repeat_p}x{i_repeat_q}, step:{i_step_p}x{i_step_q}, k:{i_k}, v:{i_v}, num_a_c:{num_agpr_per_issue}")
            return self._get_deferred()

        def mfma_loop():
            mfma = cxm.inst_mfma

            repeat_q_thread_offset = wave_step_q * num_v_q * v_pack_q_per_kpt
            local_buffer_q = wave_repeat_q * repeat_q_thread_offset
            mfma_v_pack_slot = unroll_k_slot * wave_repeat_p * wave_repeat_q    # TODO: not consider step
            cnt_mfma_v_pack_slot = 0

            def first_sld():
                # when start of mfma main loop, do this load
                with self._deferred_context():
                    for i in range(wave_repeat_q):
                        self._emit(f_sld_q(v_q(i * repeat_q_thread_offset), v_sld_q_os(), lds_base_q + mi_q(0, i * (lds_width_q // 2))))
                    if ctrl.local_prefetch_num == 2:
                        self._emit(f_sld_q(v_q(wave_step_q * wave_repeat_q * v_pack_q_per_kpt * num_v_q), v_sld_q_os(), lds_base_q + mi_q(1 * v_pack_q_per_kpt * k_per_inst, 0)))
                return self._get_deferred()

            mbb_first_sld = create_machine_basic_block(first_sld())

            def mfma_per_k_slot(i_k, i_mfma_v_pack_slot, is_last_fma):
                '''
                k slot is unroll_k / k_per_inst
                pattern:
                prefetch:1, repeat:1 (phase:1)
                                 0           0            0
                    i_k   i_r    load_i_r    load_i_buf   load_i_k  lgkmcnt     need_load
                    0     0      0           0            1         0
                    1     0      0           0            2         0
                    2     0      0           0            3         0
                    3     0      0           0            4         0           x

                prefetch:2, repeat:1 (phase:1)
                                 0           0            0
                                 0           1            1
                    i_k   i_r    load_i_r    load_i_buf   load_i_k  lgkmcnt     need_load
                    0     0      0           0            2         1
                    1     0      0           1            3         1
                    2     0      0           0            4         1           x
                    3     0      0           1            5         0           x

                prefetch:1, repeat:2 (phase:2)
                                 0           0            0
                                 1           0            0
                    i_k   i_r    load_i_r    load_i_buf   load_i_k  lgkmcnt     need_load
                    0     0      0           0            1         1
                    0     1      1           0            1         1
                    1     0      0           0            2         1
                    1     1      1           0            2         1
                    2     0      0           0            3         1
                    2     1      0           0            3         1
                    3     0      1           0            4         1           x
                    3     1      1           0            4         0           x

                prefetch:2, repeat:2 (phase:3)
                                 0           0            0
                                 1           0            0
                                 0           1            1
                    i_k   i_r    load_i_r    load_i_buf   load_i_k  lgkmcnt     need_load
                    0     0      1           1            1         2
                    0     1      0           0            2         2
                    1     0      1           0            2         2
                    1     1      0           1            3         2
                    2     0      1           1            3         2
                    2     1      0           0            4         2           x
                    3     0      1           0            4         1           x
                    3     1      0           1            5         0           x
                '''
                pref = ctrl.local_prefetch_num
                rept = wave_repeat_q
                phase = pref + rept - 1               # idx before entering main loop

                i_r_sequence = [ x & (rept - 1) for x in range(pref * rept)]
                i_b_sequence = [(x >> (rept - 1)) & (pref - 1) for x in range(pref * rept)]

                i_local_buffer_q = i_k & 1 if pref == 2 else 0
                i_k_sst_q = i_k == (unroll_k_slot - ctrl.local_prefetch_num)
                # print(f"i_k:{i_k}, i_k_sst_q:{i_k_sst_q}")
                gld_p_per_k = wave_repeat_p * wave_step_p
                cnt_mfma = 0
                def try_do_gld_per_slot(i_slot):
                    if is_last_fma:
                        if p_interleave_gld:
                            mbb_gld_p_per_k = mbb_gld_p[len(mbb_gld_p) - gld_p_per_k : ] if i_k == 0 else list()
                        else:
                            mbb_gld_p_per_k = list()
                        mbb_gld_per_k = mbb_gld_p_per_k
                    else:
                        if p_interleave_gld:
                            if i_k == 0:
                                mbb_gld_p_per_k = mbb_gld_p[len(mbb_gld_p) - gld_p_per_k : ]
                            else:
                                start_p_idx = mbb_p_clear if i_k == 1 else ((i_k - 1) * gld_p_per_k + mbb_p_clear)  # always no clear
                                mbb_gld_p_per_k = mbb_gld_p[start_p_idx : i_k * gld_p_per_k + mbb_p_clear  ]
                        else:
                            mbb_gld_p_per_k = mbb_gld_p if i_k == 0 else list()
                        mbb_gld_per_k = ((mbb_gld_p_per_k + mbb_msw_pq + mbb_msw_acc + mbb_gld_q) if p_interleave_gld else (mbb_gld_q + mbb_gld_p_per_k)) \
                            if i_k == 0 else mbb_gld_p_per_k
                    num_gld_slot_per_k = wave_repeat_p * wave_repeat_q * v_pack_p_per_kpt
                    num_gld_per_slot = utility_next_mul(len(mbb_gld_per_k), num_gld_slot_per_k) // num_gld_slot_per_k
                    for i_gld in range(num_gld_per_slot):
                        current_gld = i_slot * num_gld_per_slot + i_gld
                        if current_gld < len(mbb_gld_per_k):
                            self._emit(call_mbb(mbb_gld_per_k[current_gld]))

                def do_sst_q():
                    # print(f"do_sst_q, i_k:{i_k}")
                    if ctrl.lds_buffer_num == 1:
                        self._emit(f"s_barrier")
                    self._emit(f_sst_q())
                    if ctrl.lds_buffer_num != 1:
                        self._emit(f"v_xor_b32 v[{v_sst_q_os()}], {hex(lds_single_size)}, v[{v_sst_q_os()}]")

                def do_sld_q(i_v, i_r):
                    # interleave into different v_pack
                    i_idx = i_k * rept + i_r
                    i_idx_mod = (i_idx + phase) % (pref * rept)
                    i_idx_int = (i_idx + phase) // (pref * rept)

                    # print(f"  ==i_r_sequence:{i_r_sequence}, i_b_sequence:{i_b_sequence}, i_idx:{i_idx}, mod:{i_idx_mod}, int:{i_idx_int}")

                    load_i_r = i_r_sequence[i_idx_mod]
                    load_i_b = i_b_sequence[i_idx_mod]
                    load_i_k = i_idx_int * pref + load_i_b

                    #if i_v == (v_pack_p - 1) and load_i_k < unroll_k_slot:
                    if (i_v == (v_pack_q_per_kpt - 1) and load_i_k < unroll_k_slot):
                        the_str = f' ; i_r:{load_i_r}, i_b:{load_i_b}, i_k:{load_i_k}'
                        self._emit(f_sld_q(v_q(load_i_b * local_buffer_q + load_i_r * repeat_q_thread_offset),
                            v_sld_q_os(), lds_base_q + mi_q(load_i_k * v_pack_q_per_kpt * k_per_inst, load_i_r * (lds_width_q // 2) )) + the_str)


                if i_k == 0:
                    if not p_interleave_gld and not is_last_fma:
                        self._emit(move_slice_window_pq())
                    for mbb_1st in mbb_first_sld[1:]:
                        self._emit(call_mbb(mbb_1st))
                    if not p_interleave_gld and not is_last_fma:
                        self._emit(move_slice_window_acc())

                for i_rp in range(wave_repeat_p):
                    # cnt_p_load = cnt_p_load + 1
                    for i_rq in range(wave_repeat_q):
                        num_lgkmcnt = (pref + rept - 2) - ((pref - 1 + i_rq) if i_k == (unroll_k_slot-1) else 0)
                        if not p_interleave_gld:
                            vmcnt_str =  "vmcnt(0)" if i_k == 0 and i_rp == 0 and i_rq == 0 else \
                                    ( f"vmcnt({f_gld_p.get_issues()})" if num_lgkmcnt == 0 and  not is_last_fma else "")
                        else:
                            if i_rq != 0 and wave_repeat_q != 1:
                                vmcnt_str = ""
                            else:
                                if i_k == 0:
                                    vmcnt_str = f'vmcnt({num_p_issue - 1 - gld_p_per_k})'
                                else:
                                    if not is_last_fma:
                                        vmcnt_str = f'vmcnt({num_p_issue + num_q_issue - 2})'
                                    else:
                                        vmcnt_str = f'vmcnt({num_p_issue - i_k - 1})'

                        if MFMA_FEAT_SINGLE_PASS_THROUGH_EARLY_LAST_DS_WAIT and num_lgkmcnt == 0 and p_interleave_gld:
                            # we need a chance to put last lgkmcnt earlier
                            # assert vmcnt_str == ""
                            if is_last_fma:
                                self._emit(f's_waitcnt lgkmcnt(0)   ; vmcnt_str:{vmcnt_str}')
                            else:
                                # self._emit(f"; __ vmcnt_str:{vmcnt_str}")
                                pass
                        else:
                            self._emit(f's_waitcnt lgkmcnt({num_lgkmcnt}) {vmcnt_str}')
                            if num_lgkmcnt == 0 and not p_interleave_gld and not is_last_fma:
                                # self._emit(move_slice_window_acc())
                                do_sst_q()
                        if i_k == 0 and i_rp == 0 and i_rq == 0:
                            if not p_interleave_gld and v_gld_p_gpf:
                                # move buffer
                                for i_pnum in range(v_gld_p_num):
                                    if ctrl.pass_through_bf16_1k_in_fp16:
                                        self._emit(f".if {ctrl.pass_through_bf16_1k_in_fp16_predefine} == 1")
                                        self._emit(f"v_cvt_f32_f16 v[{v_gld_p(i_pnum)}], v[{v_gld_p_gpf(i_pnum)}]")
                                        self._emit(f"v_cvt_f32_f16 v[{v_gld_p_gpf(i_pnum)}], v[{v_gld_p_gpf(i_pnum)}] src0_sel:WORD_1")
                                        self._emit(f"v_pack_b32_f16 v[{v_gld_p(i_pnum)}], v[{v_gld_p(i_pnum)}], v[{v_gld_p_gpf(i_pnum)}] op_sel:[1,1]")
                                        self._emit(f".else")
                                        self._emit(f"v_mov_b32 v[{v_gld_p(i_pnum)}], v[{v_gld_p_gpf(i_pnum)}]")
                                        self._emit(f".endif")
                                    else:
                                        self._emit(f"v_mov_b32 v[{v_gld_p(i_pnum)}], v[{v_gld_p_gpf(i_pnum)}]")

                        for i_v in range(v_pack_p_per_kpt):
                            self._emit(mfma_step_pxq_vk(i_k, i_rp, i_rq, i_v, i_local_buffer_q))
                            if MFMA_FEAT_SINGLE_PASS_THROUGH_EARLY_LAST_DS_WAIT and p_interleave_gld:
                                if (i_mfma_v_pack_slot == mfma_v_pack_slot - 2) and (v_pack_p_per_kpt == 1 or i_v == (v_pack_p_per_kpt // 2) - 1):
                                    assert i_rq == 0
                                    if not is_last_fma:
                                        self._emit(f's_waitcnt lgkmcnt(0) vmcnt({num_p_issue - gld_p_per_k})')
                                        do_sst_q()
                            do_sld_q(i_v, i_rq) # will not emit when last ds wait, hence will never co-exist when last ds wait emit
                            #if not is_last_fma:
                            try_do_gld_per_slot(cnt_mfma)
                            cnt_mfma = cnt_mfma + 1
                        assert i_mfma_v_pack_slot < mfma_v_pack_slot, f'i_mfma_v_pack_slot:{i_mfma_v_pack_slot}, mfma_v_pack_slot:{mfma_v_pack_slot}'
                        i_mfma_v_pack_slot = i_mfma_v_pack_slot + 1

                if not is_last_fma and i_k == (unroll_k_slot - 1):
                    self._emit(f's_waitcnt lgkmcnt(0)')
                    self._emit(f"s_barrier")
                    self._emit(call_mbb(mbb_first_sld[0]))
                    self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                    self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                    self._emit(f"s_cbranch_scc1 {label_mfma_body}")
                return i_mfma_v_pack_slot

            self._emit(call_mbb(mbb_first_sld[0]))
            self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_knum()}], {unroll_k}")
            self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
            self._emit(f"s_cbranch_scc0 {label_mfma_end}")
            self._emit_empty_line()

            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}, mfma_v_pack_slot:{mfma_v_pack_slot}")

            for i_k in range(unroll_k_slot):
                cnt_mfma_v_pack_slot = mfma_per_k_slot(i_k, cnt_mfma_v_pack_slot, False)

            self._emit_front(f"{label_mfma_end}:")
            cnt_mfma_v_pack_slot = 0
            for i_k in range(unroll_k_slot):
                cnt_mfma_v_pack_slot = mfma_per_k_slot(i_k, cnt_mfma_v_pack_slot, True)

        # start emit, first load q tensor, then p tensor.
        self._emit(f"; start MFMA loop, wave tile:{cxm.wave_tile_m}x{cxm.wave_tile_n}, repeat:{cxm.wave_repeat_m}x{cxm.wave_repeat_n}, step:{cxm.wave_step_m}x{cxm.wave_step_n}" +\
                f", k_pack:{self.ctrl.lds_k_pack}, p_issue:{num_p_issue}, q_issue:{num_q_issue}, local_prefetch_num:{ctrl.local_prefetch_num}")

        if self.ctrl.accvgpr_unified:
            self._emit(f".v_clear_nc {a_c()}, {cxm.total_acc_c()}")
            set_ctrl_xdlops_mapping_accvgpr_unified(True)
        else:
            self._emit(f".v_clear_acc_c {a_c()}, {cxm.total_acc_c()}")
        # self._emit(f"; make sure acc WAR harzard, at least 1 nop for src_c")

        self._emit(f"s_waitcnt vmcnt({f_gld_p.get_issues() - ((wave_repeat_p * wave_step_p) if p_interleave_gld else 0)})")
        self._emit(f_sst_q())
        self._emit_empty_line()
        # if not p_interleave_gld:
        #     self._emit(move_slice_window_pq())
        #     self._emit(move_slice_window_acc())

        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit(f"s_barrier")
        self._emit_empty_line()

        mfma_loop()

        nop = emit_nop_t(self.mc)
        nop(cxm.inst_mfma.get_nop_count_mfma_acc_raw())   # solve dependency 


    def emit(self, **option):
        def get_dict_with_default(dictionary, key, default_value):
                if key in dictionary:
                    return dictionary[key]
                else:
                    return default_value
            
        num_gld_a_per_mbb = get_dict_with_default(option, "num_gld_a_per_mbb", 1)
        num_gld_b_per_mbb = get_dict_with_default(option, "num_gld_b_per_mbb", 1)

        if self.ctrl.pass_through_a ^ self.ctrl.pass_through_b:
            return self.emit_single_pass_through()

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
        f_move_slice_window_acc  = self.ctrl.move_slice_window_accumule_functor

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

        pad_m = self.ctrl.lds_pad_m
        pad_n = self.ctrl.lds_pad_n

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
                        # self._emit(mfma(a_c((c_index, c_index_end)), v_a(a_index), v_b(b_index), a_c((c_index, c_index_end))) + f"  ; repeat:{i_repeat_m}x{i_repeat_n}, step:{i_step_m}x{i_step_n}, num_a_c:{num_agpr_per_issue}")
                        if mfma.num_v_a == 1 and mfma.num_v_b == 1:
                            self._emit(mfma(a_c((c_index, c_index_end)), v_a(a_index), v_b(b_index), a_c((c_index, c_index_end))) + f"  ; repeat:{i_repeat_m}x{i_repeat_n}, step:{i_step_m}x{i_step_n}, num_a_c:{num_agpr_per_issue}")
                        elif mfma.num_v_a != 1 and mfma.num_v_b != 1:
                            a_index_end = a_index + mfma.num_v_a - 1
                            b_index_end = b_index + mfma.num_v_b - 1
                            self._emit(mfma(a_c((c_index, c_index_end)), v_a((a_index, a_index_end)), v_b((b_index, b_index_end)), a_c((c_index, c_index_end))) + f"  ; repeat:{i_repeat_m}x{i_repeat_n}, step:{i_step_m}x{i_step_n}, num_a_c:{num_agpr_per_issue}")

            return self._get_deferred()

        def mfma_loop_repeat_1x1_lp2():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())

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
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)))   # lds_width_m * k_per_inst
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)))   # lds_width_n * k_per_inst

            def do_unroll_k_1x1_sub():
                unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                for i_k in range(unroll_k_sub):
                    self._emit(f's_waitcnt lgkmcnt(2)')
                    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m  + mi_m((2*i_k+2) * k_per_inst)))    # (2*i_k+2) * lds_width_m * k_per_inst
                    self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n  + mi_n((2*i_k+2) * k_per_inst)))    #  (2*i_k+2) * lds_width_n * k_per_inst

                    self._emit(f's_waitcnt lgkmcnt(2)')
                    self._emit(mfma_step_mxn(0, 0, 1, 1))
                    self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst)))    # (2*i_k+3) * lds_width_m * k_per_inst
                    self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst)))    # (2*i_k+3) * lds_width_n * k_per_inst

            do_unroll_k_1x1_sub()
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(mfma_step_mxn(0, 0, 0, 0))

            self._emit_empty_line()

            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())
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
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)))   # k_per_inst * lds_width_m
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)))   # k_per_inst * lds_width_n
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
                    #print(f"unroll_k_sub={unroll_k_sub}")
                    for i_k in range(unroll_k_sub):
                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m  + mi_m((2*i_k+2) * k_per_inst)))    # (2*i_k+2) * k_per_inst *  lds_width_m
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n  + mi_n((2*i_k+2) * k_per_inst)))    # (2*i_k+2) * k_per_inst * lds_width_n

                        self._emit(f's_waitcnt lgkmcnt(2)')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m  + mi_m((2*i_k+3) * k_per_inst)))  # (2*i_k+3) * k_per_inst * lds_width_m
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n  + mi_n((2*i_k+3) * k_per_inst)))  # (2*i_k+3) * k_per_inst * lds_width_n
                return self._get_deferred()
           
            def get_interleave_gload_and_move_slice_window_mbbs():
                dup_inst_per_mbb_gld_b = f"buffer_load,{num_gld_b_per_mbb}" if num_gld_b_per_mbb != 1 else "off"
                dup_inst_per_mbb_gld_a = f"buffer_load,{num_gld_a_per_mbb}" if num_gld_a_per_mbb != 1 else "off"
                mbbs_gld_b = create_machine_basic_block(f_gld_b(), dup_inst_per_mbb=dup_inst_per_mbb_gld_b)
                mbbs_gld_a = create_machine_basic_block(f_gld_a(), dup_inst_per_mbb=dup_inst_per_mbb_gld_a)
                mbbs_msw_b = create_machine_basic_block(f_move_slice_window_b())
                mbbs_msw_a = create_machine_basic_block(f_move_slice_window_a())
                return mbbs_gld_b + mbbs_gld_a + mbbs_msw_b + mbbs_msw_a

            def do_interleave_unroll_k_last():
                with self._deferred_context():
                    # self._emit(f's_waitcnt lgkmcnt(0)')
                    self._emit(mfma_step_mxn(0, 0, 0, 0))

                    self._emit_empty_line()

                    self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                    self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                    self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

                    self._emit(mfma_step_mxn(0, 0, 1, 1))
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
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())

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
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst) ))   # k_per_inst * lds_width_m
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst) ))   # k_per_inst * lds_width_n


            if (unroll_k // k_per_inst) // 2 - 1 != 0:
                mbb_list_sub = [create_machine_basic_block(do_interleave_unroll_k_sub(), group_mbb_by_end_of_inst_op="v_mfma"),
                                get_interleave_gload_and_move_slice_window_mbbs()]
                se_sub = create_scheduler(self.mc, mbb_list_sub)
                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                                create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)

                self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))
                if f_move_slice_window_acc != None:
                    self._emit(f_move_slice_window_acc())
                self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1))
            else:
                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                                create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)
                self._emit(emit_machine_basic_blocks(self.mc, get_interleave_gload_and_move_slice_window_mbbs()))
                if f_move_slice_window_acc != None:
                    self._emit(f_move_slice_window_acc())
                self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1))

            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            self._emit(mfma_step_mxn(0, 0, 1, 1))

            
            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)))    # k_per_inst * lds_width_m
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)))    # k_per_inst * lds_width_n
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
                    unroll_k_sub = (unroll_k // k_per_inst) // 2
                    for i_k in range(unroll_k_sub):
                        if i_k < unroll_k_sub - 1:
                            self._emit(f's_waitcnt lgkmcnt(2)')
                            self._emit(mfma_step_mxn(0, 0, 0, 0))
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m  + mi_m((2*i_k+2) * k_per_inst)))
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n  + mi_n((2*i_k+2) * k_per_inst)))

                            self._emit(f's_waitcnt lgkmcnt(2)')
                            self._emit(mfma_step_mxn(0, 0, 1, 1))
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m  + mi_m((2*i_k+3) * k_per_inst)))
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n  + mi_n((2*i_k+3) * k_per_inst)))
                        else:
                            self._emit(f's_waitcnt lgkmcnt(2)')
                            self._emit(mfma_step_mxn(0, 0, 0, 0))
                            self._emit(f's_waitcnt lgkmcnt(0)')
                            self._emit(mfma_step_mxn(0, 0, 1, 1))

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
                    # self._emit(mfma_step_mxn(0, 0, 0, 0))
                    # self._emit(mfma_step_mxn(0, 0, 1, 1))

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
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)))
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)))


            mbb_list_sub = [create_machine_basic_block(do_interleave_unroll_k_sub(), group_mbb_by_end_of_inst_op="v_mfma"),
                            create_machine_basic_block(do_interleave_gload_and_move_slice_window())]
            se_sub = create_scheduler(self.mc, mbb_list_sub)

            self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))
            self._emit(f"; k iteration : {unroll_k - 2 * k_per_inst}")
            self._emit(f's_waitcnt lgkmcnt(0)')
            self._emit(do_interleave_share_store())
            self._emit(do_interleave_unroll_k_last())

            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            #self._emit(mfma_step_mxn(0, 0, 1, 1))

            
            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)))
            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)))
            self._emit(do_interleave_unroll_k_sub())


        def mfma_loop_repeat_2x2_lp2():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())

            #self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {hex(lds_single_size)}, v[{v_sst_b_os()}] ; switch double buffer b store")
            #self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {hex(lds_single_size)}, v[{v_sst_a_os()}] ; switch double buffer a store")
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            self._emit_empty_line()
            self._emit(f_gld_b())                                           # global load
            self._emit(f_gld_a())                                           # global load
            if self.ctrl.opt_1st_sld:
                self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
                self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
                self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
                self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2
            self._emit_empty_line()

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            if not self.ctrl.opt_1st_sld:
                self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
                self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
                self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
                self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2

            def do_unroll_k_sub():
                unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                for i_k in range(unroll_k_sub):
                    self._emit(f"; k iteration : {2 * i_k + 0}")
                    # 1st fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k == 0 else 5})')
                    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    if i_k == 0:
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")   # lds_width_m * k_per_inst
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")   # lds_width_n * k_per_inst
                    else:
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst, lds_width_m // 2)) + f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}") # (2*i_k+1) * lds_width_m * k_per_inst + lds_width_m // 2
                    self._emit_empty_line()

                    # 2nd fma
                    self._emit(f's_waitcnt lgkmcnt({3 if i_k == 0 else 5})')
                    self._emit(mfma_step_mxn(0, 1, 0, 0))
                    if i_k == 0:
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(k_per_inst,  lds_width_n // 2) ) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")   # lds_width_n * k_per_inst + lds_width_n // 2
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(k_per_inst,  lds_width_m // 2) ) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")   # lds_width_m * k_per_inst + lds_width_m // 2
                    else:
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")    # (2*i_k+2) * lds_width_m * k_per_inst
                    self._emit_empty_line()

                    # 3rd fma
                    self._emit(f's_waitcnt lgkmcnt({4 if i_k == 0 else 5})')
                    self._emit(mfma_step_mxn(1, 0, 0, 0))
                    if i_k == 0:
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")    # (2*i_k+2) * lds_width_m * k_per_inst
                    self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")    # (2*i_k+2) * lds_width_n * k_per_inst
                    self._emit_empty_line()

                    # 4th fma
                    self._emit(mfma_step_mxn(1, 1, 0, 0))
                    self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst, lds_width_n // 2)) + \
                                                f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}") # (2*i_k+2) * lds_width_n * k_per_inst + lds_width_n // 2
                    self._emit_empty_line()

                    self._emit(f"; k iteration : {2 * i_k + 1}")
                    # 1st fma
                    self._emit(f's_waitcnt lgkmcnt(5)')
                    self._emit(mfma_step_mxn(0, 0, 1, 1))
                    self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst, lds_width_m // 2) )+ \
                                                f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}") # (2*i_k+2) * lds_width_m * k_per_inst + lds_width_m // 2
                    self._emit_empty_line()

                    # 2nd fma
                    self._emit(f's_waitcnt lgkmcnt(5)')
                    self._emit(mfma_step_mxn(0, 1, 1, 1))
                    self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst) ) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")  # (2*i_k+3) * lds_width_m * k_per_inst
                    self._emit_empty_line()

                    # 3rd fma
                    self._emit(f's_waitcnt lgkmcnt(5)')
                    self._emit(mfma_step_mxn(1, 0, 1, 1))
                    self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")  # (2*i_k+3) * lds_width_n * k_per_inst
                    self._emit_empty_line()

                    # 4th fma
                    self._emit(mfma_step_mxn(1, 1, 1, 1))
                    self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst, lds_width_n//2)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}") # (2*i_k+3) * lds_width_n * k_per_inst + lds_width_n//2
                    if i_k == unroll_k_sub - 1:
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((unroll_k // k_per_inst - 1) * k_per_inst, lds_width_m // 2)) + f" ; load i_k:{unroll_k // k_per_inst - 1} into local buffer {1}, repeat {1}") # (unroll_k // k_per_inst - 1) * lds_width_m * k_per_inst + lds_width_m // 2
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
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())
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
            if self.ctrl.opt_1st_sld:
                self._emit(f"s_waitcnt lgkmcnt(0)")
                self._emit(f"s_barrier")
                self._emit(f_gld_b())
                self._emit(f_gld_a())
                self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
                self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
                self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
                self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2
            # 3rd fma
            self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit_empty_line()

            # 4th fma
            self._emit(mfma_step_mxn(1, 1, 1, 1))
            if not self.ctrl.opt_1st_sld:
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
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2
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
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")   #  lds_width_m * k_per_inst
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")   #   lds_width_n * k_per_inst
                        else:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst, lds_width_m // 2)) + f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")  # (2*i_k+1) * lds_width_m * k_per_inst + lds_width_m // 2
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({3 if i_k == 0 else 5})')
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(k_per_inst ,lds_width_n // 2) ) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")   # lds_width_n * k_per_inst + lds_width_n // 2
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(k_per_inst , lds_width_m // 2) ) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")   # lds_width_m * k_per_inst + lds_width_m // 2
                        else:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")    # (2*i_k+2) * lds_width_m * k_per_inst
                        self._emit_empty_line()

                        # 3rd fma
                        self._emit(f's_waitcnt lgkmcnt({4 if i_k == 0 else 5})')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")    # (2*i_k+2) * lds_width_m * k_per_inst
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}") # (2*i_k+2) * lds_width_n * k_per_inst
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 0, 0))
                        self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst, lds_width_n // 2)) + \
                                                    f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}") # (2*i_k+2) * lds_width_n * k_per_inst + lds_width_n // 2
                        self._emit_empty_line()

                        self._emit(f"; k iteration : {2 * i_k + 1}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst, lds_width_m // 2))+ \
                                                    f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}") #  (2*i_k+2) * lds_width_m * k_per_inst + lds_width_m // 2
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(0, 1, 1, 1))
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}") # (2*i_k+3) * lds_width_m * k_per_inst
                        self._emit_empty_line()

                        # 3rd fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 1, 1))
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst, lds_width_n//2)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}") # (2*i_k+3) * lds_width_n * k_per_inst + lds_width_n//2
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((unroll_k // k_per_inst - 1) * k_per_inst, lds_width_m//2)) + f" ; load i_k:{unroll_k // k_per_inst - 1} into local buffer {1}, repeat {1}")
                        self._emit_empty_line()
                    if unroll_k_sub == 0:
                        self._emit(f"; k iteration : {0}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({2})')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()
                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({3})')
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(k_per_inst, lds_width_n // 2)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(k_per_inst, lds_width_m // 2)) + \
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

            def get_interleave_gload_and_move_slice_window_mbbs():
                dup_inst_per_mbb_gld_b = f"buffer_load,{num_gld_b_per_mbb}" if num_gld_b_per_mbb != 1 else "off"
                dup_inst_per_mbb_gld_a = f"buffer_load,{num_gld_a_per_mbb}" if num_gld_a_per_mbb != 1 else "off"
                mbbs_gld_b = create_machine_basic_block(f_gld_b(), dup_inst_per_mbb=dup_inst_per_mbb_gld_b)
                mbbs_gld_a = create_machine_basic_block(f_gld_a(), dup_inst_per_mbb=dup_inst_per_mbb_gld_a)
                mbbs_msw_b = create_machine_basic_block(f_move_slice_window_b())
                mbbs_msw_a = create_machine_basic_block(f_move_slice_window_a())
                return mbbs_gld_b + mbbs_gld_a + mbbs_msw_b + mbbs_msw_a

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
                        #       iteration--
                        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                        self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")
                        if self.ctrl.opt_1st_sld:
                            self._emit(f"s_waitcnt lgkmcnt(0)")
                            self._emit(f"s_barrier")
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))

                        # 3rd fma
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        if self.ctrl.opt_1st_sld:
                            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
                            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 1, 1))
                        if not self.ctrl.opt_1st_sld:
                            self._emit(f"s_waitcnt lgkmcnt(0)")
                            self._emit(f"s_barrier")
                        self._emit(f"s_branch {label_mfma_body}")
                    else:
                        #       iteration--
                        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                        self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")
                        if self.ctrl.opt_1st_sld:
                            self._emit(f"s_waitcnt lgkmcnt(0)")
                            self._emit(f"s_barrier")
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
                            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
                            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2

                        # 3rd fma
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 1, 1))
                        if not self.ctrl.opt_1st_sld:
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
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            if self.ctrl.opt_1st_sld:
                self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
                self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
                self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
                self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            if not self.ctrl.opt_1st_sld:
                self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
                self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
                self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
                self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2

            unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
            #if (unroll_k // k_per_inst) // 2 - 1 != 0:
            if True:
                mbb_list_sub = [create_machine_basic_block(do_interleave_unroll_k_sub(), group_mbb_by_end_of_inst_op="v_mfma"),
                                get_interleave_gload_and_move_slice_window_mbbs()]
                se_sub = create_scheduler(self.mc, mbb_list_sub)

                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                                create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)
                self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))
                if f_move_slice_window_acc != None:
                    self._emit(f_move_slice_window_acc())
                mbb_0_mfma_cnt_after_branch_to_start = 2 * cxm.wave_step_m * cxm.wave_step_n - 1 # number of mfma not count into share store interleave slot, check do_interleave_unroll_k_last for last 2 mfma
                self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1, mbb_0_mfma_cnt_after_branch_to_start=mbb_0_mfma_cnt_after_branch_to_start))
            else:
                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                                create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)
                self._emit(emit_machine_basic_blocks(self.mc, get_interleave_gload_and_move_slice_window_mbbs()))
                if f_move_slice_window_acc != None:
                    self._emit(f_move_slice_window_acc())
                mbb_0_mfma_cnt_after_branch_to_start = 2 * cxm.wave_step_m * cxm.wave_step_n - 1 # number of mfma not count into share store interleave slot, check do_interleave_unroll_k_last for last 2 mfma
                self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1, mbb_0_mfma_cnt_after_branch_to_start=mbb_0_mfma_cnt_after_branch_to_start))

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
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2
            self._emit(do_interleave_unroll_k_sub())

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

        def mfma_loop_repeat_2x2_step_2x2_lp2_with_interleave():
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
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")   #  lds_width_m * k_per_inst
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")   #   lds_width_n * k_per_inst
                        else:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst, lds_width_m // 2)) + f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")  # (2*i_k+1) * lds_width_m * k_per_inst + lds_width_m // 2
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({3 if i_k == 0 else 5})')
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(k_per_inst ,lds_width_n // 2) ) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")   # lds_width_n * k_per_inst + lds_width_n // 2
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(k_per_inst , lds_width_m // 2) ) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")   # lds_width_m * k_per_inst + lds_width_m // 2
                        else:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")    # (2*i_k+2) * lds_width_m * k_per_inst
                        self._emit_empty_line()

                        # 3rd fma
                        self._emit(f's_waitcnt lgkmcnt({4 if i_k == 0 else 5})')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")    # (2*i_k+2) * lds_width_m * k_per_inst
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}") # (2*i_k+2) * lds_width_n * k_per_inst
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 0, 0))
                        self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst, lds_width_n // 2)) + \
                                                    f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}") # (2*i_k+2) * lds_width_n * k_per_inst + lds_width_n // 2
                        self._emit_empty_line()

                        self._emit(f"; k iteration : {2 * i_k + 1}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst, lds_width_m // 2))+ \
                                                    f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}") #  (2*i_k+2) * lds_width_m * k_per_inst + lds_width_m // 2
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(0, 1, 1, 1))
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}") # (2*i_k+3) * lds_width_m * k_per_inst
                        self._emit_empty_line()

                        # 3rd fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 1, 1))
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst, lds_width_n//2)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}") # (2*i_k+3) * lds_width_n * k_per_inst + lds_width_n//2
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((unroll_k // k_per_inst - 1) * k_per_inst, lds_width_m//2)) + f" ; load i_k:{unroll_k // k_per_inst - 1} into local buffer {1}, repeat {1}")
                        
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

                    if unroll_k_sub == 0:
                        self._emit(f"; k iteration : {0}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({2})')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()
                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({3})')
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(k_per_inst, lds_width_n // 2)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(k_per_inst, lds_width_m // 2)) + \
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

            def get_interleave_gload_and_move_slice_window_mbbs():
                dup_inst_per_mbb_gld_b = f"buffer_load,{num_gld_b_per_mbb}" if num_gld_b_per_mbb != 1 else "off"
                dup_inst_per_mbb_gld_a = f"buffer_load,{num_gld_a_per_mbb}" if num_gld_a_per_mbb != 1 else "off"
                mbbs_gld_b = create_machine_basic_block(f_gld_b(), dup_inst_per_mbb=dup_inst_per_mbb_gld_b)
                mbbs_gld_a = create_machine_basic_block(f_gld_a(), dup_inst_per_mbb=dup_inst_per_mbb_gld_a)
                mbbs_msw_b = create_machine_basic_block(f_move_slice_window_b())
                mbbs_msw_a = create_machine_basic_block(f_move_slice_window_a())
                return mbbs_gld_b + mbbs_gld_a + mbbs_msw_b + mbbs_msw_a

            def do_interleave_unroll_k_last():
                with self._deferred_context():
                    unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                    if unroll_k_sub != 0:
                        self._emit(f"; k iteration : {unroll_k - 2}")
                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 0, 0))

                        self._emit(f"; k iteration : {unroll_k - 1}")
                        # 1st fma
                        self._emit(mfma_step_mxn(0, 0, 1, 1))

                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(mfma_step_mxn(0, 1, 1, 1))
                        self._emit_empty_line()
                        #       iteration--
                        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                        self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")
                        if self.ctrl.opt_1st_sld:
                            self._emit(f"s_waitcnt lgkmcnt(0)")
                            self._emit(f"s_barrier")
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))

                        # 3rd fma
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        if self.ctrl.opt_1st_sld:
                            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
                            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 1, 1))
                        if not self.ctrl.opt_1st_sld:
                            self._emit(f"s_waitcnt lgkmcnt(0)")
                            self._emit(f"s_barrier")
                        self._emit(f"s_branch {label_mfma_body}")
                    else:
                        #       iteration--
                        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                        self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")
                        if self.ctrl.opt_1st_sld:
                            self._emit(f"s_waitcnt lgkmcnt(0)")
                            self._emit(f"s_barrier")
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
                            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
                            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2

                        # 3rd fma
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 1, 1))
                        if not self.ctrl.opt_1st_sld:
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
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")
            if self.ctrl.opt_1st_sld:
                self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
                self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
                self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
                self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            if not self.ctrl.opt_1st_sld:
                self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
                self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
                self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
                self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2

            unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
            #if (unroll_k // k_per_inst) // 2 - 1 != 0:
            if True:
                mbb_list_sub = [create_machine_basic_block(do_interleave_unroll_k_sub(), group_mbb_by_end_of_inst_op="v_mfma"),
                                get_interleave_gload_and_move_slice_window_mbbs()]
                se_sub = create_scheduler(self.mc, mbb_list_sub)

                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                                create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)
                self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))
                if f_move_slice_window_acc != None:
                    self._emit(f_move_slice_window_acc())
                mbb_0_mfma_cnt_after_branch_to_start = 2 * cxm.wave_step_m * cxm.wave_step_n - 1 # number of mfma not count into share store interleave slot, check do_interleave_unroll_k_last for last 2 mfma
                self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1, mbb_0_mfma_cnt_after_branch_to_start=mbb_0_mfma_cnt_after_branch_to_start))
            else:
                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                                create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)
                self._emit(emit_machine_basic_blocks(self.mc, get_interleave_gload_and_move_slice_window_mbbs()))
                if f_move_slice_window_acc != None:
                    self._emit(f_move_slice_window_acc())
                mbb_0_mfma_cnt_after_branch_to_start = 2 * cxm.wave_step_m * cxm.wave_step_n - 1 # number of mfma not count into share store interleave slot, check do_interleave_unroll_k_last for last 2 mfma
                self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1, mbb_0_mfma_cnt_after_branch_to_start=mbb_0_mfma_cnt_after_branch_to_start))

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
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2
            self._emit(do_interleave_unroll_k_sub())

            if unroll_k_sub > 0:
                self._emit(f"; k iteration : {unroll_k - 2}")

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
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2)))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2)))

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
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        else:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst, lds_width_m // 2)) + f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({3 if i_k == 0 else 5})')
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(k_per_inst, lds_width_n // 2)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(k_per_inst, lds_width_m // 2)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        else:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        # 3rd fma
                        self._emit(f's_waitcnt lgkmcnt({4 if i_k == 0 else 5})')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        if i_k == 0:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 0, 0))
                        self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst, lds_width_n // 2)) + \
                                                f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit_empty_line()

                        self._emit(f"; k iteration : {2 * i_k + 1}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst, lds_width_m // 2))+ \
                                                f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(0, 1, 1, 1))
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 3rd fma
                        self._emit(f's_waitcnt lgkmcnt(5)')
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 4th fma
                        self._emit(mfma_step_mxn(1, 1, 1, 1))
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst, lds_width_n // 2)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}")
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((unroll_k // k_per_inst - 1) * k_per_inst, lds_width_m // 2)) + f" ; load i_k:{unroll_k // k_per_inst - 1} into local buffer {1}, repeat {1}")
                        self._emit_empty_line()
                    if unroll_k_sub == 0:
                        self._emit(f"; k iteration : {0}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({2})')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()
                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({3})')
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(k_per_inst, lds_width_n // 2)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(k_per_inst, lds_width_m // 2)) + \
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
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2)))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2)))

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



        def mfma_loop_repeat_2x2():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())

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
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2) ))  # lds_width_n // 2
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) ))  # lds_width_m // 2

            # self._emit(f".itr_k = 0")
            # self._emit(f".rept {unroll_k // k_per_inst - 1}")
            #with self._indent_context():
            for i_k in range( unroll_k // k_per_inst - 1):
                # 1st fma
                self._emit(f's_waitcnt lgkmcnt(2)')
                self._emit(mfma_step_mxn(0, 0))

                # 2nd fma
                self._emit(f's_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(0, 1))

                # 3rd fma
                # self._emit(f_sld_a(v_a(), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m * k_per_inst}'))
                self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((i_k+1) * k_per_inst)))
                self._emit(f's_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(1, 0))

                # 4th fma
                # self._emit(f_sld_b(v_b(), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n * k_per_inst}'))
                self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((i_k+1)* k_per_inst)))
                self._emit(mfma_step_mxn(1, 1))
                self._emit_empty_line()

                # last
                # self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n * k_per_inst}+{lds_width_n//2}'))
                # self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m * k_per_inst}+{lds_width_m//2}'))
                self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((i_k+1) * k_per_inst, lds_width_n//2)))
                self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((i_k+1) * k_per_inst, lds_width_m//2)))
                # self._emit('.itr_k = .itr_k + 1')

            # self._emit(f".endr")
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
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())
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
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2 )))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2 )))

            # self._emit(f".itr_k = 0")
            # self._emit(f".rept {unroll_k // k_per_inst - 1}")
            # with self._indent_context():
            for i_k in range(unroll_k // k_per_inst - 1):
                # 1st fma
                self._emit('s_waitcnt lgkmcnt(2)')
                self._emit(mfma_step_mxn(0, 0))

                # 2nd fma
                self._emit('s_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(0, 1))

                # 3rd fma
                # self._emit(f_sld_a(v_a(), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m * k_per_inst}'))
                self._emit(f_sld_a(v_a(), v_sld_a_os(),  lds_base_m + mi_m((i_k+1)* k_per_inst)))
                self._emit('s_waitcnt lgkmcnt(1)')
                self._emit(mfma_step_mxn(1, 0))

                # 4th fma
                # self._emit(f_sld_b(v_b(), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n * k_per_inst}'))
                self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((i_k+1)* k_per_inst)))
                self._emit(mfma_step_mxn(1, 1))
                self._emit_empty_line()

                # last
                #self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n * k_per_inst}+{lds_width_n//2}'))
                #self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m * k_per_inst}+{lds_width_m//2}'))
                self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((i_k+1) * k_per_inst, lds_width_n//2)))
                self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((i_k+1) * k_per_inst, lds_width_m//2)))
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


        def mfma_loop_repeat_2x1_lp2():
            mfma = cxm.inst_mfma
            repeat_m_thread_offset = cxm.wave_step_m * mfma.num_v_a
            # repeat_n_thread_offset = cxm.wave_step_n * mfma.num_v_b
            local_buffer_m = cxm.inst_mfma.num_v_a * cxm.wave_step_m * cxm.wave_repeat_m
            local_buffer_n = cxm.inst_mfma.num_v_b * cxm.wave_step_n * cxm.wave_repeat_n

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())

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
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2) )) # lds_width_m // 2

            def do_unroll_k_sub():
                unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                for i_k in range(unroll_k_sub):
                    self._emit(f"; k iteration : {(2 * i_k + 0) * k_per_inst}")
                    # 1st fma
                    self._emit(f's_waitcnt lgkmcnt({1 if i_k == 0 else 2})')
                    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    if i_k == 0:
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")   # lds_width_n * k_per_inst
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")   # lds_width_m * k_per_inst
                        if unroll_k_sub == 1:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(k_per_inst, lds_width_m // 2) ) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")   # lds_width_m * k_per_inst + lds_width_m // 2
                    elif i_k == unroll_k_sub - 1:
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst)) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}") #  (2*i_k+1) * lds_width_m * k_per_inst
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst, lds_width_m // 2 )) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}") # (2*i_k+1) * lds_width_m * k_per_inst + lds_width_m // 2
                    else:
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst)) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}") # (2*i_k+1) * lds_width_m * k_per_inst
                    self._emit_empty_line()

                    # 2nd fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                    self._emit(mfma_step_mxn(1, 0, 0, 0))
                    if i_k == unroll_k_sub - 1:
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst))+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    else:
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst, lds_width_m // 2 )) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    self._emit_empty_line()

                    self._emit(f"; k iteration : {(2 * i_k + 1) * k_per_inst}")
                    # 1st fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                    self._emit(mfma_step_mxn(0, 0, 1, 1))
                    if i_k == unroll_k_sub - 1:
                        self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst, lds_width_m // 2))+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst)) + f" ; load i_k:{(2*i_k+3)} into local buffer {1}, repeat {0}")

                    else:
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    self._emit_empty_line()

                    # 2nd fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 4})')
                    self._emit(mfma_step_mxn(1, 0, 1, 1))
                    if i_k == unroll_k_sub - 1:
                        # v_b attension!
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst, lds_width_m // 2)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}")
                    else:
                        self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst, lds_width_m // 2))+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
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
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())
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
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2 )))
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
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                            if unroll_k_sub == 1:
                                self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(k_per_inst,lds_width_m // 2 )) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        elif i_k == unroll_k_sub - 1:
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst)) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst, lds_width_m // 2 )) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                        else:
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst)) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        else:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst,  lds_width_m // 2 )) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        self._emit(f"; k iteration : {(2 * i_k + 1) * k_per_inst}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst, lds_width_m // 2))+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst)) + f" ; load i_k:{(2*i_k+3)} into local buffer {1}, repeat {0}")

                        else:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 4})')
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        if i_k == unroll_k_sub - 1:
                            # v_b attension!
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst, lds_width_m // 2)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}")
                        else:
                            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst, lds_width_m // 2))+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()
                    if unroll_k_sub == 0:
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({1})')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({2})')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))

                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(k_per_inst,  lds_width_m // 2 )) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        #self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n(2 * k_per_inst)) + f" ; load i_k:{2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({1})')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))

                        #self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()


                return self._get_deferred()

            def get_interleave_gload_and_move_slice_window_mbbs():
                dup_inst_per_mbb_gld_b = f"buffer_load,{num_gld_b_per_mbb}" if num_gld_b_per_mbb != 1 else "off"
                dup_inst_per_mbb_gld_a = f"buffer_load,{num_gld_a_per_mbb}" if num_gld_a_per_mbb != 1 else "off"
                mbbs_gld_b = create_machine_basic_block(f_gld_b(), dup_inst_per_mbb=dup_inst_per_mbb_gld_b)
                mbbs_gld_a = create_machine_basic_block(f_gld_a(), dup_inst_per_mbb=dup_inst_per_mbb_gld_a)
                mbbs_msw_b = create_machine_basic_block(f_move_slice_window_b())
                mbbs_msw_a = create_machine_basic_block(f_move_slice_window_a())
                return mbbs_gld_b + mbbs_gld_a + mbbs_msw_b + mbbs_msw_a

            def do_interleave_unroll_k_last():
                with self._deferred_context():
                    unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                    if unroll_k_sub != 0:
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
                    else:
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
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2 )))

            unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
            #if (unroll_k // k_per_inst) // 2 - 1 != 0:
            if True:
                mbb_list_sub = [create_machine_basic_block(do_interleave_unroll_k_sub(), group_mbb_by_end_of_inst_op="v_mfma"),
                                get_interleave_gload_and_move_slice_window_mbbs()]

                se_sub = create_scheduler(self.mc, mbb_list_sub)

                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                                create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)
                self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))
                if f_move_slice_window_acc != None:
                    self._emit(f_move_slice_window_acc())
                mbb_0_mfma_cnt_after_branch_to_start = (2 * cxm.wave_step_m * cxm.wave_step_n - 1) if unroll_k_sub != 0 else 0 # number of mfma not count into share store interleave slot, check do_interleave_unroll_k_last for last 2 mfma
                self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1, mbb_0_mfma_cnt_after_branch_to_start=mbb_0_mfma_cnt_after_branch_to_start))
            else:
                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                                create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)
                self._emit(emit_machine_basic_blocks(self.mc, get_interleave_gload_and_move_slice_window_mbbs()))
                if f_move_slice_window_acc != None:
                    self._emit(f_move_slice_window_acc())
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
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2 )))
            self._emit(do_interleave_unroll_k_sub())

            
            if unroll_k_sub > 0:
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

        def mfma_loop_repeat_2x1_lp2_double_buffer_with_interleave():
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
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                            if unroll_k_sub == 1:
                                self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(k_per_inst,lds_width_m // 2 )) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        elif i_k == unroll_k_sub - 1:
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst)) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst, lds_width_m // 2 )) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                        else:
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst)) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        else:
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+1) * k_per_inst,  lds_width_m // 2 )) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        self._emit(f"; k iteration : {(2 * i_k + 1) * k_per_inst}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst, lds_width_m // 2))+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst)) + f" ; load i_k:{(2*i_k+3)} into local buffer {1}, repeat {0}")

                        else:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 4})')
                        self._emit(mfma_step_mxn(1, 0, 1, 1))
                        if i_k == unroll_k_sub - 1:
                            # v_b attension!
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst, lds_width_m // 2)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}")
                        else:
                            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst, lds_width_m // 2))+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()
                    if unroll_k_sub == 0:
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({1})')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({2})')
                        self._emit(mfma_step_mxn(1, 0, 0, 0))

                        self._emit(f_sld_a(v_a(local_buffer_m + repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(k_per_inst,  lds_width_m // 2 )) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        #self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n(2 * k_per_inst)) + f" ; load i_k:{2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({1})')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))

                        #self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()


                return self._get_deferred()

            def get_interleave_gload_and_move_slice_window_mbbs():
                #dup_inst_per_mbb_gld_b = f"buffer_load,{num_gld_b_per_mbb}" if num_gld_b_per_mbb != 1 else "off"
                #dup_inst_per_mbb_gld_a = f"buffer_load,{num_gld_a_per_mbb}" if num_gld_a_per_mbb != 1 else "off"
                #mbbs_gld_b = create_machine_basic_block(f_gld_b(), dup_inst_per_mbb=dup_inst_per_mbb_gld_b)
                #mbbs_gld_a = create_machine_basic_block(f_gld_a(), dup_inst_per_mbb=dup_inst_per_mbb_gld_a)
                mbbs_msw_b = create_machine_basic_block(f_move_slice_window_b())
                mbbs_msw_a = create_machine_basic_block(f_move_slice_window_a())
                return mbbs_msw_b + mbbs_msw_a #mbbs_gld_b + mbbs_gld_a + mbbs_msw_b + mbbs_msw_a

            def do_interleave_unroll_k_last():
                with self._deferred_context():
                    unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                    if unroll_k_sub != 0:
                        self._emit(f"; k iteration : {unroll_k - 2 * k_per_inst}")
                        # 1st fma
                        #self._emit(f's_waitcnt lgkmcnt(4)')
                        self._emit(mfma_step_mxn(0, 0, 0, 0))
                        self._emit_empty_line()

                        # 2nd fma
                        #self._emit(f's_waitcnt lgkmcnt(0)')
                        #self._emit(f"s_barrier")
                        # self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
                        self._emit(mfma_step_mxn(1, 0, 0, 0))
                        self._emit_empty_line()

                        self._emit(f"v_xor_b32 v[{v_sld_b_os()}], {lds_single_size}, v[{v_sld_b_os()}] ; switch double buffer b load")
                        self._emit(f"v_xor_b32 v[{v_sld_a_os()}], {lds_single_size}, v[{v_sld_a_os()}] ; switch double buffer a load")

                        # 1st fma
                        #self._emit(f"s_waitcnt vmcnt(0)")
                        self._emit(f"; k iteration : {unroll_k - 1 * k_per_inst}")
                        self._emit(mfma_step_mxn(0, 0, 1, 1))

                        # 2nd fma
                        self._emit(mfma_step_mxn(1, 0, 1, 1))

                        self._emit_empty_line()
                        #       iteration--
                        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                        self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

                        self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {lds_single_size}, v[{v_sst_b_os()}] ; switch double buffer b store")
                        self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {lds_single_size}, v[{v_sst_a_os()}] ; switch double buffer a store")

                        self._emit(f"s_branch {label_mfma_body}")
                    else:
                        # 2nd fma
                        self._emit(mfma_step_mxn(1, 0, 1, 1))

                        self._emit(f"v_xor_b32 v[{v_sld_b_os()}], {lds_single_size}, v[{v_sld_b_os()}] ; switch double buffer b load")
                        self._emit(f"v_xor_b32 v[{v_sld_a_os()}], {lds_single_size}, v[{v_sld_a_os()}] ; switch double buffer a load")

                        #       iteration--
                        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {unroll_k}")
                        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
                        self._emit(f"s_cbranch_scc0 {label_mfma_finishing}")

                        self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {lds_single_size}, v[{v_sst_b_os()}] ; switch double buffer b store")
                        self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {lds_single_size}, v[{v_sst_a_os()}] ; switch double buffer a store")

                        
                        self._emit(f"s_branch {label_mfma_body}")
                return self._get_deferred()

            def do_interleave_share_store():
                with self._deferred_context():
                    #self._emit(f's_waitcnt lgkmcnt(0)')
                    self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")
                    self._emit(f_sst_b())
                    self._emit(f"s_waitcnt vmcnt(0)")
                    self._emit(f_sst_a())
                return self._get_deferred()

            self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {hex(lds_single_size)}, v[{v_sst_b_os()}] ; switch double buffer b store")
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {hex(lds_single_size)}, v[{v_sst_a_os()}] ; switch double buffer a store")

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f_gld_b())                                           # global load
            self._emit(f_gld_a()) 
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")

            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2 )))

            unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
            mbb_list_sub = [create_machine_basic_block(do_interleave_unroll_k_sub(), group_mbb_by_end_of_inst_op="v_mfma"),
                            get_interleave_gload_and_move_slice_window_mbbs()]

            se_sub = create_scheduler(self.mc, mbb_list_sub)

            mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                            create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

            se_last = create_scheduler(self.mc, mbb_list_last)
            self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())

            self._emit(f's_waitcnt lgkmcnt(0)')

            mbb_0_mfma_cnt_after_branch_to_start = (2 * cxm.wave_step_m * cxm.wave_step_n - 1) if unroll_k_sub != 0 else 0 # number of mfma not count into share store interleave slot, check do_interleave_unroll_k_last for last 2 mfma
            self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1, mbb_0_mfma_cnt_after_branch_to_start=mbb_0_mfma_cnt_after_branch_to_start))

            # Label: finishing of fma body
            self._emit_front(f"{label_mfma_finishing}:")
            #self._emit(mfma_step_mxn(1, 0, 1, 1))
            self._emit_empty_line()

            self._emit_front(f"{label_mfma_end}:")
            self._emit("s_waitcnt lgkmcnt(0)")
            self._emit("s_barrier")

            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_a(v_a(repeat_m_thread_offset), v_sld_a_os(), lds_base_m + mi_m(0, lds_width_m // 2 )))
            self._emit(do_interleave_unroll_k_sub())

            
            if unroll_k_sub > 0:
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
            # print(f"local_buffer_m:{local_buffer_m}, local_buffer_n:{local_buffer_n}, {cxm.inst_mfma.num_v_b}, {cxm.wave_step_n}, {cxm.wave_repeat_n}")

            # right after clear acc
            self._emit(f_move_slice_window_b())
            self._emit(f_move_slice_window_a())
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())

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
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2 )))

            def do_unroll_k_sub():
                unroll_k_sub = (unroll_k // k_per_inst) // 2 - 1
                for i_k in range(unroll_k_sub):
                    self._emit(f"; k iteration : {(2 * i_k + 0) * k_per_inst}")
                    # 1st fma
                    self._emit(f's_waitcnt lgkmcnt({1 if i_k == 0 else 2})')
                    self._emit(mfma_step_mxn(0, 0, 0, 0))
                    if i_k == 0:
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                        if unroll_k_sub == 1:
                            self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(k_per_inst, lds_width_n // 2 )) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                    elif i_k == unroll_k_sub - 1:
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+1) * k_per_inst)) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+1) * k_per_inst, lds_width_n // 2 )) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                    else:
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+1) * k_per_inst)) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                    self._emit_empty_line()

                    # 2nd fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                    self._emit(mfma_step_mxn(0, 1, 0, 0))
                    if i_k == unroll_k_sub - 1:
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    else:
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+1) * k_per_inst, lds_width_n // 2 )) + \
                                                        f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    self._emit_empty_line()

                    self._emit(f"; k iteration : {(2 * i_k + 1) * k_per_inst}")
                    # 1st fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                    self._emit(mfma_step_mxn(0, 0, 1, 1))
                    if i_k == unroll_k_sub - 1:
                        self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst, lds_width_n // 2))+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst)) + f" ; load i_k:{(2*i_k+3)} into local buffer {1}, repeat {0}")

                    else:
                        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                    self._emit_empty_line()

                    # 2nd fma
                    self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 4})')
                    self._emit(mfma_step_mxn(0, 1, 1, 1))
                    if i_k == unroll_k_sub - 1:
                        # v_b attension!
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst, lds_width_n // 2)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}")
                    else:
                        self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst, lds_width_n // 2))+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                        self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
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
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())
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
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2 )))
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
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n(k_per_inst)) + \
                                                            f" ; load i_k:{1} into local buffer {1}, repeat {0}")
                            if unroll_k_sub == 1:
                                self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(k_per_inst, lds_width_n // 2 )) + \
                                                        f" ; load i_k:{1} into local buffer {1}, repeat {1}")
                        elif i_k == unroll_k_sub - 1:
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+1) * k_per_inst)) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+1) * k_per_inst, lds_width_n // 2 )) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                        else:
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+1) * k_per_inst)) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                        self._emit(mfma_step_mxn(0, 1, 0, 0))
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        else:
                            self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+1) * k_per_inst, lds_width_n // 2 )) + \
                                                            f" ; load i_k:{2*i_k+1} into local buffer {1}, repeat {1}")
                            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m + mi_m((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        self._emit(f"; k iteration : {(2 * i_k + 1) * k_per_inst}")
                        # 1st fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 3})')
                        self._emit(mfma_step_mxn(0, 0, 1, 1))
                        if i_k == unroll_k_sub - 1:
                            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst, lds_width_n // 2))+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                            self._emit(f_sld_b(v_b(local_buffer_n), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst)) + f" ; load i_k:{(2*i_k+3)} into local buffer {1}, repeat {0}")

                        else:
                            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst)) + f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {0}")
                        self._emit_empty_line()

                        # 2nd fma
                        self._emit(f's_waitcnt lgkmcnt({2 if i_k != unroll_k_sub - 1 else 4})')
                        self._emit(mfma_step_mxn(0, 1, 1, 1))
                        if i_k == unroll_k_sub - 1:
                            # v_b attension!
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                            self._emit(f_sld_b(v_b(local_buffer_n + repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+3) * k_per_inst, lds_width_n // 2))+ f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {1}")
                        else:
                            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n((2*i_k+2) * k_per_inst, lds_width_n // 2))+ f" ; load i_k:{2*i_k+2} into local buffer {0}, repeat {1}")
                            self._emit(f_sld_a(v_a(local_buffer_m), v_sld_a_os(), lds_base_m + mi_m((2*i_k+3) * k_per_inst)) + f" ; load i_k:{2*i_k+3} into local buffer {1}, repeat {0}")
                        self._emit_empty_line()
                return self._get_deferred()

            def get_interleave_gload_and_move_slice_window_mbbs():
                dup_inst_per_mbb_gld_b = f"buffer_load,{num_gld_b_per_mbb}" if num_gld_b_per_mbb != 1 else "off"
                dup_inst_per_mbb_gld_a = f"buffer_load,{num_gld_a_per_mbb}" if num_gld_a_per_mbb != 1 else "off"
                mbbs_gld_b = create_machine_basic_block(f_gld_b(), dup_inst_per_mbb=dup_inst_per_mbb_gld_b)
                mbbs_gld_a = create_machine_basic_block(f_gld_a(), dup_inst_per_mbb=dup_inst_per_mbb_gld_a)
                mbbs_msw_b = create_machine_basic_block(f_move_slice_window_b())
                mbbs_msw_a = create_machine_basic_block(f_move_slice_window_a())
                return mbbs_gld_b + mbbs_gld_a + mbbs_msw_b + mbbs_msw_a

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
            if f_move_slice_window_acc != None:
                self._emit(f_move_slice_window_acc())

            self._emit(f"s_waitcnt lgkmcnt(0)")
            self._emit(f"s_barrier")

            # Label: start of fma body
            self._emit_front(f"{label_mfma_body}:")
            self._emit(f"; do fma accumulate with unroll {unroll_k}")
            self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
            self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2 )))

            if (unroll_k // k_per_inst) // 2 - 1 != 0:
                mbb_list_sub = [create_machine_basic_block(do_interleave_unroll_k_sub(), group_mbb_by_end_of_inst_op="v_mfma"),
                                get_interleave_gload_and_move_slice_window_mbbs()]

                se_sub = create_scheduler(self.mc, mbb_list_sub)

                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                                create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)
                self._emit(se_sub.lower(interleave_pattern=INTERLEAVE_PTN_0))
                if f_move_slice_window_acc != None:
                    self._emit(f_move_slice_window_acc())
                mbb_0_mfma_cnt_after_branch_to_start = 2 * cxm.wave_step_m * cxm.wave_step_n - 1 # number of mfma not count into share store interleave slot, check do_interleave_unroll_k_last for last 2 mfma
                self._emit(se_last.lower(interleave_pattern=INTERLEAVE_PTN_1, mbb_0_mfma_cnt_after_branch_to_start=mbb_0_mfma_cnt_after_branch_to_start))
            else:
                mbb_list_last = [create_machine_basic_block(do_interleave_unroll_k_last(), group_mbb_by_end_of_inst_op="v_mfma"),
                                create_machine_basic_block(do_interleave_share_store(), group_mbb_by_end_of_inst_op="ds_write")]

                se_last = create_scheduler(self.mc, mbb_list_last)
                self._emit(emit_machine_basic_blocks(self.mc, get_interleave_gload_and_move_slice_window_mbbs()))
                if f_move_slice_window_acc != None:
                    self._emit(f_move_slice_window_acc())
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
            self._emit(f_sld_b(v_b(repeat_n_thread_offset), v_sld_b_os(), lds_base_n + mi_n(0, lds_width_n // 2 )))
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
        self._emit(f"; start MFMA loop, {cxm.wave_tile_m}x{cxm.wave_tile_n} wave tile with {cxm.wave_repeat_m}x{cxm.wave_repeat_n} repeat, {cxm.wave_step_m}x{cxm.wave_step_n} step, k_pack:{self.ctrl.lds_k_pack}")
        self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")

        self._emit(f_sst_b())
        self._emit_empty_line()
        self._emit(f"s_waitcnt vmcnt(0)")
        self._emit(f_sst_a())
        self._emit_empty_line()

        if self.ctrl.accvgpr_unified:
            self._emit(f".v_clear_nc {a_c()}, {cxm.total_acc_c()}")
            set_ctrl_xdlops_mapping_accvgpr_unified(True)
        else:
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
                mfma_loop_repeat_2x2()
            elif self.ctrl.local_prefetch_num == 2:
                if self.ctrl.interleave:
                    if self.ctrl.lds_buffer_num == 2:
                        mfma_loop_repeat_2x2_lp2_double_buffer_with_interleave()
                    else:
                        if cxm.wave_step_m == 2 or cxm.wave_step_n == 2:
                            mfma_loop_repeat_2x2_step_2x2_lp2_with_interleave()
                        else:
                            mfma_loop_repeat_2x2_lp2_with_interleave()
                else:
                    mfma_loop_repeat_2x2_lp2()
        elif cxm.wave_repeat_m == 1 and cxm.wave_repeat_n == 1:
            if self.ctrl.local_prefetch_num == 1:
                assert False
            elif self.ctrl.local_prefetch_num == 2:
                if self.ctrl.interleave:
                    if self.ctrl.lds_buffer_num == 2:
                        mfma_loop_repeat_1x1_lp2_double_buffer_with_interleave()
                    else:
                        mfma_loop_repeat_1x1_lp2_with_interleave()
                else:
                    mfma_loop_repeat_1x1_lp2()
        elif cxm.wave_repeat_m == 2 and cxm.wave_repeat_n == 1:
            if self.ctrl.local_prefetch_num == 1:
                assert False
            elif self.ctrl.local_prefetch_num == 2:
                if self.ctrl.interleave:
                    if self.ctrl.lds_buffer_num == 2:
                        mfma_loop_repeat_2x1_lp2_double_buffer_with_interleave()
                    else:
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
