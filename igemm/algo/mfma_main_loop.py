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

def inst_mfma_data_type_to_string(data_type):
    if data_type == AMDGPU_PRECISION_FP32:
        return 'fp32'
    if data_type == AMDGPU_PRECISION_FP16:
        return 'fp16'
    if data_type == AMDGPU_PRECISION_BF16:
        return 'bf16'
    assert False

class inst_mfma_t(object):
    '''
    http://llvm.org/docs/AMDGPU/AMDGPUAsmGFX908.html
    '''
    def __init__(self, m, n, k, data_type):
        #self.arch_config = arch_config
        self.m = m
        self.n = n
        self.k = k
        self.data_type = data_type
        self.cycle = 1
        self.num_v_a = 1
        self.num_v_b = 1
        self.num_a_c = 1
        self.num_blocks = 1
        #assert arch_config.arch == AMDGPU_ARCH_GFX908 and arch_config.use_xdlops

    def __call__(self, reg_d, reg_a, reg_b, reg_c, cbsz=0, abid=0, blgp=0):
        mfma_acc_type = 'f32' # TODO: int8 mfma accumulate type is i32
        mfma_trait = f'{self.m}x{self.n}x{self.k}' + inst_mfma_data_type_to_string(self.data_type)
        mfma_inst = f'v_mfma_{mfma_acc_type}_{mfma_trait}'
        with self._deferred_context():
            self._emit(f"{mfma_inst} a[{reg_d}], v[{reg_a}], v[{reg_b}], a[{reg_c}] cbsz:{cbsz} abid:{abid} blgp:{blgp}")
        return self._get_deferred()

#                                     m,  n,  k,  precision,           cycle, v_a, v_b, a_c, #block
v_mfma_f32_4x4x1f32     = inst_mfma_t(4,  4,  1,  AMDGPU_PRECISION_FP32,   8,   1,   1,  4,    16)
v_mfma_f32_16x16x1f32   = inst_mfma_t(16, 16, 1,  AMDGPU_PRECISION_FP32,  32,   1,   1,  16,   4 )
v_mfma_f32_16x16x4f32   = inst_mfma_t(16, 16, 4,  AMDGPU_PRECISION_FP32,  32,   1,   1,  4,    1 )
v_mfma_f32_32x32x1f32   = inst_mfma_t(32, 32, 1,  AMDGPU_PRECISION_FP32,  64,   1,   1,  32,   2 )
v_mfma_f32_32x32x2f32   = inst_mfma_t(32, 32, 2,  AMDGPU_PRECISION_FP32,  64,   1,   1,  16,   1 )

v_mfma_f32_4x4x4f16     = inst_mfma_t(4,  4,  4,  AMDGPU_PRECISION_FP16,   8,   2,   2,  4,    16)
v_mfma_f32_16x16x4f16   = inst_mfma_t(16, 16, 4,  AMDGPU_PRECISION_FP16,  32,   2,   2,  16,   4 )
v_mfma_f32_16x16x16f16  = inst_mfma_t(16, 16, 16, AMDGPU_PRECISION_FP16,  32,   2,   2,  4,    1 )
v_mfma_f32_32x32x4f16   = inst_mfma_t(32, 32, 4,  AMDGPU_PRECISION_FP16,  64,   2,   2,  32,   2 )
v_mfma_f32_32x32x8f16   = inst_mfma_t(32, 32, 8,  AMDGPU_PRECISION_FP16,  64,   2,   2,  16,   1 )

v_mfma_f32_4x4x2bf16    = inst_mfma_t(4,  4,  2,  AMDGPU_PRECISION_BF16,   8,   1,   1,  4,    16)
v_mfma_f32_16x16x2bf16  = inst_mfma_t(16, 16, 2,  AMDGPU_PRECISION_BF16,  32,   1,   1,  16,   4 )
v_mfma_f32_16x16x8bf16  = inst_mfma_t(16, 16, 8,  AMDGPU_PRECISION_BF16,  32,   1,   1,  4,    1 )
v_mfma_f32_32x32x2bf16  = inst_mfma_t(32, 32, 2,  AMDGPU_PRECISION_BF16,  64,   1,   1,  32,   2 )
v_mfma_f32_32x32x4bf16  = inst_mfma_t(32, 32, 4,  AMDGPU_PRECISION_BF16,  64,   1,   1,  16,   1 )

class inst_composed_mfma_t(object):
    '''
    handy class to issue several mfma to form a wave wise mxn
    '''
    pass

class inst_composed_mfma_f32_64x64x1f32_t(object):
    def __init__(self):
        self.m = 64
        self.n = 64
        self.k = 1
        self.data_type = AMDGPU_PRECISION_FP32
        self.mfma = v_mfma_f32_32x32x1f32
    def issues(self):
        return 2
    def issue0(self, reg_d, reg_a, reg_b, reg_c):
        return self.mfma(reg_d, reg_a, reg_b, reg_c, 1, 0, 0)
    def issue1(self, reg_d, reg_a, reg_b, reg_c):
        return self.mfma(reg_d, reg_a, reg_b, reg_c, 1, 1, 0)
    def __call__(self, reg_d, reg_a, reg_b, reg_c):
        with self._deferred_context():
            self._emit(self.issue0(reg_d, reg_a, reg_b, reg_c))
            self._emit(self.issue1(reg_d'+32', reg_a, reg_b, reg_c))
        return self._get_deferred()

class inst_composed_mfma_f32_32x64x1f32_t(object):
    def __init__(self):
        self.m = 32
        self.n = 64
        self.k = 1
        self.data_type = AMDGPU_PRECISION_FP32
        self.mfma = v_mfma_f32_32x32x1f32
    def issues(self):
        return 1
    def issue0(self, reg_d, reg_a, reg_b, reg_c):
        return self.mfma(reg_d, reg_a, reg_b, reg_c, 1, 0, 0)
    def __call__(self, reg_d, reg_a, reg_b, reg_c):
        return self.issue0(reg_d, reg_a, reg_b, reg_c)

class inst_composed_mfma_f32_64x32x1f32_t(object):
    def __init__(self):
        self.m = 64
        self.n = 32
        self.k = 1
        self.data_type = AMDGPU_PRECISION_FP32
        self.mfma = v_mfma_f32_32x32x1f32
    def issues(self):
        return 1
    def issue0(self, reg_d, reg_a, reg_b, reg_c):
        return self.mfma(reg_d, reg_a, reg_b, reg_c, 0, 0, 1)
    def __call__(self, reg_d, reg_a, reg_b, reg_c):
        return self.issue0(reg_d, reg_a, reg_b, reg_c)

class inst_composed_mfma_f32_16x64x1f32_t(object):
    def __init__(self):
        self.m = 16
        self.n = 64
        self.k = 1
        self.data_type = AMDGPU_PRECISION_FP32
        self.mfma = v_mfma_f32_16x16x1f32
    def issues(self):
        return 1
    def issue0(self, reg_d, reg_a, reg_b, reg_c):
        return self.mfma(reg_d, reg_a, reg_b, reg_c, 2, 0, 0)
    def __call__(self, reg_d, reg_a, reg_b, reg_c):
        return self.issue0(reg_d, reg_a, reg_b, reg_c)

class inst_composed_mfma_f32_64x16x1f32_t(object):
    def __init__(self):
        self.m = 64
        self.n = 16
        self.k = 1
        self.data_type = AMDGPU_PRECISION_FP32
        self.mfma = v_mfma_f32_16x16x1f32
    def issues(self):
        return 1
    def issue0(self, reg_d, reg_a, reg_b, reg_c):
        return self.mfma(reg_d, reg_a, reg_b, reg_c, 0, 0, 4)
    def __call__(self, reg_d, reg_a, reg_b, reg_c):
        return self.issue0(reg_d, reg_a, reg_b, reg_c)

v_mfma_f32_64x64x1f32   = inst_composed_mfma_f32_64x64x1f32_t()
v_mfma_f32_32x64x1f32   = inst_composed_mfma_f32_32x64x1f32_t()
v_mfma_f32_64x32x1f32   = inst_composed_mfma_f32_64x32x1f32_t()
v_mfma_f32_16x64x1f32   = inst_composed_mfma_f32_16x64x1f32_t()
v_mfma_f32_64x16x1f32   = inst_composed_mfma_f32_64x16x1f32_t()


class ctrl_mfma_main_loop_t(object):
    def __init__(self):
        self.wave_m                      = 0
        self.wave_n                      = 0
        self.unroll_k                    = 0
        self.label_prefix                = ''                      # usually be kernel name of caller kernel
        self.data_type                   = AMDGPU_PRECISION_FP32

        self.gemm_m_wave_step            = 0
        self.gemm_m_wave_repeat          = 0
        self.gemm_n_wave_step            = 0
        self.gemm_n_wave_repeat          = 0

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
        # self.v_c                         = None
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
        v_c = self.ctrl.v_c

        v_gld_a = self.ctrl.v_gld_a
        v_gld_b = self.ctrl.v_gld_b

        v_sst_a_os = self.ctrl.v_sst_a_os
        v_sld_a_os = self.ctrl.v_sld_a_os
        v_sst_b_os = self.ctrl.v_sst_b_os
        v_sld_b_os = self.ctrl.v_sld_b_os

        s_kitr = self.ctrl.s_kitr
        s_knum = self.ctrl.s_knum

        # assert type(v_a) is sym_t and type(s_kitr) is sym_t  # other gpr type check ignore

        data_byte = amdgpu_precision_data_byte(self.ctrl.data_type)

        lds_width_m = data_byte * self.ctrl.thread_m * self.ctrl.gemm_m_level0_cluster * self.ctrl.gemm_m_level1_cluster
        lds_width_n = data_byte * self.ctrl.thread_n * self.ctrl.gemm_n_level0_cluster * self.ctrl.gemm_n_level1_cluster
        lds_single_size = self.ctrl.lds_single_size

        # used as offset:x number. may some 
        lds_base_m = 0
        lds_base_n = 0
        unroll_k = self.ctrl.unroll_k

        assert self.ctrl.gemm_m_repeat == 2 and self.ctrl.thread_m % self.ctrl.gemm_m_repeat == 0 \
            and self.ctrl.gemm_n_repeat == 2 and self.ctrl.thread_n % self.ctrl.gemm_n_repeat == 0

        thread_m = self.ctrl.thread_m
        thread_n = self.ctrl.thread_n
        thread_sub_m = self.ctrl.thread_m // self.ctrl.gemm_m_repeat
        thread_sub_n = self.ctrl.thread_n // self.ctrl.gemm_n_repeat

        v_fma = macro_v_fma_mxn_t(self.mc, thread_sub_m, thread_sub_n, thread_n)

        # start emit
        self._emit(f"; start FMA loop, {thread_m}x{thread_n} thread tile with {thread_sub_m}x{thread_sub_n} sub-tile")
        self._emit(f"s_waitcnt vmcnt({f_gld_a.get_issues()})")

        self._emit(f_sst_b())
        self._emit_empty_line()
        self._emit(f"s_waitcnt vmcnt(0)")
        self._emit(f_sst_a())
        self._emit_empty_line()

        self._emit(f".v_clear_nc {v_c()}, {thread_m * thread_n}")

        # decrese k
        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_knum()}], {unroll_k}")
        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
        self._emit(f"s_cbranch_scc0 {label_mfma_end}")
        self._emit_empty_line()

        self._emit(f_move_slice_window_b())
        self._emit(f_move_slice_window_a())

        self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {hex(lds_single_size)}, v[{v_sst_b_os()}] ; switch double buffer b store")
        self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {hex(lds_single_size)}, v[{v_sst_a_os()}] ; switch double buffer a store")
        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit(f"s_barrier")
        self._emit_empty_line()
        self._emit(f_gld_b())
        self._emit(f_gld_a())
        self._emit_empty_line()

        # Label: start of fma body
        self._emit_front(f"{label_mfma_body}:")
        self._emit(f"; do fma accumulate with unroll {unroll_k}")
        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
        self._emit(f_sld_b(v_b(thread_sub_n), v_sld_b_os(), lds_base_n + lds_width_n // 2 ))
        self._emit(f_sld_a(v_a(thread_sub_m), v_sld_a_os(), lds_base_m + lds_width_m // 2 ))

        self._emit(f".itr_k = 0")
        self._emit(f".rept {unroll_k-1}")
        with self._indent_context():
            # 1st fma
            self._emit(f's_waitcnt lgkmcnt(2)')
            self._emit(v_fma(v_c(), v_a(), v_b()))
            #self._emit_empty_line()

            # 2nd fma
            self._emit(f's_waitcnt lgkmcnt(1)')
            self._emit(v_fma(v_c(thread_sub_n), v_a(), v_b(thread_sub_n)))
            #self._emit_empty_line()

            # 3rd fma
            self._emit(f_sld_a(v_a(), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m}'))
            self._emit(f's_waitcnt lgkmcnt(1)')
            self._emit(v_fma(v_c(thread_sub_m * thread_n), v_a(thread_sub_m), v_b()))
            #self._emit_empty_line()

            # 4th fma
            self._emit(f_sld_b(v_b(), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}'))
            self._emit(v_fma(v_c(thread_sub_m * thread_n + thread_sub_n), v_a(thread_sub_m), v_b(thread_sub_n)))
            self._emit_empty_line()

            # last
            self._emit(f_sld_b(v_b(thread_sub_n), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}+{lds_width_n//2}'))
            self._emit(f_sld_a(v_a(thread_sub_m), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m}+{lds_width_m//2}'))
            self._emit('.itr_k = .itr_k + 1')

        self._emit(f".endr")
        self._emit_empty_line()
        self._emit(f"; last unroll")
        self._emit(f"v_xor_b32 v[{v_sld_b_os()}], {lds_single_size}, v[{v_sld_b_os()}] ; switch double buffer b load")
        self._emit(f"v_xor_b32 v[{v_sld_a_os()}], {lds_single_size}, v[{v_sld_a_os()}] ; switch double buffer a load")

        # 1st fma
        self._emit(f"s_waitcnt lgkmcnt(2)")
        self._emit(v_fma(v_c(), v_a(), v_b()))
        #self._emit_empty_line()

        # 2nd fma
        self._emit(f"s_waitcnt lgkmcnt(1)")
        self._emit(v_fma(v_c(thread_sub_n), v_a(), v_b(thread_sub_n)))
        #self._emit_empty_line()

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
        self._emit(v_fma(v_c(thread_sub_m * thread_n), v_a(thread_sub_m), v_b()))
        #self._emit_empty_line()

        self._emit(f"v_xor_b32 v[{v_sst_b_os()}], {lds_single_size}, v[{v_sst_b_os()}] ; switch double buffer b store")
        self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {lds_single_size}, v[{v_sst_a_os()}] ; switch double buffer a store")
        #       barrier here!
        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit(f"s_barrier")

        #       load next from global
        self._emit(f_gld_b())
        self._emit(f_gld_a())

        # 4th fma
        self._emit(v_fma(v_c(thread_sub_m*thread_n+thread_sub_n), v_a(thread_sub_m), v_b(thread_sub_n)))
        self._emit_empty_line()
        self._emit(f"s_branch {label_mfma_body}")

        # Label: finishing of fma body
        self._emit_front(f"{label_mfma_finishing}:")
        self._emit(f"s_waitcnt lgkmcnt({f_sst_a.get_issues() + f_sst_a.get_issues()})")
        self._emit(v_fma(v_c(thread_sub_m*thread_n), v_a(thread_sub_m), v_b()))
        self._emit(v_fma(v_c(thread_sub_m*thread_n+thread_sub_n), v_a(thread_sub_m), v_b(thread_sub_n)))

        # Label: end of fma body
        self._emit_front(f"{label_mfma_end}:")
        self._emit("s_waitcnt lgkmcnt(0)")
        self._emit("s_barrier")

        self._emit(f_sld_a(v_a(), v_sld_a_os(), lds_base_m))
        self._emit(f_sld_b(v_b(), v_sld_b_os(), lds_base_n))
        self._emit(f_sld_b(v_b(thread_sub_n), v_sld_b_os(), lds_base_n + lds_width_n // 2 ))
        self._emit(f_sld_a(v_a(thread_sub_m), v_sld_a_os(), lds_base_m + lds_width_m // 2 ))

        self._emit(f".itr_k = 0")
        self._emit(f".rept {unroll_k - 1}")
        with self._indent_context():
            # 1st fma
            self._emit('s_waitcnt lgkmcnt(2)')
            self._emit(v_fma(v_c(), v_a(), v_b()))
            #self._emit_empty_line()

            # 2nd fma
            self._emit('s_waitcnt lgkmcnt(1)')
            self._emit(v_fma(v_c(thread_sub_n), v_a(), v_b(thread_sub_n)))
            #self._emit_empty_line()

            # 3rd fma
            self._emit(f_sld_a(v_a(), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m}'))
            self._emit('s_waitcnt lgkmcnt(1)')
            self._emit(v_fma(v_c(thread_sub_m*thread_n), v_a(thread_sub_m), v_b()))
            #self._emit_empty_line()

            # 4th fma
            self._emit(f_sld_b(v_b(), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}'))
            self._emit(v_fma(v_c(thread_sub_m*thread_n+thread_sub_n), v_a(thread_sub_m), v_b(thread_sub_n)))
            self._emit_empty_line()

            # last
            self._emit(f_sld_b(v_b(thread_sub_n), v_sld_b_os(), f'{lds_base_n}+(.itr_k+1)*{lds_width_n}+{lds_width_n//2}'))
            self._emit(f_sld_a(v_a(thread_sub_m), v_sld_a_os(), f'{lds_base_m}+(.itr_k+1)*{lds_width_m}+{lds_width_m//2}'))
            self._emit('.itr_k = .itr_k + 1')
        self._emit('.endr')
        self._emit_empty_line()
        self._emit('; last unroll')
        # 1st fma
        self._emit('s_waitcnt lgkmcnt(2)')
        self._emit(v_fma(v_c(), v_a(), v_b()))
        #self._emit_empty_line()

        # 2nd fma
        self._emit('s_waitcnt lgkmcnt(1)')
        self._emit(v_fma(v_c(thread_sub_n), v_a(), v_b(thread_sub_n)))
        #self._emit_empty_line()

        # 3rd fma
        self._emit('s_waitcnt lgkmcnt(0)')
        self._emit(v_fma(v_c(thread_sub_m*thread_n), v_a(thread_sub_m), v_b()))
        #self._emit_empty_line()

        # 4th fma
        self._emit(v_fma(v_c(thread_sub_m*thread_n+thread_sub_n), v_a(thread_sub_m), v_b(thread_sub_n)))
        self._emit_empty_line()