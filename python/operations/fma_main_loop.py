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

class inst_fma_t(object):
    def __init__(self, arch_config):
        self.arch_config = arch_config
    def __call__(self, reg_c, reg_a, reg_b):
        if self.arch_config.data_type == AMDGPU_PRECISION_FP32:
            if self.arch_config.arch == AMDGPU_ARCH_GFX906:
                if self.arch_config.use_dlops:
                    return 'v_fmac_f32 v[{}], v[{}], v[{}]'.format(reg_c, reg_a, reg_b)
            return 'v_mac_f32 v[{}], v[{}], v[{}]'.format(reg_c, reg_a, reg_b)
        # xdlops
        assert False, 'unimplemented fma type'

class macro_v_fma_mxn_t(macro_base_t):
    '''
    continuous fma, or strided fma
    TODO: implement any index-ed fma (for rdna)
    '''
    def name(self):
        return f".v_fma_{self.m}x{self.n}" + \
                ("" if self.stride == 1 else f"_s{self.stride}")

    def __init__(self, mc, m, n, stride):
        macro_base_t.__init__(self, mc)
        self.m = m
        self.n = n
        self.stride = stride
        assert stride >= n and stride % n == 0
    def __call__(self, c, a, b):
        return '{} {},{},{}'.format(self.name(), c, a, b)
    def emit(self):
        fma = inst_fma_t(self.mc.arch_config)
        reg_a = msym_t(sym_t('a'))
        reg_b = msym_t(sym_t('b'))
        reg_c = msym_t(sym_t('c'))
        with self._emit_macro_indented('.macro {} c, a, b'.format(self.name())):
            for idx_m in range(self.m):
                for idx_n in range(self.n):
                    self._emit(fma(reg_c(idx_m * self.stride + idx_n), reg_a(idx_m), reg_b(idx_n)))

class ctrl_fma_main_loop_t(object):
    def __init__(self):
        self.thread_m                    = 0
        self.thread_n                    = 0
        self.unroll_k                    = 0
        self.label_prefix                = ''                      # usually be kernel name of caller kernel
        self.data_type                   = AMDGPU_PRECISION_FP32
        self.gemm_m_repeat               = 0
        self.gemm_m_level0_cluster       = 0
        self.gemm_m_level1_cluster       = 0
        self.gemm_n_repeat               = 0
        self.gemm_n_level0_cluster       = 0
        self.gemm_n_level1_cluster       = 0
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


class fma_main_loop_t(mc_base_t):
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
        self._emit(f"s_cbranch_scc0 {label_fma_end}")
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
        self._emit_front(f"{label_fma_body}:")
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
        self._emit(f"s_cbranch_scc0 {label_fma_finishing}")

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
        self._emit(f"s_branch {label_fma_body}")

        # Label: finishing of fma body
        self._emit_front(f"{label_fma_finishing}:")
        self._emit(f"s_waitcnt lgkmcnt({f_sst_a.get_issues() + f_sst_a.get_issues()})")
        self._emit(v_fma(v_c(thread_sub_m*thread_n), v_a(thread_sub_m), v_b()))
        self._emit(v_fma(v_c(thread_sub_m*thread_n+thread_sub_n), v_a(thread_sub_m), v_b(thread_sub_n)))

        # Label: end of fma body
        self._emit_front(f"{label_fma_end}:")
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