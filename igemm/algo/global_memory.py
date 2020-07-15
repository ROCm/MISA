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
from __future__ import print_function
import sys
from ..codegen import *

class ctrl_2d_global_load_t(object):
    def __init__(self):
        self.length_d0 = 1
        self.length_d1 = 1
        self.vector_d1 = 1
        self.precision = 'fp32'      # 'fp32', 'fp16', ...

class macro_igemm_2d_global_load_t(mc_base_t):
    # TODO: if need vectorize further LDS write, need shuffle dst gpr while load
    def __init__(self, mc, ctrl):
        assert type(ctrl) is ctrl_2d_global_load_t
        mc_base_t.__init__(self, mc)
        self.ctrl = ctrl
    def name(self):
        ctrl = self.ctrl
        if ctrl.precision == "fp32":
            bits_str = 'b32'
        elif ctrl.precision in ("fp16", "bf16"):
            bits_str = 'b16'
        else:
            assert False

        if ctrl.vector_d1 == 4:
            vec_str = 'v4'
        elif ctrl.vector_d1 == 2:
            vec_str = 'v2'
        elif ctrl.vector_d1 == 1:
            vec_str = 'v1'
        else:
            assert False

        #assert ctrl.length_d1 % ctrl.vector_d1 = 0
        #n_d1 = ctrl.length_d1 // ctrl.vector_d1
        return f".v_gld_{ctrl.length_d0}x{ctrl.length_d1}_{bits_str}_{vec_str}"
    def __call__(self, v_dst, s_ptr, v_os, s_stride_d0, s_stride_d1, s_tmp2):
        return '{} {}, {}, {}, {}, {}, {}'.format(self.name(),
                        v_dst, s_ptr, v_os, s_stride_d0, s_stride_d1, s_tmp2)
    def emit(self):
        ctrl = self.ctrl
        assert ctrl.length_d1 % ctrl.vector_d1 == 0
        n_d1 = ctrl.length_d1 // ctrl.vector_d1
        assert ctrl.precision == 'fp32', "TO BE supported"
        buffer_load_dword = buffer_load_dword_t(ctrl.vector_d1)
        with self._emit_macro_indented('.macro {} v_dst, s_ptr, v_os, s_stride_d0, s_stride_d1, s_tmp2'.format(self.name())):
            self._emit(".v_clear_nc \\v_dst, 8")
            i_dst = 0
            for i_d0 in range(ctrl.length_d0):
                for i_d1 in range(n_d1):
                    if i_d1 == 0 and i_d0 == 0:
                        self._emit(buffer_load_dword(f"\\v_dst+{i_dst*ctrl.vector_d1}", "\\v_os", "\\s_ptr", 0, 0))
                    elif i_d1 == 0 and i_d0 != 0:
                        self._emit(buffer_load_dword(f"\\v_dst+{i_dst*ctrl.vector_d1}", "\\v_os", "\\s_ptr", "\\s_tmp2+1", 0))
                    else:
                        self._emit(buffer_load_dword(f"\\v_dst+{i_dst*ctrl.vector_d1}", "\\v_os", "\\s_ptr", "\\s_tmp2", 0))

                    if i_d1 != (n_d1 - 1):
                        if i_d1 == 0 and i_d0 ==  0:
                            self._emit("s_mov_b32 s[\\s_tmp2], s[\\s_stride_d1]")
                        elif i_d1 == 0 and i_d0 !=  0:
                            self._emit("s_add_u32 s[\\s_tmp2], s[\\s_tmp2+1], s[\\s_stride_d1]")
                        else:
                            self._emit("s_add_u32 s[\\s_tmp2], s[\\s_tmp2], s[\\s_stride_d1]")
                    i_dst = i_dst + 1

                if i_d0 != (ctrl.length_d0 - 1):
                    if i_d0 == 0:
                        self._emit("s_mov_b32 s[\\s_tmp2+1], s[\\s_stride_d0]")
                    else:
                        self._emit("s_add_u32 s[\\s_tmp2+1], s[\\s_tmp2+1], s[\\s_stride_d0]")

    def get_issues(self):
        ctrl = self.ctrl
        n_d1 = ctrl.length_d1 // ctrl.vector_d1
        return  ctrl.length_d0 * n_d1

class macro_igemm_write_4d_strided_t(mc_base_t):
    def name(self):
        return '.v_write4d_strided'
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, v_src, s_p_dst, v_dst_os, s_dst_diff1d, s_dst_diff2d, s_dst_diff3d, s_dst_diff4d,
                s_dst_os_4, t_dim_1d, t_dim_2d, t_dim_3d, t_dim_4d, vec_1d = 1):
        return '{} {},{},{},{},{},{},{},{},{},{},{},{}'.format(self.name(),
            v_src, s_p_dst, v_dst_os, s_dst_diff1d, s_dst_diff2d, s_dst_diff3d, s_dst_diff4d,
            s_dst_os_4, t_dim_1d, t_dim_2d, t_dim_3d, t_dim_4d, vec_1d)
    def emit(self):
        def emit_write1d_strided():
            self._emit('; write 1d tensor to global with stride')
            with self._emit_macro_indented('.macro .v_write1d_strided v_src, s_p_buf_dst, v_dst_os, s_dst_diff, s_dst_os, t_dim_1d, vec_1d=1'):
                self._emit('.itr_1d = 0')
                self._emit('.if \\t_dim_1d % \\vec_1d != 0')
                self._emit('    .error "\\t_dim_1d can not evenly divided by \\vec_1d"')
                self._emit('    .end')
                self._emit('.endif')
                self._emit('.t_dim_1d_v = \\t_dim_1d / \\vec_1d')
                self._emit('.rept .t_dim_1d_v')
                self._inc_indent()
                self._emit('.if \\vec_1d == 1')
                self._emit('buffer_store_dword v[\\v_src+.itr_1d], v[\\v_dst_os], s[\\s_p_buf_dst:\\s_p_buf_dst+3], s[\\s_dst_os] offen')
                self._emit('.elseif  \\vec_1d == 2')
                self._emit('buffer_store_dwordx2 v[\\v_src+.itr_1d:\\v_src+.itr_1d+1], v[\\v_dst_os], s[\\s_p_buf_dst:\\s_p_buf_dst+3], s[\\s_dst_os] offen')
                self._emit('.elseif \\vec_1d == 4')
                self._emit('buffer_store_dwordx4 v[\\v_src+.itr_1d:\\v_src+.itr_1d+3], v[\\v_dst_os], s[\\s_p_buf_dst:\\s_p_buf_dst+3], s[\\s_dst_os] offen')
                self._emit('.endif')
                self._emit('.if .itr_1d != .t_dim_1d_v - 1')
                self._inc_indent()
                self._emit('s_add_u32 s[\\s_dst_os], s[\\s_dst_os], s[\\s_dst_diff]')
                self._dec_indent()
                self._emit('.endif')
                self._emit('.itr_1d = .itr_1d + \\vec_1d')
                self._dec_indent()
                self._emit('.endr')

        def emit_write2d_strided():
            self._emit('; write 2d tensor to global with stride')
            with self._emit_macro_indented('.macro .v_write2d_strided v_src, s_p_dst, v_dst_os, s_dst_diff1d, s_dst_diff2d, s_dst_os_2, t_dim_1d, t_dim_2d, vec_1d=1'):
                self._emit('.itr_2d = 0')
                self._emit('.rept \\t_dim_2d')
                self._emit('.v_write1d_strided (\\v_src + .itr_2d * \\t_dim_1d), \\s_p_dst, \\v_dst_os, \\s_dst_diff1d, \\s_dst_os_2, \\t_dim_1d')
                self._emit('.if .itr_2d != \\t_dim_2d - 1')
                self._inc_indent()
                self._emit('s_add_u32 s[\\s_dst_os_2+1], s[\\s_dst_os_2+1], s[\\s_dst_diff2d]')
                self._emit('s_mov_b32 s[\\s_dst_os_2], s[\\s_dst_os_2+1]')
                self._dec_indent()
                self._emit('.endif')
                self._emit('.itr_2d = .itr_2d + 1')
                self._emit('.endr')

        def emit_write3d_strided():
            self._emit('; write 3d tensor to global with stride')
            with self._emit_macro_indented('.macro .v_write3d_strided v_src, s_p_dst, v_dst_os, s_dst_diff1d, s_dst_diff2d, s_dst_diff3d, s_dst_os_3, t_dim_1d, t_dim_2d, t_dim_3d, vec_1d=1'):
                self._emit('.itr_3d = 0')
                self._emit('.rept \\t_dim_3d')
                self._emit('.v_write2d_strided (\\v_src+ .itr_3d * \\t_dim_1d * \\t_dim_2d), \\s_p_dst, \\v_dst_os, \\s_dst_diff1d, \\s_dst_diff2d, \\s_dst_os_3, \\t_dim_1d, \\t_dim_2d')
                self._emit('.if .itr_3d != \\t_dim_3d - 1')
                self._inc_indent()
                self._emit('s_add_u32 s[\\s_dst_os_3+2], s[\\s_dst_os_3+2], s[\\s_dst_diff3d]')
                self._emit('s_mov_b32 s[\\s_dst_os_3+1], s[\\s_dst_os_3+2]')
                self._emit('s_mov_b32 s[\\s_dst_os_3], s[\\s_dst_os_3+1]')
                self._dec_indent()
                self._emit('.endif')
                self._emit('.itr_3d = .itr_3d + 1')
                self._emit('.endr')

        def emit_write4d_strided():
            self._emit('; write 4d tensor to global with stride')
            with self._emit_macro_indented('.macro .v_write4d_strided v_src, s_p_dst, v_dst_os, s_dst_diff1d, s_dst_diff2d, s_dst_diff3d, s_dst_diff4d, s_dst_os_4, t_dim_1d, t_dim_2d, t_dim_3d, t_dim_4d, vec_1d=1'):
                self._emit('.itr_4d = 0')
                self._emit('.rept \\t_dim_4d')
                self._emit('.v_write3d_strided (\\v_src+ .itr_4d * \\t_dim_1d * \\t_dim_2d * \\t_dim_3d), \\s_p_dst, \\v_dst_os, \\s_dst_diff1d, \\s_dst_diff2d, \\s_dst_diff3d, \\s_dst_os_4, \\t_dim_1d, \\t_dim_2d, \\t_dim_3d')
                self._emit('.if .itr_4d != \\t_dim_4d - 1')
                self._inc_indent()
                self._emit('s_add_u32 s[\\s_dst_os_4+3], s[\\s_dst_os_4+3], s[\\s_dst_diff4d]')
                self._emit('s_mov_b32 s[\\s_dst_os_4+2], s[\\s_dst_os_4+3]')
                self._emit('s_mov_b32 s[\\s_dst_os_4+1], s[\\s_dst_os_4+2]')
                self._emit('s_mov_b32 s[\\s_dst_os_4], s[\\s_dst_os_4+1]')
                self._dec_indent()
                self._emit('.endif')
                self._emit('.itr_4d = .itr_4d + 1')
                self._emit('.endr')
        
        emit_write1d_strided()
        emit_write2d_strided()
        emit_write3d_strided()
        emit_write4d_strided()

    def get_issues(self):
        # write out is ignored
        return 0