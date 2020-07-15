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

from .mc import *
from .amdgpu import *

class hsa_header_t(mc_base_t):
    '''
    only used in cov2
    '''
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def emit(self):
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V2:
            self._emit(".hsa_code_object_version 2,1")
            self._emit(".hsa_code_object_isa")
            self._emit_empty_line()

class hsa_footer_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def emit(self):
        pass

class amd_kernel_code_t(mc_base_t):
    def __init__(self, mc, kernel_info):
        mc_base_t.__init__(self, mc)
        self.ki = kernel_info

    def emit(self):
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V2:
            self._emit('.amd_kernel_code_t')
            with self._indent_context():
                if self.ki.kernel_code.enable_sgpr_private_segment_buffer:
                    self._emit('enable_sgpr_private_segment_buffer = {}'.format(    self.ki.kernel_code.enable_sgpr_private_segment_buffer))
                if self.ki.kernel_code.enable_sgpr_dispatch_ptr:
                    self._emit('enable_sgpr_dispatch_ptr = {}'.format(              self.ki.kernel_code.enable_sgpr_dispatch_ptr))
                if self.ki.kernel_code.enable_sgpr_queue_ptr:
                    self._emit('enable_sgpr_queue_ptr = {}'.format(                 self.ki.kernel_code.enable_sgpr_queue_ptr))
                self._emit('enable_sgpr_kernarg_segment_ptr = {}'.format(           self.ki.kernel_code.enable_sgpr_kernarg_segment_ptr))
                if self.ki.kernel_code.enable_sgpr_dispatch_id:
                    self._emit('enable_sgpr_dispatch_id'.format(                    self.ki.kernel_code.enable_sgpr_dispatch_id))
                # other sgpr related to be implemented 
                self._emit('user_sgpr_count = {}'.format(                           self.ki.kernel_code.user_sgpr_count))
                if self.ki.kernel_code.enable_sgpr_workgroup_id_x:
                    self._emit('enable_sgpr_workgroup_id_x = {}'.format(            self.ki.kernel_code.enable_sgpr_workgroup_id_x))
                if self.ki.kernel_code.enable_sgpr_workgroup_id_y:
                    self._emit('enable_sgpr_workgroup_id_y = {}'.format(            self.ki.kernel_code.enable_sgpr_workgroup_id_y))
                if self.ki.kernel_code.enable_sgpr_workgroup_id_z:
                    self._emit('enable_sgpr_workgroup_id_z = {}'.format(            self.ki.kernel_code.enable_sgpr_workgroup_id_z))
                self._emit('enable_vgpr_workitem_id = {}'.format(                   self.ki.kernel_code.enable_vgpr_workitem_id))
                self._emit('is_ptr64 = {}'.format(                                  self.ki.kernel_code.is_ptr64))
                self._emit('float_mode = {}'.format(                                self.ki.kernel_code.float_mode))
                self._emit('workgroup_group_segment_byte_size = {}'.format(         self.ki.kernel_code.workgroup_group_segment_byte_size))
                self._emit('kernarg_segment_byte_size = {}'.format(                 self.ki.kernel_code.kernarg_segment_byte_size))
                self._emit('wavefront_sgpr_count = {}'.format(                      self.ki.kernel_code.wavefront_sgpr_count))
                self._emit('workitem_vgpr_count = {}'.format(                       self.ki.kernel_code.workitem_vgpr_count))
                self._emit('granulated_workitem_vgpr_count = {}'.format(            self.ki.kernel_code.granulated_workitem_vgpr_count))
                self._emit('granulated_wavefront_sgpr_count = {}'.format(           self.ki.kernel_code.granulated_wavefront_sgpr_count))
            self._emit('.end_amd_kernel_code_t')
        elif self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
            self._emit('.rodata')   # v3 is 64 byte rodata for each kerenl
            self._emit('.p2align 6')
            self._emit('.amdhsa_kernel {}'.format(self.ki.kernel_name))
            with self._indent_context():
                if self.ki.kernel_code.kernarg_segment_byte_size > 0:
                    self._emit('.amdhsa_group_segment_fixed_size {}'.format(        self.ki.kernel_code.workgroup_group_segment_byte_size))
                if self.ki.kernel_code.enable_sgpr_dispatch_ptr:
                    self._emit('.amdhsa_user_sgpr_dispatch_ptr {}'.format(          self.ki.kernel_code.enable_sgpr_dispatch_ptr))

                self._emit('.amdhsa_user_sgpr_kernarg_segment_ptr {}'.format(       self.ki.kernel_code.enable_sgpr_kernarg_segment_ptr))

                if self.ki.kernel_code.enable_sgpr_workgroup_id_x:
                    self._emit('.amdhsa_system_sgpr_workgroup_id_x {}'.format(      self.ki.kernel_code.enable_sgpr_workgroup_id_x))
                if self.ki.kernel_code.enable_sgpr_workgroup_id_y:
                    self._emit('.amdhsa_system_sgpr_workgroup_id_y {}'.format(      self.ki.kernel_code.enable_sgpr_workgroup_id_y))
                if self.ki.kernel_code.enable_sgpr_workgroup_id_z:
                    self._emit('.amdhsa_system_sgpr_workgroup_id_z {}'.format(      self.ki.kernel_code.enable_sgpr_workgroup_id_z))

                self._emit('.amdhsa_system_vgpr_workitem_id {}'.format(             self.ki.kernel_code.enable_vgpr_workitem_id))
                self._emit('.amdhsa_next_free_vgpr {}'.format(                      self.ki.kernel_code.workitem_vgpr_count))
                self._emit('.amdhsa_next_free_sgpr {}'.format(                      self.ki.kernel_code.wavefront_sgpr_count))

                self._emit('.amdhsa_ieee_mode 0')   # seems everyone close this?
                self._emit('.amdhsa_dx10_clamp 0')  # seems everyone close this?
            self._emit('.end_amdhsa_kernel')
        else:
            assert False

class amd_metadata_t(mc_base_t):
    '''
    only implement in cov3
    '''
    def __init__(self, mc, kernel_info):
        mc_base_t.__init__(self, mc)
        self.ki = kernel_info

    def emit_one_kernel_metadata(self, ki_):
        self._emit('  - .name: {}'.format(                          ki_.kernel_name))
        self._emit('    .symbol: {}.kd'.format(                     ki_.kernel_name))
        self._emit('    .sgpr_count: {}'.format(                    ki_.kernel_code.wavefront_sgpr_count))
        self._emit('    .vgpr_count: {}'.format(                    ki_.kernel_code.workitem_vgpr_count))
        self._emit('    .kernarg_segment_align: {}'.format(         8))     # default set to 8
        self._emit('    .kernarg_segment_size: {}'.format(          ki_.kernel_code.kernarg_segment_byte_size))
        self._emit('    .group_segment_fixed_size: {}'.format(      ki_.kernel_code.workgroup_group_segment_byte_size))
        self._emit('    .private_segment_fixed_size: {}'.format(    0))     # hard code to 0
        self._emit('    .wavefront_size: {}'.format(                64))
        self._emit('    .reqd_workgroup_size : [{}]'.format(        '{}, 1, 1'.format( ki_.kernel_block_size) \
                                                                                if type(ki_.kernel_block_size) is int else \
                                                                    '{}, {}, {}'.format(ki_.kernel_block_size[0],
                                                                                ki_.kernel_block_size[1],ki_.kernel_block_size[2])))
        self._emit('    .max_flat_workgroup_size: {}'.format(       ki_.kernel_block_size if type(ki_.kernel_block_size) is int else \
                                                                    ki_.kernel_block_size[0]*ki_.kernel_block_size[1]*ki_.kernel_block_size[2]))
        self._emit('    .args:')
        assert ki_.kernel_args
        for kern_arg in ki_.kernel_args:
            self._emit(kern_arg.serialize_as_metadata())

    def emit(self):
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
            self._emit('.amdgpu_metadata')
            self._emit('---')
            self._emit('amdhsa.version: [ 1, 0 ]')
            self._emit('amdhsa.kernels:')
            if type(self.ki) is list:
                for k in self.ki:
                    self.emit_one_kernel_metadata(k)
            else:
                self.emit_one_kernel_metadata(self.ki)
            self._emit('...')
            self._emit('.end_amdgpu_metadata')

class macro_int_div_vv_t(mc_base_t):
    def name(self):
        return '.v_u32_div'
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, v_q, v_n, v_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}'.format(self.name(), v_q, v_n, v_d, v_tmp4, s_tmp4)
    def emit(self):
        with self._emit_macro_indented(".macro {} v_q, v_n, v_d, v_tmp4, s_tmp4".format(self.name())):
            self._emit("v_cvt_f32_u32     v[\\v_tmp4+0],   v[\\v_d]")
            self._emit("v_rcp_f32         v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_f32         v[\\v_tmp4+0],   0x4f800000, v[\\v_tmp4+0]")
            self._emit("v_cvt_u32_f32     v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   v[\\v_d],      v[\\v_tmp4+0]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+2],   v[\\v_d],      v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+3],   vcc, 0,     v[\\v_tmp4+1]")
            self._emit("v_cmp_ne_i32      s[\\s_tmp4:\\s_tmp4+1], 0,          v[\\v_tmp4+2]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+1],   v[\\v_tmp4+3],   v[\\v_tmp4+1],   s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+1],   v[\\v_tmp4+1],   v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_add_co_u32      v[\\v_tmp4+0],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_tmp4+2],   s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_n]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   v[\\v_tmp4+0],   v[\\v_d]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_ge_u32      s[\\s_tmp4:\\s_tmp4+1], v[\\v_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_ge_u32      s[\\s_tmp4+2:\\s_tmp4+3], v[\\v_tmp4+2],   v[\\v_d]")
            self._emit("v_add_co_u32      v[\\v_tmp4+2],   vcc, 1, v[\\v_tmp4+0]")
            self._emit("s_and_b64         s[\\s_tmp4+2:\\s_tmp4+3], s[\\s_tmp4:\\s_tmp4+1], s[\\s_tmp4+2:\\s_tmp4+3]")
            self._emit("v_add_co_u32      v[\\v_tmp4+1],   vcc, -1,    v[\\v_tmp4+0]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+0],   v[\\v_tmp4+2],      s[\\s_tmp4+2:\\s_tmp4+3]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+1],   v[\\v_tmp4+2],      s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_cmp_ne_i32      vcc,          0,          v[\\v_d]")
            self._emit("v_cndmask_b32     v[\\v_q],      -1,         v[\\v_tmp4+2],      vcc")

class macro_int_div_vs_t(mc_base_t):
    def name(self):
        return '.v_u32_div_vs'
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, v_q, v_n, s_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}'.format(self.name(), v_q, v_n, s_d, v_tmp4, s_tmp4)
    def emit(self):
        with self._emit_macro_indented(".macro {} v_q, v_n, s_d, v_tmp4, s_tmp4".format(self.name())):
            self._emit("v_cvt_f32_u32     v[\\v_tmp4+0],   s[\\s_d]")
            self._emit("v_rcp_f32         v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_f32         v[\\v_tmp4+0],   0x4f800000, v[\\v_tmp4+0]")
            self._emit("v_cvt_u32_f32     v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],      v[\\v_tmp4+0]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+2],   s[\\s_d],      v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+3],   vcc, 0,     v[\\v_tmp4+1]")
            self._emit("v_cmp_ne_i32      s[\\s_tmp4:\\s_tmp4+1], 0,          v[\\v_tmp4+2]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+1],   v[\\v_tmp4+3],   v[\\v_tmp4+1],   s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+1],   v[\\v_tmp4+1],   v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_add_co_u32      v[\\v_tmp4+0],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_tmp4+2],   s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_n]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],     v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_ge_u32      s[\\s_tmp4:\\s_tmp4+1], v[\\v_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_le_u32      s[\\s_tmp4+2:\\s_tmp4+3],  s[\\s_d],    v[\\v_tmp4+2]")
            self._emit("v_add_co_u32      v[\\v_tmp4+2],   vcc, 1, v[\\v_tmp4+0]")
            self._emit("s_and_b64         s[\\s_tmp4+2:\\s_tmp4+3], s[\\s_tmp4:\\s_tmp4+1], s[\\s_tmp4+2:\\s_tmp4+3]")
            self._emit("v_add_co_u32      v[\\v_tmp4+1],   vcc, -1,    v[\\v_tmp4+0]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+0],   v[\\v_tmp4+2],      s[\\s_tmp4+2:\\s_tmp4+3]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+1],   v[\\v_tmp4+2],      s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_cmp_ne_i32      vcc,          s[\\s_d],   0")
            self._emit("v_cndmask_b32     v[\\v_q],      -1,         v[\\v_tmp4+2],      vcc")

class macro_int_div_ss_t(mc_base_t):
    def name(self):
        return '.v_u32_div_ss'
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, v_q, s_n, s_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}'.format(self.name(), v_q, s_n, s_d, v_tmp4, s_tmp4)
    def emit(self):
        with self._emit_macro_indented(".macro .v_u32_div_ss v_q, s_n, s_d, v_tmp4, s_tmp4"):
            self._emit("v_cvt_f32_u32     v[\\v_tmp4+0],   s[\\s_d]")
            self._emit("v_rcp_f32         v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_f32         v[\\v_tmp4+0],   0x4f800000, v[\\v_tmp4+0]")
            self._emit("v_cvt_u32_f32     v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],      v[\\v_tmp4+0]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+2],   s[\\s_d],      v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+3],   vcc, 0,     v[\\v_tmp4+1]")
            self._emit("v_cmp_ne_i32      s[\\s_tmp4:\\s_tmp4+1], 0,          v[\\v_tmp4+2]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+1],   v[\\v_tmp4+3],   v[\\v_tmp4+1],   s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+1],   v[\\v_tmp4+1],   v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_add_co_u32      v[\\v_tmp4+0],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_tmp4+2],   s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+0],   s[\\s_n],   v[\\v_tmp4+0]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],     v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        s[\\s_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_ge_u32      s[\\s_tmp4:\\s_tmp4+1], s[\\s_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_le_u32      s[\\s_tmp4+2:\\s_tmp4+3],  s[\\s_d],    v[\\v_tmp4+2]")
            self._emit("v_add_co_u32      v[\\v_tmp4+2],   vcc, 1, v[\\v_tmp4+0]")
            self._emit("s_and_b64         s[\\s_tmp4+2:\\s_tmp4+3], s[\\s_tmp4:\\s_tmp4+1], s[\\s_tmp4+2:\\s_tmp4+3]")
            self._emit("v_add_co_u32      v[\\v_tmp4+1],   vcc, -1,    v[\\v_tmp4+0]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+0],   v[\\v_tmp4+2],      s[\\s_tmp4+2:\\s_tmp4+3]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+1],   v[\\v_tmp4+2],      s[\\s_tmp4:\\s_tmp4+1]")
            self._emit("v_cmp_ne_i32      vcc,          s[\\s_d],   0")
            self._emit("v_cndmask_b32     v[\\v_q],      -1,         v[\\v_tmp4+2],      vcc")

class macro_c_clear_t(mc_base_t):
    def name(self):
        return '.v_clear_nc'
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, vid, num):
        return '{} {}, {}'.format(self.name(), vid, num)
    def emit(self):
        with self._emit_macro_indented(".macro {} vid, num".format(self.name())):
            self._emit("_v = \\vid")
            self._emit(".rept \\num")
            with self._indent_context():
                self._emit("v_mov_b32 v[_v], 0")
                self._emit("_v = _v + 1")
            self._emit(".endr")


class emit_write_4d_strided_t(mc_base_t):
    def name(self):
        return '.v_write4d_strided'
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, v_src, s_p_dst, v_dst_os, s_dst_diff1d, s_dst_diff2d, s_dst_diff3d, s_dst_diff4d,
                s_dst_os_4, t_dim_1d, t_dim_2d, t_dim_3d, t_dim_4d):
        return '{} {},{},{},{},{},{},{},{},{},{},{},{}'.format(self.name(),
            v_src, s_p_dst, v_dst_os, s_dst_diff1d, s_dst_diff2d, s_dst_diff3d, s_dst_diff4d,
            s_dst_os_4, t_dim_1d, t_dim_2d, t_dim_3d, t_dim_4d)
    def emit(self):
        def emit_write1d_strided():
            self._emit('; write 1d tensor to global with stride')
            with self._emit_macro_indented('.macro .v_write1d_strided v_src, s_p_buf_dst, v_dst_os, s_dst_diff, s_dst_os, t_dim_1d'):
                self._emit('.itr_1d = 0')
                self._emit('.rept \\t_dim_1d')
                self._inc_indent()
                self._emit('buffer_store_dword v[\\v_src+.itr_1d], v[\\v_dst_os], s[\\s_p_buf_dst:\\s_p_buf_dst+3], s[\\s_dst_os] offen')
                self._emit('.if .itr_1d != \\t_dim_1d - 1')
                self._inc_indent()
                self._emit('s_add_u32 s[\\s_dst_os], s[\\s_dst_os], s[\\s_dst_diff]')
                self._dec_indent()
                self._emit('.endif')
                self._emit('.itr_1d = .itr_1d + 1')
                self._dec_indent()
                self._emit('.endr')

        def emit_write2d_strided():
            self._emit('; write 2d tensor to global with stride')
            with self._emit_macro_indented('.macro .v_write2d_strided v_src, s_p_dst, v_dst_os, s_dst_diff1d, s_dst_diff2d, s_dst_os_2, t_dim_1d, t_dim_2d'):
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
            with self._emit_macro_indented('.macro .v_write3d_strided v_src, s_p_dst, v_dst_os, s_dst_diff1d, s_dst_diff2d, s_dst_diff3d, s_dst_os_3, t_dim_1d, t_dim_2d, t_dim_3d'):
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
            with self._emit_macro_indented('.macro .v_write4d_strided v_src, s_p_dst, v_dst_os, s_dst_diff1d, s_dst_diff2d, s_dst_diff3d, s_dst_diff4d, s_dst_os_4, t_dim_1d, t_dim_2d, t_dim_3d, t_dim_4d'):
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

class amdgpu_swap_sequencer_t(object):
    '''
    partial-transpose 2d matrix in register, by using swap.
    currently only consider continus register in same col, aka col major

    after transpose, the num of col/row should be the same

    And be aware that, this method still is not straight-forward and not optimal,
    for v_swap_b32 have half speed. In this case better use several tmp register serve as vector buffer
    Hopefully in the future could have full speed v_swap_b32

        k0 k1 k2 k3          k0 k1 k2 k3
    e0 0  2  4  6    =>  e0 0  1  2  3
    e1 1  3  5  7        e1 4  5  6  7

        k0 k1 k2 k3         k0 k1 k2 k3
    e0  0  4  8  c       e0 0  1  2  3
    e1  1  5  9  d   =>  e1 4  5  6  7
    e2  2  6  a  e       e2 8  9  a  b
    e3  3  7  b  f       e3 c  d  e  f
    '''
    def create_2d_swap(self):
        def init_2d_indice(row, col):
            indice_2d = []
            for r in range(row):
                indice_2d.append([r+c*row for c in range(col)])
            return indice_2d
        def check_row_can_omit_swap(indice_2d, cur_row):
            '''
            if current row already fit in vector pattern, can omit out
            '''
            row = len(indice_2d)
            col = len(indice_2d[0])
            targeting_vector_pattern = []
            for c in range(col):
                targeting_vector_pattern.append(c)
            vector_diff = []
            for c in range(col):
                vector_diff.append(abs(indice_2d[cur_row][c] - targeting_vector_pattern[c]))
            lasf_diff = vector_diff[0]
            #print('xxx {}'.format(vector_diff))
            if lasf_diff % 2 != 0:
                return False
            for c in range(1, col):
                if lasf_diff != vector_diff[c]:
                    return False
            return True
        def scan_2d_indice(indice_2d):
            def locate_indice(indice_2d, target_indice, start_row):
                row = len(indice_2d)
                col = len(indice_2d[0])
                (tr, tc) = (start_row, 0)
                found = False
                for tr in range(start_row, row):
                    for tc in range(0, col):
                        #print(target_indice, indice_2d[tr][tc])
                        if target_indice == indice_2d[tr][tc]:
                            found = True
                            break
                    if found:
                        break
                assert found
                return (tr, tc)
            swap_list = []
            row = len(indice_2d)
            col = len(indice_2d[0])

            class touch_row_t(object):
                def __init__(self, row):
                    self.row = row
                    self.row_touched = [ 0 for r in range(row)]
                    self.row_touched_index = 0
                def next_untouched_row(self):
                    for r in range(self.row_touched_index, self.row):
                        if self.row_touched[r] == 0:
                            self.row_touched_index = r
                            return r
                    assert False
                def touch(self, row_index):
                    self.row_touched[row_index] = 1
            touch_row = touch_row_t(row)
            for r in range(row):
                if check_row_can_omit_swap(indice_2d, r):
                    swap_list.append('unified for row {}'.format(r))
                    touch_row.touch( indice_2d[r][0] // col)
                    continue
                swap_list_per_row = []
                for c in range(col):
                    target_indice = touch_row.next_untouched_row()*col + c
                    origin_indice = indice_2d[r][c]
                    if origin_indice == target_indice:
                        continue
                    #print('to find:{}'.format(target_indice))
                    (tr, tc) = locate_indice(indice_2d, target_indice, r)
                    # swap and record indice
                    indice_2d[tr][tc] = origin_indice
                    indice_2d[r][c] = target_indice
                    #print('swapper:{}'.format(indice_2d))
                    swap_list_per_row.append((origin_indice, target_indice))
                swap_list.append(swap_list_per_row)
                touch_row.touch(r)
            return swap_list
        indice_2d = init_2d_indice(self.row, self.col)
        #print(indice_2d)
        swap_list = scan_2d_indice(indice_2d)
        return swap_list

    def __init__(self, row, col):
        assert col != 1 and row != 1
        self.col = col
        self.row = row
        self.swap_list = self.create_2d_swap()

    def __call__(self):
        '''
        return list of tuple of the row row_idx what swap should take
        '''
        return self.swap_list

class gpr_sequencer_t(object):
    def __init__(self):
        self.cnt = 0
    def __call__(self, step = 0, alignment = 0):
        previous_cnt = self.cnt
        if alignment:
            aligned_cnt = ((previous_cnt + alignment - 1) // alignment) * alignment
            self.cnt = aligned_cnt
            previous_cnt = aligned_cnt
        self.cnt += step
        return previous_cnt
    def get(self):
        return self.cnt

class sym_t(object):
    '''
    symbol used in asm source, can use '.set <label>,   <value>'
    '''
    def __init__(self, label, value = 0):
        assert type(label) is str
        assert type(value) is int
        self.label = label
        self.value = value
    def declare(self):
        return f'.set {self.label}, {self.value}'
    @staticmethod
    def expr(label, index = 0):
        if type(index) is int:
            if index == 0:
                return label
            return f'{label}+{index}'
        elif type(index) is tuple:
            assert len(index) == 2, 'expect tuple (start-index, end-index), inclusive'
            return f'{label}+{index[0]}:{label}+{index[1]}'
        else:
            assert False

    def __call__(self, index = 0):
        return self.expr(self.label, index)

class msym_t(object):
    """ reference a symbol inside macro """
    def __init__(self, sym):
        assert type(sym) is sym_t
        self.sym = sym
        self.label_in_macro = f'\{sym.label}'

    def __call__(self, index = 0):
        return self.sym.expr(self.label_in_macro, index)





#class gpr_t(object):
#    '''
#    index-ed gpr symbol type, with label as var. not distinguish sgpr/vgpr or others
#    '''
#    def __init__(self, var):
#        assert type(var) is str
#        self.var = var
#    
#    @staticmethod
#    def expr(label, index = 0):
#        if type(index) is int:
#            if index == 0:
#                return label
#            return label + f'+{index}'
#        elif type(index) is tuple:
#            assert len(index) == 2, 'expect tuple (start-index, end-index), inclusive'
#            return f'{label}+{index[0]}:{label}+{index[1]}'
#        else:
#            assert False
#
#    def __call__(self, index = 0):
#        return self.expr(self.var, index)
#
#class mgpr_t(object):
#    """ macro-contexted-gpr label, used within macro"""
#    def __init__(self, gpr):
#        assert type(gpr) is gpr_t
#        self.gpr = gpr
#        self.macro_var = f'\\{gpr.var}'
#
#    def __call__(self, index = 0):
#        return self.gpr.expr(self.macro_var, index)
#
#
#class vgpr_t(object):
#    def __init__(self, gpr):
#        assert type(gpr) is gpr_t
#        self.gpr = gpr
#    def __call__(self, index = 0):
#        return 'v[{}]'.format(gpr(index))
#
#class sgpr_t(object):
#    def __init__(self, gpr):
#        assert type(gpr) is gpr_t
#        self.gpr = gpr
#    def __call__(self, index = 0):
#        return 's[{}]'.format(gpr(index))

class fma_inst_t(object):
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

class ds_read_t(object):
    def __init__(self, bytes):
        self.bytes = bytes
    def get_offset(self, offset):
        return '' if offset == 0 else 'offset:{}'.format(offset)
    def __call__(self, vdst, vaddr, offset):
        if self.bytes == 4:
            return 'ds_read_b32 v[{}], v[{}] {}'.format(vdst, vaddr, self.get_offset(offset))
        if self.bytes == 8:
            return 'ds_read_b64 v[{}:{}+1], v[{}] {}'.format(vdst, vdst, vaddr, self.get_offset(offset))
        if self.bytes == 12:
            return 'ds_read_b96 v[{}:{}+2], v[{}] {}'.format(vdst, vdst, vaddr, self.get_offset(offset))
        if self.bytes == 16:
            return 'ds_read_b128 v[{}:{}+3], v[{}] {}'.format(vdst, vdst, vaddr, self.get_offset(offset))
        assert False

class ds_write_t(object):
    def __init__(self, bytes):
        self.bytes = bytes

    def get_offset(self, offset):
        if type(offset) is str:
            return 'offset:{}'.format(offset)
        if type(offset) is int:
            return '' if offset == 0 else 'offset:{}'.format(offset)
        assert False

    def __call__(self, vaddr, vdata, offset):
        if self.bytes == 4:
            return 'ds_write_b32 v[{}], v[{}] {}'.format(vaddr, vdata, self.get_offset(offset))
        if self.bytes == 8:
            return 'ds_write_b64 v[{}], v[{}:{}+1] {}'.format(vaddr, vdata, vdata, self.get_offset(offset))
        if self.bytes == 12:
            return 'ds_write_b96 v[{}], v[{}:{}+2] {}'.format(vaddr, vdata, vdata, self.get_offset(offset))
        if self.bytes == 16:
            return 'ds_write_b128 v[{}], v[{}:{}+3] {}'.format(vaddr, vdata, vdata, self.get_offset(offset))
        assert False

class buffer_load_dword_t(object):
    ''' TODO: this implementation always offen '''
    def __init__(self, dwords):
        self.dwords = dwords

    def __call__(self, vdst, vaddr, srsrc, soffset, offset):
        if type(soffset) is int and soffset == 0:
            soffset_str = "0"
        else:
            soffset_str = f"s[{soffset}]"

        if self.dwords == 1:
            return f"buffer_load_dword v[{vdst}], v[{vaddr}], s[{srsrc}:{srsrc}+3], {soffset_str} offen offset:{offset}"
        if self.dwords == 2:
            return f"buffer_load_dwordx2 v[{vdst}:{vdst}+1], v[{vaddr}], s[{srsrc}:{srsrc}+3], {soffset_str} offen offset:{offset}"
        if self.dwords == 3:
            return f"buffer_load_dwordx3 v[{vdst}:{vdst}+2], v[{vaddr}], s[{srsrc}:{srsrc}+3], {soffset_str} offen offset:{offset}"
        if self.dwords == 4:
            return f"buffer_load_dwordx4 v[{vdst}:{vdst}+3], v[{vaddr}], s[{srsrc}:{srsrc}+3], {soffset_str} offen offset:{offset}"
        assert False

class buffer_store_dword_t(object):
    ''' TODO: this implementation always offen '''
    def __init__(self, dwords):
        self.dwords = dwords

    def __call__(self, vdata, vaddr, srsrc, soffset, offset):
        if type(soffset) is int and soffset == 0:
            soffset_str = "0"
        else:
            soffset_str = f"s[{soffset}]"

        if self.dwords == 1:
            return f"buffer_store_dword v[{vdata}], v[{vaddr}], s[{srsrc}:{srsrc}+3], {soffset_str} offen offset:{offset}"
        if self.dwords == 2:
            return f"buffer_store_dwordx2 v[{vdata}:{vdata}+1], v[{vaddr}], s[{srsrc}:{srsrc}+3], {soffset_str} offen offset:{offset}"
        if self.dwords == 3:
            return f"buffer_store_dwordx3 v[{vdata}:{vdata}+2], v[{vaddr}], s[{srsrc}:{srsrc}+3], {soffset_str} offen offset:{offset}"
        if self.dwords == 4:
            return f"buffer_store_dwordx4 v[{vdata}:{vdata}+3], v[{vaddr}], s[{srsrc}:{srsrc}+3], {soffset_str} offen offset:{offset}"
        assert False
