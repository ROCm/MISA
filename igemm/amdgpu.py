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
import os
import subprocess
from .codegen import *
AMDGPU_PRECISION_FP32   = (0 << 20)
AMDGPU_PRECISION_FP16   = (1 << 20)
AMDGPU_PRECISION_BF16   = (2 << 20)
AMDGPU_ARCH_GFX900      = (0 << 24)
AMDGPU_ARCH_GFX906      = (1 << 24)
AMDGPU_ARCH_GFX908      = (2 << 24)
AMDGPU_CODEOBJECT_V2    = (0 << 28)
AMDGPU_CODEOBJECT_V3    = (1 << 28)

def amdgpu_string_to_arch(amdgpu_arch_string):
    if amdgpu_arch_string == 'gfx900':
        return AMDGPU_ARCH_GFX900
    if amdgpu_arch_string == 'gfx906':
        return AMDGPU_ARCH_GFX906
    if amdgpu_arch_string == 'gfx908':
        return AMDGPU_ARCH_GFX908
    assert False

def amdgpu_arch_to_string(amdgpu_arch_gfxxxx):
    if amdgpu_arch_gfxxxx == AMDGPU_ARCH_GFX900:
        return 'gfx900'
    if amdgpu_arch_gfxxxx == AMDGPU_ARCH_GFX906:
        return 'gfx906'
    if amdgpu_arch_gfxxxx == AMDGPU_ARCH_GFX908:
        return 'gfx908'
    assert False

def amdgpu_string_to_codeobj(amdgpu_codeobj_string):
    if amdgpu_codeobj_string == 'cov2':
        return AMDGPU_CODEOBJECT_V2
    if amdgpu_codeobj_string == 'cov3':
        return AMDGPU_CODEOBJECT_V3
    assert False

def amdgpu_codeobj_to_string(amdgpu_codeobj):
    if amdgpu_codeobj == AMDGPU_CODEOBJECT_V2:
        return 'cov2'
    if amdgpu_codeobj == AMDGPU_CODEOBJECT_V3:
        return 'cov3'
    assert False

def amdgpu_precision_to_string(amdgpu_precision):
    if amdgpu_precision == AMDGPU_PRECISION_FP32:
        return 'fp32'
    if amdgpu_precision == AMDGPU_PRECISION_FP16:
        return 'fp16'
    if amdgpu_precision == AMDGPU_PRECISION_BF16:
        return 'bf16'
    assert False

def amdgpu_string_to_precision(amdgpu_precision_string):
    if amdgpu_precision_string == 'fp32':
        return AMDGPU_PRECISION_FP32
    if amdgpu_precision_string == 'fp16':
        return AMDGPU_PRECISION_FP16
    if amdgpu_precision_string == 'bf16':
        return AMDGPU_PRECISION_BF16
    assert False

def amdgpu_precision_data_byte(precision):
    if type(precision) is str:
        precision = amdgpu_precision_to_string(precision)
    if precision == AMDGPU_PRECISION_FP32:
        return 4
    if precision == AMDGPU_PRECISION_FP16:
        return 2
    if precision == AMDGPU_PRECISION_BF16:
        return 2
    assert False

class amdgpu_arch_detail_t(object):
    '''
    probe or hard code
    '''
    def __init__(self):
        self.arch           = 0
        self.num_cu         = 0
        self.simd_per_cu    = 0
        self.sclk_mhz       = 0
        self.mclk_mhz       = 0
        self.lds_size       = 0     # in byte
        self.lds_banks      = 0
        self.l1_size        = 0
        self.l1_cache_line  = 0
        self.l2_size        = 0
        self.l2_cache_line  = 0
        self.mem_channels   = 0
        self.vgpr_per_cu    = 0
        self.sgpr_per_cu    = 0
        self.agpr_per_cu    = 0
        self.wavefront_size = 64
        self.max_waves_per_cu       = 0
        self.fp32_fma_per_cycle     = 0
        self.memory_op_per_cycle    = 0     # read write
        self.memory_bus_width_bits  = 0    

    def theoretical_fp32_gflops(self):
        return self.num_cu * self.simd_per_cu * (self.sclk_mhz / 1000) * self.fp32_fma_per_cycle

    def theoretical_bandwidth_gbps(self):
        return (self.mclk_mhz / 1000) * (self.memory_bus_width_bits / 8) * self.memory_op_per_cycle

def amdgpu_calculate_occupancy(arch_detail, vgpr_per_thread, block_size, lds_per_block):
    vgpr_per_block = vgpr_per_thread * block_size
    if vgpr_per_block > arch_detail.vgpr_per_cu:
        print('vgpr required:{} is larger than hw vgpr:{}'.format(vgpr_per_block, arch_detail.vgpr_per_cu))
        return 0
    blocks_consider_vgpr = arch_detail.vgpr_per_cu // vgpr_per_block
    if lds_per_block > arch_detail.lds_size:
        print('lds required:{} is larger than hw vgpr:{}'.format(lds_per_block, arch_detail.lds_size))
        return 0
    blocks_consider_lds = arch_detail.lds_size // lds_per_block

    return min(blocks_consider_vgpr, blocks_consider_lds)

def amdgpu_valid_occupancy_with_max_waves(arch_detail, block_size, occupancy):
    assert block_size >= arch_detail.wavefront_size and \
            block_size % arch_detail.wavefront_size == 0
    waves_per_block = block_size // arch_detail.wavefront_size
    return waves_per_block * occupancy <= arch_detail.max_waves_per_cu

def get_amdgpu_gfx906_60cu():
    gfx906_60cu = amdgpu_arch_detail_t()
    gfx906_60cu.arch            = AMDGPU_ARCH_GFX906
    gfx906_60cu.num_cu          = 60
    gfx906_60cu.simd_per_cu     = 64
    gfx906_60cu.sclk_mhz        = 1725
    gfx906_60cu.mclk_mhz        = 1000
    gfx906_60cu.lds_size        = 65536
    gfx906_60cu.lds_banks       = 32
    gfx906_60cu.l1_size         = 16384
    gfx906_60cu.l2_size         = 0
    gfx906_60cu.mem_channels    = 0
    gfx906_60cu.vgpr_per_cu     = 65536
    gfx906_60cu.sgpr_per_cu     = 3200
    gfx906_60cu.agpr_per_cu     = 0
    gfx906_60cu.wavefront_size      = 64
    gfx906_60cu.max_waves_per_cu    = 40
    gfx906_60cu.fp32_fma_per_cycle  = 2
    gfx906_60cu.memory_op_per_cycle = 2     # read write
    gfx906_60cu.memory_bus_width_bits = 4096
    return gfx906_60cu

class amdgpu_arch_config_t(object):
    '''
    config some of arch related feature
    '''
    def __init__(self, arch_dict):
        ad = codegen_dict_with_default_t(arch_dict)
        self.arch           = ad('arch', AMDGPU_ARCH_GFX906)
        self.use_dlops      = ad('use_dlops', False)
        self.use_xdlops     = ad('use_sdlops', False)
        self.data_type      = ad('data_type', AMDGPU_PRECISION_FP32)
        self.code_object    = ad('code_object', AMDGPU_CODEOBJECT_V3)

class amdgpu_kernel_code_t(object):
    '''
    reuse this for both cov2 and cov3
    .amd_kernel_code_t
    	amd_code_version_major = 1
		amd_code_version_minor = 2
		amd_machine_kind = 1
		amd_machine_version_major = 9
		amd_machine_version_minor = 0
		amd_machine_version_stepping = 6
		kernel_code_entry_byte_offset = 256
		kernel_code_prefetch_byte_size = 0
		granulated_workitem_vgpr_count = 28
		granulated_wavefront_sgpr_count = 4
		priority = 0
		float_mode = 192
		priv = 0
		enable_dx10_clamp = 1
		debug_mode = 0
		enable_ieee_mode = 1
		enable_wgp_mode = 0
		enable_mem_ordered = 0
		enable_fwd_progress = 0
		enable_sgpr_private_segment_wave_byte_offset = 0
		user_sgpr_count = 6
		enable_trap_handler = 0
		enable_sgpr_workgroup_id_x = 1
		enable_sgpr_workgroup_id_y = 0
		enable_sgpr_workgroup_id_z = 0
		enable_sgpr_workgroup_info = 0
		enable_vgpr_workitem_id = 0
		enable_exception_msb = 0
		granulated_lds_size = 0
		enable_exception = 0
		enable_sgpr_private_segment_buffer = 1
		enable_sgpr_dispatch_ptr = 0
		enable_sgpr_queue_ptr = 0
		enable_sgpr_kernarg_segment_ptr = 1
		enable_sgpr_dispatch_id = 0
		enable_sgpr_flat_scratch_init = 0
		enable_sgpr_private_segment_size = 0
		enable_sgpr_grid_workgroup_count_x = 0
		enable_sgpr_grid_workgroup_count_y = 0
		enable_sgpr_grid_workgroup_count_z = 0
		; enable_wavefront_size32 = 0
		enable_ordered_append_gds = 0
		private_element_size = 1
		is_ptr64 = 1
		is_dynamic_callstack = 0
		is_debug_enabled = 0
		is_xnack_enabled = 0
		workitem_private_segment_byte_size = 0
		workgroup_group_segment_byte_size = 32768
		gds_segment_byte_size = 0
		kernarg_segment_byte_size = 32
		workgroup_fbarrier_count = 0
		wavefront_sgpr_count = 35
		workitem_vgpr_count = 115
		reserved_vgpr_first = 0
		reserved_vgpr_count = 0
		reserved_sgpr_first = 0
		reserved_sgpr_count = 0
		debug_wavefront_private_segment_offset_sgpr = 0
		debug_private_segment_buffer_sgpr = 0
		kernarg_segment_alignment = 4
		group_segment_alignment = 4
		private_segment_alignment = 4
		wavefront_size = 6
		call_convention = -1
		runtime_loader_kernel_symbol = 0
    .end_amd_kernel_code_t
    '''
    def __init__(self, kernel_code_dict):
        kc = codegen_dict_with_default_t(kernel_code_dict)
        self.enable_sgpr_private_segment_buffer     = kc('enable_sgpr_private_segment_buffer', 0)
        self.enable_sgpr_dispatch_ptr               = kc('enable_sgpr_dispatch_ptr', 0)
        self.enable_sgpr_queue_ptr                  = kc('enable_sgpr_queue_ptr', 0)
        self.enable_sgpr_kernarg_segment_ptr        = kc('enable_sgpr_kernarg_segment_ptr', 1)
        self.enable_sgpr_dispatch_id                = kc('enable_sgpr_dispatch_id', 0)
        # other sgpr related to be implemented
        self.user_sgpr_count                        = self.cal_user_sgpr_count()

        self.enable_sgpr_workgroup_id_x             = kc('enable_sgpr_workgroup_id_x', 1)
        self.enable_sgpr_workgroup_id_y             = kc('enable_sgpr_workgroup_id_y', 0)
        self.enable_sgpr_workgroup_id_z             = kc('enable_sgpr_workgroup_id_z', 0)
        self.enable_vgpr_workitem_id                = kc('enable_vgpr_workitem_id', 0)

        self.float_mode                             = kc('float_mode', 192)

        self.is_ptr64                               = kc('is_ptr64', 1)
        self.workgroup_group_segment_byte_size      = kc('workgroup_group_segment_byte_size', 0)
        self.wavefront_sgpr_count                   = kc('wavefront_sgpr_count', 0)
        self.workitem_vgpr_count                    = kc('workitem_vgpr_count', 0)

        if type(self.workitem_vgpr_count) is str and self.workitem_vgpr_count == 'v_end':
            self.granulated_workitem_vgpr_count     = '(v_end-1)/4'
        else:
            self.granulated_workitem_vgpr_count     = self.cal_granulated_workitem_vgpr_count()

        # VCC, FLAT_SCRATCH and XNACK must be counted
        if type(self.wavefront_sgpr_count) is str and self.wavefront_sgpr_count == 's_end+2*3':
            self.granulated_wavefront_sgpr_count    = '(s_end+2*3-1)/8'
        else:
            self.granulated_wavefront_sgpr_count    = self.cal_granulated_wavefront_sgpr_count()
        self.kernarg_segment_byte_size              = kc('kernarg_segment_byte_size', 0)

    def cal_user_sgpr_count(self):
        count = 0
        if self.enable_sgpr_private_segment_buffer:
            count += 4
        if self.enable_sgpr_dispatch_ptr:
            count += 2
        if self.enable_sgpr_queue_ptr:
            count += 2
        if self.enable_sgpr_kernarg_segment_ptr:
            count += 2
        if self.enable_sgpr_dispatch_id:
            count += 2
        # other sgpr related to be implemented

        return count

    def cal_granulated_workitem_vgpr_count(self):
        '''
        (workitem_vgpr_count-1)/4
        '''
        return (self.workitem_vgpr_count - 1) // 4

    def cal_granulated_wavefront_sgpr_count(self):
        '''
        (wavefront_sgpr_count-1)/8
        '''
        return (self.wavefront_sgpr_count - 1) // 8

class amdgpu_kernel_arg_t(object):
    '''
    http://llvm.org/docs/AMDGPUUsage.html#code-object-v3-metadata-mattr-code-object-v3
    '''
    def __init__(self, name, size, offset, value_kind, value_type, **misc):
        self.name = name
        self.size = size
        self.offset = offset
        self.value_kind = value_kind
        self.value_type = value_type
        self.misc = misc
    def serialize_as_metadata(self):
        misc_metadata = ''
        if self.value_kind == 'global_buffer':
            assert self.misc
            misc_metadata += ', .address_space: {}'.format(self.misc['address_space'])
            misc_metadata += ', .is_const: {}'.format(self.misc['is_const'])
        return '    - {{ .name: {:<10}, .size: {}, .offset: {:>3}, .value_kind: {}, .value_type: {}{}}}'.format(
            self.name, self.size, self.offset, self.value_kind, self.value_type, misc_metadata)

class amdgpu_kernel_info_t(object):
    def __init__(self, kernel_code, kernel_name, kernel_block_size, kernel_args):
        self.kernel_code = kernel_code
        self.kernel_name = kernel_name
        self.kernel_block_size = kernel_block_size
        self.kernel_args = kernel_args

class amdgpu_asm_utils_t(object):
    def __init__(self, mc):
        self.mc = mc
        mc.inject(self)

class emit_hsa_header_t(amdgpu_asm_utils_t):
    '''
    only used in cov2
    '''
    def __init__(self, mc):
        amdgpu_asm_utils_t.__init__(self, mc)
    def emit(self):
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V2:
            self._emit(".hsa_code_object_version 2,1")
            self._emit(".hsa_code_object_isa")
            self._emit_empty_line()

class emit_hsa_footer_t(amdgpu_asm_utils_t):
    def __init__(self, mc):
        amdgpu_asm_utils_t.__init__(self, mc)
    def emit(self):
        pass


class emit_amd_kernel_code_t(amdgpu_asm_utils_t):
    def __init__(self, mc, kernel_info):
        amdgpu_asm_utils_t.__init__(self, mc)
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

class emit_amd_metadata_t(amdgpu_asm_utils_t):
    '''
    only implement in cov3
    '''
    def __init__(self, mc, kernel_info):
        amdgpu_asm_utils_t.__init__(self, mc)
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

class emit_int_div_vv_t(amdgpu_asm_utils_t):
    def name(self):
        return '.v_u32_div'
    def __init__(self, mc):
        amdgpu_asm_utils_t.__init__(self, mc)
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

class emit_int_div_vs_t(amdgpu_asm_utils_t):
    def name(self):
        return '.v_u32_div_vs'
    def __init__(self, mc):
        amdgpu_asm_utils_t.__init__(self, mc)
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

class emit_int_div_ss_t(amdgpu_asm_utils_t):
    def name(self):
        return '.v_u32_div_ss'
    def __init__(self, mc):
        amdgpu_asm_utils_t.__init__(self, mc)
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

class emit_c_clear_t(amdgpu_asm_utils_t):
    def name(self):
        return '.v_clear_nc'
    def __init__(self, mc):
        amdgpu_asm_utils_t.__init__(self, mc)
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

class emit_write_4d_strided_t(amdgpu_asm_utils_t):
    def name(self):
        return '.v_write4d_strided'
    def __init__(self, mc):
        amdgpu_asm_utils_t.__init__(self, mc)
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
    def __call__(self, step = 0):
        previous_cnt = self.cnt
        self.cnt += step
        return previous_cnt
    def get(self):
        return self.cnt

class gpr_t(object):
    def __init__(self, var):
        assert type(var) is str
        self.var = var
    def __call__(self, index = 0):
        if index == 0:
            return self.var
        return self.var + '+{}'.format(index)

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

class emit_fma_mxn_t(amdgpu_asm_utils_t):
    def name(self):
        return '.v_fma_{}x{}{}'.format(self.m, self.n,
                '' if self.stride == 1 else '_s{}'.format(self.stride))
    def __init__(self, mc, m, n, stride):
        amdgpu_asm_utils_t.__init__(self, mc)
        self.m = m
        self.n = n
        self.stride = stride
        assert stride >= n
    def __call__(self, c, a, b):
        return '{} {},{},{}'.format(self.name(), c, a, b)
    def emit(self):
        fma = fma_inst_t(self.mc.arch_config)
        reg_a = gpr_t('\\a')
        reg_b = gpr_t('\\b')
        reg_c = gpr_t('\\c')
        with self._emit_macro_indented('.macro {} c, a, b'.format(self.name())):
            for idx_m in range(self.m):
                for idx_n in range(self.n):
                    self._emit(fma(reg_c(idx_m * self.stride + idx_n), reg_a(idx_m), reg_b(idx_n)))

def amdgpu_check_hip_clang():
    return os.path.exists('/opt/rocm/llvm/bin/clang++')

class amdgpu_build_asm_t(object):
    def __init__(self, mc, asm_file_name, target_hsaco = ''):
        self.asm_file_name = asm_file_name
        if target_hsaco == '':
            self.target_hsaco = os.path.splitext(asm_file_name)[0] + '.hsaco'
        else:
            self.target_hsaco = target_hsaco
        self.mc = mc
    def build(self, **kwargs):
        # make sure mc output is closed
        self.mc.close()

        arch_str = amdgpu_arch_to_string(self.mc.arch_config.arch)
        use_hip_clang = amdgpu_check_hip_clang()
        if use_hip_clang:
            cmd = ['/opt/rocm/llvm/bin/clang++']
        else:
            cmd = ['/opt/rocm/hcc/bin/clang']
        cmd += ['-x', 'assembler']
        cmd += ['-target', 'amdgcn--amdhsa']
        cmd += ['-mcpu={}'.format(arch_str)]
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V2:
            cmd += ['-mno-code-object-v3']
        # TODO: current compiler treat cov3 as default, so no need add extra flag
        cmd += ['{}'.format(self.asm_file_name)]
        cmd += ['-o', '{}'.format(self.target_hsaco)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.STDOUT)
        try:
            (out, _) = p.communicate()
            if p.returncode != 0:
                print('build fail:{}'.format(cmd))
                print('{}'.format(out.decode('utf-8')))
                return False
            return True
        except Exception as e:
            print('fail to run cmd:{}'.format(cmd))
            print('err:{}'.format(e))
            return False

class amdgpu_build_host_t(object):
    def __init__(self, arch_config, host_cpp, target_exec = ''):
        self.host_cpp = host_cpp
        if target_exec == '':
            if type(host_cpp) is str:
                self.target_exec = os.path.splitext(host_cpp)[0] + '.exe'
            elif type(host_cpp) is list:
                self.target_exec = os.path.splitext(host_cpp[0])[0] + '.exe'
            else:
                assert False
        else:
            self.target_exec = target_exec
        self.arch_config = arch_config
    def build(self, **kwargs):
        arch_str = amdgpu_arch_to_string(self.arch_config.arch)
        use_hip_clang = amdgpu_check_hip_clang()
        if use_hip_clang:
            cmd = ['g++']
            cmd += ['-D__HIP_PLATFORM_HCC__=','-I/opt/rocm/hip/include', '-I/opt/rocm/hcc/include', '-I/opt/rocm/hsa/include']
            cmd += ['-Wall','-O2', '-std=c++11']
            if 'cflags' in kwargs:
                cmd += kwargs['cflags']
            if 'cxxflags' in kwargs:
                cmd += kwargs['cxxflags']
            if type(self.host_cpp) is str:
                cmd += [self.host_cpp]
            elif type(self.host_cpp) is list:
                cmd += self.host_cpp     # for multiple files
            else:
                assert False
            cmd += ['-L/opt/rocm/lib', '-L/opt/rocm/lib64', '-Wl,-rpath=/opt/rocm/lib',
                    '-ldl', '-lm', '-lpthread',
                    '-Wl,--whole-archive', '-lamdhip64', '-lhsa-runtime64', '-lhsakmt', '-Wl,--no-whole-archive']
            cmd += ['-o', self.target_exec]
        else:
            cmd = ['g++']
            # from `/opt/rocm/bin/hipconfig --cpp_config`
            cmd += ['-D__HIP_PLATFORM_HCC__=','-I/opt/rocm/hip/include', '-I/opt/rocm/hcc/include', '-I/opt/rocm/hsa/include']
            cmd += ['-Wall','-O2', '-std=c++11']
            if 'cflags' in kwargs:
                cmd += kwargs['cflags']
            if 'cxxflags' in kwargs:
                cmd += kwargs['cxxflags']
            if type(self.host_cpp) is str:
                cmd += [self.host_cpp]
            elif type(self.host_cpp) is list:
                cmd += self.host_cpp     # for multiple files
            else:
                assert False
            cmd += ['-L/opt/rocm/hcc/lib', '-L/opt/rocm/lib', '-L/opt/rocm/lib64', '-Wl,-rpath=/opt/rocm/hcc/lib:/opt/rocm/lib',
                    '-ldl', '-lm', '-lpthread', '-lhc_am',
                    '-Wl,--whole-archive', '-lmcwamp', '-lhip_hcc', '-lhsa-runtime64', '-lhsakmt', '-Wl,--no-whole-archive']
            cmd += ['-o', self.target_exec]

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.STDOUT)
        try:
            (out, _) = p.communicate()
            if p.returncode != 0:
                print('build fail:{}'.format(cmd))
                print('{}'.format(out.decode('utf-8')))
                return False
            return True
        except Exception as e:
            print('fail to run cmd:{}'.format(cmd))
            print('err:{}'.format(e))
            return False