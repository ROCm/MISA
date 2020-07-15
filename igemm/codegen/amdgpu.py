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

AMDGPU_PRECISION_FP32   = (0 << 20)
AMDGPU_PRECISION_FP16   = (1 << 20)
AMDGPU_PRECISION_BF16   = (2 << 20)
AMDGPU_ARCH_GFX900      = (0 << 24)
AMDGPU_ARCH_GFX906      = (1 << 24)
AMDGPU_ARCH_GFX908      = (2 << 24)
AMDGPU_CODEOBJECT_V2    = (0 << 28)
AMDGPU_CODEOBJECT_V3    = (1 << 28)

class _dict_with_default_t(object):
    def __init__(self, d):
        self.d = d
    def __call__(self, key, default_value):
        if key in self.d:
            return self.d[key]
        return default_value

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
        precision = amdgpu_string_to_precision(precision)
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

def amdgpu_get_gfx906_60cu():
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
        ad = _dict_with_default_t(arch_dict)
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
        kc = _dict_with_default_t(kernel_code_dict)
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
