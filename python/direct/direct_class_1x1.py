from abc import ABC, abstractmethod
import enum
from typing import Union

from python.codegen.generator.allocator import onDemand_allocator
from python.codegen.generator.gpu_arch.gpu_data_types import *
from python.codegen.generator.gpu_arch.HW_components import HW_gfx9, sgpr_file_t, sgpr_hw_component, vgpr_file_t, vgpr_hw_component
from python.codegen.generator.gpu_arch.GFX10 import gfx10_instructions_caller
from python.codegen.generator.gpu_arch.gpu_instruct import gpu_instructions_caller_base

from python.codegen.generator.kernel_func import kernel_func, kernel_launcher, launcher_kernel, mfunc_func
from python.codegen.kernel_driver import base_config

from ..tools.config_parser import config_content_t
from ..codegen.mc import mc_base_t, mc_asm_printer_t
from ..codegen.generator.kernel_constructor import *



class DATA_TYPE(Enum):
    TYPE_FP32 = 'TYPE_FP32'
    TYPE_FP16 = 'TYPE_FP16'

class direct_1x1u_config(base_config):
    def __init__(self, config_content: config_content_t):
        super().__init__(config_content, '-navi')
        
        self.read_size = config_content.get_section('direct-navi')[0]['read_size']

        self.kernels = [conv_direct_1x1u]

class conv_direct_1x1u(kernel_constructor):

    def get_kernel_name(self):
        return 'conv_direct_1x1u'

    class kernel_karg_t(karg_file_t):
        '''Define kernel arguments. Used in _set_kernel_karg_t'''
        def __init__(self, mc) -> None:
            super().__init__(mc)
            pb_arg = self._pb_kernel_arg
            self.N = pb_arg('N', arg_kind.value, arg_type.I32, is_const='True')
            self.C = pb_arg('C', arg_kind.value, arg_type.I32, is_const='True')
            self.H = pb_arg('H', arg_kind.value, arg_type.I32, is_const='True')
            self.W = pb_arg('W', arg_kind.value, arg_type.I32, is_const='True')
            self.K = pb_arg('K', arg_kind.value, arg_type.I32, is_const='True')
            self.n_groups = pb_arg('n_groups', arg_kind.value, arg_type.I32, is_const='True')
            self.unused_0 = pb_arg('unused_0', arg_kind.value, arg_type.I32, is_const='True')
            self.unused_1 = pb_arg('unused_1', arg_kind.value, arg_type.I32, is_const='True')
            self.in_ptr_off  = pb_arg('in_ptr',  arg_kind.GlobBuffer, arg_type.F32, address_space='global', is_const='True')
            self.wei_ptr_off = pb_arg('wei_ptr', arg_kind.GlobBuffer, arg_type.F32, is_const='True')
            self.out_ptr_off = pb_arg('out_ptr', arg_kind.GlobBuffer, arg_type.F32)
            self.dbg_ptr_off = pb_arg('dbg_ptr_off', arg_kind.GlobBuffer, arg_type.F32)
            

    def _get_LDS_usage(self):
        return 0

    def _emit_kernel_body(self):
        from dataclasses import dataclass
        @dataclass
        class conv_params():
            batch_size:int = 64
            img_w = 8
            img_h = 8
            
            filter_hw:int = 1 * 1
            elements_in_dword:int = 1
            input_channels:int = 128
            output_channels:int = 32
            weights_layout:int = 0
            vec_c_in:int = 1
            vec_k_out:int = 1
            vec_c_filter:int = 1
            buf_type = DATA_TYPE.TYPE_FP32

            #filter_c_stride:int = (4 // elements_in_dword) * filter_hw
            #filter_c_stride=4
            #filter_k_stride:int = filter_c_stride * input_channels
            
            filter_k_stride=512 
            filter_c_stride=4 
            
            #input_w_stride = 4 // elements_in_dword
            #input_c_stride:int = (4 // elements_in_dword) * img_h * img_w
            #input_n_stride:int = input_c_stride * input_channels

            input_w_stride=4
            input_c_stride=256
            input_n_stride=32768 

            #output_w_stride = 4 // elements_in_dword
            #output_k_stride:int = (4 // elements_in_dword) * img_h * img_w
            #output_n_stride:int = output_k_stride * output_channels
            
            output_w_stride = 4
            output_k_stride=256
            output_n_stride=8192

            #input_buffer_size:int = input_n_stride * batch_size
            #filter_buffer_size:int = filter_k_stride * output_channels
            #output_buffer_size:int = 256
            input_buffer_size=2097152 
            filter_buffer_size=16384 
            output_buffer_size=524288


        @launcher_kernel
        def launch_k(self:kernel_launcher[gfx10_instructions_caller], karg:conv_direct_1x1u.kernel_karg_t):
            @dataclass
            class perf_params():
                    read_size = 4 # 1, 2, 3, 4
                    k_mult = 16 # 1..32 (preffer 4*n)
                    c_mult = 1 # 1, 2..32 (4*n) #TODO solve sgpr align isue
                    chunks_per_wave:int = 4 # 1..16
                    chunk_size = 16 # 1..64
                    balanced_n = 1 # 0..1 (deprecated)
                    balanced_chunk = 1 # 0..1 (deprecated)
                    n_mult = 2 # 1..8
                    waves_c_in_group = 1 # 1..8 (preffer 1..4)
                    waves_k_in_group = 1 # 1,2,4,8 (preffer 1,2,4,8)
                    lds_limit = 1024 // 8 #.MAX_LDS // 8,
                    disable_case_opt = 0

            class const_vars():
                def __init__(self, conv_p:conv_params, perf_p:perf_params) -> None:
                    
                    elements_in_dword = 1
                    self.output_dword_chunks_cnt = 1
                    self.input_dword_chunks_cnt = 1
                    self.maxU24 = 1 << 24
                    self.invalid_addr_lit = 0x7FFFFFFF

                    if( conv_p.vec_k_out > elements_in_dword):
                        assert(conv_p.vec_k_out == elements_in_dword)
                    else:
                        self.output_dword_chunks_cnt = elements_in_dword // conv_p.vec_k_out
                    

                    if(conv_p.vec_c_in > elements_in_dword):
                        assert(conv_p.vec_c_in == elements_in_dword)
                    else:
                        self.input_dword_chunks_cnt = elements_in_dword // conv_p.vec_c_in
                    
                    self.img_hw = conv_p.img_h * conv_p.img_w
                    img_hw_vec = (conv_p.img_h * conv_p.img_w + self.input_dword_chunks_cnt - 1) // self.input_dword_chunks_cnt
                    self.rem_hw_in  = (conv_p.img_h * conv_p.img_w) % self.input_dword_chunks_cnt


                    assert( perf_p.chunks_per_wave % self.output_dword_chunks_cnt == 0)
                    assert( perf_p.chunks_per_wave % self.input_dword_chunks_cnt == 0)

                    self.hi_input_channels = (conv_p.input_channels + conv_p.vec_c_in - 1) // conv_p.vec_c_in
                    self.hi_output_channels = (conv_p.output_channels + conv_p.vec_k_out - 1) // conv_p.vec_k_out

                    self.s_pack_instructions_available = 0
                    self.dot_instructions_available = 0
                    self.madmix_instructions_available = 0
                    self.fmamix_instructions_available = 0
                    #if (.option.machine_version_major == 9) && (.option.machine_version_minor == 0) && (.option.machine_version_stepping >= 6)
                    #    dot_instructions_available = 1
                    

                    n_per_gpr = 64 // perf_p.chunk_size
                    assert (n_per_gpr * perf_p.chunk_size == 64)
                    total_n_blocks = (conv_p.batch_size + n_per_gpr - 1) // n_per_gpr
                    
                    if perf_p.balanced_n:
                        self.active_n_per_gpr = (conv_p.batch_size + total_n_blocks - 1) // total_n_blocks
                    else:
                        self.active_n_per_gpr = n_per_gpr
                    
                    n_per_wave = perf_p.n_mult * n_per_gpr
                    self.active_n_per_wave = perf_p.n_mult * self.active_n_per_gpr

                    total_chunks = (self.img_hw + perf_p.chunk_size - 1) // perf_p.chunk_size
                    if total_chunks < perf_p.chunks_per_wave:
                        total_chunks = perf_p.chunks_per_wave

                    if perf_p.balanced_chunk:
                        active_chunk_lanes = (self.img_hw + total_chunks - 1) // total_chunks
                    else:
                        active_chunk_lanes = perf_p.chunk_size

                    hw_per_wave = perf_p.chunk_size * perf_p.chunks_per_wave
                    self.active_hw_per_wave = active_chunk_lanes * perf_p.chunks_per_wave

                    self.in_gprs = (perf_p.n_mult * perf_p.c_mult * perf_p.chunks_per_wave  + conv_p.elements_in_dword - 1) // conv_p.elements_in_dword

                    #since we use mix-precision, which accumulates fp16 into fp32, we need vec_size
                    #times fp16 registers for accumulation
                    accums_cnt = perf_p.k_mult * perf_p.chunks_per_wave * perf_p.n_mult
                    self.accums_cnt = accums_cnt

                    # exec mask
                    import math
                    self.chunk_size_log2:int = int(math.log(perf_p.chunk_size, 2))

                    if active_chunk_lanes < 64 :
                        chunk_mask = (1 << active_chunk_lanes) - 1
                    else:
                        chunk_mask = -1

                    active_mask = chunk_mask
                    for i in range(self.active_n_per_gpr-1):
                        active_mask = (active_mask << perf_p.chunk_size) + chunk_mask
                    
                    self.active_mask_lo = active_mask & 0xFFFFFFFF
                    self.active_mask_hi = active_mask >> 32

                    # group parameters
                    self.hi_c_per_wave = (self.hi_input_channels + perf_p.waves_c_in_group - 1) // perf_p.waves_c_in_group
                    last_wave_hi_c_per_wave = self.hi_input_channels - self.hi_c_per_wave * (perf_p.waves_c_in_group-1)
                    self.last_wave_hi_c_per_wave = last_wave_hi_c_per_wave 
                    lds_per_group = 0
                    self.double_lds = 0
                    self.WAVE_SIZE = 64
                    if perf_p.waves_c_in_group > 1:
                        lds_per_wave = perf_p.lds_limit // (perf_p.waves_c_in_group - 1) // perf_p.waves_k_in_group
                        #lds_gprs_per_wave = lds_per_wave // (4 * .self.WAVE_SIZE)
                        lds_gprs_per_wave = lds_per_wave // (4 * self.WAVE_SIZE)

                        if lds_gprs_per_wave >= accums_cnt:
                            self.sync_loops = 1
                            self.lds_gprs_per_loop = accums_cnt
                            lds_per_group = self.lds_gprs_per_loop * 4 * self.WAVE_SIZE * (perf_p.waves_c_in_group - 1) * perf_p.waves_k_in_group
                        else:
                            self.lds_gprs_per_loop = lds_gprs_per_wave // 2
                            self.sync_loops = (accums_cnt + self.lds_gprs_per_loop - 1) // self.lds_gprs_per_loop
                            lds_per_group = 2 * self.lds_gprs_per_loop * 4 * self.WAVE_SIZE * (perf_p.waves_c_in_group - 1) * perf_p.waves_k_in_group
                            self.double_lds = 1

                    self.raw_filter_dword_k_cnt = 1
                    if(conv_p.weights_layout == 0):
                        assert ((self.hi_c_per_wave * conv_p.vec_c_in) % perf_p.c_mult == 0 and (last_wave_hi_c_per_wave * conv_p.vec_c_in )% perf_p.c_mult == 0)

                        self.filter_c_gpr_stride = 1
                        self.filter_k_gpr_stride = perf_p.c_mult // conv_p.elements_in_dword
                        self.sequential_read_size= perf_p.c_mult // conv_p.elements_in_dword
                        self.sequential_read_stride = conv_p.filter_k_stride
                        self.sequential_reads_cnt = perf_p.k_mult
                        assert(perf_p.c_mult % conv_p.vec_c_filter == 0)
                    else:
                        assert ((self.hi_c_per_wave * conv_p.vec_c_in) % perf_p.c_mult == 0 and (last_wave_hi_c_per_wave * conv_p.vec_c_in )% perf_p.c_mult == 0)
                        self.raw_filter_dword_k_cnt = conv_p.elements_in_dword // conv_p.vec_c_filter
                        assert (perf_p.k_mult % (conv_p.elements_in_dword // conv_p.vec_c_filter) == 0)
                        assert (conv_p.output_channels % perf_p.k_mult == 0)
                        self.filter_c_gpr_stride = perf_p.k_mult // (conv_p.elements_in_dword // conv_p.vec_c_filter)
                        self.filter_k_gpr_stride = 1
                        self.sequential_read_size= perf_p.k_mult // (conv_p.elements_in_dword // conv_p.vec_c_filter)
                        self.sequential_read_stride = conv_p.filter_c_stride
                        self.sequential_reads_cnt = perf_p.c_mult // conv_p.vec_c_filter

            conv_p = conv_params()
            perf_p = perf_params()
            const_params = const_vars(conv_p, perf_p)

            class _sgpr(sgpr_file_t):
                def __init__(self, sgpr_f, HW:sgpr_hw_component, perf_p:perf_params):
                    super().__init__(sgpr_f.ic, sgpr_f._allocator)
                    add = self.add
                    self.karg_ptr = HW.get_karg_segment_ptr()
                    self.gid_hw = HW.get_gid_x()
                    self.gid_k = HW.get_gid_y()
                    self.gid_n = HW.get_gid_z()
                    self.soffset_in = add('soffset_in', 1)
                    self.soffset_out = add('soffset_out', 1)
                    self.soffset_wei = add('soffset_wei', 1)
                    self.desc_in = add('desc_in', 4)
                    self.desc_out = add('desc_out', 4)
                    self.desc_wei = add('desc_wei', 4)
                    self.filtersA = add('filtersA', perf_p.k_mult * perf_p.c_mult, 1)
                    
                    self.filtersB = add('filtersB', perf_p.k_mult * perf_p.c_mult, 4)
                    
                    self.wave_c_id = add('wave_c_id')
                    self.wave_k_id = add('wave_k_id')
                    self.loop_cnt = add('loop_cnt')
                    self.stmp_offset = add('stmp_offset')
                    self.stmp = add('stmp')

                    #self.N = add('N', 1)
                    #self.C = add('C', 1)
                    #self.H = add('H', 1)
                    #self.W = add('W', 1)
                    #self.K = add('K', 1)
                    #self.X = add('X', 1)
                    #self.Y = add('Y', 1)
                    #self.G = add('G', 1)

            
            class _vgpr(vgpr_file_t):
                def __init__(self, vgpr_f, HW:vgpr_hw_component, const_v:const_vars):
                    super().__init__(vgpr_f.ic, vgpr_f._allocator)
                    add = self.add
                    self.tid = HW.get_tid_x()
                    self.voffset_in = add('voffset_in')
                    self.voffset_out = add('voffset_out')
            
                    self.inputA = add('inputA', const_v.in_gprs)
                    self.inputB = add('inputB', const_v.in_gprs)

                    self.accums = add('accums', const_v.accums_cnt)
                    self.vtmp = add('vtmp')
                    if (const_v.madmix_instructions_available == 0 and const_v.dot_instructions_available == 0 and const_v.fmamix_instructions_available == 0):
                        self.vtmp_f_cvt = add('vtmp_f_cvt')
                    if(const_v.rem_hw_in):
                        self.current_hw_in = add('current_hw_in')

                    
            
            s = _sgpr(self.sgpr_f, self.HW, perf_p)
            v = _vgpr(self.vgpr_f, self.HW, const_params)
            
            ic = self.ic

            input_buffer_size = filter_buffer_size = output_buffer_size = literal(101)

            
            # fill format and size fields of buffer descriptors
            def fill_buff_desc(desc_reg:regVar, size:int):
                ic.s_mov_b32(desc_reg[2], size)
                ic.s_mov_b32(desc_reg[3], 0x00027000)
            
            input_buffer_size = filter_buffer_size = output_buffer_size = 256
            fill_buff_desc(s.desc_in[:], conv_p.input_buffer_size)
            fill_buff_desc(s.desc_wei[:], conv_p.filter_buffer_size)
            fill_buff_desc(s.desc_out[:], conv_p.output_buffer_size)

            ic.s_load_dwordx2(s.desc_in[0:1], s.karg_ptr[0:1], karg.in_ptr_off+0)
            ic.s_load_dwordx2(s.desc_wei[0:1], s.karg_ptr[0:1], karg.wei_ptr_off+0)
            ic.s_load_dwordx2(s.desc_out[0:1], s.karg_ptr[0:1], karg.out_ptr_off+0)
            
            #ic.s_load_dwordx2(s.C[0], s.karg_ptr[0:1], karg.C+0)

            ic.s_mov_b32(s.exec.lo, const_params.active_mask_lo)
            ic.s_mov_b32(s.exec.hi, const_params.active_mask_hi)

            ic.v_lshrrev_b32(v.vtmp[0], 6, v.tid[0])
            ic.v_readfirstlane_b32(s.wave_c_id[0], v.vtmp[0])
            
            def get_rcp(reg, val):
                if val == 1:
                    ic.s_mov_b32(reg, const(1.0))
                elif val == 2:
                   ic.s_mov_b32(reg, const(0.5))
                elif val == 3:
                   ic.s_mov_b32(reg, const(0.33333333333))
                elif val == 4:
                   ic.s_mov_b32(reg, const(0.25))
                elif val == 5:
                   ic.s_mov_b32(reg, const(0.2))
                elif val == 6:
                   ic.s_mov_b32(reg, const(0.16666666666))
                elif val == 7:
                   ic.s_mov_b32(reg, const(0.14285714285))
                elif val == 8:
                   ic.s_mov_b32(reg, const(0.125))
                else:
                    #"val > 8"
                   assert(False)

            def _v_add_nc_u32(dst, src0, src1, dpp=None):
                #if (.option.machine_version_major == 8)
                    # None No-Carry instruction in Gfx8, modifies VCC.
                 #   ic.v_add_u32( dst, vcc, src0, src1 dpp
                #else:
                    #ic.v_add_u32 dst, src0, src1 dpp
                if(dpp == None):
                    ic.v_add_nc_u32(dst,src0,src1)
                else:
                    ic.v_add_nc_u32_dpp(dst,src0,src1, dpp)


            #wave_k_id = wave_id / waves_c_in_group
            ic.v_cvt_f32_u32(v.vtmp[0], v.vtmp[0])
            
            get_rcp(s.stmp[0], perf_p.waves_c_in_group)

            ic.v_mul_f32(v.vtmp[0], v.vtmp[0], s.stmp[0])
            
            ic.v_cvt_u32_f32(v.vtmp[0], v.vtmp[0],)

            ic.v_readfirstlane_b32(s.wave_k_id[0], v.vtmp[0])
            # wave_c_id = wave_id % waves_c_in_group
            ic.s_mul_i32(s.stmp[0], s.wave_k_id[0], perf_p.waves_c_in_group)
            ic.s_sub_i32(s.wave_c_id[0], s.wave_c_id[0], s.stmp[0])
            ic.v_and_b32(v.tid[0], 0x3f, v.tid[0])


            # calculate input/output offsets
            ic.v_lshrrev_b32(v.vtmp[0], 0 + const_params.chunk_size_log2, v.tid[0]) #vtmp = wave part id
            ic.v_mul_u32_u24(v.voffset_in[0], 0 + conv_p.input_n_stride, v.vtmp[0])
            ic.v_mul_u32_u24(v.voffset_out[0], 0 + conv_p.output_n_stride, v.vtmp[0])

            ic.v_and_b32(v.vtmp[0], 0 + perf_p.chunk_size - 1, v.tid[0]) #vtmp = lane in wave part
            ic.v_mul_u32_u24(v.vtmp[0], 0 + conv_p.input_w_stride * perf_p.chunks_per_wave, v.vtmp[0])
            _v_add_nc_u32(v.voffset_in[0], v.voffset_in[0], v.vtmp[0])

            ic.v_and_b32(v.vtmp[0], 0 + perf_p.chunk_size - 1, v.tid[0]) #vtmp = lane in wave part
            ic.v_mul_u32_u24(v.vtmp[0], 0 + conv_p.output_w_stride * perf_p.chunks_per_wave, v.vtmp[0])
            _v_add_nc_u32(v.voffset_out[0], v.voffset_out[0], v.vtmp[0])

            ic.s_mul_i32(s.soffset_in[0], s.gid_n[0], 0 + conv_p.input_n_stride * const_params.active_n_per_wave)
            ic.s_mul_i32(s.soffset_out[0], s.gid_n[0], 0 + conv_p.output_n_stride * const_params.active_n_per_wave)

            ic.s_mul_i32(s.stmp[0], s.gid_hw[0], 0 + const_params.active_hw_per_wave * conv_p.input_w_stride)
            ic.s_add_u32(s.soffset_in[0], s.soffset_in[0], s.stmp[0])

            ic.s_mul_i32(s.stmp[0], s.gid_hw[0], 0 + const_params.active_hw_per_wave * conv_p.output_w_stride)
            ic.s_add_u32(s.soffset_out[0], s.soffset_out[0], s.stmp[0])

            ic.s_mul_i32(s.stmp[0], s.wave_c_id[0], 0 + const_params.hi_c_per_wave * conv_p.input_c_stride)
            ic.s_add_u32(s.soffset_in[0], s.soffset_in[0], s.stmp[0])

            ic.s_mul_i32(s.stmp[0], s.gid_k[0], 0 + conv_p.output_k_stride * perf_p.k_mult * perf_p.waves_k_in_group // conv_p.vec_k_out)
            ic.s_add_u32(s.soffset_out[0], s.soffset_out[0], s.stmp[0])
            ic.s_mul_i32(s.stmp[0], s.wave_k_id[0], 0 + conv_p.output_k_stride * perf_p.k_mult // conv_p.vec_k_out)
            ic.s_add_u32(s.soffset_out[0], s.soffset_out[0], s.stmp[0])
            ic.s_mul_i32(s.soffset_wei[0], s.gid_k[0], 0 + perf_p.k_mult * conv_p.filter_k_stride * perf_p.waves_k_in_group)
            ic.s_mul_i32(s.stmp[0], s.wave_k_id[0], 0 + perf_p.k_mult * conv_p.filter_k_stride)
            ic.s_add_u32(s.soffset_wei[0], s.soffset_wei[0], s.stmp[0])

            assert(conv_p.vec_c_in == conv_p.vec_c_filter)
            ic.s_mul_i32(s.stmp[0], s.wave_c_id[0], 0 + const_params.hi_c_per_wave * conv_p.filter_c_stride)
            ic.s_add_u32(s.soffset_wei[0], s.soffset_wei[0], s.stmp[0])


            ic.s_waitcnt (0)

            def _s_buffer_load_dwordxcnt(cnt:int, base:regVar, desc:regVar, off:regVar):
                if cnt == 2:
                    ic.s_buffer_load_dwordx2(base[0:1], desc[0:3], off[0])
                elif cnt == 4:
                    ic.s_buffer_load_dwordx4(base[0:3], desc[0:3], off[0])
                elif cnt == 8:
                    ic.s_buffer_load_dwordx8(base[0:7], desc[0:3], off[0])
                elif cnt == 16:
                    ic.s_buffer_load_dwordx16(base[0:15], desc[0:3], off[0])

            def xsload(base, xx, cnt):
                ret = 0
                for _i_1 in range(xx):
                    if cnt == 1:
                        ic.s_buffer_load_dword(base[_i_1], s.desc_wei[0:3], s.soffset_wei[0])
                    else:
                        _s_buffer_load_dwordxcnt(cnt, base[_i_1:_i_1+cnt-1], s.desc_wei[0:3], s.soffset_wei[0])
                    
                    base = base + cnt
                    ret += cnt
                    ic.s_add_u32(s.soffset_wei[0], s.soffset_wei[0], 0 + 4 * cnt)
                return ret
                
            def load_filters(base, seq_size, seq_cnt, seq_stride):
                seq_it = 0
                fbase = base
                for _i_1 in range(seq_cnt):
                    x16_chunks = seq_size // 16
                    rest = seq_size - x16_chunks * 16
                    x8_chunks = rest // 8
                    rest = rest - x8_chunks * 8
                    x4_chunks = rest // 4
                    rest = rest - x4_chunks * 4
                    x2_chunks = rest // 2
                    rest = rest - x2_chunks * 2
                    x1_chunks = rest
                    imm_off = 0

                    inc = xsload (fbase, x16_chunks, 16)
                    fbase += inc
                    inc = xsload (fbase, x8_chunks, 8)
                    fbase += inc
                    inc = xsload (fbase, x4_chunks, 4)
                    fbase += inc
                    inc = xsload (fbase, x2_chunks, 2)
                    fbase += inc
                    inc = xsload (fbase, x1_chunks, 1)
                    fbase += inc

                    seq_it = seq_it + 1
                    if(conv_p.weights_layout == 0 and seq_it == seq_cnt):
                        ic.s_add_u32(s.soffset_wei[0], s.soffset_wei[0], 0 - seq_stride * (seq_cnt - 1) )
                    else:
                        ic.s_add_u32(s.soffset_wei[0], s.soffset_wei[0], 0 + seq_stride - 4 * seq_size )


            if perf_p.chunks_per_wave % (perf_p.read_size * const_params.input_dword_chunks_cnt):
                mbufs_cnt = (perf_p.c_mult // conv_p.elements_in_dword) * perf_p.n_mult * (1 + perf_p.chunks_per_wave // (perf_p.read_size * const_params.input_dword_chunks_cnt))
            else:
                mbufs_cnt = (perf_p.c_mult // conv_p.elements_in_dword) * perf_p.n_mult * (perf_p.chunks_per_wave // (perf_p.read_size * const_params.input_dword_chunks_cnt))
            
            def load_input (base:regVar):
                def m_buffer_load_dwordx(size, dst, off, desc, soff, ioff:int=0):
                    if size == 1:
                        ic.buffer_load_dword(dst[0], off[0], desc[0:3], soff[0], f'offen offset:0+{ioff}')
                    elif size == 2:
                        ic.buffer_load_dwordx2(dst[0:size-1], off[0], desc[0:3], soff[0], f'offen offset:0+{ioff}')
                    elif size == 3:
                        ic.buffer_load_dwordx3(dst[0:size-1], off[0], desc[0:3], soff[0], f'offen offset:0+{ioff}')
                    elif size == 4:
                        ic.buffer_load_dwordx4(dst[0:size-1], off[0], desc[0:3], soff[0], f'offen offset:0+{ioff}')

                def m_buffer_load_ushort(size, dst, off, desc, soff, ioff:int=0):
                    if size == 1:
                        ic.buffer_load_ushort( dst[0], off[0],  desc[0:3], soff[0], f'offen offset:0+{ioff}')
                    

                ibase = base
                hi_c_mult = perf_p.c_mult // conv_p.vec_c_in
                full_loads =              perf_p.chunks_per_wave // (perf_p.read_size * const_params.input_dword_chunks_cnt)
                partial_load_chunks_cnt = perf_p.chunks_per_wave % (perf_p.read_size * const_params.input_dword_chunks_cnt)
                partial_load_dwords = (partial_load_chunks_cnt + const_params.input_dword_chunks_cnt -1) // const_params.input_dword_chunks_cnt
                partial_load_short  =  partial_load_chunks_cnt % const_params.input_dword_chunks_cnt
                nb = 0
                for _i_1 in range(perf_p.n_mult):
                    c_it = 0
                    ic.s_mov_b32(s.stmp_offset[0], s.soffset_in[0])
                    for _i_2 in range( hi_c_mult): # input and filter must be vectorized
                        ic.s_cmpk_le_i32(s.loop_cnt[0], imm16_t(0 + c_it))
                        ic.s_cmov_b32(s.stmp_offset[0], 0 + const_params.invalid_addr_lit)

                        ld_it = 0
                        imm_off = 0
                        current_read_cnt = perf_p.read_size
                        rem_ibase = ibase
                        for _i_3 in range( full_loads + 1):
                            if(ld_it == full_loads):
                                current_read_cnt = partial_load_dwords
                            
                            if(current_read_cnt):
                                m_buffer_load_dwordx(current_read_cnt, ibase[:], v.voffset_in[:], s.desc_in[:], s.stmp_offset[:], imm_off)
                            ibase = ibase + current_read_cnt
                            imm_off = imm_off + 4 * current_read_cnt
                            ld_it = ld_it + 1
                            #TODO change step size

                        if conv_p.elements_in_dword == 2 and const_params.rem_hw_in:

                            chunk_id = const_params.img_hw // (perf_p.chunks_per_wave)
                            rem_dword_id = (const_params.img_hw % (perf_p.chunks_per_wave)) // const_params.input_dword_chunks_cnt
                            rem_ibase = rem_ibase + rem_dword_id
                            ic.v_cmpx_eq_i32(0 + chunk_id * (perf_p.chunks_per_wave), v.current_hw_in[0])

                            m_buffer_load_ushort(1,  rem_ibase, v.voffset_in[:], s.desc_in[:], s.stmp_offset[:], rem_dword_id * 4)
                            
                            ic.s_mov_b32(s.exec.lo, const_params.active_mask_lo)
                            ic.s_mov_b32(s.exec.hi, const_params.active_mask_hi)
                        
                        c_it = c_it + 1
                        ic.s_add_u32(s.stmp_offset[0], s.stmp_offset[0], conv_p.input_c_stride)

                    nb = nb + 1 

                    if nb == perf_p.n_mult:
                        ic.s_add_u32(s.soffset_in[0], s.soffset_in[0], 0 + (conv_p.input_c_stride * hi_c_mult) - conv_p.input_n_stride * const_params.active_n_per_gpr * (perf_p.n_mult - 1) )
                    else:
                        ic.s_add_u32(s.soffset_in[0], s.soffset_in[0], 0 + conv_p.input_n_stride * const_params.active_n_per_gpr)

                ic.s_addk_i32(s.loop_cnt[0], imm16_t(0 - (1 * perf_p.c_mult)) )
                if (1 or (const_params.hi_c_per_wave % 4) or (const_params.hi_input_channels % const_params.hi_c_per_wave) ):
                    ic.s_cmpk_le_i32(s.loop_cnt[0],imm16_t(0))
                    ic.s_cmov_b32 (s.desc_in[2], 0)
                

            def get_acc_idx(k:int, n:int, chunk:int):
                acc = v.accums[perf_p.chunks_per_wave * perf_p.n_mult * k + n * perf_p.chunks_per_wave + chunk]
                return acc

            #repack imgs between two vgpr
            def exch_img(img_c0, img_c1):
                ic.v_mov_b32(v.vtmp[0], img_c0[0])
                ic.v_mov_b32_sdwa(img_c0[0], img_c1[0], 'dst_sel:WORD_1 src0_sel:WORD_0')
                ic.v_mov_b32_sdwa(img_c1[0], v.vtmp[0], 'dst_sel:WORD_0 src0_sel:WORD_1')
            

            def exch_filter(filter_c0:regVar, filter_c1:regVar, tmp0:regVar, tmp1:regVar):
                assert(filter_c0 != filter_c1 and filter_c0 != tmp0 and filter_c1 != tmp0)
                if const_params.s_pack_instructions_available:
                    ic.s_mov_b32         (tmp0[0],       filter_c0[0])
                    ic.s_pack_ll_b32_b16 (filter_c0[0],  filter_c0[0],  filter_c1[0])
                    ic.s_pack_hh_b32_b16 (filter_c1[0],  s.stmp_offset[0], filter_c1[0])
                else:
                    assert(tmp1 != filter_c0 and tmp1 != filter_c1 and tmp1 != tmp0)
                    ic.s_lshr_b32 (tmp1[0],      filter_c0[0], 16)
                    ic.s_and_b32  (tmp0[0],      filter_c0[0], 0x0000ffff)
                    ic.s_lshl_b32 (filter_c0[0], filter_c1[0], 16)
                    ic.s_or_b32   (filter_c0[0], filter_c0[0], tmp0[0])
                    ic.s_and_b32  (filter_c1[0], filter_c1[0], 0xffff0000)
                    ic.s_or_b32   (filter_c1[0], filter_c1[0], tmp1[0])
            
            #repack input across channels
            def trans_input(ibase:regVar):
                if(const_params.input_dword_chunks_cnt == 2):
                    c = 0
                    for _i_1 in range(perf_p.c_mult): # input_dword_chunks_cnt
                        n = 0
                        for _i_2 in range(perf_p.n_mult):
                            ch_gpr = 0
                            dwords_with_chunks_from_cx_lane = perf_p.chunks_per_wave // const_params.input_dword_chunks_cnt
                            for _i_3 in range(dwords_with_chunks_from_cx_lane):
                                c_gpr_inp = c * dwords_with_chunks_from_cx_lane
                                n_gpr_inp = n * perf_p.c_mult * dwords_with_chunks_from_cx_lane
                                img = ibase + ch_gpr + n_gpr_inp + c_gpr_inp
                                exch_img(img, img + dwords_with_chunks_from_cx_lane)
                                ch_gpr = ch_gpr + 1
                            n = n + 1
                        c = c + const_params.input_dword_chunks_cnt
            

            #repack filter across channels
            def trans_filter(fbase:regVar):
                if(conv_p.elements_in_dword == 2 and  const_params.raw_filter_dword_k_cnt == 2):
                    if(conv_p.weights_layout != 0):
                        c = 0
                        for _i_1 in range(const_params.sequential_reads_cnt // const_params.raw_filter_dword_k_cnt):
                            k = 0
                            for _i_2 in range(const_params.filter_c_gpr_stride):
                                c_gpr_filter = (c) * const_params.filter_c_gpr_stride
                                k_gpr_filter = k * const_params.filter_k_gpr_stride
                                wei = fbase + k_gpr_filter + c_gpr_filter
                                exch_filter(wei, wei + const_params.filter_c_gpr_stride, s.stmp_offset[0], s.stmp[0])
                                k = k + 1
                            
                            c = c + const_params.raw_filter_dword_k_cnt

            
            def conv(ibase, fbase):
                chunk_lo_intrans_gpr_stride = perf_p.chunks_per_wave // const_params.input_dword_chunks_cnt
                chunk_hi_intrans_gpr_stride = 1
                hi_c_mult = perf_p.c_mult // conv_p.elements_in_dword
                k_lo_gpr_stride = const_params.filter_c_gpr_stride
                k_hi_gpr_stride = const_params.filter_k_gpr_stride
                c_hi_ftrans_gpr_stride = const_params.filter_c_gpr_stride * const_params.raw_filter_dword_k_cnt
                c_hi_intrans_gpr_stride = perf_p.chunks_per_wave
                n_input_gpr_stride = hi_c_mult * c_hi_intrans_gpr_stride
                for c_hi in range(hi_c_mult):
                    for k in range(perf_p.k_mult):
                        for nb in range(perf_p.n_mult):
                            for chunk in range(perf_p.chunks_per_wave):
                                acc = get_acc_idx(k, nb, chunk)
                                k_lo = (k % const_params.raw_filter_dword_k_cnt) * k_lo_gpr_stride
                                k_hi = (k // const_params.raw_filter_dword_k_cnt) * k_hi_gpr_stride
                                k_gpr_filter = k_lo + k_hi

                                c_gpr_filter = c_hi * c_hi_ftrans_gpr_stride
                                f_gpr = fbase + k_gpr_filter + c_gpr_filter

                                c_gpr_inp = c_hi * c_hi_intrans_gpr_stride
                                n_gpr_inp = nb * n_input_gpr_stride

                                chunk_lo = ((chunk) % const_params.input_dword_chunks_cnt) * chunk_lo_intrans_gpr_stride
                                chunk_hi = (chunk // const_params.input_dword_chunks_cnt) * chunk_hi_intrans_gpr_stride
                                inp_gpr = ibase + c_gpr_inp + n_gpr_inp + chunk_lo + chunk_hi

                                if conv_p.buf_type == DATA_TYPE.TYPE_FP32 and conv_p.vec_c_in == 1 :
                                    #ic.v_mac_f32( acc[0], f_gpr[0], inp_gpr[0])
                                    ic.v_fma_f32( acc[0], f_gpr[0], inp_gpr[0], acc[0])
                                elif conv_p.buf_type == DATA_TYPE.TYPE_FP16:
                                    if const_params.dot_instructions_available:
                                        ic.v_dot2_f32_f16( acc[0], f_gpr[0], inp_gpr[0], acc[0])
                                    elif const_params.madmix_instructions_available:
                                        ic.v_mad_mix_f32( acc[0], f_gpr[0], inp_gpr[0], acc[0], 'op_sel:[0,0,0] op_sel_hi:[1,1,0]')
                                        ic.v_mad_mix_f32( acc[0], f_gpr[0], inp_gpr[0], acc[0], 'op_sel:[1,1,0] op_sel_hi:[1,1,0]')
                                    elif const_params.fmamix_instructions_available:
                                        ic.v_fma_mix_f32( acc[0], f_gpr[0], inp_gpr[0], acc[0], 'op_sel:[0,0,0] op_sel_hi:[1,1,0]')
                                        ic.v_fma_mix_f32( acc[0], f_gpr[0], inp_gpr[0], acc[0], 'op_sel:[1,1,0] op_sel_hi:[1,1,0]')
                                    else:
                                        ic.v_mov_b32(v.vtmp_f_cvt[0], f_gpr[0])
                                        ic.v_cvt_f32_f16( v.vtmp[0], inp_gpr[0])
                                        ic.v_cvt_f32_f16( v.vtmp_f_cvt[0], v.vtmp_f_cvt[0])
                                        ic.v_mac_f32( acc[0], v.vtmp[0], v.vtmp_f_cvt[0])

                                        ic.v_mov_b32( v.vtmp_f_cvt[0], f_gpr[0])
                                        ic.v_lshrrev_b32( v.vtmp[0], 16, inp_gpr[0])
                                        ic.v_lshrrev_b32( v.vtmp_f_cvt[0], 16, v.vtmp_f_cvt[0])

                                        ic.v_cvt_f32_f16( v.vtmp[0], v.vtmp[0])
                                        ic.v_cvt_f32_f16( v.vtmp_f_cvt[0], v.vtmp_f_cvt[0])
                                        ic.v_mac_f32(     acc[0], v.vtmp[0], v.vtmp_f_cvt[0])
                                else:
                                    assert(0)

 
            if(const_params.rem_hw_in):
                ic.v_and_b32( v.current_hw_in[0], 0 + perf_p.chunk_size - 1, v.tid[0])
                ic.v_mul_u32_u24( v.current_hw_in[0], 0 + perf_p.chunks_per_wave, v.current_hw_in[0])
                ic.s_mul_i32( s.stmp[0], s.gid_hw[0], 0 + const_params.active_hw_per_wave)
                _v_add_nc_u32( v.current_hw_in[0],  s.stmp[0], v.current_hw_in[0])

            ic.s_mov_b32( s.loop_cnt[0], 0 + const_params.hi_c_per_wave * conv_p.vec_c_in)
            ic.s_cmpk_eq_u32( s.wave_c_id[0], imm16_t(0 + perf_p.waves_c_in_group - 1))
            ic.s_cmov_b32( s.loop_cnt[0], 0 + const_params.last_wave_hi_c_per_wave * conv_p.vec_c_in)

            load_input(v.inputA[:])
            load_filters(s.filtersA[:], const_params.sequential_read_size, const_params.sequential_reads_cnt, const_params.sequential_read_stride)

            # zeroing accums

            for i in range(const_params.accums_cnt):
               ic.v_mov_b32(v.accums[i], 0)

            loop_begin = label_t('loop_begin')
            self.set_label(loop_begin)

            load_input(v.inputB[:])
            
            ic.s_wait(mbufs_cnt, 0)

            load_filters( s.filtersB[:], const_params.sequential_read_size, const_params.sequential_reads_cnt, const_params.sequential_read_stride)
            trans_input( v.inputA[:])
            trans_filter (s.filtersA[:])
            conv( v.inputA[:], s.filtersA[:])

            load_input( v.inputA[:])
            ic.s_wait( mbufs_cnt, 0)
            load_filters( s.filtersA[:], const_params.sequential_read_size, const_params.sequential_reads_cnt, const_params.sequential_read_stride)
            trans_input( v.inputB[:])
            trans_filter( s.filtersB[:])
            conv( v.inputB[:], s.filtersB[:])

            loop_end = label_t('loop_end')
            self.set_label(loop_end)

            ic.s_cmpk_gt_i32( s.loop_cnt[0], imm16_t(1 * perf_p.c_mult))
            ic.s_cbranch_scc1( loop_begin)

            load_input( v.inputB[:])
            ic.s_wait( mbufs_cnt, 0)
            load_filters( s.filtersB[:], const_params.sequential_read_size, const_params.sequential_reads_cnt, const_params.sequential_read_stride)
            trans_input( v.inputA[:])
            trans_filter( s.filtersA[:])
            conv( v.inputA[:], s.filtersA[:])
            ic.s_waitcnt( 0)

            trans_input( v.inputB[:])
            trans_filter( s.filtersB[:])
            conv( v.inputB[:], s.filtersB[:])

            # reduction across waves in group
            # all waves but last store accums to LDS and dies
            # last wave survives and read LDS

            lds_off = v.add_no_pos('lds_off')
            v.reuse(v.voffset_in, lds_off)

            lds_off_k = s.add_no_pos('lds_off_k')
            s.reuse(s.soffset_in, lds_off_k)
            
            last_wave = label_t('last_wave')

            if perf_p.waves_c_in_group > 1:
                ic.s_mul_i32(lds_off_k[0], s.wave_k_id[0], 4 * const_params.WAVE_SIZE * const_params.lds_gprs_per_loop * (perf_p.waves_c_in_group-1) * (const_params.double_lds + 1) )
                ic.s_mov_b32( s.m0, -1)
                ic.s_cmpk_eq_u32( s.wave_c_id[0], imm16_t(0 + perf_p.waves_c_in_group - 1))
                ic.s_cbranch_scc1( last_wave)

                ic.s_mul_i32( s.stmp[0], s.wave_c_id[0], 4 * const_params.WAVE_SIZE * const_params.lds_gprs_per_loop)

                ic.v_lshlrev_b32( lds_off[0], 2, v.tid[0])
                _v_add_nc_u32( lds_off[0], s.stmp[0], lds_off[0])
                _v_add_nc_u32( lds_off[0], lds_off_k[0], lds_off[0])
                acc_id = 0
                for sync_loop in range(const_params.sync_loops):
                    imm_off = (sync_loop % 2) * const_params.lds_gprs_per_loop * (perf_p.waves_c_in_group-1) * 4 * const_params.WAVE_SIZE
                    for _i_2 in range(const_params.lds_gprs_per_loop):
                        if acc_id < const_params.accums_cnt:
                            ic.ds_write_b32(lds_off[0], v.accums[acc_id], f'offset:0+{imm_off}')
                        acc_id = acc_id + 1
                        imm_off = imm_off + 4 * const_params.WAVE_SIZE
                    ic.s_waitcnt(0)
                    ic.s_barrier


                ic.s_endpgm

                self.set_label(last_wave)

                acc_id = 0

                for sync_loop in range(const_params.sync_loops):
                    ic.v_lshlrev_b32( lds_off[0], 2, v.tid[0])
                    _v_add_nc_u32(lds_off[0], lds_off_k[0], lds_off[0])
                    ic.s_barrier

                    for gpr in range(const_params.lds_gprs_per_loop):
                        for wave in range(perf_p.waves_c_in_group-1):
                            imm_off = 4 * const_params.WAVE_SIZE * (gpr + wave * const_params.lds_gprs_per_loop + (sync_loop % 2) * const_params.lds_gprs_per_loop * (perf_p.waves_c_in_group-1))
                            if acc_id < const_params.accums_cnt:
                                ic.ds_read_b32( v.vtmp[0], lds_off[0], f'offset:0+{imm_off}')
                                ic.s_waitcnt (0)

                                ic.v_add_f32(v.accums[acc_id], v.vtmp[0], v.accums[acc_id])

                        acc_id = acc_id + 1

            # Pack output

            acc_idx = []
            def set_acc_idx(idx, k, nb, chunk):
                acc_idx[idx] = get_acc_idx (k, nb, chunk)


            def set_all_acc_idx(rounded_ck_base, vec_ck, nb, chunk_base):
                _ck_local = 0
                _chunk_local = 0
                for _idx in range(1, conv_p.elements_in_dword):
                    set_acc_idx(_idx, (rounded_ck_base * vec_ck + _ck_local), nb, chunk_base + _chunk_local)
                    if(vec_ck == 1):
                        _chunk_local = _chunk_local + 1
                    else:
                        _ck_local = _ck_local + 1

            if (conv_p.vec_k_out > 1) or (conv_p.elements_in_dword > 1):

                for nb in range(perf_p.n_mult):
                    for hi_chunk in range( perf_p.chunks_per_wave // const_params.output_dword_chunks_cnt):
                        hi_k = 0
                        for hi_k in range(perf_p.k_mult // conv_p.vec_k_out):
                            set_all_acc_idx( hi_k, conv_p.vec_k_out, nb, hi_chunk  * const_params.output_dword_chunks_cnt)
                            if conv_p.buf_type == DATA_TYPE.TYPE_FP16 :
                               ic.v_cvt_pkrtz_f16_f32( acc_idx[1], acc_idx[1], acc_idx[2])
                            else:
                                assert(0)



            # store output
            current_k = s.add_no_pos('current_k')
            s.reuse(s.stmp_offset, current_k)

            ic.s_mul_i32(current_k[0], s.gid_k[0], 0 + perf_p.k_mult * perf_p.waves_k_in_group // conv_p.vec_k_out)
            ic.s_mul_i32(s.stmp[0], s.wave_k_id[0], 0 + perf_p.k_mult // conv_p.vec_k_out)
            ic.s_add_u32(current_k[0], current_k[0], s.stmp[0])

            current_hw = v.add_no_pos('current_hw')

            if(not const_params.rem_hw_in):
                v.reuse(v.inputA, current_hw)
                ic.v_and_b32(current_hw[0], 0 + perf_p.chunk_size - 1, v.tid[0])
                ic.v_mul_u32_u24(current_hw[0], 0 + perf_p.chunks_per_wave, current_hw[0])
                ic.s_mul_i32(s.stmp[0], s.gid_hw[0], 0 + const_params.active_hw_per_wave)
                _v_add_nc_u32(current_hw[0],  s.stmp[0], current_hw[0])
            else:
                GPR_REUSE(v.current_hw_in, current_hw)

            def store_result():
                rem_hw_out = (conv_p.img_h * conv_p.img_w) % const_params.output_dword_chunks_cnt
                for k in range(perf_p.k_mult // conv_p.vec_k_out):

                    ic.s_cmpk_ge_i32( current_k[0], imm16_t(0 + const_params.hi_output_channels - k))
                    ic.s_cmov_b32( s.desc_out[2], 0)

                    for nb in range(perf_p.n_mult):
                        ic.s_mov_b32(s.exec.lo, const_params.active_mask_lo)
                        ic.s_mov_b32(s.exec.hi, const_params.active_mask_hi)

                        for chunk in range(perf_p.chunks_per_wave // const_params.output_dword_chunks_cnt):
                            ic.v_cmpx_ge_i32(0 + (const_params.img_hw - rem_hw_out) - (chunk + 1) * const_params.output_dword_chunks_cnt, current_hw[0])
                            acc = get_acc_idx(conv_p.vec_k_out * k, nb, chunk * const_params.output_dword_chunks_cnt)
                            ic.buffer_store_dword( acc[0], v.voffset_out[0], s.desc_out[0:3], s.soffset_out[0], f'offen offset:0+{4 * chunk}')

                        if(rem_hw_out != 0):
                            #TODO add support for int8
                            ic.s_mov_b32(s.exec.lo, const_params.active_mask_lo)
                            ic.s_mov_b32(s.exec.hi, const_params.active_mask_hi)

                            chunk_id = const_params.img_hw // (perf_p.chunks_per_wave)

                            ic.v_cmpx_eq_i32(0 + chunk_id * perf_p.chunks_per_wave, current_hw[0])

                            last_dword = (const_params.img_hw % perf_p.chunks_per_wave) // const_params.output_dword_chunks_cnt
                            acc = get_acc_idx(conv_p.vec_k_out * k, nb, last_dword * const_params.output_dword_chunks_cnt)
                            ic.buffer_store_short(acc[0], v.voffset_out[0], s.desc_out[0:3], s.soffset_out[0], f'offen offset:0+{4 * last_dword}')


                        if (nb+1) == perf_p.n_mult:
                            ic.s_add_u32( s.soffset_out[0], s.soffset_out[0], 0 + conv_p.output_k_stride - const_params.active_n_per_gpr * conv_p.output_n_stride * (perf_p.n_mult - 1))
                        else:
                            ic.s_add_u32( s.soffset_out[0], s.soffset_out[0], 0 + const_params.active_n_per_gpr * conv_p.output_n_stride)

            store_result()

            ic.s_endpgm()


        launch_k(self.k_config, self.kargs)

    def get_kernel_block_size(self):
        return (256, 1, 1)

    def __init__(self, mc_asm_printer: mc_asm_printer_t, **kwargs):
        super().__init__(mc_asm_printer, **kwargs)
    class custom_caller(gfx10_instructions_caller):
        def __init__(self, insturction_list) -> None:
            super().__init__(insturction_list)

    def _set_kernel_karg_t(self) -> None:
        '''Should be called before get_amdgpu_metadata_list in kernel_constructor.__init__.
        Defined in kernel class to make overwrited kernel_karg_t trackable by IDE.'''
        self.kargs=self.kernel_karg_t(self.mc)
    
    def set_GPU_HW(self):
        self.HW = HW_gfx9(self.instructions_caller, stack_allocator, stack_allocator)

    def _instructions_init(self):
        '''Defined in kernel class to make overwrited sgpr and vgpr trackable by IDE.'''
        self.instructions_caller = conv_direct_1x1u.custom_caller(self.instr_ctrl.instructions_list)
