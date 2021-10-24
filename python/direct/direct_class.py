from abc import ABC, abstractmethod
from python.codegen.gpu_data_types import *
from python.codegen.gpu_arch.HW_components import HW_gfx9, sgpr_file_t, vgpr_file_t
from python.codegen.kernel_func import kernel_func, kernel_launcher, launcher_kernel, mfunc_func
from python.codegen.gpu_arch.GFX10 import gfx10_instructions_caller
from python.codegen.kernel_driver import base_config
from ..codegen.config_parser import config_content_t
from ..codegen.mc import mc_base_t, mc_asm_printer_t
from .kernel_constructor import *
from python.codegen.gpu_instruct import gpu_instructions_caller_base

class direct_navi_config(base_config):
    def __init__(self, config_content: config_content_t):
        super().__init__(config_content, '-navi')
        
        self.read_size = config_content.get_section('direct-navi')[0]['read_size']

class conv_direct_navi(kernel_constructor):

    def get_kernel_name(self):
        return 'conv_direct_navi'

    class kernel_karg_t(karg_file_t):
        '''Define kernel arguments. Used in _set_kernel_karg_t'''
        def __init__(self, mc) -> None:
            super().__init__(mc)
            pb_arg = self._pb_kernel_arg
            self.in_ptr  = pb_arg('in_ptr',  arg_kind.GlobBuffer, arg_type.F32)
            self.out_ptr = pb_arg('out_ptr', arg_kind.GlobBuffer, arg_type.F32)
            self.wei_ptr = pb_arg('wei_ptr', arg_kind.GlobBuffer, arg_type.F32)
            self.N = pb_arg('N', arg_kind.value, arg_type.I32)
            self.C = pb_arg('C', arg_kind.value, arg_type.I32)
            self.H = pb_arg('H', arg_kind.value, arg_type.I32)
            self.W = pb_arg('W', arg_kind.value, arg_type.I32)
            self.K = pb_arg('K', arg_kind.value, arg_type.I32)
            self.X = pb_arg('Y', arg_kind.value, arg_type.I32)
            self.Y = pb_arg('X', arg_kind.value, arg_type.I32)
            self.G = pb_arg('G', arg_kind.value, arg_type.I32)

    def _get_LDS_usage(self):
        return 0

    def _emit_kernel_body(self):

        @launcher_kernel
        def launch_k(self:kernel_launcher[gfx10_instructions_caller], karg:conv_direct_navi.kernel_karg_t):
            class _sgpr(sgpr_file_t):
                def __init__(self, sgpr_f):
                    super().__init__(sgpr_f.ic, sgpr_f._allocator)
                    add = self.add 
                    self.karg_ptr = add('karg_ptr', 2)
                    self.group_id = add('group_id', 1)
                    self.gid_off = add('gid_off', 1)
                    self.in_buff_ptr = add('in_buff_ptr', 4)
                    self.out_buff_ptr = add('out_buff_ptr', 4)
                    self.wei_buff_ptr = add('wei_buff_ptr', 4)
                    self.N = add('N', 1)
                    self.C = add('C', 1)
                    self.H = add('H', 1)
                    self.W = add('W', 1)
                    self.K = add('K', 1)
                    self.X = add('X', 1)
                    self.Y = add('Y', 1)
                    self.G = add('G', 1)
            
            class _vgpr(vgpr_file_t):
                def __init__(self, vgpr_f):
                    super().__init__(vgpr_f.ic, vgpr_f._allocator)
                    add = self.add 
                    self.tid = add('tid', 1)
                    self.in_off = add('in_off', 1)
                    
            
            s = _sgpr(self.sgpr_f)
            v = _vgpr(self.vgpr_f)
            
            ic = self.ic

            input_buffer_size = filter_buffer_size = output_buffer_size = literal(101)

            ic.s_mov_b32(s.in_buff_ptr[2], input_buffer_size)
            ic.s_mov_b32(s.in_buff_ptr[3], 0x00027000)

            ic.s_mov_b32(s.out_buff_ptr[2], output_buffer_size)
            ic.s_mov_b32(s.out_buff_ptr[3], 0x00027000)

            ic.s_mov_b32(s.wei_buff_ptr[2], filter_buffer_size)
            ic.s_mov_b32(s.wei_buff_ptr[3], 0x00027000)
            ic.s_mov_b32(s.wei_buff_ptr[3], s.wei_buff_ptr[2])
            ic.v_add3_u32(v.in_off[:],v.in_off[:],v.in_off[:],v.in_off[:])


            #self.instr_ctrl._emmit_all(self._emit)

            ic.s_load_dwordx2(s.in_buff_ptr[0:1], s.karg_ptr[0:1], karg.in_ptr+0)
            ic.s_load_dwordx2(s.out_buff_ptr[0:1], s.karg_ptr[0:1], karg.out_ptr+0)
            ic.s_load_dwordx2(s.wei_buff_ptr[0:1], s.karg_ptr[0:1], karg.wei_ptr+0)
            ic.s_load_dwordx2(v.in_off[0:1], s.wei_buff_ptr[0:1], karg.wei_ptr+0)
            ic.v_add3_u32(v.in_off[:],v.in_off[:],v.in_off[:],v.in_off[:])

            ic.s_load_dwordx4(s.wei_buff_ptr[0:3], s.karg_ptr[0:1], karg.H+0)
            ic.s_load_dwordx2(s.Y[0:1], s.karg_ptr[0:1], karg.Y+0)

            def fill_buff_desc(desc_reg:regVar, size:int):
                ic.s_mov_b32(desc_reg[2], size)
                ic.s_mov_b32(desc_reg[3], 0x00027000)

            input_buffer_size = 25
            filter_buffer_size = 50
            output_buffer_size = 100

            fill_buff_desc(s.in_buff_ptr[:], input_buffer_size)
            fill_buff_desc(s.out_buff_ptr[:], filter_buffer_size)
            fill_buff_desc(s.wei_buff_ptr[:], output_buffer_size)

            ic.v_dot2_i32_i16(s.wei_buff_ptr[0], s.karg_ptr[1],s.wei_buff_ptr[1], s.karg_ptr[0])

            @mfunc_func
            def func_x1(self:kernel_func):
                s = self.sgpr_f
                H1O = s.add('H1O', 1, 4)
                H2O = s.add('H2O', 4, 4)

                H4O = s.add_no_pos('H30', 4)
                H31O = s.add_no_pos('H310', 1)

                block = s.add_block('block', [H4O, H31O])

                fill_buff_desc(H4O[:], 15)

            func_x1(self)
            H33O = s.add('H33O', 4, 2)

        launch_k(self.k_config, self.kargs)


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
        self.instructions_caller = conv_direct_navi.custom_caller(self.instr_ctrl.instructions_list)
