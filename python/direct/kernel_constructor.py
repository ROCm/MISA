from abc import ABC, abstractmethod
from python.codegen.kernel_func import kernel_launcher
from python.codegen.gpu_arch.allocator import stack_allocator
from python.codegen.gpu_arch.HW_components import base_HW
from typing import List, Type
from python.codegen.gpu_instruct import gpu_instructions_caller_base, instruction_ctrl
from python.codegen.amdgpu import amd_kernel_code_t, amdgpu_kernel_code_t, amdgpu_kernel_info_t, hsa_kernel_header
from python.codegen.kernel_arg import _args_manager_t, karg_file_t
from ..codegen.mc import mc_base_t, mc_asm_printer_t

#public dep
from python.codegen.kernel_arg import arg_kind, arg_type


class kernel_constructor(mc_base_t, ABC):

    class kernel_karg_t(karg_file_t):
        '''Empty class, should be overwritten in child class'''
        def __init__(self, mc) -> None:
            super().__init__(mc)

    @abstractmethod
    def _set_kernel_karg_t(self):
        '''Should be called before get_amdgpu_metadata_list in kernel_constructor.__init__.
        Defined in kernel class to make overwrited kernel_karg_t trackable by IDE.'''
        self.kargs=self.kernel_karg_t(self.mc)

    def generate_kernel_body(self):
        self._emit_kernel_body()
        self.HW.ABI_HW_setregs()
        self.instr_ctrl.plot_the_graph()
        self.instr_ctrl.execute_all()
        #some optimize

    def __init__(self, mc_asm_printer: mc_asm_printer_t, **kwargs):
        mc_base_t.__init__(self, mc_asm_printer)
        self.instr_ctrl = instruction_ctrl()
        self._instructions_init()
        self.set_GPU_HW()
        t = type(self.instructions_caller)
        self.k_config = kernel_launcher[t](self.instructions_caller, self.HW)
        self._set_kernel_karg_t()
        self.generate_kernel_body()
        self.kernel_info = self._construct_kernel_info()

    @abstractmethod    
    def set_GPU_HW(self):
        self.HW = base_HW(self.instructions_caller, stack_allocator, stack_allocator, 104, 256, 65000)
        
    def _construct_kernel_info(self) -> amdgpu_kernel_info_t:
        return amdgpu_kernel_info_t(
            kernel_code=self._get_kernel_code_obj_t(),
            kernel_args=self.kargs.get_amdgpu_metadata_list(),
            kernel_block_size=0, kernel_name=self.get_kernel_name())
    
    @abstractmethod
    def _instructions_init(self):
        '''Defined in kernel class to make overwrited sgpr and vgpr trackable by IDE.'''
        self.instructions_caller = gpu_instructions_caller_base(self.instr_ctrl.instructions_list)

    def _get_kernel_code_obj_t(self) -> amdgpu_kernel_code_t:
        ''' 
        Set .amd_kernel_code_t for kernel metadata
        '''
        kernel_code_dict = {
                'enable_sgpr_kernarg_segment_ptr'   :   self.HW._special_reg_used.get('karg_segment_ptr', 0),
                'enable_sgpr_workgroup_id_x'        :   self.HW._special_reg_used.get('gid_x', 0),
                'enable_sgpr_workgroup_id_y'        :   self.HW._special_reg_used.get('gid_y', 0),
                'enable_sgpr_workgroup_id_z'        :   self.HW._special_reg_used.get('gid_z', 0),
                'enable_vgpr_workitem_id'           :   sum(
                                                            map(
                                                                lambda x: self.HW._special_reg_used.get(x, 0),
                                                                ['tid_x', 'tid_y', 'tid_z']
                                                            )
                                                        ) - 1,
                'workgroup_group_segment_byte_size' :   self._get_LDS_usage(),
                'kernarg_segment_byte_size'         :   self.kargs._get_arg_byte_size(),
                # self.HW.sgpr_alloc.get_required_size() + VCC, FLAT_SCRATCH and XNACK 
                'wavefront_sgpr_count'              :   self.HW.sgpr_alloc.get_required_size() + 2*3,
                'workitem_vgpr_count'               :   self.HW.vgpr_alloc.get_required_size()}
        
        kernel_code_dict['accum_offset']        =   self.HW.vgpr_alloc.get_required_size()
        kernel_code = amdgpu_kernel_code_t(kernel_code_dict)
        return kernel_code

    @abstractmethod
    def get_kernel_name(self) -> str:
        return str('base')

    @abstractmethod
    def _get_LDS_usage(self):
        return 0

    def _emit_kernel_header(self):
        hsa_kernel_header(mc=self.mc, amdgpu_kernel_info=self.kernel_info).emit()

    @abstractmethod
    def _emit_kernel_body(self):
        pass

    def _emit_kernel_end(self):
        self._emit('s_endpgm')

    def emit_kernel_footer(self):
        self._emit_empty_line()

    def _emit_kernel_symbols(self):
        self.kargs.emit_symb()
        self._emit_empty_line()

    def emit_kernel_code(self):
        self._emit_kernel_header()
        self._emit_kernel_symbols()
        self.instr_ctrl._emmit_created_code(self._emit)
        self._emit_kernel_end()

    def emit_kernel_amd_kernel_code_t(self):
        amd_kernel_code_t(self.mc, self.kernel_info).emit()
