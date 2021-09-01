from abc import ABC, abstractmethod
from python.codegen.amdgpu import amd_kernel_code_t, amdgpu_kernel_code_t, amdgpu_kernel_info_t, hsa_kernel_header
from python.codegen.kernel_arg import _args_manager_t, karg_file_t
from python.codegen.gpu_reg import sgpr_file_t, vgpr_file_t
from ..codegen.mc import mc_base_t, mc_asm_printer_t

#public dep
from python.codegen.kernel_arg import arg_kind, arg_type


class kernel_constructor(mc_base_t, _args_manager_t, ABC):

    class kernel_karg_t(karg_file_t):
        '''Empty class, should be overwritten in child class'''
        def __init__(self, k_ptr, mc) -> None:
            super().__init__(mc)

    @abstractmethod
    def _set_kernel_karg_t(self):
        '''Should be called before get_amdgpu_metadata_list in kernel_constructor.__init__.
        Defined in kernel class to make overwrited kernel_karg_t trackable by IDE.'''
        self.kargs=self.kernel_karg_t(self, self.mc)

    def __init__(self, mc_asm_printer: mc_asm_printer_t, **kwargs):
        mc_base_t.__init__(self, mc_asm_printer)
        _args_manager_t.__init__(self)
        self._gpr_init(mc_asm_printer)
        self._set_kernel_karg_t()
        self.kernel_info = self._construct_kernel_info()
        
    
    def _construct_kernel_info(self) -> amdgpu_kernel_info_t:
        return amdgpu_kernel_info_t(
            kernel_code=self._get_kernel_code_obj_t(),
            kernel_args=self.get_amdgpu_metadata_list(),
            kernel_block_size=0, kernel_name=self.get_kernel_name())
    
    @abstractmethod
    def _gpr_init(self, mc :mc_asm_printer_t):
        '''Should be called before get_amdgpu_metadata_list in kernel_constructor.__init__.
        Defined in kernel class to make overwrited sgpr and vgpr trackable by IDE.'''
        self.sgpr = self._sgpr(mc)
        self.vgpr = self._vgpr(mc)
        self.agpr = self._agpr(mc)

    class _sgpr(sgpr_file_t):
        def __init__(self, mc):
            super().__init__(mc)
            
    class _vgpr(vgpr_file_t):
        def __init__(self, mc):
            super().__init__(mc)
    
    class _agpr(vgpr_file_t):
        def __init__(self, mc):
            super().__init__(mc)

    def _get_kernel_code_obj_t(self) -> amdgpu_kernel_code_t:
        ''' 
        Set .amd_kernel_code_t for kernel metadata
        '''
        kernel_code_dict = {
                'enable_sgpr_kernarg_segment_ptr'   :   1,
                'enable_sgpr_workgroup_id_x'        :   1,
                'enable_sgpr_workgroup_id_y'        :   1,
                'enable_vgpr_workitem_id'           :   0,
                'workgroup_group_segment_byte_size' :   self._get_LDS_usage(),
                'kernarg_segment_byte_size'         :   self._get_kernel_arg_byte_size(),
                'wavefront_sgpr_count'              :   self.sgpr.get_count() + 2*3,
                'workitem_vgpr_count'               :   self.vgpr.get_count()}
        
        kernel_code_dict['accum_offset']        =   self.vgpr.get_count()
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
        self.kargs.emit()
        self._emit_empty_line()
        self.sgpr.emit()
        self._emit_empty_line()
        self.vgpr.emit()
        self._emit_empty_line()
        self.agpr.emit()
        self._emit_empty_line()

    def emit_kernel_code(self):
        self._emit_kernel_header()
        self._emit_kernel_symbols()
        self._emit_kernel_body()
        self._emit_kernel_end()

    def emit_kernel_amd_kernel_code_t(self):
        amd_kernel_code_t(self.mc, self.kernel_info).emit()
