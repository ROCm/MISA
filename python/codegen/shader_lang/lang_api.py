from abc import ABC, abstractmethod

from python.codegen.mc import mc_asm_printer_t, mc_base_t



class amdgpu_kernel_info_t(object):
    def __init__(self, kernel_code, kernel_name, kernel_block_size, kernel_args):
        self.kernel_code = kernel_code
        self.kernel_name = kernel_name
        self.kernel_block_size = kernel_block_size
        self.kernel_args = kernel_args

class base_lang_api(mc_base_t, ABC):    
    
    def _emit_kernel_header(self):
        pass

    def __init__(self, mc_asm_printer: mc_asm_printer_t, kernel_info:amdgpu_kernel_info_t, emmit_created_code):
        super().__init__(mc_asm_printer)
        self._kernel_info = kernel_info
        self._emmit_created_code = emmit_created_code
    

    def emit_kernel_code(self, kernel_obj, **kwargs):
        pass
