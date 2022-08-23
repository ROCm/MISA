from abc import ABC, abstractmethod
from python.codegen.generator.base_components import inst_base, all_atrs
from python.codegen.mc import mc_asm_printer_t, mc_base_t



class amdgpu_kernel_info_t(object):
    def __init__(self, kernel_code, kernel_name, kernel_block_size, kernel_args):
        self.kernel_code = kernel_code
        self.kernel_name = kernel_name
        self.kernel_block_size = kernel_block_size
        self.kernel_args = kernel_args

class base_lang_class(mc_base_t, ABC):    
    
    def __init__(self, mc_asm_printer: mc_asm_printer_t, emmit_created_code, real_variable_mode=True):
        super().__init__(mc_asm_printer)
        self._emmit_created_code = emmit_created_code
        self._set_variable_mod(real_variable_mode)
    
    def _set_variable_mod(self, real_variable_mode):
        if(real_variable_mode):
            self.variable_print_mode = ''
        else:
            self.variable_print_mode = '#'

    @abstractmethod
    def _emit_kernel_header(self):
        pass

    @abstractmethod
    def create_variable(self, name:str, value=0):
        pass
    
    @abstractmethod
    def set_variable_val(self, name:str, value):
        pass

    @abstractmethod
    def unset_variable(self, name:str):
        pass
    
    @abstractmethod
    def emit_kernel_code(self, kernel_info:amdgpu_kernel_info_t, **kwargs):
        pass

    def emit_instruction(self, instr:inst_base, is_comment:bool=False):
        print_mode = self.variable_print_mode if is_comment else ''
        str_op = getattr(instr, "__str__", None)
        
        if callable(str_op):
            self._emit(f'{print_mode}{str_op()}')
        else:
            curent_args = map(lambda atr:getattr(instr, atr, None), all_atrs)
            curent_args = filter(None.__ne__, curent_args)
            self._emit(f"{print_mode}{self.label} {', '.join(map(str, curent_args))} {getattr(instr, 'MODIFIERS', '')}")
        
