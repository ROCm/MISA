from abc import ABC, abstractmethod
from python.codegen.kernel_driver import base_config
from ..codegen.config_parser import config_content_t
from ..codegen.mc import mc_base_t, mc_asm_printer_t
from .kernel_constructor import *


class direct_navi_config(base_config):
    def __init__(self, config_content: config_content_t):
        super().__init__(config_content, '-navi')
        
        self.read_size = config_content.get_section('direct-navi')[0]['read_size']

class conv_direct_navi(kernel_constructor):

    def get_kernel_name(self):
        return 'conv_direct_navi'

    class kernel_karg_t(karg_file_t):
        '''Define kernel arguments. Used in _set_kernel_karg_t'''
        def __init__(self, k_ptr:kernel_constructor, mc) -> None:
            super().__init__(mc)
            self.in_buff = k_ptr._pb_kernel_arg('in_buff', arg_kind.GlobBuffer, arg_type.F32)    
            self.H = k_ptr._pb_kernel_arg('H', arg_kind.value, arg_type.I32)
            self.W = k_ptr._pb_kernel_arg('W', arg_kind.value, arg_type.I32)

    class _sgpr(sgpr_file_t):
        def __init__(self, mc):
            super().__init__(mc)
            sseq = self._sq
            add = self.add 
            self.karg_ptr = add('karg_ptr', sseq(2))
            self.group_id = add('group_id', sseq(1))
            self.gid_off = add('gid_off', sseq(1))
            self.in_buff_ptr = add('in_buff_ptr', sseq(1))
            self.H = add('H', sseq(1))
            self.W = add('W', sseq(1))
    
    class _vgpr(vgpr_file_t):
        def __init__(self, mc):
            super().__init__(mc)
            sseq = self._sq
            add = self.add 
            self.tid = add('tid', sseq(1))
            self.in_off = add('in_off', sseq(1))
            self.in_off = add('in_off', sseq(1))

    def _get_LDS_usage(self):
        return 0

    def __init__(self, mc_asm_printer: mc_asm_printer_t, **kwargs):
        print(conv_direct_navi.__mro__)
        super().__init__(mc_asm_printer, **kwargs)
                
        
    def _emit_kernel_body(self):
        s = self.sgpr
        v = self.vgpr
        karg = self.kargs
        
        self._emit(f"s_load_dwordx2  s[{s.in_buff_ptr(0, 1)}],    s[{s.karg_ptr(0, 1)}],    0+{karg.in_buff()}")
        self._emit(f"s_load_dwordx  s[{s.H()}],   s[{s.karg_ptr(0, 1)}],    0+{karg.H()}")
        self._emit(f"s_load_dwordx4  {s.in_buff_ptr[0, 3]},  {s.karg_ptr[0, 1]},    0+{karg.in_buff()}")
        self._emit(f"s_load_dwordx  {s.H[0]},   {s.karg_ptr[0, 1]},    0+{karg.H()}")


    def _set_kernel_karg_t(self) -> None:
        '''Should be called before get_amdgpu_metadata_list in kernel_constructor.__init__.
        Defined in kernel class to make overwrited kernel_karg_t trackable by IDE.'''
        self.kargs=self.kernel_karg_t(self, self.mc)
    
    def _gpr_init(self, mc :mc_asm_printer_t) -> None:
        '''Should be called before get_amdgpu_metadata_list in kernel_constructor.__init__.
        Defined in kernel class to make overwrited sgpr and vgpr trackable by IDE.'''
        
        self.sgpr = self._sgpr(mc)
        self.vgpr = self._vgpr(mc)
        self.agpr = self._agpr(mc)
