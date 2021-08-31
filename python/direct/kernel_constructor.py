from abc import ABC, abstractmethod
from typing import List

from ..codegen.mc import mc_base_t, mc_asm_printer_t
from ..codegen.amdgpu import *
from ..codegen.symbol import *

from enum import Enum
class arg_type(Enum):
    I64 = 'I64'
    F64 = 'F64'
    I32 = 'I32'
    F32 = 'F32'
    F16 = 'F16'
    I8 = 'I8'
    F8 = 'F8'

class arg_kind(Enum):
    GlobBuffer = 'global_buffer'
    value = 'by_value'

#@dataclass
class _singl_arg():
    __slots__ = ['name', 'size', 'offset', 'value_kind', 'value_type', 'address_space', 'is_const']
    def __init__(self, name, size, offset, value_kind, value_type, **misc) -> None:
        self.name = name
        self.size = size
        self.offset = offset
        self.value_kind = value_kind
        self.value_type = value_type
        self.address_space = getattr(misc, 'address_space', None)
        self.is_const = getattr(misc, 'is_const', None)
    
    def get_amdgpu_arg(self) -> amdgpu_kernel_arg_t:
        misc = {'address_space':self.address_space, 'is_const':self.is_const}
        return amdgpu_kernel_arg_t(self.name, self.size, self.offset, self.value_kind, self.value_type, **misc)
        
class _args_manager_t(ABC):
    #__slots__ = ['args_size', 'args_list']
    def __init__(self) -> None:
        self.args_size = 0
        self.args_list:List[_singl_arg] = []
    
    def _get_arg_type_size(self, value_kind:arg_kind, value_type:arg_type) -> int:
        if value_kind == arg_kind.GlobBuffer:
            return 8
        else:
            if value_type in [arg_type.F8, arg_type.I8]:
                return 1
            elif value_type in [arg_type.F16]:
                return 2
            elif value_type in [arg_type.F32, arg_type.I32]:
                return 4
            elif value_type in [arg_type.F64, arg_type.I64]:
                return 8
            else:
                assert False

    def _get_new_offset(self, val_size:int) -> int:
            alignment    = val_size
            padding      = (alignment - (self.args_size % alignment)) % alignment
            return self.args_size + padding

    def _pb_kernel_arg(self, name:str, value_kind:arg_kind, value_type:arg_type, **misc) ->sym_t:
        '''Add new kernel_argument record to args_list.'''
        assert type(value_kind) is arg_kind
        assert type(value_type) is arg_type
        
        val_size = self._get_arg_type_size(value_kind, value_type)
        offset = self._get_new_offset(val_size)
        last_arg = _singl_arg(name, val_size, offset, value_kind, value_type, **misc)

        self.args_list.append(last_arg)
        self.args_size = offset + val_size

        symbol = sym_t(name, offset)
        return symbol

    def get_amdgpu_metadata_list(self):
        '''Create list of arguments for amdgpu_metadata. As source used args_list '''
        meta_args:List[amdgpu_kernel_arg_t] = []
        for i in self.args_list:
            #meta_args.append(i.get_amdgpu_arg().serialize_as_metadata())
            meta_args.append(i.get_amdgpu_arg())
        return meta_args
    
    def _get_kernel_arg_byte_size(self) -> int:
        return self.args_size

class gpr_off_sequencer_t(object):
    def __init__(self, offset = 0):
        self.start_offset = offset
    def __call__(self, step = 0, alignment = 0):
        previous_offset = self.start_offset
        if alignment:
            aligned_offset = ((previous_offset + alignment - 1) // alignment) * alignment
            self.start_offset = aligned_offset
            previous_offset = aligned_offset
        self.start_offset += step
        return previous_offset
    def get_start_offset(self):
        return self.start_offset

class gpr_file_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
        self._sq = gpr_off_sequencer_t()
    
    def get_count(self):
        return self._sq.get_start_offset()
    
    def emit(self, pref):
        _end_val = self._sq.get_start_offset()
        assert _end_val <= amdgpu_sgpr_limit(self.mc.arch_config.arch), f"{pref}end:{_end_val} "
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                self._emit(pref+v.declare())

class sgpr_file_t(gpr_file_t):
    def __init__(self, mc):
        super().__init__(mc)
    
    def emit(self):
        super().emit('s_')

class vgpr_file_t(gpr_file_t):
    def __init__(self, mc):
        super().__init__(mc)

    def emit(self):
        super().emit('v_')

class karg_file_t(mc_base_t, ABC):
    '''base class, should be overwritten in child class'''
    def __init__(self, mc) -> None:
        mc_base_t.__init__(self, mc)
        #self.sample = k_ptr._pb_kernel_arg('sample', arg_kind.value, arg_type.I32)

    def emit(self):
        for k, v in self.__dict__.items():
            self._emit(v.declare())

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
