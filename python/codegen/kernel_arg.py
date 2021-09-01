from abc import ABC
from enum import Enum
from python.codegen.mc import mc_base_t
from python.codegen.amdgpu import amdgpu_kernel_arg_t
from python.codegen.symbol import sym_t
from typing import List

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

class karg_file_t(mc_base_t, ABC):
    '''base class, should be overwritten in child class'''
    def __init__(self, mc) -> None:
        mc_base_t.__init__(self, mc)
        #self.sample = k_ptr._pb_kernel_arg('sample', arg_kind.value, arg_type.I32)

    def emit(self):
        for k, v in self.__dict__.items():
            self._emit(v.declare())