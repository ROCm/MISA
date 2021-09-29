from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Union
from python.codegen.gpu_data_types import reg_block, abs, neg

class inst_mode(Enum):
    allocation: auto
    emmition: auto

class instruction_type(Enum):
    SMEM = 'SMEM'
    VMEM = 'VMEM'
    VOP1 = 'VOP1'
    REGALLOC = 'REGA'
    REGDEALLOC = 'REGA'
    FLOW_CONTROL = 'FC'

class inst_base(ABC):
    __slots__ = ['label', 'inst_type', 'trace']
    def __init__(self, inst_type:instruction_type, label:str) -> None:
        self.inst_type:instruction_type = inst_type
        self.label = label
        self.trace = ''

    @abstractmethod
    def __str__(self) -> str:
        return f'{self.label}'

    def emit(self) -> str:
        return self.__str__()

    def emit_trace(self) -> str:
        return self.trace

class inst_caller_base(ABC):
    __slots__ = ['il']
    def __init__(self, insturction_list:List[inst_base]) -> None:
        self.il = insturction_list

    def ic_pb(self, inst):
        self.il.append(inst)
        return inst

class reg_allocator_base(inst_base):
    def __init__(self, reg:Union[reg_block, List[reg_block], tuple], alignment, allocator_f) -> None:
        super().__init__(instruction_type.REGALLOC, 'reg_alloc')
        self.reg = reg
        self.allocator_f = allocator_f
        self.alignment = alignment

    def __str__(self) -> str:
        return self.allocator_f(self.reg, self.alignment)

class reg_allocator_caller(inst_caller_base):
    def __init__(self, insturction_list: List[inst_base]) -> None:
        super().__init__(insturction_list)
    
    def reg_alloc(self, dst:reg_block, alignment:int, alloc_f):
        return self.ic_pb(reg_allocator_base(dst, alignment, alloc_f))
    
    def Block_alloc(self, dst:List[reg_block], block_offsets:List[int], alignment:int, alloc_f):
        return self.ic_pb(reg_allocator_base( (dst, block_offsets), alignment, alloc_f))

    def reg_dealloc(self, dst:reg_block, dealloc_f):
        return self.ic_pb(reg_allocator_base(dst, 0, dealloc_f))

#class gpu_instructions_caller(VOP1_instr_caller, VMEM_instr_caller, SMEM_instr_caller, SOP1_instr_caller):

class flow_control_base(inst_base):
    def __init__(self, label) -> None:
        super().__init__(instruction_type.FLOW_CONTROL, 'label')
        self.label = label

    def __str__(self) -> str:
        return f'//{self.label}'

class flow_control_caller(inst_caller_base):
    def __init__(self, insturction_list: List[inst_base]) -> None:
        super().__init__(insturction_list)
    
    def kernel_func_begin(self, func_name):
        return self.ic_pb(flow_control_base(f'{func_name}.begin()'))
    
    def kernel_func_end(self, func_name):
        return self.ic_pb(flow_control_base(f'{func_name}.end()'))


class gpu_instructions_caller_base(reg_allocator_caller, flow_control_caller):
    def __init__(self, insturction_list) -> None:
        super().__init__(insturction_list)

class instruction_ctrl():
    __slots__ = ['instructions_list']

    def __init__(self) -> None:
        instructions_list:List[inst_base] = []
        self.instructions_list = instructions_list
    
    def _emmit_all(self, emmiter):
        e = emmiter
        for i in self.instructions_list:
            e(f'{i}')
    
    def _emmit_range(self, emmiter, strt:int, end:int):
        e = emmiter
        i_list = self.instructions_list
        for i in i_list:
            e(f'{i}')
