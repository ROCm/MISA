from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import inspect
from python.codegen.generator_instructions import flow_control_caller, reg_allocator_caller
from typing import List
from python.codegen.gpu_data_types import reg_block, abs, neg

class inst_mode(Enum):
    allocation: auto
    emmition: auto

class instruction_type(Enum):
    SMEM = 'SMEM'
    VMEM = 'VMEM'
    VOP1 = 'VOP1'
    REGALLOC = 'REGA'
    REGDEALLOC = 'REGD'
    BLOCKALLOC = 'BLCKA'
    BLOCKSPLIT = 'BLCKS'
    #BLOCKDEALLOC = 'BLCKD'
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

from python.codegen.generator_instructions import flow_control_caller, reg_allocator_caller
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
