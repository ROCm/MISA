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

class inst_base(ABC):
    __slots__ = ['label', 'inst_type']
    def __init__(self, inst_type:instruction_type, label:str) -> None:
        self.inst_type:instruction_type = inst_type
        self.label = label

    @abstractmethod
    def __str__(self) -> str:
        return f'{self.label}'

    def emit(self) -> str:
        return self.__str__()

class inst_caller_base(ABC):
    __slots__ = ['ic']
    def __init__(self, insturction_container:List[inst_base]) -> None:
        self.ic = insturction_container

    def ic_pb(self, inst):
        self.ic.append(inst)
        return inst

class reg_allocator_base(inst_base):
    def __init__(self, reg:Union[reg_block, List[reg_block], tuple], alignment, alloc_f) -> None:
        super().__init__(instruction_type.REGALLOC, 'reg_alloc')
        self.reg = reg
        self.alloc_f = alloc_f
        self.alignment = alignment

    def __str__(self) -> str:
        return self.alloc_f(self.reg, self.alignment)

class reg_allocator_caller(inst_caller_base):
    def __init__(self, insturction_container: List[inst_base]) -> None:
        super().__init__(insturction_container)
    
    def reg_alloc(self, dst:reg_block, alignment:int, alloc_f):
        return self.ic_pb(reg_allocator_base(dst, alignment, alloc_f))
    
    def Block_alloc(self, dst:List[reg_block], block_offsets:List[int], alignment:int, alloc_f):
        return self.ic_pb(reg_allocator_base( (dst, block_offsets), alignment, alloc_f))

    def reg_dealloc(self, dst:reg_block, dealloc_f):
        return self.ic_pb(reg_allocator_base(dst, 0, dealloc_f))

#class gpu_instructions_caller(VOP1_instr_caller, VMEM_instr_caller, SMEM_instr_caller, SOP1_instr_caller):

from python.codegen.GFX10 import *
from python.codegen.GFX1011 import *

class gfx1011_1012(dpp16_instr_caller_gfx10Ex,dpp8_instr_caller_gfx10Ex,
 vop2_instr_caller_gfx10Ex, vop3p_instr_caller_gfx10Ex):
 def __init__(self, insturction_container) -> None:
        super().__init__(insturction_container)

class gpu_instructions_caller(dpp16_instr_caller, dpp8_instr_caller, ds_instr_caller,
    exp_instr_caller, flat_instr_caller, mimg_instr_caller, mtbuf_instr_caller,
    mubuf_instr_caller, sdwa_instr_caller, smem_instr_caller, sop1_instr_caller,
    sop2_instr_caller, sopc_instr_caller, sopk_instr_caller, sopp_instr_caller,
    vintrp_instr_caller, vop1_instr_caller, vop2_instr_caller, vop3_instr_caller,
    vop3p_instr_caller, vopc_instr_caller, reg_allocator_caller, gfx1011_1012):
    def __init__(self, insturction_container) -> None:
        super().__init__(insturction_container)

class instruction_ctrl():
    __slots__ = ['instructions_caller', 'instructions_list']

    def __init__(self) -> None:
        instructions_list:List[inst_base] = []
        self.instructions_caller = gpu_instructions_caller(instructions_list)
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