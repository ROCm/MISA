from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import inspect
from typing import Dict, List

class inst_mode(Enum):
    allocation: auto
    emmition: auto

class instruction_type(Enum):
    SMEM = 'SMEM'
    VMEM = 'VMEM'
    VOP1 = 'VOP1'
    VOP2 = 'VOP2'
    VOP3 = 'VOP3'
    VOP3P = 'VOP3P'
    VOPC = 'VOPC'
    SOP2 = 'SOP2'
    SOP1 = 'SOP1'
    VINTRP='VINTRP'
    DPP16 = 'DPP16'
    DPP8 = 'DPP8'
    DS = 'DS'
    EXP = 'EXP'
    MIMG = 'MIMG'
    MTBUF = 'MTBUF'
    MUBUF = 'MUBUF'
    SDWA = 'SDWA'
    SOPC = 'SOPC'
    SOPK = 'SOPK'
    SOPP = 'SOPP'
    FLAT = 'FLAT'
    REGALLOC = 'REGA'
    REGDEALLOC = 'REGD'
    REGREUSE = 'REGREUSE'
    BLOCKALLOC = 'BLCKA'
    BLOCKSPLIT = 'BLCKS'
    #BLOCKDEALLOC = 'BLCKD'
    FLOW_CONTROL = 'FC'
    HW_REG_INIT= "HW_INIT"


dst_atrs = ['DST', 'DST0','DST1', 'DST2']
src_atrs = ['SRC', 'SRC0', 'SRC1', 'SRC2', 'SRC3']

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
    
    def get_srs_regs(self):
        ret = []
        for srs_a in src_atrs:
            a = getattr(self, srs_a, None)
            if(a):
                ret.append(a)
        return ret

    def get_dst_regs(self):
        ret = []
        for dst_a in dst_atrs:
            a = getattr(self, dst_a, None)
            if(a):
                ret.append(a)
        return ret

class inst_caller_base(ABC):
    __slots__ = ['il']
    def __init__(self, insturction_list:List[inst_base]) -> None:
        self.il = insturction_list

    def ic_pb(self, inst):
        self.il.append(inst)
        st = inspect.stack()
        return inst

from python.codegen.generator_instructions import HW_Reg_Init, flow_control_caller, instr_label_caller, reg_allocator_caller
class gpu_instructions_caller_base(reg_allocator_caller, flow_control_caller, instr_label_caller):
    def __init__(self, insturction_list) -> None:
        super().__init__(insturction_list)


class instruction_ctrl():
    __slots__ = ['instructions_list', 'code_str']

    def __init__(self) -> None:
        hw_init = HW_Reg_Init()
        instructions_list:List[inst_base] = [hw_init]
        self.instructions_list = instructions_list

        self.code_str = []
    
    def get_HW_Reg_Init(self)->HW_Reg_Init:
        for i in self.instructions_list:
            if (type(i) is HW_Reg_Init):
                return i
        assert(False)

    def execute_all(self):
        self._emmit_all(self.code_str.append)


    def _emmit_all(self, emmiter):
        e = emmiter
        for i in self.instructions_list:
            e(f'{i}')
    
    def _emmit_created_code(self, emmiter):
        e = emmiter
        for i in self.code_str:
            e(i)

    def _emmit_range(self, emmiter, strt:int, end:int):
        e = emmiter
        i_list = self.instructions_list
        for i in i_list:
            e(f'{i}')
