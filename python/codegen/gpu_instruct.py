from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Union
from python.codegen.gpu_reg_block import reg_block

class inst_mode(Enum):
    allocation: auto
    emmition: auto

class instruction_type(Enum):
    SMEM = 'SMEM'
    VMEM = 'VMEM'
    VOP1 = 'VOP1'

class inst_base(ABC):
    __slots__ = ['label', 'inst_type']
    def __init__(self, inst_type:instruction_type, label:str) -> None:
        self.inst_type:instruction_type = inst_type
        self.label = label
class SMEM_base(inst_base):
    __slots__ = ['sdata','sbase','soffset', 'glc_dlc']
    def __init__(self,
            label: str, sdata:reg_block, sbase:reg_block, soffset=0,
            glc:bool = False, dlc:bool = False
            ) -> None:
        super().__init__(instruction_type.SMEM, label)
        self.sdata = sdata
        self.sbase = sbase
        self.soffset = soffset
        self.glc_dlc = [' glc' if glc else '', ' dlc' if dlc else '']
    
    def emit(self):
        return f'{self.label} {self.sdata}, {self.sbase}, 0+{self.soffset}{self.glc_dlc[0]}{self.glc_dlc[1]}'

class SMEM_instr_caller():
    
    def s_load_dword(self, dwords_cnt:int, sdata:reg_block, sbase:reg_block, soffset=0, glc=False, dlc=False) -> SMEM_base:
        label = f's_load_dwordx{dwords_cnt}' if dwords_cnt > 1 else 's_load_dword'
        return SMEM_base(label, sdata, sbase, soffset)

    def s_buffer_load_dword(self, dwords_cnt:int, sdst:reg_block, sbase:reg_block, soffset=0, glc=False, dlc=False) -> SMEM_base:
        label = f's_buffer_load_dwordx{dwords_cnt}' if dwords_cnt > 1 else 's_buffer_load_dword'
        return SMEM_base(label, sdst, sbase, soffset, glc, dlc)

    def s_buffer_store_dword(self, dwords_cnt:int, sdata:reg_block, sbase:reg_block, soffset=0, glc=False) -> SMEM_base:
        label = f's_buffer_store_dwordx{dwords_cnt}' if dwords_cnt > 1 else 's_buffer_store_dword'
        return SMEM_base(label, sdata, sbase, soffset)
