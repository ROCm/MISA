from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from ..base_components import inst_base
from ..generator_instructions import flow_control_caller, instr_label_caller, reg_allocator_caller

class gpu_instructions_caller_base(reg_allocator_caller, flow_control_caller, instr_label_caller):
    def __init__(self, insturction_list) -> None:
        super().__init__(insturction_list)

from ..generator_instructions import HW_Reg_Init
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
