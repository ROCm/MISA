

from python.codegen.gpu_data_types import block_of_reg_blocks, reg_block
from python.codegen.gpu_instruct import inst_base, inst_caller_base, instruction_type
from typing import List, Optional, Union

class reg_allocator_base(inst_base):
    def __init__(self, inst_type:instruction_type, reg:Union[reg_block, List[reg_block], block_of_reg_blocks, tuple], alignment, allocator_f) -> None:
        super().__init__(inst_type, 'reg_alloc')
        self.reg = reg
        self.allocator_f = allocator_f
        self.alignment = alignment

    def __str__(self) -> str:
        return self.allocator_f(self.reg, self.alignment)


class reg_allocator_caller(inst_caller_base):
    def __init__(self, insturction_list: List[inst_base]) -> None:
        super().__init__(insturction_list)
    
    def reg_alloc(self, dst:reg_block, alignment:int, alloc_f):
        return self.ic_pb(reg_allocator_base(instruction_type.REGALLOC, dst, alignment, alloc_f))
    
    def Block_alloc(self, dst:List[reg_block], block_offsets:List[int], alignment:int, alloc_f):
        return self.ic_pb(reg_allocator_base( instruction_type.BLOCKALLOC, (dst, block_offsets), alignment, alloc_f))

    def Block_split(self, block_of_reg_blocks, split_f):
        return self.ic_pb(reg_allocator_base( instruction_type.BLOCKSPLIT, block_of_reg_blocks, 0, split_f))

    def reg_dealloc(self, dst:reg_block, dealloc_f):
        return self.ic_pb(reg_allocator_base(instruction_type.REGDEALLOC, dst, 0, dealloc_f))

class flow_control_base(inst_base):
    def __init__(self, label, func_ptr) -> None:
        super().__init__(instruction_type.FLOW_CONTROL, 'label')
        self.label = label
        self.func_ptr = func_ptr

    def __str__(self) -> str:
        return f'//{self.label}'

class flow_control_caller(inst_caller_base):
    def __init__(self, insturction_list: List[inst_base]) -> None:
        super().__init__(insturction_list)
    
    def kernel_func_begin(self, func_ptr):
        return self.ic_pb(flow_control_base(f'{func_ptr.func_name}.begin()', func_ptr))
    
    def kernel_func_end(self, func_ptr):
        return self.ic_pb(flow_control_base(f'{func_ptr.func_name}.end()', func_ptr))
    
class HW_Reg_Init(inst_base):
    def __init__(self) -> None:
        super().__init__(instruction_type.HW_REG_INIT, 'label')
        self.label = 'HW_REG_INIT'
        self.dst_regs = []

    def __str__(self) -> str:
        return f'//{self.label}'

    def get_srs_regs(self):
        return []

    def get_dst_regs(self):
        return self.dst_regs