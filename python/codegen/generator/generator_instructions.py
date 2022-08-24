from typing import Callable, List, Optional, Union

from .gpu_arch.gpu_data_types import block_of_reg_blocks, label_t, reg_block
from .base_components import inst_base, inst_caller_base, instruction_type


class reg_allocator_base(inst_base):
    def __init__(self, inst_type:instruction_type, reg:Union[reg_block, List[reg_block], block_of_reg_blocks, tuple], alignment, allocator_f, print_f) -> None:
        super().__init__(inst_type, 'reg_alloc')
        self.reg = reg
        self.allocator_f = allocator_f
        self.alignment = alignment
        self.print_f = print_f

    def __str__(self) -> str:
        return self.print_f(self.reg)

    def execute(self):
        self.allocator_f(self.reg, self.alignment)

class reg_allocator_caller(inst_caller_base):
    def __init__(self, insturction_list: List[inst_base]) -> None:
        super().__init__(insturction_list)
    
    def reg_alloc(self, dst:reg_block, alignment:int, alloc_f, print_f=None):
        return self.ic_pb(reg_allocator_base(instruction_type.REGALLOC, dst, alignment, alloc_f, print_f))
    
    def Block_alloc(self, dst:List[reg_block], block_offsets:List[int], alignment:int, alloc_f, print_f=None):
        return self.ic_pb(reg_allocator_base( instruction_type.BLOCKALLOC, (dst, block_offsets), alignment, alloc_f, print_f))

    def Block_split(self, block_of_reg_blocks, split_f, print_f=None):
        return self.ic_pb(reg_allocator_base( instruction_type.BLOCKSPLIT, block_of_reg_blocks, 0, split_f, print_f))

    def reg_dealloc(self, dst:reg_block, dealloc_f, print_f=None):
        return self.ic_pb(reg_allocator_base(instruction_type.REGDEALLOC, dst, 0, dealloc_f, print_f))
    
    def reg_pos_reuse(self, src:reg_block, dst:reg_block, reuse_f, print_f=None):
        return self.ic_pb(reg_allocator_base(instruction_type.REGREUSE, (src, dst), 0, reuse_f, print_f))

class flow_control_base(inst_base):
    def __init__(self, label, func_ptr) -> None:
        super().__init__(instruction_type.FLOW_CONTROL, label)
        self.func_ptr = func_ptr

    def __str__(self) -> str:
        return f'{self.label}'

    def emit(self, emiter_f:Callable[[inst_base, any], None]):
        emiter_f(self, True)

class flow_control_caller(inst_caller_base):
    def __init__(self, insturction_list: List[inst_base]) -> None:
        super().__init__(insturction_list)
    
    def kernel_func_begin(self, func_ptr):
        return self.ic_pb(flow_control_base(f'{func_ptr.func_name}.begin()', func_ptr))
    
    def kernel_func_end(self, func_ptr):
        return self.ic_pb(flow_control_base(f'{func_ptr.func_name}.end()', func_ptr))
    
class HW_Reg_Init(inst_base):
    def __init__(self) -> None:
        super().__init__(instruction_type.HW_REG_INIT, 'HW_REG_INIT')
        self.dst_regs = []

    def __str__(self) -> str:
        return f'{self.label}'

    def emit(self, emiter_f:Callable[[inst_base], None]):
        return

    def get_srs_regs(self):
        return []

    def get_dst_regs(self):
        return self.dst_regs

class instr_label_base(inst_base):
    def __init__(self, instr_label:label_t) -> None:
        super().__init__(instruction_type.FLOW_CONTROL, 'label')
        self.instr_label = instr_label

    def __str__(self) -> str:
        return f'{self.instr_label.define()}'

class instr_label_caller(inst_caller_base):
    def __init__(self, insturction_list: List[inst_base]) -> None:
        super().__init__(insturction_list)
    
    def set_label(self, label:label_t):
        return self.ic_pb(instr_label_base(label))

