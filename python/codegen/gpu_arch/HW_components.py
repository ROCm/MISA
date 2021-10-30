from enum import Enum
from python.codegen.generator_instructions import HW_Reg_Init, reg_allocator_caller
from python.codegen.gpu_arch.allocator import base_allocator, stack_allocator
from typing import Dict, List
from python.codegen.amdgpu import amdgpu_sgpr_limit
from python.codegen.gpu_data_types import *
import copy

class gpr_off_sequencer_t(object):
    def __init__(self, offset = 0):
        self._last_pos = offset
    def __call__(self, step = 0, alignment = 0):
        previous_offset = self._last_pos
        if alignment:
            aligned_offset = ((previous_offset + alignment - 1) // alignment) * alignment
            self._last_pos = aligned_offset
            previous_offset = aligned_offset
        self._last_pos += step
        return previous_offset
    def get_last_pos(self):
        return self._last_pos

from python.codegen.gpu_instruct import inst_caller_base
from python.codegen.gpu_arch.allocator import stack_allocator

from typing import Type
from python.codegen.gpu_arch.allocator import base_allocator

class gpr_file_t():#mc_base_t):
    __slots__ = ['_allocator', 'reg_t', 'define_on_creation', 'ic', 'active_blocks']
    def __init__(self, ic:inst_caller_base, reg_t:reg_type, gpr_allocator:base_allocator=None):
        #mc_base_t.__init__(self, mc)
        self._allocator = gpr_allocator
        self.reg_t = reg_t
        self.define_on_creation = False
        self.ic = reg_allocator_caller(ic.il)
        self.active_blocks:List[reg_block] = []
    
    @classmethod
    def create_from_other_inst(cls, other):
        return cls(other.ic, other.reg_t, other.gpr_allocator)

    def get_count(self):
        return self._allocator.get_required_size()
       
    def _alloc(self, reg:reg_block, alignment):
        reg.set_position(self._allocator.malloc(reg.dwords, alignment))
        return f'.set {reg.label}, {reg.position}'

    def _alloc_block(self, block_info:tuple, alignment):
        s = []
        regs:List[reg_block] = block_info[0]
        ofsets:List[int]  = block_info[1]
        base_reg = regs[0]
        base_reg.set_position(self._allocator.malloc(base_reg.dwords, alignment))
        s.append(f'.set {base_reg.label}, {base_reg.position}')
        base_reg_pos = base_reg.position
        
        for i in range(len(ofsets)):
            cur_reg = regs[i+1]
            cur_reg.set_position(base_reg_pos + ofsets[i])
            s.append(f'.set {cur_reg.label}, {cur_reg.position}')

        return '\n'.join(s)

    def _dealloc(self, reg:reg_block, alignment):
        #self.active_blocks.remove(reg)
        self._allocator.mfree(reg.position)
        return f'//dealock .unset {reg.label}, {reg.position}'

    def _dealloc_all(self):
        blocks = self.active_blocks
        dealoc = self._dealloc
        return [dealoc(i,0) for i in blocks]

    def add(self, label:str, dwords:int = 1, alignment:int = 0):
        ret = reg_block.declare(label, self.reg_t, dwords=dwords)
        self.active_blocks.append(ret)
        self.ic.reg_alloc(ret, alignment, self._alloc)
        return ret

    def add_no_pos(self, label:str, dwords:int = 1):
        return reg_block.declare(label, self.reg_t, dwords=dwords)

    def add_block(self, label:str, reg_list:List[reg_block], alignment:int = 0) ->block_of_reg_blocks:
        in_block_define = gpr_off_sequencer_t()
        block_pos = []
        block_regs = []
        for i in reg_list:
            assert i.reg_t == self.reg_t, f" reg_t of element {i} doesn't match the block reg_t"
            block_pos.append(in_block_define(i.dwords))

        block = block_of_reg_blocks.declare(label, self.reg_t, reg_list, dwords=in_block_define.get_last_pos())

        self.ic.Block_alloc([block,*reg_list], block_pos, alignment, self._alloc_block)
        
        return block
    
    def _split_block(self, supper_block:block_of_reg_blocks, ):
        supper_block_position = supper_block.position
        sub_blocks = supper_block._reg_blocks
        sub_units = []
        for i in range(len(sub_blocks)):
            
            offset = sub_blocks[i].position - supper_block_position
            sz = sub_blocks[i].dwords
            sub_units.append((offset,sz))

        self._allocator.unit_split(supper_block, sub_units)
        

    def split_block(self, block_of_reg_blocks):
        self.ic.Block_split(block_of_reg_blocks, self._split_block)

class sgpr_file_t(gpr_file_t):
    def __init__(self, gpu_instructions_caller_base, gpr_allocator:base_allocator):
        super().__init__(gpu_instructions_caller_base, reg_type.sgpr, gpr_allocator)
        
        vcc = VCC_reg()
        self.vcc = vcc
        self.vcc_lo = vcc.lo
        self.vcc_hi = vcc.hi

        exec = EXEC_reg()
        self.exec = exec
        self.vcc_lo = exec.lo
        self.vcc_hi = exec.hi
    
class vgpr_file_t(gpr_file_t):
    def __init__(self, gpu_instructions_caller_base, gpr_allocator:base_allocator):
        super().__init__(gpu_instructions_caller_base, reg_type.vgpr, gpr_allocator)
    
class special_regs_base():
    def __init__(self) -> None:
        self._special_regs_storage:Dict[str,reg_block] = {}
        self._special_reg_dirty = {}
        self._special_reg_used = {}
    
    def create_special_reg(self, name, size, reg_t):
        self._special_reg_dirty[name] = False
        self._special_reg_used[name] = False
        self._special_regs_storage[name] = reg_block.declare(name, reg_t, dwords=size)

    def get_special_reg(self, name):
        assert(self._special_reg_dirty[name] == False)
        self._special_reg_dirty[name] = True
        self._special_reg_used[name] = True
        return self._special_regs_storage[name]

    def clean_special_reg(self, name):
        self._special_reg_dirty[name] = False

class sgpr_hw_component(special_regs_base):
    def __init__(self, gpu_instructions_caller_base, sgpr_size, sgpr_alloc:Type[base_allocator], *args, **kwargs) -> None:
        super().__init__(gpu_instructions_caller_base=gpu_instructions_caller_base, *args, **kwargs)
        self.sgpr_alloc = sgpr_alloc(sgpr_size)
        s_type = reg_type.sgpr
        self.create_special_reg('karg_segment_ptr', 2, s_type)
        self.create_special_reg('gid_x', 1, s_type)
        self.create_special_reg('gid_y', 1, s_type)
        self.create_special_reg('gid_z', 1, s_type)

    def get_gid_x(self):
        return self.get_special_reg('gid_x')

    def get_gid_y(self):
        return self.get_special_reg('gid_y')
    def get_gid_z(self):
        return self.get_special_reg('gid_z')
    
    def get_karg_segment_ptr(self):
        return self.get_special_reg('karg_segment_ptr')
    
    def ABI_sgpr_setregs(self, Reg_Init_instr:HW_Reg_Init):
        off_seq = 0
        alloc = self.sgpr_alloc
        ABI_reg_list = ['karg_segment_ptr',
                    'gid_x', 'gid_y', 'gid_z'
                ]
        for i in ABI_reg_list:
            if(self._special_reg_used[i]):
                reg = self._special_regs_storage[i]
                alloc.malloc_at_fixed_pos(reg.dwords, off_seq)
                reg.set_position(off_seq)
                Reg_Init_instr.dst_regs.append(reg[:])
                off_seq += reg.dwords
        
class vgpr_hw_component(special_regs_base):
    def __init__(self, gpu_instructions_caller_base, vgpr_size, vgpr_alloc:Type[base_allocator], *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self.vgpr_alloc = vgpr_alloc(vgpr_size)
        v_type = reg_type.vgpr
        self.create_special_reg('tid_x', 1, v_type)
        self._special_reg_used['tid_x'] = True
        self.create_special_reg('tid_y', 1, v_type)
        self.create_special_reg('tid_z', 1, v_type)

    def get_tid_x(self):
        return self.get_special_reg('tid_x')
    def get_tid_y(self):
        return self.get_special_reg('tid_y')
    def get_tid_z(self):
        self._special_reg_used['tid_y'] = True
        return  self.get_special_reg('tid_z')

    def ABI_vgpr_setregs(self, Reg_Init_instr:HW_Reg_Init):
        off_seq = 0
        alloc = self.vgpr_alloc
        ABI_reg_list = ['tid_x', 'tid_y', 'tid_z']

        for i in ABI_reg_list:
            if(self._special_reg_used[i]):
                reg = self._special_regs_storage[i]
                alloc.malloc_at_fixed_pos(reg.dwords, off_seq)
                reg.set_position(off_seq)
                Reg_Init_instr.dst_regs.append(reg[:])
                off_seq += reg.dwords

class base_HW(sgpr_hw_component, vgpr_hw_component):
    def __init__(self, gpu_instructions_caller_base,sgpr_alloc:Type[base_allocator], vgpr_alloc:Type[base_allocator], sgpr_size, vgpr_size, LDS_size) -> None:
        super().__init__(
            gpu_instructions_caller_base=gpu_instructions_caller_base,
            sgpr_size=sgpr_size,
            sgpr_alloc=sgpr_alloc,
            vgpr_size=vgpr_size, vgpr_alloc=vgpr_alloc)
        #super(sgpr_hw_component, self).__init__(gpu_instructions_caller_base, vgpr_size, vgpr_alloc)
        
        self.LDS_size = LDS_size

    def ABI_HW_setregs(self, Reg_Init_instr:HW_Reg_Init):
        self.ABI_sgpr_setregs(Reg_Init_instr)
        self.ABI_vgpr_setregs(Reg_Init_instr)

class HW_gfx9(base_HW):
    def __init__(self, gpu_instructions_caller_base, sgpr_alloc: Type[base_allocator], vgpr_alloc: Type[base_allocator]) -> None:
        super().__init__(gpu_instructions_caller_base, sgpr_alloc, vgpr_alloc, 104, 256, 65536)