from enum import Enum
from typing import List
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



from python.codegen.gpu_instruct import gpu_instructions_caller_base

class gpr_file_t():#mc_base_t):
    __slots__ = ['_allocator', 'reg_t', 'define_on_creation', 'ic']
    def __init__(self, ic:gpu_instructions_caller_base, reg_t:reg_type):
        #mc_base_t.__init__(self, mc)
        self._allocator = gpr_off_sequencer_t()
        self.reg_t = reg_t
        self.define_on_creation = False
        self.ic = ic
    
    def get_count(self):
        return self._allocator.get_last_pos()
    
    #def emit(self):
    #    _end_val = self._allocator.get_last_pos()
    #    assert _end_val <= amdgpu_sgpr_limit(self.mc.arch_config.arch), f"{self.reg_t.value}_end:{_end_val} "
    #    for k, v in self.__dict__.items():
    #        if not k.startswith('_'):
    #            reg_alloc(v)
    #            self._emit(v.define())
    #    self.define_on_creation = True
    
    def _alloc(self, reg:reg_block, alignment):
        reg.set_position(self._allocator(reg.dwords, alignment))
        return f'.set {reg.label}, {reg.position}'

    def _alloc_block(self, block_info:tuple, alignment):
        s = []
        regs:List[reg_block] = block_info[0]
        ofsets:List[int]  = block_info[1]
        base_reg = regs[0]
        base_reg.set_position(self._allocator(base_reg.dwords, alignment))
        s.append(f'.set {base_reg.label}, {base_reg.position}')
        base_reg_pos = base_reg.position
        
        for i in range(len(ofsets)):
            cur_reg = regs[i+1]
            cur_reg.set_position(self._allocator(base_reg_pos + ofsets[i], alignment))
            s.append(f'.set {cur_reg.label}, {cur_reg.position}')

        return '\n'.join(s)

    def _dealloc(self, reg:reg_block, alignment):
        return f'#dealock .unset {reg.label}, {reg.position}'

    def add(self, label:str, dwords:int = 1, alignment:int = 0):
        ret = reg_block.declare(label, self.reg_t, dwords=dwords)
        self.ic.reg_alloc(ret, alignment, self._alloc)
        return ret

    def add_no_pos(self, label:str, dwords:int = 1):
        return reg_block.declare(label, self.reg_t, dwords=dwords)

    def add_block(self, label:str, reg_list:List[reg_block], alignment:int = 0):
        in_block_define = gpr_off_sequencer_t()
        block_pos = []
        block_regs = []
        for i in reg_list:
            assert i.reg_t == self.reg_t, f" reg_t of element {i} doesn't match the block reg_t"
            block_pos.append(in_block_define(i.dwords))

        block = reg_block.declare(label, self.reg_t, dwords=in_block_define.get_last_pos())

        self.ic.Block_alloc([block,*reg_list], block_pos, alignment, self._alloc_block)
        
        return block[:]

class sgpr_file_t(gpr_file_t):
    def __init__(self, mc):
        super().__init__(mc, reg_type.sgpr)
    
class vgpr_file_t(gpr_file_t):
    def __init__(self, mc):
        super().__init__(mc, reg_type.vgpr)
