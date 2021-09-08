from enum import Enum
from typing import List
from python.codegen.amdgpu import amdgpu_sgpr_limit
from python.codegen.mc import mc_base_t
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

#should be 1 symbol len
class reg_type(Enum):
    sgpr = 's'
    vgpr = 'v'

max_vgpr_size = 256

class reg_block(object):
    __slots__ = ['label', 'offset', 'dwords', 'reg_t', 'start']
    def __init__(self, label:str, reg_t:reg_type, start:int = 0, dwords:int = 1, offset:int = 0):
        
        assert type(label) is str
        assert type(start) is int
        assert type(dwords) is int

        self.label = f'{reg_t.value}_{label}'
        self.start = start
        self.dwords = dwords
        self.reg_t = reg_t
        self.offset = offset

    @classmethod
    def declare(cls, label:str, reg_t:reg_type, dwords:int = 1):
        '''Declaration without definition, only block type, label and size will be defined'''
        return reg_block(label, reg_t, start=-1, dwords=dwords)

    def define(self):
        if(self.start < 0):
            assert False, 'block can`t be defined, the block position has not been set'
        if type(self.label) in (tuple, list):
            assert False, "not support label is tuple and call define"
        return f'.set {self.label}, {self.start}'

    def set_position(self, start:int):
        self.start = start
    
    def set_offset(self, offset:int):
        self.offset = offset

    def expr(self, index = 0):
        if type(index) is int:
            if index == 0:
                return self.label
            return f'{self.label}+{index}'
        elif type(index) is tuple:
            assert len(index) == 2, 'expect tuple (start-index, end-index), inclusive'
            return f'{self.label}+{index[0]}:{self.label}+{index[1]}'
        else:
            assert False

    def __call__(self, index1:int=0, index2:int=0):
        if(index2<=index1):
            return self.expr(index1)
        else:
            return self.expr((index1,index2))
    
    def __getitem__(self, key):
        slice_size = 1
        new_offset = self.offset

        if(type(key) is tuple):
            assert len(key) == 2
            slice_size = key[1] - key[0]
            new_offset = new_offset + key[0]
        elif (type(key) is slice):
            slice_size = key.stop - key.start
            new_offset = new_offset + key.start
        else:
            new_offset = new_offset + key
        #send label without reg_type prefix
        block_slice = copy.deepcopy(self)
        block_slice.dwords = slice_size
        block_slice.offset = new_offset
        return block_slice
        
    
    def __str__(self) -> str:
        if(self.dwords == 0):
            return f'{self.reg_t.value}[{self.label}+{self.offset}]'
        else:
            return f'{self.reg_t.value}[{self.label}+{self.offset}:{self.label}+{self.offset}+{self.dwords}]'
        
class gpr_file_t(mc_base_t):
    __slots__ = ['_sq', 'reg_t', 'define_on_creation']
    def __init__(self, mc, reg_t:reg_type):
        mc_base_t.__init__(self, mc)
        self._sq = gpr_off_sequencer_t()
        self.reg_t = reg_t
        self.define_on_creation = False
    
    def get_count(self):
        return self._sq.get_last_pos()
    
    def emit(self):
        _end_val = self._sq.get_last_pos()
        assert _end_val <= amdgpu_sgpr_limit(self.mc.arch_config.arch), f"{self.reg_t.value}_end:{_end_val} "
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                self._emit(v.define())
        self.define_on_creation = True
    
    def add(self, label:str, dwords:int = 1, alignment:int = 0):
        ret = reg_block(label, self.reg_t, self._sq(dwords, alignment))
        if self.define_on_creation :
            self._emit(ret.define())
        return ret

    def add_no_pos(self, label:str, dwords:int = 1):
        return reg_block.declare(label, self.reg_t, dwords=dwords)

    def add_block(self, label:str, reg_list:List[reg_block], alignment:int = 0):
        in_block_define = gpr_off_sequencer_t()
        
        for i in reg_list:
            assert i.reg_t == self.reg_t, f" reg_t of element {i} doesn't match the block reg_t"
            i.set_offset(in_block_define(i.dwords))

        block = reg_block(label, self.reg_t, self._sq(in_block_define.get_last_pos(), alignment))
        
        if self.define_on_creation :
            self._emit(block.define())

        for i in reg_list:
            i.set_position(block.start)
            if self.define_on_creation :
                self._emit(i.define())
        
        return block

class sgpr_file_t(gpr_file_t):
    def __init__(self, mc):
        super().__init__(mc, reg_type.sgpr)
    
class vgpr_file_t(gpr_file_t):
    def __init__(self, mc):
        super().__init__(mc, reg_type.vgpr)

