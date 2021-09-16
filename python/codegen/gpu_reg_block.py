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


class const():
    def __init__(self, val) -> None:
        if(type(val) is int):
            if(val <= 64 and val >= -16):
                self.val = val
            else:
                assert False
        elif(type(val) is float):
            if (val in [0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0, 0.1592, 0.15915494, 0.15915494309189532]):
                self.val = val
            else:
                assert False
        else:
            assert False
    
    def __str__(self) -> str:
        return f' {self.val}'

class literal():
    def __init__(self, val) -> None:
        if(type(val) in [int, float]):
            self.val = val
        else:
            assert False
            
    def __str__(self) -> str:
        return f' {self.val}'        

class imm16_t():
    def __init__(self, val) -> None:
        if(type(val) is int and (val <= 65535) and (val >= -32768)):
            self.val = val
        else:
            assert False
            
    def __str__(self) -> str:
        return f' {self.val}'   

class simm32_t():
    def __init__(self, val) -> None:
        if(type(val) is int and (val < 1**32) and (val >= -(1**32))):
            self.val = val
        else:
            assert False
            
    def __str__(self) -> str:
        return f' {self.val}' 

class simm21_t():
    def __init__(self, val) -> None:
        if(type(val) is int and (val < 1**20) and (val >= -(1**20) )):
            self.val = val
        else:
            assert False
            
    def __str__(self) -> str:
        return f' {self.val}'  


class uimm20_t():
    def __init__(self, val) -> None:
        if(type(val) is int and (val < 1**20) and (val >= 0)):
            self.val = val
        else:
            assert False
            
    def __str__(self) -> str:
        return f' {self.val}' 

class label_t():
    def __init__(self, val:str) -> None:
        if(type(val) is str):
            self.val = val
        else:
            assert False
            
    def __str__(self) -> str:
        return f' {self.val}' 

    def define(self):
        return f' {self.val}:'

class reg_block(object):
    __slots__ = ['label', 'dwords', 'reg_t', 'position']
    def __init__(self, label:str, reg_t:reg_type, position:int = 0, dwords:int = 1):
        
        assert type(label) is str
        assert type(position) is int
        assert type(dwords) is int

        self.label = f'{reg_t.value}_{label}'
        self.position = position
        self.dwords = dwords
        self.reg_t = reg_t

    @classmethod
    def declare(cls, label:str, reg_t:reg_type, dwords:int = 1):
        '''Declaration without definition, only block type, label and size will be defined'''
        return reg_block(label, reg_t, position=-1, dwords=dwords)

    def define(self):
        if(self.position < 0):
            assert False, 'block can`t be defined, the block position has not been set'
        if type(self.label) in (tuple, list):
            assert False, "not support label is tuple and call define"
        return f'.set {self.label}, {self.position}'

    def set_position(self, position:int):
        self.position = position

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

        if(type(key) is tuple):
            assert len(key) == 2
            slice_size = key[1] - key[0]
            new_offset = key[0]
        elif (type(key) is slice):
            indices = key.indices(self.dwords)
            assert(indices[2] <= 1)
            slice_size = indices[1] - indices[0]
            new_offset = indices[0]
        else:
            new_offset = key
        #send label without reg_type prefix
        view_slice = regVar('', self, new_offset, slice_size)
        return view_slice

class regVar(object):
    __slots__ = ['label', 'base_reg', 'reg_offset', 'right_index']
    def __init__(self, label:str, base_reg:reg_block, reg_offset:int = 0, right_index:int = 0):
        self.label = label
        self.base_reg = base_reg
        self.reg_offset = reg_offset
        self.right_index = right_index
    
    #@classmethod
    #def init_working_label(cls, label:str, reg:reg_block, reg_offset:int = 0, dwords:int = 0):
    #    #cls = self.__class__
    #    result = regVar.__new__(cls)
    #    #result.__dict__.update(self.__dict__)
    #    result.label = label #set as reg.label + offset
    #    result.base_reg = reg
    #    result.right_index = dwords

    def __getitem__(self, key):
        slice_size = 1
        new_offset = self.reg_offset

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
        block_slice = regVar('', self.base_reg, new_offset, slice_size)
        return block_slice

    def set_lable(self, label:str):
        self.label = label

    def define(self):
        if(self.base_reg.position < 0):
            assert False, 'block can`t be defined, the block position has not been set'
        if type(self.label) in (tuple, list):
            assert False, "not support label is tuple and call define"
        return f'.set {self.label}, {self.base_reg.position + self.reg_offset}'

    def define_as(self, label:str):
        self.label = label
        #self.define()
        if(self.base_reg.position < 0):
            assert False, 'block can`t be defined, the block position has not been set'
        if type(label) in (tuple, list):
            assert False, "not support label is tuple and call define"
        return f'.set {label}, {self.base_reg.position + self.reg_offset}'

    def __str__(self) -> str:
        right_index = self.right_index
        if(right_index == 0):
            #if (self.label is present) #TODO
            #return f'{self.base_reg.reg_t.value}[{self.label}]'
            return f'{self.base_reg.reg_t.value}[{self.base_reg.label}+{self.reg_offset}]'
        else:
            return f'{self.base_reg.reg_t.value}[{self.base_reg.label}+{self.reg_offset}:{self.base_reg.label}+{self.reg_offset}+{self.right_index}]'

class regAbs(regVar):
    def __init__(self, reg_src:regVar):
        self.__dict__.update(reg_src.__dict__)
    def __str__(self) -> str:
        return f'abs({super().__str__()})'

class regNeg(regVar):
    def __init__(self, reg_src:regVar):
        self.__dict__.update(reg_src.__dict__)
    def __str__(self) -> str:
        return f'neg({super().__str__()})'

def abs(reg:regVar):
    return regAbs

def neg(reg:regVar):
    return regAbs

from python.codegen.gpu_instruct import gpu_instructions_caller

class gpr_file_t():#mc_base_t):
    __slots__ = ['_allocator', 'reg_t', 'define_on_creation', 'ic']
    def __init__(self, ic:gpu_instructions_caller, reg_t:reg_type):
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

