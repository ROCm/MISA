from enum import Enum
from typing import List

#should be 1 symbol len
class reg_type(Enum):
    sgpr = 's'
    vgpr = 'v'


class const():
    def __init__(self, val) -> None:
        self.label = val
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
        self.label = val
        if(type(val) in [int, float]):
            self.val = val
        else:
            assert False
            
    def __str__(self) -> str:
        return f' {self.val}'        

class imm16_t():
    def __init__(self, val) -> None:
        self.label = val
        if(type(val) is int and (val <= 65535) and (val >= -32768)):
            self.val = val
        else:
            assert False
            
    def __str__(self) -> str:
        return f' {self.val}'   

class simm32_t():
    def __init__(self, val) -> None:
        self.label = val
        if(type(val) is int and (val < 1**32) and (val >= -(1**32))):
            self.val = val
        else:
            assert False
            
    def __str__(self) -> str:
        return f' {self.val}' 

class simm21_t():
    def __init__(self, val) -> None:
        self.label = val
        if(type(val) is int and (val < 1**20) and (val >= -(1**20) )):
            self.val = val
        else:
            assert False
            
    def __str__(self) -> str:
        return f' {self.val}'  

class uimm20_t():
    def __init__(self, val) -> None:
        self.label = val
        if(type(val) is int and (val < 1**20) and (val >= 0)):
            self.val = val
        else:
            assert False
            
    def __str__(self) -> str:
        return f' {self.val}' 


class label_t():
    def __init__(self, val:str) -> None:
        self.label = val
        if(type(val) is str):
            self.val = val
        else:
            assert False
            
    def __str__(self) -> str:
        return f' {self.val}' 

    def define(self):
        return f' {self.val}:'

class reg_block(object):
    __slots__ = ['label', 'dwords', 'reg_t', 'position', 'label_as_pos', 'reg_align']
    def __init__(self, label:str, reg_t:reg_type, position:int = 0, dwords:int = 1, label_as_pos=False, reg_align=1):
        
        assert type(label) is str
        assert type(position) is int
        assert type(dwords) is int

        self.label = f'{reg_t.value}_{label}'
        self.position = position
        self.dwords = dwords
        self.reg_t = reg_t
        self.reg_align = reg_align
        self.label_as_pos = label_as_pos

    @classmethod
    def declare(cls, label:str, reg_t:reg_type, dwords:int = 1, label_as_pos=False, reg_align=1):
        '''Declaration without definition, only block type, label and size will be defined'''
        return reg_block(label, reg_t, position=-1, dwords=dwords, label_as_pos=label_as_pos, reg_align=reg_align)

    #def define(self):
    #    if(self.position < 0):
    #        assert False, 'block can`t be defined, the block position has not been set'
    #    if type(self.label) in (tuple, list):
    #        assert False, "not support label is tuple and call define"
    #    return f'.set {self.label}, {self.position}'

    def define(self):
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
    
    def get_str(self, index=None) -> str:
        if self.label_as_pos:
            self_pos = self.label
        else:
            self_pos = self.position

        if index is None:
            return f'{self.reg_t.value}[{self_pos}]'
        elif type(index) is int:
            if index == 0:
                return f'{self.reg_t.value}[{self_pos}]'
            return f'{self.reg_t.value}[{self_pos}+{index}]'
        elif type(index) is tuple:
            assert len(index) == 2, 'expect tuple (start-index, end-index), inclusive'
            return f'{self.reg_t.value}[{self_pos}+{index[0]}:{self_pos}+{index[1]}]'
        else:
            assert False
    
    def __getitem__(self, key):
        slice_size = 1

        if(type(key) is tuple):
            assert len(key) == 2
            slice_size = key[1] - key[0] + 1
            new_offset = key[0]
        elif (type(key) is slice):
            indices = key.indices(self.dwords-1)
            assert(indices[2] <= 1)
            slice_size = indices[1] - indices[0] + 1
            new_offset = indices[0]
        else:
            new_offset = key
        #send label without reg_type prefix
        view_slice = regVar('', self, new_offset, slice_size)
        return view_slice

class reg_block_custom_reg(reg_block):
    def __init__(self, label:str, reg_t:reg_type, dwords:int = 1):
        
        assert type(label) is str
        assert type(dwords) is int
        super().__init__(label=label, reg_t=reg_t, position=-1, dwords=dwords)
        self.position = label

    def define(self):
        raise AttributeError( "'custom_reg' object has no attribute 'define'" )

    def set_position(self, position:int):
        raise AttributeError( "'custom_reg' object has no attribute 'set_position'" )
    
    def expr(self, index = 0):
        raise AttributeError( "'custom_reg' object has no attribute 'expr'" )


class block_of_reg_blocks(reg_block):
    def __init__(self, label: str, reg_t: reg_type, reg_blocks:List[reg_block], position: int = 0, dwords: int = 1, reg_align=1):
        super().__init__(label, reg_t, position=position, dwords=dwords, reg_align=reg_align)
        self._reg_blocks = reg_blocks
    
    @classmethod
    def declare(cls, label:str, reg_t:reg_type, reg_blocks:List[reg_block], dwords:int = 1, reg_align=1):
        '''Declaration without definition, only block type, label and size will be defined'''
        return block_of_reg_blocks(label, reg_t, reg_blocks, position=-1, dwords=dwords, reg_align=reg_align)

class regVar(object):
    __slots__ = ['label', 'base_reg', 'reg_offset', 'regVar_size']
    def __init__(self, label:str, base_reg:reg_block, reg_offset:int = 0, regVar_size:int = 1):
        self.label = label
        self.base_reg = base_reg
        self.reg_offset = reg_offset
        self.regVar_size = regVar_size
        assert(regVar_size >= 0)
    
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
        assert(self.regVar_size > 0)
        if(type(key) is tuple):
            assert len(key) == 2
            slice_size = key[1] - key[0] + 1
            new_offset = new_offset + key[0]
        elif (type(key) is slice):
            indices = key.indices(self.regVar_size - 1)
            assert(indices[2] <= 1)
            slice_size = indices[1] - indices[0] + 1
            new_offset = indices[0]
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
        assert(self.regVar_size > 0)
        right_index = self.regVar_size - 1
        if(right_index == 0):
            #if (self.label is present) #TODO
            #return f'{self.base_reg.reg_t.value}[{self.label}]'
            return f'{self.base_reg.reg_t.value}[{self.base_reg.label}+{self.reg_offset}]'
        else:
            return f'{self.base_reg.reg_t.value}[{self.base_reg.label}+{self.reg_offset}:{self.base_reg.label}+{self.reg_offset}+{right_index}]'
    
    def __add__(self, offset):
        assert(type(offset) is int)
        return regVar(self.label, self.base_reg, self.reg_offset + offset, self.regVar_size - offset)


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

class  M0_reg(regVar):
    def __init__(self):
        self.label = 'm0'
        self.dwords = 2

    
    def set_lable(self, label:str):
        raise AttributeError( "'m0' object has no attribute 'set_lable'" )

    def define(self):
        raise AttributeError( "'m0' object has no attribute 'define'" )

    def define_as(self, label:str):
        raise AttributeError( "'m0' object has no attribute 'define_as'" )

    def __getitem__(self, key):
        slice_size = 1
        l = 0
        r = 0
        if(type(key) is tuple):
            assert len(key) == 2
            r = key[1]
            l = key[0]
        elif (type(key) is slice):
            r = key.stop
            l = key.start
        else:
            l = key
            r = key
        #send label without reg_type prefix
        slice_size = r - l
        assert(slice_size <= 1 and slice_size >= 0)
        return self

    def __str__(self) -> str:
        return f'{self.label}'

vcc_reg_block = reg_block('vcc', reg_type.sgpr, -1, 2) 

class  VCC_reg(regVar):
    def __init__(self, baseVCC=True):
        super().__init__(vcc_reg_block.label, vcc_reg_block, 0, 2)
        if(baseVCC):
            self.lo = _VCC_LO()
            self.hi = _VCC_HI()
    
    def set_lable(self, label:str):
        raise AttributeError( "'VCC' object has no attribute 'set_lable'" )

    def define(self):
        raise AttributeError( "'VCC' object has no attribute 'define'" )

    def define_as(self, label:str):
        raise AttributeError( "'VCC' object has no attribute 'define_as'" )

    def __getitem__(self, key):
        slice_size = 1
        l = 0
        r = 0
        if(type(key) is tuple):
            assert len(key) == 2
            r = key[1]
            l = key[0]
        elif (type(key) is slice):
            r = key.stop
            l = key.start
        else:
            l = key
            r = key
        #send label without reg_type prefix
        slice_size = r - l
        assert(slice_size <= 1 and slice_size >= 0)
        if(slice_size > 0):
            return self
        else:
            if(l == 0):
                return self.lo
            else:
                return self.hi

    def __str__(self) -> str:
        return f'{self.label}'

class  _VCC_LO(VCC_reg):
    def __init__(self):

        super().__init__(baseVCC=False)
        self.label = 'vcc_lo'
        self.regVar_size = 1

    def __getitem__(self, key):
        l = 0
        r = 0
        if(type(key) is tuple):
            assert len(key) == 2
            r = key[1]
            l = key[0]
        elif (type(key) is slice):
            r = key.stop
            l = key.start
        else:
            l = key
            r = key
        #send label without reg_type prefix
        assert(l == r)
        
        return self
        
class  _VCC_HI(VCC_reg):
    def __init__(self):
        super().__init__(baseVCC=False)
        self.reg_offset = 1
        self.regVar_size = 1
        self.label = 'vcc_hi'
    
    def __getitem__(self, key):
        l = 0
        r = 0
        if(type(key) is tuple):
            assert len(key) == 2
            r = key[1]
            l = key[0]
        elif (type(key) is slice):
            r = key.stop
            l = key.start
        else:
            l = key
            r = key
        #send label without reg_type prefix
        assert(l == r)
        
        return self

EXEC_reg_block = reg_block('exec', reg_type.sgpr, -1, 2) 

class  EXEC_reg(regVar):
    def __init__(self, baseEXEC=True):
        super().__init__(EXEC_reg_block.label, EXEC_reg_block, 0, 2)
        if(baseEXEC):
            self.lo = _EXEC_LO()
            self.hi = _EXEC_HI()
    
    def set_lable(self, label:str):
        raise AttributeError( "'EXEC' object has no attribute 'set_lable'" )

    def define(self):
        raise AttributeError( "'EXEC' object has no attribute 'define'" )

    def define_as(self, label:str):
        raise AttributeError( "'EXEC' object has no attribute 'define_as'" )

    def __getitem__(self, key):
        slice_size = 1
        l = 0
        r = 0
        if(type(key) is tuple):
            assert len(key) == 2
            r = key[1]
            l = key[0]
        elif (type(key) is slice):
            r = key.stop
            l = key.start
        else:
            l = key
            r = key
        #send label without reg_type prefix
        slice_size = r - l
        assert(slice_size <= 1 and slice_size >= 0)
        if(slice_size > 0):
            return self
        else:
            if(l == 0):
                return self.lo
            else:
                return self.hi

    def __str__(self) -> str:
        return f'{self.label}'

class  _EXEC_LO(EXEC_reg):
    def __init__(self):
        super().__init__(baseEXEC=False)
        self.label = 'exec_lo'
        self.regVar_size = 1
    
    def __getitem__(self, key):
        l = 0
        r = 0
        if(type(key) is tuple):
            assert len(key) == 2
            r = key[1]
            l = key[0]
        elif (type(key) is slice):
            r = key.stop
            l = key.start
        else:
            l = key
            r = key
        #send label without reg_type prefix
        assert(l == r)
        
        return self
        
class  _EXEC_HI(EXEC_reg):
    def __init__(self):

        super().__init__(baseEXEC=False)
        self.regVar_size = 1
        self.reg_offset = 1
        self.label = 'exec_hi'

    
    def __getitem__(self, key):
        l = 0
        r = 0
        if(type(key) is tuple):
            assert len(key) == 2
            r = key[1]
            l = key[0]
        elif (type(key) is slice):
            r = key.stop
            l = key.start
        else:
            l = key
            r = key
        #send label without reg_type prefix
        assert(l == r)
        
        return self

