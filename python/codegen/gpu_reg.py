from enum import Enum
from python.codegen.amdgpu import amdgpu_sgpr_limit
from python.codegen.mc import mc_base_t

class gpr_off_sequencer_t(object):
    def __init__(self, offset = 0):
        self.start_offset = offset
    def __call__(self, step = 0, alignment = 0):
        previous_offset = self.start_offset
        if alignment:
            aligned_offset = ((previous_offset + alignment - 1) // alignment) * alignment
            self.start_offset = aligned_offset
            previous_offset = aligned_offset
        self.start_offset += step
        return previous_offset
    def get_start_offset(self):
        return self.start_offset

class reg_type(Enum):
    sgpr = 's'
    vgpr = 'v'

class any_reg(object):
    __slots__ = ['label', 'offset', 'dwords', 'reg_t']
    def __init__(self, label:str, reg_t:reg_type, offset:int = 0, dwords:int = 1):
        
        assert type(label) is str
        assert type(offset) is int
        assert type(dwords) is int

        self.label = f'{reg_t.value}_{label}'
        self.offset = offset
        self.dwords = dwords
        self.reg_t = reg_t

    def declare(self):
        if type(self.label) in (tuple, list):
            assert False, "not support label is tuple and call declare"
        return f'.set {self.label}, {self.offset}'
    
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
        
        if(type(key) is tuple):
            #index_f, index_s = key
            assert len(key) == 2
            return f'{self.reg_t.value}[{self.label}+{key[0]}:{self.label}+{key[1]}]'
        return f'{self.reg_t.value}[{self.label}+{key}]'
        
    def __eq__(self, other):
        if type(other) is not any_reg:
            return False
        if type(self.label) in (tuple, list):
            if type(other.label) not in (tuple, list):
                return False
            if len(self.label) != len(other.label):
                return False
            for a, b in zip(self.label, other.label):
                if a != b:
                    return False
            return True
        return self.label == other.label and self.offset == other.offset
    def __ne__(self, other):
        return not self == other

class gpr_file_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
        self._sq = gpr_off_sequencer_t()
    
    def get_count(self):
        return self._sq.get_start_offset()
    
    def emit(self, pref):
        _end_val = self._sq.get_start_offset()
        assert _end_val <= amdgpu_sgpr_limit(self.mc.arch_config.arch), f"{pref}end:{_end_val} "
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                self._emit(v.declare())

class sgpr_file_t(gpr_file_t):
    def __init__(self, mc):
        super().__init__(mc)
    
    def emit(self):
        super().emit('s_')
    
    def add(self, label:str, offset:int = 0, dwords:int = 1):
        return any_reg(label, reg_type.sgpr, offset, dwords)

class vgpr_file_t(gpr_file_t):
    def __init__(self, mc):
        super().__init__(mc)

    def emit(self):
        super().emit('v_')
    
    def add(self, label:str, offset:int = 0, dwords:int = 1):
        return any_reg(label, reg_type.vgpr, offset, dwords)
