from enum import Enum, auto
from abc import ABC, abstractmethod
import inspect
from typing import Callable, Dict, List, NewType

class reg_type(Enum):
    sgpr = 's'
    vgpr = 'v'
    
class inst_mode(Enum):
    allocation: auto
    emmition: auto

class instruction_type(Enum):
    SMEM = 'SMEM'
    VMEM = 'VMEM'
    VOP1 = 'VOP1'
    VOP2 = 'VOP2'
    VOP3 = 'VOP3'
    VOP3P = 'VOP3P'
    VOPC = 'VOPC'
    SOP2 = 'SOP2'
    SOP1 = 'SOP1'
    VINTRP='VINTRP'
    DPP16 = 'DPP16'
    DPP8 = 'DPP8'
    DS = 'DS'
    EXP = 'EXP'
    MIMG = 'MIMG'
    MTBUF = 'MTBUF'
    MUBUF = 'MUBUF'
    SDWA = 'SDWA'
    SOPC = 'SOPC'
    SOPK = 'SOPK'
    SOPP = 'SOPP'
    FLAT = 'FLAT'
    REGALLOC = 'REGA'
    REGDEALLOC = 'REGD'
    REGREUSE = 'REGREUSE'
    BLOCKALLOC = 'BLCKA'
    BLOCKSPLIT = 'BLCKS'
    #BLOCKDEALLOC = 'BLCKD'
    FLOW_CONTROL = 'FC'
    HW_REG_INIT= "HW_INIT"

dst_atrs = ['DST', 'DST0','DST1', 'DST2']
src_atrs = ['SRC', 'SRC0', 'SRC1', 'SRC2', 'SRC3']

all_atrs = [*dst_atrs, *src_atrs]


def get_gfxXXX_instructions_sets():
    i_t = instruction_type

    def is_scalar(inst:inst_base):
        if (inst.inst_type in [i_t.SOP1, i_t.SOP2, i_t.SOPC, i_t.SOPK, i_t.SOPP, i_t.VOP3P, i_t.SMEM]):
            return True
        return False
    def is_vector(inst:inst_base):
        if (inst.inst_type in [i_t.VOPC, i_t.VOP1, i_t.VOP2, i_t.VOP3, i_t.VOP3P, i_t.DPP8, i_t.DPP16, i_t.VINTRP, i_t.VMEM]):
            return True
        return False
    def is_memory(inst:inst_base):
        if (inst.inst_type in [i_t.VMEM, i_t.SMEM, i_t.MTBUF, i_t.MUBUF, i_t.DS, i_t.FLAT]):
            return True
        if(inst.label in ['s_waitcnt']):
            return True
        if(inst.inst_type is i_t.SOPK and inst.label in 
            ['s_waitcnt_expcnt','s_waitcnt_lgkmcnt','s_waitcnt_vmcnt','s_waitcnt_vscnt']):
            return True
        return False
    def is_program_flow(inst:inst_base):
        if (inst.inst_type in [i_t.SOPP, i_t.FLOW_CONTROL]):
            #if(not (inst.label in ['s_nop', 's_waitcnt'])):
            if(inst.label in ['s_waitcnt']):
                return False
            return True
        #if(inst.inst_type is i_t.SOPK and inst.label in 
        #    ['s_waitcnt_expcnt','s_waitcnt_lgkmcnt','s_waitcnt_vmcnt','s_waitcnt_vscnt']):
        #    return True
        return False
    def is_exec_dependent(inst:inst_base):
        if( is_vector(inst) or (inst.label in [i_t.EXP, i_t.MTBUF, i_t.MUBUF, i_t.DS, i_t.FLAT]) ):
            return True
        return False

    return { 
        'scalar' : is_scalar,
        'vector' : is_vector,
        'memory' : is_memory,
        'program_flow' : is_program_flow,
        'exec_dep' : is_exec_dependent
    }

class inst_base(ABC):
    __slots__ = ['label', 'inst_type', 'trace']
    def __init__(self, inst_type:instruction_type, label:str) -> None:
        self.inst_type:instruction_type = inst_type
        self.label = label
        self.trace = ''

    @abstractmethod
    def __str__(self) -> str:
        return f'{self.label}'
    
    def execute(self):
        pass

    def emit(self, emiter_f:Callable[[NewType], None]):
        emiter_f(self)

    def emit_trace(self) -> str:
        return self.trace
    
    def get_srs_regs(self):
        ret = []
        for srs_a in src_atrs:
            a = getattr(self, srs_a, None)
            if(a):
                ret.append(a)
        return ret

    def get_dst_regs(self):
        ret = []
        for dst_a in dst_atrs:
            a = getattr(self, dst_a, None)
            if(a):
                ret.append(a)
        return ret

class inst_caller_base(ABC):
    __slots__ = ['il']
    def __init__(self, insturction_list:List[inst_base]) -> None:
        self.il = insturction_list

    def ic_pb(self, inst):
        self.il.append(inst)
        st = inspect.stack()
        return inst


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
        return f'//.set {self.label}, {self.position}'
        #TODO lang based definition
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

        assert(self.regVar_size > 0)
        if(type(key) is tuple):
            assert len(key) == 2
            slice_size = key[1] - key[0] + 1
            new_offset = key[0]
        elif (type(key) is slice):
            indices = key.indices(self.regVar_size - 1)
            assert(indices[2] <= 1)
            slice_size = indices[1] - indices[0] + 1
            new_offset = indices[0]
        else:
            new_offset = key
        #send label without reg_type prefix
        assert(new_offset < self.regVar_size and slice_size <= self.regVar_size - new_offset)
        new_offset = self.reg_offset + new_offset
        
        block_slice = regVar('', self.base_reg, new_offset, slice_size)
        return block_slice

    def set_lable(self, label:str):
        self.label = label

    def define(self):
        if(self.base_reg.position < 0):
            assert False, 'block can`t be defined, the block position has not been set'
        if type(self.label) in (tuple, list):
            assert False, "not support label is tuple and call define"
        return f'//.set {self.label}, {self.base_reg.position + self.reg_offset}'
        #TODO lang based definition
        return f'.set {self.label}, {self.base_reg.position + self.reg_offset}'

    def define_as(self, label:str):
        self.label = label
        #self.define()
        if(self.base_reg.position < 0):
            assert False, 'block can`t be defined, the block position has not been set'
        if type(label) in (tuple, list):
            assert False, "not support label is tuple and call define"
        return f'//.set {label}, {self.base_reg.position + self.reg_offset}'
        #TODO lang based definition
        return f'.set {label}, {self.base_reg.position + self.reg_offset}'

    def __str__(self) -> str:
        assert(self.regVar_size > 0)
        right_index = self.regVar_size - 1
        
        if(right_index == 0):
            return self.base_reg.get_str(index=self.reg_offset)
        else:
            return self.base_reg.get_str(index=(self.reg_offset, self.reg_offset+right_index))
    
    def __add__(self, offset):
        assert(type(offset) is int)
        return regVar(self.label, self.base_reg, self.reg_offset + offset, self.regVar_size - offset)
    
    def get_view_range(self):
        l = self.reg_offset
        r = self.regVar_size + l
        return (l, r)

