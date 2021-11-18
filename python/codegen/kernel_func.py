
from os import name
from python.codegen.gpu_arch.HW_components import base_HW, sgpr_file_t, vgpr_file_t
from python.codegen.gpu_arch.allocator import base_allocator
from typing import Generic, List, Type, TypeVar
from python.codegen.gpu_data_types import block_of_reg_blocks, reg_block, reg_type
from python.codegen.generator_instructions import instr_label_caller, reg_allocator_base
from python.codegen.gpu_instruct import gpu_instructions_caller_base, inst_caller_base, instruction_type

import inspect

T = TypeVar('T')
class kernel_func(Generic[T]):
    def __init__(self, instructions_caller_base:T, func_name:str=None, sgpr_a:base_allocator=None, vgpr_a:base_allocator=None, agpr_a:base_allocator=None) -> None:
        
        if(sgpr_a != None ):#and issubclass(type(sgpr_a), base_allocator)):
            self.sgpr_f = sgpr_file_t(instructions_caller_base, sgpr_a)
        
        if(vgpr_a != None ):#and issubclass(type(sgpr_a), base_allocator)):
            self.vgpr_f = vgpr_file_t(instructions_caller_base, vgpr_a)

        self.agpr_f = None

        self.set_label = instr_label_caller(instructions_caller_base.il).set_label
        

        self.ic = instructions_caller_base
        self.PIc = gpu_instructions_caller_base(instructions_caller_base.il)
        if(func_name == None):
            self.func_name = self.__class__.__name__
        else:
            self.func_name = func_name
        
        self.ic_begin_pos = -1

        self.reg_ignore_list = []
    
    @classmethod
    def create_from_other_inst(cls, other, func_name: str = None):
        
        sgpr_f = getattr(other, 'sgpr_f', None)
        sgpr_a = getattr(sgpr_f, '_allocator', None)

        vgpr_f = getattr(other, 'vgpr_f', None)
        vgpr_a = getattr(vgpr_f, '_allocator', None)

        agpr_f = getattr(other, 'agpr_f', None)
        agpr_a = getattr(agpr_f, '_allocator', None)

        return cls(other.ic, func_name, sgpr_a = sgpr_a, vgpr_a=vgpr_a, agpr_a=agpr_a)

    def _func_begin(self):
        self.PIc.kernel_func_begin(self)
        self.ic_begin_pos = len(self.PIc.il)

    def _func_end(self):
        ic_end_pos = len(self.PIc.il)
        il = self.PIc.il
        not_dealocated_list = []
        ignore_list = self.reg_ignore_list

        for i in range(self.ic_begin_pos, ic_end_pos, 1):
            cur_instruction = il[i]
            if (type(cur_instruction) is reg_allocator_base):
                if(cur_instruction.inst_type is instruction_type.REGALLOC):
                    not_dealocated_list.append(cur_instruction.reg)
                elif(cur_instruction.inst_type is instruction_type.BLOCKALLOC):
                    not_dealocated_list.append(cur_instruction.reg[0][0])
                elif(cur_instruction.inst_type is instruction_type.BLOCKSPLIT):
                    assert(type(cur_instruction.reg) is block_of_reg_blocks)
                    block_list = cur_instruction.reg._reg_blocks
                    split_pos = not_dealocated_list.index(cur_instruction.reg)
                    not_dealocated_list.pop(split_pos)
                    for reg in block_list:
                        not_dealocated_list.insert(split_pos, reg)
                        split_pos += 1
                elif(cur_instruction.inst_type is instruction_type.REGDEALLOC):
                    try:
                        not_dealocated_list.remove(cur_instruction.reg)
                    except ValueError :
                        try:
                            ignore_list.index(cur_instruction.reg)
                        except IndexError:
                            assert(False)
                elif(cur_instruction.inst_type is instruction_type.REGREUSE):
                        not_dealocated_list.remove(cur_instruction.reg[0])
                        not_dealocated_list.append(cur_instruction.reg[1])
        
        sgpr_dealloc = self.sgpr_f.free
        vgpr_dealloc = self.vgpr_f.free
        
        for i in reversed(not_dealocated_list):
            assert(type(i) is reg_block or block_of_reg_blocks)
            if(i.reg_t is reg_type.sgpr):
                sgpr_dealloc(i)
            else:
                vgpr_dealloc(i)

        self.PIc.kernel_func_end(self)

    def wrap_call(self, *args, **kwargs):
        self._func_begin()
        ret = self.wrapped_call(self, *args, **kwargs)
        self._func_end()
    
    def func(self, *args, **kwargs):
        return self.wrap_call(*args, **kwargs)
    
    def wrapped_call(self, *args, **kwargs):
        pass
class kernel_launcher(kernel_func[T]):
    
    def __init__(self, instructions_caller_base: T, gpu_HW:base_HW, func_name: str = None):
        sgpr_a = getattr(gpu_HW, 'sgpr_alloc', None)
        vgpr_a = getattr(gpu_HW, 'vgpr_alloc', None)
        agpr_a = getattr(gpu_HW, 'agpr_alloc', None)
        
        super().__init__(instructions_caller_base, func_name=func_name, sgpr_a=sgpr_a, vgpr_a=vgpr_a, agpr_a=agpr_a)
        self.HW = gpu_HW
    
    def _func_begin(self):
        self.PIc.kernel_func_begin(self)
        self.ic_begin_pos = 0

    def _func_end(self):
        self.reg_ignore_list = self.HW.get_ABI_active_reg_list()
        super()._func_end()

    @classmethod
    def create_from_other_inst(cls, other, func_name: str = None):
        return cls(other.ic, other.HW, func_name)


def mfunc_class(cls):
    def wrapped_class(*args, **kwargs):
        save = cls(*args, **kwargs)
    
        save.wrapped_call = save.func
        save.func = save.wrap_call
        return save
    return wrapped_class

def mfunc_func(src_func):    
    def func(kf:kernel_func, *args, **kwargs):
        name = src_func.__name__
        save = kernel_func.create_from_other_inst(kf, name)
        save.wrapped_call = src_func
        save.func(*args, **kwargs)
    func.__signature__ = inspect.signature(src_func)
    return func

def launcher_kernel(src_func):
    def func(kf, *args, **kwargs):
        name = src_func.__name__
        save = kernel_launcher.create_from_other_inst(kf, name)
        save.wrapped_call = src_func
        save.func(*args, **kwargs)
    #func.__signature__ = inspect.signature(src_func)
    return func


#sample
@mfunc_class
class __maccro_1(kernel_func):
    def func(self, arg1, arg2:str):
        self.sgpr_f.add('s', 2)
        return 

@mfunc_func
def __maccro_2(self:kernel_func, arg1, arg2:str):
    sgpr_f = self.sgpr_f
    sgpr_f.add('s', 2)


class __macro_ctrl():
    def __init__(self, ic) -> None:
        kf = kernel_func(ic)
        self.x = __maccro_2(kf, 1, 2)
        kl = inspect.signature(maccro_2)
        __maccro_2(ic, 1, 2)

        
        
        
