
from os import name
from python.codegen.gpu_instruct import gpu_instructions_caller_base
from python.codegen.gpu_reg_block import gpr_file_t, sgpr_file_t, vgpr_file_t
import inspect

class kernel_func():
    def __init__(self, instructions_caller_base, func_name:str=None, sgpr_f:gpr_file_t=None, vgpr_f:gpr_file_t=None, agpr_f:gpr_file_t=None) -> None:
        if(sgpr_f == None):
            self.sgpr_f = sgpr_file_t(instructions_caller_base)
        else:
            self.sgpr_f = sgpr_f

        if(vgpr_f == None):
            self.vgpr_f = vgpr_file_t(instructions_caller_base)
        else:
            self.vgpr_f = vgpr_f

        if(agpr_f == None):
            self.agpr_f = vgpr_file_t(instructions_caller_base)
        else:
            self.agpr_f = agpr_f

        self.ic = instructions_caller_base
        if(func_name == None):
            self.func_name = self.__class__.__name__
        else:
            self.func_name = func_name

    def _func_begin(self):
        name = self.__class__.__name__
        self.ic.kernel_func_begin(name)

    def _func_end(self):
        name = self.__class__.__name__
        self.ic.kernel_func_end(name)

    def wrap_call(self, *args, **kwargs):
        self._func_begin()
        ret = self.wrapped_call(self, *args, **kwargs)
        self._func_end()
    
    def func(self, *args, **kwargs):
        return self.wrap_call(*args, **kwargs)
    
    def wrapped_call(self, *args, **kwargs):
        pass

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
        save = kernel_func(kf.ic, name)
        save.wrapped_call = src_func
        save.func(*args, **kwargs)
    func.__signature__ = inspect.signature(src_func)
    return func

#sample
@mfunc_class
class __maccro_1(kernel_func):
    def func(self, arg1, arg2:str):
        self.sgpr_f.add('s', 2)
        return 

@mfunc_func
def __maccro_2(self:kernel_func, arg1, arg2:str):
    sgpr_f = sgpr_file_t(self.ic)
    sgpr_f.add('s', 2)



class __macro_ctrl():
    def __init__(self, ic) -> None:
        kf = kernel_func(ic)
        self.x = __maccro_2(kf, 1, 2)
        kl = inspect.signature(maccro_2)
        __maccro_2(ic, 1, 2)

        
        
        
