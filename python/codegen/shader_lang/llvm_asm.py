from python.codegen.mc import mc_base_t
from python.codegen.runtime.amdgpu import amd_kernel_code_t
from python.codegen.shader_lang.base_api import amdgpu_kernel_info_t, base_lang_class

AMDGPU_CODEOBJECT_V2    = (0 << 28)
AMDGPU_CODEOBJECT_V3    = (1 << 28)

class hsa_kernel_header(mc_base_t):
    '''
    only used in cov2
    '''
    def __init__(self, mc, amdgpu_kernel_info:amdgpu_kernel_info_t):
        mc_base_t.__init__(self, mc)
        self._kernel_info = amdgpu_kernel_info

    def emit(self):
        kernel_name = self._kernel_info.kernel_name
        self._emit('.text')
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
            self._emit('.globl {}'.format(kernel_name))
        self._emit('.p2align 8')
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
            self._emit('.type {},@function'.format(kernel_name))
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V2:
            self._emit('.amdgpu_hsa_kernel {}'.format(kernel_name))
        self._emit('{}:'.format(kernel_name))

class llvm_kernel(base_lang_class):    
    
    def __init__(self, mc, kernel_info: amdgpu_kernel_info_t, emmit_created_code):
        super().__init__(mc, kernel_info, emmit_created_code)
    
    def _emit_kernel_header(self):
        hsa_kernel_header(mc=self.mc, amdgpu_kernel_info=self._kernel_info).emit()
        
    def _emit_kernel_end(self):
        self._emit('s_endpgm')
        self._emit('')

    def _emit_kernel_amd_kernel_code_t(self):
        amd_kernel_code_t(self.mc, self._kernel_info).emit()

    def emit_kernel_code(self, ):
        self._emit_kernel_header()
        #self._emit_kernel_symbols()
        self._emmit_created_code(self._emit)
        self._emit_kernel_end()
        self._emit_kernel_amd_kernel_code_t()