################################################################################
# 
#  MIT License
# 
#  Copyright (c) 2020 Advanced Micro Devices, Inc.
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
# 
################################################################################
# pylint: disable=maybe-no-member

from .algo import *
from .codegen import *

class igemm_codegen_driver_t(mc_base_t):
    def __init__(self, mc, tunable_dicts):
        mc_base_t.__init__(self, mc)
        self.tunable_dicts = tunable_dicts

        kernel_list = []

        # gtc bwd
        kernel_list.extend([igemm_bwd_gtc_t(mc, igemm_gtc_tunable_parameter_t(td)) for td in tunable_dicts])

        self.kernel_list = kernel_list

    def emit_hsa_header(self):
        hsa_header_t(self.mc).emit()

    def _emit_fma_macro(self):
        macro_v_fma_mxn_t(self.mc, 4, 4, 8).emit()
        macro_v_fma_mxn_t(self.mc, 2, 4, 8).emit()
        macro_v_fma_mxn_t(self.mc, 4, 2, 4).emit()
        macro_v_fma_mxn_t(self.mc, 2, 2, 4).emit()

    def emit_global_macro(self):
        # emit global macro, independent of tunable
        macro_int_div_vv_t(self.mc).emit()
        macro_int_div_vs_t(self.mc).emit()
        macro_int_div_ss_t(self.mc).emit()
        macro_int_div_rem_vv_t(self.mc).emit()
        macro_int_div_rem_vs_t(self.mc).emit()
        macro_int_div_rem_ss_t(self.mc).emit()
        # emit_write_4d_strided_t(self.mc).emit()
        if self.mc.arch_config.arch == AMDGPU_ARCH_GFX908 and self.mc.arch_config.use_xdlops:
            macro_acc_c_clear_t(self.mc).emit()
        macro_c_clear_t(self.mc).emit()
        if self.mc.arch_config.arch == AMDGPU_ARCH_GFX908 and not self.mc.arch_config.use_dlops:
            self._emit_fma_macro()

    def emit_igemm_macro(self):
        # igemm algorithm related macros
        # emit_v4r1_dynamic_macros(self.mc, self.tunable_dicts)
        for kernel in self.kernel_list:
            macro_list = kernel.get_kernel_macros()
            # assert len(macro_list), ''
            for macro in macro_list:
                self.mc.insert_unique(macro.name(), macro)
        self.mc.emit_all_unique()

    def emit_igemm_kernel(self):
        # emit the kernel
        #emit_v4r1_dynamic_kernel(self.mc, self.tunable_dicts)
        for kernel in self.kernel_list:
            self._emit(';----------------------------------------------------------')
            self._emit('; starting of kernel {}'.format(kernel.name()))
            self._emit(kernel.tunable.serialize())

            kernel.emit_kernel_symbol()

            kernel.emit_kernel_header()
            with kernel._indent_context():
                if kernel.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V2:
                    kernel.emit_kernel_amd_kernel_code_t()
                kernel.emit_kernel_body()
                kernel.emit_kernel_end()
            if kernel.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
                kernel.emit_kernel_amd_kernel_code_t()
            kernel.emit_kernel_footer()

    def emit_metadata(self):
        kernel_info_list = [kernel.get_kernel_info() for kernel in self.kernel_list]
        amdgpu_metadata_t(self.mc, kernel_info_list).emit()

    def do_emit(self):
        self.emit_hsa_header()
        self.emit_global_macro()
        self.emit_igemm_macro()
        self.emit_igemm_kernel()
        self.emit_metadata()

    def do_compile(self):
        ass = compile_asm_t(self.mc, self.mc.emitter.file_name)
        rtn = ass.compile()
        if not rtn:
            assert False

        disass = compile_disass_t(self.mc, ass.target_hsaco)
        rtn = disass.compile()
        if not rtn:
            assert False

    def __call__(self):
        self.do_emit()
        self.do_compile()
