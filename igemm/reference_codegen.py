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
from .ref import *
from .algo import *
from .codegen import *

CPP_DIR="driver"

class naive_conv_codegen_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)

    def do_emit_naive_conv(self):
        self._emit(f".ifndef MIOPEN_USE_RNE_BFLOAT16")
        self._emit(f".set MIOPEN_USE_RNE_BFLOAT16, 1")
        self._emit(f".endif")
        macro_int_div_vv_t(self.mc).emit()
        macro_int_div_vs_t(self.mc).emit()
        macro_int_div_ss_t(self.mc).emit()
        macro_int_div_rem_vv_t(self.mc).emit()
        macro_int_div_rem_vs_t(self.mc).emit()
        macro_int_div_rem_ss_t(self.mc).emit()

        #kernel_info_list = [kernel.get_kernel_info() for kernel in self.kernel_list]
        #amdgpu_metadata_t(self.mc, kernel_info_list).emit()
        kernel_info_list = []

        for direction in ['fwd']:
            for tensor_layout in ["nchw", "ncdhw"]:
                for precision in ["fp32", "fp16", "bf16"]:
                    ctrl = naive_conv_ctrl_t(direction, tensor_layout, precision)
                    if direction == 'fwd':
                        kernel = naive_conv_fwd_t(self.mc, ctrl)
                    else:
                        assert False
                    kernel_info_list.append(kernel.get_kernel_info())

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

        amdgpu_metadata_t(self.mc, kernel_info_list).emit()

    def do_emit(self):
        self.do_emit_naive_conv()

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

def reference_host_compile(args):
    cpp_src = os.path.join(CPP_DIR, "reference_driver.cpp")
    target_exe = os.path.join(args.dir, "reference_driver.exe")

    arch = amdgpu_arch_config_t({
        'arch'          :  amdgpu_string_to_arch( args.arch )})
    builder = compile_host_t(arch, cpp_src, target_exe)

    rtn = builder.compile(cxxflags=['-DIGEMM_HSACO=\"{}\"'.format("naive_conv.hsaco")])
    if not rtn:
        assert False

def reference_codegen_naive_conv(args):
    asm_target = os.path.join(args.dir, 'naive_conv.s')
    emitter = mc_emit_to_file_t(asm_target)
    arch = amdgpu_arch_config_t({
        'arch'          :   amdgpu_string_to_arch( args.arch ),
        'code_object'   :   amdgpu_string_to_codeobj( 'cov3') })
    # create mc
    mc = mc_asm_printer_t(emitter, arch)
    naive_conv_codegen_t(mc)()

def reference_codegen(args):
    reference_host_compile(args)
    reference_codegen_naive_conv(args)

