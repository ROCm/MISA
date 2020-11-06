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

import os
import copy

IGEMM_EMIT_KERNEL_PER_INC_FILE = 1
IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE = 0     # it seems fail to find symbol if seperate metadata of different kernel using multiple .amdgpu_metadata

class igemm_codegen_driver_t(mc_base_t):
    def __init__(self, mc, tunable_dicts):
        mc_base_t.__init__(self, mc)
        self.tunable_dicts = tunable_dicts

        kernel_list = []

        # currently only support direction in tunable_dicts all the same.
        if tunable_dicts[0]['direction'] == 'fwd':
            for tdd in tunable_dicts:
                assert tdd['direction'] == 'fwd'
            # gtc fwd
            kernel_list.extend([igemm_fwd_gtc_t(mc, igemm_gtc_tunable_parameter_t(td)) for td in tunable_dicts])

        elif tunable_dicts[0]['direction'] == 'bwd':
            for tdd in tunable_dicts:
                assert tdd['direction'] == 'bwd'
            # gtc bwd
            kernel_list.extend([igemm_bwd_gtc_t(mc, igemm_gtc_tunable_parameter_t(td)) for td in tunable_dicts])

        elif tunable_dicts[0]['direction'] == 'wrw':
            for tdd in tunable_dicts:
                assert tdd['direction'] == 'wrw'
            # gtc bwd
            kernel_list.extend([igemm_wrw_gtc_t(mc, igemm_gtc_tunable_parameter_t(td)) for td in tunable_dicts])

        else:	
            assert False, f"unknown direcrion? {tunable_dicts[0]['direction']}"

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

        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            macro_mdiv_u32_ss_t(self.mc).emit()
            macro_mdiv_u32_rem_ss_t(self.mc).emit()
            macro_mdiv_u32_vs_t(self.mc).emit()
            macro_mdiv_u32_rem_vs_t(self.mc).emit()

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
        def get_kernel_per_inc_file_name(ker, origin_file_name):
            root_file_name = os.path.splitext(origin_file_name)[0]
            return root_file_name + f"_{ker.tunable.gemm_m_per_block:03}x{ker.tunable.gemm_n_per_block:03}" + ".inc"

        # emit the kernel
        #emit_v4r1_dynamic_kernel(self.mc, self.tunable_dicts)
        if IGEMM_EMIT_KERNEL_PER_INC_FILE:
            origin_emitter = self.mc.emitter
            assert type(origin_emitter) is mc_emit_to_file_t
            emitter_per_inc_dict = dict()
            if IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE:
                kinfo_per_inc_dict = dict()
            self._emit_empty_line()
            self._emit(f";---------------------------------------------------")

        for kernel in self.kernel_list:
            if IGEMM_EMIT_KERNEL_PER_INC_FILE:
                kpi_file_name = get_kernel_per_inc_file_name(kernel, origin_emitter.file_name)
                if kpi_file_name not in emitter_per_inc_dict:
                    origin_emitter.emit(f".include \"{os.path.basename(kpi_file_name)}\"")

                    kpi_emitter = mc_emit_to_file_t(kpi_file_name, copy.copy(origin_emitter.indent))
                    kpi_emitter.open()
                    self.mc.emitter = kpi_emitter
                    self.mc.emit_license()
                    self.mc.emit('; generated by igemm_codegen.py')
                    self.mc.emit(';')
                    emitter_per_inc_dict[kpi_file_name] = kpi_emitter
                    if IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE:
                        kinfo_per_inc_dict[kpi_file_name] = [kernel.get_kernel_info()]

                else:
                    self.mc.emitter = emitter_per_inc_dict[kpi_file_name]
                    if IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE:
                        kinfo_per_inc_dict[kpi_file_name].append(kernel.get_kernel_info())
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

        if IGEMM_EMIT_KERNEL_PER_INC_FILE:
            for k, v in emitter_per_inc_dict.items():
                if IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE:
                    self.mc.emitter = emitter_per_inc_dict[k]
                    amdgpu_metadata_t(self.mc, kinfo_per_inc_dict[k]).emit()
                # os.chmod(k, 0x777)
                v.close()
            self.mc.emitter = origin_emitter
            self._emit(f";---------------------------------------------------")
            self._emit_empty_line()

    def emit_metadata(self):
        kernel_info_list = [kernel.get_kernel_info() for kernel in self.kernel_list]
        amdgpu_metadata_t(self.mc, kernel_info_list).emit()

    def do_emit(self):
        self.emit_hsa_header()
        self.emit_global_macro()
        self.emit_igemm_macro()
        self.emit_igemm_kernel()
        if not IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE:
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

