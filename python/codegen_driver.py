################################################################################
# 
#  MIT License
# 
#  Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

from .igemm import *
from .codegen import *

import os
import copy
import multiprocessing as mp

IGEMM_EMIT_KERNEL_PER_INC_FILE = 1
IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE = 0     # it seems fail to find symbol if seperate metadata of different kernel using multiple .amdgpu_metadata

class codegen_driver_t(mc_base_t):
    def __init__(self, mc, tunable_dicts):
        mc_base_t.__init__(self, mc)
        self.tunable_dicts = tunable_dicts

        kernel_list = []

        assert type(mc) is mc_asm_printer_t

        # currently only support direction in tunable_dicts all the same.
        if tunable_dicts[0]['direction'] == 'fwd':
            for tdd in tunable_dicts:
                assert tdd['direction'] == 'fwd'
            # gtc fwd
            if 'tensor_layout' in tunable_dicts[0] and tunable_dicts[0]['tensor_layout'] == 'nhwc':
                kernel_list.extend([igemm_fwd_gtc_nhwc_t(mc_asm_printer_t(mc.emitter, mc.arch_config), igemm_gtc_tunable_parameter_t(td)) for td in tunable_dicts])
            else:
                kernel_list.extend([igemm_fwd_gtc_t(mc_asm_printer_t(mc.emitter, mc.arch_config), igemm_gtc_tunable_parameter_t(td)) for td in tunable_dicts])

        elif tunable_dicts[0]['direction'] == 'bwd':
            for tdd in tunable_dicts:
                assert tdd['direction'] == 'bwd'
            # gtc bwd
            if 'tensor_layout' in tunable_dicts[0] and tunable_dicts[0]['tensor_layout'] == 'nhwc':
                kernel_list.extend([igemm_bwd_gtc_nhwc_t(mc_asm_printer_t(mc.emitter, mc.arch_config), igemm_gtc_tunable_parameter_t(td)) for td in tunable_dicts])
            else:
                kernel_list.extend([igemm_bwd_gtc_t(mc_asm_printer_t(mc.emitter, mc.arch_config), igemm_gtc_tunable_parameter_t(td)) for td in tunable_dicts])
            # in bwd direction, need such upsampling clear kernel
            # kernel_list.extend([igemm_upsampling_clear_t(mc_asm_printer_t(mc.emitter, mc.arch_config), igemm_gtc_tunable_parameter_t(tunable_dicts[0]))])

        elif tunable_dicts[0]['direction'] == 'wrw':
            for tdd in tunable_dicts:
                assert tdd['direction'] == 'wrw'
            # gtc wrw
            if 'tensor_layout' in tunable_dicts[0] and tunable_dicts[0]['tensor_layout'] == 'nhwc':
                kernel_list.extend([igemm_wrw_gtc_nhwc_t(mc_asm_printer_t(mc.emitter, mc.arch_config), igemm_gtc_tunable_parameter_t(td)) for td in tunable_dicts])
            else:
                kernel_list.extend([igemm_wrw_gtc_t(mc_asm_printer_t(mc.emitter, mc.arch_config), igemm_gtc_tunable_parameter_t(td)) for td in tunable_dicts])

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
        if self.mc.arch_config.arch in (AMDGPU_ARCH_GFX908, AMDGPU_ARCH_GFX90A) and self.mc.arch_config.use_xdlops:
            macro_acc_c_clear_t(self.mc).emit()
        macro_c_clear_t(self.mc).emit()
        if self.mc.arch_config.use_dlops:
            self._emit_fma_macro()
        if hasattr(self.kernel_list[0], 'use_bf16_1k_in_fp16'):
            if self.kernel_list[0].use_bf16_1k_in_fp16():
                sym = self.kernel_list[0].get_predefine_for_bf16_1k_in_fp16()
                dfv = self.kernel_list[0].get_predefine_for_bf16_1k_in_fp16_default_value()
                inst_mfma_emit_macro_mfma_16f(self.mc, sym, dfv)

    def emit_global_macro_per_s_file(self, mc):
        # emit global macro, independent of tunable
        if self.tunable_dicts[0]['direction'] == 'wrw':
            macro_int_div_vv_t(mc).emit()
            macro_int_div_vs_t(mc).emit()
            macro_int_div_ss_t(mc).emit()
            macro_int_div_rem_vv_t(mc).emit()
            macro_int_div_rem_vs_t(mc).emit()
            macro_int_div_rem_ss_t(mc).emit()

        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            macro_mdiv_u32_ss_t(mc).emit()
            macro_mdiv_u32_rem_ss_t(mc).emit()
            macro_mdiv_u32_vs_t(mc).emit()
            macro_mdiv_u32_rem_vs_t(mc).emit()

        # emit_write_4d_strided_t(self.mc).emit()
        if self.mc.arch_config.arch in (AMDGPU_ARCH_GFX908, AMDGPU_ARCH_GFX90A) and self.mc.arch_config.use_xdlops:
            macro_acc_c_clear_t(mc).emit()
        macro_c_clear_t(mc).emit()
        if self.mc.arch_config.use_dlops:
            self._emit_fma_macro()

    def emit_igemm_macro(self):
        # igemm algorithm related macros
        # emit_v4r1_dynamic_macros(self.mc, self.tunable_dicts)
        for kernel in self.kernel_list:
            if hasattr(kernel, "get_kernel_macros"):
                macro_list = kernel.get_kernel_macros()
                # assert len(macro_list), ''
                for macro in macro_list:
                    self.mc.insert_unique(macro.name(), macro)
        self.mc.emit_all_unique()

    def get_kernel_per_inc_file_name(self, ker, origin_file_name):
        if type(ker) is igemm_upsampling_clear_t:
            return os.path.join(os.path.dirname(origin_file_name), f"{ker.name()}.inc")
        root_file_name = os.path.splitext(origin_file_name)[0]
        return root_file_name + f"_{ker.tunable.gemm_m_per_block:03}x{ker.tunable.gemm_n_per_block:03}x{ker.tunable.gemm_k_per_block:03}" + ".inc"

    def get_kernel_per_s_file_name(self, ker, origin_file_name):
        if type(ker) is igemm_upsampling_clear_t:
            return os.path.join(os.path.dirname(origin_file_name), f"{ker.name()}.s")
        root_file_name = os.path.dirname(origin_file_name)
        return root_file_name + '/' + ker.name() + '.s'

    def get_kernel_util_inc_name(self, origin_file_name):
        kernel_base_name = igemm_gtc_encode_kernel_base_name(igemm_gtc_tunable_parameter_t(self.tunable_dicts[0]), self.mc.arch_config.arch)
        root_file_name = os.path.dirname(origin_file_name)
        return root_file_name + '/' + kernel_base_name + '_utils.inc'

    def emit_kernel_util_inc(self, origin_file_name):
        origin_emitter = self.mc.emitter
        kutil_file_name = self.get_kernel_util_inc_name(origin_file_name)
        kutil_emitter = mc_emit_to_file_t(kutil_file_name, copy.copy(origin_emitter.indent))
        kutil_emitter.open()
        self.mc.emitter = kutil_emitter
        self.emit_global_macro()
        self.mc.emitter = origin_emitter
        kutil_emitter.close()

    def emit_igemm_kernel(self, **options):
        is_multiprocess = True if "emit_kernel_mp" in options and options["emit_kernel_mp"] == True else False
        emit_kernel_per_s = options["split_kernel"]
        emit_kernel_per_inc = IGEMM_EMIT_KERNEL_PER_INC_FILE if not emit_kernel_per_s else False

        # emit the kernel
        #emit_v4r1_dynamic_kernel(self.mc, self.tunable_dicts)
        if emit_kernel_per_inc or emit_kernel_per_s:
            origin_emitter = self.mc.emitter
            assert type(origin_emitter) is mc_emit_to_file_t
            emitter_per_inc_dict = dict()
            if IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE or emit_kernel_per_s:
                kinfo_per_inc_dict = dict()
            self._emit_empty_line()
            self._emit(f";---------------------------------------------------")

        if is_multiprocess:
            kernel_per_inc_dict = dict()
            for kernel in self.kernel_list:
                if emit_kernel_per_inc:
                    kpi_file_name = self.get_kernel_per_inc_file_name(kernel, origin_emitter.file_name)
                    if kpi_file_name not in emitter_per_inc_dict:
                        origin_emitter.emit(f".include \"{os.path.basename(kpi_file_name)}\"")

                        kpi_emitter = mc_emit_to_file_t(kpi_file_name, copy.copy(origin_emitter.indent))
                        kernel.mc.emitter = kpi_emitter
                        # ATTENTION! never open file in one thread/process and use it in another thread/process
                        # kpi_emitter.open()

                        emitter_per_inc_dict[kpi_file_name] = kpi_emitter
                        if IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE:
                            kinfo_per_inc_dict[kpi_file_name] = [kernel.get_kernel_info()]
                        kernel_per_inc_dict[kpi_file_name] = [kernel]
                    else:
                        kernel.mc.emitter = emitter_per_inc_dict[kpi_file_name]
                        if IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE:
                            kinfo_per_inc_dict[kpi_file_name].append(kernel.get_kernel_info())
                        kernel_per_inc_dict[kpi_file_name].append(kernel)

            def concurrent_emit_kernel(emitter, con_kernels):
                emitter.open()  # open/close file in same process
                file_name = con_kernels[0].mc.emitter.file_name
                for kernel in con_kernels:
                    if type(kernel) is not igemm_upsampling_clear_t:
                        kernel._emit(';----------------------------------------------------------')
                        kernel._emit('; starting of kernel {}'.format(kernel.name()))
                        kernel._emit(kernel.tunable.serialize())
                    assert file_name == kernel.mc.emitter.file_name

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
                emitter.close() # open/close file in same process

            workers = list()
            # mp.set_start_method('spawn')
            for k, v in kernel_per_inc_dict.items():
                worker = mp.Process(target=concurrent_emit_kernel, args=(emitter_per_inc_dict[k], v))
                worker.start()
                workers.append(worker)

            for worker in workers:
                worker.join()

        else:
            for kernel in self.kernel_list:
                if emit_kernel_per_inc:
                    kpi_file_name = self.get_kernel_per_inc_file_name(kernel, origin_emitter.file_name)
                    if kpi_file_name not in emitter_per_inc_dict:
                        origin_emitter.emit(f".include \"{os.path.basename(kpi_file_name)}\"")

                        kpi_emitter = mc_emit_to_file_t(kpi_file_name, copy.copy(origin_emitter.indent))
                        kernel.mc.emitter = kpi_emitter
                        kpi_emitter.open()

                        emitter_per_inc_dict[kpi_file_name] = kpi_emitter
                        if IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE:
                            kinfo_per_inc_dict[kpi_file_name] = [kernel.get_kernel_info()]

                    else:
                        kernel.mc.emitter = emitter_per_inc_dict[kpi_file_name]
                        if IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE:
                            kinfo_per_inc_dict[kpi_file_name].append(kernel.get_kernel_info())

                elif emit_kernel_per_s:
                    self.emit_kernel_util_inc(origin_emitter.file_name)
                    kutil_file_name = self.get_kernel_util_inc_name(origin_emitter.file_name)

                    kps_file_name = self.get_kernel_per_s_file_name(kernel, origin_emitter.file_name)
                    if kps_file_name not in emitter_per_inc_dict:

                        kps_emitter = mc_emit_to_file_t(kps_file_name, copy.copy(origin_emitter.indent))
                        kernel.mc.emitter = kps_emitter
                        kps_emitter.open()
                        kernel._emit(f".include \"{os.path.basename(kutil_file_name)}\"")
                        kernel._emit_empty_line()
                        #self.emit_global_macro_per_s_file(kernel.mc)

                        emitter_per_inc_dict[kps_file_name] = kps_emitter
                        kinfo_per_inc_dict[kps_file_name] = [kernel.get_kernel_info()]
                    else:
                        kernel.mc.emitter = emitter_per_inc_dict[kps_file_name]
                        kinfo_per_inc_dict[kps_file_name].append(kernel.get_kernel_info())

                if type(kernel) is not igemm_upsampling_clear_t:
                    kernel._emit(';----------------------------------------------------------')
                    kernel._emit('; starting of kernel {}'.format(kernel.name()))
                    kernel._emit(kernel.tunable.serialize())

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

        if emit_kernel_per_inc:
            for k, v in emitter_per_inc_dict.items():
                if IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE:
                    self.mc.emitter = emitter_per_inc_dict[k]
                    amdgpu_metadata_t(self.mc, kinfo_per_inc_dict[k]).emit()
                # os.chmod(k, 0x777)
                v.close()
            self.mc.emitter = origin_emitter
            self._emit(f";---------------------------------------------------")
            self._emit_empty_line()
        elif emit_kernel_per_s:
            for k, v in emitter_per_inc_dict.items():
                self.mc.emitter = emitter_per_inc_dict[k]
                amdgpu_metadata_t(self.mc, kinfo_per_inc_dict[k]).emit()
                # os.chmod(k, 0x777)
                v.close()
                self._emit(f";---------------------------------------------------")
                self._emit_empty_line()
            origin_emitter.close()
            os.remove(origin_emitter.file_name)

    def emit_metadata(self):
        kernel_info_list = [kernel.get_kernel_info() for kernel in self.kernel_list]
        amdgpu_metadata_t(self.mc, kernel_info_list).emit()

    def do_emit(self, **options):
        self.emit_hsa_header()
        self.emit_global_macro()
        self.emit_igemm_macro()
        self.emit_igemm_kernel(**options)
        if not IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE:
            self.emit_metadata()

    def do_compile(self, **options):
        emit_kernel_per_s = options["split_kernel"]
        if emit_kernel_per_s:
            for kernel in self.kernel_list:
                file_name = self.get_kernel_per_s_file_name(kernel, self.mc.emitter.file_name)
                ass = compile_asm_t(self.mc, file_name)
                rtn = ass.compile()
                if not rtn:
                    assert False
        else:
            ass = compile_asm_t(self.mc, self.mc.emitter.file_name)
            rtn = ass.compile()
            if not rtn:
                assert False

            is_skip_disass = True if "compile_skip_disass" in options and options["compile_skip_disass"] == True else False
            if not is_skip_disass:
                disass = compile_disass_t(self.mc, ass.target_hsaco)
                rtn = disass.compile()
                if not rtn:
                    assert False

    def __call__(self, **options):
        self.do_emit(**options)
        self.do_compile(**options)
