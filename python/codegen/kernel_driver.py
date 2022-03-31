################################################################################
# 
#  MIT License
# 
#  Copyright (c) 2021 Advanced Micro Devices, Inc.
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

import os
import copy
import multiprocessing as mp
from abc import ABC, abstractmethod
from python.codegen.mc import *
from python.codegen.config_parser import config_content_t

class base_config(ABC):
    def __init__(self, config_content: config_content_t, suf:str=''):

        def get_gpu_arch(config_content: config_content_t) -> str:
            arch = config_content.get_section('codegen')[0]['arch']
            return arch

        def get_data_type(config_content: config_content_t) -> str:
            data_type = config_content.get_section('direct' + suf)[0]['data_type']
            return data_type

        def get_layout(config_content: config_content_t) -> str:
            layout = config_content.get_section('direct' + suf)[0]['layout']
            return layout
        
        def get_direction(config_content: config_content_t) -> str:
            direction = config_content.get_section('direct' + suf)[0]['direction']
            return direction
        
        self.gpu = get_gpu_arch(config_content)
        self.data_type = get_data_type(config_content)
        self.layout = get_layout(config_content)
        self.direction = get_direction(config_content)


from ..codegen import *
from ..direct.kernel_constructor import kernel_constructor
from typing import List
from typing import Dict

class base_driver_t(mc_base_t, ABC):
    def __init__(self, mc : mc_asm_printer_t, _config:base_config):
        mc_base_t.__init__(self, mc)
        #self.tunable_dicts = direct_config
        assert  issubclass(type(_config), base_config)
        self._config = _config

        self.kernel_list:List[kernel_constructor] = []

    def emit_hsa_header(self):
        hsa_header_t(self.mc).emit()
    
    def emit_metadata(self):
        kernel_info_list = [kernel.kernel_info for kernel in self.kernel_list]
        amdgpu_metadata_t(self.mc, kernel_info_list).emit()

    def emit_kernel(self, **options):
        is_multiprocess = True if "emit_kernel_mp" in options and options["emit_kernel_mp"] == True else False
        emit_kernel_per_s = options.get("split_kernel", True)
        emit_kernel_per_inc = True if not emit_kernel_per_s else False

        origin_emitter = self.mc.emitter
        assert type(origin_emitter) is mc_emit_to_file_t
        
        emitter_per_inc_dict:Dict[str,mc_emit_to_file_t] = dict()
        
        #if IGEMM_EMIT_KERNEL_METADATA_PER_INC_FILE or emit_kernel_per_s:
        kinfo_per_inc_dict = dict()

        if emit_kernel_per_inc or emit_kernel_per_s:                
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
                        #kernel._emit(kernel.tunable.serialize())
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

                    kps_file_name = self.get_kernel_per_s_file_name(kernel, origin_emitter.file_name)
                    if kps_file_name not in emitter_per_inc_dict.keys():

                        kps_emitter = mc_emit_to_file_t(kps_file_name, copy.copy(origin_emitter.indent))
                        kernel.mc.emitter = kps_emitter
                        kps_emitter.open()
                        #kernel._emit(f".include \"{os.path.basename(origin_emitter.file_name)}\"")
                        #self.emit_global_macro_per_s_file(kernel.mc)

                        emitter_per_inc_dict[kps_file_name] = kps_emitter
                        kinfo_per_inc_dict[kps_file_name] = [kernel.kernel_info]
                    else:
                        kernel.mc.emitter = emitter_per_inc_dict[kps_file_name]
                        kinfo_per_inc_dict[kps_file_name].append(kernel.kernel_info)


                kernel._emit(';----------------------------------------------------------')
                kernel._emit('; starting of kernel {}'.format(kernel.get_kernel_name()))
                #kernel._emit(kernel.tunable.serialize())

                kernel.emit_kernel_code()


                #with kernel._indent_context():
                #    if kernel.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V2:
                #        kernel.emit_kernel_amd_kernel_code_t()
                #    kernel.emit_kernel_body()
                #    kernel.emit_kernel_end()
                
                #if kernel.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
                kernel.emit_kernel_amd_kernel_code_t()
                kernel.emit_kernel_footer()

        if emit_kernel_per_inc:
            for k, v in emitter_per_inc_dict.items():
                if True:
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



    def do_emit(self, **options):
        self.emit_hsa_header()
        self.emit_kernel(**options)
        if(not options.get('separate_metadata', False)==True):
            self.emit_metadata()

    #def get_filename(self, **options):
    def get_kernel_per_inc_file_name(self, ker:kernel_constructor, origin_file_name):
        if (issubclass(type(ker), kernel_constructor)):
            return os.path.join(os.path.dirname(origin_file_name), f"{ker.get_kernel_name()}.inc")
        
        root_file_name = os.path.splitext(origin_file_name)[0]
        return root_file_name + f"_{ker.tunable.gemm_m_per_block:03}x{ker.tunable.gemm_n_per_block:03}x{ker.tunable.gemm_k_per_block:03}" + ".inc"

    def get_kernel_per_s_file_name(self, ker, origin_file_name):
        if (not issubclass(type(ker), kernel_constructor)):
            return os.path.join(os.path.dirname(origin_file_name), f"{ker.name()}.s")
        
        root_file_name = os.path.dirname(origin_file_name)
        return root_file_name + '/' + ker.get_kernel_name() + '.s'

    def emit_global_macro_per_s_file(self):
        #TODO list to track duplicates
        self._emit(f";TODO global macro")

    def do_compile(self, **options):
        return 1
        emit_kernel_per_s = options["split_kernel"]
        if emit_kernel_per_s:
            for kernel in self.kernel_list:
                file_name = self.get_kernel_per_s_file_name(kernel, self.mc.emitter.get_filename())
                ass = compile_asm_t(self.mc, file_name)
                rtn = ass.compile()
                if not rtn:
                    assert False
        else:
            ass = compile_asm_t(self.mc, self.mc.emitter.get_filename())
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
        if(not options.get('no_compile', False)):
            self.do_compile(**options)