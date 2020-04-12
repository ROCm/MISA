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
from __future__ import print_function
import sys, os, shutil
import numpy as np

from igemm.amdgpu import *
from igemm.codegen import *
from igemm.igemm_base import *
from igemm.igemm_algo_v4r1 import *

OUT_DIR='out'
CPP_DIR='driver'

def igemm_host_driver():
    cpp_src = os.path.join(CPP_DIR, "conv_driver.cpp")
    target_exe = os.path.join(OUT_DIR, "conv_driver.exe")
    arch = amdgpu_arch_config_t({
        'Arch'          :   AMDGPU_ARCH_GFX906})
    builder = amdgpu_build_host_t(arch, cpp_src, target_exe)
    rtn = builder.build()
    if not rtn:
        assert False

def igemm_v4r1_emit():
    '''
    codegen driver for v4r1
    '''
    asm_target = os.path.join(OUT_DIR, "igemm_v4r1_dynamic.s")
    emitter = codegen_emit_to_file_t(asm_target)
    arch = amdgpu_arch_config_t({
        'Arch'          :   AMDGPU_ARCH_GFX906,
        'DataType'      :   AMDGPU_PRECISION_FP32 })

    # create mc
    mc = codegen_asm_printer_t(emitter, arch)

    # emit hsa header, for once. This be will ignored in cov3
    emit_hsa_header_t(mc).emit()

    # emit global macro, independent of tunable
    emit_int_div_vv_t(mc).emit()
    emit_int_div_vs_t(mc).emit()
    emit_int_div_ss_t(mc).emit()
    emit_write_4d_strided_t(mc).emit()
    emit_c_clear_t(mc).emit()

    tunable_dicts = igemm_get_v4r1_tunable_dict_array()

    # emit v4r1 related macros, with different tunable
    emit_v4r1_dynamic_macros(mc, tunable_dicts)

    # emit the kernel
    emit_v4r1_dynamic_kernel(mc, tunable_dicts)


    builder = amdgpu_build_asm_t(mc, asm_target)
    rtn = builder.build()
    if not rtn:
        assert False

if __name__ == '__main__':
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.mkdir(OUT_DIR)
    igemm_host_driver()
    igemm_v4r1_emit()
