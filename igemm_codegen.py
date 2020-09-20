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
import argparse
import sys, os, shutil

from igemm import *

OUT_DIR='out'
CPP_DIR='driver'

def igemm_host_driver(args, config_content):
    cpp_src = os.path.join(CPP_DIR, "conv_driver.cpp")
    target_exe = os.path.join(args.dir, "conv_driver.exe")
    sec_root = config_content.get_section('codegen')[0]
    arch = amdgpu_arch_config_t({
        'arch'          :   amdgpu_string_to_arch(sec_root['arch'])})
    builder = compile_host_t(arch, cpp_src, target_exe)
    config_file_name = os.path.abspath(args.config_file)
    hsaco_name = os.path.splitext(os.path.basename(args.config_file))[0] + '.hsaco'
    rtn = builder.compile(cxxflags=['-DIGEMM_CONFIG_FILE=\"{}\"'.format(config_file_name), \
                        '-DIGEMM_HSACO=\"{}\"'.format(hsaco_name)])
    if not rtn:
        assert False

def igemm_flatten(args, config_content):
    asm_target = os.path.join(args.dir, os.path.splitext(os.path.basename(args.config_file))[0] + '.s')
    emitter = mc_emit_to_file_t(asm_target)
    sec_root = config_content.get_section('codegen')[0]
    arch = amdgpu_arch_config_t({
        'arch'          :   amdgpu_string_to_arch( sec_root['arch'] ),
        'data_type'     :   AMDGPU_PRECISION_FP32,
        'code_object'   :   amdgpu_string_to_codeobj( sec_root['code_object']) })

    # create mc
    mc = mc_asm_printer_t(emitter, arch)

    tunable_dicts = [sec.to_dict() for sec in config_content if sec.get_name().startswith('igemm_')]
    for td in tunable_dicts:
        td['arch'] = sec_root['arch']       # append arch to each section

    igemm_codegen_driver_t(mc, tunable_dicts)()

def igemm_out_tunable_param(output_file, config_content):
    sec_root = config_content.get_section('codegen')[0]
    list_emitter = mc_emit_to_file_t(output_file)
    list_emitter.open()
    tunable_dicts = [sec.to_dict() for sec in config_content if sec.get_name().startswith('igemm_')]
    for td in tunable_dicts:
        td['arch'] = sec_root['arch']       # append arch to each section
        td_item = igemm_gtc_tunable_parameter_t(td)
        list_emitter.emit(td_item.output())
    list_emitter.close()

#def igemm_sequence(args, config_content):
#    kseq = v4r1_dynamic_kernel_sequencer_t(amdgpu_get_gfx906_60cu(),
#            config_content.get_section('v4r1_dynamic_kernel')[0].to_dict())
#    kseq()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="config file as input")
    parser.add_argument("-d", "--dir", help="directory of output files", default = OUT_DIR)
    parser.add_argument("-output", nargs='?', const='tunable_parameter_list.txt', help="output tunable parameter list")
    args = parser.parse_args()

    config_parser = config_parser_t(args.config_file)
    #print(os.getcwd())
    config_content = config_parser()
    #config_content.dump()
    #print(args.output)
    if args.output:
        igemm_out_tunable_param(args.output, config_content)

    if config_content.get_section('codegen')[0]['mode'] in ('flat', 'flatten'):
        if os.path.exists(args.dir):
            shutil.rmtree(args.dir)
        os.mkdir(args.dir)
        igemm_host_driver(args, config_content)
        igemm_flatten(args, config_content)

    if config_content.get_section('codegen')[0]['mode'] in ('seq', 'sequencer'):
        # config_content.dump()
        # igemm_sequence(args, config_content)
        pass
