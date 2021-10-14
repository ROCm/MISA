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

from python import *

OUT_DIR='out'

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
    mc_set_current(mc)

    tunable_dicts = [sec.to_dict() for sec in config_content if sec.get_name().startswith('igemm_')]
    for td in tunable_dicts:
        td['arch'] = sec_root['arch']       # append arch to each section

    codegen_driver_t(mc, tunable_dicts)(split_kernel = args.split_kernel)

    # os.chmod(asm_target, 0x777)

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

def igemm_check_fp16_configs(config_content):
    tunable_dicts = [sec.to_dict() for sec in config_content if sec.get_name().startswith('igemm_')]
    for td in tunable_dicts:
        if "fp16" in td['precision']:
            return True
    return False

def igemm_check_int8_configs(config_content):
    tunable_dicts = [sec.to_dict() for sec in config_content if sec.get_name().startswith('igemm_')]
    for td in tunable_dicts:
        if "int8" in td['precision']:
            return True
    return False

def igemm_check_bf16_configs(config_content):
    tunable_dicts = [sec.to_dict() for sec in config_content if sec.get_name().startswith('igemm_')]
    for td in tunable_dicts:
        if "bf16" in td['precision']:
            return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="config file as input")
    parser.add_argument("-d", "--dir", help="directory of output files", default = OUT_DIR)
    parser.add_argument("-output", nargs='?', const='tunable_parameter_list.txt', help="output tunable parameter list")
    parser.add_argument("-s", "--split_kernel", action="store_true")
    args = parser.parse_args()

    config_parser = config_parser_t(args.config_file)
    #print(os.getcwd())
    config_content = config_parser()
    #config_content.dump()
    #print(args.output)
    if args.output:
        igemm_out_tunable_param(args.output, config_content)

    arch = config_content.get_section('codegen')[0]['arch']
    code_object = config_content.get_section('codegen')[0]['code_object']
    has_fp16_config = igemm_check_fp16_configs(config_content)
    has_int8_config = igemm_check_int8_configs(config_content)
    has_bf16_config = igemm_check_bf16_configs(config_content)

    if config_content.get_section('codegen')[0]['mode'] in ('flat', 'flatten'):
        if os.path.exists(args.dir):
            shutil.rmtree(args.dir)
        os.mkdir(args.dir)
        cxxflags = []
        if args.split_kernel:
            cxxflags += ["-DIGEMM_SPLIT_KERNEL"]
        host_driver(cxxflags=cxxflags, arch=arch, config_file=args.config_file, out_dir=args.dir, has_fp16_config=has_fp16_config, has_int8_config=has_int8_config, has_bf16_config=has_bf16_config)
        igemm_flatten(args, config_content)

    if config_content.get_section('codegen')[0]['mode'] in ('seq', 'sequencer'):
        # config_content.dump()
        # igemm_sequence(args, config_content)
        if os.path.exists(args.dir):
            shutil.rmtree(args.dir)
        os.mkdir(args.dir)
        sequence_driver(arch=arch, code_object=code_object,
                            config_content=config_content, out_dir=args.dir )


