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
from __future__ import print_function
import argparse
import sys, os, shutil

from python.direct_driver import *

OUT_DIR='out_direct'

def arch_cfg_from_main_cfg(config_content : config_content_t) -> amdgpu_arch_config_t:
    sec_root = config_content.get_section('codegen')[0]
    return amdgpu_arch_config_t({
        'arch'          :   amdgpu_string_to_arch( sec_root['arch'] ),
        'data_type'     :   AMDGPU_PRECISION_FP32,
        'code_object'   :   amdgpu_string_to_codeobj( sec_root['code_object']) })




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="config file as input")
    parser.add_argument("-out_file", help="output dest file for code")
    parser.add_argument("-d", "--dir", help="directory of output files", default = OUT_DIR)
    parser.add_argument("-output_tun", nargs='?', const='tunable_parameter_list.txt', help="output tunable parameter list")
    args = parser.parse_args()
else:
    args = obj = lambda: None
    setattr(args, 'dir', OUT_DIR)
    setattr(args, 'config_file', 'config/direct.config')
    setattr(args, 'out_file', None)

config_parser = config_parser_t(args.config_file)
#print(os.getcwd())
config_content = config_parser()
#config_content.dump()
#print(args.output)

#config_direct = direct_navi_config(config_parser())
config_direct = direct_1x1u_config(config_parser())

if os.path.exists(args.dir):
    shutil.rmtree(args.dir)
os.mkdir(args.dir)

#direct = direct_navi(config_content)
if(args.out_file):
    asm_target = os.path.join(args.dir, args.out_file)
else:
    asm_target = os.path.join(args.dir, os.path.splitext(os.path.basename(args.config_file))[0] + '.s')

emitter = mc_emit_to_file_t(asm_target)
arch = arch_cfg_from_main_cfg(config_content)
mc = mc_asm_printer_t(emitter, arch)

direct_driver_t(mc, config_direct)()

    