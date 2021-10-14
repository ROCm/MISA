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

from .codegen import *

IGEMM_HOST_USE_GPU_NAIVE_CONV = True


def host_driver(**options):
    def get_dict_with_default(some_dict, key, default_value):
        if key in some_dict:
            return some_dict[key]
        return default_value

    arch = get_dict_with_default(options, 'arch', 'gfx908')
    has_fp16_config = get_dict_with_default(options, 'has_fp16_config', False)
    has_int8_config = get_dict_with_default(options, 'has_int8_config', False)
    has_bf16_config = get_dict_with_default(options, 'has_bf16_config', False)
    out_dir = get_dict_with_default(options, 'out_dir', 'out')
    cpp_dir = get_dict_with_default(options, 'cpp_dir', 'driver')
    cpp_name = get_dict_with_default(options, 'cpp_name', 'conv_driver.cpp')
    target_name = get_dict_with_default(options, 'target_name', 'conv_driver.exe')
    config_file = get_dict_with_default(options, 'config_file', 'config/igemm_bwd_gtc_gfx908.config')
    config_file_name = os.path.abspath(config_file)
    hsaco_name = get_dict_with_default(options, "hsaco_name", os.path.splitext(os.path.basename(config_file))[0] + '.hsaco')
    cxxflags = get_dict_with_default(options, "cxxflags", list())
    use_gpu_reference_kernel = get_dict_with_default(options, "use_gpu_reference_kernel", IGEMM_HOST_USE_GPU_NAIVE_CONV)

    #cpp_src = os.path.join(cpp_dir, cpp_name)
    cpp_src = [os.path.join(cpp_dir, 'conv_driver.cpp'), os.path.join(cpp_dir, 'perf', 'gmap.cpp')]
    target_exe = os.path.join(out_dir, target_name)
    arch_config = amdgpu_arch_config_t({
        'arch'          :   amdgpu_string_to_arch(arch)})
    builder = compile_host_t(arch_config, cpp_src, target_exe)

    host_cxxflags = ['-DIGEMM_CONFIG_FILE=\"{}\"'.format(config_file_name), '-DIGEMM_HSACO=\"{}\"'.format(hsaco_name)]
    host_cxxflags += ['-I{}'.format(cpp_dir)]
    if has_fp16_config:
        host_cxxflags += ['-DUSE_HALF']
    if has_int8_config:
        host_cxxflags += ['-DUSE_INT8']
    if has_bf16_config:
        host_cxxflags += ['-DUSE_BF16']
    if use_gpu_reference_kernel:
        host_cxxflags += ['-DUSE_GPU_NAIVE_CONV']
    if len(cxxflags) != 0:
        assert type(cxxflags) is list
        host_cxxflags.extend(cxxflags)
    rtn = builder.compile(cxxflags=host_cxxflags)
    if not rtn:
        assert False

    if use_gpu_reference_kernel:
        hip_src = os.path.join(cpp_dir, "gpu_naive_conv", "naive_conv.cpp")
        target_hsaco = os.path.join(out_dir, "naive_conv.hsaco")
        hip_builder = compile_hip_t(arch_config, hip_src, target_hsaco)
        rtn = hip_builder.compile()
        if not rtn:
            assert False

    # compile tensor cast code
    hip_src = os.path.join(cpp_dir, "gpu_tensor_cast", "gpu_tensor_cast.cpp")
    target_hsaco = os.path.join(out_dir, "igemm_gtc_tensor_cast.hsaco")
    hip_builder = compile_hip_t(arch_config, hip_src, target_hsaco)
    rtn = hip_builder.compile()
    if not rtn:
        assert False
