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
import os
import subprocess

from .amdgpu import *
import os

IGEMM_HOST_USE_GPU_NAIVE_CONV = True
IGEMM_HOST_USE_XDNN = False
IGEMM_HOST_USE_MAGIC_DIV = True
IGEMM_HOST_USE_HIPCC = True # hipclang perfer use hipcc to compile host code

def _check_hip_clang():
    return os.path.exists('/opt/rocm/llvm/bin/clang++')

class compile_hip_t(object):
    def __init__(self, arch_config, hip_file_name, target_hsaco = ''):
        self.hip_file_name = hip_file_name
        if target_hsaco == '':
            self.target_hsaco = os.path.splitext(hip_file_name)[0] + '.hsaco'
        else:
            self.target_hsaco = target_hsaco
        self.arch_config = arch_config
    def compile(self, **kwargs):
        # make sure mc output is closed
        # self.mc.close()

        arch_str = amdgpu_arch_to_string(self.arch_config.arch)
        use_hip_clang = _check_hip_clang()
        if use_hip_clang:
            cmd = ['/opt/rocm/hip/bin/hipcc']
        else:
            cmd = ['/opt/rocm/hip/bin/hipcc']
        cmd += ['-x', 'hip']
        cmd += ['--cuda-gpu-arch={}'.format(arch_str)]
        cmd += ['--cuda-device-only', '-c', '-O3']
        cmd += ['{}'.format(self.hip_file_name)]
        cmd += ['-o', '{}'.format(self.target_hsaco)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.STDOUT)
        # print("[hip] " + " ".join(cmd))
        try:
            (out, _) = p.communicate()
            if p.returncode != 0:
                print('build fail:{}'.format(" ".join(cmd)))
                print('{}'.format(out.decode('utf-8')))
                return False
            return True
        except Exception as e:
            print('fail to run cmd:{}'.format(" ".join(cmd)))
            print('err:{}'.format(e))
            return False


class compile_asm_t(object):
    def __init__(self, mc, asm_file_name, target_hsaco = ''):
        self.asm_file_name = asm_file_name
        if target_hsaco == '':
            self.target_hsaco = os.path.splitext(asm_file_name)[0] + '.hsaco'
        else:
            self.target_hsaco = target_hsaco
        self.mc = mc
    def compile(self, **kwargs):
        # make sure mc output is closed
        self.mc.close()

        arch_str = amdgpu_arch_to_string(self.mc.arch_config.arch)
        use_hip_clang = _check_hip_clang()
        if use_hip_clang:
            cmd = ['/opt/rocm/llvm/bin/clang++']
        else:
            cmd = ['/opt/rocm/hcc/bin/clang']
        cmd += ['-x', 'assembler']
        cmd += ['-I{}'.format(os.path.dirname(self.asm_file_name))]
        cmd += ['-target', 'amdgcn--amdhsa']
        cmd += ['-mcpu={}'.format(arch_str)]
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V2:
            cmd += ['-mno-code-object-v3']
        # TODO: current compiler treat cov3 as default, so no need add extra flag
        cmd += ['{}'.format(self.asm_file_name)]
        cmd += ['-o', '{}'.format(self.target_hsaco)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.STDOUT)
        # print("[asm] " + " ".join(cmd))
        try:
            (out, _) = p.communicate()
            if p.returncode != 0:
                print('build fail:{}'.format(" ".join(cmd)))
                print('{}'.format(out.decode('utf-8')))
                return False
            return True
        except Exception as e:
            print('fail to run cmd:{}'.format(" ".join(cmd)))
            print('err:{}'.format(e))
            return False

class compile_disass_t(object):
    def __init__(self, mc, hsaco_file_name, target_disass = ''):
        self.hsaco_file_name = hsaco_file_name
        if target_disass == '':
            self.target_disass = os.path.splitext(hsaco_file_name)[0] + '.disass.s'
        else:
            self.target_disass = target_disass
        self.mc = mc
    def compile(self, **kwargs):
        arch_str = amdgpu_arch_to_string(self.mc.arch_config.arch)
        use_hip_clang = _check_hip_clang()
        if use_hip_clang:
            cmd = ['/opt/rocm/llvm/bin/llvm-objdump']
            cmd += ['--disassemble']
            cmd += ['--mcpu={}'.format(arch_str)]
        else:
            cmd = ['/opt/rocm/hcc/bin/llvm-objdump']
            cmd += ['-disassemble']
            cmd += ['-mcpu={}'.format(arch_str)]

        cmd += ['{}'.format(self.hsaco_file_name)]
        # cmd += ['>', '{}'.format(self.target_disass)]
        # print("[dis] " + " ".join(cmd))
        try:
            fp = open(self.target_disass, "w")
            p = subprocess.Popen(cmd, stdout=fp)
            try:
                (out, _) = p.communicate()
                if p.returncode != 0:
                    print('build fail:{}'.format(" ".join(cmd)))
                    print('{}'.format(out.decode('utf-8')))
                    return False
                return True
            except Exception as e:
                print('fail to run cmd:{}'.format(" ".join(cmd)))
                print('err:{}'.format(e))
                return False
        except IOError as e:
                print("can't open file:{}({})".format(self.target_disass, e))
                sys.exit()

class compile_host_t(object):
    def __init__(self, arch_config, host_cpp, target_exec = ''):
        self.host_cpp = host_cpp
        if target_exec == '':
            if type(host_cpp) is str:
                self.target_exec = os.path.splitext(host_cpp)[0] + '.exe'
            elif type(host_cpp) is list:
                self.target_exec = os.path.splitext(host_cpp[0])[0] + '.exe'
            else:
                assert False
        else:
            self.target_exec = target_exec
        self.arch_config = arch_config
    def compile(self, **kwargs):
        arch_str = amdgpu_arch_to_string(self.arch_config.arch)
        use_hip_clang = _check_hip_clang()
        xdnnroot ='2f6f70742f696e74656c2f696e74656c6f6e656170692f6f6e65444e4e2f6c61746573742f6370755f676f6d702f'
        if use_hip_clang:
            if IGEMM_HOST_USE_HIPCC:
                cmd = ['/opt/rocm/hip/bin/hipcc']
                cmd += ['-std=c++14']
            else:
                cmd = ['g++']
                cmd += ['-D__HIP_PLATFORM_HCC__=','-I/opt/rocm/hip/include', '-I/opt/rocm/hcc/include', '-I/opt/rocm/hsa/include']
                cmd += ['-Wall','-O2', '-std=c++11']
            if IGEMM_HOST_USE_XDNN:
                cmd += [f'-I{bytes.fromhex(xdnnroot).decode()}/include', '-DUSE_XDNN']
            if IGEMM_HOST_USE_MAGIC_DIV:
                cmd += ['-DUSE_MAGIC_DIV=1']
            if 'cflags' in kwargs:
                cmd += kwargs['cflags']
            if 'cxxflags' in kwargs:
                cmd += kwargs['cxxflags']
            if type(self.host_cpp) is str:
                cmd += [self.host_cpp]
            elif type(self.host_cpp) is list:
                cmd += self.host_cpp     # for multiple files
            else:
                assert False
            if not IGEMM_HOST_USE_HIPCC:
                cmd += ['-L/opt/rocm/lib', '-L/opt/rocm/lib64', '-Wl,-rpath=/opt/rocm/lib',
                        '-ldl', '-lm', '-lpthread',
                        '-Wl,--whole-archive', '-lamdhip64', '-lhsa-runtime64', '-lhsakmt', '-Wl,--no-whole-archive']
            if IGEMM_HOST_USE_XDNN:
                cmd += [f'-L{bytes.fromhex(xdnnroot).decode()}/lib', f"-l{bytes.fromhex('646e6e6c').decode()}", f'-Wl,-rpath={bytes.fromhex(xdnnroot).decode()}/lib']
            cmd += ['-o', self.target_exec]
        else:
            cmd = ['g++']
            # from `/opt/rocm/bin/hipconfig --cpp_config`
            cmd += ['-D__HIP_PLATFORM_HCC__=','-I/opt/rocm/hip/include', '-I/opt/rocm/hcc/include', '-I/opt/rocm/hsa/include']
            cmd += ['-Wall','-O2', '-std=c++11']
            if IGEMM_HOST_USE_XDNN:
                cmd += [f'-I{bytes.fromhex(xdnnroot).decode()}/include', '-DUSE_XDNN']
            if IGEMM_HOST_USE_MAGIC_DIV:
                cmd += ['-DUSE_MAGIC_DIV']
            if 'cflags' in kwargs:
                cmd += kwargs['cflags']
            if 'cxxflags' in kwargs:
                cmd += kwargs['cxxflags']
            if type(self.host_cpp) is str:
                cmd += [self.host_cpp]
            elif type(self.host_cpp) is list:
                cmd += self.host_cpp     # for multiple files
            else:
                assert False
            cmd += ['-L/opt/rocm/hcc/lib', '-L/opt/rocm/lib', '-L/opt/rocm/lib64', '-Wl,-rpath=/opt/rocm/hcc/lib:/opt/rocm/lib',
                    '-ldl', '-lm', '-lpthread', '-lhc_am',
                    '-Wl,--whole-archive', '-lmcwamp', '-lhip_hcc', '-lhsa-runtime64', '-lhsakmt', '-Wl,--no-whole-archive']
            if IGEMM_HOST_USE_XDNN:
                cmd += [f'-L{bytes.fromhex(xdnnroot).decode()}/lib', f"-l{bytes.fromhex('646e6e6c').decode()}", f'-Wl,-rpath={bytes.fromhex(xdnnroot).decode()}/lib']
            cmd += ['-o', self.target_exec]
        # print("[host] " + " ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.STDOUT)
        try:
            (out, _) = p.communicate()
            if p.returncode != 0:
                print('build fail:{}'.format(" ".join(cmd)))
                print('{}'.format(out.decode('utf-8')))
                return False
            return True
        except Exception as e:
            print('fail to run cmd:{}'.format(" ".join(cmd)))
            print('err:{}'.format(e))
            return False
