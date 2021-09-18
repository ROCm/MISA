import re
import argparse
import os

class solver_insertion(object):
    def __init__(self, dir, precision, arch, miopen):
        self.dir = dir
        self.miopen = miopen
        self.precision = precision
        self.arch = arch
        assert self.dir in ("fwd", "bwd", "wrw")
        assert self.precision in ("fp16", "fp32")
        assert self.arch in ("gfx908", "gfx90a")

    def remove_old_asm_files(self):
        arch = "" if self.arch == "gfx908" else "_gfx90a"
        asm_path = os.path.join(self.miopen, f'src/kernels/dynamic_igemm/igemm_gtc_xdlops_nhwc{arch}/{self.dir}_{self.precision}/*.s')
        inc_path = os.path.join(self.miopen, f'src/kernels/dynamic_igemm/igemm_gtc_xdlops_nhwc{arch}/{self.dir}_{self.precision}/*.inc')
        os.system(f"rm {asm_path} {inc_path}")

    def insert_new_asm_files(self):
        asm_src_path = f'out/*.s'
        inc_src_path = f'out/*.inc'
        arch = "" if self.arch == "gfx908" else "_gfx90a"
        dst_path = os.path.join(self.miopen, f'src/kernels/dynamic_igemm/igemm_gtc_xdlops_nhwc{arch}/{self.dir}_{self.precision}/')
        os.system(f"cp {asm_src_path} {dst_path}")
        os.system(f"cp {inc_src_path} {dst_path}")

    def get_new_param_list(self):
        with open("./tunable_parameter_list.txt", 'r') as f:
            log_lines = f.readlines()
            param_list = []
            for line in log_lines:
                if self.precision == "fp16":
                    if self.dir == 'fwd':
                        res = re.search(r'.*{"fwd", "nhwc", miopenHalf,.*\n', line)
                    if self.dir == 'bwd':
                        res = re.search(r'.*{"bwd", "nhwc", miopenHalf,.*\n', line)
                    if self.dir == 'wrw':
                        res = re.search(r'.*{"wrw", "nhwc", miopenHalf,.*\n', line)
                if self.precision == "fp32":
                    if self.dir == 'fwd':
                        res = re.search(r'.*{"fwd", "nhwc", miopenFloat,.*\n', line)
                    if self.dir == 'bwd':
                        res = re.search(r'.*{"bwd", "nhwc", miopenFloat,.*\n', line)
                    if self.dir == 'wrw':
                        res = re.search(r'.*{"wrw", "nhwc", miopenFloat,.*\n', line)
                if res != None:
                    param_list.append(res.group())
            return param_list

    def insert_new_param_list_to_solver(self, new_param_list):
        assert isinstance(new_param_list, list)
        solver_path = os.path.join(self.miopen, f'src/solver/conv_asm_implicit_gemm_gtc_{self.dir}_nhwc.cpp')
        with open(solver_path, 'r') as f:
            solver_lines = f.readlines()
            new_solver_lines = []
            solver_before_insertion = []
            solver_after_insertion = []
            find_insert_pos = 0
            for line in solver_lines:
                if self.precision == "fp16":
                    if self.dir == 'fwd':
                        res = re.search(r'.*{"fwd", "nhwc", miopenHalf,.*\n', line)
                    if self.dir == 'bwd':
                        res = re.search(r'.*{"bwd", "nhwc", miopenHalf,.*\n', line)
                    if self.dir == 'wrw':
                        res = re.search(r'.*{"wrw", "nhwc", miopenHalf,.*\n', line)
                if self.precision == "fp32":
                    if self.dir == 'fwd':
                        res = re.search(r'.*{"fwd", "nhwc", miopenFloat,.*\n', line)
                    if self.dir == 'bwd':
                        res = re.search(r'.*{"bwd", "nhwc", miopenFloat,.*\n', line)
                    if self.dir == 'wrw':
                        res = re.search(r'.*{"wrw", "nhwc", miopenFloat,.*\n', line)
                if res != None:
                    find_insert_pos = 1
                else:
                    if find_insert_pos == 0:
                        solver_before_insertion.append(line)
                    else:
                        solver_after_insertion.append(line)

            for line in solver_before_insertion:
                new_solver_lines.append(line)
            for line in new_param_list:
                new_solver_lines.append(line)
            for line in solver_after_insertion:
                new_solver_lines.append(line)

        with open(solver_path, 'w') as f:
            f.writelines(new_solver_lines)

    def python_code_generation(self):
        prec = "" if self.precision == "fp32" else "_fp16"
        config_file = f'config/igemm_{self.dir}_gtc_{self.arch}_nhwc{prec}.config'
        os.system(f'python igemm_codegen.py {config_file} -output -s')

def merge_igemmgen_to_miopen(dir, precision, arch, miopen):
    def param_and_asm_insertion(dir, precision, arch,miopen):
        s_i = solver_insertion(dir, precision, arch, miopen)
        s_i.python_code_generation()
        new_parpam_list = s_i.get_new_param_list()
        s_i.insert_new_param_list_to_solver(new_parpam_list)
        s_i.remove_old_asm_files()
        s_i.insert_new_asm_files()

    if dir == 1:
        param_and_asm_insertion("fwd", precision, arch, miopen)

    if dir == 2:
        param_and_asm_insertion("bwd", precision, arch, miopen)

    if dir == 4:
        param_and_asm_insertion("wrw", precision, arch, miopen)

    if dir == 0:
        for each_dir in ["fwd", "bwd", "wrw"]:
            param_and_asm_insertion(each_dir, precision, arch, miopen)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", default='0', help="direction, all/fwd/bwd/wrw, encoding by 0/1/2/4")
    parser.add_argument("-p", default='fp16', help='pricision: data type fp32/fp16')
    parser.add_argument("-a", default='gfx908', help='arch: gfx908/gfx90a')
    parser.add_argument("-miopen", default='../../miopen_develop/nhwc_miopen_fp16/', help="dir of miopen")
    args = parser.parse_args()

    assert args.d in ('0', '1', '2', '4')
    assert os.path.exists(args.miopen)

    merge_igemmgen_to_miopen(int(args.d), args.p, args.a, args.miopen)
    
