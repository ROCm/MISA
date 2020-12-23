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
from .igemm_codegen_driver import igemm_codegen_driver_t

import os
import copy
import math

class igemm_sequence_xdlops_t(mc_base_t):
    def __init__(self, mc, config):
        mc_base_t.__init__(self, mc)
        self.config = config

    def __call__(self):
        config = self.config
        gemm_m_per_block_list = config["gemm_m_per_block"] if type(config["gemm_m_per_block"]) is list else config["gemm_m_per_block"]
        gemm_n_per_block_list = config["gemm_n_per_block"] if type(config["gemm_n_per_block"]) is list else config["gemm_n_per_block"]
        gemm_k_per_block_list = config["gemm_k_per_block"] if type(config["gemm_k_per_block"]) is list else config["gemm_k_per_block"]
        options = config["options"] if "options" in config else dict()
        assert type(options) is dict, f"fail to get options:{options}, type:{type(options)}"

        def search_xdlops_mapping_from_m_n(macro_tile_m, macro_tile_n):
            valid_mapping_list = []
            for ctrl in ctrl_xdlops_mapping_fp32:
                if ctrl.macro_tile_m == macro_tile_m and \
                    ctrl.macro_tile_n == macro_tile_n:
                    valid_mapping_list.append(ctrl)
            # assert len(valid_mapping_list) != 0, f"no macro_tile hit for {macro_tile_m}x{macro_tile_n}"
            return valid_mapping_list

        def search_xdlops_thread_cluster_lengths(direction, macro_tile_m, macro_tile_n, macro_tile_k, block_size):
            # to generate a combination of ta[4], ca[4], tb[4], cb[4], nxb, nxe
            #
            # ta_0, ta_1, ta_2, ta_3
            # ca_0, ca_1, ca_2, ca_3
            # tb_0, tb_1, tb_2, tb_3
            # cb_0, cb_1, cb_2, cb_3
            #
            thread_cluster_list = []
            ta_0 = ca_0 = tb_0 = cb_0 = 1
            ca_2 = cb_2 = 1
            if direction == 'fwd':
                # a, weight, C0xC1ExK0xK1
                # b, input,  C0xC1ExN0xN1B
                # gemm_k iter
                vector_load_size = [1, 2, 4]
                for ta_1 in vector_load_size:
                    if ta_1 > macro_tile_k or (macro_tile_k % ta_1 != 0):
                        continue
                    ca_1 = macro_tile_k // ta_1
                    if ca_1 > block_size or (block_size % ca_1 != 0):
                        continue

                    ca_3 = block_size // ca_1
                    if ca_3 > macro_tile_m:
                        continue

                    for tb_1 in vector_load_size:
                        if tb_1 > macro_tile_k or (macro_tile_k % tb_1 != 0):
                            continue
                        cb_1 = macro_tile_k // tb_1
                        if cb_1 > block_size or (block_size % cb_1 != 0):
                            continue
                        cb_3 = block_size // cb_1
                        if cb_3 > macro_tile_n:
                            continue

                        # gemm_m iter
                        for ta_3 in vector_load_size:
                            if ca_3 * ta_3 > macro_tile_m:
                                continue

                            if ta_1 != 1 and ca_3 * ta_3 != macro_tile_m: # constrain that thread length only have 2 copy dim
                                continue

                            if macro_tile_m % (ca_3 * ta_3) != 0:
                                continue
                            ta_2 = macro_tile_m // (ca_3 * ta_3)

                            # gemm_n iter
                            for tb_3 in vector_load_size:
                                if cb_3 * tb_3 > macro_tile_n:
                                    continue

                                if tb_1 != 1 and cb_3 * tb_3 != macro_tile_n: # constrain that thread length only have 2 copy dim
                                    continue

                                if macro_tile_n % (cb_3 * tb_3) != 0:
                                    continue
                                tb_2 = macro_tile_n // (cb_3 * tb_3)

                                # nxb, nxe
                                # for fwd, nxb is in gemm_n direction
                                nxb_list = [2 ** i for i in range(int(math.log2(macro_tile_n) + 1))]
                                if 2 in nxb_list:
                                    nxb_list.remove(2)      # remove item equal to 2
                                for nxe in (0, 1):
                                    for nxb in nxb_list:
                                        # remove constrain due to unmerge_sub_n
                                        unmerge_sub_n = macro_tile_n // nxb
                                        if unmerge_sub_n % (tb_2 * cb_2) != 0:      # unmerge_sub_n % nb_n0 == 0
                                            continue
                                        unmerge_sub_n1 = unmerge_sub_n // (tb_2 * cb_2)
                                        if (tb_3 * cb_3) % unmerge_sub_n1 != 0:
                                            continue                                # nb_n1b % unmerge_sub_n1 == 0
                                        item = ([ta_0, ta_1, ta_2, ta_3],
                                                [ca_0, ca_1, ca_2, ca_3],
                                                [tb_0, tb_1, tb_2, tb_3],
                                                [cb_0, cb_1, cb_2, cb_3],
                                                nxb, nxe)
                                        thread_cluster_list.append(item)
            return thread_cluster_list

        def gen_all_configs():
            tunable_dicts = []
            for gemm_m_per_block in gemm_m_per_block_list:
                for gemm_n_per_block in gemm_n_per_block_list:
                    xdlops_mapping_list = search_xdlops_mapping_from_m_n(gemm_m_per_block, gemm_n_per_block)
                    if len(xdlops_mapping_list) == 0:
                        continue
                    for xdlops_mapping in xdlops_mapping_list:
                        for gemm_k_per_block in gemm_k_per_block_list:
                            if gemm_k_per_block % xdlops_mapping.wave_tile_k != 0:
                                continue
                            if "lmk" in options:
                                print(f"options:{options}, {type(options)}")
                                if gemm_k_per_block // xdlops_mapping.wave_tile_k < options["lmk"]:
                                    continue
                            block_size = xdlops_mapping.waves * amdgpu_wave_size(self.mc.arch_config.arch)
                            thread_cluster_list = search_xdlops_thread_cluster_lengths(
                                                        config["current_direction"],
                                                        gemm_m_per_block,
                                                        gemm_n_per_block,
                                                        gemm_k_per_block,
                                                        block_size)
                            if len(thread_cluster_list) == 0:
                                continue
                            for thread_cluster in thread_cluster_list:
                                tensor_a_thread_lengths, tensor_a_cluster_lengths, \
                                        tensor_b_thread_lengths, tensor_b_cluster_lengths, nxb, nxe = thread_cluster
                                # populate the dict
                                tunable_dict = dict()
                                tunable_dict["arch"]                        =   'gfx908'
                                tunable_dict["gemm_m_per_block"]            =   gemm_m_per_block
                                tunable_dict["gemm_n_per_block"]            =   gemm_n_per_block
                                tunable_dict["gemm_k_per_block"]            =   gemm_k_per_block
                                tunable_dict["wave_tile_m"]                 =   xdlops_mapping.wave_tile_m
                                tunable_dict["wave_step_m"]                 =   xdlops_mapping.wave_step_m
                                tunable_dict["wave_repeat_m"]               =   xdlops_mapping.wave_repeat_m
                                tunable_dict["wave_tile_n"]                 =   xdlops_mapping.wave_tile_n
                                tunable_dict["wave_step_n"]                 =   xdlops_mapping.wave_step_n
                                tunable_dict["wave_repeat_n"]               =   xdlops_mapping.wave_repeat_n
                                tunable_dict["wave_tile_k"]                 =   xdlops_mapping.wave_tile_k
                                tunable_dict["tensor_a_thread_lengths"]     =   tensor_a_thread_lengths
                                tunable_dict["tensor_a_cluster_lengths"]    =   tensor_a_cluster_lengths
                                tunable_dict["tensor_b_thread_lengths"]     =   tensor_b_thread_lengths
                                tunable_dict["tensor_b_cluster_lengths"]    =   tensor_b_cluster_lengths
                                tunable_dict['direction']                   =   config["current_direction"]
                                tunable_dict['precision']                   =   config["precision"]
                                tunable_dict['nxb']                         =   nxb
                                tunable_dict['nxe']                         =   nxe

                                # post constrain, coalescing constrain
                                tentative_tunable = igemm_gtc_tunable_parameter_t(tunable_dict)
                                tentative_ctrl_coalescing_store_xdlops = ctrl_coalescing_store_xdlops_t()
                                tentative_ctrl_coalescing_store_xdlops.cxm = xdlops_mapping
                                tentative_ctrl_coalescing_store_xdlops.coalescing_groups = tentative_tunable.coalescing_store_groups
                                tentative_ctrl_coalescing_store_xdlops.data_byte = amdgpu_precision_data_byte(tentative_tunable.precision)
                                tentative_ctrl_coalescing_store_xdlops.vector_write_out = 1                      # TODO: some cases this can be set to other value
                                tentative_ctrl_coalescing_store_xdlops.block_size = tentative_tunable.block_size
                                if tentative_ctrl_coalescing_store_xdlops.get_length_m_max_groups() % \
                                        tentative_ctrl_coalescing_store_xdlops.coalescing_groups != 0:
                                    continue

                                tunable_dicts.append(tunable_dict)

            return tunable_dicts

        def serialize_all_configs(tunable_dicts):
            assert len(tunable_dicts) != 0
            # first, get config file name.
            arch_str = amdgpu_arch_to_string(self.mc.arch_config.arch)
            direction = tunable_dicts[0]['direction']
            config_file_base_name = f'igemm_{direction}_gtc_{arch_str}.config'
            config_file = os.path.join(os.path.dirname(self.mc.emitter.file_name), config_file_base_name)
            with open(config_file, "w") as fp:
                fp.write('[codegen]\n')
                fp.write('arch = {}\n'.format('\'' + arch_str + '\''))
                fp.write('code_object = {}\n'.format('\'' + amdgpu_codeobj_to_string(self.mc.arch_config.code_object) + '\''))
                fp.write('mode = \'flat\'\n')
                fp.write('\n')
                for td in tunable_dicts:
                    fp.write(igemm_gtc_tunable_parameter_t(td).serialize_as_section())
                    fp.write('\n')

        tunable_dicts = gen_all_configs()
        if len(tunable_dicts) == 0:
            print(f"no config generated")
            return
        print(f"total configs:{len(tunable_dicts)}")
        #for td in tunable_dicts:
        #    print(igemm_gtc_tunable_parameter_t(td).serialize())
        igemm_codegen_driver_t(self.mc, tunable_dicts)(emit_kernel_mp=True, compile_skip_disass=True)
        serialize_all_configs(tunable_dicts)

class igemm_sequence_driver_t(mc_base_t):
    def __init__(self, mc, config):
        mc_base_t.__init__(self, mc)
        self.config = config

    def __call__(self):
        if self.mc.arch_config.arch == 908:
            igemm_sequence_xdlops_t(self.mc, self.config)()
        else:
            assert False

def igemm_sequence(args, config_content):
    sec_root = config_content.get_section('codegen')[0]
    arch = amdgpu_arch_config_t({
        'arch'          :   amdgpu_string_to_arch( sec_root['arch'] ),
        'data_type'     :   AMDGPU_PRECISION_FP32,
        'code_object'   :   amdgpu_string_to_codeobj( sec_root['code_object']) })

    config_dicts = [sec.to_dict() for sec in config_content if sec.get_name().startswith('igemm_')]
    for config in config_dicts:
        config['arch'] = sec_root['arch']       # append arch to each section

    def sequece_one_direction(direction, config):
        arch_str = sec_root['arch']
        asm_file = f'igemm_{direction}_gtc_{arch_str}.s'
        asm_target = os.path.join(args.dir, asm_file)
        emitter = mc_emit_to_file_t(asm_target)
        mc = mc_asm_printer_t(emitter, arch)
        igemm_sequence_driver_t(mc, config)()

    for config in config_dicts:
        assert "direction" in config
        if type(config["direction"]) is list:
            for direction in config["direction"]:
                config["current_direction"] = direction             # give a flag for current target direction
                sequece_one_direction(direction, config)
        else:
            config["current_direction"] = config["direction"]       # give a flag for current target direction
            sequece_one_direction(config["direction"], config)
