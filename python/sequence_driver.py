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
from .codegen_driver import codegen_driver_t
from .host_driver import host_driver

import os
import copy
import math


def sequence_get_config_file_name(direction, arch_str, out_dir):
    return os.path.join(out_dir, f'igemm_{direction}_gtc_{arch_str}.config')

def sequence_serialize_all_configs(arch_str, code_object, config_file, tunable_dicts):
    assert len(tunable_dicts) != 0
    with open(config_file, "w") as fp:
        fp.write('[codegen]\n')
        fp.write('arch = {}\n'.format('\'' + arch_str + '\''))
        fp.write('code_object = {}\n'.format('\'' +code_object + '\''))
        fp.write('mode = \'flat\'\n')
        fp.write('\n')
        for cnt, td in enumerate(tunable_dicts):
            fp.write('# kernel:{}\n'.format(cnt))
            fp.write(igemm_gtc_tunable_parameter_t(td).serialize_as_section())
            fp.write('\n')

def sequence_is_cgt(t0, t1, t2, t3, c0, c1, c2, c3):
    '''
    check if cluster length is >= thread length
    '''
    return (c0 * c1) >= (t0 * t1) and (c2 * c3) >= (t2 * t3)


def sequence_is_tunable_resource_valid(direction, mc, tunable):
    if direction == 'fwd':
        igemm = igemm_fwd_gtc_t(mc, tunable)
    elif direction == 'bwd':
        igemm = igemm_bwd_gtc_t(mc, tunable)
    elif direction == 'wrw':
        igemm = igemm_wrw_gtc_t(mc, tunable)

    if igemm.sgpr.s_end.value > amdgpu_sgpr_limit(mc.arch_config.arch):
        return False

    return True

class sequence_xdlops_t(mc_base_t):
    def __init__(self, mc, config):
        mc_base_t.__init__(self, mc)
        self.config = config

    def __call__(self):
        '''
        return all tunables
        '''
        config = self.config
        gemm_m_per_block_list = config["gemm_m_per_block"] if type(config["gemm_m_per_block"]) is list else config["gemm_m_per_block"]
        gemm_n_per_block_list = config["gemm_n_per_block"] if type(config["gemm_n_per_block"]) is list else config["gemm_n_per_block"]
        gemm_k_per_block_list = config["gemm_k_per_block"] if type(config["gemm_k_per_block"]) is list else config["gemm_k_per_block"]
        options = config["options"] if "options" in config else dict()
        assert type(options) is dict, f"fail to get options:{options}, type:{type(options)}"
        # print("options:{},{}, lmk:{}".format("options" in config, options, options["lmk"] if "lmk" in options else 0))

        def search_xdlops_mapping_from_m_n(macro_tile_m, macro_tile_n):
            valid_mapping_list = []
            for ctrl in ctrl_xdlops_mapping_fp32:
                if ctrl.macro_tile_m == macro_tile_m and \
                    ctrl.macro_tile_n == macro_tile_n:
                    valid_mapping_list.append(ctrl)
            # assert len(valid_mapping_list) != 0, f"no macro_tile hit for {macro_tile_m}x{macro_tile_n}"
            return valid_mapping_list

        def search_xdlops_sub_configs(direction, macro_tile_m, macro_tile_n, macro_tile_k, block_size):
            # to generate a combination of ta[4], ca[4], tb[4], cb[4], nxb, nxe, gemm_k_global_split
            #
            # ta_0, ta_1, ta_2, ta_3
            # ca_0, ca_1, ca_2, ca_3
            # tb_0, tb_1, tb_2, tb_3
            # cb_0, cb_1, cb_2, cb_3
            #
            sub_configs = []
            if direction == 'fwd':
                ta_0 = ca_0 = tb_0 = cb_0 = 1
                ca_2 = cb_2 = 1
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

                    # tb_1 is input c1e, stride is in_stride_c. This is indeed not vector load
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

                                if "cgt" in options and options["cgt"] == 1:
                                    if not sequence_is_cgt(ta_0, ta_1, ta_2, ta_3, ca_0, ca_1, ca_2, ca_3):
                                        continue
                                    if not sequence_is_cgt(tb_0, tb_1, tb_2, tb_3, cb_0, cb_1, cb_2, cb_3):
                                        continue

                                # nxb, nxe
                                # for fwd, nxb is in gemm_n direction
                                nxb_list = [2 ** i for i in range(int(math.log2(macro_tile_n) + 1))]
                                if 2 in nxb_list:
                                    nxb_list.remove(2)      # remove item equal to 2
                                for nxe in (0, 1):
                                    if "exv" in options and options["exv"] == 1:
                                        if nxe == 1 and tb_3 > 1:
                                            continue
                                    else:
                                        pass
                                    for nxb in nxb_list:
                                        if nxe == 0 and "bev" in options and options["bev"] == 1:
                                            if tb_3 > nxb or nxb % tb_3 != 0:
                                                continue
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
                                                nxb, nxe, 0)
                                        sub_configs.append(item)
            elif direction == 'bwd':
                    # bwd, for simplicity, have following rules:
                    # 1) ta_0 = tb_0, ta_1 = tb_1, ca_0 = cb_0, ca_1 = cb_1
                    #   in other words, gemm_k thread/cluster distribution is the same in a/b matrix
                    # 2) only have >1 value in thread_lengths K0, cluster length K1E (no vector load)
                    # 3) cluster length C0 always 1, cluster length N0 always 1
                    # 4) for weight, thread_length c0, c1 can't >1 at the same time
                    # 5) when nxe!=0, thread_length n1b can't >1  (fwd can, and can work. but maybe useless?)
                    #
                    # a, weight, K0xK1ExC0xC1
                    # b, output, K0xK1ExN0xN1B

                    ta_1 = tb_1 = ca_0 = cb_0 = 1
                    ca_2 = cb_2 = 1
                    vector_load_size = [1, 2, 4]

                    # gemm_k iter
                    #for ta_0 in [2 ** i for i in range(int(math.log2(macro_tile_k) + 1))]:
                    for ta_0 in vector_load_size:
                        if ta_0 > macro_tile_k or (macro_tile_k % ta_0 != 0):
                            continue
                        ca_1 = macro_tile_k // ta_0

                        tb_0 = ta_0
                        cb_1 = ca_1

                        # try find ca_3, cb_3
                        if ca_1 > block_size or (block_size % ca_1 != 0):
                            continue
                        ca_3 = block_size // ca_1
                        cb_3 = ca_3

                        # check remaning element of gemm_m, genn_n
                        if ca_3 > macro_tile_m or (macro_tile_m % ca_3 != 0):
                            continue
                        rem_gemm_m = macro_tile_m // ca_3
                        if cb_3 > macro_tile_n or (macro_tile_n % cb_3 != 0):
                            continue
                        rem_gemm_n = macro_tile_n // cb_3

                        for ta_3 in [2 ** i for i in range(int(math.log2(rem_gemm_m) + 1))]:
                            if ta_3 > vector_load_size[-1]:     # LDS share store need this as vector_d1
                                continue
                            ta_2 = rem_gemm_m // ta_3
                            if ta_3 > 1 and ta_2 > 1 and ta_0 > 1:
                                # only support 2 copy dimension
                                continue 
                            for tb_3 in [2 ** i for i in range(int(math.log2(rem_gemm_n) + 1))]:
                                if tb_3 > vector_load_size[-1]:      # LDS share store need this as vector_d1
                                    continue
                                tb_2 = rem_gemm_n // tb_3
                                if tb_3 > 1 and tb_2 > 1 and tb_0 > 1:
                                    # only support 2 copy dimension
                                    continue 

                                if "cgt" in options and options["cgt"] == 1:
                                    if not sequence_is_cgt(ta_0, ta_1, ta_2, ta_3, ca_0, ca_1, ca_2, ca_3):
                                        continue
                                    if not sequence_is_cgt(tb_0, tb_1, tb_2, tb_3, cb_0, cb_1, cb_2, cb_3):
                                        continue

                                # nxb, nxe
                                # for bwd, nxb is in gemm_n direction
                                nxb_list = [2 ** i for i in range(int(math.log2(macro_tile_n) + 1))]
                                if 2 in nxb_list:
                                    nxb_list.remove(2)      # remove item equal to 2
                                for nxe in (0, 1):
                                    if "exv" in options and options["exv"] == 1:
                                        if nxe == 1 and tb_3 > 1:
                                            continue
                                    else:
                                        assert False, "currently in bwd, exv must be 1"
                                    for nxb in nxb_list:
                                        if nxe == 0 and "bev" in options and options["bev"] == 1:
                                            if tb_3 > nxb or nxb % tb_3 != 0:
                                                continue
                                        # remove constrain due to unmerge_sub_n
                                        unmerge_sub_n = macro_tile_n // nxb
                                        # gemm_n_unmerge_cluster=0 case. TODO, support gemm_n_unmerge_cluster=1
                                        if unmerge_sub_n % (tb_2 * cb_2) != 0:      # unmerge_sub_n % nb_n0 == 0
                                            continue
                                        unmerge_sub_n1 = unmerge_sub_n // (tb_2 * cb_2)
                                        if (tb_3 * cb_3) % unmerge_sub_n1 != 0:
                                            continue                                # nb_n1b % unmerge_sub_n1 == 0
                                        item = ([ta_0, ta_1, ta_2, ta_3],
                                                [ca_0, ca_1, ca_2, ca_3],
                                                [tb_0, tb_1, tb_2, tb_3],
                                                [cb_0, cb_1, cb_2, cb_3],
                                                nxb, nxe, 0)
                                        sub_configs.append(item)
            elif direction == 'wrw':
                    #
                    # a, output, N0xN1BxK0xK1
                    # b, input,  N0xN1BxC0xC1E
                    # 1) ta_0 = tb_0, ta_1 = tb_1, ca_0 = cb_0, ca_1 = cb_1
                    #   in other words, gemm_k thread/cluster distribution is the same in a/b matrix
                    # 2) ta_0 = tb_0 = 1, ca_0 = cb_0 = 1. in gemm_k, wrw only have value in n1b 
                    # 3) ca_0, cb_0, ca_2, cb_2 always 1
                    # 4) thread length n1b is either 1 or vector size. and be affected by exv. indeed always need exv
                    # 5) tb_3 always be 1

                    ta_0 = tb_0 = ca_0 = cb_0 = 1
                    ca_2 = cb_2 = 1
                    vector_load_size = [1, 2, 4]

                    # gemm_k iter
                    for ta_1 in vector_load_size:
                        if ta_1 > macro_tile_k or (macro_tile_k % ta_1 != 0):
                            continue
                        ca_1 = macro_tile_k // ta_1

                        tb_1 = ta_1
                        cb_1 = ca_1

                        # try find ca_3, cb_3
                        if ca_1 > block_size or (block_size % ca_1 != 0):
                            continue
                        ca_3 = block_size // ca_1
                        cb_3 = ca_3

                        # check remaning element of gemm_m, genn_n
                        if ca_3 > macro_tile_m or (macro_tile_m % ca_3 != 0):
                            continue
                        rem_gemm_m = macro_tile_m // ca_3
                        if cb_3 > macro_tile_n or (macro_tile_n % cb_3 != 0):
                            continue
                        rem_gemm_n = macro_tile_n // cb_3

                        for ta_3 in [2 ** i for i in range(int(math.log2(rem_gemm_m) + 1))]:
                            if ta_3 > vector_load_size[-1]:     # LDS share store need this as vector_d1
                                continue
                            ta_2 = rem_gemm_m // ta_3
                            if ta_3 > 1 and ta_2 > 1 and ta_1 > 1:
                                # only support 2 copy dimension
                                continue 
                            for tb_3 in [2 ** i for i in range(int(math.log2(rem_gemm_n) + 1))]:
                                if tb_3 != 1:
                                    continue
                                if tb_3 > vector_load_size[-1]:      # LDS share store need this as vector_d1
                                    continue

                                tb_2 = rem_gemm_n // tb_3
                                if tb_3 > 1 and tb_2 > 1 and tb_1 > 1:
                                    # only support 2 copy dimension
                                    continue 

                                if "cgt" in options and options["cgt"] == 1:
                                    if not sequence_is_cgt(ta_0, ta_1, ta_2, ta_3, ca_0, ca_1, ca_2, ca_3):
                                        continue
                                    if not sequence_is_cgt(tb_0, tb_1, tb_2, tb_3, cb_0, cb_1, cb_2, cb_3):
                                        continue

                                # nxb, nxe
                                # for bwd, nxb is in gemm_n direction
                                nxb_list = [2 ** i for i in range(int(math.log2(macro_tile_k) + 1))]
                                #nxb_list = [1, 4]
                                if 2 in nxb_list:
                                    nxb_list.remove(2)      # remove item equal to 2
                                for nxe in (0, 1):
                                    if "exv" in options and options["exv"] == 1:
                                        if nxe == 1 and (ta_1 > 1 or tb_1 > 1):
                                            continue
                                    else:
                                        assert False, "currently in bwd, exv must be 1"
                                    for nxb in nxb_list:
                                        if nxe == 0 and "bev" in options and options["bev"] == 1:
                                            if ta_1 > nxb or nxb % ta_1 != 0:
                                                continue
                                        # remove constrain due to unmerge_sub_n
                                        unmerge_sub_n = macro_tile_k // nxb
                                        # gemm_k_unmerge_cluster=0 case. TODO, support gemm_k_unmerge_cluster=1
                                        if unmerge_sub_n % (ta_0 * ca_0) != 0:      # unmerge_sub_n % n_n0 == 0
                                            continue
                                        unmerge_sub_n1 = unmerge_sub_n // (ta_0 * ca_0)
                                        if (ta_1 * ca_1) % unmerge_sub_n1 != 0:
                                            continue                                # nb_n1b % unmerge_sub_n1 == 0
                                        for gemm_k_global_split in (0, 1):
                                            item = ([ta_0, ta_1, ta_2, ta_3],
                                                    [ca_0, ca_1, ca_2, ca_3],
                                                    [tb_0, tb_1, tb_2, tb_3],
                                                    [cb_0, cb_1, cb_2, cb_3],
                                                    nxb, nxe, gemm_k_global_split)
                                            sub_configs.append(item)

            else:
                assert False, f'unsupported direction:{direction}'

            return sub_configs

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
                                if gemm_k_per_block // xdlops_mapping.wave_tile_k < options["lmk"]:
                                    continue
                            block_size = xdlops_mapping.waves * amdgpu_wave_size(self.mc.arch_config.arch)
                            sub_configs = search_xdlops_sub_configs(
                                                        config["current_direction"],
                                                        gemm_m_per_block,
                                                        gemm_n_per_block,
                                                        gemm_k_per_block,
                                                        block_size)
                            if len(sub_configs) == 0:
                                continue
                            for sub_config in sub_configs:
                                tensor_a_thread_lengths, tensor_a_cluster_lengths, \
                                        tensor_b_thread_lengths, tensor_b_cluster_lengths, nxb, nxe, gemm_k_global_split = sub_config
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
                                if config["current_direction"] == 'wrw':
                                    tunable_dict['gemm_k_global_split']     =   gemm_k_global_split

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

                                if not sequence_is_tunable_resource_valid(config["current_direction"], self.mc, tentative_tunable):
                                    continue

                                tunable_dicts.append(tunable_dict)

            return tunable_dicts

        tunable_dicts = gen_all_configs()
        if len(tunable_dicts) == 0:
            print(f"no config generated")
            return None
        print(f"[{config['current_direction']}] total configs:{len(tunable_dicts)}")
        #for td in tunable_dicts:
        #    print(igemm_gtc_tunable_parameter_t(td).serialize())
        codegen_driver_t(self.mc, tunable_dicts)(emit_kernel_mp=True, compile_skip_disass=True)
        #serialize_all_configs(tunable_dicts)
        return tunable_dicts

class sequence_driver_t(mc_base_t):
    def __init__(self, mc, config):
        mc_base_t.__init__(self, mc)
        self.config = config

    def __call__(self, **options):
        def get_dict_with_default(some_dict, key, default_value):
            if key in some_dict:
                return some_dict[key]
            return default_value
        tunable_dicts = None
        out_dir = get_dict_with_default(options, 'out_dir', 'out')
        arch = get_dict_with_default(options, 'arch', 'gfx908')
        code_object = get_dict_with_default(options, 'code_object', 'cov3')

        if self.mc.arch_config.arch == 908:
            tunable_dicts = sequence_xdlops_t(self.mc, self.config)()
        else:
            assert False
        
        if tunable_dicts == None:
            return

        sequence_serialize_all_configs(arch, code_object,
                                sequence_get_config_file_name(tunable_dicts[0]['direction'],
                                        arch,
                                        out_dir),
                                tunable_dicts)


def igemm_sequence_driver(**options):
    def get_dict_with_default(some_dict, key, default_value):
        if key in some_dict:
            return some_dict[key]
        return default_value
    arch = get_dict_with_default(options, 'arch', 'gfx908')
    code_object = get_dict_with_default(options, 'code_object', 'cov3')
    config_content = get_dict_with_default(options, 'config_content', None)
    out_dir = get_dict_with_default(options, 'out_dir', 'out')

    arch_config = amdgpu_arch_config_t({
        'arch'          :   amdgpu_string_to_arch( arch ),
        'data_type'     :   AMDGPU_PRECISION_FP32,
        'code_object'   :   amdgpu_string_to_codeobj( code_object) })

    config_dicts = [sec.to_dict() for sec in config_content if sec.get_name().startswith('igemm_')]
    for config in config_dicts:
        config['arch'] = arch       # append arch to each section

    def sequece_one_direction(direction, config, **options):
        asm_file = f'igemm_{direction}_gtc_{arch}.s'
        asm_target = os.path.join(out_dir, asm_file)
        emitter = mc_emit_to_file_t(asm_target)
        mc = mc_asm_printer_t(emitter, arch_config)
        mc_set_current(mc)
        igemm_sequence_driver_t(mc, config)(**options)

    for config in config_dicts:
        assert "direction" in config
        if type(config["direction"]) is list:
            for direction in config["direction"]:
                config["current_direction"] = direction             # give a flag for current target direction
                sequece_one_direction(direction, config, **options)
        else:
            config["current_direction"] = config["direction"]       # give a flag for current target direction
            sequece_one_direction(config["direction"], config, **options)

    # build host
    direction = config_dicts[0]["direction"][0] if type(config_dicts[0]["direction"]) is list else config_dicts[0]["direction"]
    config_file = sequence_get_config_file_name(direction, arch, out_dir)
    host_driver(arch=arch, config_file=config_file, out_dir=out_dir)
