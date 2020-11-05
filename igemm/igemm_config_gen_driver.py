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

import os
import copy

# micro-tile: 
# 128x128,256x128,256x64,128x256,64x256,128x64,64x128,256x32,32x256,64x64,128x32,32x128,256x16,16x256,128x16,16x128,64x32,32x64,32x32,64x16,16x64,32x16,16x32,64x8,8x64,16x16,64x4,4x64
ctrl_xdlops_mapping_fp32_config = [
        ctrl_xdlops_mapping_t( 256, 128,  64,  32,  4,  2,  2,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 128, 256,  32,  64,  4,  2,  2,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 256, 64 ,  64,  16,  4,  2,  2,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 256, 64 ,  64,  32,  4,  1,  1,  2,  1,  v_mfma_f32_32x32x1f32),  #add by jane
        ctrl_xdlops_mapping_t( 64 , 256,  16,  64,  4,  2,  2,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 64 , 256,  32,  64,  4,  1,  1,  1,  2,  v_mfma_f32_32x32x1f32), #add by jane

        ctrl_xdlops_mapping_t( 256, 32 ,  64,  4 ,  4,  2,  2,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 256,  4 ,  64,  4,  2,  2,  2,  1,  v_mfma_f32_4x4x1f32),
        #ctrl_xdlops_mapping_t( 32 , 256,  16 ,  64,  4,  1,  1,  1,  2,  v_mfma_f32_16x16x1f32),   #add by jane can not because coleasing group assert

        ctrl_xdlops_mapping_t( 256, 16 ,  64,  4 ,  4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 256,  4 ,  64,  4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),

        ctrl_xdlops_mapping_t( 128, 128,  32,  32,  4,  2,  2,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 128, 128,  32,  64,  4,  1,  1,  2,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 128, 64 ,  32,  8 ,  4,  2,  2,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 128, 64 ,  64,  32 ,  4,  1,  1,  1,  1,  v_mfma_f32_32x32x1f32),   #add by jane

        ctrl_xdlops_mapping_t( 64 , 128,  8 ,  32,  4,  2,  2,  2,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 128,  32,  64,  4,  1,  1,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 64 , 128,  64,  32,  4,  1,  1,  1,  1,  v_mfma_f32_32x32x1f32),
        ctrl_xdlops_mapping_t( 128, 32 ,  32,  8 ,  4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 128, 32 ,  64,  16 ,  4,  1,  1,  1,  1,  v_mfma_f32_16x16x1f32), #add by jane

        ctrl_xdlops_mapping_t( 32 , 128,  8 ,  32,  4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),
        #ctrl_xdlops_mapping_t( 32 , 128,  16,  64,  4,  1,  1,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 64 , 64 ,  16,  16,  4,  2,  2,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 64 ,  32,  32,  4,  1,  1,  1,  1,  v_mfma_f32_16x16x1f32), #add by jane

        ctrl_xdlops_mapping_t( 128, 16 ,  64,  16,  2,  1,  1,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 16 , 128,  16,  64,  2,  1,  1,  1,  1,  v_mfma_f32_16x16x1f32),
        ctrl_xdlops_mapping_t( 64 , 32 ,  32,  8 ,  4,  1,  1,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 32 ,  32,  32 ,  2,  1,  1,  1,  1,  v_mfma_f32_16x16x1f32),  #add by jane;
        ctrl_xdlops_mapping_t( 32 , 64 ,  8 ,  32,  4,  1,  1,  2,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 64 ,  32 ,  32,  2,  1,  1,  1,  1,  v_mfma_f32_16x16x1f32),   #add by jane;

        ctrl_xdlops_mapping_t( 32 , 32 ,  16,  16,  4,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),

        ctrl_xdlops_mapping_t( 64 , 16 ,  64,  4 ,  4,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 64 ,  4 ,  64,  4,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 16 ,  64,  4 ,  2,  1,  1,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 64 ,  4 ,  64,  2,  1,  1,  2,  1,  v_mfma_f32_4x4x1f32),

        # 2waves, block_size=128
        ctrl_xdlops_mapping_t( 64 , 8  ,  64,  4 ,  2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 8  , 64 ,  4 ,  64,  2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 32 , 16 ,  32,  8 ,  2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 32 ,  8 ,  32,  2,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        # 1 wave
        ctrl_xdlops_mapping_t( 32 , 16 ,  32,  8 ,  1,  1,  1,  1,  2,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 32 ,  8 ,  32,  1,  1,  1,  2,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 64 , 4 ,  64,  4 ,  1,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 4  , 64,  4 ,  64,  1,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32),
        ctrl_xdlops_mapping_t( 16 , 16,  16,  16,  1,  1,  1,  1,  1,  v_mfma_f32_4x4x1f32)
        ]

class igemm_config_gen_driver_t():
    def __init__(self, emitter, config_content):
        self.emitter = emitter
        self.emitter.open()
        self.config_content = config_content
        
    def non1numbers(self, length_list):
        n = 0
        for i in range(len(length_list)):
            if length_list[i] > 1:
                n += 1
        return n
    def __del__(self):
        self.emitter.close()

    def emit(self, s):
        self.emitter.emit(s)

    def emit_one_valid_config(self):
        self.emit(f"[igemm_bwd_gtc]")
        self.emit(f"{'gemm_m_per_block':25}= {self.gemm_m_per_block}")
        self.emit(f"{'gemm_n_per_block':25}= {self.gemm_n_per_block}")
        self.emit(f"{'gemm_k_per_block':25}= {self.gemm_k_per_block}")
        self.emit(f"{'wave_tile_m':25}= {self.wave_tile_m}")
        self.emit(f"{'wave_step_m':25}= {self.wave_step_m}")
        self.emit(f"{'wave_repeat_m':25}= {self.wave_repeat_m}")
        self.emit(f"{'wave_tile_n':25}= {self.wave_tile_n}")
        self.emit(f"{'wave_step_n':25}= {self.wave_step_n}")
        self.emit(f"{'wave_repeat_n':25}= {self.wave_repeat_n}")
        self.emit(f"{'tensor_a_thread_lengths':25}= [{self.t_k0}, {self.t_k1e}, {self.t_c0},{self.t_c1}]")
        self.emit(f"{'tensor_a_cluster_lengths':25}= [{self.c_k0}, {self.c_k1e}, {self.c_c0},{self.c_c1}]")
        self.emit(f"{'tensor_b_thread_lengths':25}= [{self.t_k0}, {self.t_k1e}, {self.t_n0},{self.t_n1b}]")
        self.emit(f"{'tensor_b_cluster_lengths':25}= [{self.c_k0}, {self.c_k1e}, {self.c_n0},{self.c_n1b}]")
        self.emit(f"{'direction':25}= '{self.direction}'")
        self.emit(f"{'precision':25}= '{self.precision}'")
        self.emit(f"{'nxb':25}= {self.nxb}")
        self.emit(f"{'nxe':25}= {self.nxe}")
        self.emit('')

    def get_specific_ctrl_xdlops_mapping_t(self, macro_tile_m, macro_tile_n):
        target_mfma_tiling = list()
        for t in ctrl_xdlops_mapping_fp32_config:
            if t.macro_tile_m == macro_tile_m and t.macro_tile_n == macro_tile_n:
                target_mfma_tiling.append(t)
        return target_mfma_tiling

    def __call__(self):
        sec_root = self.config_content.get_section('codegen')[0]

        self.emit("[codegen]")
        self.emit(f"arch = '{sec_root['arch']}'")
        self.emit(f"code_object = '{sec_root['code_object']}'")
        self.emit("mode = 'flat'")
        self.emit('')

        gen_config_lit = [sec.to_dict() for sec in self.config_content if sec.get_name().startswith('codegen_config')]
        for gen_conf in gen_config_lit:
            if sec_root['direction'] == 'bwd':
                self.emit_bwd_configs(gen_conf)



    def emit_bwd_configs(self, gen_conf):
        self.direction = 'bwd'
        self.precision = gen_conf['precision']
        if 'micro_tile' in gen_conf:
            micro_tile_array = gen_conf['micro_tile'].split(',')
            potential_micro_tile_list = [i for i in micro_tile_array]
        else:
            potential_micro_tile_list = ['128x128','256x128','256x64','128x256','64x256','128x64','64x128','256x32','32x256','64x64','128x32','32x128','256x16','16x256','128x16','16x128','64x32','32x64','32x32','64x16','16x64','32x16','16x32','64x8','8x64','16x16','64x4','4x64']

        if 'nxb' in gen_conf:
            nxb_array = gen_conf['nxb'].split(',')
            potential_nxb_list = [int(i) for i in nxb_array]
        else:
            potential_nxb_list = [128,64,32,16,8,4,1]

        if 'nxe' in gen_conf:
            nxe_array = gen_conf['nxe'].split(',')
            potential_nxe_list = [int(i) for i in nxe_array]
        else:
            potential_nxe_list = [0,1]

        if 'gemm_k' in gen_conf:
            gemm_k_array = gen_conf['gemm_k'].split(',')
            potential_k_list = [int(i) for i in gemm_k_array]
        else:
            potential_k_list = [16,8,4]
        
        if 'micro_tile_with_gemm_k_4' in gen_conf:
            micro_tile_with_gemm_k_4_array = gen_conf['micro_tile_with_gemm_k_4'].split(',')
            micro_tile_with_gemm_k_4_list = [i for i in micro_tile_with_gemm_k_4_array]
        else:
            micro_tile_with_gemm_k_4_list = ['32x16','16x32']

        for item in potential_micro_tile_list:
            tile = item.split('x')
            self.gemm_m_per_block = int(tile[0])
            self.gemm_n_per_block = int(tile[1])
            target_xdlops_t_list = self.get_specific_ctrl_xdlops_mapping_t(self.gemm_m_per_block, self.gemm_n_per_block)
            for t in target_xdlops_t_list:
                self.emit(f"### {t.macro_tile_m}x{t.macro_tile_n}")
                self.gemm_m_per_block                   = t.macro_tile_m
                self.gemm_n_per_block                   = t.macro_tile_n
                self.wave_tile_m                    = t.wave_tile_m
                self.wave_step_m                    = t.wave_step_m
                self.wave_repeat_m                  = t.wave_repeat_m
                self.wave_tile_n                    = t.wave_tile_n
                self.wave_step_n                    = t.wave_step_n
                self.wave_repeat_n                  = t.wave_repeat_n
                waves_per_m = self.gemm_m_per_block // (self.wave_tile_m * self.wave_step_m * self.wave_repeat_m)
                waves_per_n = self.gemm_n_per_block // (self.wave_tile_n * self.wave_step_n * self.wave_repeat_n)
                self.block_size                     = waves_per_m * waves_per_n * AMDGPU_WAVE_SIZE

                for i_k in potential_k_list:
                    self.gemm_k_per_block = i_k
                    if i_k == 4:
                        cur_tile_str = f"{self.gemm_m_per_block}x{self.gemm_n_per_block}"
                        if cur_tile_str not in micro_tile_with_gemm_k_4_list:
                            break


                    b_data_per_thread = (self.gemm_n_per_block*self.gemm_k_per_block)//self.block_size
                    a_data_per_thread = (self.gemm_m_per_block*self.gemm_k_per_block)//self.block_size

                    # when nxe=0, nxb=[1,4,8,16,32,64,128], t_n1b<nxb, t_n1b=[1,2,4], t_c1=1 to avoid address caculation
                    # tensor_a_thread_lengths  = [c?,  1,  f?, 1]      # K0xK1ExC0xC1
                    # tensor_a_cluster_lengths = [1,  a?,  1, d?]      # K0xK1ExC0xC1
                    # tensor_b_thread_lengths  = [c?,  1,  b?,  y]      # K0xK1ExN0xN1B
                    # tensor_b_cluster_lengths = [1,  a?,  1, x]      # K0xK1ExN0xN1B
                    for i_nxb in potential_nxb_list:
                        if i_nxb > self.gemm_n_per_block:
                            continue
                        for i_nxe in potential_nxe_list:
                            if i_nxe == 0:
                                potential_t_n1b_list = [4,2,1]
                            elif i_nxe == 1:
                                potential_t_n1b_list = [1]

                            for i_t_n1b in potential_t_n1b_list:
                                self.t_n1b = i_t_n1b
                                self.nxb = i_nxb
                                self.nxe = i_nxe
                                self.t_k1e = 1
                                self.t_c1 = 1
                                self.c_k0 = 1
                                self.c_n0 = 1
                                self.c_c0 = 1
                                self.c_n1b = self.gemm_n_per_block*2
                                while self.c_n1b>1:
                                    self.c_n1b = self.c_n1b//2
                                    if self.c_n1b*self.t_n1b > self.gemm_n_per_block:
                                        continue
                                    self.c_k1e = self.block_size//self.c_n1b  #a?
                                    if self.c_k1e > self.gemm_k_per_block:
                                        continue
                                    self.t_n0 = self.gemm_n_per_block//(self.c_n1b*self.t_n1b)  #b?
                                    self.t_k0 = self.gemm_k_per_block//self.c_k1e  #c?
                                    if self.t_k0 != (b_data_per_thread//(self.t_n0*self.t_n1b)):  
                                        continue
                                    self.c_c1 = self.c_n1b  #d?
                                    if i_nxe == 0:
                                        potential_t_c1_list = [4,2,1]
                                    elif i_nxe == 1:
                                        potential_t_c1_list = [1]
                                    #assert unmerge_sub_n % n_n0 == 0, f"unmerge_sub_n:{unmerge_sub_n}, n_n0:{n_n0}"
                                    if (self.gemm_n_per_block//self.nxb % (self.t_n0*self.c_n0) != 0):
                                        continue
                                    if self.non1numbers([self.t_k0, self.t_n0, self.t_n1b]) > 2:  #check [t_k0, t_k1e, t_c0] 
                                        continue
                                    for i_t_c1 in potential_t_c1_list: #e?
                                        self.t_c1 = i_t_c1
                                        self.t_c0 = self.gemm_m_per_block//(self.c_c1*self.t_c1)  #f?
                                        if self.t_c0 == 0:
                                            continue
                                        if self.t_k0*self.t_c0*self.t_c1 !=a_data_per_thread:
                                            continue
                                        if self.non1numbers([self.t_k0, self.t_c0, self.t_c1]) > 2:  #check [t_k0, t_k1e, t_c0] 
                                            continue
                                        self.emit_one_valid_config()

