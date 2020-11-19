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
from ..algo import *
from ..codegen import *

def naive_get_spatial_dimension(tensor_layout):
    assert type(tensor_layout) is str
    # TODO: expect string like nchw, ncdhw
    return len(tensor_layout) - 2

class naive_fma_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        self.ctrl = ctrl
    def __call__(self, v_c, v_a, v_b, v_tmp2):
        with self._deferred_context():
            if self.ctrl.precision == 'fp32':
                self._emit(f"v_mac_f32 v[{v_c}], v[{v_a}], v[{v_b}]")
            elif self.ctrl.precision == 'fp16':
                # here we can use pack-math instruction like v_fma_mix_f32, v_mad_mix_f32
                # but that we lead to less arch compatibility. e.g., gfx900, gfx803 not support above
                # hence we prefer a more *naive* way
                self._emit(f"v_cvt_f32_f16 v[{v_tmp2}], v[{v_a}]")
                self._emit(f"v_cvt_f32_f16 v[{v_tmp2}+1], v[{v_b}]")
                self._emit(f"v_mac_f32 v[{v_c}], v[{v_tmp2}], v[{v_tmp2}+1]")
            elif self.ctrl.precision == 'bf16':
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}], 16, v[{v_a}]")
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}+1], 16, v[{v_b}]")
                self._emit(f"v_mac_f32 v[{v_c}], v[{v_tmp2}], v[{v_tmp2}+1]")
            else:
                assert False
        return self._get_deferred()

class naive_float_to_bfloat16_t(mc_base_t):
    '''
    __device__ __host__ ushort __float_to_bfloat16(float src_val)
    {
        _cvt_bf16_fp32_t target_val;
        target_val.f32 = src_val;

        if((~target_val.u32 & 0x7f800000) == 0) // Inf or NaN
        {
            if((target_val.u32 & 0xffff) != 0)
            {
                target_val.u32 |= 0x10000; // Preserve signaling NaN
            }
        }
        else
        {
    #ifdef MIOPEN_USE_RNE_BFLOAT16
            target_val.u32 += (0x7fff + (target_val.ushortvec[1] & 1));
    #endif // MIOPEN_USE_RNE_BFLOAT16
        }
        return target_val.ushortvec[1];
    }
    '''
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        self.ctrl = ctrl
    def __call__(self, v_dst, v_src, v_10000, s_nan_inf, s_7fff, s_tmp4, v_tmp2):
        with self._deferred_context():
            self._emit(f"v_not_b32 v[{v_tmp2}+1], v[{v_src}]")
            self._emit(f"v_and_b32 v[{v_tmp2}], s[{s_nan_inf}], v[{v_tmp2}+1]")
            self._emit(f"v_cmp_eq_u32 s[{s_tmp4}:{s_tmp4}+1], 0, v[{v_tmp2}]")
            self._emit(f"v_and_b32 v[{v_tmp2}], 0xffff, v[{v_src}]")
            self._emit(f"v_cmp_ne_u32 s[{s_tmp4}+2:{s_tmp4}+3], 0, v[{v_tmp2}]")
            self._emit(f"s_and_b64 s[{s_tmp4}+2:{s_tmp4}+3], s[{s_tmp4}:{s_tmp4}+1], s[{s_tmp4}+2:{s_tmp4}+3]")
            self._emit(f"v_cndmask_b32 v[{v_tmp2}], 0, v[{v_10000}], s[{s_tmp4}+2:{s_tmp4}+3]")
            self._emit(f"v_or_b32 v[{v_dst}], v[{v_src}], v[{v_tmp2}]")
            self._emit_front(f".if MIOPEN_USE_RNE_BFLOAT16 == 1")
            self._emit(f"v_bfe_u32 v[{v_tmp2}], v[{v_src}], 16, 1")
            self._emit(f"v_add_u32 v[{v_tmp2}+1], s[{s_7fff}], v[{v_tmp2}]")
            self._emit(f"v_cndmask_b32 v[{v_tmp2}], v[{v_tmp2}+1], 0, s[{s_tmp4}:{s_tmp4}+1]")
            self._emit(f"v_add_u32 v[{v_dst}], v[{v_tmp2}], v[{v_src}]")
            self._emit_front(f".endif")
            self._emit(f"v_lshrrev_b32 v[{v_dst}], 16, v[{v_dst}]")
        return self._get_deferred()

class naive_set_flag_hw(macro_base_t):
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)

    def __call__(self, v_flag, v_ih, v_iw, s_h, s_w):
        with self._deferred_context():
            self._emit(f"v_cmp_gt_u32 vcc, s[{s_h}], v[{v_ih}]")
            self._emit(f"v_cndmask_b32 v[{v_flag}], 0, 1, vcc")
            self._emit(f"v_cmp_gt_u32 vcc, s[{s_w}], v[{v_iw}]")
            self._emit(f"v_cndmask_b32 v[{v_flag}], 0, v[{v_flag}], vcc")
        return self._get_deferred()

class naive_conv_ctrl_t(object):
    def __init__(self, direction, tensor_layout, precision, block_size = 256):
        self.direction = direction
        self.tensor_layout = tensor_layout
        self.precision = precision
        self.block_size = block_size

class naive_conv_fwd_t(mc_base_t):
    '''
    // prototype is as below:
    extern "C" __global__ void naive_conv_fwd_nchw_fp32(const float* __restrict__ p_in,
                                                        const float* __restrict__ p_wei,
                                                        float* __restrict__ p_out,
                                                        int hi,
                                                        int wi,
                                                        int n,
                                                        int k_per_group,
                                                        int c_per_group,
                                                        int ho,
                                                        int wo,
                                                        int sy,
                                                        int sx,
                                                        int dy,
                                                        int dx,
                                                        int py,
                                                        int px,
                                                        int fy,
                                                        int fx,
                                                        int group)
    {
        /*
        *  need to compute total output pixel: `group * n * k_per_group * ho * wo`.
        *  to distribute this workload, let one workgroup compute `ho * wo` pixel,
        *  hence need `group * n * k_per_group` workgroups (grid_size).
        */
        int k             = k_per_group * group;
        int c             = c_per_group * group;
        int thread_length = ho * wo;
        int bid           = blockIdx.x;
        int ik            = bid % k_per_group;
        int in            = (bid / k_per_group) % n;
        int ig            = bid / (n * k_per_group);

        p_in += in * c * hi * wi + ig * c_per_group * hi * wi;
        p_wei += ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx;
        p_out += in * k * ho * wo + ig * k_per_group * ho * wo + ik * ho * wo;

        for(int tid = threadIdx.x; tid < thread_length; tid += 256)
        {
            int iho = tid / wo;
            int iwo = tid % wo;

            float value = .0f;

            for(int ic = 0; ic < c_per_group; ic++)
            {
                for(int iy = 0; iy < fy; iy++)
                {
                    int valid_h = 1;
                    int cur_h   = sy * iho - py + dy * iy;
                    if(cur_h < 0 || cur_h >= hi)
                        valid_h &= 0;
                    for(int ix = 0; ix < fx; ix++)
                    {
                        int valid_w = 1;
                        int cur_w   = sx * iwo - px + dx * ix;
                        if(cur_w < 0 || cur_w >= wi)
                            valid_w &= 0;

                        if(valid_w & valid_h)
                        {
                            int i_idx = ic * hi * wi + cur_h * wi + cur_w;
                            int w_idx = ic * fy * fx + iy * fx + ix;
                            value += p_in[i_idx] * p_wei[w_idx];
                        }
                    }
                }
            }
            int o_idx    = iho * wo + iwo;
            p_out[o_idx] = value;
        }
    }

    extern "C" __global__ void naive_conv_fwd_ncdhw_fp32(const float* __restrict__ p_in,
                                                        const float* __restrict__ p_wei,
                                                        float* __restrict__ p_out,
                                                        int di,
                                                        int hi,
                                                        int wi,
                                                        int n,
                                                        int k_per_group,
                                                        int c_per_group,
                                                        int do_,
                                                        int ho,
                                                        int wo,
                                                        int sz,
                                                        int sy,
                                                        int sx,
                                                        int dz,
                                                        int dy,
                                                        int dx,
                                                        int pz,
                                                        int py,
                                                        int px,
                                                        int fz,
                                                        int fy,
                                                        int fx,
                                                        int group)
    {
        /*
        *  need to compute total output pixel: `group * n * k_per_group * do_ * ho * wo`.
        *  to distribute this workload, let one workgroup compute `do_ * ho * wo` pixel,
        *  hence need `group * n * k_per_group` workgroups (grid_size).
        */
        int k             = k_per_group * group;
        int c             = c_per_group * group;
        int thread_length = do_ * ho * wo;
        int bid           = blockIdx.x;
        int ik            = bid % k_per_group;
        int in            = (bid / k_per_group) % n;
        int ig            = bid / (n * k_per_group);

        p_in += in * c * di * hi * wi + ig * c_per_group * di * hi * wi;
        p_wei += ig * k_per_group * c_per_group * fz * fy * fx + ik * c_per_group * fz * fy * fx;
        p_out += in * k * do_ * ho * wo + ig * k_per_group * do_ * ho * wo + ik * do_ * ho * wo;

        for(int tid = threadIdx.x; tid < thread_length; tid += 256)
        {
            int iwo = tid % wo;
            int iho = (tid / wo) % ho;
            int ido = tid / (ho * wo);

            float value = .0f;

            for(int ic = 0; ic < c_per_group; ic++)
            {
                for(int iz = 0; iz < fz; iz++)
                {
                    int valid_d = 1;
                    int cur_d   = sz * ido - pz + dz * iz;
                    if(cur_d < 0 || cur_d >= di)
                        valid_d &= 0;
                    for(int iy = 0; iy < fy; iy++)
                    {
                        int valid_h = 1;
                        int cur_h   = sy * iho - py + dy * iy;
                        if(cur_h < 0 || cur_h >= hi)
                            valid_h &= 0;
                        for(int ix = 0; ix < fx; ix++)
                        {
                            int valid_w = 1;
                            int cur_w   = sx * iwo - px + dx * ix;
                            if(cur_w < 0 || cur_w >= wi)
                                valid_w &= 0;

                            if(valid_d & valid_w & valid_h)
                            {
                                int i_idx = ic * di * hi * wi + cur_d * hi * wi + cur_h * wi + cur_w;
                                int w_idx = ic * fz * fy * fx + iz * fy * fx + iy * fx + ix;
                                value += p_in[i_idx] * p_wei[w_idx];
                            }
                        }
                    }
                }
            }
            int o_idx    = ido * ho * wo + iho * wo + iwo;
            p_out[o_idx] = value;
        }
    }
    '''
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is naive_conv_ctrl_t
        self.ctrl = ctrl
        self.spatial_dim = naive_get_spatial_dimension(ctrl.tensor_layout)
        self.karg = self.kernel_karg_t(mc, self)
        self.sgpr = self.kernel_sgpr_t(mc, self)
        self.vgpr = self.kernel_vgpr_t(mc, self)

    def name(self):
        return f"naive_conv_{self.ctrl.direction}_{self.ctrl.tensor_layout}_{self.ctrl.precision}"

    class kernel_karg_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.k_p_in             = sym_t("k_p_in",           0)
            self.k_p_wei            = sym_t("k_p_wei",          8)
            self.k_p_out            = sym_t("k_p_out",          16)
            if outer.spatial_dim == 2:
                self.k_hi           = sym_t("k_hi",             24)
                self.k_wi           = sym_t("k_wi",             28)
                self.k_n            = sym_t("k_n",              32)
                self.k_k_per_group  = sym_t("k_k_per_group",    36)
                self.k_c_per_group  = sym_t("k_c_per_group",    40)
                self.k_ho           = sym_t("k_ho",             44)
                self.k_wo           = sym_t("k_wo",             48)
                self.k_sy           = sym_t("k_sy",             52)
                self.k_sx           = sym_t("k_sx",             56)
                self.k_dy           = sym_t("k_dy",             60)
                self.k_dx           = sym_t("k_dx",             64)
                self.k_py           = sym_t("k_py",             68)
                self.k_px           = sym_t("k_px",             72)
                self.k_fy           = sym_t("k_fy",             76)
                self.k_fx           = sym_t("k_fx",             80)
                self.k_group        = sym_t("k_group",          84)
                self.k_end          = sym_t("k_end",            88)
            else:
                self.k_di           = sym_t("k_di",             24)
                self.k_hi           = sym_t("k_hi",             28)
                self.k_wi           = sym_t("k_wi",             32)
                self.k_n            = sym_t("k_n",              36)
                self.k_k_per_group  = sym_t("k_k_per_group",    40)
                self.k_c_per_group  = sym_t("k_c_per_group",    44)
                self.k_do           = sym_t("k_do",             48)
                self.k_ho           = sym_t("k_ho",             52)
                self.k_wo           = sym_t("k_wo",             56)
                self.k_sz           = sym_t("k_sz",             60)
                self.k_sy           = sym_t("k_sy",             64)
                self.k_sx           = sym_t("k_sx",             68)
                self.k_dz           = sym_t("k_dz",             72)
                self.k_dy           = sym_t("k_dy",             76)
                self.k_dx           = sym_t("k_dx",             80)
                self.k_pz           = sym_t("k_pz",             84)
                self.k_py           = sym_t("k_py",             88)
                self.k_px           = sym_t("k_px",             92)
                self.k_fz           = sym_t("k_fz",             96)
                self.k_fy           = sym_t("k_fy",             100)
                self.k_fx           = sym_t("k_fx",             104)
                self.k_group        = sym_t("k_group",          108)
                self.k_end          = sym_t("k_end",            112)

        def get_count(self):
            return self.k_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('k_'):
                    self._emit(v.declare())
        
    class kernel_sgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
            self.s_ka               = sym_t("s_ka",             0)
            self.s_bx               = sym_t("s_bx",             2)
            self.s_p_in             = sym_t("s_p_in",           4)
            self.s_p_wei            = sym_t("s_p_wei",          8)
            self.s_p_out            = sym_t("s_p_out",          12)
            sseq                    = gpr_sequencer_t(16)
            if outer.spatial_dim == 2:
                self.s_hi           = sym_t("s_hi",             sseq(1))
                self.s_wi           = sym_t("s_wi",             sseq(1))
                self.s_n            = sym_t("s_n",              sseq(1))
                self.s_k_per_group  = sym_t("s_k_per_group",    sseq(1))
                self.s_c_per_group  = sym_t("s_c_per_group",    sseq(1))
                self.s_ho           = sym_t("s_ho",             sseq(1))
                self.s_wo           = sym_t("s_wo",             sseq(1))
                self.s_sy           = sym_t("s_sy",             sseq(1))
                self.s_sx           = sym_t("s_sx",             sseq(1))
                self.s_dy           = sym_t("s_dy",             sseq(1))
                self.s_dx           = sym_t("s_dx",             sseq(1))
                self.s_py           = sym_t("s_py",             sseq(1))
                self.s_px           = sym_t("s_px",             sseq(1))
                self.s_fy           = sym_t("s_fy",             sseq(1))
                self.s_fx           = sym_t("s_fx",             sseq(1))
                self.s_group        = sym_t("s_group",          sseq(1))
            else:
                self.s_di           = sym_t("s_di",             sseq(1))
                self.s_hi           = sym_t("s_hi",             sseq(1))
                self.s_wi           = sym_t("s_wi",             sseq(1))
                self.s_n            = sym_t("s_n",              sseq(1))
                self.s_k_per_group  = sym_t("s_k_per_group",    sseq(1))
                self.s_c_per_group  = sym_t("s_c_per_group",    sseq(1))
                self.s_do           = sym_t("s_do",             sseq(1))
                self.s_ho           = sym_t("s_ho",             sseq(1))
                self.s_wo           = sym_t("s_wo",             sseq(1))
                self.s_sz           = sym_t("s_sz",             sseq(1))
                self.s_sy           = sym_t("s_sy",             sseq(1))
                self.s_sx           = sym_t("s_sx",             sseq(1))
                self.s_dz           = sym_t("s_dz",             sseq(1))
                self.s_dy           = sym_t("s_dy",             sseq(1))
                self.s_dx           = sym_t("s_dx",             sseq(1))
                self.s_pz           = sym_t("s_pz",             sseq(1))
                self.s_py           = sym_t("s_py",             sseq(1))
                self.s_px           = sym_t("s_px",             sseq(1))
                self.s_fz           = sym_t("s_fz",             sseq(1))
                self.s_fy           = sym_t("s_fy",             sseq(1))
                self.s_fx           = sym_t("s_fx",             sseq(1))
                self.s_group        = sym_t("s_group",          sseq(1))
            self.s_c                = sym_t("s_c",              sseq(1))
            self.s_k                = sym_t("s_k",              sseq(1))
            self.s_in               = sym_t("s_in",             sseq(1))
            self.s_ik               = sym_t("s_ik",             sseq(1))
            self.s_ic               = sym_t("s_ic",             sseq(1))
            self.s_ig               = sym_t("s_ig",             sseq(1))
            self.s_iy               = sym_t("s_iy",             sseq(1))
            self.s_ix               = sym_t("s_ix",             sseq(1))
            self.s_step             = sym_t("s_step",           sseq(1))
            self.s_thread_length    = sym_t("s_thread_length",  sseq(1))
            #self.s_k_num            = sym_t("s_k_num",          0)
            #self.s_k_itr            = sym_t("s_k_itr",          1)
            self.s_in_offset        = sym_t("s_in_offset",      sseq(1))
            self.s_wei_offset       = sym_t("s_wei_offset",     sseq(1))
            self.s_out_offset       = sym_t("s_out_offset",     sseq(1))
            self.s_nan_inf          = sym_t("s_nan_inf",        sseq(1))
            self.s_7fff             = sym_t("s_7fff",           sseq(1))
            self.s_exec_buf         = sym_t("s_exec_buf",       sseq(4, 2))
            self.s_tmp              = sym_t("s_tmp"                    ,sseq(4, 2))
            self.s_end              = sym_t("s_end"                    ,sseq())

        def get_count(self):
            return self.s_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('s_'):
                    self._emit(v.declare())

    class kernel_vgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            vseq = gpr_sequencer_t()
            self.v_tid                  = sym_t("v_tid"                 ,vseq(1))
            self.v_buf_in               = sym_t("v_buf_in"              ,vseq(1))
            self.v_buf_wei              = sym_t("v_buf_wei"             ,vseq(1))
            self.v_buf_out              = sym_t("v_buf_out"             ,vseq(1))
            self.v_iho                  = sym_t("v_iho"                 ,vseq(1))
            self.v_iwo                  = sym_t("v_iwo"                 ,vseq(1))
            self.v_valid_flag           = sym_t("v_valid_flag"          ,vseq(1))
            self.v_cur_h                = sym_t("v_cur_h"               ,vseq(1))
            self.v_cur_w                = sym_t("v_cur_w"               ,vseq(1))
            self.v_in_offset            = sym_t("v_in_offset"           ,vseq(1))
            self.v_wei_offset           = sym_t("v_wei_offset"          ,vseq(1))
            self.v_out_offset           = sym_t("v_out_offset"          ,vseq(1))
            self.v_10000                = sym_t("v_10000"               ,vseq(1))

            self.v_tmp                  = sym_t("v_tmp"                 ,vseq(6, 2))
            self.v_end                  = sym_t("v_end"                 ,vseq())

        def get_count(self):
            return self.v_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('v_'):
                    self._emit(v.declare())

    def get_kernel_code(self):
        kernel_code = amdgpu_kernel_code_t({
                'enable_sgpr_kernarg_segment_ptr'   :   1,
                'enable_sgpr_workgroup_id_x'        :   1,
                'enable_vgpr_workitem_id'           :   0,
                'workgroup_group_segment_byte_size' :   0,
                'kernarg_segment_byte_size'         :   self.karg.get_count(),
                'wavefront_sgpr_count'              :   self.sgpr.get_count() + 2*3,
                'workitem_vgpr_count'               :   self.vgpr.get_count()
                })
        return kernel_code

    def get_kernel_args(self):
        kas = []
        kas.append(amdgpu_kernel_arg_t('p_in'               , 8,   0, 'global_buffer','f32',address_space='global',is_const='false'))
        kas.append(amdgpu_kernel_arg_t('p_wei'              , 8,   8, 'global_buffer','f32',address_space='global',is_const='false'))
        kas.append(amdgpu_kernel_arg_t('p_out'              , 8,  16, 'global_buffer','f32',address_space='global',is_const='false'))
        if self.spatial_dim == 2:
            kas.append(amdgpu_kernel_arg_t('hi'             , 4,  24, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('wi'             , 4,  28, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('n'              , 4,  32, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('k_per_group'    , 4,  36, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('c_per_group'    , 4,  40, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('ho'             , 4,  44, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('wo'             , 4,  48, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('sy'             , 4,  52, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('sx'             , 4,  56, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('dy'             , 4,  60, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('dx'             , 4,  64, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('py'             , 4,  68, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('px'             , 4,  72, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('fy'             , 4,  76, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('fx'             , 4,  80, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('group'          , 4,  84, 'by_value', 'i32'))
        else:
            kas.append(amdgpu_kernel_arg_t('di'             , 4,  24, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('hi'             , 4,  28, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('wi'             , 4,  32, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('n'              , 4,  36, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('k_per_group'    , 4,  40, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('c_per_group'    , 4,  44, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('do_'            , 4,  48, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('ho'             , 4,  52, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('wo'             , 4,  56, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('sz'             , 4,  60, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('sy'             , 4,  64, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('sx'             , 4,  68, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('dz'             , 4,  72, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('dy'             , 4,  76, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('dx'             , 4,  80, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('pz'             , 4,  84, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('py'             , 4,  88, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('px'             , 4,  92, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('fz'             , 4,  96, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('fy'             , 4, 100, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('fx'             , 4, 104, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('group'          , 4, 108, 'by_value', 'i32'))
        return kas
    
    def get_kernel_info(self):
        kernel_code = self.get_kernel_code()
        kernel_args = self.get_kernel_args()
        kernel_info = amdgpu_kernel_info_t(kernel_code, self.name(), self.ctrl.block_size, kernel_args)
        return kernel_info

    def emit_kernel_symbol(self):
        self.karg.emit()
        self._emit_empty_line()
        self.sgpr.emit()
        self._emit_empty_line()
        self.vgpr.emit()
        self._emit_empty_line()

    def emit_kernel_header(self):
        kernel_name = self.name()
        self._emit('.text')
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
            self._emit('.globl {}'.format(kernel_name))
        self._emit('.p2align 8')
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
            self._emit('.type {},@function'.format(kernel_name))
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V2:
            self._emit('.amdgpu_hsa_kernel {}'.format(kernel_name))
        self._emit('{}:'.format(kernel_name))

    def emit_kernel_body(self):
        data_byte = amdgpu_precision_data_byte(self.ctrl.precision)

        m_int_div_rem_vs = macro_int_div_rem_vs_t(self.mc)
        m_int_div_rem_ss = macro_int_div_rem_ss_t(self.mc)
        m_set_flag_hw = naive_set_flag_hw(self.mc)
        naive_fma = naive_fma_t(self.mc, self.ctrl)
        if self.ctrl.precision == 'bf16':
            naive_float_to_bfloat16 = naive_float_to_bfloat16_t(self.mc, self.ctrl)

        s = self.sgpr
        v = self.vgpr
        k = self.karg

        label_thread_loop_start = f"{self.name()}_thread_loop_start"
        label_fma_start = f"{self.name()}_fma_start"
        label_fma_end = f"{self.name()}_fma_end"
        label_out = f"{self.name()}_out"

        def emit_cur_h():
            with self._deferred_context():
                # int cur_h   = sy * iho - py + dy * iy;
                self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_dy()}], s[{s.s_iy()}]")
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_sy()}], v[{v.v_iho()}]")
                self._emit(f"s_sub_i32 s[{s.s_tmp()}], s[{s.s_tmp(1)}], s[{s.s_py()}]")
                self._emit(f"v_add_i32 v[{v.v_cur_h()}], s[{s.s_tmp()}], v[{v.v_tmp()}]")
            return self._get_deferred()

        def emit_cur_w():
            with self._deferred_context():
                # int cur_w   = sx * iwo - px + dx * ix;
                self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_dx()}], s[{s.s_ix()}]")
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_sx()}], v[{v.v_iwo()}]")
                self._emit(f"s_sub_i32 s[{s.s_tmp()}], s[{s.s_tmp(1)}], s[{s.s_px()}]")
                self._emit(f"v_add_i32 v[{v.v_cur_w()}], s[{s.s_tmp()}], v[{v.v_tmp()}]")
            return self._get_deferred()
        
        def emit_i_idx():
            # int i_idx = ic * hi * wi + cur_h * wi + cur_w;
            with self._deferred_context():
                self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_hi()}], s[{s.s_wi()}]")
                self._emit(f"s_mul_i32 s[{s.s_in_offset()}], s[{s.s_tmp()}], s[{s.s_ic()}]")
                self._emit(f"s_lshl_b32 s[{s.s_in_offset()}], s[{s.s_in_offset()}], {igemm_log2(data_byte)}")
                self._emit(f"v_mad_u32_u24 v[{v.v_tmp()}], v[{v.v_cur_h()}], s[{s.s_wi()}], v[{v.v_cur_w()}]")
                self._emit(f"v_lshlrev_b32 v[{v.v_in_offset()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")
            return self._get_deferred()

        def emit_w_idx():
            # ic * fy * fx + iy * fx + ix;
            with self._deferred_context():
                self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_fy()}], s[{s.s_fx()}]")
                self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_tmp(1)}], s[{s.s_ic()}]")
                self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_iy()}], s[{s.s_fx()}]")
                self._emit(f"s_add_u32 s[{s.s_tmp(1)}], s[{s.s_tmp(2)}], s[{s.s_ix()}]")
                self._emit(f"s_add_u32 s[{s.s_tmp()}], s[{s.s_tmp()}], s[{s.s_tmp(1)}]")
                self._emit(f"s_lshl_b32 s[{s.s_wei_offset()}], s[{s.s_tmp()}], {igemm_log2(data_byte)}")
            return self._get_deferred()

        def emit_o_idx():
            # int o_idx    = iho * wo + iwo;
            with self._deferred_context():
                self._emit(f"v_mad_u32_u24 v[{v.v_tmp()}], v[{v.v_iho()}], s[{s.s_wo()}], v[{v.v_iwo()}]")
                self._emit(f"v_lshlrev_b32 v[{v.v_out_offset()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")
            return self._get_deferred()

        def emit_global_load():
            with self._deferred_context():
                self._emit(f"v_cmp_eq_u32 vcc, 1, v[{v.v_valid_flag()}]")
                self._emit(f"s_and_saveexec_b64 s[{s.s_exec_buf((2,3))}], vcc")
                if self.ctrl.precision is 'fp32':
                    self._emit(f"buffer_load_dword v[{v.v_buf_in()}], v[{v.v_in_offset()}], s[{s.s_p_in((0, 3))}], s[{s.s_in_offset()}] offen offset:0")
                    self._emit(f"buffer_load_dword v[{v.v_buf_wei()}], v[{v.v_wei_offset()}], s[{s.s_p_wei((0, 3))}], s[{s.s_wei_offset()}] offen offset:0")
                elif self.ctrl.precision in ('fp16', 'bf16'):
                    self._emit(f"buffer_load_ushort v[{v.v_buf_in()}], v[{v.v_in_offset()}], s[{s.s_p_in((0, 3))}], s[{s.s_in_offset()}] offen offset:0")
                    self._emit(f"buffer_load_ushort v[{v.v_buf_wei()}], v[{v.v_wei_offset()}], s[{s.s_p_wei((0, 3))}], s[{s.s_wei_offset()}] offen offset:0")
                else:
                    assert False
                self._emit(f"s_or_b64 exec, exec, s[{s.s_exec_buf((2,3))}]")
            return self._get_deferred()

        def emit_global_store():
            with self._deferred_context():
                if self.ctrl.precision is 'fp32':
                    self._emit(f"buffer_store_dword v[{v.v_buf_out()}], v[{v.v_out_offset()}], s[{s.s_p_out((0, 3))}], s[{s.s_out_offset()}] offen offset:0")
                elif self.ctrl.precision is 'fp16':
                    # convert back
                    self._emit(f"v_cvt_f16_f32 v[{v.v_buf_out()}], v[{v.v_buf_out()}]")
                    self._emit(f"buffer_store_short v[{v.v_buf_out()}], v[{v.v_out_offset()}], s[{s.s_p_out((0, 3))}], s[{s.s_out_offset()}] offen offset:0")
                elif self.ctrl.precision is 'bf16':
                    # convert back
                    self._emit(naive_float_to_bfloat16(v.v_tmp(3), v.v_buf_out(), v.v_10000(), s.s_nan_inf(), s.s_7fff(), s.s_tmp(), v.v_tmp()))
                    self._emit(f"buffer_store_short v[{v.v_tmp(3)}], v[{v.v_out_offset()}], s[{s.s_p_out((0, 3))}], s[{s.s_out_offset()}] offen offset:0")
                else:
                    assert False
            return self._get_deferred()

        def emit_move_slice_window():
            # naive slice window emulation
            with self._deferred_context():
                self._emit(f"; move slice window along c_per_group*y*x")
                self._emit(f"s_addk_i32 s[{s.s_ix()}], 1")
                self._emit(f"s_cmp_lt_u32 s[{s.s_ix()}], s[{s.s_fx()}]")
                self._emit(f"s_cselect_b32 s[{s.s_ix()}], s[{s.s_ix()}], 0")
                self._emit(f"s_cselect_b32 s[{s.s_tmp()}], 0, 1")
                self._emit(f"s_add_u32 s[{s.s_iy()}], s[{s.s_iy()}], s[{s.s_tmp()}]")
                self._emit(f"s_cmp_lt_u32 s[{s.s_iy()}], s[{s.s_fy()}]")
                self._emit(f"s_cselect_b32 s[{s.s_iy()}], s[{s.s_iy()}], 0")
                self._emit(f"s_cselect_b32 s[{s.s_tmp()}], 0, 1")
                self._emit(f"s_add_u32 s[{s.s_ic()}], s[{s.s_ic()}], s[{s.s_tmp()}]")
            return self._get_deferred()

        self._emit(f"s_load_dwordx2  s[{s.s_p_in((0,1))}],    s[{s.s_ka((0, 1))}],    0+{k.k_p_in()}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_wei((0,1))}],    s[{s.s_ka((0, 1))}],    0+{k.k_p_wei()}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_out((0,1))}],    s[{s.s_ka((0, 1))}],    0+{k.k_p_out()}")
        if self.spatial_dim == 2:
            self._emit(f"s_load_dwordx8 s[{s.s_hi((0, 7))}],    s[{s.s_ka((0, 1))}],    0+{k.k_hi()}")
            self._emit(f"s_load_dwordx8 s[{s.s_sx((0, 7))}],    s[{s.s_ka((0, 1))}],    0+{k.k_sx()}")
        else:
            self._emit(f"s_load_dwordx8 s[{s.s_di((0, 7))}],    s[{s.s_ka((0, 1))}],    0+{k.k_di()}")
            self._emit(f"s_load_dwordx8 s[{s.s_wo((0, 7))}],    s[{s.s_ka((0, 1))}],    0+{k.k_wo()}")
            self._emit(f"s_load_dwordx4 s[{s.s_py((0, 3))}],    s[{s.s_ka((0, 1))}],    0+{k.k_py()}")
            self._emit(f"s_load_dwordx2 s[{s.s_fx((0, 1))}],    s[{s.s_ka((0, 1))}],    0+{k.k_fx()}")

        self._emit(f"s_mov_b32 s[{s.s_p_in(2)}], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_in(3)}], 0x27000")
        self._emit(f"s_mov_b32 s[{s.s_p_wei(2)}], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_wei(3)}], 0x27000")
        self._emit(f"s_mov_b32 s[{s.s_p_out(2)}], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_out(3)}], 0x27000")
        self._emit(f"s_mov_b32 s[{s.s_step()}], {self.ctrl.block_size}     ; block_size")
        self._emit(f"v_mov_b32 v[{v.v_in_offset()}], 0")
        self._emit(f"s_mov_b32 s[{s.s_in_offset()}], 0")
        self._emit(f"v_mov_b32 v[{v.v_wei_offset()}], 0")
        self._emit(f"s_mov_b32 s[{s.s_wei_offset()}], 0")
        self._emit(f"v_mov_b32 v[{v.v_out_offset()}], 0")
        self._emit(f"s_mov_b32 s[{s.s_out_offset()}], 0")
        self._emit(f"v_mov_b32 v[{v.v_10000()}], 0x10000")
        self._emit(f"s_mov_b32 s[{s.s_nan_inf()}], 0x7f800000")
        self._emit(f"s_mov_b32 s[{s.s_7fff()}], 0x7fff")
        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit_empty_line()
        self._emit(f"s_mul_i32 s[{s.s_c()}], s[{s.s_c_per_group()}], s[{s.s_group()}]")
        self._emit(f"s_mul_i32 s[{s.s_k()}], s[{s.s_k_per_group()}], s[{s.s_group()}]")
        self._emit(f"s_mul_i32 s[{s.s_thread_length()}], s[{s.s_ho()}], s[{s.s_wo()}]")
        self._emit(m_int_div_rem_ss(s.s_ik(), s.s_tmp(4), s.s_bx(), s.s_k_per_group(), v.v_tmp(5), v.v_tmp(), s.s_tmp()))
        self._emit(m_int_div_rem_ss(s.s_in(), s.s_ig(), s.s_tmp(4), s.s_n(), v.v_tmp(5), v.v_tmp(), s.s_tmp()))
        self._emit_empty_line()

        self._emit(f"; p_in += in * c * hi * wi + ig * c_per_group * hi * wi;")
        self._emit(f"s_mul_i32 s[{s.s_tmp(4)}], s[{s.s_hi()}], s[{s.s_wi()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(4)}], s[{s.s_tmp(4)}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp(3)}], s[{s.s_in()}], s[{s.s_c()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_tmp(4)}], s[{s.s_tmp(3)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_tmp(4)}], s[{s.s_tmp(3)}]")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], s[{s.s_tmp(1)}]")

        self._emit(f"s_mul_i32 s[{s.s_tmp(3)}], s[{s.s_ig()}], s[{s.s_c_per_group()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_tmp(4)}], s[{s.s_tmp(3)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_tmp(4)}], s[{s.s_tmp(3)}]")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], s[{s.s_tmp(1)}]")

        self._emit(f"; p_wei += ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx;")
        self._emit(f"s_mul_i32 s[{s.s_tmp(4)}], s[{s.s_fy()}], s[{s.s_fx()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp(4)}], s[{s.s_tmp(4)}], s[{s.s_c_per_group()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(4)}], s[{s.s_tmp(4)}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp(3)}], s[{s.s_ig()}], s[{s.s_k_per_group()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_tmp(4)}], s[{s.s_tmp(3)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_tmp(4)}], s[{s.s_tmp(3)}]")
        self._emit(f"s_add_u32 s[{s.s_p_wei()}], s[{s.s_p_wei()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_wei(1)}], s[{s.s_p_wei(1)}], s[{s.s_tmp(1)}]")

        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_tmp(4)}], s[{s.s_ik()}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_tmp(4)}], s[{s.s_ik()}]")
        self._emit(f"s_add_u32 s[{s.s_p_wei()}], s[{s.s_p_wei()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_wei(1)}], s[{s.s_p_wei(1)}], s[{s.s_tmp(1)}]")

        self._emit(f"; p_out += in * k * ho * wo + ig * k_per_group * ho * wo + ik * ho * wo;")
        self._emit(f"s_mul_i32 s[{s.s_tmp(4)}], s[{s.s_ho()}], s[{s.s_wo()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(4)}], s[{s.s_tmp(4)}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp(3)}], s[{s.s_ig()}], s[{s.s_k_per_group()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_tmp(4)}], s[{s.s_tmp(3)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_tmp(4)}], s[{s.s_tmp(3)}]")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], s[{s.s_tmp(1)}]")

        self._emit(f"s_mul_i32 s[{s.s_tmp(3)}], s[{s.s_in()}], s[{s.s_k()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_tmp(4)}], s[{s.s_tmp(3)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_tmp(4)}], s[{s.s_tmp(3)}]")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], s[{s.s_tmp(1)}]")

        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_tmp(4)}], s[{s.s_ik()}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_tmp(4)}], s[{s.s_ik()}]")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], s[{s.s_tmp(1)}]")
        self._emit_empty_line()

        self._emit(f"v_cmp_lt_u32 vcc, v[{v.v_tid()}], s[{s.s_thread_length()}]")
        self._emit(f"s_cbranch_vccz {label_out}")
        self._emit_front(f"{label_thread_loop_start}:")
        self._emit(f"s_and_saveexec_b64 s[{s.s_exec_buf((0, 1))}], vcc")
        self._emit(m_int_div_rem_vs(v.v_iwo(), v.v_iho(), v.v_tid(), s.s_wo(), v.v_tmp(), s.s_tmp()))

        self._emit(f"s_mov_b32 s[{s.s_ic()}], 0")
        self._emit(f"s_mov_b32 s[{s.s_iy()}], 0")
        self._emit(f"s_mov_b32 s[{s.s_ix()}], 0")
        self._emit(f"v_mov_b32 v[{v.v_buf_out()}], 0")

        self._emit(emit_cur_h())
        self._emit(emit_cur_w())
        self._emit(emit_i_idx())
        self._emit(emit_w_idx())
        self._emit(m_set_flag_hw(v.v_valid_flag(), v.v_cur_h(), v.v_cur_w(), s.s_hi(), s.s_wi()))
        self._emit(emit_global_load())
        self._emit(emit_o_idx())            # calculate out offset here
        self._emit_empty_line()

        self._emit_front(f"{label_fma_start}:")
        self._emit(emit_move_slice_window())
        self._emit(f"s_cmp_lt_u32 s[{s.s_ic()}], s[{s.s_c_per_group()}]")
        self._emit(f"s_cbranch_scc0 {label_fma_end}")
        self._emit(emit_cur_h())
        self._emit(emit_cur_w())
        self._emit(emit_i_idx())
        self._emit(emit_w_idx())
        self._emit(m_set_flag_hw(v.v_valid_flag(), v.v_cur_h(), v.v_cur_w(), s.s_hi(), s.s_wi()))
        self._emit(f"s_waitcnt vmcnt(0)")
        self._emit(naive_fma(v.v_buf_out(), v.v_buf_in(), v.v_buf_wei(), v.v_tmp()))
        self._emit(emit_global_load())
        self._emit(f"s_branch {label_fma_start}")
        self._emit_empty_line()

        self._emit_front(f"{label_fma_end}:")
        self._emit(f"s_waitcnt vmcnt(0)")
        self._emit(naive_fma(v.v_buf_out(), v.v_buf_in(), v.v_buf_wei(), v.v_tmp()))
        self._emit(emit_global_store())
        self._emit_empty_line()

        self._emit(f"s_or_b64 exec, exec, s[{s.s_exec_buf((0, 1))}]")
        self._emit(f"v_add_u32 v[{v.v_tid()}], s[{s.s_step()}], v[{v.v_tid()}]")
        self._emit(f"v_cmp_lt_u32 vcc, v[{v.v_tid()}], s[{s.s_thread_length()}]")
        self._emit(f"s_cbranch_vccnz {label_thread_loop_start}")
        self._emit_empty_line()

        self._emit_front(f"{label_out}:")
        self._emit(f"s_endpgm")


    def emit_kernel_end(self):
        self._emit('s_endpgm')
    def emit_kernel_footer(self):
        self._emit_empty_line()

    def emit_kernel_amd_kernel_code_t(self):
        amd_kernel_code_t(self.mc, self.get_kernel_info()).emit()
