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
from ..codegen import *
from ..operations import *
from .igemm_base import *



class igemm_upsampling_clear_t(mc_base_t):
    '''
    // in upsampling of bwd, require zero initilize some pixel in input-gradient
    // this kernel is to deal with such case

    // prototype is as below:
    #include <hip/hip_runtime.h>

    // design block_size 256
    extern "C" __global__
    void igemm_upsampling_clear(float * input,
        int hi,
        int wi,
        int n,
        int k,                      // this is indeed k_per_group
        int c,                      // this is indeed c_per_group
        int ho,
        int wo,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int pad_h,
        int pad_w,
        int y,
        int x)
    {
        input = input + blockIdx.x * hi * wi;

        for(int tid = threadIdx.x;tid < (hi*wi); tid += 256){
            if(tid >= (hi*wi))
                continue;
            int current_hi = tid / wi;
            int current_wi = tid % wi;
            int need_fill_zero = 1;

            for(int iy = 0; iy < y; iy++){
                int current_ho = current_hi + pad_h - dilation_h * iy;
                if (current_ho < 0 || current_ho % stride_h)
                    continue;
                current_ho /= stride_h;
                if(current_ho >= ho)
                    continue;
                for(int ix = 0; ix < x; ix++){
                    int current_wo = current_wi + pad_w - dilation_w * ix;
                    if(current_wo < 0 || current_wo % stride_w)
                        continue;
                    current_wo /= stride_w;
                    if(current_wo >= wo)
                        continue;
                    // finally, here is the place to multiple input with weight, hence no need fill zero
                    need_fill_zero = 0;
                }
            }
            if(need_fill_zero)
                input[current_hi * wi + current_wi] = .0f;
        }
    }
    '''
    def __init__(self, mc, tunable):
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
        self.karg = self.kernel_karg_t(mc, self)
        self.sgpr = self.kernel_sgpr_t(mc, self)
        self.vgpr = self.kernel_vgpr_t(mc, self)

    def name(self):
        return "igemm_upsampling_clear" + "_" + self.tunable.tensor_layout + "_" + self.tunable.precision
    
    class kernel_karg_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.k_p_in          = sym_t("k_p_in",          0)
            self.k_hi            = sym_t("k_hi",            8)
            self.k_wi            = sym_t("k_wi",            12)
            self.k_n             = sym_t("k_n",             16)
            self.k_k             = sym_t("k_k",             20)
            self.k_c             = sym_t("k_c",             24)
            self.k_ho            = sym_t("k_ho",            28)
            self.k_wo            = sym_t("k_wo",            32)
            self.k_stride_h      = sym_t("k_stride_h",      36)
            self.k_stride_w      = sym_t("k_stride_w",      40)
            self.k_dilation_h    = sym_t("k_dilation_h",    44)
            self.k_dilation_w    = sym_t("k_dilation_w",    48)
            self.k_pad_h         = sym_t("k_pad_h",         52)
            self.k_pad_w         = sym_t("k_pad_w",         56)
            self.k_y             = sym_t("k_y",             60)
            self.k_x             = sym_t("k_x",             64)
            self.k_group         = sym_t("k_group",         68)
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self.k_magic_0       = sym_t("k_magic_0",       72)
                self.k_magic_1       = sym_t("k_magic_1",       76)
                self.k_magic_2       = sym_t("k_magic_2",       80)
                self.k_shift_pack_0  = sym_t("k_shift_pack_0",  84)
                self.k_end           = sym_t("k_end",           88)
            else:
                self.k_end           = sym_t("k_end",           72)

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

            self.s_ka                      = sym_t("s_ka"                     ,0)
            self.s_bx                      = sym_t("s_bx"                     ,2)
            self.s_p_in                    = sym_t("s_p_in"                   ,4)
            self.s_hi                      = sym_t("s_hi"                     ,8)
            self.s_wi                      = sym_t("s_wi"                     ,9)
            self.s_n                       = sym_t("s_n"                      ,10)
            self.s_k                       = sym_t("s_k"                      ,11)
            self.s_c                       = sym_t("s_c"                      ,12)
            self.s_ho                      = sym_t("s_ho"                     ,13)
            self.s_wo                      = sym_t("s_wo"                     ,14)
            self.s_stride_h                = sym_t("s_stride_h"               ,15)
            self.s_stride_w                = sym_t("s_stride_w"               ,16)
            self.s_dilation_h              = sym_t("s_dilation_h"             ,17)
            self.s_dilation_w              = sym_t("s_dilation_w"             ,18)
            self.s_pad_h                   = sym_t("s_pad_h"                  ,19)
            self.s_pad_w                   = sym_t("s_pad_w"                  ,20)
            self.s_y                       = sym_t("s_y"                      ,21)
            self.s_x                       = sym_t("s_x"                      ,22)
            self.s_group                   = sym_t("s_group"                  ,23)
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self.s_magic_0             = sym_t("s_magic_0"                ,24)
                self.s_magic_1             = sym_t("s_magic_1"                ,25)
                self.s_magic_2             = sym_t("s_magic_2"                ,26)
                self.s_shift_pack_0        = sym_t("s_shift_pack_0"           ,27)
            sseq                           = gpr_sequencer_t(28)
            self.s_step                    = sym_t("s_step"                   ,sseq(1))
            self.s_in_stride_c             = sym_t("s_in_stride_c"            ,sseq(1))
            if IGEMM_GTC_FEAT_MAGIC_DIVISION:
                self.s_shift_0             = sym_t("s_shift_0"                ,sseq(1))
                self.s_shift_1             = sym_t("s_shift_1"                ,sseq(1))
                self.s_shift_2             = sym_t("s_shift_2"                ,sseq(1))
            self.s_iy                      = sym_t("s_iy"                     ,sseq(1))
            self.s_ix                      = sym_t("s_ix"                     ,sseq(1))
            self.s_exec_buf                = sym_t("s_exec_buf"               ,sseq(4, 2))
            self.s_tmp                     = sym_t("s_tmp"                    ,sseq(4, 2))
            self.s_end                     = sym_t("s_end"                    ,sseq())

        def get_count(self):
            return self.s_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('s_'):
                    self._emit(v.declare())

    class kernel_vgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            is_vgpr_acc_c = outer.tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS
            vseq = gpr_sequencer_t()
            self.v_tid                   = sym_t("v_tid"                    ,vseq(1))
            self.v_current_hi            = sym_t("v_current_hi"             ,vseq(1))
            self.v_current_wi            = sym_t("v_current_wi"             ,vseq(1))
            self.v_current_input_valid   = sym_t("v_current_input_valid"    ,vseq(1))
            self.v_zero                  = sym_t("v_zero"                   ,vseq(1))
            self.v_current_ho            = sym_t("v_current_ho"             ,vseq(1))
            self.v_current_ho_quo        = sym_t("v_current_ho_quo"         ,vseq(1))
            self.v_current_ho_rem        = sym_t("v_current_ho_rem"         ,vseq(1))
            self.v_current_ho_valid      = sym_t("v_current_ho_valid"       ,vseq(1))
            self.v_current_wo            = sym_t("v_current_wo"             ,vseq(1))
            self.v_current_wo_quo        = sym_t("v_current_wo_quo"         ,vseq(1))
            self.v_current_wo_rem        = sym_t("v_current_wo_rem"         ,vseq(1))
            self.v_current_wo_valid      = sym_t("v_current_wo_valid"       ,vseq(1))
            self.v_tmp                   = sym_t("v_tmp"                    ,vseq(6, 2))
            total_vgpr                   = vseq()
            self.v_end                   = sym_t("v_end"                    ,total_vgpr)

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
        '''
            float *p_in;
            int hi;
            int wi;
            int n;
            int k;
            int c;
            int ho;
            int wo;
            int stride_h;
            int stride_w;
            int dilation_h;
            int dilation_w;
            int pad_h;
            int pad_w;
            int y;
            int x;
            int group;
            int magic_0;            // denom of wi
            int magic_1;            // denom of stride_h
            int magic_2;            // denom of stride_w
            int shift_pack_0;
        '''
        kas = []
        # name: {}, .size: {}, .offset: {}, .value_kind: {}, .value_type
        kas.append(amdgpu_kernel_arg_t('p_in'          , 8,   0, 'global_buffer','f32',address_space='global',is_const='false'))
        kas.append(amdgpu_kernel_arg_t('hi'            , 4,   8, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('wi'            , 4,  12, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('n'             , 4,  16, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('k'             , 4,  20, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('c'             , 4,  24, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('ho'            , 4,  28, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('wo'            , 4,  32, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('stride_h'      , 4,  36, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('stride_w'      , 4,  40, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_h'    , 4,  44, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_w'    , 4,  48, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('pad_h'         , 4,  52, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('pad_w'         , 4,  56, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('y'             , 4,  60, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('x'             , 4,  64, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('group'         , 4,  68, 'by_value', 'i32'))
        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            kas.append(amdgpu_kernel_arg_t('magic_0'       , 4,  72, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('magic_1'       , 4,  76, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('magic_2'       , 4,  80, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('shift_pack_0'  , 4,  84, 'by_value', 'i32'))
        return kas
    
    def get_kernel_info(self):
        kernel_code = self.get_kernel_code()
        kernel_args = self.get_kernel_args()
        kernel_info = amdgpu_kernel_info_t(kernel_code, self.name(), self.tunable.block_size, kernel_args)
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
        data_byte = amdgpu_precision_data_byte(self.tunable.precision)
        label_start     = self.name() + "_start"
        label_next      = self.name() + "_next"
        label_end       = self.name() + "_end"
        label_y_start   = self.name() + "_y_start"
        label_x_start   = self.name() + "_x_start"

        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            m_mdiv_u32_vs = macro_mdiv_u32_rem_vs_t(self.mc)
        else:
            m_int_div_rem_vs = macro_int_div_rem_vs_t(self.mc)

        s = self.sgpr
        v = self.vgpr
        k = self.karg

        self._emit(f"s_load_dwordx2  s[{s.s_p_in((0,1))}],    s[{s.s_ka((0, 1))}],    0+{k.k_p_in()}")
        self._emit(f"s_load_dwordx8 s[{s.s_hi((0, 7))}],    s[{s.s_ka((0, 1))}],    0+{k.k_hi()}")
        self._emit(f"s_load_dwordx8 s[{s.s_stride_w((0, 7))}],    s[{s.s_ka((0, 1))}],    0+{k.k_stride_w()}")
        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            self._emit(f"s_load_dwordx4  s[{s.s_magic_0((0, 3))}]  s[{s.s_ka((0, 1))}],    0+{k.k_magic_0()}")
        self._emit(f"s_mov_b32 s[{s.s_p_in(2)}], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_in(3)}], 0x27000")
        self._emit(f"s_mov_b32 s[{s.s_step()}], 256     ; block_size")
        self._emit(f"v_mov_b32 v[{v.v_zero()}], 0x0     ; 0 for fp32/fp16/bf16")
        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit_empty_line()

        self._emit(f"s_mul_i32 s[{s.s_in_stride_c()}], s[{s.s_hi()}], s[{s.s_wi()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], s[{s.s_bx()}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_in_stride_c()}], s[{s.s_tmp(3)}]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_in_stride_c()}], s[{s.s_tmp(3)}]")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], s[{s.s_tmp(1)}]")

        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            self._emit(f"s_bfe_u32 s[{s.s_shift_0()}], s[{s.s_shift_pack_0()}], 0x00080000 ; offset:0,  width:8")
            self._emit(f"s_bfe_u32 s[{s.s_shift_1()}], s[{s.s_shift_pack_0()}], 0x00080008 ; offset:8,  width:8")
            self._emit(f"s_bfe_u32 s[{s.s_shift_2()}], s[{s.s_shift_pack_0()}], 0x00080010 ; offset:16, width:8")


        self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_in_stride_c()}], v[{v.v_tid()}]")
        self._emit(f"s_cbranch_vccz {label_end}")
        self._emit_front(f"{label_start}:")
        self._emit(f"s_and_saveexec_b64 s[{s.s_exec_buf((0, 1))}], vcc")
        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            self._emit(m_mdiv_u32_vs(v.v_current_wi(), v.v_current_hi(), v.v_tid(), s.s_magic_0(), s.s_shift_0(), s.s_wi(), v.v_tmp()))
        else:
            self._emit(m_int_div_rem_vs(v.v_current_wi(), v.v_current_hi(), v.v_tid(), s.s_wi(), v.v_tmp(), s.s_tmp()))
        self._emit_empty_line()

        self._emit(f"v_mov_b32 v[{v.v_current_ho_valid()}], 0")
        self._emit(f"v_mov_b32 v[{v.v_current_wo_valid()}], 0")
        self._emit(f"s_mov_b32 s[{s.s_iy()}], 0")
        self._emit_front(f"{label_y_start}:")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_dilation_h()}], s[{s.s_iy()}]")
        self._emit(f"v_add_i32 v[{v.v_tmp()}], s[{s.s_pad_h()}], v[{v.v_current_hi()}]")
        self._emit(f"v_sub_i32 v[{v.v_current_ho()}], v[{v.v_tmp()}], s[{s.s_tmp()}]")
        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            self._emit(m_mdiv_u32_vs(v.v_current_ho_rem(), v.v_current_ho_quo(), v.v_current_ho(), s.s_magic_1(), s.s_shift_1(), s.s_stride_h(), v.v_tmp()))
        else:
            self._emit(m_int_div_rem_vs(v.v_current_ho_rem(), v.v_current_ho_quo(), v.v_current_ho(), s.s_stride_h(), v.v_tmp(), s.s_tmp()))
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], 1")
        self._emit(f"v_cmp_le_i32 vcc, 0, v[{v.v_current_ho()}]")
        self._emit(f"v_cndmask_b32 v[{v.v_tmp()}], 0, v[{v.v_tmp()}], vcc")
        self._emit(f"v_cmp_eq_u32 vcc, 0, v[{v.v_current_ho_rem()}]")
        self._emit(f"v_cndmask_b32 v[{v.v_tmp()}], 0, v[{v.v_tmp()}], vcc")
        self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_ho()}],  v[{v.v_current_ho_quo()}]")
        self._emit(f"v_cndmask_b32 v[{v.v_tmp()}], 0, v[{v.v_tmp()}], vcc")
        self._emit(f"v_or_b32 v[{v.v_current_ho_valid()}], v[{v.v_current_ho_valid()}], v[{v.v_tmp()}]")
        self._emit(f"s_add_u32 s[{s.s_iy()}], 1, s[{s.s_iy()}]")
        self._emit(f"s_cmp_lt_u32 s[{s.s_iy()}], s[{s.s_y()}]")
        self._emit(f"s_cbranch_scc1 {label_y_start}")        
        self._emit_empty_line()

        self._emit(f"s_mov_b32 s[{s.s_ix()}], 0")
        self._emit_front(f"{label_x_start}:")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_dilation_w()}], s[{s.s_ix()}]")
        self._emit(f"v_add_i32 v[{v.v_tmp()}], s[{s.s_pad_w()}], v[{v.v_current_wi()}]")
        self._emit(f"v_sub_i32 v[{v.v_current_wo()}], v[{v.v_tmp()}], s[{s.s_tmp()}]")
        if IGEMM_GTC_FEAT_MAGIC_DIVISION:
            self._emit(m_mdiv_u32_vs(v.v_current_wo_rem(), v.v_current_wo_quo(), v.v_current_wo(), s.s_magic_2(), s.s_shift_2(), s.s_stride_w(), v.v_tmp()))
        else:
            self._emit(m_int_div_rem_vs(v.v_current_wo_rem(), v.v_current_wo_quo(), v.v_current_wo(), s.s_stride_w(), v.v_tmp(), s.s_tmp()))
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], 1")
        self._emit(f"v_cmp_le_i32 vcc, 0, v[{v.v_current_wo()}]")
        self._emit(f"v_cndmask_b32 v[{v.v_tmp()}], 0, v[{v.v_tmp()}], vcc")
        self._emit(f"v_cmp_eq_u32 vcc, 0, v[{v.v_current_wo_rem()}]")
        self._emit(f"v_cndmask_b32 v[{v.v_tmp()}], 0, v[{v.v_tmp()}], vcc")
        self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_wo()}],  v[{v.v_current_wo_quo()}]")
        self._emit(f"v_cndmask_b32 v[{v.v_tmp()}], 0, v[{v.v_tmp()}], vcc")
        self._emit(f"v_or_b32 v[{v.v_current_wo_valid()}], v[{v.v_current_wo_valid()}], v[{v.v_tmp()}]")
        self._emit(f"s_add_u32 s[{s.s_ix()}], 1, s[{s.s_ix()}]")
        self._emit(f"s_cmp_lt_u32 s[{s.s_ix()}], s[{s.s_x()}]")
        self._emit(f"s_cbranch_scc1 {label_x_start}")
        self._emit_empty_line()

        self._emit(f"v_and_b32 v[{v.v_current_input_valid()}], v[{v.v_current_wo_valid()}], v[{v.v_current_ho_valid()}]")
        self._emit(f"v_cmp_eq_u32 vcc, 0, v[{v.v_current_input_valid()}]")
        self._emit(f"s_and_saveexec_b64 s[{s.s_exec_buf((2, 3))}], vcc")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_wi()}], v[{v.v_current_hi()}]")
        self._emit(f"v_add_lshl_u32 v[{v.v_tmp()}], v[{v.v_tmp(1)}], v[{v.v_current_wi()}], {igemm_log2(data_byte)}")

        if self.tunable.precision == 'fp32':
            self._emit(f"buffer_store_dword v[{v.v_zero()}], v[{v.v_tmp()}], s[{s.s_p_in((0, 3))}], 0  offen offset:0")
        elif self.tunable.precision in ('fp16',  'bf16'):
            self._emit(f"buffer_store_short v[{v.v_zero()}], v[{v.v_tmp()}], s[{s.s_p_in((0, 3))}], 0  offen offset:0")
        else:
            assert False

        self._emit(f"s_or_b64 exec, exec, s[{s.s_exec_buf((2, 3))}]")

        self._emit_front(f"{label_next}:")
        self._emit(f"s_or_b64 exec, exec, s[{s.s_exec_buf((0, 1))}]")
        self._emit(f"v_add_u32 v[{v.v_tid()}], s[{s.s_step()}], v[{v.v_tid()}]")
        self._emit(f"v_cmp_gt_u32 vcc, s[{s.s_in_stride_c()}], v[{v.v_tid()}]")
        self._emit(f"s_cbranch_vccnz {label_start}")

        self._emit_front(f"{label_end}:")


    def emit_kernel_end(self):
        self._emit('s_endpgm')
    def emit_kernel_footer(self):
        self._emit_empty_line()

    def emit_kernel_amd_kernel_code_t(self):
        amd_kernel_code_t(self.mc, self.get_kernel_info()).emit()
