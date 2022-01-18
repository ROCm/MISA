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

from ..codegen import *
from .utility import *

class ctrl_spatial_tiling_t(object):
    def __init__(self):
        self.nxe = 1

class spatial_tiling_t(mc_base_t):
    '''
    // spatial-slice size of output
    size_t tmp_sps_ho = ho - (i_tile_h * tile_h);
    size_t sps_ho = tmp_sps_ho < tile_h ? tmp_sps_ho : tile_h;
    size_t tmp_sps_wo = wo - (i_tile_w * tile_w);
    size_t sps_wo = tmp_sps_wo < tile_w ? tmp_sps_wo : tile_w;

    // start idx along h/w of current tile
    size_t i_tho = i_tile_h * tile_h;
    size_t i_two = i_tile_w * tile_w;
    size_t i_thi = sy * i_tho - py;
    size_t i_twi = sx * i_two - px;

    // spatial-slice size of input, need further modify
    size_t sps_hi = (sps_ho - 1) * sy + 1 + dy * (fy - 1);
    size_t sps_wi = (sps_wo - 1) * sx + 1 + dx * (fx - 1);

    size_t tmp_sps_end_hi = sps_hi + i_thi;
    size_t tmp_sps_end_wi = sps_wi + i_twi;

    size_t sps_py = 0;  // left pad for each sec
    size_t sps_px = 0;  // left pad for each sec

    // modify input spatial-slice size according to left pad
    if(tmp_sps_end_hi < sps_hi){
        sps_py = sps_hi - tmp_sps_end_hi;
        sps_hi = tmp_sps_end_hi;
    }

    if(tmp_sps_end_wi < sps_wi){
        sps_px = sps_wi - tmp_sps_end_wi;
        sps_wi = tmp_sps_end_wi;
    }

    // modify input spatial-slice size according to right pad
    if(tmp_sps_end_hi > h)
        sps_hi -= tmp_sps_end_hi - h;

    if(tmp_sps_end_wi > w)
        sps_wi -= tmp_sps_end_wi - w;

    // tile index should start from 0
    if(i_thi < 0 || i_thi >= h)
        i_thi = 0;
    if(i_twi < 0 || i_twi >= w)
        i_twi = 0;

    // accumulate offset of each tile, and parse to tiled_conv
    p_src_t tile_src = src + i_thi * w + i_twi;
    p_dst_t tile_dst = dst + i_tho * wo + i_two;

    tiled_conv(tile_src, tile_dst, sps_hi, sps_wi, sps_ho, sps_wo, sps_py, sps_px);
    '''
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_spatial_tiling_t
        self.ctrl = ctrl

    def __call__(self, s_sps_in, s_tos_in, s_i_tile, s_size_in, s_size_out, s_tile, s_tmp2,
                s_sps_pad, s_sps_out, s_tos_out, s_filter, s_pad, s_stride, s_dilation):
        ctrl = self.ctrl
        s_sps_in        = sym_t(s_sps_in)   # current spatial-slice input size
        s_tos_in        = sym_t(s_tos_in)   # start offset along input image of this tile
        s_i_tile        = sym_t(s_i_tile)   # current index of tile
        s_size_in       = sym_t(s_size_in)  # original input image size
        s_size_out      = sym_t(s_size_out) # original output image size
        s_tile          = sym_t(s_tile)     # tile size to split this image
        s_tmp2          = sym_t(s_tmp2)
        if ctrl.nxe != 0:
            s_sps_pad   = sym_t(s_sps_pad)  # padding for current spatial-slice   -> require set zero before!
            s_sps_out   = sym_t(s_sps_out)  # current spatial-slice output size
            s_tos_out   = sym_t(s_tos_out)  # start offset along output image of this tile

            s_filter    = sym_t(s_filter)
            s_pad       = sym_t(s_pad)
            s_stride    = sym_t(s_stride)
            s_dilation  = sym_t(s_dilation)

        with self._deferred_context():
            self._emit(f"; calculate spatial tiling")
            if ctrl.nxe != 0:
                self._emit(f"s_mul_i32 s[{s_tos_out()}], s[{s_i_tile()}], s[{s_tile()}]")
                self._emit(f"s_mul_i32 s[{s_tmp2()}], s[{s_tos_out()}], s[{s_stride()}]")
                self._emit(f"s_sub_u32 s[{s_tos_in()}], s[{s_tmp2()}], s[{s_pad()}]")
            else:
                self._emit(f"s_mul_i32 s[{s_tos_in()}], s[{s_i_tile()}], s[{s_tile()}]")

            if ctrl.nxe != 0:
                self._emit(f"s_sub_u32 s[{s_sps_out()}], s[{s_size_out()}], s[{s_tos_out()}]")
                self._emit(f"s_cmp_ge_u32 s[{s_sps_out()}], s[{s_tile()}]")
                self._emit(f"s_cmov_b32 s[{s_sps_out()}], s[{s_tile()}]")

                self._emit(f"s_sub_u32 s[{s_tmp2()}], s[{s_sps_out()}], 1")
                self._emit(f"s_mul_i32 s[{s_sps_in()}], s[{s_tmp2()}], s[{s_stride()}]")
                self._emit(f"s_add_u32 s[{s_sps_in()}], s[{s_sps_in()}], 1")
                self._emit(f"s_sub_u32 s[{s_tmp2()}], s[{s_filter()}], 1")
                self._emit(f"s_mul_i32 s[{s_tmp2()}], s[{s_tmp2()}], s[{s_dilation()}]")
                self._emit(f"s_add_u32 s[{s_sps_in()}], s[{s_sps_in()}], s[{s_tmp2()}]")

                self._emit(f"s_add_u32 s[{s_tmp2(1)}], s[{s_sps_in()}], s[{s_tos_in()}]")   # tmp_sps_end_hi = sps_hi + i_thi;
                self._emit(f"s_sub_u32 s[{s_tmp2()}], s[{s_sps_in()}], s[{s_tmp2(1)}]")     # sps_hi - tmp_sps_end_hi;
                self._emit(f"s_cmp_lt_u32 s[{s_tmp2(1)}], s[{s_sps_in()}]")
                self._emit(f"s_cmov_b32 s[{s_sps_pad()}], s[{s_tmp2()}]")
                self._emit(f"s_cmov_b32 s[{s_sps_in()}], s[{s_tmp2(1)}]")

                self._emit(f"s_sub_u32 s[{s_tmp2()}], s[{s_tmp2(1)}], s[{s_size_in()}]")       # sps_hi -= tmp_sps_end_hi - h;
                self._emit(f"s_cmp_le_u32 s[{s_tmp2(1)}], s[{s_size_in()}]")
                self._emit(f"s_cmov_b32 s[{s_tmp2()}], 0")
                self._emit(f"s_sub_u32 s[{s_sps_in()}], s[{s_sps_in()}], s[{s_tmp2()}]")

                self._emit(f"s_cmp_ge_u32 s[{s_tos_in()}], s[{s_size_in()}]")
                self._emit(f"s_cmov_b32 s[{s_tos_in()}], 0")
                self._emit_empty_line()

            else:
                self._emit(f"s_sub_u32 s[{s_sps_in()}], s[{s_size_in()}], s[{s_tos_in()}]")
                self._emit(f"s_cmp_ge_u32 s[{s_sps_in()}], s[{s_tile()}]")
                self._emit(f"s_cmov_b32 s[{s_sps_in()}], s[{s_tile()}]")

        return self._get_deferred()
