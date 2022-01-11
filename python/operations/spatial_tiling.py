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
    // section size of output
    size_t tmp_sec_ho = ho - (i_tile_h * tile_h);
    size_t sec_ho = tmp_sec_ho < tile_h ? tmp_sec_ho : tile_h;
    size_t tmp_sec_wo = wo - (i_tile_w * tile_w);
    size_t sec_wo = tmp_sec_wo < tile_w ? tmp_sec_wo : tile_w;

    // start idx along h/w of current tile
    size_t i_tho = i_tile_h * tile_h;
    size_t i_two = i_tile_w * tile_w;
    size_t i_thi = sy * i_tho - py;
    size_t i_twi = sx * i_two - px;

    // section size of input, need further modify
    size_t sec_hi = (sec_ho - 1) * sy + 1 + dy * (fy - 1);
    size_t sec_wi = (sec_wo - 1) * sx + 1 + dx * (fx - 1);

    size_t tmp_sec_end_hi = sec_hi + i_thi;
    size_t tmp_sec_end_wi = sec_wi + i_twi;

    size_t sec_py = 0;  // left pad for each sec
    size_t sec_px = 0;  // left pad for each sec

    // modify input section size according to left pad
    if(tmp_sec_end_hi < sec_hi){
        sec_py = sec_hi - tmp_sec_end_hi;
        sec_hi = tmp_sec_end_hi;
    }

    if(tmp_sec_end_wi < sec_wi){
        sec_px = sec_wi - tmp_sec_end_wi;
        sec_wi = tmp_sec_end_wi;
    }

    // modify input section size according to right pad
    if(tmp_sec_end_hi > h)
        sec_hi -= tmp_sec_end_hi - h;

    if(tmp_sec_end_wi > w)
        sec_wi -= tmp_sec_end_wi - w;

    // tile index should start from 0
    if(i_thi < 0 || i_thi >= h)
        i_thi = 0;
    if(i_twi < 0 || i_twi >= w)
        i_twi = 0;

    // accumulate offset of each tile, and parse to tiled_conv
    p_src_t tile_src = src + i_thi * w + i_twi;
    p_dst_t tile_dst = dst + i_tho * wo + i_two;

    tiled_conv(tile_src, tile_dst, sec_hi, sec_wi, sec_ho, sec_wo, sec_py, sec_px);
    '''
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_spatial_tiling_t
        self.ctrl = ctrl

    def __call__(self, s_sec_in, s_tos_in, s_i_tile, s_size, s_tile, s_tmp2,
                s_sec_pad, s_sec_out, s_tos_out, s_filter, s_pad, s_stride, s_dilation):
        ctrl = self.ctrl
        s_sec_in        = sym_t(s_sec_in)   # current section input size
        s_tos_in        = sym_t(s_tos_in)   # start offset along input image of this tile
        s_i_tile        = sym_t(s_i_tile)   # current index of tile
        s_size          = sym_t(s_size)     # original image size
        s_tile          = sym_t(s_tile)     # tile size to split this image
        s_tmp2          = sym_t(s_tmp2)
        if ctrl.nxe != 0:
            s_sec_pad   = sym_t(s_sec_pad)  # padding for current section   -> require set zero before!
            s_sec_out   = sym_t(s_sec_out)  # current section output size
            s_tos_out   = sym_t(s_tos_out)  # start offset along output image of this tile

            s_filter    = sym_t(s_filter)
            s_pad       = sym_t(s_pad)
            s_stride    = sym_t(s_stride)
            s_dilation  = sym_t(s_dilation)

        with self._deferred_context():
            if ctrl.nxe != 0:
                self._emit(f"s_mul_i32 s[{s_tos_out()}], s[{s_i_tile()}], s[{s_tile()}]")
                self._emit(f"s_mul_i32 s[{s_tmp2()}], s[{s_tos_out()}], s[{s_stride()}]")
                self._emit(f"s_sub_u32 s[{s_tos_in()}], s[{s_tmp2()}], s[{s_pad()}]")
            else:
                self._emit(f"s_mul_i32 s[{s_tos_in()}], s[{s_i_tile()}], s[{s_tile()}]")

            if ctrl.nxe != 0:
                self._emit(f"s_sub_u32 s[{s_sec_out()}], s[{s_size()}], s[{s_tos_out()}]")
                self._emit(f"s_cmp_ge_u32 s[{s_sec_out()}], s[{s_tile()}]")
                self._emit(f"s_cmov_b32 s[{s_sec_out()}], s[{s_tile()}]")

                self._emit(f"s_sub_u32 s[{s_tmp2()}], s[{s_sec_out()}], 1")
                self._emit(f"s_mul_i32 s[{s_sec_in()}], s[{s_tmp2()}], s[{s_stride()}]")
                self._emit(f"s_add_u32 s[{s_sec_in()}], s[{s_sec_in()}], 1")
                self._emit(f"s_sub_u32 s[{s_tmp2()}], s[{s_filter()}], 1")
                self._emit(f"s_mul_i32 s[{s_tmp2()}], s[{s_tmp2()}], s[{s_dilation()}]")
                self._emit(f"s_add_u32 s[{s_sec_in()}], s[{s_sec_in()}], s[{s_tmp2()}]")

                self._emit(f"s_add_u32 s[{s_tmp2(1)}], s[{s_sec_in()}], s[{s_tos_in()}]")   # tmp_sec_end_hi = sec_hi + i_thi;
                self._emit(f"s_sub_u32 s[{s_tmp2()}], s[{s_sec_in()}], s[{s_tmp2(1)}]")     # sec_hi - tmp_sec_end_hi;
                self._emit(f"s_cmp_lt_u32 s[{s_tmp2(1)}], s[{s_sec_in()}]")
                self._emit(f"s_cmov_b32 s[{s_sec_pad()}], s[{s_tmp2()}]")
                self._emit(f"s_cmov_b32 s[{s_sec_in()}], s[{s_tmp2(1)}]")

                self._emit(f"s_sub_u32 s[{s_tmp2()}], s[{s_tmp2(1)}], s[{s_size()}]")       # sec_hi -= tmp_sec_end_hi - h;
                self._emit(f"s_cmp_le_u32 s[{s_tmp2(1)}], s[{s_size()}]")
                self._emit(f"s_cmov_b32 s[{s_tmp2()}], 0")
                self._emit(f"s_sub_u32 s[{s_sec_in()}], s[{s_sec_in()}], s[{s_tmp2()}]")

                self._emit(f"s_cmp_ge_u32 s[{s_tos_in()}], s[{s_size()}]")
                self._emit(f"s_cmov_b32 s[{s_tos_in()}], 0")

            else:
                self._emit(f"s_sub_u32 s[{s_sec_in()}], s[{s_size()}], s[{s_tos_in()}]")
                self._emit(f"s_cmp_ge_u32 s[{s_sec_in()}], s[{s_tile()}]")
                self._emit(f"s_cmov_b32 s[{s_sec_in()}], s[{s_tile()}]")

        return self._get_deferred()
