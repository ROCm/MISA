/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef __NAIVE_TILED_CONV_H
#define __NAIVE_TILED_CONV_H

// implement convolution pre tiled in h-w
static inline size_t naive_tiled_conv_out_size(size_t in_size, size_t pad,
                                         size_t dilation, size_t ksize,
                                         size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

template<typename p_src_t, typename p_dst_t, typename tiled_conv_func_t>
void naive_2d_tiled_conv_iterator(
    p_src_t src, p_dst_t dst,
    size_t i_tile_h, size_t i_tile_w, size_t tile_h, size_t tile_w,
    size_t w, size_t h, size_t fx, size_t fy, size_t px, size_t py,
    size_t sx, size_t sy, size_t dx, size_t dy,
    tiled_conv_func_t tiled_conv)
{
    // 2d tiled iterator. each tile we call a section,
    // since each section size (h/w) may actually different from tile size,
    // because of original conv padding.
    // This iterator is responsible for calculating each sec size, and padding of each sec
    // ... to alow each tile becomea an independent conv, with h/w strided
    size_t ho = naive_tiled_conv_out_size(h, py, dy, fy, sy);
    size_t wo = naive_tiled_conv_out_size(w, px, dx, fx, sx);

    assert((tile_w <= wo) && (tile_h <= ho));

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

    // printf("tile_h:%lu, tile_w:%lu, sec_hi:%lu, sec_wi:%lu, sec_ho:%lu, sec_wo:%lu\n", tile_h, tile_w, sec_hi, sec_wi, sec_ho, sec_wo); fflush(stdout);

    // accumulate offset of each tile, and parse to tiled_conv
    p_src_t tile_src = src + i_thi * w + i_twi;
    p_dst_t tile_dst = dst + i_tho * wo + i_two;

    tiled_conv(tile_src, tile_dst, sec_hi, sec_wi, sec_ho, sec_wo, sec_py, sec_px);
}

static inline void naive_tiled_conv_fwd_nchw(const float *src, const float *filter,
                                       float *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx, size_t sy,
                                       size_t dx, size_t dy, size_t group,
                                       size_t tx, size_t ty)
{
    // tx, ty is used to tile output h, w
    size_t ho = naive_tiled_conv_out_size(h, py, dy, fy, sy);
    size_t wo = naive_tiled_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;

    assert((tx <= wo) && (ty <= ho));
    size_t tiles_w = (wo + tx - 1) / tx;
    size_t tiles_h = (ho + ty - 1) / ty;

    auto tiled_conv = [&](const float * tile_src, float * tile_dst,
                    size_t sec_hi, size_t sec_wi, size_t sec_ho, size_t sec_wo,
                    size_t sec_py, size_t sec_px){
        for (size_t ig = 0; ig < group; ig++) {
            for (size_t in = 0; in < n; in++) {
                for (size_t ik = 0; ik < k_per_group; ik++) {
                    for (size_t i_sho = 0; i_sho < sec_ho; i_sho++) {
                        for (size_t i_swo = 0; i_swo < sec_wo; i_swo++) {
                            double value = .0f;
                            size_t o_idx = in * k * ho * wo + ig * k_per_group * ho * wo + ik * ho * wo + i_sho * wo + i_swo;
                            for (size_t ic = 0; ic < c_per_group; ic++) {
                                for (size_t ir = 0; ir < fy; ir++) {
                                    size_t i_shi = sy * i_sho - sec_py + dy * ir;
                                    if (i_shi < 0 || i_shi >= sec_hi)
                                        continue;
                                    for (size_t is = 0; is < fx; is++) {
                                        size_t i_swi = sx * i_swo - sec_px + dx * is;
                                        if (i_swi < 0 || i_swi >= sec_wi)
                                            continue;
                                        size_t i_idx = in * c * h * w + ig * c_per_group * h * w + ic * h * w +
                                                i_shi * w + i_swi;
                                        size_t f_idx = ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx + ic * fy * fx +
                                                ir * fx + is;
                                        value += static_cast<double>(tile_src[i_idx]) * filter[f_idx];
                                    }
                                }
                            }
                            tile_dst[o_idx] = static_cast<float>(value);
                        }
                    }
                }
            }
        }
    };

    for(size_t i_tile_h = 0; i_tile_h < tiles_h; i_tile_h++){
        for(size_t i_tile_w = 0; i_tile_w < tiles_w; i_tile_w++){
            naive_2d_tiled_conv_iterator(
                src, dst,
                i_tile_h, i_tile_w, ty, tx,
                w, h, fx, fy, px, py,
                sx, sy, dx, dy,
                tiled_conv);
        }
    }
}

#endif
