/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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
#ifndef __NAIVE_CONV_H
#define __NAIVE_CONV_H

// #define NAIVE_CONV_THREADED

#ifdef NAIVE_CONV_THREADED
// if use threaded conv, need c++11
#include <assert.h>
#include <thread>
#include <vector>
#include <functional>

using naive_conv_threadwise_conv_5d_t = std::function<void(size_t,size_t,size_t,size_t,size_t)>;

class naive_conv_blockwise_5d_t{
public:
    naive_conv_blockwise_5d_t(naive_conv_threadwise_conv_5d_t f):mf(f){}
    void operator()(size_t thread_id, size_t block_size,
        size_t d0, size_t d1, size_t d2, size_t d3, size_t d4)
    {
        size_t total_length = d0 * d1 * d2 * d3 * d4;
        for(size_t tid = thread_id; tid<total_length ; tid += block_size){
            size_t id0, id1, id2, id3, id4;
            get_index_5d(tid, d0, d1, d2, d3, d4, &id0, &id1, &id2, &id3, &id4);
            mf(id0, id1, id2, id3, id4);
        }
    }
private:
    naive_conv_threadwise_conv_5d_t mf;
    void get_index_5d(size_t idx, size_t d0, size_t d1, size_t d2, size_t d3, size_t d4,
                        size_t *id0, size_t *id1, size_t *id2, size_t *id3, size_t *id4)
    {
        *id4 = idx % d4;
        *id3 = (idx / d4) % d3;
        *id2 = (idx / (d3*d4)) % d2;
        *id1 = (idx / (d2*d3*d4)) % d1;
        *id0 = (idx / (d1*d2*d3*d4));
        (void)d0;
    }
};

using naive_conv_threadwise_conv_6d_t = std::function<void(size_t,size_t,size_t,size_t,size_t,size_t)>;

class naive_conv_blockwise_6d_t{
public:
    naive_conv_blockwise_6d_t(naive_conv_threadwise_conv_6d_t f):mf(f){}
    void operator()(size_t thread_id, size_t block_size,
        size_t d0, size_t d1, size_t d2, size_t d3, size_t d4, size_t d5)
    {
        size_t total_length = d0 * d1 * d2 * d3 * d4 * d5;
        for(size_t tid = thread_id; tid<total_length ; tid += block_size){
            size_t id0, id1, id2, id3, id4, id5;
            get_index_6d(tid, d0, d1, d2, d3, d4, d5, &id0, &id1, &id2, &id3, &id4, &id5);
            mf(id0, id1, id2, id3, id4, id5);
        }
    }
private:
    naive_conv_threadwise_conv_6d_t mf;
    void get_index_6d(size_t idx, size_t d0, size_t d1, size_t d2, size_t d3, size_t d4, size_t d5,
                        size_t *id0, size_t *id1, size_t *id2, size_t *id3, size_t *id4, size_t *id5)
    {
        *id5 = idx % d5;
        *id4 = (idx / d5) % d4;
        *id3 = (idx / (d4*d5)) % d3;
        *id2 = (idx / (d3*d4*d5)) % d2;
        *id1 = (idx / (d2*d3*d4*d5)) % d1;
        *id0 = (idx / (d1*d2*d3*d4*d5));
        (void)d0;
    }
};

template<class blockwise_t, class threadwise_t, class... args_t>
void naive_conv_blockwise_in_parallel(threadwise_t thread_func, args_t... args)
{
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    for(size_t tid = 0; tid < num_threads; tid++)
        threads[tid] = std::thread(blockwise_t(thread_func), tid, num_threads, args...);
    for(size_t tid = 0; tid < num_threads; tid++)
        threads[tid].join();
}

static inline void naive_conv_blockwise_in_parallel_5d(naive_conv_threadwise_conv_5d_t && thread_func,
    size_t d0, size_t d1, size_t d2, size_t d3, size_t d4){
    naive_conv_blockwise_in_parallel<naive_conv_blockwise_5d_t, naive_conv_threadwise_conv_5d_t,
            size_t, size_t, size_t, size_t, size_t>(thread_func, d0, d1, d2, d3, d4);
}

static inline void naive_conv_blockwise_in_parallel_6d(naive_conv_threadwise_conv_6d_t && thread_func,
    size_t d0, size_t d1, size_t d2, size_t d3, size_t d4, size_t d5){
    naive_conv_blockwise_in_parallel<naive_conv_blockwise_6d_t, naive_conv_threadwise_conv_6d_t,
            size_t, size_t, size_t, size_t, size_t, size_t>(thread_func, d0, d1, d2, d3, d4, d5);
}

#endif
static inline size_t naive_conv_out_size(size_t in_size, size_t pad,
                                         size_t dilation, size_t ksize,
                                         size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}
static inline void naive_conv_fwd_nchw(const float *src, const float *filter,
                                       float *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy, size_t group) {
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
#ifdef NAIVE_CONV_THREADED
    auto conv_one_pixel = [&](size_t ig, size_t in, size_t ik, size_t ioh, size_t iow){
        size_t ic, is, ir, cur_h, cur_w, o_idx, i_idx, f_idx;
        float value = .0f;
        o_idx = in * k * oh * ow + ig * k_per_group * oh * ow + ik * oh * ow + ioh * ow + iow;
        for (ic = 0; ic < c_per_group; ic++) {
            for (ir = 0; ir < fy; ir++) {
                cur_h = sy * ioh - py + dy * ir;
                if (cur_h < 0 || cur_h >= h)
                    continue;
                for (is = 0; is < fx; is++) {
                    cur_w = sx * iow - px + dx * is;
                    if (cur_w < 0 || cur_w >= w)
                        continue;
                    i_idx = in * c * h * w + ig * c_per_group * h * w + ic * h * w +
                            cur_h * w + cur_w;
                    f_idx = ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx + ic * fy * fx +
                            ir * fx + is;
                    value += src[i_idx] * filter[f_idx];
                }
            }
        }
        dst[o_idx] = value;
    };
    naive_conv_blockwise_in_parallel_5d(conv_one_pixel, group, n, k_per_group, oh, ow);
#else
    size_t ig, in, ik, ioh, iow, ic, is, ir;
    size_t cur_h, cur_w, o_idx, i_idx, f_idx;
    for (ig = 0; ig < group; ig++) {
        for (in = 0; in < n; in++) {
            for (ik = 0; ik < k_per_group; ik++) {
                for (ioh = 0; ioh < oh; ioh++) {
                    for (iow = 0; iow < ow; iow++) {
                        // sliding window for this filter
                        float value = .0f;
                        o_idx = in * k * oh * ow + ig * k_per_group * oh * ow + ik * oh * ow + ioh * ow + iow;
                        for (ic = 0; ic < c_per_group; ic++) {
                            for (ir = 0; ir < fy; ir++) {
                                cur_h = sy * ioh - py + dy * ir;
                                if (cur_h < 0 || cur_h >= h)
                                    continue;
                                for (is = 0; is < fx; is++) {
                                    cur_w = sx * iow - px + dx * is;
                                    if (cur_w < 0 || cur_w >= w)
                                        continue;
                                    i_idx = in * c * h * w + ig * c_per_group * h * w + ic * h * w +
                                            cur_h * w + cur_w;
                                    f_idx = ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx + ic * fy * fx +
                                            ir * fx + is;
                                    value += src[i_idx] * filter[f_idx];
                                }
                            }
                        }
                        dst[o_idx] = value;
                    }
                }
            }
        }
    }
#endif
}

static inline void naive_conv_bwd_nchw(float *src_grad, const float *filter,
                                         const float *dst_grad, size_t n,
                                         size_t w, size_t h, size_t c, size_t k,
                                         size_t fx, size_t fy, size_t px,
                                         size_t py, size_t sx, size_t sy,
                                         size_t dx, size_t dy, size_t group) {
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
#ifdef NAIVE_CONV_THREADED
    auto conv_one_pixel = [&](size_t ig, size_t in, size_t ic, size_t ih, size_t iw){
        size_t ik, is, ir;
        size_t cur_oh, cur_ow, o_idx, i_idx, f_idx;
        float value = .0f;
        i_idx = in * c * h * w + ig * c_per_group * h * w + ic * h * w + ih * w + iw;
        for (ik = 0; ik < k_per_group; ik++) {
            for (ir = 0; ir < fy; ir++) {
                cur_oh =
                    ih + py - dy * ir; // cur_h = sy*ioh-py+dy*ir;
                if (cur_oh < 0 || cur_oh % sy)
                    continue;
                cur_oh /= sy;
                if (cur_oh >= oh)
                    continue;
                for (is = 0; is < fx; is++) {
                    cur_ow = iw + px -
                                dx * is; // cur_w = sx*iow-px+dx*is;
                    if (cur_ow < 0 || cur_ow % sx)
                        continue;
                    cur_ow /= sx;
                    if (cur_ow >= ow)
                        continue;

                    o_idx = in * k * oh * ow + ig * k_per_group * oh * ow + ik * oh * ow + 
                            cur_oh * ow + cur_ow;
                    f_idx = ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx + ic * fy * fx +
                            ir * fx + is;

                    value += dst_grad[o_idx] * filter[f_idx];
                }
            }
        }
        src_grad[i_idx] = value;
    };
    // naive_conv_blockwise_in_parallel(conv_one_pixel, n, c, h, w);
    naive_conv_blockwise_in_parallel_5d(conv_one_pixel, group, n, c_per_group, h, w);
#else
    size_t ig, in, ik, ih, iw, ic, is, ir;
    size_t cur_oh, cur_ow, o_idx, i_idx, f_idx;

    for (ig = 0; ig < group; ig++) {
        for (in = 0; in < n; in++) {
            for (ic = 0; ic < c_per_group; ic++) {
                for (ih = 0; ih < h; ih++) {
                    for (iw = 0; iw < w; iw++) {
                        float value = .0f;
                        i_idx = in * c * h * w + ig * c_per_group * h * w + ic * h * w + ih * w + iw;
                        for (ik = 0; ik < k_per_group; ik++) {
                            for (ir = 0; ir < fy; ir++) {
                                cur_oh =
                                    ih + py - dy * ir; // cur_h = sy*ioh-py+dy*ir;
                                if (cur_oh < 0 || cur_oh % sy)
                                    continue;
                                cur_oh /= sy;
                                if (cur_oh >= oh)
                                    continue;
                                for (is = 0; is < fx; is++) {
                                    cur_ow = iw + px -
                                            dx * is; // cur_w = sx*iow-px+dx*is;
                                    if (cur_ow < 0 || cur_ow % sx)
                                        continue;
                                    cur_ow /= sx;
                                    if (cur_ow >= ow)
                                        continue;

                                    o_idx = in * k * oh * ow + ig * k_per_group * oh * ow + ik * oh * ow +
                                            cur_oh * ow + cur_ow;
                                    f_idx = ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx + ic * fy * fx +
                                            ir * fx + is;

                                    value += dst_grad[o_idx] * filter[f_idx];
                                }
                            }
                        }
                        src_grad[i_idx] = value;
                    }
                }
            }
        }
    }
#endif
}

static inline void naive_conv_wrw_nchw(const float *src, float *filter_grad,
                                         const float *dst_grad, size_t n,
                                         size_t w, size_t h, size_t c, size_t k,
                                         size_t fx, size_t fy, size_t px,
                                         size_t py, size_t sx, size_t sy,
                                         size_t dx, size_t dy, size_t group) {
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
#ifdef NAIVE_CONV_THREADED
    auto conv_one_pixel = [&](size_t ig, size_t ik, size_t ic, size_t ir, size_t is){
        size_t in, ioh, iow;
        size_t cur_h, cur_w, o_idx, i_idx, f_idx;
        float value = .0f;
        f_idx = ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx + ic * fy * fx + ir * fx + is;
        for (in = 0; in < n; in++) {
            for (ioh = 0; ioh < oh; ioh++) {
                cur_h = sy * ioh - py + dy * ir;
                if (cur_h < 0 || cur_h >= h)
                    continue;
                for (iow = 0; iow < ow; iow++) {
                    cur_w = sx * iow - px + dx * is;
                    if (cur_w < 0 || cur_w >= w)
                        continue;
                    i_idx = in * c * h * w + ig * c_per_group * h * w + ic * h * w +
                            cur_h * w + cur_w;
                    o_idx = in * k * oh * ow + ig * k_per_group * oh * ow + ik * oh * ow +
                            ioh * ow + iow;
                    value += src[i_idx] * dst_grad[o_idx];
                }
            }
        }
        filter_grad[f_idx] = value;
    };
    // naive_conv_blockwise_in_parallel(conv_one_pixel, k, c, fy, fx);
    naive_conv_blockwise_in_parallel_5d(conv_one_pixel, group, k_per_group, c_per_group, fy, fx);
#else
    size_t ig, in, ik, ioh, iow, ic, is, ir;
    size_t cur_h, cur_w, o_idx, i_idx, f_idx;

    for (ig = 0; ig < group; ig++) {
        for (ik = 0; ik < k_per_group; ik++) {
            for (ic = 0; ic < c_per_group; ic++) {
                for (ir = 0; ir < fy; ir++) {
                    for (is = 0; is < fx; is++) {
                        float value = .0f;
                        f_idx = ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx + ic * fy * fx + ir * fx + is;
                        for (in = 0; in < n; in++) {
                            for (ioh = 0; ioh < oh; ioh++) {
                                cur_h = sy * ioh - py + dy * ir;
                                if (cur_h < 0 || cur_h >= h)
                                    continue;
                                for (iow = 0; iow < ow; iow++) {
                                    cur_w = sx * iow - px + dx * is;
                                    if (cur_w < 0 || cur_w >= w)
                                        continue;
                                    i_idx = in * c * h * w + ig * c_per_group * h * w + ic * h * w +
                                            cur_h * w + cur_w;
                                    o_idx = in * k * oh * ow + ig * k_per_group * oh * ow + ik * oh * ow +
                                            ioh * ow + iow;
                                    value += src[i_idx] * dst_grad[o_idx];
                                }
                            }
                        }
                        filter_grad[f_idx] = value;
                    }
                }
            }
        }
    }
#endif
}


static inline void naive_conv_fwd_ncdhw(const float *src, const float *filter, float *dst,
                                       size_t n, size_t w, size_t h, size_t d, size_t c, size_t k,
                                       size_t fx, size_t fy, size_t fz, size_t px, size_t py, size_t pz,
                                       size_t sx, size_t sy, size_t sz, size_t dx, size_t dy, size_t dz, size_t group) {
    size_t od = naive_conv_out_size(d, pz, dz, fz, sz);
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
#ifdef NAIVE_CONV_THREADED
    auto conv_one_pixel = [&](size_t ig, size_t in, size_t ik, size_t iod, size_t ioh, size_t iow){
        size_t ic, iz, is, ir, cur_d, cur_h, cur_w, o_idx, i_idx, f_idx;
        float value = .0f;
        o_idx = in * k * od * oh * ow + ig * k_per_group * od * oh * ow + ik * od * oh * ow + iod * oh * ow + ioh * ow + iow;
        for (ic = 0; ic < c_per_group; ic++) {
            for (iz = 0; iz < fz; iz++) {
                cur_d = sz * iod - pz + dz * iz;
                if (cur_d < 0 || cur_d >= d)
                    continue;
                for (ir = 0; ir < fy; ir++) {
                    cur_h = sy * ioh - py + dy * ir;
                    if (cur_h < 0 || cur_h >= h)
                        continue;
                    for (is = 0; is < fx; is++) {
                        cur_w = sx * iow - px + dx * is;
                        if (cur_w < 0 || cur_w >= w)
                            continue;
                        i_idx = in * c * d * h * w + ig * c_per_group * d * h * w + ic * d * h * w + cur_d * h * w + cur_h * w + cur_w;
                        f_idx = ig * k_per_group * c_per_group * fz * fy * fx + ik * c_per_group * fz * fy * fx + ic * fz * fy * fx + iz * fy * fx + ir * fx + is;
                        value += src[i_idx] * filter[f_idx];
                    }
                }
            }
        }
        dst[o_idx] = value;
    };
    naive_conv_blockwise_in_parallel_6d(conv_one_pixel, group, n, k_per_group, od, oh, ow);
#else
    size_t ig, in, ik, iod, ioh, iow, ic, iz, is, ir;
    size_t cur_d, cur_h, cur_w, o_idx, i_idx, f_idx;
    for (ig = 0; ig < group; ig++) {
        for (in = 0; in < n; in++) {
            for (ik = 0; ik < k_per_group; ik++) {
                for (iod = 0; iod < od; iod++) {
                    for (ioh = 0; ioh < oh; ioh++) {
                        for (iow = 0; iow < ow; iow++) {
                            // sliding window for this filter
                            float value = .0f;
                            o_idx = in * k * od * oh * ow + ig * k_per_group * od * oh * ow + ik * od * oh * ow + iod * oh * ow + ioh * ow + iow;
                            for (ic = 0; ic < c_per_group; ic++) {
                                for (iz = 0; iz < fz; iz++) {
                                    cur_d = sz * iod - pz + dz * iz;
                                    if (cur_d < 0 || cur_d >= d)
                                        continue;
                                    for (ir = 0; ir < fy; ir++) {
                                        cur_h = sy * ioh - py + dy * ir;
                                        if (cur_h < 0 || cur_h >= h)
                                            continue;
                                        for (is = 0; is < fx; is++) {
                                            cur_w = sx * iow - px + dx * is;
                                            if (cur_w < 0 || cur_w >= w)
                                                continue;
                                            i_idx = in * c * d * h * w + ig * c_per_group * d * h * w + ic * d * h * w + cur_d * h * w + cur_h * w + cur_w;
                                            f_idx = ig * k_per_group * c_per_group * fz * fy * fx + ik * c_per_group * fz * fy * fx + ic * fz * fy * fx + iz * fy * fx + ir * fx + is;
                                            value += src[i_idx] * filter[f_idx];
                                        }
                                    }
                                }
                            }
                            dst[o_idx] = value;
                        }
                    }
                }
            }
        }
    }
#endif
}

static inline void naive_conv_bwd_ncdhw(float *src_grad, const float *filter, const float *dst_grad,
                                       size_t n, size_t w, size_t h, size_t d, size_t c, size_t k,
                                       size_t fx, size_t fy, size_t fz, size_t px, size_t py, size_t pz,
                                       size_t sx, size_t sy, size_t sz, size_t dx, size_t dy, size_t dz, size_t group) {
    size_t od = naive_conv_out_size(d, pz, dz, fz, sz);
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
#ifdef NAIVE_CONV_THREADED
    auto conv_one_pixel = [&](size_t ig, size_t in, size_t ic, size_t id, size_t ih, size_t iw){
        size_t ik, iz, is, ir;
        size_t cur_od, cur_oh, cur_ow, o_idx, i_idx, f_idx;
        float value = .0f;
        i_idx = in * c * d * h * w + ig * c_per_group * d * h * w + ic * d * h * w + id * h * w + ih * w + iw;
        for (ik = 0; ik < k_per_group; ik++) {
            for (iz = 0; iz < fz; iz++) {
                cur_od = id + pz - dz * iz;
                if (cur_od < 0 || cur_od % sz)
                    continue;
                cur_od /= sz;
                if (cur_od >= od)
                    continue;
                for (ir = 0; ir < fy; ir++) {
                    cur_oh = ih + py - dy * ir; // cur_h = sy*ioh-py+dy*ir;
                    if (cur_oh < 0 || cur_oh % sy)
                        continue;
                    cur_oh /= sy;
                    if (cur_oh >= oh)
                        continue;
                    for (is = 0; is < fx; is++) {
                        cur_ow = iw + px - dx * is; // cur_w = sx*iow-px+dx*is;
                        if (cur_ow < 0 || cur_ow % sx)
                            continue;
                        cur_ow /= sx;
                        if (cur_ow >= ow)
                            continue;

                        o_idx = in * k * od * oh * ow + ig * k_per_group * od * oh * ow + ik * od * oh * ow + cur_od * oh * ow +
                                cur_oh * ow + cur_ow;
                        f_idx = ig * k_per_group * c_per_group * fz * fy * fx + ik * c_per_group * fz * fy * fx + ic * fz * fy * fx + iz * fy * fx +
                                ir * fx + is;

                        value += dst_grad[o_idx] * filter[f_idx];
                    }
                }
            }
        }
        src_grad[i_idx] = value;
    };
    naive_conv_blockwise_in_parallel_6d(conv_one_pixel, group, n, c_per_group, d, h, w);
#else
    size_t ig, in, ik, id, ih, iw, ic, iz, is, ir;
    size_t cur_od, cur_oh, cur_ow, o_idx, i_idx, f_idx;

    for (ig = 0; ig < group; ig++) {
        for (in = 0; in < n; in++) {
            for (ic = 0; ic < c_per_group; ic++) {
                for (id = 0; id < d; id++) {
                    for (ih = 0; ih < h; ih++) {
                        for (iw = 0; iw < w; iw++) {
                            float value = .0f;
                            i_idx = in * c * d *h * w + ig * c_per_group * d *h * w + ic * d *h * w + id * h * w + ih * w + iw;
                            for (ik = 0; ik < k_per_group; ik++) {
                                for (iz = 0; iz < fz; iz++) {
                                    cur_od = id + pz - dz * iz;
                                    if (cur_od < 0 || cur_od % sz)
                                        continue;
                                    cur_od /= sz;
                                    if (cur_od >= od)
                                        continue;
                                    for (ir = 0; ir < fy; ir++) {
                                        cur_oh = ih + py - dy * ir; // cur_h = sy*ioh-py+dy*ir;
                                        if (cur_oh < 0 || cur_oh % sy)
                                            continue;
                                        cur_oh /= sy;
                                        if (cur_oh >= oh)
                                            continue;
                                        for (is = 0; is < fx; is++) {
                                            cur_ow = iw + px - dx * is; // cur_w = sx*iow-px+dx*is;
                                            if (cur_ow < 0 || cur_ow % sx)
                                                continue;
                                            cur_ow /= sx;
                                            if (cur_ow >= ow)
                                                continue;

                                            o_idx = in * k * od * oh * ow + ig * k_per_group * od * oh * ow + ik * od * oh * ow + cur_od * oh * ow +
                                                    cur_oh * ow + cur_ow;
                                            f_idx = ig * k_per_group * c_per_group * fz * fy * fx + ik * c_per_group * fz * fy * fx + ic * fz * fy * fx + iz * fy * fx +
                                                    ir * fx + is;

                                            value += dst_grad[o_idx] * filter[f_idx];
                                        }
                                    }
                                }
                            }
                            src_grad[i_idx] = value;
                        }
                    }
                }
            }
        }
    }
#endif
}

static inline void naive_conv_wrw_ncdhw(const float *src, float *filter_grad, const float *dst_grad,
                                       size_t n, size_t w, size_t h, size_t d, size_t c, size_t k,
                                       size_t fx, size_t fy, size_t fz, size_t px, size_t py, size_t pz,
                                       size_t sx, size_t sy, size_t sz, size_t dx, size_t dy, size_t dz, size_t group) {
    size_t od = naive_conv_out_size(d, pz, dz, fz, sz);
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
#ifdef NAIVE_CONV_THREADED
    auto conv_one_pixel = [&](size_t ig, size_t ik, size_t ic, size_t iz, size_t ir, size_t is){
        size_t in, iod, ioh, iow;
        size_t cur_d, cur_h, cur_w, o_idx, i_idx, f_idx;
        float value = .0f;
        f_idx = ig * k_per_group * c_per_group * fz * fy * fx + ik * c_per_group * fz * fy * fx + ic * fz * fy * fx + iz * fy * fx + ir * fx + is;
        for (in = 0; in < n; in++) {
            for (iod = 0; iod < od; iod++) {
                cur_d = sz * iod - pz + dz * iz;
                if (cur_d < 0 || cur_d >= d)
                    continue;
                for (ioh = 0; ioh < oh; ioh++) {
                    cur_h = sy * ioh - py + dy * ir;
                    if (cur_h < 0 || cur_h >= h)
                        continue;
                    for (iow = 0; iow < ow; iow++) {
                        cur_w = sx * iow - px + dx * is;
                        if (cur_w < 0 || cur_w >= w)
                            continue;
                        i_idx = in * c * d * h * w + ig * c_per_group * d * h * w + ic * d * h * w + cur_d * h * w +
                                cur_h * w + cur_w;
                        o_idx = in * k * od * oh * ow + ig * k_per_group * od * oh * ow + ik * od * oh * ow + iod * oh * ow +
                                ioh * ow + iow;
                        value += src[i_idx] * dst_grad[o_idx];
                    }
                }
            }
        }
        filter_grad[f_idx] = value;
    };
    naive_conv_blockwise_in_parallel_6d(conv_one_pixel, group, k_per_group, c_per_group, fz, fy, fx);
#else
    size_t ig, in, ik, iod, ioh, iow, ic, iz, is, ir;
    size_t cur_d, cur_h, cur_w, o_idx, i_idx, f_idx;

    for (ig = 0; ig < group; ig++) {
        for (ik = 0; ik < k_per_group; ik++) {
            for (ic = 0; ic < c_per_group; ic++) {
                for (iz = 0; iz < fz; iz++) {
                    for (ir = 0; ir < fy; ir++) {
                        for (is = 0; is < fx; is++) {
                            float value = .0f;
                            f_idx = ig * k_per_group * c_per_group * fz * fy * fx + ik * c_per_group * fz * fy * fx + ic * fz * fy * fx + iz * fy * fx + ir * fx + is;
                            for (in = 0; in < n; in++) {
                                for (iod = 0; iod < od; iod++) {
                                    cur_d = sz * iod - pz + dz * iz;
                                    if (cur_d < 0 || cur_d >= d)
                                        continue;
                                    for (ioh = 0; ioh < oh; ioh++) {
                                        cur_h = sy * ioh - py + dy * ir;
                                        if (cur_h < 0 || cur_h >= h)
                                            continue;
                                        for (iow = 0; iow < ow; iow++) {
                                            cur_w = sx * iow - px + dx * is;
                                            if (cur_w < 0 || cur_w >= w)
                                                continue;
                                            i_idx = in * c * d * h * w + ig * c_per_group * d * h * w + ic * d * h * w + cur_d * h * w +
                                                    cur_h * w + cur_w;
                                            o_idx = in * k * od * oh * ow + ig * k_per_group * od * oh * ow + ik * od * oh * ow + iod * oh * ow +
                                                    ioh * ow + iow;
                                            value += src[i_idx] * dst_grad[o_idx];
                                        }
                                    }
                                }
                            }
                            filter_grad[f_idx] = value;
                        }
                    }
                }
            }
        }
    }
#endif
}

/************************** nhwc ****************************/
static inline void naive_conv_fwd_nhwc(const float *src, const float *filter,
                                       float *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy, size_t group) {
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
#ifdef NAIVE_CONV_THREADED
    auto conv_one_pixel = [&](size_t ig, size_t in, size_t ioh, size_t iow, size_t ik){
        size_t ic, is, ir, cur_h, cur_w, o_idx, i_idx, f_idx;
        float value = .0f;
        o_idx = in * oh * ow * k + ioh * ow * k + iow * k + ig * k_per_group + ik;
        for (ir = 0; ir < fy; ir++) {
            cur_h = sy * ioh - py + dy * ir;
            if (cur_h < 0 || cur_h >= h)
                continue;
            for (is = 0; is < fx; is++) {
                cur_w = sx * iow - px + dx * is;
                if (cur_w < 0 || cur_w >= w)
                    continue;
                for (ic = 0; ic < c_per_group; ic++) {
                    i_idx = in * h * w * c + cur_h * w * c + cur_w * c + ig * c_per_group + ic;
                    f_idx = ig * k_per_group * fy * fx * c_per_group + ik * fy * fx * c_per_group +
                            ir * fx * c_per_group + is * c_per_group + ic;
                    value += src[i_idx] * filter[f_idx];
                }
            }
        }
        dst[o_idx] = value;
    };
    naive_conv_blockwise_in_parallel_5d(conv_one_pixel, group, n, oh, ow, k_per_group);
#else
    size_t ig, in, ik, ioh, iow, ic, is, ir;
    size_t cur_h, cur_w, o_idx, i_idx, f_idx;
    for (ig = 0; ig < group; ig++) {
        for (in = 0; in < n; in++) {
            for (ioh = 0; ioh < oh; ioh++) {
                for (iow = 0; iow < ow; iow++) {
                    for (ik = 0; ik < k_per_group; ik++) {
                        // sliding window for this filter
                        float value = .0f;
                        o_idx = in * oh * ow * k + + ioh * ow * k_per_group + iow * k_per_group + ig * k_per_group + ik;
                        for (ir = 0; ir < fy; ir++) {
                            cur_h = sy * ioh - py + dy * ir;
                            if (cur_h < 0 || cur_h >= h)
                                continue;
                            for (is = 0; is < fx; is++) {
                                cur_w = sx * iow - px + dx * is;
                                if (cur_w < 0 || cur_w >= w)
                                    continue;
                                for (ic = 0; ic < c_per_group; ic++) {
                                    i_idx = in * h * w * c + cur_h * w * c + cur_w * c + ig * c_per_group + ic;
                                    f_idx = ig * k_per_group * fy * fx * c_per_group + ik * fy * fx * c_per_group +
                                                            ir * fx * c_per_group + is * c_per_group + ic;
                                    value += src[i_idx] * filter[f_idx];
                                }
                            }
                        }
                        dst[o_idx] = value;
                    }
                }
            }
        }
    }
#endif
}

static inline void naive_conv_bwd_nhwc(float *src_grad, const float *filter,
                                         const float *dst_grad, size_t n,
                                         size_t w, size_t h, size_t c, size_t k,
                                         size_t fx, size_t fy, size_t px,
                                         size_t py, size_t sx, size_t sy,
                                         size_t dx, size_t dy, size_t group) {
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
#ifdef NAIVE_CONV_THREADED
    auto conv_one_pixel = [&](size_t ig, size_t in, size_t ih, size_t iw, size_t ic){
        size_t ik, is, ir;
        size_t cur_oh, cur_ow, o_idx, i_idx, f_idx;
        float value = .0f;
        i_idx = in * h * w * c + ih * w * c + iw * c + ig * c_per_group + ic;
        for (ir = 0; ir < fy; ir++) {
            cur_oh = ih + py - dy * ir; // cur_h = sy*ioh-py+dy*ir;
            if (cur_oh < 0 || cur_oh % sy)
                continue;
            cur_oh /= sy;
            if (cur_oh >= oh)
                continue;
            for (is = 0; is < fx; is++) {
                cur_ow = iw + px - dx * is; // cur_w = sx*iow-px+dx*is;
                if (cur_ow < 0 || cur_ow % sx)
                    continue;
                cur_ow /= sx;
                if (cur_ow >= ow)
                    continue;
                for (ik = 0; ik < k_per_group; ik++) {
                    o_idx = in * oh * ow * k + cur_oh * ow * k + cur_ow * k + ig * k_per_group + ik;
                    f_idx = ig * k_per_group * fy * fx * c_per_group + ik * fy * fx * c_per_group + ir * fx * c_per_group + is * c_per_group + ic;
                    value += dst_grad[o_idx] * filter[f_idx];
                }
            }
        }
        src_grad[i_idx] = value;
    };
    naive_conv_blockwise_in_parallel_5d(conv_one_pixel, group, n, h, w, c_per_group);
#else
    size_t ig, in, ik, ih, iw, ic, is, ir;
    size_t cur_oh, cur_ow, o_idx, i_idx, f_idx;

    for (ig = 0; ig < group; ig++) {
        for (in = 0; in < n; in++) {
            for (ih = 0; ih < h; ih++) {
                for (iw = 0; iw < w; iw++) {
                    for (ic = 0; ic < c_per_group; ic++) {
                        float value = .0f;
                        i_idx = in * h * w * c + ih * w * c + iw * c + ig * c_per_group + ic;
                        for (ir = 0; ir < fy; ir++) {
                            cur_oh = ih + py - dy * ir; // cur_h = sy*ioh-py+dy*ir;
                            if (cur_oh < 0 || cur_oh % sy)
                                continue;
                            cur_oh /= sy;
                            if (cur_oh >= oh)
                                continue;
                            for (is = 0; is < fx; is++) {
                                cur_ow = iw + px - dx * is; // cur_w = sx*iow-px+dx*is;
                                if (cur_ow < 0 || cur_ow % sx)
                                    continue;
                                cur_ow /= sx;
                                if (cur_ow >= ow)
                                    continue;
                                for (ik = 0; ik < k_per_group; ik++) {
                                    o_idx = in * oh * ow * k + cur_oh * ow * k + cur_ow * k + ig * k_per_group + ik;
                                    f_idx = ig * k_per_group * fy * fx * c_per_group + ik * fy * fx * c_per_group + ir * fx * c_per_group + is * c_per_group + ic;
                                    value += dst_grad[o_idx] * filter[f_idx];
                                }
                            }
                        }
                        src_grad[i_idx] = value;
                    }
                }
            }
        }
    }
#endif
}

static inline void naive_conv_wrw_nhwc(const float *src, float *filter_grad,
                                         const float *dst_grad, size_t n,
                                         size_t w, size_t h, size_t c, size_t k,
                                         size_t fx, size_t fy, size_t px,
                                         size_t py, size_t sx, size_t sy,
                                         size_t dx, size_t dy, size_t group) {
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
#ifdef NAIVE_CONV_THREADED
    auto conv_one_pixel = [&](size_t ig, size_t ik, size_t ir, size_t is, size_t ic){
        size_t in, ioh, iow;
        size_t cur_h, cur_w, o_idx, i_idx, f_idx;
        float value = .0f;
        f_idx = ig * k_per_group * fy * fx * c_per_group + ik * fy * fx * c_per_group + ir * fx * c_per_group + is * c_per_group + ic;
        for (in = 0; in < n; in++) {
            for (ioh = 0; ioh < oh; ioh++) {
                cur_h = sy * ioh - py + dy * ir;
                if (cur_h < 0 || cur_h >= h)
                    continue;
                for (iow = 0; iow < ow; iow++) {
                    cur_w = sx * iow - px + dx * is;
                    if (cur_w < 0 || cur_w >= w)
                        continue;
                    i_idx = in * h * w * c + cur_h * w * c + cur_w * c + ig * c_per_group + ic;
                    o_idx = in * oh * ow * k + ioh * ow * k + iow * k + ig * k_per_group + ik;
                    value += src[i_idx] * dst_grad[o_idx];
                }
            }
        }
        filter_grad[f_idx] = value;
    };
    naive_conv_blockwise_in_parallel_5d(conv_one_pixel, group, k_per_group, fy, fx, c_per_group);
#else
    size_t ig, in, ik, ioh, iow, ic, is, ir;
    size_t cur_h, cur_w, o_idx, i_idx, f_idx;
    for (ig = 0; ig < group; ig++) {
        for (ik = 0; ik < k_per_group; ik++) {
            for (ir = 0; ir < fy; ir++) {
                for (is = 0; is < fx; is++) {
                    for (ic = 0; ic < c_per_group; ic++) {
                        float value = .0f;
                        f_idx = ig * k_per_group * fy * fx * c_per_group + ik * fy * fx * c_per_group + ir * fx * c_per_group + is * c_per_group + ic;
                        for (in = 0; in < n; in++) {
                            for (ioh = 0; ioh < oh; ioh++) {
                                cur_h = sy * ioh - py + dy * ir;
                                if (cur_h < 0 || cur_h >= h)
                                    continue;
                                for (iow = 0; iow < ow; iow++) {
                                    cur_w = sx * iow - px + dx * is;
                                    if (cur_w < 0 || cur_w >= w)
                                        continue;
                                    i_idx = in * h * w * c + cur_h * w * c + cur_w * c + ig * c_per_group + ic;
                                    o_idx = in * oh * ow * k + ioh * ow * k + iow * k + ig * k_per_group + ik;
                                    value += src[i_idx] * dst_grad[o_idx];
                                }
                            }
                        }
                        filter_grad[f_idx] = value;
                    }
                }
            }
        }
    }
#endif
}

static inline void naive_conv_fwd_ndhwc(const float *src, const float *filter, float *dst,
                                       size_t n, size_t w, size_t h, size_t d, size_t c, size_t k,
                                       size_t fx, size_t fy, size_t fz, size_t px, size_t py, size_t pz,
                                       size_t sx, size_t sy, size_t sz, size_t dx, size_t dy, size_t dz, size_t group) {
    size_t od = naive_conv_out_size(d, pz, dz, fz, sz);
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
#ifdef NAIVE_CONV_THREADED
    auto conv_one_pixel = [&](size_t ig, size_t in, size_t iod, size_t ioh, size_t iow, size_t ik){
        size_t ic, iz, is, ir, cur_d, cur_h, cur_w, o_idx, i_idx, f_idx;
        float value = .0f;
        o_idx = in * od * oh * ow * k + iod * oh * ow * k + ioh * ow * k + iow * k + ig * k_per_group + ik;
        for (iz = 0; iz < fz; iz++) {
            cur_d = sz * iod - pz + dz * iz;
            if (cur_d < 0 || cur_d >= d)
                continue;
            for (ir = 0; ir < fy; ir++) {
                cur_h = sy * ioh - py + dy * ir;
                if (cur_h < 0 || cur_h >= h)
                    continue;
                for (is = 0; is < fx; is++) {
                    cur_w = sx * iow - px + dx * is;
                    if (cur_w < 0 || cur_w >= w)
                        continue;
                    for (ic = 0; ic < c_per_group; ic++) {
                        i_idx = in * d * h * w * c + cur_d * h * w * c + cur_h * w * c + cur_w * c + ig * c_per_group + ic;
                        f_idx = ig * k_per_group * fz * fy * fx * c_per_group + ik * fz * fy * fx * c_per_group + iz * fy * fx * c_per_group + ir * fx * c_per_group + is * c_per_group + ic;
                        value += src[i_idx] * filter[f_idx];
                    }
                }
            }
        }
        dst[o_idx] = value;
    };
    naive_conv_blockwise_in_parallel_6d(conv_one_pixel, group, n, od, oh, ow, k_per_group);
#else
    size_t ig, in, ik, iod, ioh, iow, ic, iz, is, ir;
    size_t cur_d, cur_h, cur_w, o_idx, i_idx, f_idx;
    for (ig = 0; ig < group; ig++) {
        for (in = 0; in < n; in++) {
            for (iod = 0; iod < od; iod++) {
                for (ioh = 0; ioh < oh; ioh++) {
                    for (iow = 0; iow < ow; iow++) {
                        for (ik = 0; ik < k_per_group; ik++) {
                            // sliding window for this filter
                            float value = .0f;
                            o_idx = in * od * oh * ow * k + iod * oh * ow * k + ioh * ow * k + iow * k + ig * k_per_group + ik;
                            for (iz = 0; iz < fz; iz++) {
                                cur_d = sz * iod - pz + dz * iz;
                                if (cur_d < 0 || cur_d >= d)
                                    continue;
                                for (ir = 0; ir < fy; ir++) {
                                    cur_h = sy * ioh - py + dy * ir;
                                    if (cur_h < 0 || cur_h >= h)
                                        continue;
                                    for (is = 0; is < fx; is++) {
                                        cur_w = sx * iow - px + dx * is;
                                        if (cur_w < 0 || cur_w >= w)
                                            continue;
                                        for (ic = 0; ic < c_per_group; ic++) {
                                            i_idx = in * d * h * w * c + cur_d * h * w * c + cur_h * w * c + cur_w * c + ig * c_per_group + ic;
                                            f_idx = ig * k_per_group * fz * fy * fx * c_per_group + ik * fz * fy * fx * c_per_group + iz * fy * fx * c_per_group + ir * fx * c_per_group + is * c_per_group + ic;
                                            value += src[i_idx] * filter[f_idx];
                                        }
                                    }
                                }
                            }
                            dst[o_idx] = value;
                        }
                    }
                }
            }
        }
    }
#endif
}

static inline void naive_conv_bwd_ndhwc(float *src_grad, const float *filter, const float *dst_grad,
                                       size_t n, size_t w, size_t h, size_t d, size_t c, size_t k,
                                       size_t fx, size_t fy, size_t fz, size_t px, size_t py, size_t pz,
                                       size_t sx, size_t sy, size_t sz, size_t dx, size_t dy, size_t dz, size_t group) {
    size_t od = naive_conv_out_size(d, pz, dz, fz, sz);
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
#ifdef NAIVE_CONV_THREADED
    auto conv_one_pixel = [&](size_t ig, size_t in, size_t id, size_t ih, size_t iw, size_t ic){
        size_t ik, iz, is, ir;
        size_t cur_od, cur_oh, cur_ow, o_idx, i_idx, f_idx;
        float value = .0f;
        i_idx = in * d * h * w * c + id * h * w * c + ih * w * c + iw * c + ig * c_per_group + ic;
        for (iz = 0; iz < fz; iz++) {
            cur_od = id + pz - dz * iz;
            if (cur_od < 0 || cur_od % sz)
                continue;
            cur_od /= sz;
            if (cur_od >= od)
                continue;
            for (ir = 0; ir < fy; ir++) {
                cur_oh = ih + py - dy * ir; // cur_h = sy*ioh-py+dy*ir;
                if (cur_oh < 0 || cur_oh % sy)
                    continue;
                cur_oh /= sy;
                if (cur_oh >= oh)
                    continue;
                for (is = 0; is < fx; is++) {
                    cur_ow = iw + px - dx * is; // cur_w = sx*iow-px+dx*is;
                    if (cur_ow < 0 || cur_ow % sx)
                        continue;
                    cur_ow /= sx;
                    if (cur_ow >= ow)
                        continue;
                    for (ik = 0; ik < k_per_group; ik++) {
                        o_idx = in * od * oh * ow * k + cur_od * oh * ow * k + cur_oh * ow * k + cur_ow * k + ig * k_per_group + ik;
                        f_idx = ig * k_per_group * fz * fy * fx * c_per_group + ik * fz * fy * fx * c_per_group + iz * fy * fx * c_per_group + ir * fx * c_per_group + is * c_per_group + ic;

                        value += dst_grad[o_idx] * filter[f_idx];
                    }
                }
            }
        }
        src_grad[i_idx] = value;
    };
    naive_conv_blockwise_in_parallel_6d(conv_one_pixel, group, n, d, h, w, c_per_group);
#else
    size_t ig, in, ik, id, ih, iw, ic, iz, is, ir;
    size_t cur_od, cur_oh, cur_ow, o_idx, i_idx, f_idx;

    for (ig = 0; ig < group; ig++) {
        for (in = 0; in < n; in++) { 
            for (id = 0; id < d; id++) {
                for (ih = 0; ih < h; ih++) {
                    for (iw = 0; iw < w; iw++) {
                        for (ic = 0; ic < c_per_group; ic++) {
                            float value = .0f;
                            i_idx = in * d * h * w * c + id * h * w * c + ih * w * c + iw * c + ig * c_per_group + ic;
                            for (iz = 0; iz < fz; iz++) {
                                cur_od = id + pz - dz * iz;
                                if (cur_od < 0 || cur_od % sz)
                                    continue;
                                cur_od /= sz;
                                if (cur_od >= od)
                                    continue;
                                for (ir = 0; ir < fy; ir++) {
                                    cur_oh = ih + py - dy * ir; // cur_h = sy*ioh-py+dy*ir;
                                    if (cur_oh < 0 || cur_oh % sy)
                                        continue;
                                    cur_oh /= sy;
                                    if (cur_oh >= oh)
                                        continue;
                                    for (is = 0; is < fx; is++) {
                                        cur_ow = iw + px - dx * is; // cur_w = sx*iow-px+dx*is;
                                        if (cur_ow < 0 || cur_ow % sx)
                                            continue;
                                        cur_ow /= sx;
                                        if (cur_ow >= ow)
                                            continue;
                                        for (ik = 0; ik < k_per_group; ik++) {
                                            o_idx = in * od * oh * ow * k + cur_od * oh * ow * k + cur_oh * ow * k + cur_ow * k + ig * k_per_group + ik;
                                            f_idx = ig * k_per_group * fz * fy * fx * c_per_group + ik * fz * fy * fx * c_per_group + iz * fy * fx * c_per_group + ir * fx * c_per_group + is * c_per_group + ic;
                                            value += dst_grad[o_idx] * filter[f_idx];
                                        }
                                    }
                                }
                            }
                            src_grad[i_idx] = value;
                        }
                    }
                }
            }
        }
    }
#endif
}

static inline void naive_conv_wrw_ndhwc(const float *src, float *filter_grad, const float *dst_grad,
                                       size_t n, size_t w, size_t h, size_t d, size_t c, size_t k,
                                       size_t fx, size_t fy, size_t fz, size_t px, size_t py, size_t pz,
                                       size_t sx, size_t sy, size_t sz, size_t dx, size_t dy, size_t dz, size_t group) {
    size_t od = naive_conv_out_size(d, pz, dz, fz, sz);
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
#ifdef NAIVE_CONV_THREADED
    auto conv_one_pixel = [&](size_t ig, size_t ik, size_t iz, size_t ir, size_t is, size_t ic){
        size_t in, iod, ioh, iow;
        size_t cur_d, cur_h, cur_w, o_idx, i_idx, f_idx;
        float value = .0f;
        f_idx = ig * k_per_group * fz * fy * fx * c_per_group + ik * fz * fy * fx * c_per_group + iz * fy * fx * c_per_group + ir * fx * c_per_group + is * c_per_group + ic;
        for (in = 0; in < n; in++) {
            for (iod = 0; iod < od; iod++) {
                cur_d = sz * iod - pz + dz * iz;
                if (cur_d < 0 || cur_d >= d)
                    continue;
                for (ioh = 0; ioh < oh; ioh++) {
                    cur_h = sy * ioh - py + dy * ir;
                    if (cur_h < 0 || cur_h >= h)
                        continue;
                    for (iow = 0; iow < ow; iow++) {
                        cur_w = sx * iow - px + dx * is;
                        if (cur_w < 0 || cur_w >= w)
                            continue;
                        i_idx = in * d * h * w * c + cur_d * h * w * c + cur_h * w * c + cur_w * c + ig * c_per_group + ic;
                        o_idx = in * od * oh * ow * k + iod * oh * ow * k + ioh * ow * k + iow * k + ig * k_per_group + ik;
                        value += src[i_idx] * dst_grad[o_idx];
                    }
                }
            }
        }
        filter_grad[f_idx] = value;
    };
    naive_conv_blockwise_in_parallel_6d(conv_one_pixel, group, k_per_group, fz, fy, fx, c_per_group);
#else
    size_t ig, in, ik, iod, ioh, iow, ic, iz, is, ir;
    size_t cur_d, cur_h, cur_w, o_idx, i_idx, f_idx;

    for (ig = 0; ig < group; ig++) {
        for (ik = 0; ik < k_per_group; ik++) {
            for (iz = 0; iz < fz; iz++) {
                for (ir = 0; ir < fy; ir++) {
                    for (is = 0; is < fx; is++) {
                        for (ic = 0; ic < c_per_group; ic++) {
                            float value = .0f;
                            f_idx = ig * k_per_group * fz * fy * fx * c_per_group + ik * fz * fy * fx * c_per_group + iz * fy * fx * c_per_group + ir * fx * c_per_group + is * c_per_group + ic;
                            for (in = 0; in < n; in++) {
                                for (iod = 0; iod < od; iod++) {
                                    cur_d = sz * iod - pz + dz * iz;
                                    if (cur_d < 0 || cur_d >= d)
                                        continue;
                                    for (ioh = 0; ioh < oh; ioh++) {
                                        cur_h = sy * ioh - py + dy * ir;
                                        if (cur_h < 0 || cur_h >= h)
                                            continue;
                                        for (iow = 0; iow < ow; iow++) {
                                            cur_w = sx * iow - px + dx * is;
                                            if (cur_w < 0 || cur_w >= w)
                                                continue;
                                            i_idx = in * d * h * w * c + cur_d * h * w * c + cur_h * w * c + cur_w * c + ig * c_per_group + ic;
                                            o_idx = in * od * oh * ow * k + iod * oh * ow * k + ioh * ow * k + iow * k + ig * k_per_group + ik;
                                            value += src[i_idx] * dst_grad[o_idx];
                                        }
                                    }
                                }
                            }
                            filter_grad[f_idx] = value;
                        }
                    }
                }
            }
        }
    }
#endif
}

#endif
