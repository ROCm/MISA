#ifndef __NAIVE_CONV_H
#define __NAIVE_CONV_H
static inline size_t naive_conv_out_size(size_t in_size, size_t pad,
                                         size_t dilation, size_t ksize,
                                         size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}
static inline void naive_conv_fwd_nchw(const float *src, const float *filter,
                                       float *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy) {
    size_t in, ik, ioh, iow, ic, is, ir;
    size_t cur_h, cur_w, o_idx, i_idx, f_idx;
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    for (in = 0; in < n; in++) {
        for (ik = 0; ik < k; ik++) {
            for (ioh = 0; ioh < oh; ioh++) {
                for (iow = 0; iow < ow; iow++) {
                    // sliding window for this filter
                    float value = .0f;
                    o_idx = in * k * oh * ow + ik * oh * ow + ioh * ow + iow;
                    for (ic = 0; ic < c; ic++) {
                        for (ir = 0; ir < fy; ir++) {
                            cur_h = sy * ioh - py + dy * ir;
                            if (cur_h < 0 || cur_h >= h)
                                continue;
                            for (is = 0; is < fx; is++) {
                                cur_w = sx * iow - px + dx * is;
                                if (cur_w < 0 || cur_w >= w)
                                    continue;
                                i_idx = in * c * h * w + ic * h * w +
                                        cur_h * w + cur_w;
                                f_idx = ik * c * fy * fx + ic * fy * fx +
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
static inline void naive_conv_fwd_cnhw(const float *src, const float *filter,
                                       float *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy) {
    size_t in, ik, ioh, iow, ic, is, ir;
    size_t cur_h, cur_w, o_idx, i_idx, f_idx;
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    for (ik = 0; ik < k; ik++) {
        for (in = 0; in < n; in++) {
            for (ioh = 0; ioh < oh; ioh++) {
                for (iow = 0; iow < ow; iow++) {
                    // sliding window for this filter
                    float value = .0f;
                    o_idx = ik * n * oh * ow + in * oh * ow + ioh * ow + iow;
                    for (ic = 0; ic < c; ic++) {
                        for (ir = 0; ir < fy; ir++) {
                            cur_h = sy * ioh - py + dy * ir;
                            if (cur_h < 0 || cur_h >= h)
                                continue;
                            for (is = 0; is < fx; is++) {
                                cur_w = sx * iow - px + dx * is;
                                if (cur_w < 0 || cur_w >= w)
                                    continue;
                                i_idx = ic * n * h * w + in * h * w +
                                        cur_h * w + cur_w;
                                f_idx = ik * c * fy * fx + ic * fy * fx +
                                        ir * fx + is;
                                value += src[i_idx] * filter[f_idx];
                            }
                        }
                    }
                    dst[o_idx] = value;
                    // prsize_tf("o)idx:%d, value:%f\n",o_idx, value);
                }
            }
        }
    }
}
static inline void naive_conv_bwd_d_nchw(float *src_grad, const float *filter,
                                         const float *dst_grad, size_t n,
                                         size_t w, size_t h, size_t c, size_t k,
                                         size_t fx, size_t fy, size_t px,
                                         size_t py, size_t sx, size_t sy,
                                         size_t dx, size_t dy) {
    size_t in, ik, ih, iw, ic, is, ir;
    size_t cur_oh, cur_ow, o_idx, i_idx, f_idx;
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    for (in = 0; in < n; in++) {
        for (ic = 0; ic < c; ic++) {
            for (ih = 0; ih < h; ih++) {
                for (iw = 0; iw < w; iw++) {
                    float value = .0f;
                    i_idx = in * c * h * w + ic * h * w + ih * w + iw;
                    for (ik = 0; ik < k; ik++) {
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

                                o_idx = in * k * oh * ow + ik * oh * ow +
                                        cur_oh * ow + cur_ow;
                                f_idx = ik * c * fy * fx + ic * fy * fx +
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
static inline void naive_conv_bwd_d_cnhw(float *src_grad, const float *filter,
                                         const float *dst_grad, size_t n,
                                         size_t w, size_t h, size_t c, size_t k,
                                         size_t fx, size_t fy, size_t px,
                                         size_t py, size_t sx, size_t sy,
                                         size_t dx, size_t dy) {
    size_t in, ik, ih, iw, ic, is, ir;
    size_t cur_oh, cur_ow, o_idx, i_idx, f_idx;
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    for (ic = 0; ic < c; ic++) {
        for (in = 0; in < n; in++) {
            for (ih = 0; ih < h; ih++) {
                for (iw = 0; iw < w; iw++) {
                    float value = .0f;
                    i_idx = ic * n * h * w + in * h * w + ih * w + iw;
                    for (ik = 0; ik < k; ik++) {
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

                                o_idx = ik * n * oh * ow + in * oh * ow +
                                        cur_oh * ow + cur_ow;
                                f_idx = ik * c * fy * fx + ic * fy * fx +
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

static inline void naive_conv_bwd_f_nchw(const float *src, float *filter_grad,
                                         const float *dst_grad, size_t n,
                                         size_t w, size_t h, size_t c, size_t k,
                                         size_t fx, size_t fy, size_t px,
                                         size_t py, size_t sx, size_t sy,
                                         size_t dx, size_t dy) {
    size_t in, ik, ioh, iow, ic, is, ir;
    size_t cur_h, cur_w, o_idx, i_idx, f_idx;
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    for (ik = 0; ik < k; ik++) {
        for (ic = 0; ic < c; ic++) {
            for (ir = 0; ir < fy; ir++) {
                for (is = 0; is < fx; is++) {
                    float value = .0f;
                    f_idx = ik * c * fy * fx + ic * fy * fx + ir * fx + is;
                    for (in = 0; in < n; in++) {
                        for (ioh = 0; ioh < oh; ioh++) {
                            cur_h = sy * ioh - py + dy * ir;
                            if (cur_h < 0 || cur_h >= h)
                                continue;
                            for (iow = 0; iow < ow; iow++) {
                                cur_w = sx * iow - px + dx * is;
                                if (cur_w < 0 || cur_w >= w)
                                    continue;
                                i_idx = in * c * h * w + ic * h * w +
                                        cur_h * w + cur_w;
                                o_idx = in * k * oh * ow + ik * oh * ow +
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

static inline void naive_conv_bwd_f_cnhw(const float *src, float *filter_grad,
                                         const float *dst_grad, size_t n,
                                         size_t w, size_t h, size_t c, size_t k,
                                         size_t fx, size_t fy, size_t px,
                                         size_t py, size_t sx, size_t sy,
                                         size_t dx, size_t dy) {
    size_t in, ik, ioh, iow, ic, is, ir;
    size_t cur_h, cur_w, o_idx, i_idx, f_idx;
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    for (ik = 0; ik < k; ik++) {
        for (ic = 0; ic < c; ic++) {
            for (ir = 0; ir < fy; ir++) {
                for (is = 0; is < fx; is++) {
                    float value = .0f;
                    f_idx = ik * c * fy * fx + ic * fy * fx + ir * fx + is;
                    for (in = 0; in < n; in++) {
                        for (ioh = 0; ioh < oh; ioh++) {
                            cur_h = sy * ioh - py + dy * ir;
                            if (cur_h < 0 || cur_h >= h)
                                continue;
                            for (iow = 0; iow < ow; iow++) {
                                cur_w = sx * iow - px + dx * is;
                                if (cur_w < 0 || cur_w >= w)
                                    continue;
                                i_idx = ic * n * h * w + in * h * w +
                                        cur_h * w + cur_w;
                                o_idx = ik * n * oh * ow + in * oh * ow +
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