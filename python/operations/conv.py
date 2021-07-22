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


CONV_TENSOR_LAYOUT_NCHW = ((1 << 4) | 0)
CONV_TENSOR_LAYOUT_NHWC = ((1 << 4) | 1)
CONV_TENSOR_LAYOUT_CNHW = ((1 << 4) | 2)

CONV_DIRECTION_FWD = 0      # forward
CONV_DIRECTION_BWD = 1      # backward data
CONV_DIRECTOIN_WRW = 2      # backward weight

def conv_string_to_direction(direction_string):
    if direction_string == 'fwd':
        return CONV_DIRECTION_FWD
    if direction_string == 'bwd':
        return CONV_DIRECTION_BWD
    if direction_string == 'wrw':
        return CONV_DIRECTOIN_WRW
    assert False

def conv_direction_to_string(direction):
    if direction == CONV_DIRECTION_FWD:
        return 'fwd'
    if direction == CONV_DIRECTION_BWD:
        return 'bwd'
    if direction == CONV_DIRECTOIN_WRW:
        return 'wrw'
    assert False

def conv_string_to_tensor_layout(tensor_layout):
    if tensor_layout == 'nchw':
        return CONV_TENSOR_LAYOUT_NCHW
    if tensor_layout == 'nhwc':
        return CONV_TENSOR_LAYOUT_NHWC
    if tensor_layout == 'cnhw':
        return CONV_TENSOR_LAYOUT_CNHW
    assert False

def conv_tensor_layout_to_string(tensor_layout):
    if tensor_layout == CONV_TENSOR_LAYOUT_NCHW:
        return 'nchw'
    if tensor_layout == CONV_TENSOR_LAYOUT_NHWC:
        return 'nhwc'
    if tensor_layout == CONV_TENSOR_LAYOUT_CNHW:
        return 'cnhw'

def conv_out_size(in_size, pad, dilation,  ksize,  stride):
     return (in_size + 2*pad- dilation*(ksize-1) - 1) // stride + 1

class conv_param_t:
    def __init__(self, n, g, c, hi, wi, k, y, x, py, px, sy, sx, dy, dx, ho, wo, direction, precision):
        self.n = n
        self.g = g
        self.c = c
        self.hi = hi
        self.wi = wi
        self.k = k
        self.y = y
        self.x = x
        self.py = py
        self.px = px
        self.sy = sy
        self.sx = sx
        self.dy = dy
        self.dx = dx

        if ho == -1 or ho == 0:
            self.ho = conv_out_size(hi, py, dy, y, sy)
        else:
            self.ho = ho
        if wo == -1 or wo == 0:
            self.wo = conv_out_size(wi, px, dx, x, sx)
        else:
            self.wo = wo

        self.direction = direction
        self.precision = precision

    def dump(self):
        print("n:{}, g:{}, c:{}, hi:{}, wi:{}, k:{}, y:{}, x:{}, py:{}, px:{}, sy:{}, sx:{}, dy:{}, dx:{}, ho:{}, wo:{}, {},{}".format(
        self.n ,
        self.g ,
        self.c ,
        self.hi,
        self.wi,
        self.k ,
        self.y ,
        self.x ,
        self.py,
        self.px,
        self.sy,
        self.sx,
        self.dy,
        self.dx,
        self.ho,
        self.wo,
        self.direction,
        self.precision))
