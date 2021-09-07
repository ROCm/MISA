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

from collections import OrderedDict
import sys
import numpy as np
from ..codegen import *
# from .utility import *

CELLFFT_FEAT_BTFL_MERGE_S_C = True
CELLFFT_FEAT_BTFL_SCHED = True
CELLFFT_FEAT_BTFL_MINUS_THETA = True

def twiddle_isclose(f, target, atol=0.00001):
    d = abs(f-target)
    return d <= atol

'''
CAUSION!
need use np.float32 to do arithmetic
otherwise the error may be significant due to hw only use fp32 arithmetic

https://docs.python.org/3/tutorial/floatingpoint.html
str(float_value) is 17 digit by default.

'''
# https://oeis.org/A030109
brev_dict={
    "1" : [0],
    "2" : [0,1],
    "4" : [0,2,1,3],
    "8" : [0,4,2,6,1,5,3,7],
    "16": [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15],
    "32": [0,16,8,24,4,20,12,28,
           2,18,10,26,6,22,14,30,
           1,17,9,25,5,21,13,29,
           3,19,11,27,7,23,15,31],
    "64": [0,32,16,48,8,40,24,56,
           4,36,20,52,12,44,28,60,
           2,34,18,50,10,42,26,58,
           6,38,22,54,14,46,30,62,
           1,33,17,49,9,41,25,57,
           5,37,21,53,13,45,29,61,
           3,35,19,51,11,43,27,59,
           7,39,23,55,15,47,31,63],
    "128":[0,64,32,96,16,80,48,112,   8,72,40,104,24,88,56,120,
           4,68,36,100,20,84,52,116,  12,76,44,108,28,92,60,124,
           2,66,34,98,18,82,50,114,   10,74,42,106,26,90,58,122,
           6,70,38,102,22,86,54,118,  14,78,46,110,30,94,62,126,
           1,65,33,97,17,81,49,113,    9,73,41,105,25,89,57,121,
           5,69,37,101,21,85,53,117,  13,77,45,109,29,93,61,125,
           3,67,35,99,19,83,51,115,   11,75,43,107,27,91,59,123,
           7,71,39,103,23,87,55,119,  15,79,47,111,31,95,63,127]
}

BUTTERFLY_DIRECTION_FORWARD     = 0
BUTTERFLY_DIRECTION_BACKWARD    = 1


class ctrl_btfl_t(object):
    def __init__(self, n, k, direction):
        self.n          = n                 # total length of a sequence
        self.k          = k                 # target frequence at phase k
        self.direction  = direction

class btfl_t(macro_base_t):
    '''
    this is a basic implementation of butterfly operation
    and this is not optimal, due to it can not schedule instructions to avoid register bank.

    xx = x+y*w
    yy = x-y*w
           w: e^(i*theta) = cos(theta)+i*sin(theta), theta = -2*PI*k/N
           w: e^(i*theta) = cos(t)-i*sin(t), t = 2*PI*k/N = -theta

    xx = x_r+x_i*i + (y_r+y_i*i)*(cos(t)-i*sin(t))
    yy = x_r+x_i*i - (y_r+y_i*i)*(cos(t)-i*sin(t))

       xx_r = x_r + y_r*cos + y_i*sin = x_r + cos*(y_r + y_i*sin/cos)
       xx_i = x_i + y_i*cos - y_r*sin = x_i + cos*(y_i - y_r*sin/cos)
       yy_r = x_r - y_r*cos - y_i*sin = x_r - cos*(y_r + y_i*sin/cos)
       yy_i = x_i - y_i*cos + y_r*sin = x_i - cos*(y_i - y_r*sin/cos)

    inverse:
       xx_r = x_r + y_r*cos - y_i*sin = x_r + cos*(y_r - y_i*sin/cos)
       xx_i = x_i + y_i*cos + y_r*sin = x_i + cos*(y_i + y_r*sin/cos)
       yy_r = x_r - y_r*cos + y_i*sin = x_r - cos*(y_r - y_i*sin/cos)
       yy_i = x_i - y_i*cos - y_r*sin = x_i - cos*(y_i + y_r*sin/cos)


    if t = -2*PI*k/N = theta :
       xx_r = x_r + y_r*cos - y_i*sin = x_r + cos*(y_r - y_i*sin/cos)
       xx_i = x_i + y_i*cos + y_r*sin = x_i + cos*(y_i + y_r*sin/cos)
       yy_r = x_r - y_r*cos + y_i*sin = x_r - cos*(y_r - y_i*sin/cos)
       yy_i = x_i - y_i*cos - y_r*sin = x_i - cos*(y_i + y_r*sin/cos)

    inverse:
       xx_r = x_r + y_r*cos + y_i*sin = x_r + cos*(y_r + y_i*sin/cos)
       xx_i = x_i + y_i*cos - y_r*sin = x_i + cos*(y_i - y_r*sin/cos)
       yy_r = x_r - y_r*cos - y_i*sin = x_r - cos*(y_r + y_i*sin/cos)
       yy_i = x_i - y_i*cos + y_r*sin = x_i - cos*(y_i - y_r*sin/cos)

    '''
    def __init__(self, mc, ctrl, inline = False):
        assert type(ctrl) is ctrl_btfl_t
        macro_base_t.__init__(self, mc, inline)
        self.ctrl = ctrl
        self.declare_arg("v_x")    # complex value, 2 vgpr
        self.declare_arg("v_y")    # complex value, 2 vgpr
        self.declare_arg("v_t")    # complex value, 2 vgpr

    def name(self):
        return ".btfl_{}_w{}_{}".format('fwd' if self.ctrl.direction == BUTTERFLY_DIRECTION_FORWARD else 'bwd', n, k)

    def expr(self):
        def cal_theta(n, k):
            if CELLFFT_FEAT_BTFL_MINUS_THETA:
                return -2*np.pi*(k*1.0/n)
            else:
                return 2*np.pi*(k*1.0/n)
        def gen_fwd():
            theta = cal_theta(self.ctrl.n, self.ctrl.k)
            c = np.float32(np.cos(theta))
            s = np.float32(np.sin(theta))
            self._emit(f"; c:{c}, s:{s}")
            with self._deferred_context():
                if twiddle_isclose(theta, 0):
                    self._emit(f"v_sub_f32 v[{self.v_t(0)}], v[{self.v_x(0)}], v[{self.v_y(0)}]")
                    self._emit(f"v_sub_f32 v[{self.v_t(1)}], v[{self.v_x(1)}], v[{self.v_y(1)}]")
                    self._emit(f"v_add_f32 v[{self.v_x(0)}], v[{self.v_x(0)}], v[{self.v_y(0)}]")
                    self._emit(f"v_add_f32 v[{self.v_x(1)}], v[{self.v_x(1)}], v[{self.v_y(1)}]")
                    self._emit(f"v_mov_b32 v[{self.v_y(0)}], v[{self.v_t(0)}]")
                    self._emit(f"v_mov_b32 v[{self.v_y(1)}], v[{self.v_t(1)}]")
                elif twiddle_isclose(theta, np.pi/2):
                    if CELLFFT_FEAT_BTFL_MINUS_THETA:
                        # self._emit(f"v_add_f32 v[{self.v_t(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        # self._emit(f"v_sub_f32 v[{self.v_t(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        # self._emit(f"v_sub_f32 v[{self.v_x(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        # self._emit(f"v_add_f32 v[{self.v_x(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        # self._emit(f"v_mov_b32 v[{self.v_y(0)}], v[{self.v_t(0)}]")
                        # self._emit(f"v_mov_b32 v[{self.v_y(1)}], v[{self.v_t(1)}]")
                        assert False, "minus theta shuold not go into this branch"
                    else:
                        self._emit(f"v_sub_f32 v[{self.v_t(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        self._emit(f"v_add_f32 v[{self.v_t(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        self._emit(f"v_add_f32 v[{self.v_x(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        self._emit(f"v_sub_f32 v[{self.v_x(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        self._emit(f"v_mov_b32 v[{self.v_y(0)}], v[{self.v_t(0)}]")
                        self._emit(f"v_mov_b32 v[{self.v_y(1)}], v[{self.v_t(1)}]")
                elif twiddle_isclose(theta, -np.pi/2):
                    if CELLFFT_FEAT_BTFL_MINUS_THETA:
                        self._emit(f"v_sub_f32 v[{self.v_t(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        self._emit(f"v_add_f32 v[{self.v_t(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        self._emit(f"v_add_f32 v[{self.v_x(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        self._emit(f"v_sub_f32 v[{self.v_x(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        self._emit(f"v_mov_b32 v[{self.v_y(0)}], v[{self.v_t(0)}]")
                        self._emit(f"v_mov_b32 v[{self.v_y(1)}], v[{self.v_t(1)}]")
                    else:
                        # self._emit(f"v_sub_f32 v[{self.v_t(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        # self._emit(f"v_add_f32 v[{self.v_t(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        # self._emit(f"v_add_f32 v[{self.v_x(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        # self._emit(f"v_sub_f32 v[{self.v_x(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        # self._emit(f"v_mov_b32 v[{self.v_y(0)}], v[{self.v_t(0)}]")
                        # self._emit(f"v_mov_b32 v[{self.v_y(1)}], v[{self.v_t(1)}]")
                        assert False, "possitive theta shuold not go into this branch"
                else:
                    if twiddle_isclose(s/c, 1):
                        if CELLFFT_FEAT_BTFL_MINUS_THETA:
                            self._emit(f"v_sub_f32 v[{self.v_t(0)}], v[{self.v_y(0)}], v[{self.v_y(1)}]")
                            self._emit(f"v_add_f32 v[{self.v_t(1)}], v[{self.v_y(1)}], v[{self.v_y(0)}]")
                        else:
                            self._emit(f"v_add_f32 v[{self.v_t(0)}], v[{self.v_y(0)}], v[{self.v_y(1)}]")
                            self._emit(f"v_sub_f32 v[{self.v_t(1)}], v[{self.v_y(1)}], v[{self.v_y(0)}]")
                        self._emit(v_madmk(self.v_y(0), self.v_t(0), np.float32(-1) * c, self.v_x(0)))
                        self._emit(v_madmk(self.v_y(1), self.v_t(1), np.float32(-1) * c, self.v_x(1)))
                        self._emit(v_madmk(self.v_x(0), self.v_t(0), c, self.v_x(0)))
                        self._emit(v_madmk(self.v_x(1), self.v_t(1), c, self.v_x(1)))
                    elif twiddle_isclose(s/c, -1):
                        if CELLFFT_FEAT_BTFL_MINUS_THETA:
                            self._emit(f"v_add_f32 v[{self.v_t(0)}], v[{self.v_y(0)}], v[{self.v_y(1)}]")
                            self._emit(f"v_sub_f32 v[{self.v_t(1)}], v[{self.v_y(1)}], v[{self.v_y(0)}]")
                        else:
                            self._emit(f"v_sub_f32 v[{self.v_t(0)}], v[{self.v_y(0)}], v[{self.v_y(1)}]")
                            self._emit(f"v_add_f32 v[{self.v_t(1)}], v[{self.v_y(1)}], v[{self.v_y(0)}]")
                        self._emit(v_madmk(self.v_y(0), self.v_t(0), np.float32(-1) * c, self.v_x(0)))
                        self._emit(v_madmk(self.v_y(1), self.v_t(1), np.float32(-1) * c, self.v_x(1)))
                        self._emit(v_madmk(self.v_x(0), self.v_t(0), c, self.v_x(0)))
                        self._emit(v_madmk(self.v_x(1), self.v_t(1), c, self.v_x(1)))
                    else:
                        if CELLFFT_FEAT_BTFL_MERGE_S_C:
                            if CELLFFT_FEAT_BTFL_MINUS_THETA:
                                self._emit(v_madmk(self.v_t(0), self.v_y(1), np.float32(-1)*s/c, self.v_y(0)))
                                self._emit(v_madmk(self.v_t(1), self.v_y(0), s/c, self.v_y(1)))
                            else:
                                self._emit(v_madmk(self.v_t(0), self.v_y(1), s/c, self.v_y(0)))
                                self._emit(v_madmk(self.v_t(1), self.v_y(0), np.float32(-1)*s/c, self.v_y(1)))
                            self._emit(v_madmk(self.v_y(0), self.v_t(0), np.float32(-1) * c, self.v_x(0)))
                            self._emit(v_madmk(self.v_y(1), self.v_t(1), np.float32(-1) * c, self.v_x(1)))
                            self._emit(v_madmk(self.v_x(0), self.v_t(0), c, self.v_x(0)))
                            self._emit(v_madmk(self.v_x(1), self.v_t(1), c, self.v_x(1)))
                        else:
                            if CELLFFT_FEAT_BTFL_MINUS_THETA:
                                self._emit(v_madmk(self.v_t(0), self.v_y(1), s, self.v_x(0)))
                                self._emit(v_madmk(self.v_t(1), self.v_y(0), np.float32(-1)*s, self.v_x(1)))
                                self._emit(v_madmk(self.v_x(0), self.v_y(1), np.float32(-1)*s, self.v_x(0)))
                                self._emit(v_madmk(self.v_x(1), self.v_y(0), s, self.v_x(1)))
                            else:
                                self._emit(v_madmk(self.v_t(0), self.v_y(1), np.float32(-1)*s, self.v_x(0)))
                                self._emit(v_madmk(self.v_t(1), self.v_y(0), s, self.v_x(1)))
                                self._emit(v_madmk(self.v_x(0), self.v_y(1), s, self.v_x(0)))
                                self._emit(v_madmk(self.v_x(1), self.v_y(0), np.float32(-1)*s, self.v_x(1)))
                            self._emit(v_madmk(self.v_y(0), self.v_y(0), np.float32(-1)*c, self.v_t(0)))
                            self._emit(v_madmk(self.v_y(1), self.v_y(1), np.float32(-1)*c, self.v_t(1)))
                            self._emit(v_madmk(self.v_x(0), self.v_y(0), c, self.v_x(0)))
                            self._emit(v_madmk(self.v_x(1), self.v_y(1), c, self.v_x(1)))

            return self._get_deferred()

        def gen_bwd():
            theta = cal_theta(self.ctrl.n, self.ctrl.k)
            c = np.float32(np.cos(theta))
            s = np.float32(np.sin(theta))
            with self._deferred_context():
                if twiddle_isclose(theta, 0):
                    self._emit(f"v_sub_f32 v[{self.v_t(0)}], v[{self.v_x(0)}], v[{self.v_y(0)}]")
                    self._emit(f"v_sub_f32 v[{self.v_t(1)}], v[{self.v_x(1)}], v[{self.v_y(1)}]")
                    self._emit(f"v_add_f32 v[{self.v_x(0)}], v[{self.v_x(0)}], v[{self.v_y(0)}]")
                    self._emit(f"v_add_f32 v[{self.v_x(1)}], v[{self.v_x(1)}], v[{self.v_y(1)}]")
                    self._emit(f"v_mov_b32 v[{self.v_y(0)}], v[{self.v_t(0)}]")
                    self._emit(f"v_mov_b32 v[{self.v_y(1)}], v[{self.v_t(1)}]")
                elif twiddle_isclose(theta, np.pi/2):
                    if CELLFFT_FEAT_BTFL_MINUS_THETA:
                        # self._emit(f"v_sub_f32 v[{self.v_t(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        # self._emit(f"v_add_f32 v[{self.v_t(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        # self._emit(f"v_add_f32 v[{self.v_x(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        # self._emit(f"v_sub_f32 v[{self.v_x(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        # self._emit(f"v_mov_b32 v[{self.v_y(0)}], v[{self.v_t(0)}]")
                        # self._emit(f"v_mov_b32 v[{self.v_y(1)}], v[{self.v_t(1)}]")
                        assert False, "minus theta shuold not go into this branch"
                    else:
                        self._emit(f"v_add_f32 v[{self.v_t(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        self._emit(f"v_sub_f32 v[{self.v_t(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        self._emit(f"v_sub_f32 v[{self.v_x(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        self._emit(f"v_add_f32 v[{self.v_x(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        self._emit(f"v_mov_b32 v[{self.v_y(0)}], v[{self.v_t(0)}]")
                        self._emit(f"v_mov_b32 v[{self.v_y(1)}], v[{self.v_t(1)}]")
                elif twiddle_isclose(theta, -np.pi/2):
                    if CELLFFT_FEAT_BTFL_MINUS_THETA:
                        self._emit(f"v_add_f32 v[{self.v_t(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        self._emit(f"v_sub_f32 v[{self.v_t(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        self._emit(f"v_sub_f32 v[{self.v_x(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        self._emit(f"v_add_f32 v[{self.v_x(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        self._emit(f"v_mov_b32 v[{self.v_y(0)}], v[{self.v_t(0)}]")
                        self._emit(f"v_mov_b32 v[{self.v_y(1)}], v[{self.v_t(1)}]")
                    else:
                        # self._emit(f"v_add_f32 v[{self.v_t(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        # self._emit(f"v_sub_f32 v[{self.v_t(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        # self._emit(f"v_sub_f32 v[{self.v_x(0)}], v[{self.v_x(0)}], v[{self.v_y(1)}]")
                        # self._emit(f"v_add_f32 v[{self.v_x(1)}], v[{self.v_x(1)}], v[{self.v_y(0)}]")
                        # self._emit(f"v_mov_b32 v[{self.v_y(0)}], v[{self.v_t(0)}]")
                        # self._emit(f"v_mov_b32 v[{self.v_y(1)}], v[{self.v_t(1)}]")
                        assert False, "possitive theta shuold not go into this branch"
                else:
                    if twiddle_isclose(s/c, 1):
                        if CELLFFT_FEAT_BTFL_MINUS_THETA:
                            self._emit(f"v_add_f32 v[{self.v_t(0)}], v[{self.v_y(0)}], v[{self.v_y(1)}]")
                            self._emit(f"v_sub_f32 v[{self.v_t(1)}], v[{self.v_y(1)}], v[{self.v_y(0)}]")
                        else:
                            self._emit(f"v_sub_f32 v[{self.v_t(0)}], v[{self.v_y(0)}], v[{self.v_y(1)}]")
                            self._emit(f"v_add_f32 v[{self.v_t(1)}], v[{self.v_y(1)}], v[{self.v_y(0)}]")
                        self._emit(v_madmk(self.v_y(0), self.v_t(0), np.float32(-1)*c, self.v_x(0)))
                        self._emit(v_madmk(self.v_y(1), self.v_t(1), np.float32(-1)*c, self.v_x(1)))
                        self._emit(v_madmk(self.v_x(0), self.v_t(0), c, self.v_x(0)))
                        self._emit(v_madmk(self.v_x(1), self.v_t(1), c, self.v_x(1)))
                    elif twiddle_isclose(s/c, -1):
                        if CELLFFT_FEAT_BTFL_MINUS_THETA:
                            self._emit(f"v_sub_f32 v[{self.v_t(0)}], v[{self.v_y(0)}], v[{self.v_y(1)}]")
                            self._emit(f"v_add_f32 v[{self.v_t(1)}], v[{self.v_y(1)}], v[{self.v_y(0)}]")
                        else:
                            self._emit(f"v_add_f32 v[{self.v_t(0)}], v[{self.v_y(0)}], v[{self.v_y(1)}]")
                            self._emit(f"v_sub_f32 v[{self.v_t(1)}], v[{self.v_y(1)}], v[{self.v_y(0)}]")
                        self._emit(v_madmk(self.v_y(0), self.v_t(0), np.float32(-1)*c, self.v_x(0)))
                        self._emit(v_madmk(self.v_y(1), self.v_t(1), np.float32(-1)*c, self.v_x(1)))
                        self._emit(v_madmk(self.v_x(0), self.v_t(0), c, self.v_x(0)))
                        self._emit(v_madmk(self.v_x(1), self.v_t(1), c, self.v_x(1)))
                    else:
                        if CELLFFT_FEAT_BTFL_MERGE_S_C:
                            if CELLFFT_FEAT_BTFL_MINUS_THETA:
                                self._emit(v_madmk(self.v_t(0), self.v_y(1), s/c, self.v_y(0)))
                                self._emit(v_madmk(self.v_t(1), self.v_y(0), np.float32(-1)*s/c, self.v_y(1)))
                            else:
                                self._emit(v_madmk(self.v_t(0), self.v_y(1), np.float32(-1)*s/c, self.v_y(0)))
                                self._emit(v_madmk(self.v_t(1), self.v_y(0), s/c, self.v_y(1)))
                            self._emit(v_madmk(self.v_y(0), self.v_t(0), np.float32(-1)*c, self.v_x(0)))
                            self._emit(v_madmk(self.v_y(1), self.v_t(1), np.float32(-1)*c, self.v_x(1)))
                            self._emit(v_madmk(self.v_x(0), self.v_t(0), c, self.v_x(0)))
                            self._emit(v_madmk(self.v_x(1), self.v_t(1), c, self.v_x(1)))
                        else:
                            if CELLFFT_FEAT_BTFL_MINUS_THETA:
                                self._emit(v_madmk(self.v_t(0), self.v_y(1), np.float32(-1)*s, self.v_x(0)))
                                self._emit(v_madmk(self.v_t(1), self.v_y(0), s, self.v_x(1)))
                                self._emit(v_madmk(self.v_x(0), self.v_y(1), s, self.v_x(0)))
                                self._emit(v_madmk(self.v_x(1), self.v_y(0), np.float32(-1)*s, self.v_x(1)))
                            else:
                                self._emit(v_madmk(self.v_t(0), self.v_y(1), s, self.v_x(0)))
                                self._emit(v_madmk(self.v_t(1), self.v_y(0), np.float32(-1)*s, self.v_x(1)))
                                self._emit(v_madmk(self.v_x(0), self.v_y(1), np.float32(-1)*s, self.v_x(0)))
                                self._emit(v_madmk(self.v_x(1), self.v_y(0), s, self.v_x(1)))
                            self._emit(v_madmk(self.v_y(0), self.v_t(0), np.float32(-1)*c, self.v_x(0)))
                            self._emit(v_madmk(self.v_y(1), self.v_t(1), np.float32(-1)*c, self.v_x(1)))
                            self._emit(v_madmk(self.v_x(0), self.v_t(0), c, self.v_x(0)))
                            self._emit(v_madmk(self.v_x(1), self.v_t(1), c, self.v_x(1)))
            return self._get_deferred()

        self._emit(gen_fwd() if self.ctrl.direction == BUTTERFLY_DIRECTION_FORWARD else gen_bwd())


class ctrl_fft_t(object):
    def __init__(self, n, opt_n, direction, use_sched = CELLFFT_FEAT_BTFL_SCHED):
        self.n          = n                 # total length of a sequence
        self.opt_n      = opt_n             # first opt_n value have number, other are zero
        self.direction  = direction
        self.use_sched  = use_sched

class fft_t(macro_base_t):
    '''
    basic implementation of n-point fft
    attension! after this operation, point in register is in brev order
    e.g, 0,1,2,3,4,5,6,7 -- fft(8) --> 0.4.2.6.1.5.3.7

    pay attension to register request when use_sched is true or false
    '''
    def __init__(self, mc, ctrl, inline = False):
        macro_base_t.__init__(self, mc, inline)
        assert type(ctrl) is ctrl_fft_t
        self.ctrl = ctrl
        self.declare_arg("v_pt")       # complex value, 2 vgpr each, n*2 total
        self.declare_arg("v_tt")       # complex value, 2 vgpr each, 4 vgpr if not use_sched else n vgpr

    def name(self):
        opt_list = opt_list_t(self.ctrl.n, self.ctrl.opt_n)
        base_str = "fft_{}_".format('fwd' if self.ctrl.direction == BUTTERFLY_DIRECTION_FORWARD else 'bwd' )

        return "fft{}_{}{}".format(self.ctrl.n,
                        'fwd' if self.ctrl.direction == BUTTERFLY_DIRECTION_FORWARD else 'bwd',
                        self.ctrl.opt_n if opt_list.is_opt() else '')

    def expr(self):
        class opt_list_t(object):
            def __init__(self,n,opt_n):
                if opt_n==0 or opt_n >= n:
                    self._list = [1]*n
                    self._opt = False
                else:
                    self._list = [0]*n
                    for i in range(opt_n):
                        self._list[i] = 1
                    self._opt = True
            def check(self,idx):
                if idx >= len(self._list):
                    raise ValueError("required idx bigger than list")
                if self._list[idx] == 1:
                    return True
                return False
            def touch(self,idx):
                if idx >= len(self._list):
                    raise ValueError("required idx bigger than list")
                self._list[idx] = 1
            def is_opt(self):
                return self._opt

        if self.ctrl.use_sched:
            # give a chance that handcraft fast transform can be selected if possible
            if self.ctrl.n == 4:
                fft4 = fft4_fwd_sched_t(self.mc, self.inline) if self.ctrl.direction == BUTTERFLY_DIRECTION_FORWARD else \
                        fft4_bwd_sched_t(self.mc, self.inline)
                self._emit(fft4(self.v_pt(), self.v_tt()))
                return
            elif self.ctrl.n == 8:
                fft8 = fft8_fwd_sched_t(self.mc, self.inline) if self.ctrl.direction == BUTTERFLY_DIRECTION_FORWARD else \
                        fft8_bwd_sched_t(self.mc, self.inline)
                self._emit(fft8(self.v_pt(), self.v_tt()))
                return
            elif self.ctrl.n == 16:
                fft16 = fft16_fwd_sched_t(self.mc, self.inline) if self.ctrl.direction == BUTTERFLY_DIRECTION_FORWARD else \
                        fft16_bwd_sched_t(self.mc, self.inline)
                self._emit(fft16(self.v_pt(), self.v_tt()))
                return
                


        opt_list = opt_list_t(self.ctrl.n, self.ctrl.opt_n)
        radix = int(np.log2(self.ctrl.n))
        for itr in range(radix):
            groups = int(np.power(2,itr))
            omega = int(np.power(2,itr+1))
            stride = self.ctrl.n//omega
            group_width = self.ctrl.n//omega
            if str(omega//2) not in brev_dict:
                raise ValueError("omega not in brev list")
            brev_list = brev_dict[str(omega//2)]

            for g in range(groups):
                k = brev_list[g]
                btfl = btfl_t(self.mc, ctrl_btfl_t(omega, k, self.ctrl.direction), self.inline)        # TODO: hardcode inline here
                for gtr in range(group_width):
                    ia = 2*stride*g+gtr
                    ib = 2*stride*g+gtr+stride
                    self._emit(f"; {ia} -- {ib}, omega:{omega}, k:{k}")
                    if opt_list.check(ia) and opt_list.check(ib):
                        self._emit(btfl(self.v_pt(2*ia), self.v_pt(2*ib), self.v_tt()  ))
                    elif not opt_list.check(ia) and not opt_list.check(ib):
                        pass
                    elif opt_list.check(ia) and not opt_list.check(ib):
                        self._emit(f"v_mov_b32 v[{self.v_pt(2*ib)}], v[{self.v_pt(2*ia)}]")
                        self._emit(f"v_mov_b32 v[{self.v_pt(2*ib+1)}], v[{self.v_pt(2*ia+1)}]")
                        opt_list.touch(ib)
                    elif not opt_list.check(ia) and opt_list.check(ib):
                        raise ValueError("not implemented and should not happen in current algo")
                    self._emit_empty_line()

            if itr != radix-1:
                self._emit_empty_line()

class fft4_fwd_sched_t(macro_base_t):
    '''
    handcraft scheduled fft implementation
    target no pipeline stall if possible
    '''
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_pt")    # continus pixels
        self.declare_arg("v_tt")    # half the size of above pixel

    def expr(self):
        # omega 2_0
        self._emit(f"v_sub_f32 v[{self.v_tt(2)}], v[{self.v_pt(2)}], v[{self.v_pt(6)}]")    # 6
        self._emit(f"v_add_f32 v[{self.v_pt(6)}], v[{self.v_pt(2)}], v[{self.v_pt(6)}]")    # 2
        self._emit(f"v_sub_f32 v[{self.v_tt(0)}], v[{self.v_pt(0)}], v[{self.v_pt(4)}]")    # 4
        self._emit(f"v_add_f32 v[{self.v_pt(4)}], v[{self.v_pt(0)}], v[{self.v_pt(4)}]")    # 0
        self._emit(f"v_sub_f32 v[{self.v_tt(3)}], v[{self.v_pt(3)}], v[{self.v_pt(7)}]")    # 7
        self._emit(f"v_add_f32 v[{self.v_pt(7)}], v[{self.v_pt(3)}], v[{self.v_pt(7)}]")    # 3
        self._emit(f"v_sub_f32 v[{self.v_tt(1)}], v[{self.v_pt(1)}], v[{self.v_pt(5)}]")    # 5
        self._emit(f"v_add_f32 v[{self.v_pt(5)}], v[{self.v_pt(1)}], v[{self.v_pt(5)}]")    # 1

        # omega 4_0, 4_1
        self._emit(f"v_add_f32 v[{self.v_pt(0)}], v[{self.v_pt(4)}], v[{self.v_pt(6)}]")
        self._emit(f"v_sub_f32 v[{self.v_pt(2)}], v[{self.v_pt(4)}], v[{self.v_pt(6)}]")
        self._emit(f"v_add_f32 v[{self.v_pt(4)}], v[{self.v_tt(0)}], v[{self.v_tt(3)}]")
        self._emit(f"v_sub_f32 v[{self.v_pt(6)}], v[{self.v_tt(0)}], v[{self.v_tt(3)}]")
        self._emit(f"v_add_f32 v[{self.v_pt(1)}], v[{self.v_pt(5)}], v[{self.v_pt(7)}]")
        self._emit(f"v_sub_f32 v[{self.v_pt(3)}], v[{self.v_pt(5)}], v[{self.v_pt(7)}]")
        self._emit(f"v_sub_f32 v[{self.v_pt(5)}], v[{self.v_tt(1)}], v[{self.v_tt(2)}]")
        self._emit(f"v_add_f32 v[{self.v_pt(7)}], v[{self.v_tt(1)}], v[{self.v_tt(2)}]")

class fft4_bwd_sched_t(macro_base_t):
    '''
    handcraft scheduled fft implementation
    target no pipeline stall if possible
    '''
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_pt")    # continus pixels
        self.declare_arg("v_tt")    # half the size of above pixel

    def expr(self):
        # omega 2_0
        self._emit(f"v_sub_f32 v[{self.v_tt(2)}], v[{self.v_pt(2)}], v[{self.v_pt(6)}]")    # 6
        self._emit(f"v_add_f32 v[{self.v_pt(6)}], v[{self.v_pt(2)}], v[{self.v_pt(6)}]")    # 2
        self._emit(f"v_sub_f32 v[{self.v_tt(0)}], v[{self.v_pt(0)}], v[{self.v_pt(4)}]")    # 4
        self._emit(f"v_add_f32 v[{self.v_pt(4)}], v[{self.v_pt(0)}], v[{self.v_pt(4)}]")    # 0
        self._emit(f"v_sub_f32 v[{self.v_tt(3)}], v[{self.v_pt(3)}], v[{self.v_pt(7)}]")    # 7
        self._emit(f"v_add_f32 v[{self.v_pt(7)}], v[{self.v_pt(3)}], v[{self.v_pt(7)}]")    # 3
        self._emit(f"v_sub_f32 v[{self.v_tt(1)}], v[{self.v_pt(1)}], v[{self.v_pt(5)}]")    # 5
        self._emit(f"v_add_f32 v[{self.v_pt(5)}], v[{self.v_pt(1)}], v[{self.v_pt(5)}]")    # 1

        # omega 4_0, 4_1
        self._emit(f"v_add_f32 v[{self.v_pt(0)}], v[{self.v_pt(4)}], v[{self.v_pt(6)}]")
        self._emit(f"v_sub_f32 v[{self.v_pt(2)}], v[{self.v_pt(4)}], v[{self.v_pt(6)}]")
        self._emit(f"v_sub_f32 v[{self.v_pt(4)}], v[{self.v_tt(0)}], v[{self.v_tt(3)}]")
        self._emit(f"v_add_f32 v[{self.v_pt(6)}], v[{self.v_tt(0)}], v[{self.v_tt(3)}]")
        self._emit(f"v_add_f32 v[{self.v_pt(1)}], v[{self.v_pt(5)}], v[{self.v_pt(7)}]")
        self._emit(f"v_sub_f32 v[{self.v_pt(3)}], v[{self.v_pt(5)}], v[{self.v_pt(7)}]")
        self._emit(f"v_add_f32 v[{self.v_pt(5)}], v[{self.v_tt(1)}], v[{self.v_tt(2)}]")
        self._emit(f"v_sub_f32 v[{self.v_pt(7)}], v[{self.v_tt(1)}], v[{self.v_tt(2)}]")

class fft8_fwd_sched_t(macro_base_t):
    '''
    handcraft scheduled fft implementation
    target no pipeline stall if possible
    '''
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_pt")    # continus pixels
        self.declare_arg("v_tt")    # half the size of above pixel

    def expr(self):
        # omega 2_0
        self._emit(f"v_sub_f32 v[{self.v_tt( 0)}], v[{self.v_pt( 0)}], v[{self.v_pt( 8)}]") # 8
        self._emit(f"v_sub_f32 v[{self.v_tt( 1)}], v[{self.v_pt( 1)}], v[{self.v_pt( 9)}]") # 9
        self._emit(f"v_sub_f32 v[{self.v_tt( 2)}], v[{self.v_pt( 2)}], v[{self.v_pt(10)}]") # 10
        self._emit(f"v_sub_f32 v[{self.v_tt( 3)}], v[{self.v_pt( 3)}], v[{self.v_pt(11)}]") # 11
        self._emit(f"v_sub_f32 v[{self.v_tt( 4)}], v[{self.v_pt( 4)}], v[{self.v_pt(12)}]") # 12
        self._emit(f"v_sub_f32 v[{self.v_tt( 5)}], v[{self.v_pt( 5)}], v[{self.v_pt(13)}]") # 13
        self._emit(f"v_sub_f32 v[{self.v_tt( 6)}], v[{self.v_pt( 6)}], v[{self.v_pt(14)}]") # 14
        self._emit(f"v_sub_f32 v[{self.v_tt( 7)}], v[{self.v_pt( 7)}], v[{self.v_pt(15)}]") # 15
        self._emit(f"v_add_f32 v[{self.v_pt( 8)}], v[{self.v_pt( 0)}], v[{self.v_pt( 8)}]") # 0
        self._emit(f"v_add_f32 v[{self.v_pt( 9)}], v[{self.v_pt( 1)}], v[{self.v_pt( 9)}]") # 1
        self._emit(f"v_add_f32 v[{self.v_pt(10)}], v[{self.v_pt( 2)}], v[{self.v_pt(10)}]") # 2
        self._emit(f"v_add_f32 v[{self.v_pt(11)}], v[{self.v_pt( 3)}], v[{self.v_pt(11)}]") # 3
        self._emit(f"v_add_f32 v[{self.v_pt(12)}], v[{self.v_pt( 4)}], v[{self.v_pt(12)}]") # 4
        self._emit(f"v_add_f32 v[{self.v_pt(13)}], v[{self.v_pt( 5)}], v[{self.v_pt(13)}]") # 5
        self._emit(f"v_add_f32 v[{self.v_pt(14)}], v[{self.v_pt( 6)}], v[{self.v_pt(14)}]") # 6
        self._emit(f"v_add_f32 v[{self.v_pt(15)}], v[{self.v_pt( 7)}], v[{self.v_pt(15)}]") # 7

        # omega 4_0, 4_1
        self._emit(f"v_add_f32 v[{self.v_pt( 0)}], v[{self.v_tt( 0)}], v[{self.v_tt( 5)}]") # 8
        self._emit(f"v_sub_f32 v[{self.v_pt( 1)}], v[{self.v_tt( 1)}], v[{self.v_tt( 4)}]") # 9
        self._emit(f"v_sub_f32 v[{self.v_pt( 4)}], v[{self.v_tt( 0)}], v[{self.v_tt( 5)}]") # 12
        self._emit(f"v_add_f32 v[{self.v_pt( 5)}], v[{self.v_tt( 1)}], v[{self.v_tt( 4)}]") # 13
        self._emit(f"v_add_f32 v[{self.v_pt( 2)}], v[{self.v_tt( 2)}], v[{self.v_tt( 7)}]") # 10
        self._emit(f"v_sub_f32 v[{self.v_pt( 3)}], v[{self.v_tt( 3)}], v[{self.v_tt( 6)}]") # 11
        self._emit(f"v_sub_f32 v[{self.v_pt( 6)}], v[{self.v_tt( 2)}], v[{self.v_tt( 7)}]") # 14
        self._emit(f"v_add_f32 v[{self.v_pt( 7)}], v[{self.v_tt( 3)}], v[{self.v_tt( 6)}]") # 15
        self._emit(f"v_add_f32 v[{self.v_tt( 0)}], v[{self.v_pt( 8)}], v[{self.v_pt(12)}]") # 0
        self._emit(f"v_add_f32 v[{self.v_tt( 1)}], v[{self.v_pt( 9)}], v[{self.v_pt(13)}]") # 1
        self._emit(f"v_sub_f32 v[{self.v_tt( 4)}], v[{self.v_pt( 8)}], v[{self.v_pt(12)}]") # 4
        self._emit(f"v_sub_f32 v[{self.v_tt( 5)}], v[{self.v_pt( 9)}], v[{self.v_pt(13)}]") # 5
        self._emit(f"v_add_f32 v[{self.v_tt( 2)}], v[{self.v_pt(10)}], v[{self.v_pt(14)}]") # 2
        self._emit(f"v_add_f32 v[{self.v_tt( 3)}], v[{self.v_pt(11)}], v[{self.v_pt(15)}]") # 3
        self._emit(f"v_sub_f32 v[{self.v_tt( 6)}], v[{self.v_pt(10)}], v[{self.v_pt(14)}]") # 6

        self._emit(f"v_add_f32 v[{self.v_pt( 8)}], v[{self.v_pt( 2)}], v[{self.v_pt( 3)}]") #   4 -- 5 tmp_0
        self._emit(f"v_sub_f32 v[{self.v_pt( 9)}], v[{self.v_pt( 3)}], v[{self.v_pt( 2)}]") #   4 -- 5 tmp_1
        self._emit(f"v_sub_f32 v[{self.v_pt(12)}], v[{self.v_pt( 6)}], v[{self.v_pt( 7)}]") #   6 -- 7 tmp_1
        self._emit(f"v_add_f32 v[{self.v_pt(13)}], v[{self.v_pt( 7)}], v[{self.v_pt( 6)}]") #   6 -- 7 tmp_0
        self._emit(f"v_sub_f32 v[{self.v_tt( 7)}], v[{self.v_pt(11)}], v[{self.v_pt(15)}]") # 7

        # omega 8_0, 8_2, 8_1, 8_3
        self._emit(v_madmk(self.v_pt(10), self.v_pt( 8), -0.7071067690849304, self.v_pt( 0)))  # 10
        self._emit(f"v_sub_f32 v[{self.v_pt( 2)}], v[{self.v_tt( 0)}], v[{self.v_tt( 2)}]") # 2
        self._emit(f"v_sub_f32 v[{self.v_pt( 3)}], v[{self.v_tt( 1)}], v[{self.v_tt( 3)}]") # 3
        self._emit(v_madmk(self.v_pt(14), self.v_pt(12),  0.7071067690849304, self.v_pt( 4)))  # 14
        self._emit(v_madmk(self.v_pt(11), self.v_pt( 9), -0.7071067690849304, self.v_pt( 1)))  # 11
        self._emit(v_madmk(self.v_pt(15), self.v_pt(13),  0.7071067690849304, self.v_pt( 5)))  # 15
        self._emit(v_madmk(self.v_pt( 8), self.v_pt( 8),  0.7071067690849304, self.v_pt( 0)))  # 8
        self._emit(f"v_sub_f32 v[{self.v_pt( 6)}], v[{self.v_tt( 4)}], v[{self.v_tt( 7)}]") # 6
        self._emit(v_madmk(self.v_pt(12), self.v_pt(12), -0.7071067690849304, self.v_pt( 4)))  # 12
        self._emit(v_madmk(self.v_pt( 9), self.v_pt( 9),  0.7071067690849304, self.v_pt( 1)))  # 9
        self._emit(v_madmk(self.v_pt(13), self.v_pt(13), -0.7071067690849304, self.v_pt( 5)))  # 13
        self._emit(f"v_add_f32 v[{self.v_pt( 7)}], v[{self.v_tt( 5)}], v[{self.v_tt( 6)}]") # 7
        self._emit(f"v_add_f32 v[{self.v_pt( 0)}], v[{self.v_tt( 0)}], v[{self.v_tt( 2)}]") # 0
        self._emit(f"v_add_f32 v[{self.v_pt( 4)}], v[{self.v_tt( 4)}], v[{self.v_tt( 7)}]") # 4
        self._emit(f"v_add_f32 v[{self.v_pt( 1)}], v[{self.v_tt( 1)}], v[{self.v_tt( 3)}]") # 1
        self._emit(f"v_sub_f32 v[{self.v_pt( 5)}], v[{self.v_tt( 5)}], v[{self.v_tt( 6)}]") # 5
        


class fft8_bwd_sched_t(macro_base_t):
    '''
    handcraft scheduled fft implementation
    target no pipeline stall if possible
    '''
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_pt")    # continus pixels
        self.declare_arg("v_tt")    # half the size of above pixel

    def expr(self):
        # omega 2_0
        self._emit(f"v_sub_f32 v[{self.v_tt( 0)}], v[{self.v_pt( 0)}], v[{self.v_pt( 8)}]") # 8
        self._emit(f"v_sub_f32 v[{self.v_tt( 1)}], v[{self.v_pt( 1)}], v[{self.v_pt( 9)}]") # 9
        self._emit(f"v_sub_f32 v[{self.v_tt( 2)}], v[{self.v_pt( 2)}], v[{self.v_pt(10)}]") # 10
        self._emit(f"v_sub_f32 v[{self.v_tt( 3)}], v[{self.v_pt( 3)}], v[{self.v_pt(11)}]") # 11
        self._emit(f"v_sub_f32 v[{self.v_tt( 4)}], v[{self.v_pt( 4)}], v[{self.v_pt(12)}]") # 12
        self._emit(f"v_sub_f32 v[{self.v_tt( 5)}], v[{self.v_pt( 5)}], v[{self.v_pt(13)}]") # 13
        self._emit(f"v_sub_f32 v[{self.v_tt( 6)}], v[{self.v_pt( 6)}], v[{self.v_pt(14)}]") # 14
        self._emit(f"v_sub_f32 v[{self.v_tt( 7)}], v[{self.v_pt( 7)}], v[{self.v_pt(15)}]") # 15
        self._emit(f"v_add_f32 v[{self.v_pt( 8)}], v[{self.v_pt( 0)}], v[{self.v_pt( 8)}]") # 0
        self._emit(f"v_add_f32 v[{self.v_pt( 9)}], v[{self.v_pt( 1)}], v[{self.v_pt( 9)}]") # 1
        self._emit(f"v_add_f32 v[{self.v_pt(10)}], v[{self.v_pt( 2)}], v[{self.v_pt(10)}]") # 2
        self._emit(f"v_add_f32 v[{self.v_pt(11)}], v[{self.v_pt( 3)}], v[{self.v_pt(11)}]") # 3
        self._emit(f"v_add_f32 v[{self.v_pt(12)}], v[{self.v_pt( 4)}], v[{self.v_pt(12)}]") # 4
        self._emit(f"v_add_f32 v[{self.v_pt(13)}], v[{self.v_pt( 5)}], v[{self.v_pt(13)}]") # 5
        self._emit(f"v_add_f32 v[{self.v_pt(14)}], v[{self.v_pt( 6)}], v[{self.v_pt(14)}]") # 6
        self._emit(f"v_add_f32 v[{self.v_pt(15)}], v[{self.v_pt( 7)}], v[{self.v_pt(15)}]") # 7

        # omega 4_0, 4_1
        self._emit(f"v_sub_f32 v[{self.v_pt( 0)}], v[{self.v_tt( 0)}], v[{self.v_tt( 5)}]") # 8
        self._emit(f"v_add_f32 v[{self.v_pt( 1)}], v[{self.v_tt( 1)}], v[{self.v_tt( 4)}]") # 9
        self._emit(f"v_add_f32 v[{self.v_pt( 4)}], v[{self.v_tt( 0)}], v[{self.v_tt( 5)}]") # 12
        self._emit(f"v_sub_f32 v[{self.v_pt( 5)}], v[{self.v_tt( 1)}], v[{self.v_tt( 4)}]") # 13
        self._emit(f"v_sub_f32 v[{self.v_pt( 2)}], v[{self.v_tt( 2)}], v[{self.v_tt( 7)}]") # 10
        self._emit(f"v_add_f32 v[{self.v_pt( 3)}], v[{self.v_tt( 3)}], v[{self.v_tt( 6)}]") # 11
        self._emit(f"v_add_f32 v[{self.v_pt( 6)}], v[{self.v_tt( 2)}], v[{self.v_tt( 7)}]") # 14
        self._emit(f"v_sub_f32 v[{self.v_pt( 7)}], v[{self.v_tt( 3)}], v[{self.v_tt( 6)}]") # 15
        self._emit(f"v_add_f32 v[{self.v_tt( 0)}], v[{self.v_pt( 8)}], v[{self.v_pt(12)}]") # 0
        self._emit(f"v_add_f32 v[{self.v_tt( 1)}], v[{self.v_pt( 9)}], v[{self.v_pt(13)}]") # 1
        self._emit(f"v_sub_f32 v[{self.v_tt( 4)}], v[{self.v_pt( 8)}], v[{self.v_pt(12)}]") # 4
        self._emit(f"v_sub_f32 v[{self.v_tt( 5)}], v[{self.v_pt( 9)}], v[{self.v_pt(13)}]") # 5
        self._emit(f"v_add_f32 v[{self.v_tt( 2)}], v[{self.v_pt(10)}], v[{self.v_pt(14)}]") # 2
        self._emit(f"v_add_f32 v[{self.v_tt( 3)}], v[{self.v_pt(11)}], v[{self.v_pt(15)}]") # 3
        self._emit(f"v_sub_f32 v[{self.v_tt( 6)}], v[{self.v_pt(10)}], v[{self.v_pt(14)}]") # 6

        self._emit(f"v_sub_f32 v[{self.v_pt( 8)}], v[{self.v_pt( 2)}], v[{self.v_pt( 3)}]") #   4 -- 5 tmp_0
        self._emit(f"v_add_f32 v[{self.v_pt( 9)}], v[{self.v_pt( 3)}], v[{self.v_pt( 2)}]") #   4 -- 5 tmp_1
        self._emit(f"v_add_f32 v[{self.v_pt(12)}], v[{self.v_pt( 6)}], v[{self.v_pt( 7)}]") #   6 -- 7 tmp_1
        self._emit(f"v_sub_f32 v[{self.v_pt(13)}], v[{self.v_pt( 7)}], v[{self.v_pt( 6)}]") #   6 -- 7 tmp_0
        self._emit(f"v_sub_f32 v[{self.v_tt( 7)}], v[{self.v_pt(11)}], v[{self.v_pt(15)}]") # 7

        # omega 8_0, 8_2, 8_1, 8_3
        self._emit(v_madmk(self.v_pt(10), self.v_pt( 8), -0.7071067690849304, self.v_pt( 0)))  # 10
        self._emit(f"v_sub_f32 v[{self.v_pt( 2)}], v[{self.v_tt( 0)}], v[{self.v_tt( 2)}]") # 2
        self._emit(f"v_sub_f32 v[{self.v_pt( 3)}], v[{self.v_tt( 1)}], v[{self.v_tt( 3)}]") # 3
        self._emit(v_madmk(self.v_pt(14), self.v_pt(12),  0.7071067690849304, self.v_pt( 4)))  # 14
        self._emit(v_madmk(self.v_pt(11), self.v_pt( 9), -0.7071067690849304, self.v_pt( 1)))  # 11
        self._emit(v_madmk(self.v_pt(15), self.v_pt(13),  0.7071067690849304, self.v_pt( 5)))  # 15
        self._emit(v_madmk(self.v_pt( 8), self.v_pt( 8),  0.7071067690849304, self.v_pt( 0)))  # 8
        self._emit(f"v_add_f32 v[{self.v_pt( 6)}], v[{self.v_tt( 4)}], v[{self.v_tt( 7)}]") # 6
        self._emit(v_madmk(self.v_pt(12), self.v_pt(12), -0.7071067690849304, self.v_pt( 4)))  # 12
        self._emit(v_madmk(self.v_pt( 9), self.v_pt( 9),  0.7071067690849304, self.v_pt( 1)))  # 9
        self._emit(v_madmk(self.v_pt(13), self.v_pt(13), -0.7071067690849304, self.v_pt( 5)))  # 13
        self._emit(f"v_sub_f32 v[{self.v_pt( 7)}], v[{self.v_tt( 5)}], v[{self.v_tt( 6)}]") # 7
        self._emit(f"v_add_f32 v[{self.v_pt( 0)}], v[{self.v_tt( 0)}], v[{self.v_tt( 2)}]") # 0
        self._emit(f"v_sub_f32 v[{self.v_pt( 4)}], v[{self.v_tt( 4)}], v[{self.v_tt( 7)}]") # 4
        self._emit(f"v_add_f32 v[{self.v_pt( 1)}], v[{self.v_tt( 1)}], v[{self.v_tt( 3)}]") # 1
        self._emit(f"v_add_f32 v[{self.v_pt( 5)}], v[{self.v_tt( 5)}], v[{self.v_tt( 6)}]") # 5

class fft16_fwd_sched_t(macro_base_t):
    '''
    handcraft scheduled fft implementation
    target no pipeline stall if possible
    '''
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_pt")    # continus pixels
        self.declare_arg("v_tt")    # half the size of above pixel

    def expr(self):
        # omega 2_0
        self._emit(f"v_sub_f32 v[{self.v_tt( 0)}], v[{self.v_pt( 0)}], v[{self.v_pt(16)}]") # 16
        self._emit(f"v_sub_f32 v[{self.v_tt( 1)}], v[{self.v_pt( 1)}], v[{self.v_pt(17)}]") # 17
        self._emit(f"v_sub_f32 v[{self.v_tt( 2)}], v[{self.v_pt( 2)}], v[{self.v_pt(18)}]") # 18
        self._emit(f"v_sub_f32 v[{self.v_tt( 3)}], v[{self.v_pt( 3)}], v[{self.v_pt(19)}]") # 19
        self._emit(f"v_sub_f32 v[{self.v_tt( 4)}], v[{self.v_pt( 4)}], v[{self.v_pt(20)}]") # 20
        self._emit(f"v_sub_f32 v[{self.v_tt( 5)}], v[{self.v_pt( 5)}], v[{self.v_pt(21)}]") # 21
        self._emit(f"v_sub_f32 v[{self.v_tt( 6)}], v[{self.v_pt( 6)}], v[{self.v_pt(22)}]") # 22
        self._emit(f"v_sub_f32 v[{self.v_tt( 7)}], v[{self.v_pt( 7)}], v[{self.v_pt(23)}]") # 23
        self._emit(f"v_sub_f32 v[{self.v_tt( 8)}], v[{self.v_pt( 8)}], v[{self.v_pt(24)}]") # 24
        self._emit(f"v_sub_f32 v[{self.v_tt( 9)}], v[{self.v_pt( 9)}], v[{self.v_pt(25)}]") # 25
        self._emit(f"v_sub_f32 v[{self.v_tt(10)}], v[{self.v_pt(10)}], v[{self.v_pt(26)}]") # 26
        self._emit(f"v_sub_f32 v[{self.v_tt(11)}], v[{self.v_pt(11)}], v[{self.v_pt(27)}]") # 27
        self._emit(f"v_sub_f32 v[{self.v_tt(12)}], v[{self.v_pt(12)}], v[{self.v_pt(28)}]") # 28
        self._emit(f"v_sub_f32 v[{self.v_tt(13)}], v[{self.v_pt(13)}], v[{self.v_pt(29)}]") # 29
        self._emit(f"v_sub_f32 v[{self.v_tt(14)}], v[{self.v_pt(14)}], v[{self.v_pt(30)}]") # 30
        self._emit(f"v_sub_f32 v[{self.v_tt(15)}], v[{self.v_pt(15)}], v[{self.v_pt(31)}]") # 31

        self._emit(f"v_add_f32 v[{self.v_pt( 0)}], v[{self.v_pt( 0)}], v[{self.v_pt(16)}]") # 0
        self._emit(f"v_add_f32 v[{self.v_pt( 1)}], v[{self.v_pt( 1)}], v[{self.v_pt(17)}]") # 1
        self._emit(f"v_add_f32 v[{self.v_pt( 2)}], v[{self.v_pt( 2)}], v[{self.v_pt(18)}]") # 2
        self._emit(f"v_add_f32 v[{self.v_pt( 3)}], v[{self.v_pt( 3)}], v[{self.v_pt(19)}]") # 3
        self._emit(f"v_add_f32 v[{self.v_pt( 4)}], v[{self.v_pt( 4)}], v[{self.v_pt(20)}]") # 4
        self._emit(f"v_add_f32 v[{self.v_pt( 5)}], v[{self.v_pt( 5)}], v[{self.v_pt(21)}]") # 5
        self._emit(f"v_add_f32 v[{self.v_pt( 6)}], v[{self.v_pt( 6)}], v[{self.v_pt(22)}]") # 6
        self._emit(f"v_add_f32 v[{self.v_pt( 7)}], v[{self.v_pt( 7)}], v[{self.v_pt(23)}]") # 7
        self._emit(f"v_add_f32 v[{self.v_pt( 8)}], v[{self.v_pt( 8)}], v[{self.v_pt(24)}]") # 8
        self._emit(f"v_add_f32 v[{self.v_pt( 9)}], v[{self.v_pt( 9)}], v[{self.v_pt(25)}]") # 9
        self._emit(f"v_add_f32 v[{self.v_pt(10)}], v[{self.v_pt(10)}], v[{self.v_pt(26)}]") # 10
        self._emit(f"v_add_f32 v[{self.v_pt(11)}], v[{self.v_pt(11)}], v[{self.v_pt(27)}]") # 11
        self._emit(f"v_add_f32 v[{self.v_pt(12)}], v[{self.v_pt(12)}], v[{self.v_pt(28)}]") # 12
        self._emit(f"v_add_f32 v[{self.v_pt(13)}], v[{self.v_pt(13)}], v[{self.v_pt(29)}]") # 13
        self._emit(f"v_add_f32 v[{self.v_pt(14)}], v[{self.v_pt(14)}], v[{self.v_pt(30)}]") # 14
        self._emit(f"v_add_f32 v[{self.v_pt(15)}], v[{self.v_pt(15)}], v[{self.v_pt(31)}]") # 15


        # omega 4_0, 4_1
        self._emit(f"v_add_f32 v[{self.v_pt(16)}], v[{self.v_tt( 0)}], v[{self.v_tt( 9)}]") # 16
        self._emit(f"v_sub_f32 v[{self.v_pt(17)}], v[{self.v_tt( 1)}], v[{self.v_tt( 8)}]") # 17
        self._emit(f"v_add_f32 v[{self.v_pt(18)}], v[{self.v_tt( 2)}], v[{self.v_tt(11)}]") # 18
        self._emit(f"v_sub_f32 v[{self.v_pt(19)}], v[{self.v_tt( 3)}], v[{self.v_tt(10)}]") # 19
        self._emit(f"v_add_f32 v[{self.v_pt(20)}], v[{self.v_tt( 4)}], v[{self.v_tt(13)}]") # 20
        self._emit(f"v_sub_f32 v[{self.v_pt(21)}], v[{self.v_tt( 5)}], v[{self.v_tt(12)}]") # 21
        self._emit(f"v_add_f32 v[{self.v_pt(22)}], v[{self.v_tt( 6)}], v[{self.v_tt(15)}]") # 22
        self._emit(f"v_sub_f32 v[{self.v_pt(23)}], v[{self.v_tt( 7)}], v[{self.v_tt(14)}]") # 23
        self._emit(f"v_sub_f32 v[{self.v_pt(24)}], v[{self.v_tt( 0)}], v[{self.v_tt( 9)}]") # 24
        self._emit(f"v_add_f32 v[{self.v_pt(25)}], v[{self.v_tt( 1)}], v[{self.v_tt( 8)}]") # 25
        self._emit(f"v_sub_f32 v[{self.v_pt(26)}], v[{self.v_tt( 2)}], v[{self.v_tt(11)}]") # 26
        self._emit(f"v_add_f32 v[{self.v_pt(27)}], v[{self.v_tt( 3)}], v[{self.v_tt(10)}]") # 27
        self._emit(f"v_sub_f32 v[{self.v_pt(28)}], v[{self.v_tt( 4)}], v[{self.v_tt(13)}]") # 28
        self._emit(f"v_add_f32 v[{self.v_pt(29)}], v[{self.v_tt( 5)}], v[{self.v_tt(12)}]") # 29
        self._emit(f"v_sub_f32 v[{self.v_pt(30)}], v[{self.v_tt( 6)}], v[{self.v_tt(15)}]") # 30
        self._emit(f"v_add_f32 v[{self.v_pt(31)}], v[{self.v_tt( 7)}], v[{self.v_tt(14)}]") # 31

        self._emit(f"v_add_f32 v[{self.v_tt( 0)}], v[{self.v_pt( 0)}], v[{self.v_pt( 8)}]") #  0
        self._emit(f"v_add_f32 v[{self.v_tt( 1)}], v[{self.v_pt( 1)}], v[{self.v_pt( 9)}]") #  1
        self._emit(f"v_add_f32 v[{self.v_tt( 2)}], v[{self.v_pt( 2)}], v[{self.v_pt(10)}]") #  2
        self._emit(f"v_add_f32 v[{self.v_tt( 3)}], v[{self.v_pt( 3)}], v[{self.v_pt(11)}]") #  3
        self._emit(f"v_add_f32 v[{self.v_tt( 4)}], v[{self.v_pt( 4)}], v[{self.v_pt(12)}]") #  4
        self._emit(f"v_add_f32 v[{self.v_tt( 5)}], v[{self.v_pt( 5)}], v[{self.v_pt(13)}]") #  5
        self._emit(f"v_add_f32 v[{self.v_tt( 6)}], v[{self.v_pt( 6)}], v[{self.v_pt(14)}]") #  6
        self._emit(f"v_add_f32 v[{self.v_tt( 7)}], v[{self.v_pt( 7)}], v[{self.v_pt(15)}]") #  7
        self._emit(f"v_sub_f32 v[{self.v_tt( 8)}], v[{self.v_pt( 0)}], v[{self.v_pt( 8)}]") #  8
        self._emit(f"v_sub_f32 v[{self.v_tt( 9)}], v[{self.v_pt( 1)}], v[{self.v_pt( 9)}]") #  9
        self._emit(f"v_sub_f32 v[{self.v_tt(10)}], v[{self.v_pt( 2)}], v[{self.v_pt(10)}]") # 10
        self._emit(f"v_sub_f32 v[{self.v_tt(11)}], v[{self.v_pt( 3)}], v[{self.v_pt(11)}]") # 11
        self._emit(f"v_sub_f32 v[{self.v_tt(12)}], v[{self.v_pt( 4)}], v[{self.v_pt(12)}]") # 12
        self._emit(f"v_sub_f32 v[{self.v_tt(13)}], v[{self.v_pt( 5)}], v[{self.v_pt(13)}]") # 13
        self._emit(f"v_sub_f32 v[{self.v_tt(14)}], v[{self.v_pt( 6)}], v[{self.v_pt(14)}]") # 14
        self._emit(f"v_sub_f32 v[{self.v_tt(15)}], v[{self.v_pt( 7)}], v[{self.v_pt(15)}]") # 15

        # omega 8_0, 8_2, 8_1, 8_3
        self._emit(f"v_add_f32 v[{self.v_pt( 0)}], v[{self.v_pt(20)}], v[{self.v_pt(21)}]") # 8 -- 10 tmp0
        self._emit(f"v_sub_f32 v[{self.v_pt( 1)}], v[{self.v_pt(21)}], v[{self.v_pt(20)}]") # 8 -- 10 tmp1
        self._emit(f"v_add_f32 v[{self.v_pt( 2)}], v[{self.v_pt(22)}], v[{self.v_pt(23)}]") # 9 -- 11 tmp0
        self._emit(f"v_sub_f32 v[{self.v_pt( 3)}], v[{self.v_pt(23)}], v[{self.v_pt(22)}]") # 9 -- 11 tmp1
        self._emit(f"v_sub_f32 v[{self.v_pt( 4)}], v[{self.v_pt(28)}], v[{self.v_pt(29)}]") # 12 -- 14 tmp0
        self._emit(f"v_add_f32 v[{self.v_pt( 5)}], v[{self.v_pt(29)}], v[{self.v_pt(28)}]") # 12 -- 14 tmp1
        self._emit(f"v_sub_f32 v[{self.v_pt( 6)}], v[{self.v_pt(30)}], v[{self.v_pt(31)}]") # 13 -- 15 tmp0
        self._emit(f"v_add_f32 v[{self.v_pt( 7)}], v[{self.v_pt(31)}], v[{self.v_pt(30)}]") # 13 -- 15 tmp1

        self._emit(v_madmk(self.v_pt(20), self.v_pt( 0), -0.7071067690849304, self.v_pt(16))) # 20
        self._emit(v_madmk(self.v_pt(21), self.v_pt( 1), -0.7071067690849304, self.v_pt(17))) # 21
        self._emit(v_madmk(self.v_pt(22), self.v_pt( 2), -0.7071067690849304, self.v_pt(18))) # 22
        self._emit(v_madmk(self.v_pt(23), self.v_pt( 3), -0.7071067690849304, self.v_pt(19))) # 23
        self._emit(v_madmk(self.v_pt(28), self.v_pt( 4),  0.7071067690849304, self.v_pt(24))) # 28
        self._emit(v_madmk(self.v_pt(29), self.v_pt( 5),  0.7071067690849304, self.v_pt(25))) # 29
        self._emit(v_madmk(self.v_pt(30), self.v_pt( 6),  0.7071067690849304, self.v_pt(26))) # 30
        self._emit(v_madmk(self.v_pt(31), self.v_pt( 7),  0.7071067690849304, self.v_pt(27))) # 31

        self._emit(v_madmk(self.v_pt(16), self.v_pt( 0),  0.7071067690849304, self.v_pt(16))) # 16
        self._emit(v_madmk(self.v_pt(17), self.v_pt( 1),  0.7071067690849304, self.v_pt(17))) # 17
        self._emit(v_madmk(self.v_pt(18), self.v_pt( 2),  0.7071067690849304, self.v_pt(18))) # 18
        self._emit(v_madmk(self.v_pt(19), self.v_pt( 3),  0.7071067690849304, self.v_pt(19))) # 19
        self._emit(v_madmk(self.v_pt(24), self.v_pt( 4), -0.7071067690849304, self.v_pt(24))) # 24
        self._emit(v_madmk(self.v_pt(25), self.v_pt( 5), -0.7071067690849304, self.v_pt(25))) # 25
        self._emit(v_madmk(self.v_pt(26), self.v_pt( 6), -0.7071067690849304, self.v_pt(26))) # 26
        self._emit(v_madmk(self.v_pt(27), self.v_pt( 7), -0.7071067690849304, self.v_pt(27))) # 27

        self._emit(f"v_add_f32 v[{self.v_pt( 8)}], v[{self.v_tt( 8)}], v[{self.v_tt(13)}]") #  8
        self._emit(f"v_sub_f32 v[{self.v_pt( 9)}], v[{self.v_tt( 9)}], v[{self.v_tt(12)}]") #  9
        self._emit(f"v_add_f32 v[{self.v_pt(10)}], v[{self.v_tt(10)}], v[{self.v_tt(15)}]") #  10
        self._emit(f"v_sub_f32 v[{self.v_pt(11)}], v[{self.v_tt(11)}], v[{self.v_tt(14)}]") #  11
        self._emit(f"v_sub_f32 v[{self.v_pt(12)}], v[{self.v_tt( 8)}], v[{self.v_tt(13)}]") #  12
        self._emit(f"v_add_f32 v[{self.v_pt(13)}], v[{self.v_tt( 9)}], v[{self.v_tt(12)}]") #  13
        self._emit(f"v_sub_f32 v[{self.v_pt(14)}], v[{self.v_tt(10)}], v[{self.v_tt(15)}]") #  14
        self._emit(f"v_add_f32 v[{self.v_pt(15)}], v[{self.v_tt(11)}], v[{self.v_tt(14)}]") #  15

        self._emit(f"v_add_f32 v[{self.v_pt( 0)}], v[{self.v_tt( 0)}], v[{self.v_tt( 4)}]") #  0
        self._emit(f"v_add_f32 v[{self.v_pt( 1)}], v[{self.v_tt( 1)}], v[{self.v_tt( 5)}]") #  1
        self._emit(f"v_sub_f32 v[{self.v_pt( 4)}], v[{self.v_tt( 0)}], v[{self.v_tt( 4)}]") #  4
        self._emit(f"v_sub_f32 v[{self.v_pt( 5)}], v[{self.v_tt( 1)}], v[{self.v_tt( 5)}]") #  5
        self._emit(f"v_add_f32 v[{self.v_tt(12)}], v[{self.v_tt( 2)}], v[{self.v_tt( 6)}]") #  2
        self._emit(f"v_add_f32 v[{self.v_tt(13)}], v[{self.v_tt( 3)}], v[{self.v_tt( 7)}]") #  3
        self._emit(f"v_sub_f32 v[{self.v_tt(14)}], v[{self.v_tt( 2)}], v[{self.v_tt( 6)}]") #  6
        self._emit(f"v_sub_f32 v[{self.v_tt(15)}], v[{self.v_tt( 3)}], v[{self.v_tt( 7)}]") #  7

        # omega 16_0,4,2,6,1,5,3,7
        self._emit(v_madmk(self.v_tt( 4), self.v_pt(19),  0.4142135679721832, self.v_pt(18))) #  8 --  9 tmp0
        self._emit(v_madmk(self.v_tt( 5), self.v_pt(18), -0.4142135679721832, self.v_pt(19))) #  8 --  9 tmp1
        self._emit(v_madmk(self.v_tt( 8), self.v_pt(27),  2.4142134189605713, self.v_pt(26))) # 12 -- 13 tmp0
        self._emit(v_madmk(self.v_tt( 9), self.v_pt(26), -2.4142134189605713, self.v_pt(27))) # 12 -- 13 tmp1
        self._emit(v_madmk(self.v_tt( 6), self.v_pt(23), -2.4142134189605713, self.v_pt(22))) # 10 -- 11 tmp0
        self._emit(v_madmk(self.v_tt( 7), self.v_pt(22),  2.4142134189605713, self.v_pt(23))) # 10 -- 11 tmp1
        self._emit(v_madmk(self.v_tt(10), self.v_pt(31), -0.4142135679721832, self.v_pt(30))) # 14 -- 15 tmp0
        self._emit(v_madmk(self.v_tt(11), self.v_pt(30),  0.4142135679721832, self.v_pt(31))) # 14 -- 15 tmp1
        self._emit(f"v_add_f32 v[{self.v_tt( 0)}], v[{self.v_pt(10)}], v[{self.v_pt(11)}]") #  4 --  5 tmp0
        self._emit(f"v_sub_f32 v[{self.v_tt( 1)}], v[{self.v_pt(11)}], v[{self.v_pt(10)}]") #  4 --  5 tmp1
        self._emit(f"v_sub_f32 v[{self.v_tt( 2)}], v[{self.v_pt(14)}], v[{self.v_pt(15)}]") #  6 --  7 tmp0
        self._emit(f"v_add_f32 v[{self.v_tt( 3)}], v[{self.v_pt(15)}], v[{self.v_pt(14)}]") #  6 --  7 tmp1

        self._emit(v_madmk(self.v_pt(18), self.v_tt( 4), -0.9238795042037964, self.v_pt(16))) # 18
        self._emit(v_madmk(self.v_pt(19), self.v_tt( 5), -0.9238795042037964, self.v_pt(17))) # 19
        self._emit(v_madmk(self.v_pt(22), self.v_tt( 6),  0.3826834261417389, self.v_pt(20))) # 22
        self._emit(v_madmk(self.v_pt(23), self.v_tt( 7),  0.3826834261417389, self.v_pt(21))) # 23
        self._emit(v_madmk(self.v_pt(26), self.v_tt( 8), -0.3826834261417389, self.v_pt(24))) # 26
        self._emit(v_madmk(self.v_pt(27), self.v_tt( 9), -0.3826834261417389, self.v_pt(25))) # 27
        self._emit(v_madmk(self.v_pt(30), self.v_tt(10),  0.9238795042037964, self.v_pt(28))) # 30
        self._emit(v_madmk(self.v_pt(31), self.v_tt(11),  0.9238795042037964, self.v_pt(29))) # 31
        self._emit(v_madmk(self.v_pt(16), self.v_tt( 4),  0.9238795042037964, self.v_pt(16))) # 16
        self._emit(v_madmk(self.v_pt(17), self.v_tt( 5),  0.9238795042037964, self.v_pt(17))) # 17
        self._emit(v_madmk(self.v_pt(20), self.v_tt( 6), -0.3826834261417389, self.v_pt(20))) # 20
        self._emit(v_madmk(self.v_pt(21), self.v_tt( 7), -0.3826834261417389, self.v_pt(21))) # 21
        self._emit(v_madmk(self.v_pt(24), self.v_tt( 8),  0.3826834261417389, self.v_pt(24))) # 24
        self._emit(v_madmk(self.v_pt(25), self.v_tt( 9),  0.3826834261417389, self.v_pt(25))) # 25
        self._emit(v_madmk(self.v_pt(28), self.v_tt(10), -0.9238795042037964, self.v_pt(28))) # 28
        self._emit(v_madmk(self.v_pt(29), self.v_tt(11), -0.9238795042037964, self.v_pt(29))) # 29

        self._emit(f"v_sub_f32 v[{self.v_pt( 2)}], v[{self.v_pt( 0)}], v[{self.v_tt(12)}]") #  2
        self._emit(f"v_sub_f32 v[{self.v_pt( 3)}], v[{self.v_pt( 1)}], v[{self.v_tt(13)}]") #  3

        self._emit(f"v_sub_f32 v[{self.v_pt( 6)}], v[{self.v_pt( 4)}], v[{self.v_tt(15)}]") #  6
        self._emit(f"v_add_f32 v[{self.v_pt( 7)}], v[{self.v_pt( 5)}], v[{self.v_tt(14)}]") #  7
        
        self._emit(v_madmk(self.v_pt(10), self.v_tt( 0), -0.7071067690849304, self.v_pt( 8))) # 10
        self._emit(v_madmk(self.v_pt(11), self.v_tt( 1), -0.7071067690849304, self.v_pt( 9))) # 11
        
        self._emit(v_madmk(self.v_pt(14), self.v_tt( 2),  0.7071067690849304, self.v_pt(12))) # 14
        self._emit(v_madmk(self.v_pt(15), self.v_tt( 3),  0.7071067690849304, self.v_pt(13))) # 15

        self._emit(f"v_add_f32 v[{self.v_pt( 0)}], v[{self.v_pt( 0)}], v[{self.v_tt(12)}]") #  0
        self._emit(f"v_add_f32 v[{self.v_pt( 1)}], v[{self.v_pt( 1)}], v[{self.v_tt(13)}]") #  1

        self._emit(f"v_add_f32 v[{self.v_pt( 4)}], v[{self.v_pt( 4)}], v[{self.v_tt(15)}]") #  4
        self._emit(f"v_sub_f32 v[{self.v_pt( 5)}], v[{self.v_pt( 5)}], v[{self.v_tt(14)}]") #  5

        self._emit(v_madmk(self.v_pt( 8), self.v_tt( 0),  0.7071067690849304, self.v_pt( 8))) #  8
        self._emit(v_madmk(self.v_pt( 9), self.v_tt( 1),  0.7071067690849304, self.v_pt( 9))) #  9

        self._emit(v_madmk(self.v_pt(12), self.v_tt( 2), -0.7071067690849304, self.v_pt(12))) # 12
        self._emit(v_madmk(self.v_pt(13), self.v_tt( 3), -0.7071067690849304, self.v_pt(13))) # 13


class fft16_bwd_sched_t(macro_base_t):
    '''
    handcraft scheduled fft implementation
    target no pipeline stall if possible
    '''
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_pt")    # continus pixels
        self.declare_arg("v_tt")    # half the size of above pixel

    def expr(self):
        # omega 2_0
        self._emit(f"v_sub_f32 v[{self.v_tt( 0)}], v[{self.v_pt( 0)}], v[{self.v_pt(16)}]") # 16
        self._emit(f"v_sub_f32 v[{self.v_tt( 1)}], v[{self.v_pt( 1)}], v[{self.v_pt(17)}]") # 17
        self._emit(f"v_sub_f32 v[{self.v_tt( 2)}], v[{self.v_pt( 2)}], v[{self.v_pt(18)}]") # 18
        self._emit(f"v_sub_f32 v[{self.v_tt( 3)}], v[{self.v_pt( 3)}], v[{self.v_pt(19)}]") # 19
        self._emit(f"v_sub_f32 v[{self.v_tt( 4)}], v[{self.v_pt( 4)}], v[{self.v_pt(20)}]") # 20
        self._emit(f"v_sub_f32 v[{self.v_tt( 5)}], v[{self.v_pt( 5)}], v[{self.v_pt(21)}]") # 21
        self._emit(f"v_sub_f32 v[{self.v_tt( 6)}], v[{self.v_pt( 6)}], v[{self.v_pt(22)}]") # 22
        self._emit(f"v_sub_f32 v[{self.v_tt( 7)}], v[{self.v_pt( 7)}], v[{self.v_pt(23)}]") # 23
        self._emit(f"v_sub_f32 v[{self.v_tt( 8)}], v[{self.v_pt( 8)}], v[{self.v_pt(24)}]") # 24
        self._emit(f"v_sub_f32 v[{self.v_tt( 9)}], v[{self.v_pt( 9)}], v[{self.v_pt(25)}]") # 25
        self._emit(f"v_sub_f32 v[{self.v_tt(10)}], v[{self.v_pt(10)}], v[{self.v_pt(26)}]") # 26
        self._emit(f"v_sub_f32 v[{self.v_tt(11)}], v[{self.v_pt(11)}], v[{self.v_pt(27)}]") # 27
        self._emit(f"v_sub_f32 v[{self.v_tt(12)}], v[{self.v_pt(12)}], v[{self.v_pt(28)}]") # 28
        self._emit(f"v_sub_f32 v[{self.v_tt(13)}], v[{self.v_pt(13)}], v[{self.v_pt(29)}]") # 29
        self._emit(f"v_sub_f32 v[{self.v_tt(14)}], v[{self.v_pt(14)}], v[{self.v_pt(30)}]") # 30
        self._emit(f"v_sub_f32 v[{self.v_tt(15)}], v[{self.v_pt(15)}], v[{self.v_pt(31)}]") # 31

        self._emit(f"v_add_f32 v[{self.v_pt( 0)}], v[{self.v_pt( 0)}], v[{self.v_pt(16)}]") # 0
        self._emit(f"v_add_f32 v[{self.v_pt( 1)}], v[{self.v_pt( 1)}], v[{self.v_pt(17)}]") # 1
        self._emit(f"v_add_f32 v[{self.v_pt( 2)}], v[{self.v_pt( 2)}], v[{self.v_pt(18)}]") # 2
        self._emit(f"v_add_f32 v[{self.v_pt( 3)}], v[{self.v_pt( 3)}], v[{self.v_pt(19)}]") # 3
        self._emit(f"v_add_f32 v[{self.v_pt( 4)}], v[{self.v_pt( 4)}], v[{self.v_pt(20)}]") # 4
        self._emit(f"v_add_f32 v[{self.v_pt( 5)}], v[{self.v_pt( 5)}], v[{self.v_pt(21)}]") # 5
        self._emit(f"v_add_f32 v[{self.v_pt( 6)}], v[{self.v_pt( 6)}], v[{self.v_pt(22)}]") # 6
        self._emit(f"v_add_f32 v[{self.v_pt( 7)}], v[{self.v_pt( 7)}], v[{self.v_pt(23)}]") # 7
        self._emit(f"v_add_f32 v[{self.v_pt( 8)}], v[{self.v_pt( 8)}], v[{self.v_pt(24)}]") # 8
        self._emit(f"v_add_f32 v[{self.v_pt( 9)}], v[{self.v_pt( 9)}], v[{self.v_pt(25)}]") # 9
        self._emit(f"v_add_f32 v[{self.v_pt(10)}], v[{self.v_pt(10)}], v[{self.v_pt(26)}]") # 10
        self._emit(f"v_add_f32 v[{self.v_pt(11)}], v[{self.v_pt(11)}], v[{self.v_pt(27)}]") # 11
        self._emit(f"v_add_f32 v[{self.v_pt(12)}], v[{self.v_pt(12)}], v[{self.v_pt(28)}]") # 12
        self._emit(f"v_add_f32 v[{self.v_pt(13)}], v[{self.v_pt(13)}], v[{self.v_pt(29)}]") # 13
        self._emit(f"v_add_f32 v[{self.v_pt(14)}], v[{self.v_pt(14)}], v[{self.v_pt(30)}]") # 14
        self._emit(f"v_add_f32 v[{self.v_pt(15)}], v[{self.v_pt(15)}], v[{self.v_pt(31)}]") # 15


        # omega 4_0, 4_1
        self._emit(f"v_sub_f32 v[{self.v_pt(16)}], v[{self.v_tt( 0)}], v[{self.v_tt( 9)}]") # 16
        self._emit(f"v_add_f32 v[{self.v_pt(17)}], v[{self.v_tt( 1)}], v[{self.v_tt( 8)}]") # 17
        self._emit(f"v_sub_f32 v[{self.v_pt(18)}], v[{self.v_tt( 2)}], v[{self.v_tt(11)}]") # 18
        self._emit(f"v_add_f32 v[{self.v_pt(19)}], v[{self.v_tt( 3)}], v[{self.v_tt(10)}]") # 19
        self._emit(f"v_sub_f32 v[{self.v_pt(20)}], v[{self.v_tt( 4)}], v[{self.v_tt(13)}]") # 20
        self._emit(f"v_add_f32 v[{self.v_pt(21)}], v[{self.v_tt( 5)}], v[{self.v_tt(12)}]") # 21
        self._emit(f"v_sub_f32 v[{self.v_pt(22)}], v[{self.v_tt( 6)}], v[{self.v_tt(15)}]") # 22
        self._emit(f"v_add_f32 v[{self.v_pt(23)}], v[{self.v_tt( 7)}], v[{self.v_tt(14)}]") # 23
        self._emit(f"v_add_f32 v[{self.v_pt(24)}], v[{self.v_tt( 0)}], v[{self.v_tt( 9)}]") # 24
        self._emit(f"v_sub_f32 v[{self.v_pt(25)}], v[{self.v_tt( 1)}], v[{self.v_tt( 8)}]") # 25
        self._emit(f"v_add_f32 v[{self.v_pt(26)}], v[{self.v_tt( 2)}], v[{self.v_tt(11)}]") # 26
        self._emit(f"v_sub_f32 v[{self.v_pt(27)}], v[{self.v_tt( 3)}], v[{self.v_tt(10)}]") # 27
        self._emit(f"v_add_f32 v[{self.v_pt(28)}], v[{self.v_tt( 4)}], v[{self.v_tt(13)}]") # 28
        self._emit(f"v_sub_f32 v[{self.v_pt(29)}], v[{self.v_tt( 5)}], v[{self.v_tt(12)}]") # 29
        self._emit(f"v_add_f32 v[{self.v_pt(30)}], v[{self.v_tt( 6)}], v[{self.v_tt(15)}]") # 30
        self._emit(f"v_sub_f32 v[{self.v_pt(31)}], v[{self.v_tt( 7)}], v[{self.v_tt(14)}]") # 31

        self._emit(f"v_add_f32 v[{self.v_tt( 0)}], v[{self.v_pt( 0)}], v[{self.v_pt( 8)}]") #  0
        self._emit(f"v_add_f32 v[{self.v_tt( 1)}], v[{self.v_pt( 1)}], v[{self.v_pt( 9)}]") #  1
        self._emit(f"v_add_f32 v[{self.v_tt( 2)}], v[{self.v_pt( 2)}], v[{self.v_pt(10)}]") #  2
        self._emit(f"v_add_f32 v[{self.v_tt( 3)}], v[{self.v_pt( 3)}], v[{self.v_pt(11)}]") #  3
        self._emit(f"v_add_f32 v[{self.v_tt( 4)}], v[{self.v_pt( 4)}], v[{self.v_pt(12)}]") #  4
        self._emit(f"v_add_f32 v[{self.v_tt( 5)}], v[{self.v_pt( 5)}], v[{self.v_pt(13)}]") #  5
        self._emit(f"v_add_f32 v[{self.v_tt( 6)}], v[{self.v_pt( 6)}], v[{self.v_pt(14)}]") #  6
        self._emit(f"v_add_f32 v[{self.v_tt( 7)}], v[{self.v_pt( 7)}], v[{self.v_pt(15)}]") #  7
        self._emit(f"v_sub_f32 v[{self.v_tt( 8)}], v[{self.v_pt( 0)}], v[{self.v_pt( 8)}]") #  8
        self._emit(f"v_sub_f32 v[{self.v_tt( 9)}], v[{self.v_pt( 1)}], v[{self.v_pt( 9)}]") #  9
        self._emit(f"v_sub_f32 v[{self.v_tt(10)}], v[{self.v_pt( 2)}], v[{self.v_pt(10)}]") # 10
        self._emit(f"v_sub_f32 v[{self.v_tt(11)}], v[{self.v_pt( 3)}], v[{self.v_pt(11)}]") # 11
        self._emit(f"v_sub_f32 v[{self.v_tt(12)}], v[{self.v_pt( 4)}], v[{self.v_pt(12)}]") # 12
        self._emit(f"v_sub_f32 v[{self.v_tt(13)}], v[{self.v_pt( 5)}], v[{self.v_pt(13)}]") # 13
        self._emit(f"v_sub_f32 v[{self.v_tt(14)}], v[{self.v_pt( 6)}], v[{self.v_pt(14)}]") # 14
        self._emit(f"v_sub_f32 v[{self.v_tt(15)}], v[{self.v_pt( 7)}], v[{self.v_pt(15)}]") # 15

        # omega 8_0, 8_2, 8_1, 8_3
        self._emit(f"v_sub_f32 v[{self.v_pt( 0)}], v[{self.v_pt(20)}], v[{self.v_pt(21)}]") # 8 -- 10 tmp0
        self._emit(f"v_add_f32 v[{self.v_pt( 1)}], v[{self.v_pt(21)}], v[{self.v_pt(20)}]") # 8 -- 10 tmp1
        self._emit(f"v_sub_f32 v[{self.v_pt( 2)}], v[{self.v_pt(22)}], v[{self.v_pt(23)}]") # 9 -- 11 tmp0
        self._emit(f"v_add_f32 v[{self.v_pt( 3)}], v[{self.v_pt(23)}], v[{self.v_pt(22)}]") # 9 -- 11 tmp1
        self._emit(f"v_add_f32 v[{self.v_pt( 4)}], v[{self.v_pt(28)}], v[{self.v_pt(29)}]") # 12 -- 14 tmp0
        self._emit(f"v_sub_f32 v[{self.v_pt( 5)}], v[{self.v_pt(29)}], v[{self.v_pt(28)}]") # 12 -- 14 tmp1
        self._emit(f"v_add_f32 v[{self.v_pt( 6)}], v[{self.v_pt(30)}], v[{self.v_pt(31)}]") # 13 -- 15 tmp0
        self._emit(f"v_sub_f32 v[{self.v_pt( 7)}], v[{self.v_pt(31)}], v[{self.v_pt(30)}]") # 13 -- 15 tmp1

        self._emit(v_madmk(self.v_pt(20), self.v_pt( 0), -0.7071067690849304, self.v_pt(16))) # 20
        self._emit(v_madmk(self.v_pt(21), self.v_pt( 1), -0.7071067690849304, self.v_pt(17))) # 21
        self._emit(v_madmk(self.v_pt(22), self.v_pt( 2), -0.7071067690849304, self.v_pt(18))) # 22
        self._emit(v_madmk(self.v_pt(23), self.v_pt( 3), -0.7071067690849304, self.v_pt(19))) # 23
        self._emit(v_madmk(self.v_pt(28), self.v_pt( 4),  0.7071067690849304, self.v_pt(24))) # 28
        self._emit(v_madmk(self.v_pt(29), self.v_pt( 5),  0.7071067690849304, self.v_pt(25))) # 29
        self._emit(v_madmk(self.v_pt(30), self.v_pt( 6),  0.7071067690849304, self.v_pt(26))) # 30
        self._emit(v_madmk(self.v_pt(31), self.v_pt( 7),  0.7071067690849304, self.v_pt(27))) # 31

        self._emit(v_madmk(self.v_pt(16), self.v_pt( 0),  0.7071067690849304, self.v_pt(16))) # 16
        self._emit(v_madmk(self.v_pt(17), self.v_pt( 1),  0.7071067690849304, self.v_pt(17))) # 17
        self._emit(v_madmk(self.v_pt(18), self.v_pt( 2),  0.7071067690849304, self.v_pt(18))) # 18
        self._emit(v_madmk(self.v_pt(19), self.v_pt( 3),  0.7071067690849304, self.v_pt(19))) # 19
        self._emit(v_madmk(self.v_pt(24), self.v_pt( 4), -0.7071067690849304, self.v_pt(24))) # 24
        self._emit(v_madmk(self.v_pt(25), self.v_pt( 5), -0.7071067690849304, self.v_pt(25))) # 25
        self._emit(v_madmk(self.v_pt(26), self.v_pt( 6), -0.7071067690849304, self.v_pt(26))) # 26
        self._emit(v_madmk(self.v_pt(27), self.v_pt( 7), -0.7071067690849304, self.v_pt(27))) # 27

        self._emit(f"v_sub_f32 v[{self.v_pt( 8)}], v[{self.v_tt( 8)}], v[{self.v_tt(13)}]") #  8
        self._emit(f"v_add_f32 v[{self.v_pt( 9)}], v[{self.v_tt( 9)}], v[{self.v_tt(12)}]") #  9
        self._emit(f"v_sub_f32 v[{self.v_pt(10)}], v[{self.v_tt(10)}], v[{self.v_tt(15)}]") #  10
        self._emit(f"v_add_f32 v[{self.v_pt(11)}], v[{self.v_tt(11)}], v[{self.v_tt(14)}]") #  11
        self._emit(f"v_add_f32 v[{self.v_pt(12)}], v[{self.v_tt( 8)}], v[{self.v_tt(13)}]") #  12
        self._emit(f"v_sub_f32 v[{self.v_pt(13)}], v[{self.v_tt( 9)}], v[{self.v_tt(12)}]") #  13
        self._emit(f"v_add_f32 v[{self.v_pt(14)}], v[{self.v_tt(10)}], v[{self.v_tt(15)}]") #  14
        self._emit(f"v_sub_f32 v[{self.v_pt(15)}], v[{self.v_tt(11)}], v[{self.v_tt(14)}]") #  15

        self._emit(f"v_add_f32 v[{self.v_pt( 0)}], v[{self.v_tt( 0)}], v[{self.v_tt( 4)}]") #  0
        self._emit(f"v_add_f32 v[{self.v_pt( 1)}], v[{self.v_tt( 1)}], v[{self.v_tt( 5)}]") #  1
        self._emit(f"v_sub_f32 v[{self.v_pt( 4)}], v[{self.v_tt( 0)}], v[{self.v_tt( 4)}]") #  4
        self._emit(f"v_sub_f32 v[{self.v_pt( 5)}], v[{self.v_tt( 1)}], v[{self.v_tt( 5)}]") #  5
        self._emit(f"v_add_f32 v[{self.v_tt(12)}], v[{self.v_tt( 2)}], v[{self.v_tt( 6)}]") #  2
        self._emit(f"v_add_f32 v[{self.v_tt(13)}], v[{self.v_tt( 3)}], v[{self.v_tt( 7)}]") #  3
        self._emit(f"v_sub_f32 v[{self.v_tt(14)}], v[{self.v_tt( 2)}], v[{self.v_tt( 6)}]") #  6
        self._emit(f"v_sub_f32 v[{self.v_tt(15)}], v[{self.v_tt( 3)}], v[{self.v_tt( 7)}]") #  7

        # omega 16_0,4,2,6,1,5,3,7
        self._emit(v_madmk(self.v_tt( 4), self.v_pt(19), -0.4142135679721832, self.v_pt(18))) #  8 --  9 tmp0
        self._emit(v_madmk(self.v_tt( 5), self.v_pt(18),  0.4142135679721832, self.v_pt(19))) #  8 --  9 tmp1
        self._emit(v_madmk(self.v_tt( 8), self.v_pt(27), -2.4142134189605713, self.v_pt(26))) # 12 -- 13 tmp0
        self._emit(v_madmk(self.v_tt( 9), self.v_pt(26),  2.4142134189605713, self.v_pt(27))) # 12 -- 13 tmp1
        self._emit(v_madmk(self.v_tt( 6), self.v_pt(23),  2.4142134189605713, self.v_pt(22))) # 10 -- 11 tmp0
        self._emit(v_madmk(self.v_tt( 7), self.v_pt(22), -2.4142134189605713, self.v_pt(23))) # 10 -- 11 tmp1
        self._emit(v_madmk(self.v_tt(10), self.v_pt(31),  0.4142135679721832, self.v_pt(30))) # 14 -- 15 tmp0
        self._emit(v_madmk(self.v_tt(11), self.v_pt(30), -0.4142135679721832, self.v_pt(31))) # 14 -- 15 tmp1
        self._emit(f"v_sub_f32 v[{self.v_tt( 0)}], v[{self.v_pt(10)}], v[{self.v_pt(11)}]") #  4 --  5 tmp0
        self._emit(f"v_add_f32 v[{self.v_tt( 1)}], v[{self.v_pt(11)}], v[{self.v_pt(10)}]") #  4 --  5 tmp1
        self._emit(f"v_add_f32 v[{self.v_tt( 2)}], v[{self.v_pt(14)}], v[{self.v_pt(15)}]") #  6 --  7 tmp0
        self._emit(f"v_sub_f32 v[{self.v_tt( 3)}], v[{self.v_pt(15)}], v[{self.v_pt(14)}]") #  6 --  7 tmp1

        self._emit(v_madmk(self.v_pt(18), self.v_tt( 4), -0.9238795042037964, self.v_pt(16))) # 18
        self._emit(v_madmk(self.v_pt(19), self.v_tt( 5), -0.9238795042037964, self.v_pt(17))) # 19
        self._emit(v_madmk(self.v_pt(22), self.v_tt( 6),  0.3826834261417389, self.v_pt(20))) # 22
        self._emit(v_madmk(self.v_pt(23), self.v_tt( 7),  0.3826834261417389, self.v_pt(21))) # 23
        self._emit(v_madmk(self.v_pt(26), self.v_tt( 8), -0.3826834261417389, self.v_pt(24))) # 26
        self._emit(v_madmk(self.v_pt(27), self.v_tt( 9), -0.3826834261417389, self.v_pt(25))) # 27
        self._emit(v_madmk(self.v_pt(30), self.v_tt(10),  0.9238795042037964, self.v_pt(28))) # 30
        self._emit(v_madmk(self.v_pt(31), self.v_tt(11),  0.9238795042037964, self.v_pt(29))) # 31
        self._emit(v_madmk(self.v_pt(16), self.v_tt( 4),  0.9238795042037964, self.v_pt(16))) # 16
        self._emit(v_madmk(self.v_pt(17), self.v_tt( 5),  0.9238795042037964, self.v_pt(17))) # 17
        self._emit(v_madmk(self.v_pt(20), self.v_tt( 6), -0.3826834261417389, self.v_pt(20))) # 20
        self._emit(v_madmk(self.v_pt(21), self.v_tt( 7), -0.3826834261417389, self.v_pt(21))) # 21
        self._emit(v_madmk(self.v_pt(24), self.v_tt( 8),  0.3826834261417389, self.v_pt(24))) # 24
        self._emit(v_madmk(self.v_pt(25), self.v_tt( 9),  0.3826834261417389, self.v_pt(25))) # 25
        self._emit(v_madmk(self.v_pt(28), self.v_tt(10), -0.9238795042037964, self.v_pt(28))) # 28
        self._emit(v_madmk(self.v_pt(29), self.v_tt(11), -0.9238795042037964, self.v_pt(29))) # 29

        self._emit(f"v_sub_f32 v[{self.v_pt( 2)}], v[{self.v_pt( 0)}], v[{self.v_tt(12)}]") #  2
        self._emit(f"v_sub_f32 v[{self.v_pt( 3)}], v[{self.v_pt( 1)}], v[{self.v_tt(13)}]") #  3

        self._emit(f"v_add_f32 v[{self.v_pt( 6)}], v[{self.v_pt( 4)}], v[{self.v_tt(15)}]") #  6
        self._emit(f"v_sub_f32 v[{self.v_pt( 7)}], v[{self.v_pt( 5)}], v[{self.v_tt(14)}]") #  7
        
        self._emit(v_madmk(self.v_pt(10), self.v_tt( 0), -0.7071067690849304, self.v_pt( 8))) # 10
        self._emit(v_madmk(self.v_pt(11), self.v_tt( 1), -0.7071067690849304, self.v_pt( 9))) # 11
        
        self._emit(v_madmk(self.v_pt(14), self.v_tt( 2),  0.7071067690849304, self.v_pt(12))) # 14
        self._emit(v_madmk(self.v_pt(15), self.v_tt( 3),  0.7071067690849304, self.v_pt(13))) # 15

        self._emit(f"v_add_f32 v[{self.v_pt( 0)}], v[{self.v_pt( 0)}], v[{self.v_tt(12)}]") #  0
        self._emit(f"v_add_f32 v[{self.v_pt( 1)}], v[{self.v_pt( 1)}], v[{self.v_tt(13)}]") #  1

        self._emit(f"v_sub_f32 v[{self.v_pt( 4)}], v[{self.v_pt( 4)}], v[{self.v_tt(15)}]") #  4
        self._emit(f"v_add_f32 v[{self.v_pt( 5)}], v[{self.v_pt( 5)}], v[{self.v_tt(14)}]") #  5

        self._emit(v_madmk(self.v_pt( 8), self.v_tt( 0),  0.7071067690849304, self.v_pt( 8))) #  8
        self._emit(v_madmk(self.v_pt( 9), self.v_tt( 1),  0.7071067690849304, self.v_pt( 9))) #  9

        self._emit(v_madmk(self.v_pt(12), self.v_tt( 2), -0.7071067690849304, self.v_pt(12))) # 12
        self._emit(v_madmk(self.v_pt(13), self.v_tt( 3), -0.7071067690849304, self.v_pt(13))) # 13
