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
from __future__ import print_function
import sys

from ..codegen import *

class ds_write2_likely_t(mc_base_t):
    '''
    generate ds_write2 if possible. otherwise fallback to ds_write.
    Design this not as macro, but inlined into other LDS store operation
    So need upper caller to make sure the uniqueness

    For wei load from global is {k, e}, and store to LDS is {e, k}, so need consider swap
    '''
    
    def name(self):
        return ''
    def __init__(self, mc, tunable, t_n_vec, t_vec_size, t_vec_stride, t_sst_base):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        self.t_n_vec        = t_n_vec
        self.t_vec_size     = t_vec_size
        self.t_vec_stride   = t_vec_stride
        self.t_sst_base     = t_sst_base
    def likely_write2_b32(self):
        if self.t_n_vec % 2 != 0:
            return False
        if (self.t_sst_base % 4 == 0) and (self.t_vec_stride % 4 == 0):
            if (self.t_sst_base // 4) + (self.t_vec_stride // 4) * (self.t_n_vec - 1) < 256:
                return True
        return False
    def likely_write2st64_b32(self):
        if self.t_n_vec % 2 != 0:
            return False
        if (self.t_sst_base % (4*64) == 0) and (self.t_vec_stride % 4 == 0):
            if (self.t_sst_base // (4*64)) + (self.t_vec_stride // (4*64)) * (self.t_n_vec - 1) < 256:
                return True
        return False
    def likely_write2_b64(self):
        if self.t_n_vec % 2 != 0:
            return False
        if (self.t_sst_base % 8 == 0) and (self.t_vec_stride % 8 == 0):
            if (self.t_sst_base // 8) + (self.t_vec_stride // 8) * (self.t_n_vec - 1) < 256:
                return True
        return False
    def likely_write2st64_b64(self):
        if self.t_n_vec % 2 != 0:
            return False
        if (self.t_sst_base % (8*64) == 0) and (self.t_vec_stride % (8*64) == 0):
            if (self.t_sst_base // (8*64)) + (self.t_vec_stride // (8*64)) * (self.t_n_vec - 1) < 256:
                return True
        return False
    def __call__(self, v_src, v_sst):
        v_src = sym_t(v_src)
        v_sst = sym_t(v_sst)
        def emit_write2_fallback():
            with self._deferred_context():
                if self.t_vec_size == 1:
                    for n in range(self.t_n_vec):
                        self._emit('ds_write_b32 v[{}], v[{}] offset:{}'.format(v_sst(), v_src(n), self.t_sst_base + n * self.t_vec_stride))
                elif self.t_vec_size == 2:
                    if self.t_n_vec == 1:
                        self._emit('ds_write_b64 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(), v_src(1), self.t_sst_base ))
                    else:
                        swap_start = (self.t_n_vec*self.t_vec_size) // 2
                        for n in range(self.t_n_vec // 2):
                            self._emit('v_swap_b32 v[{}], v[{}]'.format(v_src(2*n + 1), v_src(2*n + swap_start)))
                            self._emit('ds_write_b64 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(2*n), v_src(2*n + 1), self.t_sst_base + 2*n * self.t_vec_stride))
                            self._emit('ds_write_b64 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(2*n + swap_start) , v_src(2*n + swap_start + 1), self.t_sst_base + (2*n+1) * self.t_vec_stride))
                elif self.t_vec_size == 4:
                    if self.t_n_vec == 1:
                        self._emit('ds_write_b128 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(), v_src(3), self.t_sst_base ))
                    else:
                        # though we use algorithm in swap_seq to interleave swap with ds_write, but it is still wise to use extra tmp register for swap is half speed
                        swap_list = amdgpu_swap_sequencer_t(self.t_n_vec , self.t_vec_size)()
                        # print('self.t_n_vec:{}, self.t_vec_size:{}, {}'.format(self.t_n_vec , self.t_vec_size, swap_list))
                        for n in range(self.t_n_vec):
                            sw = swap_list[n]
                            if type(sw) is str:
                                pass
                            else:
                                for sw_item in sw:
                                    self._emit('v_swap_b32 v[{}], v[{}]'.format(v_src(sw_item[0]) , v_src(sw_item[1]) ))
                            self._emit('ds_write_b128 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(4*n), v_src(4*n + 3), self.t_sst_base + n * self.t_vec_stride))
                else:
                    assert False, 'unsupported vector size'
            return self._get_deferred()

        def emit_write2_b32():
            with self._deferred_context():
                for n in range(self.t_n_vec // 2):
                    self._emit('ds_write2_b32 v[{}], v[{}], v[{}], offset0:{}, offset1:{}'.format(v_sst(),
                                v_src(2*n), v_src(2*n+1),
                                (self.t_sst_base//4)+2*n*(self.t_vec_stride//4), (self.t_sst_base//4)+(2*n+1)*(self.t_vec_stride//4)))
            return self._get_deferred()

        def emit_write2st64_b32():
            with self._deferred_context():
                for n in range(self.t_n_vec // 2):
                    self._emit('ds_write2st64_b32 v[{}], v[{}], v[{}], offset0:{}, offset1:{}'.format(v_sst(),
                                v_src(2*n), v_src(2*n+1),
                                (self.t_sst_base//(4*64))+2*n*(self.t_vec_stride//(4*64)), (self.t_sst_base//(4*64))+(2*n+1)*(self.t_vec_stride//(4*64))))
            return self._get_deferred()

        def emit_write2_b64():
            swap_start = (self.t_n_vec*self.t_vec_size) // 2
            with self._deferred_context():
                for n in range(self.t_n_vec // 2):
                    self._emit('v_swap_b32 v[{}], v[{}]'.format(v_src(2*n+1), v_src(2*n+swap_start)))
                    self._emit('ds_write2_b64 v[{}], v[{}:{}], v[{}:{}], offset0:{}, offset1:{}'.format(v_sst(),
                            v_src(2*n), v_src(2*n+1), v_src(2*n+swap_start), v_src(2*n+swap_start+1),
                            (self.t_sst_base//8)+2*n*(self.t_vec_stride//8), (self.t_sst_base//8)+(2*n+1)*(self.t_vec_stride//8)))
            return self._get_deferred()

        def emit_write2st64_b64():
            swap_start = (self.t_n_vec*self.t_vec_size) // 2
            with self._deferred_context():
                for n in range(self.t_n_vec // 2):
                    self._emit('v_swap_b32 v[{}], v[{}]'.format(v_src(2*n+1), v_src(2*n+swap_start)))
                    self._emit('ds_write2st64_b64 v[{}], v[{}:{}], v[{}:{}], offset0:{}, offset1:{}'.format(v_sst(),
                            v_src(2*n), v_src(2*n+1), v_src(2*n+swap_start), v_src(2*n+swap_start+1),
                            (self.t_sst_base//(8*64))+2*n*(self.t_vec_stride//(8*64)), (self.t_sst_base//(8*64))+(2*n+1)*(self.t_vec_stride//(8*64))))
            return self._get_deferred()

        def likely_emit():
            if self.t_vec_size == 1:
                if self.likely_write2_b32():
                    return emit_write2_b32()
                if self.likely_write2st64_b32():
                    return emit_write2st64_b32()
                return emit_write2_fallback()
            if self.t_vec_size == 2:
                if self.likely_write2_b64():
                    return emit_write2_b64()
                if self.likely_write2st64_b64():
                    return emit_write2st64_b64()
                return emit_write2_fallback()
            return emit_write2_fallback()

        return likely_emit()
    def emit(self):
        assert False, 'dont use emit of this'
    def get_issues(self):
        if self.t_vec_size == 1:
            if self.likely_write2_b32() or self.likely_write2st64_b32():
                return self.t_n_vec // 2
        if self.t_vec_size == 2:
            if self.likely_write2_b64() or self.likely_write2st64_b64():
                return self.t_n_vec // 2
        return self.t_n_vec

class ds_read_t(object):
    def __init__(self, bytes):
        self.bytes = bytes
    def get_offset(self, offset):
        return '' if offset == 0 else 'offset:{}'.format(offset)
    def __call__(self, vdst, vaddr, offset):
        if self.bytes == 4:
            return 'ds_read_b32 v[{}], v[{}] {}'.format(vdst, vaddr, self.get_offset(offset))
        if self.bytes == 8:
            return 'ds_read_b64 v[{}:{}+1], v[{}] {}'.format(vdst, vdst, vaddr, self.get_offset(offset))
        if self.bytes == 12:
            return 'ds_read_b96 v[{}:{}+2], v[{}] {}'.format(vdst, vdst, vaddr, self.get_offset(offset))
        if self.bytes == 16:
            return 'ds_read_b128 v[{}:{}+3], v[{}] {}'.format(vdst, vdst, vaddr, self.get_offset(offset))
        assert False
    def get_issues(self):
        return 1

class ds_write_t(object):
    def __init__(self, bytes):
        self.bytes = bytes

    def get_offset(self, offset):
        if type(offset) is str:
            return 'offset:{}'.format(offset)
        if type(offset) is int:
            return '' if offset == 0 else 'offset:{}'.format(offset)
        assert False

    def __call__(self, vaddr, vdata, offset = 0):
        if self.bytes == 4:
            return 'ds_write_b32 v[{}], v[{}] {}'.format(vaddr, vdata, self.get_offset(offset))
        if self.bytes == 8:
            return 'ds_write_b64 v[{}], v[{}:{}+1] {}'.format(vaddr, vdata, vdata, self.get_offset(offset))
        if self.bytes == 12:
            return 'ds_write_b96 v[{}], v[{}:{}+2] {}'.format(vaddr, vdata, vdata, self.get_offset(offset))
        if self.bytes == 16:
            return 'ds_write_b128 v[{}], v[{}:{}+3] {}'.format(vaddr, vdata, vdata, self.get_offset(offset))
        assert False

    def get_issues(self):
        return 1

class ctrl_2d_shared_store_t(object):
    '''
    d0xd1
    '''
    def __init__(self):
        self.length_d0 = 1
        self.length_d1 = 1
        self.vector_d1 = 1
        # self.offset_d1 = 0      # base offset
        self.stride_d0 = 0      # stride
        self.precision = 'fp32'      # 'fp32', 'fp16', ...
        self.src_order = 0  # 0-d0,d1, 1-d1,d0

class macro_igemm_2d_shared_store_t(mc_base_t):
    def __init__(self, mc, ctrl):
        assert type(ctrl) is ctrl_2d_shared_store_t
        mc_base_t.__init__(self, mc)
        self.ctrl = ctrl
    def name(self):
        ctrl = self.ctrl
        if ctrl.precision == "fp32":
            bits_str = 'b32'
        elif ctrl.precision in ("fp16", "bf16"):
            bits_str = 'b16'
        else:
            assert False

        if ctrl.vector_d1 == 4:
            vec_str = 'v4'
        elif ctrl.vector_d1 == 2:
            vec_str = 'v2'
        elif ctrl.vector_d1 == 1:
            vec_str = 'v1'
        else:
            assert False

        assert ctrl.length_d1 == ctrl.vector_d1
        #n_d1 = ctrl.length_d1 // ctrl.vector_d1
        return f".v_sst_so{ctrl.src_order}_{ctrl.length_d0}x{ctrl.length_d1}_{bits_str}_{vec_str}_st{ctrl.stride_d0}"
    def __call__(self, v_src, v_sst_os):
        return '{} {}, {}'.format(self.name(), v_src, v_sst_os)
    def emit(self):
        ctrl = self.ctrl
        assert ctrl.length_d1 == ctrl.vector_d1
        assert ctrl.precision == 'fp32', "TO BE supported"
        ds_write = ds_write_t(ctrl.vector_d1 * 4)
        with self._emit_macro_indented('.macro {} v_src, v_sst_os'.format(self.name())):
            if ctrl.src_order == 0:
                for i_d0 in range(ctrl.length_d0):
                    self._emit(ds_write('\\v_sst_os', f'\\v_src+{i_d0*ctrl.vector_d1}', i_d0 * ctrl.stride_d0))
            else:
                assert "unimplemented"
    def get_issues(self):
        return self.ctrl.length_d0
