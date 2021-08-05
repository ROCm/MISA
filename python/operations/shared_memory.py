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
from __future__ import print_function
import sys

from ..codegen import *
from .utility import *

class amdgpu_swap_sequencer_t(object):
    '''
    partial-transpose 2d matrix in register, by using swap.
    currently only consider continus register in same col, aka col major

    after transpose, the num of col/row should be the same

    And be aware that, this method still is not straight-forward and not optimal,
    for v_swap_b32 have half speed. In this case better use several tmp register serve as vector buffer
    Hopefully in the future could have full speed v_swap_b32

        k0 k1 k2 k3          k0 k1 k2 k3
    e0 0  2  4  6    =>  e0 0  1  2  3
    e1 1  3  5  7        e1 4  5  6  7

        k0 k1 k2 k3         k0 k1 k2 k3
    e0  0  4  8  c       e0 0  1  2  3
    e1  1  5  9  d   =>  e1 4  5  6  7
    e2  2  6  a  e       e2 8  9  a  b
    e3  3  7  b  f       e3 c  d  e  f
    '''
    def create_2d_swap(self):
        def init_2d_indice(row, col):
            indice_2d = []
            for r in range(row):
                indice_2d.append([r+c*row for c in range(col)])
            return indice_2d
        def check_row_can_omit_swap(indice_2d, cur_row):
            '''
            if current row already fit in vector pattern, can omit out
            '''
            row = len(indice_2d)
            col = len(indice_2d[0])
            targeting_vector_pattern = []
            for c in range(col):
                targeting_vector_pattern.append(c)
            vector_diff = []
            for c in range(col):
                vector_diff.append(abs(indice_2d[cur_row][c] - targeting_vector_pattern[c]))
            lasf_diff = vector_diff[0]
            #print('xxx {}'.format(vector_diff))
            if lasf_diff % 2 != 0:
                return False
            for c in range(1, col):
                if lasf_diff != vector_diff[c]:
                    return False
            return True
        def scan_2d_indice(indice_2d):
            def locate_indice(indice_2d, target_indice, start_row):
                row = len(indice_2d)
                col = len(indice_2d[0])
                (tr, tc) = (start_row, 0)
                found = False
                for tr in range(start_row, row):
                    for tc in range(0, col):
                        #print(target_indice, indice_2d[tr][tc])
                        if target_indice == indice_2d[tr][tc]:
                            found = True
                            break
                    if found:
                        break
                assert found
                return (tr, tc)
            swap_list = []
            row = len(indice_2d)
            col = len(indice_2d[0])

            class touch_row_t(object):
                def __init__(self, row):
                    self.row = row
                    self.row_touched = [ 0 for r in range(row)]
                    self.row_touched_index = 0
                def next_untouched_row(self):
                    for r in range(self.row_touched_index, self.row):
                        if self.row_touched[r] == 0:
                            self.row_touched_index = r
                            return r
                    assert False
                def touch(self, row_index):
                    self.row_touched[row_index] = 1
            touch_row = touch_row_t(row)
            for r in range(row):
                if check_row_can_omit_swap(indice_2d, r):
                    swap_list.append('unified for row {}'.format(r))
                    touch_row.touch( indice_2d[r][0] // col)
                    continue
                swap_list_per_row = []
                for c in range(col):
                    target_indice = touch_row.next_untouched_row()*col + c
                    origin_indice = indice_2d[r][c]
                    if origin_indice == target_indice:
                        continue
                    #print('to find:{}'.format(target_indice))
                    (tr, tc) = locate_indice(indice_2d, target_indice, r)
                    # swap and record indice
                    indice_2d[tr][tc] = origin_indice
                    indice_2d[r][c] = target_indice
                    #print('swapper:{}'.format(indice_2d))
                    swap_list_per_row.append((origin_indice, target_indice))
                swap_list.append(swap_list_per_row)
                touch_row.touch(r)
            return swap_list
        indice_2d = init_2d_indice(self.row, self.col)
        #print(indice_2d)
        swap_list = scan_2d_indice(indice_2d)
        return swap_list

    def __init__(self, row, col):
        assert col != 1 and row != 1
        self.col = col
        self.row = row
        self.swap_list = self.create_2d_swap()

    def __call__(self):
        '''
        return list of tuple of the row row_idx what swap should take
        '''
        return self.swap_list


class simple_swap_sequencer_t(object):
    '''
    use swap only when col is 2. other case is much complicated, better use tmp register

             0    1     0   1
        0    0    2   | 0   1
        1    1    3   | 2   3
    ------------------+--------
             0    1   | 
        0    0    4   | 0   1
        1    1    5   | 4   5 
        2    2    6   | 2   3 
        3    3    7   | 6   7 
    ------------------+--------
             0    1   |
        0    0    6   | 0   1
        1    1    7   | 6   7
        2    2    8   | 2   3
        3    3    9   | 8   9
        4    4    10  | 4   5
        5    5    11  | 10  11
    ------------------+--------
             0    1   |
        0    0    8   | 0   1
        1    1    9   | 8   9
        2    2    10  | 2   3
        3    3    11  | 10  11
        4    4    12  | 4   5
        5    5    13  | 12  13
        6    6    14  | 6   7
        7    7    15  | 14  15

    '''
    def __init__(self, row, col):
        assert col == 2 and row in (2, 4, 6, 8), f"col:{col}, row:{row}"
        self.col = col
        self.row = row

    def get_swap_per_row(self):
        if self.row == 2:
            return [(1,2), None]
        if self.row == 4:
            return [(1, 4), None, (3, 6), None]
        if self.row == 6:
            return [(1, 6), None, (3, 8), None, (5, 10), None]
        if self.row == 8:
            return [(1, 8), None, (3, 10), None, (5, 12), None, (7, 14), None]
        assert False

    def get_start_id_per_row(self):
        if self.row == 2:
            return [0, 2]
        if self.row == 4:
            return [0, 4, 2, 6]
        if self.row == 6:
            return [0, 6, 2, 8, 4, 10]
        if self.row == 8:
            return [0, 8, 2, 10, 4, 12, 6, 14]
        assert False

class simple_transpose_sequencer_t(object):
    def __init__(self, row, col, col_major = True):
        self.col = col
        self.row = row
        self.col_major = col_major

    def get_start_id_per_row(self):
        s_id_list = list()
        for r in range(self.row):
            row_id_list = list()
            for c in range(self.col):
                row_id_list.append(r+c*self.row)
            s_id_list.append(row_id_list)
        return s_id_list

class inst_ds_read2_likely_t(mc_base_t):
    '''
    generate ds_read2 if possible. otherwise fallback to ds_read
    Design this not as macro, but inlined into other LDS store operation
    So need upper caller to make sure the uniqueness
    '''
    def name(self):
        return ''

    def __init__(self, mc, vec_count, vec_byte, vec_stride, sld_base = 0):
        mc_base_t.__init__(self, mc)
        self.vec_count = vec_count
        self.vec_byte = vec_byte
        #assert vec_byte in (4, 8)
        self.vec_stride = vec_stride
        self.sld_base = sld_base
    
    def likely_read2_b32(self, sld_offset = 0):
        if self.vec_byte != 4:
            return False
        if self.vec_count % 2 != 0:
            return False
        if ((self.sld_base + sld_offset) % 4 == 0) and (self.vec_stride % 4 == 0):
            if ((self.sld_base + sld_offset) // 4) + (self.vec_stride // 4) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_read2st64_b32(self, sld_offset = 0):
        if self.vec_byte != 4:
            return False
        if self.vec_count % 2 != 0:
            return False
        if ((self.sld_base + sld_offset) % (4*64) == 0) and (self.vec_stride % (4*64) == 0):
            if ((self.sld_base + sld_offset) // (4*64)) + (self.vec_stride // (4*64)) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_read2_b64(self, sld_offset = 0):
        if self.vec_byte != 8:
            return False
        if self.vec_count % 2 != 0:
            return False
        if ((self.sld_base + sld_offset) % 8 == 0) and (self.vec_stride % 8 == 0):
            if ((self.sld_base + sld_offset) // 8) + (self.vec_stride // 8) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_read2st64_b64(self, sld_offset = 0):
        if self.vec_byte != 8:
            return False
        if self.vec_count % 2 != 0:
            return False
        if ((self.sld_base + sld_offset) % (8*64) == 0) and (self.vec_stride % (8*64) == 0):
            if ((self.sld_base + sld_offset) // (8*64)) + (self.vec_stride // (8*64)) * (self.vec_count - 1) < 256:
                return True
        return False
    def __call__(self, v_dst, v_sld_os, sld_offset = 0):
        v_dst = sym_t(v_dst)
        v_sld_os = sym_t(v_sld_os)
        def emit_read2_fallback(sld_offset = 0):
            sldx1 = inst_ds_read_t(self.vec_byte)
            with self._deferred_context():
                for n in range(self.vec_count):
                    self._emit(sldx1(v_dst(n*(self.vec_byte // 4)), v_sld_os(), (self.sld_base + sld_offset) + n * self.vec_stride))
            return self._get_deferred()

        def emit_read2_b32(sld_offset = 0):
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_read2_b32 v[{v_dst((2*n, 2*n+1))}], v[{v_sld_os()}], offset0:{((self.sld_base + sld_offset)//4)+2*n*(self.vec_stride//4)}, offset1:{((self.sld_base + sld_offset)//4)+(2*n+1)*(self.vec_stride//4)}')
            return self._get_deferred()

        def emit_read2st64_b32(sld_offset = 0):
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_read2st64_b32 v[{v_dst((2*n,2*n+1))}], v[{v_sld_os()}], offset0:{((self.sld_base + sld_offset)//(4*64))+2*n*(self.vec_stride//(4*64))}, offset1:{((self.sld_base + sld_offset)//(4*64))+(2*n+1)*(self.vec_stride//(4*64))}')
            return self._get_deferred()

        def emit_read2_b64(sld_offset = 0):
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_read2_b64 v[{v_dst((4*n, 4*n+3))}], v[{v_sld_os()}], offset0:{((self.sld_base + sld_offset)//8)+2*n*(self.vec_stride//8)}, offset1:{((self.sld_base + sld_offset)//8)+(2*n+1)*(self.vec_stride//8)}')
            return self._get_deferred()

        def emit_read2st64_b64(sld_offset = 0):
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_read2st64_b64 v[{v_dst((4*n,4*n+3))}], v[{v_sld_os()}], offset0:{((self.sld_base + sld_offset)//(8*64))+2*n*(self.vec_stride//(8*64))}, offset1:{((self.sld_base + sld_offset)//(8*64))+(2*n+1)*(self.vec_stride//(8*64))}')
            return self._get_deferred()

        def likely_emit(sld_offset = 0):
            if self.vec_byte == 4:
                if self.likely_read2_b32(sld_offset):
                    return emit_read2_b32(sld_offset)
                if self.likely_read2st64_b32(sld_offset):
                    return emit_read2st64_b32(sld_offset)
                return emit_read2_fallback(sld_offset)
            if self.vec_byte == 8:
                if self.likely_read2_b64(sld_offset):
                    return emit_read2_b64(sld_offset)
                if self.likely_read2st64_b64(sld_offset):
                    return emit_read2st64_b64(sld_offset)
                return emit_read2_fallback(sld_offset)
            return emit_read2_fallback(sld_offset)

        return likely_emit(sld_offset)
    #def emit(self):
    #    assert False, 'dont use emit of this'
    def get_issues(self, sld_offset = 0):
        if self.vec_byte == 4:
            if self.likely_read2_b32(sld_offset) or self.likely_read2st64_b32(sld_offset):
                return self.vec_count // 2
        if self.vec_byte == 8:
            if self.likely_read2_b64(sld_offset) or self.likely_read2st64_b64(sld_offset):
                return self.vec_count // 2
        return self.vec_count


class inst_ds_read2_likely_accumulate_offset_t(mc_base_t):
    '''
    used in fma main loop, that if ds_read2 can't be generated (aka have fall back), accumulate offset, then use ds_read2 again.
    '''
    def name(self):
        return ''

    def __init__(self, mc, vec_count, vec_byte, vec_stride, v_tmp_sld = sym_t('v_tmp'), sld_base = 0):
        mc_base_t.__init__(self, mc)
        self.ds_read2_likely = inst_ds_read2_likely_t(mc, vec_count, vec_byte, vec_stride, sld_base)
        self.init_sld_offset = 0
        self.first_call = 1

        self.v_tmp_sld = v_tmp_sld     # by default use this as accumulator! this should be convention
        #self.acc_into_v_tmp = False
        self.last_sld_offset = 0

    def any_read2_likely(self, sld_offset = 0):
        return self.ds_read2_likely.likely_read2_b32(sld_offset) or \
                self.ds_read2_likely.likely_read2st64_b32(sld_offset) or \
                self.ds_read2_likely.likely_read2_b64(sld_offset) or \
                self.ds_read2_likely.likely_read2st64_b64(sld_offset)

    def __call__(self, v_dst, v_sld_os, sld_offset = 0):
        if self.first_call:
            self.init_sld_offset = sld_offset       # record the first offset, as a marker for loop over from start
            self.last_sld_offset = sld_offset
            self.first_call = 0
            assert self.any_read2_likely(sld_offset), "currently we must make sure the first call is using read2"
            return self.ds_read2_likely(v_dst, v_sld_os, sld_offset)
        else:
            if self.init_sld_offset == sld_offset:
                self.last_sld_offset = self.init_sld_offset
                # this means we loop over from start
                return self.ds_read2_likely(v_dst, v_sld_os, sld_offset)
            else:
                if self.last_sld_offset == self.init_sld_offset:
                    diff_sld_offset = sld_offset - self.last_sld_offset
                    if self.any_read2_likely(sld_offset):
                        return self.ds_read2_likely(v_dst, v_sld_os, sld_offset)
                    else:
                        with self._deferred_context():
                            self._emit(f"v_add_u32 v[{self.v_tmp_sld()}], {diff_sld_offset}, v[{v_sld_os}]")
                            self._emit(self.ds_read2_likely(v_dst, self.v_tmp_sld(), self.init_sld_offset))
                        self.last_sld_offset = sld_offset
                        return self._get_deferred()
                else:
                    diff_sld_offset = sld_offset - self.last_sld_offset
                    if self.any_read2_likely(diff_sld_offset):
                        return self.ds_read2_likely(v_dst, self.v_tmp_sld(), diff_sld_offset)
                    else:
                        # diff_sld_offset = sld_offset - self.last_sld_offset
                        with self._deferred_context():
                            self._emit(f"v_add_u32 v[{self.v_tmp_sld()}], {diff_sld_offset}, v[{self.v_tmp_sld()}]")
                            self._emit(self.ds_read2_likely(v_dst, self.v_tmp_sld(), self.init_sld_offset))
                        self.last_sld_offset = sld_offset
                        return self._get_deferred()

    def get_issue(self, sld_offset = 0):
        # TODO: might have bug
        return self.ds_read2_likely.get_issues()


'''
class inst_ds_write2_likely_t(mc_base_t):   
    def name(self):
        return ''
    def __init__(self, mc, tunable, vec_count, vec_byte, vec_stride, sst_base):
        igemm_v4r1_dynamic_t.__init__(self, mc, tunable)
        self.vec_count        = vec_count
        self.vec_byte     = vec_byte
        self.vec_stride   = vec_stride
        (self.sst_base + sst_offset)     = sst_base
    def likely_write2_b32(self):
        if self.vec_count % 2 != 0:
            return False
        if ((self.sst_base + sst_offset) % 4 == 0) and (self.vec_stride % 4 == 0):
            if ((self.sst_base + sst_offset) // 4) + (self.vec_stride // 4) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_write2st64_b32(self):
        if self.vec_count % 2 != 0:
            return False
        if ((self.sst_base + sst_offset) % (4*64) == 0) and (self.vec_stride % 4 == 0):
            if ((self.sst_base + sst_offset) // (4*64)) + (self.vec_stride // (4*64)) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_write2_b64(self):
        if self.vec_count % 2 != 0:
            return False
        if ((self.sst_base + sst_offset) % 8 == 0) and (self.vec_stride % 8 == 0):
            if ((self.sst_base + sst_offset) // 8) + (self.vec_stride // 8) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_write2st64_b64(self):
        if self.vec_count % 2 != 0:
            return False
        if ((self.sst_base + sst_offset) % (8*64) == 0) and (self.vec_stride % (8*64) == 0):
            if ((self.sst_base + sst_offset) // (8*64)) + (self.vec_stride // (8*64)) * (self.vec_count - 1) < 256:
                return True
        return False
    def __call__(self, v_src, v_sst):
        v_src = sym_t(v_src)
        v_sst = sym_t(v_sst)
        def emit_write2_fallback():
            with self._deferred_context():
                if self.vec_byte == 1:
                    for n in range(self.vec_count):
                        self._emit('ds_write_b32 v[{}], v[{}] offset:{}'.format(v_sst(), v_src(n), (self.sst_base + sst_offset) + n * self.vec_stride))
                elif self.vec_byte == 2:
                    if self.vec_count == 1:
                        self._emit('ds_write_b64 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(), v_src(1), (self.sst_base + sst_offset) ))
                    else:
                        swap_start = (self.vec_count*self.vec_byte) // 2
                        for n in range(self.vec_count // 2):
                            self._emit('v_swap_b32 v[{}], v[{}]'.format(v_src(2*n + 1), v_src(2*n + swap_start)))
                            self._emit('ds_write_b64 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(2*n), v_src(2*n + 1), (self.sst_base + sst_offset) + 2*n * self.vec_stride))
                            self._emit('ds_write_b64 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(2*n + swap_start) , v_src(2*n + swap_start + 1), (self.sst_base + sst_offset) + (2*n+1) * self.vec_stride))
                elif self.vec_byte == 4:
                    if self.vec_count == 1:
                        self._emit('ds_write_b128 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(), v_src(3), (self.sst_base + sst_offset) ))
                    else:
                        # though we use algorithm in swap_seq to interleave swap with ds_write, but it is still wise to use extra tmp register for swap is half speed
                        swap_list = amdgpu_swap_sequencer_t(self.vec_count , self.vec_byte)()
                        # print('self.vec_count:{}, self.vec_byte:{}, {}'.format(self.vec_count , self.vec_byte, swap_list))
                        for n in range(self.vec_count):
                            sw = swap_list[n]
                            if type(sw) is str:
                                pass
                            else:
                                for sw_item in sw:
                                    self._emit('v_swap_b32 v[{}], v[{}]'.format(v_src(sw_item[0]) , v_src(sw_item[1]) ))
                            self._emit('ds_write_b128 v[{}], v[{}:{}] offset:{}'.format(v_sst(), v_src(4*n), v_src(4*n + 3), (self.sst_base + sst_offset) + n * self.vec_stride))
                else:
                    assert False, 'unsupported vector size'
            return self._get_deferred()

        def emit_write2_b32():
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit('ds_write2_b32 v[{}], v[{}], v[{}], offset0:{}, offset1:{}'.format(v_sst(),
                                v_src(2*n), v_src(2*n+1),
                                ((self.sst_base + sst_offset)//4)+2*n*(self.vec_stride//4), ((self.sst_base + sst_offset)//4)+(2*n+1)*(self.vec_stride//4)))
            return self._get_deferred()

        def emit_write2st64_b32():
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit('ds_write2st64_b32 v[{}], v[{}], v[{}], offset0:{}, offset1:{}'.format(v_sst(),
                                v_src(2*n), v_src(2*n+1),
                                ((self.sst_base + sst_offset)//(4*64))+2*n*(self.vec_stride//(4*64)), ((self.sst_base + sst_offset)//(4*64))+(2*n+1)*(self.vec_stride//(4*64))))
            return self._get_deferred()

        def emit_write2_b64():
            swap_start = (self.vec_count*self.vec_byte) // 2
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit('v_swap_b32 v[{}], v[{}]'.format(v_src(2*n+1), v_src(2*n+swap_start)))
                    self._emit('ds_write2_b64 v[{}], v[{}:{}], v[{}:{}], offset0:{}, offset1:{}'.format(v_sst(),
                            v_src(2*n), v_src(2*n+1), v_src(2*n+swap_start), v_src(2*n+swap_start+1),
                            ((self.sst_base + sst_offset)//8)+2*n*(self.vec_stride//8), ((self.sst_base + sst_offset)//8)+(2*n+1)*(self.vec_stride//8)))
            return self._get_deferred()

        def emit_write2st64_b64():
            swap_start = (self.vec_count*self.vec_byte) // 2
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit('v_swap_b32 v[{}], v[{}]'.format(v_src(2*n+1), v_src(2*n+swap_start)))
                    self._emit('ds_write2st64_b64 v[{}], v[{}:{}], v[{}:{}], offset0:{}, offset1:{}'.format(v_sst(),
                            v_src(2*n), v_src(2*n+1), v_src(2*n+swap_start), v_src(2*n+swap_start+1),
                            ((self.sst_base + sst_offset)//(8*64))+2*n*(self.vec_stride//(8*64)), ((self.sst_base + sst_offset)//(8*64))+(2*n+1)*(self.vec_stride//(8*64))))
            return self._get_deferred()

        def likely_emit():
            if self.vec_byte == 1:
                if self.likely_write2_b32():
                    return emit_write2_b32()
                if self.likely_write2st64_b32():
                    return emit_write2st64_b32()
                return emit_write2_fallback()
            if self.vec_byte == 2:
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
        if self.vec_byte == 1:
            if self.likely_write2_b32() or self.likely_write2st64_b32():
                return self.vec_count // 2
        if self.vec_byte == 2:
            if self.likely_write2_b64() or self.likely_write2st64_b64():
                return self.vec_count // 2
        return self.vec_count
'''

class inst_ds_write2_oneshot_t(mc_base_t):
    '''
    single issue a ds_write2. if can't use ds_write2, simply change to 2 ds_write
    '''
    def name(self):
        return ''
    def __init__(self, mc, vec_byte):
        mc_base_t.__init__(self, mc)
        assert vec_byte in (4, 8)
        self.vec_byte = vec_byte

    def _valid_range_div(self, num, div, upper):
        return True if num % div == 0 and num // div < upper else False

    def can_write2(self, offset0 , offset1):
        vrd = self._valid_range_div
        if self.vec_byte == 4:
            return True if vrd(offset0, 4, 256) and vrd(offset1, 4, 256) else False
        elif self.vec_byte == 8:
            return True if vrd(offset0, 8, 256) and vrd(offset1, 8, 256) else False
        else:
            assert False

    def can_write2st64(self, offset0 , offset1):
        vrd = self._valid_range_div
        if self.vec_byte == 4:
            return True if vrd(offset0, 4 * 64, 256) and vrd(offset1, 4 * 64, 256) else False
        elif self.vec_byte == 8:
            return True if vrd(offset0, 8 * 64, 256) and vrd(offset1, 8 * 64, 256) else False
        else:
            assert False

    def __call__(self, v_addr, v_src0, v_src1, offset0, offset1):
        with self._deferred_context():
            if self.vec_byte == 4:
                if self.can_write2(offset0, offset1):
                    if self.can_write2st64(offset0, offset1):
                        self._emit(f'ds_write2st64_b32 v[{v_addr}], v[{v_src0}], v[{v_src1}], offset0:{offset0 // (4*64)}, offset1:{offset1 // (4*64)}')
                    else:
                        self._emit(f'ds_write2_b32 v[{v_addr}], v[{v_src0}], v[{v_src1}], offset0:{offset0 // 4}, offset1:{offset1 // 4}')
                else:
                    self._emit(f'ds_write_b32 v[{v_addr}], v[{v_src0}], offset:{offset0}')
                    self._emit(f'ds_write_b32 v[{v_addr}], v[{v_src1}], offset:{offset1}')
            elif self.vec_byte == 8:
                if self.can_write2(offset0, offset1):
                    if self.can_write2st64(offset0, offset1):
                        self._emit(f'ds_write2st64_b64 v[{v_addr}], v[{v_src0}:{v_src0}+1], v[{v_src1}:{v_src1}+1], offset0:{offset0 // (8*64)}, offset1:{offset1 // (8*64)}')
                    else:
                        self._emit(f'ds_write2_b64 v[{v_addr}], v[{v_src0}:{v_src0}+1], v[{v_src1}:{v_src1}+1], offset0:{offset0 // 8}, offset1:{offset1 // 8}')
                else:
                    self._emit(f'ds_write_b64 v[{v_addr}], v[{v_src0}:{v_src0}+1] offset:{offset0}')
                    self._emit(f'ds_write_b64 v[{v_addr}], v[{v_src1}:{v_src1}+1] offset:{offset1}')
            else:
                assert False
        return self._get_deferred()

    def get_issues(self, offset0, offset1):
        if self.can_write2(offset0, offset1):
            return 1
        return 2

class inst_ds_write2_likely_t(mc_base_t):   
    def name(self):
        return ''
    def __init__(self, mc, vec_count, vec_byte, vec_stride, sst_base=0):
        mc_base_t.__init__(self, mc)
        self.vec_count    = vec_count
        self.vec_byte     = vec_byte
        self.vec_stride   = vec_stride
        self.sst_base     = sst_base
    def likely_write2_b32(self, sst_offset = 0):
        if self.vec_byte != 4:
            return False
        if self.vec_count % 2 != 0:
            return False
        if ((self.sst_base + sst_offset) % 4 == 0) and (self.vec_stride % 4 == 0):
            if ((self.sst_base + sst_offset) // 4) + (self.vec_stride // 4) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_write2st64_b32(self, sst_offset = 0):
        if self.vec_byte != 4:
            return False
        if self.vec_count % 2 != 0:
            return False
        if ((self.sst_base + sst_offset) % (4*64) == 0) and (self.vec_stride % (4*64) == 0):
            if ((self.sst_base + sst_offset) // (4*64)) + (self.vec_stride // (4*64)) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_write2_b64(self, sst_offset = 0):
        if self.vec_byte != 8:
            return False
        if self.vec_count % 2 != 0:
            return False
        if ((self.sst_base + sst_offset) % 8 == 0) and (self.vec_stride % 8 == 0):
            if ((self.sst_base + sst_offset) // 8) + (self.vec_stride // 8) * (self.vec_count - 1) < 256:
                return True
        return False
    def likely_write2st64_b64(self, sst_offset = 0):
        if self.vec_byte != 8:
            return False
        if self.vec_count % 2 != 0:
            return False
        if ((self.sst_base + sst_offset) % (8*64) == 0) and (self.vec_stride % (8*64) == 0):
            if ((self.sst_base + sst_offset) // (8*64)) + (self.vec_stride // (8*64)) * (self.vec_count - 1) < 256:
                return True
        return False
    def __call__(self, v_sst, v_src, sst_offset = 0):
        if type(v_src) in (tuple , list):
            assert len(v_src) == self.vec_count
        #else:
        v_src = sym_t(v_src)
        v_sst = sym_t(v_sst)
        def emit_write2_fallback(sst_offset = 0):
            sstx1 = inst_ds_write_t(self.vec_byte)
            with self._deferred_context():
                for n in range(self.vec_count):
                    self._emit(sstx1(v_sst(), v_src(n*(self.vec_byte // 4)), (self.sst_base + sst_offset) + n * self.vec_stride))
            return self._get_deferred()

        def emit_write2_b32(sst_offset = 0):
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_write2_b32 v[{v_sst()}], v[{v_src(2*n)}], v[{v_src(2*n+1)}], offset0:{((self.sst_base + sst_offset)//4)+2*n*(self.vec_stride//4)}, offset1:{((self.sst_base + sst_offset)//4)+(2*n+1)*(self.vec_stride//4)}')
            return self._get_deferred()

        def emit_write2st64_b32(sst_offset = 0):
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_write2st64_b32 v[{v_sst()}], v[{v_src(2*n)}], v[{v_src(2*n+1)}], offset0:{((self.sst_base + sst_offset)//(4*64))+2*n*(self.vec_stride//(4*64))}, offset1:{((self.sst_base + sst_offset)//(4*64))+(2*n+1)*(self.vec_stride//(4*64))}')
            return self._get_deferred()

        def emit_write2_b64(sst_offset = 0):
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_write2_b64 v[{v_sst()}], v[{v_src((4*n, 4*n+1))}], v[{v_src((4*n+2, 4*n+3))}], offset0:{((self.sst_base + sst_offset)//8)+2*n*(self.vec_stride//8)}, offset1:{((self.sst_base + sst_offset)//8)+(2*n+1)*(self.vec_stride//8)}')
            return self._get_deferred()

        def emit_write2st64_b64(sst_offset = 0):
            with self._deferred_context():
                for n in range(self.vec_count // 2):
                    self._emit(f'ds_write2st64_b64 v[{v_sst()}], v[{v_src((4*n, 4*n+1))}], v[{v_src((4*n+2, 4*n+3))}], offset0:{((self.sst_base + sst_offset)//(8*64))+2*n*(self.vec_stride//(8*64))}, offset1:{((self.sst_base + sst_offset)//(8*64))+(2*n+1)*(self.vec_stride//(8*64))}')
            return self._get_deferred()

        def likely_emit(sst_offset = 0):
            if self.vec_byte == 4:
                if self.likely_write2_b32(sst_offset):
                    return emit_write2_b32(sst_offset)
                if self.likely_write2st64_b32(sst_offset):
                    return emit_write2st64_b32(sst_offset)
                return emit_write2_fallback(sst_offset)
            if self.vec_byte == 8:
                if self.likely_write2_b64(sst_offset):
                    return emit_write2_b64(sst_offset)
                if self.likely_write2st64_b64(sst_offset):
                    return emit_write2st64_b64(sst_offset)
                return emit_write2_fallback(sst_offset)
            return emit_write2_fallback(sst_offset)
        return likely_emit(sst_offset)

    #def emit(self):
    #    assert False, 'dont use emit of this'
    def get_issues(self, sst_offset = 0):
        if self.vec_byte == 4:
            if self.likely_write2_b32(sst_offset) or self.likely_write2st64_b32(sst_offset):
                return self.vec_count // 2
        if self.vec_byte == 8:
            if self.likely_write2_b64(sst_offset) or self.likely_write2st64_b64(sst_offset):
                return self.vec_count // 2
        return self.vec_count


class inst_ds_write2_likely_accumulate_offset_t(mc_base_t):   
    def name(self):
        return ''



class inst_ds_read_t(object):
    def __init__(self, bytes):
        self.bytes = bytes
    def get_offset(self, offset):
        return '' if offset == 0 else 'offset:{}'.format(offset)
    def __call__(self, vdst, vaddr, offset):
        if self.bytes == 1:
            return 'ds_read_u8 v[{}], v[{}] {}'.format(vdst, vaddr, self.get_offset(offset))
        if self.bytes == 2:
            return 'ds_read_u16 v[{}], v[{}] {}'.format(vdst, vaddr, self.get_offset(offset))
        if self.bytes == 4:
            return 'ds_read_b32 v[{}], v[{}] {}'.format(vdst, vaddr, self.get_offset(offset))
        if self.bytes == 8:
            return 'ds_read_b64 v[{}:{}+1], v[{}] {}'.format(vdst, vdst, vaddr, self.get_offset(offset))
        if self.bytes == 12:
            return 'ds_read_b96 v[{}:{}+2], v[{}] {}'.format(vdst, vdst, vaddr, self.get_offset(offset))
        if self.bytes == 16:
            return 'ds_read_b128 v[{}:{}+3], v[{}] {}'.format(vdst, vdst, vaddr, self.get_offset(offset))
        assert False, f'bytes:{self.bytes}'
    def get_issues(self, sld_offset = 0):
        return 1

class inst_ds_write_t(object):
    def __init__(self, bytes):
        self.bytes = bytes

    def get_offset(self, offset):
        if type(offset) is str:
            return 'offset:{}'.format(offset)
        if type(offset) is int:
            return '' if offset == 0 else 'offset:{}'.format(offset)
        assert False

    def __call__(self, vaddr, vdata, offset = 0, lo_hi = 0):
        if self.bytes == 1:
            return 'ds_write_b8 v[{}], v[{}] {}'.format(vaddr, vdata, self.get_offset(offset))
        if self.bytes == 2:
            if lo_hi == 0:
                return 'ds_write_b16 v[{}], v[{}] {}'.format(vaddr, vdata, self.get_offset(offset))
            else:
                return 'ds_write_b16_d16_hi v[{}], v[{}] {}'.format(vaddr, vdata, self.get_offset(offset))
        if self.bytes == 4:
            return 'ds_write_b32 v[{}], v[{}] {}'.format(vaddr, vdata, self.get_offset(offset))
        if self.bytes == 8:
            return 'ds_write_b64 v[{}], v[{}:{}+1] {}'.format(vaddr, vdata, vdata, self.get_offset(offset))
        if self.bytes == 12:
            return 'ds_write_b96 v[{}], v[{}:{}+2] {}'.format(vaddr, vdata, vdata, self.get_offset(offset))
        if self.bytes == 16:
            return 'ds_write_b128 v[{}], v[{}:{}+3] {}'.format(vaddr, vdata, vdata, self.get_offset(offset))
        assert False

    def get_issues(self, sst_offset = 0):
        return 1

class ctrl_2d_shared_store_t(object):
    '''
    d0xd1
    '''
    def __init__(self):
        self.length_d0 = 1        # is d0 is 1, it is indeed 1d access
        self.length_d1 = 1
        self.vector_d1 = 1
        # self.offset_d1 = 0      # base offset
        self.stride_d0 = 1        # stride
        self.stride_d1 = 1         # if have stride_d1, then each d1 may have stride
        self.precision = 'fp32'      # 'fp32', 'fp16', ...
        self.src_order = 0  # 0-d0,d1, 1-d1,d0
        self.need_transpose = 1
        self.v_tmp = None   # used when order is 1 and consider shuffle

    def serialize(self):
        return f"length_d0:{self.length_d0}, length_d1:{self.length_d1}, vector_d1:{self.vector_d1}, stride_d0:{self.stride_d0}, stride_d1:{self.stride_d1}, precision:{self.precision}, src_order:{self.src_order}"

class macro_igemm_2d_shared_store_t(macro_base_t):
    def __init__(self, mc, ctrl, inline = False):
        assert type(ctrl) is ctrl_2d_shared_store_t
        macro_base_t.__init__(self, mc, inline)
        self.ctrl = ctrl
        self.issue_cnt = 0
        self.declare_arg("v_src")
        self.declare_arg("v_sst_os")
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

        # assert ctrl.length_d1 == ctrl.vector_d1

        return f".v_sst_so{ctrl.src_order}_{ctrl.length_d0}x{ctrl.length_d1}_{bits_str}_{vec_str}" + \
                f"_st{ctrl.stride_d0}x{ctrl.stride_d1}"

    def expr(self):
        ctrl = self.ctrl
        # assert ctrl.length_d1 == ctrl.vector_d1
        assert ctrl.precision == 'fp32', "TO BE supported"
        data_byte = amdgpu_precision_data_byte(ctrl.precision)
        issue_cnt = 0
        #with self._emit_macro_indented('.macro {} v_src, v_sst_os'.format(self.name())):
        if ctrl.length_d1 == ctrl.vector_d1:
            if ctrl.src_order == 0 or (ctrl.src_order == 1 and ctrl.vector_d1 == 1):
                if ctrl.length_d0 % 2 == 0 and data_byte == 4 and ctrl.vector_d1 in (1, 2):
                    ds_write2 = inst_ds_write2_likely_t(self.mc, 2, ctrl.vector_d1 * data_byte, ctrl.stride_d0)
                    for i_d0 in range(ctrl.length_d0 // 2):
                        self._emit(ds_write2(f'{self.v_sst_os()}', f'{self.v_src()}+{2 * i_d0*ctrl.vector_d1}', 2 * i_d0 * ctrl.stride_d0))
                        issue_cnt += ds_write2.get_issues(2 * i_d0 * ctrl.stride_d0)
                else:
                    ds_write = inst_ds_write_t(ctrl.vector_d1 * data_byte)
                    for i_d0 in range(ctrl.length_d0):
                        self._emit(ds_write(f'{self.v_sst_os()}', f'{self.v_src()}+{i_d0*ctrl.vector_d1}', i_d0 * ctrl.stride_d0))
                        issue_cnt += ds_write.get_issues()
            else:
                #if ctrl.length_d1 == 2 and ctrl.length_d0 in (2, 4, 6, 8):
                if ctrl.length_d1 == 2 and ctrl.length_d0 == 2 and ctrl.need_transpose == 1:
                    swap_sequencer = simple_swap_sequencer_t(ctrl.length_d0, ctrl.length_d1)
                    swap_per_row = swap_sequencer.get_swap_per_row()
                    start_id_per_row = swap_sequencer.get_start_id_per_row()

                    assert ctrl.length_d0 == len(swap_per_row) and ctrl.length_d0 == len(start_id_per_row), f"length_d0:{ctrl.length_d0}, len:{len(swap_per_row)}, s:{start_id_per_row}"

                    ds_write = inst_ds_write_t(ctrl.vector_d1 * data_byte)
                    for i_d0 in range(ctrl.length_d0):
                        swap = swap_per_row[i_d0]
                        s_id = start_id_per_row[i_d0]
                        if swap:
                            self._emit(f"v_swap_b32 v[{self.v_src()}+{swap[0]}], v[{self.v_src()}+{swap[1]}]")
                        self._emit(ds_write(f'{self.v_sst_os()}', f'{self.v_src()}+{s_id}', i_d0 * ctrl.stride_d0))
                        issue_cnt += ds_write.get_issues()

                else:
                    if ctrl.need_transpose == 1:
                        assert ctrl.v_tmp != None
                        trans_seq = simple_transpose_sequencer_t(ctrl.length_d0, ctrl.length_d1)
                        ds_write = inst_ds_write_t(ctrl.vector_d1 * data_byte)
                        for i_d0 in range(ctrl.length_d0):
                            s_id = trans_seq.get_start_id_per_row()[i_d0]
                            for j in range(len(s_id)):
                                self._emit(f"v_mov_b32 v[{ctrl.v_tmp(j)}], v[{self.v_src()}+{s_id[j]}]")
                                self._emit(ds_write(f'{self.v_sst_os()}', f'{ctrl.v_tmp()}', i_d0 * ctrl.stride_d0))
                                issue_cnt += ds_write.get_issues()
                    else:
                        ds_write = inst_ds_write_t(ctrl.vector_d1 * data_byte)
                        for i_d0 in range(ctrl.length_d0):
                            self._emit(ds_write(f'{self.v_sst_os()}', f'{self.v_src()}+{i_d0*ctrl.vector_d1}', i_d0 * ctrl.stride_d0))
                            issue_cnt += ds_write.get_issues()
        else:
            assert ctrl.length_d1 % ctrl.vector_d1 == 0
            assert ctrl.stride_d1 != 1
            num_vector_d1 = ctrl.length_d1 // ctrl.vector_d1
            ds_write2 = inst_ds_write2_likely_t(self.mc, 2, ctrl.vector_d1 * data_byte, ctrl.stride_d1)
            if ctrl.src_order == 0:
                for i_d0 in range(ctrl.length_d0):
                    for i_d1 in range(num_vector_d1 // 2):
                        i_offset = i_d0 * ctrl.stride_d0 + 2* i_d1 * ctrl.stride_d1
                        self._emit(ds_write2(f'{self.v_sst_os()}',
                                f'{self.v_src()}+{i_d0 * ctrl.length_d1 + 2*i_d1*ctrl.vector_d1}',
                                i_offset))
                        issue_cnt += ds_write2.get_issues(i_offset)
            else:
                # assert False, "this order, length_d1 and ctrl.vector_d1 has no means if not equal"
                # assert ctrl.v_tmp != None
                trans_seq = simple_transpose_sequencer_t(ctrl.length_d0, ctrl.length_d1)
                for i_d0 in range(ctrl.length_d0):
                    s_id = trans_seq.get_start_id_per_row()[i_d0]
                    # for j in range(len(s_id)):
                    #     self._emit(f"v_mov_b32 v[{ctrl.v_tmp(j)}], v[{self.v_src()}+{s_id[j]}]")
                    # for i_d1 in range(num_vector_d1 // 2):
                    #     i_offset = i_d0 * ctrl.stride_d0 + 2* i_d1 * ctrl.stride_d1
                    #     self._emit(ds_write2(f'{self.v_sst_os()}',
                    #             ctrl.v_tmp(i_d1 * num_vector_d1 // 2),
                    #             i_offset))
                    #     issue_cnt += ds_write2.get_issues(i_offset)
                    for i_d1 in range(num_vector_d1 // 2):
                        i_offset = i_d0 * ctrl.stride_d0 + 2* i_d1 * ctrl.stride_d1
                        self._emit(ds_write2(f'{self.v_sst_os()}',
                                (self.v_src(s_id[i_d1 * 2]), self.v_src(s_id[i_d1 * 2 + 1])),
                                i_offset))
                        issue_cnt += ds_write2.get_issues(i_offset)

        self.issue_cnt = issue_cnt

    def get_issues(self):
        #assert False, "tobe implemented"
        #return self.ctrl.length_d0
        with self._deferred_context():
            self.emit()
        return self.issue_cnt

class ctrl_3d_shared_store_t(object):
    '''
    d0 x d1 x dp (d pack)
    '''
    def __init__(self):
        self.length_d0 = 1        # is d0 is 1, it is indeed 1d access
        self.length_d1 = 1
        self.length_dp = 1
        self.vector_dp = 1
        self.length_dv = 1
        self.vector_dv = 1
        self.stride_d0 = 1        # stride
        self.stride_d1 = 1         # if have stride_d1, then each d1 may have stride
        self.precision = 'fp32'      # 'fp32', 'fp16', ...
        self.src_order = 0  # 0-d0,d1, 1-d1,d0
        self.need_transpose = 1
        self.v_tmp = None   # used when order is 1 and consider shuffle

    def serialize(self):
        return f"length_d0:{self.length_d0}, length_d1:{self.length_d1}, length_dp:{self.length_dp}, vector_dp:{self.vector_dp}, stride_d0:{self.stride_d0}, stride_d1:{self.stride_d1}, precision:{self.precision}, src_order:{self.src_order}"

class macro_igemm_3d_shared_store_t(macro_base_t):
    '''
    this is indeed for
        0: gemm_k * gemm_m/n * k_pack, src_order = 0
        1: gemm_m/n * gemm_k * k_pack, src_order = 1 (unsupported)
    we always want to use k_pack as vector store
    '''
    def __init__(self, mc, ctrl, inline = False):
        assert type(ctrl) is ctrl_3d_shared_store_t
        macro_base_t.__init__(self, mc, inline)
        self.ctrl = ctrl
        self.issue_cnt = 0
        self.declare_arg("v_src")
        self.declare_arg("v_sst_os")
    def name(self):
        ctrl = self.ctrl
        if ctrl.precision == "fp32":
            bits_str = 'b32'
        elif ctrl.precision in ("fp16", "bf16"):
            bits_str = 'b16'
        else:
            assert False

        return f".v_sst_so{ctrl.src_order}_{ctrl.length_d0}x{ctrl.length_d1}x{ctrl.length_dp}x{ctrl.vector_dp}_{bits_str}" + \
                f"_st{ctrl.stride_d0}x{ctrl.stride_d1}"

    def expr(self):
        ctrl = self.ctrl
        # assert ctrl.precision == 'fp32', "TO BE supported"
        data_byte = amdgpu_precision_data_byte(ctrl.precision)
        issue_cnt = 0
        pixel_per_vgpr = 4 // data_byte
        vgpr_per_vector = (ctrl.vector_dp + pixel_per_vgpr - 1) // pixel_per_vgpr
        assert ctrl.length_dp % ctrl.vector_dp == 0
        num_dp = ctrl.length_dp // ctrl.vector_dp
        ds_write = inst_ds_write_t(ctrl.vector_dp * data_byte)
        assert ctrl.length_dv in (1, 2) and ctrl.vector_dv == 1
        num_dv = ctrl.length_dv // ctrl.vector_dv
        assert not(num_dp > 1 and num_dv > 1)
        if ctrl.length_d0 == 1 or ctrl.length_d1 == 1:
            # this is indeed a 2d case.
            if ctrl.length_d0 == 1 and ctrl.length_d1 == 1:
                # further, 1d case
                for i_p in range(num_dp):
                    self._emit(ds_write(f'{self.v_sst_os()}', f'{self.v_src()}+{i_p*vgpr_per_vector}', i_p * ctrl.vector_dp * data_byte))
                    issue_cnt += ds_write.get_issues()

            else:
                length_d = ctrl.length_d0 if ctrl.length_d0 != 1 else ctrl.length_d1
                stride_d = ctrl.stride_d0 if ctrl.length_d0 != 1 else ctrl.stride_d1
                if length_d % 2 == 0 and data_byte == 4 and ctrl.vector_dp in (1, 2):
                    ds_write2 = inst_ds_write2_likely_t(self.mc, 2, ctrl.vector_dp * data_byte, stride_d)
                    for i_d in range(length_d // 2):
                        for i_p in range(num_dp):
                            self._emit(ds_write2(f'{self.v_sst_os()}', f'{self.v_src()}+{(2 * i_d * num_dp + i_p)*vgpr_per_vector}', 2 * i_d * stride_d + i_p * ctrl.vector_dp * data_byte))
                            issue_cnt += ds_write2.get_issues(2 * i_d * stride_d + i_p * ctrl.vector_dp * data_byte)
                else:
                    # nhwc almost all case goes here
                    for i_d in range(0, length_d, num_dv):
                        if num_dv == 1:
                            for i_p in range(num_dp):
                                self._emit(ds_write(f'{self.v_sst_os()}', f'{self.v_src()}+{(i_d * num_dp + i_p)*vgpr_per_vector}', i_d * stride_d + i_p * ctrl.vector_dp * data_byte))
                                issue_cnt += ds_write.get_issues()
                        if num_dv > 1:
                            for i_v in range(num_dv):
                                lo_hi = i_v % 2
                                self._emit(ds_write(f'{self.v_sst_os()}', f'{self.v_src()}+{(i_d // num_dv)*vgpr_per_vector}', (i_d + i_v) * stride_d, lo_hi))
                                issue_cnt += ds_write.get_issues()
        else:
            for i_d0 in range(ctrl.length_d0):
                if ctrl.length_d1 % 2 == 0 and data_byte == 4 and ctrl.vector_dp in (1, 2):
                    ds_write2 = inst_ds_write2_likely_t(self.mc, 2, ctrl.vector_dp * data_byte, ctrl.stride_d1)
                    for i_d1 in range(ctrl.length_d1 // 2):
                        for i_p in range(num_dp):
                            self._emit(ds_write2(f'{self.v_sst_os()}', f'{self.v_src()}+{(i_d0 * ctrl.length_d1 * num_dp + 2 * i_d1 * num_dp + i_p)*vgpr_per_vector}',
                                    i_d0 * ctrl.stride_d0 + 2 * i_d1 * ctrl.stride_d1 + i_p * ctrl.vector_dp * data_byte))
                            issue_cnt += ds_write2.get_issues(i_d0 * ctrl.stride_d0 + 2 * i_d * stride_d + i_p * ctrl.vector_dp * data_byte)
                else:
                    for i_d1 in range(ctrl.length_d1):
                        for i_p in range(num_dp):
                            self._emit(ds_write(f'{self.v_sst_os()}', f'{self.v_src()}+{(i_d0 * ctrl.length_d1 * num_dp + i_d1 * num_dp + i_p)*vgpr_per_vector}',
                                    i_d0 * ctrl.stride_d0 + i_d1 * ctrl.stride_d1 + i_p * ctrl.vector_dp * data_byte))
                            issue_cnt += ds_write.get_issues()

        self.issue_cnt = issue_cnt

    def get_issues(self):
        #assert False, "tobe implemented"
        #return self.ctrl.length_d0
        with self._deferred_context():
            self.emit()
        return self.issue_cnt
