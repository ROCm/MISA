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

from .mc import *
from .mbb import *


SCHEDULER_TYPE_SIMPLE_INTERLEAVE = 0

INTERLEAVE_PTN_0 = "mbb0 mfma and related share load, mbb1 global_load and move_slice_window"
INTERLEAVE_PTN_1 = "mbb0 mfma, mbb1 share_store"
INTERLEAVE_PTN_2 = "mbb0 global store, mbb 1 share store"

class simple_interleave_scheduler_t(mc_base_t):
    '''
    2 mbb list, mbb_0 and mbb_1. interleave mbb_1 into first mbb list mbb_0
    mbb_0 is also base mbb
    '''
    def __init__(self, mc, mbb_lists):
        mc_base_t.__init__(self, mc)
        self.mbb_lists = mbb_lists

    def call_mbb(self, mbb):
        '''
        basically this is to deal with indent...
        within mbb, there is no concept of indent.
        but while lowering, we need indent to pretty emit
        '''
        mbb_lines = mbb().split('\n')
        with self._deferred_context():
            for line in mbb_lines:
                self._emit(line)
        return self._get_deferred()

    def lower(self, **options):
        '''
        options:
            interleave_pattern:       str,   desc of inst pattern of mbb_0, mbb_1, see INTERLEAVE_PTN_*
            start_position:           int,   at which base mbb to start interleave. allow -1 to start before base mbb
            min_interval:             list,  how much mbb interval to interleave, per mbb_list[0], [1], ...
            max_interleave_space:     int,   max size of mc_inst that can be interleaved into base mbb
            arch_alu_per_interval:    int,
            global_mem_per_interval:  int,
            share_mem_per_interval:   int,
            
        '''
        def get_dict_with_default(dictionary, key, default_value):
            if key in dictionary:
                return dictionary[key]
            else:
                return default_value
        assert len(self.mbb_lists) == 2, "currently only support 2 mbb list interleave together"

        interleave_pattern = get_dict_with_default(options, "interleave_pattern", INTERLEAVE_PTN_0)
        start_position = get_dict_with_default(options, "start_position", 0)
        min_interval = get_dict_with_default(options, "min_interval", [1] * len(self.mbb_lists))
        max_interleave_space = get_dict_with_default(options, "max_interleave_space", 14)       # mfma 32x32 inst allow at most 14
        arch_alu_per_interval = get_dict_with_default(options, "arch_alu_per_interval", 2)
        global_mem_per_interval = get_dict_with_default(options, "global_mem_per_interval", 1)
        share_mem_per_interval = get_dict_with_default(options, "share_mem_per_interval", 1)

        assert min_interval[0] == 1, 'currently only support base mbb interval is 1'
        interleave_slot = len(self.mbb_lists[0])
        if start_position == -1:
            interleave_slot += 1

        interleave_space = (len(self.mbb_lists[1]) + interleave_slot - 1) // interleave_slot
        assert interleave_space <= max_interleave_space

        mbb_0 = self.mbb_lists[0]
        mbb_1 = self.mbb_lists[1]

        if interleave_pattern == INTERLEAVE_PTN_0:
            # first check how many global load in mbb_1
            assert mbb_1[0].mc_inst(-1).type() == MC_INST_TYPE_GLOBAL_MEM
            num_gmem = 0
            for i in range(len(mbb_1)):
                if mbb_1[i].mc_inst(-1).type() == MC_INST_TYPE_GLOBAL_MEM:
                    num_gmem = num_gmem + 1
                else:
                    break

            # second decide how many global mem to interleave per interval
            gmem_mbb_0_ratio = 2 / 3                          # if num global mem bigger than this of mbb_0 length, need add more per interval
            gmem_per_interval = 1
            while num_gmem * gmem_per_interval >= int(len(mbb_0) * gmem_mbb_0_ratio):
                gmem_per_interval += 1

            num_mbb_0_interleave_gmem = (num_gmem + gmem_per_interval - 1) // gmem_per_interval
            num_mbb_1_left = len(mbb_1) - num_gmem
            num_mbb_0_left = len(mbb_0) - num_mbb_0_interleave_gmem
            mbb_1_left_per_interval = (num_mbb_0_left + num_mbb_1_left - 1) // num_mbb_1_left

            # finaly, go interleave
            with self._deferred_context():
                m0_idx = 0
                m1_idx = 0
                for i in range(num_mbb_0_interleave_gmem):
                    self._emit(self.call_mbb(mbb_0[m0_idx])) ; m0_idx += 1
                    for j in range(gmem_per_interval):
                        if m1_idx < num_gmem:
                            self._emit(self.call_mbb(mbb_1[m1_idx])) ; m1_idx += 1

                for i in range(num_mbb_0_left):
                    self._emit(self.call_mbb(mbb_0[m0_idx])) ; m0_idx += 1
                    for j in range(mbb_1_left_per_interval):
                        if m1_idx < len(mbb_1):
                            self._emit(self.call_mbb(mbb_1[m1_idx])) ; m1_idx += 1
                assert m0_idx == len(mbb_0)
                assert m1_idx == len(mbb_1)
            return self._get_deferred()

        if interleave_pattern == INTERLEAVE_PTN_1:
            # mbb_1[0].dump()
            assert mbb_1[0].mc_inst(-1).type() == MC_INST_TYPE_SHARE_MEM
            num_smem = 0
            for i in range(len(mbb_1)):
                if mbb_1[i].mc_inst(-1).type() == MC_INST_TYPE_SHARE_MEM:
                    num_smem = num_smem + 1
                else:
                    pass         # here we might face with some
            assert num_smem == len(mbb_1)
            smem_per_interleave = (len(mbb_0) - 1 + num_smem - 1) // num_smem
            with self._deferred_context():
                m0_idx = 0
                m1_idx = 0
                for i in range(len(mbb_0)):
                    smem_per_interleave_cnt = 0
                    while True:
                        if m1_idx >= len(mbb_1):
                            break
                        # print(f' --- inst:{mbb_1[m1_idx]()} === {m1_idx}/{len(mbb_1)}, {smem_per_interleave_cnt}/smem_per_interleave:{smem_per_interleave}')
                        self._emit(self.call_mbb(mbb_1[m1_idx]))
                        if mbb_1[m1_idx].mc_inst(-1).type() == MC_INST_TYPE_SHARE_MEM:
                            smem_per_interleave_cnt = smem_per_interleave_cnt + 1
                        m1_idx += 1
                        if smem_per_interleave_cnt >= smem_per_interleave:
                            break
                    self._emit(self.call_mbb(mbb_0[m0_idx]))
                    m0_idx += 1
            return self._get_deferred()


def create_scheduler(mc, mbb_lists, type = SCHEDULER_TYPE_SIMPLE_INTERLEAVE):
    '''
    mbb_lists: list of machine basic blocks, every element is also a list of mbb.
    '''
    if type == SCHEDULER_TYPE_SIMPLE_INTERLEAVE:
        return simple_interleave_scheduler_t(mc, mbb_lists)
    else:
        # TODO might have other type of scheduler
        assert False, "unimplemented scheduler"
