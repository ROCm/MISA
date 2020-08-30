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

import re
import copy


MC_INST_TYPE_SALU = 0
MC_INST_TYPE_VALU = 1
MC_INST_TYPE_SHARE_MEM = 2
MC_INST_TYPE_GLOBAL_MEM = 3
MC_INST_TYPE_OTHER = 4


def get_mc_inst_type(inst_str):
    '''
    we don't care some type of inst like s_mem, typed memory
    '''
    def check_inst_prefix(self, istr, check_list):
        for cl in check_list:
            if inst.startswith(cl):
                return True
        else:
            return False

    def is_salu(istr):
        return check_inst_prefix(istr, ['s_'])
    def is_value(istr):
        return check_inst_prefix(istr, ['v_'])
    def is_share_mem(istr):
        return check_inst_prefix(istr, ['ds_'])
    def is_global_mem(istr):
        return check_inst_prefix(istr, ['global_', 'buffer_'])

    istr = inst_str.strip()
    istr = re.sub(' +', ' ', istr)     # remove multiple space character
    istr = istr.split(' ')[0]
    if is_salu(istr):
        return MC_INST_TYPE_SALU
    if is_value(istr):
        return MC_INST_TYPE_VALU
    if is_share_mem(istr):
        return MC_INST_TYPE_SHARE_MEM
    if is_global_mem(istr):
        return MC_INST_TYPE_GLOBAL_MEM
    return MC_INST_TYPE_OTHER

class mc_inst_t(object):
    '''
    MCInst, single machine code instruction
    '''
    def __init__(self, inst_str):
        self.inst_str = inst_str

    def type(self):
        return get_mc_inst_type(self.inst_str)

    def __call__(self):
        return self.inst_str


def create_mc_inst(inst_str):
    '''
    since we use simple string as instruction, no concept of opcode, oprand, type, etc...
    so every single line string is just like a machine instruction
    '''
    assert type(inst_str) is str
    istr = inst_str.strip()
    if len(istr) == 0:
        return None
    if istr[0] in (';', '/', '.'):      # ignore comment, directive like .set, .macro
        return None
    return mc_inst_t(inst_str)


class machine_basic_block_t(object):
    '''
    machine basic block(mbb), sequence of mc_inst_t
    '''
    def __init__(self, mc_inst_list):
        assert type(mc_inst_list) is list
        self.mc_inst_list = mc_inst_list

    def length(self):
        return len(self.mc_inst_list)
    
    def mc_inst(self, idx = 0):
        assert idx <= len(self.mc_inst_list)
        return self.mc_inst_list[idx]
    
    def __call__(self):
        return '\n'.join([inst() for inst in self.mc_inst_list])

def create_machine_basic_block(multi_line_inst_str):
    '''
    an post analysis and construction of mbb, only based on string parse.
    input is multiple string (\n), output is list of mbb

    basically one instruction construct a mbb.
    but sometimes we'd like several inst together, like that between INST_MBB_START/INST_MBB_END pair

    TODO: not support recursive start/end pair
    '''
    class parse_mbb_list_t(object):
        STATE_NORMAL = 0
        STATE_PARSING_MBB = 0

        INST_MBB_START = ['v_cmpx', 's_and_saveexec']
        INST_MBB_END = ['s_mov_b64 exec', 's_or_b64 exec']

        def is_mbb_start(self, istr):
            _istr = istr.strip()
            _istr = re.sub(' +', ' ', _istr)     # remove multiple space character
            for ms in INST_MBB_START:
                if _istr.startswith(ms):
                    return True
            return False

        def is_mbb_end(self, istr):
            _istr = istr.strip()
            _istr = re.sub(' +', ' ', _istr)     # remove multiple space character
            for ms in INST_MBB_END:
                if _istr.startswith(ms):
                    return True
            return False

        def parse(self, multi_line_inst_str):
            istrs = multi_line_inst_str.split('\n')
            mbbs = list()
            mc_inst_buffer = list()
            state = self.STATE_NORMAL
            if len(istrs) == 0:
                return None
            for istr in istrs:
                mc_inst = create_mc_inst(istr)
                if not mc_inst:
                    continue
                if state == self.STATE_NORMAL:
                    if self.is_mbb_start(istr):
                        mc_inst_buffer.clear()
                        mc_inst_buffer.append(mc_inst)
                        state = self.STATE_PARSING_MBB
                    else:
                        mbbs.append(machine_basic_block_t(copy.copy([mc_inst])))
                else:
                    if self.is_mbb_start(istr):
                        assert False, f'not support recursive start/end for now, with {istr}'
                    if self.is_mbb_end(istr):
                        mc_inst_buffer.append(mc_inst)
                        mbbs.append(machine_basic_block_t(copy.copy(mc_inst_buffer)))
                        state = self.STATE_NORMAL
                        mc_inst_buffer.clear()
                    else:
                        mc_inst_buffer.append(mc_inst)
            assert len(mbbs) != 0, f"nonthing parsed from input inst: {multi_line_inst_str}"
            return mbbs
    parser = parse_mbb_list_t()
    return parser(multi_line_inst_str)
