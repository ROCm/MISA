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

import re
import copy
from .mc import *

MC_INST_TYPE_SALU = 0
MC_INST_TYPE_VALU = 1
MC_INST_TYPE_SHARE_MEM = 2
MC_INST_TYPE_GLOBAL_MEM = 3
MC_INST_TYPE_LEGACY_MACRO = 4       # like macro_c_clear_t. this is a hack
MC_INST_TYPE_COMMENTS = 5
MC_INST_TYPE_PREDEFINE_IF = 6
MC_INST_TYPE_PREDEFINE_ENDIF = 7
MC_INST_TYPE_OTHER = 8

def get_mc_inst_op(inst_str):
    istr = inst_str.strip()
    istr = re.sub(' +', ' ', istr)     # remove multiple space character
    istr = istr.split(' ')[0]
    return istr

def _check_inst_prefix(istr, check_list):
    for cl in check_list:
        if istr.startswith(cl):
            return True
    else:
        return False

def mc_inst_is_salu(inst_op):
    return _check_inst_prefix(inst_op, ['s_'])
def mc_inst_is_value(inst_op):
    return _check_inst_prefix(inst_op, ['v_'])
def mc_inst_is_share_mem(inst_op):
    return _check_inst_prefix(inst_op, ['ds_'])
def mc_inst_is_global_mem(inst_op):
    return _check_inst_prefix(inst_op, ['global_', 'buffer_'])
def mc_inst_is_legacy_macro(inst_op):
    return _check_inst_prefix(inst_op, ['.v_clear_nc'])
def mc_inst_is_predefine_if(inst_op):
    return _check_inst_prefix(inst_op, ['.if'])
def mc_inst_is_predefine_endif(inst_op):
    return _check_inst_prefix(inst_op, ['.endif'])

def get_mc_inst_type(inst_str):
    '''
    we don't care some type of inst like s_mem, typed memory
    '''
    inst_op = get_mc_inst_op(inst_str)

    if mc_inst_is_salu(inst_op):
        return MC_INST_TYPE_SALU
    if mc_inst_is_value(inst_op):
        return MC_INST_TYPE_VALU
    if mc_inst_is_share_mem(inst_op):
        return MC_INST_TYPE_SHARE_MEM
    if mc_inst_is_global_mem(inst_op):
        return MC_INST_TYPE_GLOBAL_MEM
    if mc_inst_is_legacy_macro(inst_op):
        return MC_INST_TYPE_LEGACY_MACRO
    if mc_inst_is_predefine_if(inst_op):
        return MC_INST_TYPE_PREDEFINE_IF
    if mc_inst_is_predefine_endif(inst_op):
        return MC_INST_TYPE_PREDEFINE_ENDIF
    return MC_INST_TYPE_OTHER

class mc_inst_t(object):
    '''
    MCInst, single machine code instruction
    '''
    def __init__(self, inst_str):
        self.inst_str = inst_str.strip()    # remove leading/trailing space here!

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

    if mc_inst_is_legacy_macro(get_mc_inst_op(istr)):
        return mc_inst_t(inst_str)

    def inst_in_directive_white_list(inst):
        # TODO: with the .if .. .else, this should group into a single mbb
        if istr[0] != '.':
            return False
        asm_directive_white_list = ['.if', '.ifdef', '.else', '.endif']
        for itm in asm_directive_white_list:
            if inst.startswith(itm):
                return True
        return False

    if istr[0] in (';', '/', '.', '\n'):      # ignore comment, directive like .set, .macro
        if not inst_in_directive_white_list(istr):
            return None
    # print(f'[XX] {istr[0]}, {inst_str}')
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
        # assert idx <= len(self.mc_inst_list)
        # print(f'xxxx {self.mc_inst_list}')
        return self.mc_inst_list[idx]
    
    def __call__(self):
        return '\n'.join([inst() for inst in self.mc_inst_list])

    def count(self, inst_str):
        cnt = 0
        for mc_inst in self.mc_inst_list:
            if inst_str in mc_inst.inst_str:
                cnt = cnt + 1
        return cnt

    def dump(self):
        print(f'mbb len:{self.length()}, content:')
        print(self())
        print('-----------------------------------')

def machine_basic_block_call(p, mbb):
    '''
    to pretty print mbb, the indent
    currently p can not be mc_base_t directly. must be some child class
    '''
    mbb_lines = mbb().split('\n')
    with p._deferred_context():
        for line in mbb_lines:
            p._emit(line)
    return p._get_deferred()

def create_machine_basic_block(multi_line_inst_str, **option):
    '''
    an post analysis and construction of mbb, only based on string parse.
    input is multiple string (\n), output is list of mbb

    basically one instruction construct a mbb.
    but sometimes we'd like several inst together, like that between INST_MBB_START/INST_MBB_END pair

    TODO: not support recursive start/end pair

    option:
    group_mbb_by_end_of_inst_op    :   str,  group several mc_inst into mbb, each mbb is by end of this value
    merge_mbb                      :   int,  do not split into multiple mbb
    dup_inst_per_mbb               ;   str,  each mbb shold contains multiple instructions.
                                             should have format: "xxxx,n" where "xxxx" is instruction need to duplicate, "n" is number of duplication
    '''
    class parse_mbb_list_t(object):
        STATE_NORMAL = 0
        STATE_PARSING_MBB = 1
        STATE_PARSING_MBB_IN_PREDEFINE = 2

        INST_MBB_START = ['v_cmpx']
        INST_MBB_END = ['s_mov_b64 exec', 's_or_b64 exec']

        INST_MBB_START_PREDEFINE = ['.if']
        INST_MBB_END_PREDEFINE = ['.endif']

        def is_mbb_start_macro_c_clear(self, current_index, istrs_list):
            '''
            special rule for macro_c_clear_t
            '''
            assert type(istrs_list) is list
            current_istr = istrs_list[current_index]
            current_inst_op = get_mc_inst_op(current_istr)
            if mc_inst_is_legacy_macro(current_inst_op):
                if current_inst_op.startswith('.v_clear_nc'):
                    return True
            return False

        def is_mbb_start_cmp_and_exec_block(self, current_index, istrs_list):
            assert type(istrs_list) is list
            current_istr = istrs_list[current_index]
            # current_mc_inst = create_mc_inst(current_istr)
            current_inst_op = get_mc_inst_op(current_istr)
            if current_inst_op.startswith('v_cmp_'):
                #print('asdadds XXXXX')
                for next_index in range(current_index+1, len(istrs_list)):
                    next_istr =  istrs_list[next_index]
                    next_mc_inst = create_mc_inst(next_istr)
                    next_inst_op = get_mc_inst_op(next_istr)
                    #print(f'   next_inst_op:{next_inst_op} ')
                    if not next_mc_inst:
                        continue
                    if next_inst_op.startswith('s_and_saveexec'):
                        return True
                    return False
            return False
        
        def is_mbb_start_bfe_and_cmpx_block(self, current_index, istrs_list):
            assert type(istrs_list) is list
            current_istr = istrs_list[current_index]
            # current_mc_inst = create_mc_inst(current_istr)
            current_inst_op = get_mc_inst_op(current_istr)
            if current_inst_op.startswith('v_bfe_u32'):
                #print('asdadds XXXXX')
                for next_index in range(current_index+1, len(istrs_list)):
                    next_istr =  istrs_list[next_index]
                    next_mc_inst = create_mc_inst(next_istr)
                    next_inst_op = get_mc_inst_op(next_istr)
                    #print(f'   next_inst_op:{next_inst_op} ')
                    if not next_mc_inst:
                        continue
                    if next_inst_op.startswith('v_cmp'):
                        return True
                    return False
            return False

        def is_mbb_start(self, istr):
            _istr = istr.strip()
            _istr = re.sub(' +', ' ', _istr)     # remove multiple space character
            for ms in self.INST_MBB_START:
                if _istr.startswith(ms):
                    return True
            return False

        def is_mbb_end(self, istr):
            _istr = istr.strip()
            _istr = re.sub(' +', ' ', _istr)     # remove multiple space character
            for ms in self.INST_MBB_END:
                if _istr.startswith(ms):
                    return True
            return False

        def is_mbb_start_predefine(self, istr):
            _istr = istr.strip()
            _istr = re.sub(' +', ' ', _istr)     # remove multiple space character
            for ms in self.INST_MBB_START_PREDEFINE:
                if _istr.startswith(ms):
                    return True
            return False

        def is_mbb_end_predefine(self, istr):
            _istr = istr.strip()
            _istr = re.sub(' +', ' ', _istr)     # remove multiple space character
            for ms in self.INST_MBB_END_PREDEFINE:
                if _istr.startswith(ms):
                    return True
            return False

        def parse(self, multi_line_inst_str, **option):
            def get_dict_with_default(dictionary, key, default_value):
                if key in dictionary:
                    return dictionary[key]
                else:
                    return default_value

            merge_mbb = get_dict_with_default(option, "merge_mbb", 0)
            dup_inst_per_mbb = get_dict_with_default(option, "dup_inst_per_mbb", "off")

            group_mbb_by_end_of_inst_op = get_dict_with_default(option, "group_mbb_by_end_of_inst_op", "")
            def match_group_mbb_by_end_of_inst_op(inst_op):
                if type(group_mbb_by_end_of_inst_op) is str:
                    return inst_op.startswith(group_mbb_by_end_of_inst_op)
                if type(group_mbb_by_end_of_inst_op) in (list, tuple):
                    for g in group_mbb_by_end_of_inst_op:
                        if inst_op.startswith(g):
                            return True
                    return False
                assert False
            
            def match_group_mbb_by_end_of_inst_op_lookback(current_index, istrs_list):
                if current_index <= 0:
                    return False
                for prev_index in range(current_index - 1, -1, -1):
                    prev_istr =  istrs_list[prev_index]
                    prev_mc_inst = create_mc_inst(prev_istr)
                    prev_inst_op = get_mc_inst_op(prev_istr)
                    if not prev_inst_op:
                        continue
                    return match_group_mbb_by_end_of_inst_op(prev_inst_op)
                return False    # nonthing to search

            istrs = multi_line_inst_str.split('\n')
            mbbs = list()
            mc_inst_buffer = list()
            state = self.STATE_NORMAL

            #print(multi_line_inst_str)
            #print(f'========================================={ 1 if group_mbb_by_end_of_inst_op else 0}')

            if len(istrs) == 0:
                return None
            
            for i, istr in enumerate(istrs):
                mc_inst = create_mc_inst(istr)
                if not mc_inst:
                    continue

                # merge every string into a single mbb
                if merge_mbb:
                    mc_inst_buffer.append(mc_inst)
                    continue

                # early pass rule
                if self.is_mbb_start_macro_c_clear(i, istrs):
                    '''
                    whatever state, always insert a new mbb
                    '''
                    mbbs.append(machine_basic_block_t(copy.copy([mc_inst])))
                    continue

                if group_mbb_by_end_of_inst_op:
                    inst_op = get_mc_inst_op(istr)
                    if state == self.STATE_NORMAL:
                        if self.is_mbb_start_predefine(istr):
                            mc_inst_buffer.append(mc_inst)
                            state = self.STATE_PARSING_MBB_IN_PREDEFINE
                        elif match_group_mbb_by_end_of_inst_op(inst_op):
                            mc_inst_buffer.append(mc_inst)
                            mbbs.append(machine_basic_block_t(copy.copy(mc_inst_buffer)))
                            # for yy in mc_inst_buffer:
                            #     print(f"  +++inst:{yy()}")
                            # print(f"  +++--------------------")
                            mc_inst_buffer.clear()
                        else:
                            mc_inst_buffer.append(mc_inst)
                            state = self.STATE_PARSING_MBB
                    elif state == self.STATE_PARSING_MBB:
                        if self.is_mbb_start_predefine(istr):
                            mc_inst_buffer.append(mc_inst)
                            state = self.STATE_PARSING_MBB_IN_PREDEFINE
                        elif match_group_mbb_by_end_of_inst_op(inst_op):
                            mc_inst_buffer.append(mc_inst)
                            mbbs.append(machine_basic_block_t(copy.copy(mc_inst_buffer)))
                            # print(f'xxxxx_ {mc_inst_buffer}, len:{len(mc_inst_buffer)}')
                            # for yy in mc_inst_buffer:
                            #     print(f"  inst:{yy()}")
                            # machine_basic_block_t(copy.copy(mc_inst_buffer)).dump()
                            state = self.STATE_NORMAL
                            #for yy in mc_inst_buffer:
                            #    print(f"  +++inst:{yy()}")
                            #print(f"  +++--------------------")
                            mc_inst_buffer.clear()
                        else:
                            mc_inst_buffer.append(mc_inst)
                    elif state == self.STATE_PARSING_MBB_IN_PREDEFINE:
                        if self.is_mbb_end_predefine(istr):
                            '''
                            only switch back, but not cut the mc_inst_buffer into mbb here
                            '''
                            mc_inst_buffer.append(mc_inst)
                            if match_group_mbb_by_end_of_inst_op_lookback(i, istrs):
                                mbbs.append(machine_basic_block_t(copy.copy(mc_inst_buffer)))
                                #for yy in mc_inst_buffer:
                                #    print(f"  +++inst:{yy()}")
                                #print(f"  +++--------------------")
                                mc_inst_buffer.clear()
                            state = self.STATE_NORMAL
                        else:
                            mc_inst_buffer.append(mc_inst)
                    else:
                        assert False
                else:
                    if state == self.STATE_NORMAL:
                        if self.is_mbb_start(istr) or self.is_mbb_start_predefine(istr) or self.is_mbb_start_cmp_and_exec_block(i, istrs) \
                                or self.is_mbb_start_bfe_and_cmpx_block(i, istrs):
                            mc_inst_buffer.clear()
                            mc_inst_buffer.append(mc_inst)
                            state = self.STATE_PARSING_MBB
                        else:
                            mbbs.append(machine_basic_block_t(copy.copy([mc_inst])))
                    else:
                        if self.is_mbb_start(istr):
                            assert i > 1
                            if self.is_mbb_start_bfe_and_cmpx_block(i - 1, istrs):
                                # TODO: this require bfe and cmpx have no other lines in between
                                pass
                            else:
                                assert False, f'not support recursive start/end for now, with {i}:{istr}, {istrs}'
                        if self.is_mbb_end(istr) or self.is_mbb_end_predefine(istr):
                            mc_inst_buffer.append(mc_inst)
                            mbbs.append(machine_basic_block_t(copy.copy(mc_inst_buffer)))
                            state = self.STATE_NORMAL
                            mc_inst_buffer.clear()
                        else:
                            mc_inst_buffer.append(mc_inst)

            # give a chance that inst buffer still contains no-terminated mc_inst
            if len(mc_inst_buffer) != 0:
                #print('<*****************>')
                mbbs.append(machine_basic_block_t(copy.copy(mc_inst_buffer)))
                mc_inst_buffer.clear()
            #assert len(mbbs) != 0, f"nonthing parsed from input inst: {multi_line_inst_str}"
            if len(mbbs) == 0:
                return list()   # silently return empty list
            #print('************************')
            #for y in mbbs:
            #    y.dump()
            #print('************************')
            if dup_inst_per_mbb != "off":
                _dup_str = dup_inst_per_mbb.split(',')
                assert len(_dup_str) == 2
                dup_i = str(_dup_str[0].strip())
                dup_n = int(_dup_str[1].strip())
                new_mbbs = list()
                dup_state = self.STATE_NORMAL
                buf_mbb = None
                buf_n = 0
                for itm in mbbs:
                    if dup_state == self.STATE_NORMAL:
                        current_dup_n = itm.count(dup_i)
                        if current_dup_n != 0 and dup_n - current_dup_n > 0:
                            buf_mbb = copy.copy(itm)
                            buf_n = dup_n - current_dup_n
                            dup_state = self.STATE_PARSING_MBB
                        else:
                            new_mbbs.append(copy.copy(itm))
                    elif dup_state == self.STATE_PARSING_MBB:
                        if buf_n != 0:
                            buf_mbb.mc_inst_list += itm.mc_inst_list
                            current_dup_n = itm.count(dup_i)
                            if current_dup_n != 0:
                                buf_n -= current_dup_n
                            if buf_n <= 0:
                                new_mbbs.append(copy.copy(buf_mbb))
                                buf_mbb = None
                                dup_state = self.STATE_NORMAL
                        else:
                            assert False, "should not happen"
                    else:
                        assert False
                if buf_mbb != None:
                    new_mbbs.append(copy.copy(buf_mbb))
                return new_mbbs

            return mbbs
    parser = parse_mbb_list_t()
    return parser.parse(multi_line_inst_str, **option)

def emit_machine_basic_blocks(mc, mbb_lists):
    class simple_mbb_emit_t(mc_base_t):
        def __init__(self, mc, mbb_lists):
            mc_base_t.__init__(self, mc)
            self.mbb_lists = mbb_lists

        def call_mbb(self, mbb):
            mbb_lines = mbb().split('\n')
            with self._deferred_context():
                for line in mbb_lines:
                    self._emit(line)
            return self._get_deferred()
        
        def emit(self):
            with self._deferred_context():
                for mbb in mbb_lists:
                    self._emit(self.call_mbb(mbb))
            return self._get_deferred()

    assert type(mbb_lists) is list
    return simple_mbb_emit_t(mc, mbb_lists).emit()
