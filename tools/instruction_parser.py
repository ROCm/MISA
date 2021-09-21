import argparse
from os import name
import sys
from typing import Any, List
import urllib.error
import urllib.request

from bs4 import BeautifulSoup
from bs4 import element as bs4Element
import re
import itertools


class instr_type():

    class arg_format():
        def __init__(self) -> None:
            self.names:List[str] = []
            self.arg_positions:List[int] = []
        
        def add_arg(self, name, offset):
            self.names.append(name)
            self.arg_positions.append(offset)
            if(name == 'MODIFIERS'):
                self.mod_offset = offset

        def get_arg_type(self, offset):
            if(offset >= self.arg_positions[-1]):
                return self.names[-1]
            return self.names[self.arg_positions.index(offset)]

    class i_argument():
        def __init__(self, name, f_type, arg_type) -> None:
            self.name = name
            self.f_type = f_type
            self.arg_type = arg_type
            self.opt_type = None

    class instruction():
        def __init__(self, name:str) -> None:
            self.name = name
            self.arg_list:List[instr_type.i_argument] = []

    def __init__(self, name:str) -> None:
        self.name = name
        self._arg_format = instr_type.arg_format()
        self.inst_list:List[instr_type.instruction]=[]

    def arg_format_append(self, name, offset):
        self._arg_format.add_arg(name, offset)
    
    def add_new_instr(self, name):
        last_inst = instr_type.instruction(name)
        self.inst_list.append(last_inst)
        self.last_inst = last_inst

    def add_arg_to_last_inst(self, name, offset, arg_type):
        arg_f_type = self._arg_format.get_arg_type(offset)
        arg =  self.i_argument(name, arg_f_type, arg_type)
        self.last_inst.arg_list.append(arg)

    def add_opt_type_to_last_arg(self, opt_type):
        last_arg = self.last_inst.arg_list[-1]
        last_arg.opt_type = opt_type


def parse_instruction_name(l:str):
    return l.translate(str.maketrans('', '', '— '))

def parse_instruction_argument(arg:bs4Element.Tag):
    type_name = arg['href']
    name = arg.string
    if(type_name.find('AMDGPUModifierSyntax') != -1):
        return (name, type_name.split("amdgpu-synid-",1)[1])
    return (name, type_name[:type_name.index('.')])

def parse_argument_mod(a_type:bs4Element.Tag)->str:
    name, type_name = parse_instruction_argument(a_type)
    if(type_name in ['gfx10_m', 'gfx10_dst']):
        return type_name
    return ''


def parse_gfx_instruction_html_file(fileName):
    instr_types:List[instr_type] = []

    with open(fileName) as fp:
        soup = BeautifulSoup(fp, "html.parser")
    inst = soup.find("section", id="instructions") 
    inst_pack_list = inst.find_all("section")


    for i_p in inst_pack_list:
        curent_pack = instr_type(i_p['id'])
        instr_types.append(curent_pack)
        

        last_offset = 0
        next_elem_is_arg_mod = False

        litblock = i_p.find("pre")
        
        for i in litblock:
            if(type(i) == bs4Element.Tag):
                if(i.name == 'a'):
                    if(next_elem_is_arg_mod):
                        a_type = parse_argument_mod(i)
                        if(a_type != ''):
                            curent_pack.add_opt_type_to_last_arg(a_type)
                        next_elem_is_arg_mod=False
                    else:
                        arg_name, type_name = parse_instruction_argument(i)
                        curent_pack.add_arg_to_last_inst(arg_name, last_offset, type_name)
                elif(i.name == 'strong'):
                    curent_pack.arg_format_append(i.string, last_offset)
            else: #string
                lines = re.split('\\n', i)
                if(len(lines) > 1):
                    for line in lines:
                        p = parse_instruction_name(line)
                        if(p != ''):
                            curent_pack.add_new_instr(p)
                            last_offset = len(line)
                    continue
                elif(lines[0]==':'):
                    next_elem_is_arg_mod = True
            
            last_offset += len(i.string)
    return instr_types


def create_G_class_code(instr:instr_type, name) -> List:
    header = f'class {name}(inst_base): \n'

    init_head_l = [f'\tdef __init__(self, INSTRUCTION:str']
    init_body_l = [f'\t\tsuper().__init__(instruction_type.SMEM, INSTRUCTION)\n']
    str_header = f'\tdef __str__(self): \n'
    

    args = instr._arg_format.names
    no_mods = True
    for i in args[1:]:
        if(i != 'MODIFIERS'): 
            cur_str_head = f', {i}:Union[regVar,None,Any]'
            cur_str_body = f'\t\tself.{i} = {i} \n'
        else:
            no_mods = False
            # TODO
            #cur_str_head = f', **MODIFIERS'
            #cur_str_body = f'\t\tself.{i} = \' \'.join({i}.values()) \n'
            # instead of TODO
            cur_str_head = f', MODIFIERS:str'
            cur_str_body = f'\t\tself.{i} = {i} \n'
        init_head_l.append(cur_str_head)
        init_body_l.append(cur_str_body)
    init_head_l.append('): \n')
    init_head = ''.join(init_head_l)

    str_body_l = []
    args_len = len(args) - (0 if no_mods else 1)
    str_mod = '' if no_mods else '{self.MODIFIERS}'
    str_label = '{self.label}'
    if(args_len > 1):
        self_args_l = []
        for i in args[1:args_len]:
            self_args_l.append(f'self.{i}')

        str_body_l.append(f"\t\targs_l = filter(None.__ne__, [{','.join(self_args_l)}]) \n")

        str_join = "{', '.join(map(str, args_l))}"

        str_body_end = f"\t\treturn f\"{str_label} {str_join} {str_mod}\" \n"
        str_body_l.append(str_body_end)
    else:
        str_mod = '{self.MODIFIERS}'
        str_body_l.append(f"\t\treturn f\"{str_label} {str_mod}\" \n")

    return [header, init_head, *init_body_l, str_header, *str_body_l]


def create_caller_class_code(name, suffix_name='') -> List:
    header = f'class {name}_instr_caller{suffix_name}(inst_caller_base): \n'

    init = f'\tdef __init__(self, insturction_container) -> None:\n \
    \t\tsuper().__init__(insturction_container)\n'

    return [header, init]

def create_instruction_caller_func_code(cur_i:instr_type.instruction, arg_format_l:List[str], group_name:str):
    i_name = cur_i.name
    header_l = [f'\tdef {i_name}(self']
    i_body_l = [f'\t\treturn self.ic_pb({group_name}(\'{i_name}\'']
    arg_list = cur_i.arg_list
    cur_i_arg = 0
    if(len(arg_list) > 1):
        cur_arg = arg_list[cur_i_arg]
    else:
        cur_arg = instr_type.i_argument("NONE", "NONE", "NONE")
    
    s_arg_t = ':regVar'
    for i in arg_format_l[1:]:
        if(i != 'MODIFIERS'):
            if(cur_arg.f_type == i):
                if(cur_arg.name == 'vcc'):
                    s = f', {cur_arg.name}_{cur_arg.f_type}'
                else:
                    s = f', {cur_arg.name}'
                i_body_l.append(s)
                header_l.append(s)
                arg_type = cur_arg.arg_type
                if(arg_type == 'gfx10_vaddr_5'):
                    header_l.append(':Union[regVar]')
                elif (arg_type in ['gfx10_ssrc', 'gfx10_ssrc_1', 'gfx10_src_2', 'gfx10_src_3', 'gfx10_src_4','gfx10_src_8', 'gfx10_ssrc_8']):
                    header_l.append(':Union[regVar,literal,const, int]')
                elif (arg_type in ['gfx10_src_6, gfx10_src_7', 'gfx10_src_1', 'gfx10_src', 'gfx10_ssrc_6', 'gfx10_ssrc_7', 'gfx10_soffset']):
                    header_l.append(':Union[regVar,const]')
                elif(arg_type in ['gfx10_imm16', 'gfx10_simm32']):
                    header_l.append(f':{arg_type.split("gfx10_",1)[1]}_t')
                elif(arg_type in ['gfx10_soffset_1']):
                    header_l.append(':Union[regVar,simm21_t,int]')
                elif(arg_type in ['gfx10_soffset_2']):
                    header_l.append(':Union[regVar,uimm20_t,int]')
                elif(arg_type in ['gfx10_label']):
                    header_l.append(f':{arg_type.split("gfx10_",1)[1]}_t')
                elif(arg_type == 'gfx10_msg'):
                    header_l.append(':str')
                else:
                    header_l.append(':regVar')
                
                cur_i_arg += 1
                if(len(arg_list) > cur_i_arg):
                    cur_arg = arg_list[cur_i_arg]
            else:
                i_body_l.append(f', None')

    # TODO
    #if(arg_format_l[-1] == 'MODIFIERS'):
    #    for cur_i_arg in range(len(arg_list)):
    #        cur_arg = arg_list[cur_i_arg]
    #        if(cur_arg.opt_type in ['fmt', 'offset', 'dim', 'dmask', 'ufmt',
    #             'dpp8_sel', 'dpp16_sel', 'dpp16_ctrl','dpp32_ctrl', 'fi',
    #             'dpp64_ctrl', 'row_mask', 'bank_mask', 'dst_sel', 'dst_unused',
    #             'src0_sel', 'src1_sel', 'op_sel', 'dpp_op_sel', 'omod', 'op_sel_hi',
    #             'neg_lo', 'neg_hi', 'm_op_sel', 'm_op_sel_hi', 'cbsz', 'abid', 'blgp']):
    #Instead of TODO
    doc = []
    if(cur_arg.f_type == 'MODIFIERS'):
        s = f', {cur_arg.f_type}'
        i_body_l.append(s)
        header_l.append(s)
        header_l.append(':str=\'\'')
        
        doc.append('\t\t""":param str MODIFIERS:')
        for cur_i_arg in range(cur_i_arg, len(arg_list)):
            cur_arg = arg_list[cur_i_arg]
            doc.append(f' {cur_arg.name}')
        doc.append('"""\n')
    else:
        if(arg_format_l[-1] == 'MODIFIERS'):
            i_body_l.append(', \'\'')

    header_l.append(f'):\n')
    i_body_l.append(f'))\n')
    if(doc):
        header_l.extend(doc)
    return [*header_l, *i_body_l]

def create_py_code(i_types:List[instr_type], outfile, suffix_name=''):
    outfile.writelines([
        "#generated by instruction_parser.py\n",
        "from typing import Any, Union\n",
        "from python.codegen.gpu_instruct import inst_base, inst_caller_base, instruction_type\n",
        "from python.codegen.gpu_data_types import *\n\n"
        ])

    for i_t in i_types:
        group_name = i_t.name
        G_class_name = f'{group_name}_base{suffix_name}'
        outfile.writelines(create_G_class_code(i_t, G_class_name))
        outfile.writelines(create_caller_class_code(group_name,suffix_name))
        i_list = i_t.inst_list
        for cur_i in i_list:
            outfile.writelines(create_instruction_caller_func_code(cur_i,i_t._arg_format.names, G_class_name))
            

def main():
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--infile', nargs='?', type=argparse.FileType('r'),
                           default=sys.stdin)
    argparser.add_argument('--class_suffix', nargs='?',default='')
    argparser.add_argument('--outfile', nargs='?', type=argparse.FileType('w'),
                           default=sys.stdout)
    args = argparser.parse_args()
    
    outfile = args.outfile
    if(outfile == None):
        out_name = os.path.splitext(args.infile)[0]
        outfile = open(out_name, "r")

    print('Reading', args.infile.name)
    
    instr_types = parse_gfx_instruction_html_file(args.infile.name)
    
    print('Writing', outfile.name)
    
    create_py_code(instr_types, outfile, args.class_suffix)


    
if __name__ == '__main__':
    main()