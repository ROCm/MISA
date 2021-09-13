import argparse
from os import name
import sys
from typing import List
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
        def __init__(self, name, f_type) -> None:
            self.name = name
            self.f_type = f_type
            self.opt_type = None

    class instruction():
        def __init__(self, name) -> None:
            self.name = name
            self.arg_list:List[instr_type.i_argument] = []

    def __init__(self, name) -> None:
        self.name = name
        self._arg_format = instr_type.arg_format()
        self.inst_list:List[instr_type.instruction]=[]

    def arg_format_append(self, name, offset):
        self._arg_format.add_arg(name, offset)
    
    def add_new_instr(self, name):
        last_inst = instr_type.instruction(name)
        self.inst_list.append(last_inst)
        self.last_inst = last_inst

    def add_arg_to_last_inst(self, name, offset):
        arg_type = self._arg_format.get_arg_type(offset)
        arg =  self.i_argument(name, arg_type)
        self.last_inst.arg_list.append(arg)

    def add_opt_type_to_last_arg(self, opt_type):
        last_arg = self.last_inst.arg_list[-1]
        last_arg.opt_type = opt_type


def parse_instruction_name(l:str):
    return l.translate(str.maketrans('', '', '— '))

def parse_instruction_argument(arg:bs4Element.Tag):
    name = arg['href']
    if(name.find('AMDGPUModifierSyntax') != -1):
        return name.split("amdgpu-synid-",1)[1] 
    return name[:name.index('.')]

def parse_argument_type(a_type:bs4Element.Tag)->str:
    name = parse_instruction_argument(a_type)
    if(name in ['gfx10_m', 'gfx10_dst']):
        return name
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
        
        last_i_type:instr_type = None
        last_offset = 0
        next_elem_is_arg_type = False

        litblock = i_p.find("pre")
        
        for i in litblock:
            if(type(i) == bs4Element.Tag):
                if(i.name == 'a'):
                    if(next_elem_is_arg_type):
                        a_type = parse_argument_type(i)
                        if(a_type != ''):
                            curent_pack.add_opt_type_to_last_arg(a_type)
                        next_elem_is_arg_type=False
                    else:
                        arg_name = parse_instruction_argument(i)
                        curent_pack.add_arg_to_last_inst(arg_name, last_offset)
                elif(i.name == 'strong'):
                    curent_pack.arg_format_append(i.string, last_offset)
            else: #string
                lines = re.split('\\n', i)
                if(len(lines) > 1):
                    for line in lines:
                        p = parse_instruction_name(line)
                        if(p != ''):
                            last_i_type = instr_type(p)
                            curent_pack.add_new_instr(last_i_type)
                            last_offset = len(line)
                    continue
                elif(lines[0]==':'):
                    next_elem_is_arg_type = True
            
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
            cur_str_head = f', {i}:reg_block'
            cur_str_body = f'\t\tself.{i} = {i} \n'

        else:
            no_mods = False
            cur_str_head = f', **MODIFIERS'
            cur_str_body = f'\t\tself.{i} = \' \'.join({i}.values()) \n'
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

        str_join = "{','.join(map(str, args_l))}"

        str_body_end = f"\t\treturn f\"{str_label} {str_join} {str_mod}\" \n"
        str_body_l.append(str_body_end)
    else:
        str_mod = '{self.MODIFIERS}'
        str_body_l.append(f"\t\treturn f\"{str_label} {str_mod}\" \n")

    return [header, init_head, *init_body_l, str_header, *str_body_l]

def create_py_code(i_types:List[instr_type], outfile):
    
    outfile.writelines(f"from python.codegen.gpu_instruct import * \n\n")

    for i_t in i_types:
        group_name = i_t.name
        G_class_name = f'{group_name}_base'
        outfile.writelines(create_G_class_code(i_t, G_class_name))



def main():
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--infile', nargs='?', type=argparse.FileType('r'),
                           default=sys.stdin)

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
    
    create_py_code(instr_types, outfile)


    
if __name__ == '__main__':
    main()