from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import inspect
from typing import Dict, List

from python.codegen.gpu_data_types import *
#from python.codegen.gpu_data_types import abs, neg

class inst_mode(Enum):
    allocation: auto
    emmition: auto

class instruction_type(Enum):
    SMEM = 'SMEM'
    VMEM = 'VMEM'
    VOP1 = 'VOP1'
    REGALLOC = 'REGA'
    REGDEALLOC = 'REGD'
    BLOCKALLOC = 'BLCKA'
    BLOCKSPLIT = 'BLCKS'
    #BLOCKDEALLOC = 'BLCKD'
    FLOW_CONTROL = 'FC'

class inst_base(ABC):
    __slots__ = ['label', 'inst_type', 'trace']
    def __init__(self, inst_type:instruction_type, label:str) -> None:
        self.inst_type:instruction_type = inst_type
        self.label = label
        self.trace = ''

    @abstractmethod
    def __str__(self) -> str:
        return f'{self.label}'

    def emit(self) -> str:
        return self.__str__()

    def emit_trace(self) -> str:
        return self.trace

class inst_caller_base(ABC):
    __slots__ = ['il']
    def __init__(self, insturction_list:List[inst_base]) -> None:
        self.il = insturction_list

    def ic_pb(self, inst):
        self.il.append(inst)
        st = inspect.stack()
        return inst

from python.codegen.generator_instructions import flow_control_caller, reg_allocator_caller
class gpu_instructions_caller_base(reg_allocator_caller, flow_control_caller):
    def __init__(self, insturction_list) -> None:
        super().__init__(insturction_list)


import igraph
class instruction_graph():
    
    class Node():
        def __init__(self, name:str, id:int, is_var:bool = False) -> None:
            self.name = name
            self.id = id
            self.is_var = is_var
            self.connections_out:List[instruction_graph.Node] = []
            self.connections_in:List[instruction_graph.Node] = []

    def __init__(self, instructions_list:List[inst_base]) -> None:
        self.instructions_list = instructions_list

        reg_to_edge_translation = Dict
        self.reg_translation = reg_to_edge_translation
        self.vert_list:List[instruction_graph.Node] = []
        self.sub_var_list = []
        self._build_graph()

    def _build_graph(self):
        i_list = self.instructions_list
        node_id = 0
        sub_var_id = 0
        var_list = []
        sub_var_list = []
        vert_list = []
        current_var_to_sub:List[instruction_graph.Node] = []
        dst_atr = ['DST', 'DST0','DST1', 'DST2']
        src_atr = ['SRC', 'SRC0', 'SRC1', 'SRC2', 'SRC3']
        for i in i_list:            
            cur_vert = instruction_graph.Node(i.label, node_id, False)
            vert_list.append(cur_vert)
            node_id += 1

            for src_a in src_atr:
                src = getattr(i, src_a, None)
                if(src):
                    if(type(src) in [reg_block, regAbs, regNeg, regVar, VCC_reg, EXEC_reg]):
                        #pre defined HW values
                        src = src.base_reg
                        try:
                            index = var_list.index(src)
                        except ValueError:
                            index = len(var_list)
                            var_list.append(src)
                            cur_sub_var = instruction_graph.Node(src.label, sub_var_id, True)
                            sub_var_list.append(cur_sub_var)
                            sub_var_id += 1
                            current_var_to_sub.append(cur_sub_var)

                        cur_sub_var = current_var_to_sub[index]
                        cur_vert.connections_in.append(cur_sub_var)
                        cur_sub_var.connections_out.append(cur_vert)
                    else:
                        cur_sub_var = instruction_graph.Node(src, sub_var_id, True)
                        sub_var_list.append(cur_sub_var)
                        sub_var_id += 1
                        cur_vert.connections_in.append(cur_sub_var)
                        cur_sub_var.connections_out.append(cur_vert)

            for dst_a in dst_atr:
                dst = getattr(i, dst_a, None)
                if(dst):
                    dst = dst.base_reg
                    cur_sub_var = instruction_graph.Node(dst.label, sub_var_id, True)
                    sub_var_list.append(cur_sub_var)
                    sub_var_id += 1
                    cur_vert.connections_out.append(cur_sub_var)
                    try:
                        index = var_list.index(dst)
                        current_var_to_sub[index] = cur_sub_var
                    except ValueError:
                        index = len(var_list)
                        var_list.append(dst)
                        current_var_to_sub.append(cur_sub_var)
                    
        self.vert_list = vert_list
        self.sub_var_list = sub_var_list


    def build_plot(self):
        vert_list = self.vert_list
        g = igraph.Graph()
        g.add_vertices(len(vert_list))
        
        for i in vert_list:
            id = i.id
            for dst_var in i.connections_out:
                for dst_vert in dst_var.connections_out:
                    g.add_edge(id, dst_vert.id)
        layout = g.layout("kk")
        igraph.plot(g, layout=layout)


class instruction_ctrl():
    __slots__ = ['instructions_list', 'code_str']

    def __init__(self) -> None:
        instructions_list:List[inst_base] = []
        self.instructions_list = instructions_list

        self.code_str = []
    
    def execute_all(self):
        self._emmit_all(self.code_str.append)
    
    def plot_the_graph(self):
        instruction_graph(self.instructions_list).build_plot()

    def _emmit_all(self, emmiter):
        e = emmiter
        for i in self.instructions_list:
            e(f'{i}')
    
    def _emmit_created_code(self, emmiter):
        e = emmiter
        for i in self.code_str:
            e(i)

    def _emmit_range(self, emmiter, strt:int, end:int):
        e = emmiter
        i_list = self.instructions_list
        for i in i_list:
            e(f'{i}')
