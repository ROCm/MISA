from typing import Dict, List, Type
from bokeh.models import CustomJS
from bokeh.models import renderers
from bokeh.models.layouts import Row
from bokeh.models.widgets.groups import CheckboxGroup
import networkx as nx

from bokeh.io import output_file, show

from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                            MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool)
from bokeh.models.graphs import NodesOnly, StaticLayoutProvider
from bokeh.models.renderers import GraphRenderer
from bokeh.models.tools import BoxZoomTool, WheelZoomTool

from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx

from python.codegen.generator_instructions import flow_control_base, instr_label_base,  reg_allocator_base
from python.codegen.gpu_data_types import *
from python.codegen.gpu_instruct import inst_base, instruction_type

class instruction_graph():
    
    class Node():
        def __init__(self, name:str, id:int, is_var:bool = False, color='blue', line_dash=[]) -> None:
            self.name = name
            self.id = id
            self.is_var = is_var
            self.connections_out:List[instruction_graph.Node] = []
            self.connections_in:List[instruction_graph.Node] = []

            self.position_dep_before:List[instruction_graph.Node] = []
            self.position_dep_after:List[instruction_graph.Node] = []

            self.color = color
            self.line_dash = line_dash
        
        def get_networkx_node(self):
            return (self.id, {'name':self.name, 'color':self.color, 'line_dash':self.line_dash})

    def __init__(self, instructions_list:List[inst_base]) -> None:
        self.instructions_list = instructions_list

        reg_to_edge_translation = Dict
        self.reg_translation = reg_to_edge_translation
        self.vert_list:List[instruction_graph.Node] = []
        self._build_graph()


    def _add_new_var_node(self, name) -> Node:
        new_var = instruction_graph.Node(name, self._max_sub_var_id, is_var=True)
        self._max_sub_var_id += 1
        return new_var
    
    def _add_new_vert_node(self, name, color='blue', line_dash=[]):
        cur_vert = instruction_graph.Node(name, self._max_node_id, is_var=False, color=color, line_dash=line_dash)
        self.vert_list.append(cur_vert)
        self._max_node_id += 1
        return cur_vert
    
    def _bound_vert_by_pos(self, vert:Node, vertexes_before:List[Node]):
        vert.position_dep_before.extend(vertexes_before)
        for v in vertexes_before:
            v.position_dep_after.append(vert)



    def _build_graph(self):
        i_list = self.instructions_list
        # node -> vertex -> instruction
        self._max_node_id = 0
        # sub variable -> edge -> register
        self._max_sub_var_id = 0
        
        empty_Node = instruction_graph.Node('None', -1, True)
        #
        baseReg_list = []
        baseView_list = []

        self.vert_list = []
        base_subNodes = List[instruction_graph.Node]
        baseSubNodes_list:List[base_subNodes] = []

        position_constraints_enabled = True
#        if(position_constraints_enabled):
        last_label_pos = 0
        last_smem_pos = 0
        first_vertex_from_label_block = 0
        def get_gfx10_instructions_sets():
            i_t = instruction_type

            def is_scalar(inst:inst_base):
                if (inst.inst_type in [i_t.SOP1, i_t.SOP2, i_t.SOPC, i_t.SOPK, i_t.SOPP, i_t.VOP3P, i_t.SMEM]):
                    return True
                return False
            def is_vector(inst:inst_base):
                if (inst.inst_type in [i_t.VOPC, i_t.VOP1, i_t.VOP2, i_t.VOP3, i_t.VOP3P, i_t.DPP8, i_t.DPP16, i_t.VINTRP, i_t.VMEM]):
                    return True
                return False
            def is_memory(inst:inst_base):
                if (inst.inst_type in [i_t.VMEM, i_t.SMEM, i_t.MTBUF, i_t.MUBUF, i_t.DS, i_t.FLAT]):
                    return True
                if(inst.label in ['s_waitcnt']):
                    return True
                return False
            def is_program_flow(inst:inst_base):
                if (inst.inst_type in [i_t.SOPP, i_t.FLOW_CONTROL]):
                    #if(not (inst.label in ['s_nop', 's_waitcnt'])):
                    return True
                if(inst.inst_type is i_t.SOPK and inst.label in 
                    ['s_waitcnt_expcnt','s_waitcnt_lgkmcnt','s_waitcnt_vmcnt','s_waitcnt_vscnt']):
                    return True
                return False
            def is_exec_dependent(inst:inst_base):
                if( is_vector(inst) or (inst.label in [i_t.EXP, i_t.MTBUF, i_t.MUBUF, i_t.DS, i_t.FLAT]) ):
                    return True
                return False

            return { 
                'scalar' : is_scalar,
                'vector' : is_vector,
                'memory' : is_memory,
                'program_flow' : is_program_flow,
                'exec_dep' : is_exec_dependent
            }

        def get_instruction_color(inst:inst_base, inst_set:Dict):
            t = inst
            i_t = instruction_type
            
            if (inst_set['memory'](t)):
                return 'red'
            if(inst_set['program_flow'](t)):
                return 'green'
            if(inst_set['vector'](t)):
                return 'blue'
            if (inst_set['scalar'](t)):
                return 'yellow'
            if(t in [i_t.HW_REG_INIT]):
                return 'black'
            
        
        def get_instruction_dash(inst:inst_base, inst_set:Dict):
            if(inst_set['scalar'](inst)):
                return [3, 3]
            return []
        
        is_gfx10_instruct_set = get_gfx10_instructions_sets()
        for i in i_list:
            #pseudo instractions ignored
            if issubclass(type(i),(reg_allocator_base, flow_control_base)):
                continue
            
            if issubclass(type(i),(instr_label_base)) and not position_constraints_enabled:
                continue
            
            cur_vert = self._add_new_vert_node(
                i.label, 
                color=get_instruction_color(i, is_gfx10_instruct_set), 
                line_dash=get_instruction_dash(i, is_gfx10_instruct_set)
            )

            # Two labels define a code segment.
            # Instructions declared inside a segment cannot move beyond the labels 
            #   defining this segment.
            if(position_constraints_enabled):
                cur_vert_pos = len(self.vert_list) - 1
                # Bound new label to instructions from last segment.
                if is_gfx10_instruct_set['program_flow'](i):
                    vert_range = slice(last_label_pos, cur_vert_pos)
                    vertexes_before = self.vert_list[vert_range]
                    self._bound_vert_by_pos(cur_vert, vertexes_before)
                    last_label_pos = cur_vert_pos
                else:
                    # Boud new instruction to last label.
                    vert_range = slice(last_label_pos, last_label_pos+1)
                    vertexes_before = self.vert_list[vert_range]
                    self._bound_vert_by_pos(cur_vert, vertexes_before)

                # Bound current mem_op tp the previous
                # flat instructions seq in src oreder
                if is_gfx10_instruct_set['memory'](i):
                    vert_range = slice(last_smem_pos, last_smem_pos+1)
                    vertexes_before = self.vert_list[vert_range]
                    self._bound_vert_by_pos(cur_vert, vertexes_before)
                    last_smem_pos =  cur_vert_pos


            src_regs = i.get_srs_regs()

            if is_gfx10_instruct_set['exec_dep'](i):
                src_regs.append(EXEC_reg())

            for src in src_regs:
                if(src):
                    if(type(src) in [regAbs, regNeg, regVar, VCC_reg, EXEC_reg]):
                        #pre defined HW values
                        src_view:tuple = src.get_view_range()
                        src_base = src.base_reg
                        
                        try:
                            index = baseReg_list.index(src_base)
                        except ValueError:
                            assert(False)

                        cur_sub_var = baseSubNodes_list[index][src_view[0]:src_view[1]]
                        
                    elif(type(src) in [reg_block]):
                        assert(False)
                        continue
                    else:
                        cur_sub_var = [self._add_new_var_node(src)]

                    cur_vert.connections_in.extend(cur_sub_var)
                    #map(lambda x:x.connections_out.append(cur_vert), cur_sub_var)
                    [x.connections_out.append(cur_vert) for x in cur_sub_var]

            dst_regs = i.get_dst_regs()
            for dst in dst_regs:
                if(dst):
                    dst_view:tuple = dst.get_view_range()
                    dst_base:reg_block = dst.base_reg
                    
                    cur_sub_var = list(
                        map(lambda x:self._add_new_var_node(dst_base.label), 
                            range(dst_view[0], dst_view[1]))
                    )

                    cur_vert.connections_out.extend(cur_sub_var)
                    try:
                        index = baseReg_list.index(dst_base)
                    except ValueError:
                        index = len(baseReg_list)
                        baseReg_list.append(dst_base)
                        empty_base_subNodes = [empty_Node] * dst_base.dwords
                        baseSubNodes_list.append(empty_base_subNodes)
                    
                    cur_baseSubNodes = baseSubNodes_list[index]
                    
                    if(position_constraints_enabled):
                        # Write in register only afther read.
                        cur_w_nodes = cur_baseSubNodes[dst_view[0]:dst_view[1]]
                        for cur_node in cur_w_nodes:
                            clean_connections = [
                                *filter(lambda x: not x is cur_vert, cur_node.connections_out)
                            ]
                            self._bound_vert_by_pos(cur_vert, clean_connections)

                    #update var after write
                    for i, j in zip(range(dst_view[0],dst_view[1]), range(dst_view[1]-dst_view[0])):
                        cur_baseSubNodes[i] = cur_sub_var[j]
                    
                    
        
        #remove from hw_reg_init self directed edge
        hw_reg_init = self.vert_list[0]
        hw_reg_init.position_dep_before.pop(0)
        hw_reg_init.position_dep_after.pop(0)



    #def build_plot(self):
    #    vert_list = self.vert_list
    #    g = igraph.Graph()
    #    g.add_vertices(len(vert_list))
    #    root = g.vs[0]
    #    root["color"] = "black"
    #    for i in vert_list:
    #        id = i.id
    #        for dst_var in i.connections_out:
    #            for dst_vert in dst_var.connections_out:
    #                g.add_edge(id, dst_vert.id)
    #    layout = g.layout("kk")
    #    igraph.plot(g, layout=layout)

    def set_Ypos_BFS(self, G:nx.DiGraph, start_id):
        # Mark all the vertices as not visited
        nodes_cnt = len(G.nodes)
        node_succs = G.succ

        # Create a queue for BFS
        queue = []
        node_dep_cnt = list(map(lambda x: len(G.pred[x]), range(nodes_cnt)))
        # Mark the source node as
        # visited and enqueue it
        for i in range(len(node_dep_cnt)):
            if(node_dep_cnt[i] == 0):
                queue.append(i)
                G.nodes[i]['Y_pos'] = 1

        G.nodes[start_id]['Y_pos'] = 0
        max_X = 100
        prew_Y = 0
        cur_X = 0
        cur_X_step = max_X / 2

        while queue:
            cur = queue.pop(0)
            cur_Ypos = G.nodes[cur]['Y_pos']
            if(prew_Y != cur_Ypos):
                prew_Y = cur_Ypos
                cur_X_step = max_X / ((sum(1 for i in queue if G.nodes[i]['Y_pos'] == cur_Ypos)) + 2)
                cur_X = 0

            cur_X += cur_X_step
            G.nodes[cur]['pos'] = (cur_X*2, cur_Ypos*300)

            for i in node_succs[cur]:
                node_dep_cnt[i] -= 1
                if node_dep_cnt[i] == 0:
                    queue.append(i)
                    G.nodes[i]['Y_pos'] = cur_Ypos + 1

    def get_graph(self):
        
        vert_list = self.vert_list
        G = nx.DiGraph()
        networkx_nodes = map(lambda x:x.get_networkx_node(), vert_list)

        G.add_nodes_from(networkx_nodes)
                
        #root = 
        G.nodes[0]["color"] = "black"
        for i in vert_list:
            id = i.id
            for dst_var in i.connections_out:
                for dst_vert in dst_var.connections_out:
                    G.add_edge(id, dst_vert.id, data_dep=True)
            
            for after_vert in i.position_dep_after:
                G.add_edge(id, after_vert.id, data_dep=False)
        
        self.set_Ypos_BFS(G, 0)
        return G

    def bokeh_show(self, G:nx.DiGraph):
        #plot = Plot(width=2000, height=1000,
        #    x_range=Range1d(-50.0,110), y_range=Range1d(0,1000))
        
        from bokeh.plotting import figure
        plot = figure(title="Instructions graph", x_axis_label="x", y_axis_label="iRank", width=2000, height=1000)

        
        node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("name", "@name")])
        plot.add_tools(node_hover_tool, TapTool(), BoxSelectTool())
        pos=nx.get_node_attributes(G,'pos')
        graph_renderer = from_networkx(G, nx.fruchterman_reingold_layout)
        
        fixed_layout_provider = StaticLayoutProvider(graph_layout=pos)  
        graph_renderer.layout_provider = fixed_layout_provider

        #graph_renderer = GraphRenderer()

        graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="color", line_dash="line_dash", fill_alpha=0.5)
        graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
        graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

        graph_renderer.node_renderer.glyph.properties_with_values()

        graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

        graph_renderer.selection_policy = NodesAndLinkedEdges()

        graph_renderer.inspection_policy = NodesOnly()
        
        #from holoviews.element.graphs import layout_nodes
        #import holoviews as hv
        #cities_graph_fruchterman = layout_nodes(graph_renderer, layout=nx.layout.fruchterman_reingold_layout)

        #labels = hv.Labels(cities_graph_fruchterman.nodes, ['x', 'y'], ["index"])
        
        #graph.node_renderer.data_source.add(node_indices, 'index')
        #graph.node_renderer.data_source.add(Spectral8, 'color')
        #graph.node_renderer.data_source.add(["circle", "square"] * 4, 'marker')
        #graph.node_renderer.glyph = Scatter(size=20, fill_color='color', marker="marker")
        
        def plan_A():
            from holoviews.element.graphs import layout_nodes
            import holoviews as hv
            import hvplot.networkx as hvnx
            import hvplot as hvplot
            #cities_graph_fruchterman = layout_nodes(graph_renderer, layout=nx.layout.fruchterman_reingold_layout)
            #
            #labels = hv.Labels(cities_graph_fruchterman.nodes, ['x', 'y'], ["index"])
            #
            #labels.opts(xoffset=-0.05, yoffset=0.04, text_font_size='8pt',)
            #
            #cities_graph_fruchterman * labels

            
            ##pos = nx.layout.spring_layout(G)
        
            node_sizes = [30 for i in range(len(G))]
            M = G.number_of_edges()
            edge_colors = range(2, M + 2)

            G2 = G.to_directed()
            nodes = hvnx.draw_networkx_nodes(G2, pos, node_size=node_sizes, arrowstyle='->',
                                        arrows=True, edge_color=edge_colors, aspect='equal', arrowhead_length=0.005, 
                                        edge_cmap='Blues', edge_width=1, colorbar=True, directed=True)

            edges = hvnx.draw(G2, pos, node_size=node_sizes, arrowstyle='->',
                                        arrows=True, edge_color=edge_colors, aspect='equal', arrowhead_length=0.005, 
                                        edge_cmap='Blues', edge_width=1, colorbar=True, directed=True)
            

            #r = nodes * edges
            #graph = hv.Graph((edges, nodes, ), )
            #graph.opts(directed=True)

            hv.render(edges)
            
            hvplot.save(edges, 'test.html')

        #plan_A()

        def plan_B():
            import numpy as np
            import holoviews as hv
            from holoviews import opts
            import panel as pn
            from bokeh.resources import INLINE
            hv.extension('bokeh')
            opts.defaults(opts.Graph(width=400, height=400))
            #N = 8
            #node_indices = np.arange(N)
            #source = np.zeros(N)
            #target = node_indices

            #= hv.Graph(((source, target),))
            #position = nx.spring_layout(G, scale=2)
            #nx.draw(G,position)
            #padding = dict(x=(-1.1, 1.1), y=(-1.1, 1.1))
            #hv.Graph.from_networkx
            edges = G.edges
            edges = [1, 2, 3]
            nodes = G.nodes
            nodes = [0,0,0]
            simple_graph = hv.Graph(((edges, nodes),))
            #simple_graph = graph_renderer
            
            panel_object = pn.pane.HoloViews(simple_graph)
            pn.pane.HoloViews(simple_graph).save('test2', embed=True, resources=INLINE)
        
        #plan_B()

        def paln_c():
            def choose_node_outline_colors(nodes_clicked):
                outline_colors = []
                for node in G.nodes():
                    if str(node) in nodes_clicked:
                        outline_colors.append('pink')
                    else:
                        outline_colors.append('black')
                return outline_colors


            def update_node_highlight(event):
                nodes_clicked_ints = source.selected.indices
                nodes_clicked = list(map(str, nodes_clicked_ints))
                source.data['line_color'] = choose_node_outline_colors(nodes_clicked)

            source = graph.node_renderer.data_source
            source.data['line_color'] = choose_node_outline_colors('1')
            TOOLTIPS = [
                ("Index", "@index"),
            ]
            plot.add_tools(HoverTool(tooltips=TOOLTIPS), TapTool(), BoxSelectTool())
            taptool = plot.select(type=TapTool)

            plot.on_event(Tap, update_node_highlight)
            curdoc().add_root(plot)

        edge_dep_t = graph_renderer.edge_renderer.data_source.data['data_dep']
        edge_start = graph_renderer.edge_renderer.data_source.data['start']
        edge_end = graph_renderer.edge_renderer.data_source.data['end']

        data_dep = {'start': [], 'end': [], 'data_dep' : []}
        pos_dep = {'start': [], 'end': [], 'data_dep' : []}

        for i in range(len(edge_start)):
            if(edge_dep_t[i] == True):
                data_dep['start'].append(edge_start[i])
                data_dep['end'].append(edge_end[i])
                data_dep['data_dep'].append(edge_dep_t[i])
            else:
                pos_dep['start'].append(edge_start[i])
                pos_dep['end'].append(edge_end[i])
                pos_dep['data_dep'].append(edge_dep_t[i])

        plot.renderers.append(graph_renderer)

        checkbox = CheckboxGroup(labels=["data dep", "Position dep"],
                                active=[0, 1], width=100)
        
        update_edges_args = dict(
            edge_renderer=graph_renderer.edge_renderer,
            checkbox=checkbox, 
            data_dep=data_dep,
            pos_dep=pos_dep
        )

        update_edges_str = """
            var new_data_edge = {'start': [], 'end': [], 'data_dep' : []};
            if (checkbox.active.includes(0)){
                new_data_edge['start'] = new_data_edge['start'].concat(data_dep['start']);
                new_data_edge['end'] = new_data_edge['end'].concat(data_dep['end']);
                new_data_edge['data_dep'] = new_data_edge['data_dep'].concat(data_dep['data_dep']);
            }
            if (checkbox.active.includes(1)){
                new_data_edge['start'] = new_data_edge['start'].concat(pos_dep['start']);
                new_data_edge['end'] = new_data_edge['end'].concat(pos_dep['end']);
                new_data_edge['data_dep'] = new_data_edge['data_dep'].concat(pos_dep['data_dep']);
            }

            edge_renderer.data_source.data = new_data_edge;
            edge_renderer.data_source.change.emit();
            console.log('checkbox: active=' + checkbox.active, checkbox.toString())
        """

        update_edges2 = CustomJS(
            args=update_edges_args,
            code=update_edges_str,
        )
        checkbox.js_on_change('active', update_edges2)
        #graph_renderer.visible=False

        output_file("interactive_graphs.html")

        show(Row(plot,checkbox))
