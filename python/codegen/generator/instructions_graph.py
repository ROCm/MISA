from typing import Dict, List
import networkx as nx

from bokeh.io import output_file, show

from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                            MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool)
from bokeh.models.graphs import NodesOnly, StaticLayoutProvider
from bokeh.models.renderers import GraphRenderer
from bokeh.models.tools import BoxZoomTool, WheelZoomTool

from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx

from python.codegen.generator_instructions import flow_control_base,  reg_allocator_base
from python.codegen.gpu_data_types import *
from python.codegen.gpu_instruct import inst_base

class instruction_graph():
    
    class Node():
        def __init__(self, name:str, id:int, is_var:bool = False) -> None:
            self.name = name
            self.id = id
            self.is_var = is_var
            self.connections_out:List[instruction_graph.Node] = []
            self.connections_in:List[instruction_graph.Node] = []
        
        def get_networkx_node(self):
            return (self.id, {'name':self.name, 'color':'blue'})

    def __init__(self, instructions_list:List[inst_base]) -> None:
        self.instructions_list = instructions_list

        reg_to_edge_translation = Dict
        self.reg_translation = reg_to_edge_translation
        self.vert_list:List[instruction_graph.Node] = []
        self._build_graph()


    def add_new_var_node(self, name) -> Node:
        new_var = instruction_graph.Node(name, self.max_sub_var_id, True)
        self.max_sub_var_id += 1
        return new_var

    def _build_graph(self):
        i_list = self.instructions_list
        node_id = 0
        self.max_sub_var_id = 0
        var_list = []

        vert_list = []
        current_var_to_sub:List[instruction_graph.Node] = []
        

        
        for i in i_list:
            if issubclass(type(i),(reg_allocator_base, flow_control_base)):
                continue

            cur_vert = instruction_graph.Node(i.label, node_id, False)
            vert_list.append(cur_vert)
            node_id += 1
            src_regs = i.get_srs_regs()

            for src in src_regs:
                if(src):
                    if(type(src) in [regAbs, regNeg, regVar, VCC_reg, EXEC_reg]):
                        #pre defined HW values
                        src = src.base_reg
                        try:
                            index = var_list.index(src)
                        except ValueError:
                            assert(False)
                            index = len(var_list)
                            var_list.append(src)
                            cur_sub_var = self.add_new_var_node(src.label)
                            current_var_to_sub[index] = cur_sub_var

                        cur_sub_var = current_var_to_sub[index]
                        
                    elif(type(src) in [reg_block]):
                        assert(False)
                        continue
                    else:
                        cur_sub_var = self.add_new_var_node(src)

                    cur_vert.connections_in.append(cur_sub_var)
                    cur_sub_var.connections_out.append(cur_vert)

            dst_regs = i.get_dst_regs()
            for dst in dst_regs:
                if(dst):
                    dst = dst.base_reg
                    cur_sub_var = self.add_new_var_node(dst.label)

                    cur_vert.connections_out.append(cur_sub_var)
                    try:
                        index = var_list.index(dst)
                    except ValueError:
                        index = len(var_list)
                        var_list.append(dst)
                        current_var_to_sub.append(cur_sub_var)

                    current_var_to_sub[index] = cur_sub_var
                    
        self.vert_list = vert_list


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
                    G.add_edge(id, dst_vert.id)
        
        self.set_Ypos_BFS(G, 0)
        return G

    def bokeh_show(self, G:nx.DiGraph):
        plot = Plot(width=2000, height=1000,
            x_range=Range1d(-50.0,110), y_range=Range1d(0,1000))
        plot.title.text = "Graph Interaction Demonstration"

        
        node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("name", "@name")])
        plot.add_tools(node_hover_tool, TapTool(), BoxSelectTool(), BoxZoomTool(), WheelZoomTool())
        pos=nx.get_node_attributes(G,'pos')
        graph_renderer = from_networkx(G, nx.fruchterman_reingold_layout)
        
        fixed_layout_provider = StaticLayoutProvider(graph_layout=pos)  
        graph_renderer.layout_provider = fixed_layout_provider

        #graph_renderer = GraphRenderer()

        graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="color")
        graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
        graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])


        graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = NodesOnly()
        from holoviews.element.graphs import layout_nodes
        import holoviews as hv

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


        plot.renderers.append(graph_renderer)

        output_file("interactive_graphs.html")
        show(plot)
