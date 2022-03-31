# -*- coding: utf-8 -*-
# %%
# +
# #!/usr/bin/env python
# -

# %%
import graph_tool.all as gt
import numpy as np
import aid_graph
import matplotlib


# %%
def stats_block_flow(g, aff, B):
    flow = np.zeros((B,B))
    for v1 in g.vertices():
        if(aff[v1] >= 0):
            for e in v1.out_edges():
                v2 = e.target()
                if(aff[v2] >= 0):
                    flow[aff[v1],aff[v2]] += g.ep.weight[e]
    return flow


# %%
def assign_edge_weight(g, bg, B):
    ep_weight = bg.new_ep("double")
    bg.ep['weight'] = ep_weight
    flow = stats_block_flow(g, g.vp.aff, B)
    for bi in range(B):
        for bj in range(B):
            if(flow[bi,bj]>0):
                e = bg.add_edge(bg.vertex(bi), bg.vertex(bj))
                ep_weight[e] = flow[bi,bj]


# %%
def assign_node_size(g, aff, bg):
    vp_size = bg.new_vp("int")
    bg.vp["size"] = vp_size
    vp_size.a = 0
    for v in g.vertices():
        if(aff[v] >= 0):
            vp_size[bg.vertex(aff[v])] += 1


# %%
def assign_blocknode_stats(g, bg, B):
    bg.vp['in_degree_ave'] = bg.new_vp("double")
    bg.vp['in_degree_std'] = bg.new_vp("double")
    bg.vp['out_degree_ave'] = bg.new_vp("double")
    bg.vp['out_degree_std'] = bg.new_vp("double")
    insum = np.zeros(B)
    insqr = np.zeros(B)
    outsum = np.zeros(B)
    outsqr = np.zeros(B)

    for v in g.vertices():
        if(g.vp.aff[v] >= 0):
            insum[g.vp.aff[v]] += v.in_degree()
            insqr[g.vp.aff[v]] += v.in_degree() * v.in_degree()
            outsum[g.vp.aff[v]] += v.out_degree()
            outsqr[g.vp.aff[v]] += v.out_degree() * v.out_degree()

    for b in range(B):
        bv = bg.vertex(b)
        size = bg.vp.size[bv]
        if(size > 0):
            mean = bg.vp.in_degree_ave[bv] = insum[b]/size
            bg.vp.in_degree_std[bv] = np.sqrt(insqr[b]/size - mean * mean)
            mean = bg.vp.out_degree_ave[bv] = outsum[b]/size
            bg.vp.out_degree_std[bv] = np.sqrt(outsqr[b]/size - mean * mean)
        else:
            bg.vp.in_degree_ave[bv] = bg.vp.in_degree_std[bv] = bg.vp.out_degree_ave[bv] = bg.vp.out_degree_std[bv] = 0


# %%
def assign_member(g, aff, bg):
    vp_member = bg.new_vp("object")
    bg.vp["member"] = vp_member
    vp_top = bg.new_vp("string")
    bg.vp["top"] = vp_top
    for bv in bg.vertices():
        vp_member[bv] = []
    for v in g.vertices():
        if(aff[v] >= 0):
            vp_member[bg.vertex(aff[v])].append({"nodeID": g.vertex_index[v],"name":g.vp.name[v],"all_strength":g.vp.all_strength[v]})

    num_tops = 3
    for bv in bg.vertices():
        vp_member[bv] = sorted(vp_member[bv], key = lambda x: x["all_strength"], reverse=True)
        vp_top[bv] = vp_member[bv][0]["name"] if bg.vp.size[bv] > 0 else ""
        for cnt in range(1, num_tops):
            if cnt < bg.vp.size[bv]:
                vp_top[bv] += ", " + vp_member[bv][cnt]["name"]
            else:
                break


# %%
def remove_vacant_node(g, size):
    for v in g.vertices():
        if size[v] == 0:
            g.remove_vertex(v)


# %%
def make_block_graph(g):
    bg = gt.Graph()
    B = int(g.vp.aff.a.max())+1
    bg.add_vertex(B)
    assign_edge_weight(g, bg, B)
    assign_node_size(g, g.vp.aff, bg)
    assign_blocknode_stats(g, bg, B)
    assign_member(g, g.vp.aff, bg)
    aid_graph.assign_node_strength(bg)
    remove_vacant_node(bg, bg.vp.size)
    return bg


# %%
def draw_block_graph(bg, output):
    gt.graph_draw(bg,
                  output = output,
                  output_size=(1000,1000),
                  vertex_fill_color=gt.prop_to_size(bg.vp.outratio,log=False),
                  vcmap=(matplotlib.cm.inferno, .5),
                  #vertex_text=bg.vertex_index,
                  vertex_text=bg.vp.top,
                  vertex_font_size = 15,
                  vertex_text_position = 0,
                  vertex_text_offset = [-0.2,0],
                  vertex_size=gt.prop_to_size(bg.vp.all_strength, mi=10, ma=100,log=False),
                  edge_pen_width=gt.prop_to_size(bg.ep.weight,mi=1,ma=20,log=False)
                 )


# %%
def draw_block_graph_SBM(g, output):
    y = g.ep.weight.copy()
    y.a = np.log(y.a)
    state = gt.NestedBlockState(g, state_args=dict(deg_corr=False,recs=[y],rec_types=["real-normal"]))
    gt.mcmc_equilibrate(state, wait=100, mcmc_args=dict(niter=10))
    state.draw(output = output,
               output_size=(1000,1000),
               vertex_fill_color=gt.prop_to_size(g.vp.outratio,log=False),
               vertex_color = gt.prop_to_size(g.vp.outratio,log=False),
               vcmap=(matplotlib.cm.inferno, .5),
               vertex_text=g.vp.top,
               vertex_text_position = 0,
               vertex_font_size = 15,
               vertex_size=gt.prop_to_size(g.vp.all_strength, mi=10, ma=100,log=False),
               edge_pen_width=gt.prop_to_size(g.ep.weight,mi=1,ma=20,log=False),
               hide = 20
              )


# %%
if __name__ == '__main__':
    start = 1970
    end = 2013
    step = 10

    np.random.seed(777)
    gt.seed_rng(777)

    for year in range(start, end+1, step):
        print(year)
        np.random.seed(7)
        gt.seed_rng(7)
        g = aid_graph.read_graph(year)
        aid_graph.read_aff(g, year)
        bg = make_block_graph(g)
        #draw_block_graph(bg, f"output/block_interaction_{year}.png")
        draw_block_graph_SBM(bg, f'output/block_interaction_ext_{year}.png')
