# -*- coding: utf-8 -*-
# %%
# +
# #!/usr/bin/env python
# -

# %%
import os
import graph_tool.all as gt
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt

# %%
def assign_node_strength(g):
    vp_out = g.new_vp("double")
    g.vp["out_strength"] = vp_out

    vp_in = g.new_vp("double")
    g.vp["in_strength"] = vp_in

    vp_all = g.new_vp("double")
    g.vp["all_strength"] = vp_all

    vp_outratio = g.new_vp("double")
    g.vp["outratio"] = vp_outratio

    for v in g.vertices():
        out_list = [g.ep.weight[e] for e in v.out_edges()]
        vp_out[v] = np.array(out_list).sum()

        in_list = [g.ep.weight[e] for e in v.in_edges()]
        vp_in[v] = np.array(in_list).sum()
        vp_all[v] = vp_out[v] + vp_in[v]
        vp_outratio[v] = vp_out[v]/vp_all[v] if vp_all[v] > 0 else -1


# %%
def assign_edge_gravity(g):
    ep_gravity = g.new_ep("double")
    g.ep['gravity'] = ep_gravity
    for e in g.edges():
        ep_gravity[e] = g.ep.weight[e] / (g.vp.out_strength[e.source()]* g.vp.in_strength[e.target()])


# %%
def read_graph(year):
    g = gt.Graph()
    vp_name = g.new_vp("string")
    g.vp['name'] = vp_name
    ep_weight = g.new_ep("double")
    g.ep['weight'] = ep_weight
    vp_is_donor = g.new_vp("int")
    g.vp['is_donor'] = vp_is_donor

    with open(f"data/year_{year}_s.json") as f:
        j = json.load(f)

    nodes = set()
    for x in j:
        nodes.add(x['donor'])
        nodes.add(x['recip'])
    node_names = sorted(list(nodes))

    g.add_vertex(len(node_names))
    for i,name in enumerate(node_names):
        g.vp.name[g.vertex(i)] = name

    for x in j:
        d = node_names.index(x['donor'])
        r = node_names.index(x['recip'])
        e = g.add_edge(g.vertex(d), g.vertex(r))
        g.ep.weight[e] = x['weight']
        g.vp.is_donor[g.vertex(d)] = 1

    assign_node_strength(g)
    assign_edge_gravity(g)
    return g

# %%
def init_blocks(g, ep_weight):
    state = gt.NestedBlockState(g, state_args=dict(deg_corr=False,recs=[ep_weight],rec_types=["real-normal"]))
    return state

# %%
def eqm_blocks(g, ep_weight, wait=1000):
    state = init_blocks(g, ep_weight)
    gt.mcmc_equilibrate(state, wait=wait, mcmc_args=dict(niter=10))
    return state


# %%
def sample_blocks(g, wait, num_samp, lognorm=True, gravity=True):
    y = g.ep.weight.copy()
    if gravity:
        y = g.ep.gravity.copy()
    if lognorm:
        y.a = np.log(y.a)
    state = eqm_blocks(g, y, wait)
    s_samples = []
    def collect_partitions(s):
        s_samples.append(s.copy())
    gt.mcmc_equilibrate(state, force_niter=num_samp+1, mcmc_args=dict(niter=10), callback=collect_partitions)
    return state, s_samples


# %%
def draw_nested_with_node_names(state, output1, output2):
    b = state.g.new_vertex_property("int")
    b.a = state.get_bs()[0]
    pos,t,t_pos = state.draw(vertex_text = b, vertex_text_position=0, vertex_font_size = 12)
    x_pos = [pos[v][0] for v in g.vertices()]
    y_pos = [pos[v][1] for v in g.vertices()]
    angle = g.new_vertex_property("double")
    angle.a = np.arctan2(y_pos,x_pos)
    state.draw(
        pos=pos,
        edge_color=gt.prop_to_size(g.ep.gravity, power=1, log=True),
        ecmap=(matplotlib.cm.inferno, .6),
        eorder=g.ep.weight,
        edge_pen_width=gt.prop_to_size(g.ep.weight, 1, 4, power=1, log=True),
        edge_gradient=[],
        vertex_text = g.vp.name,
        vertex_text_position=0,
        vertex_text_offset=[0,0],
        vertex_font_size=20,
        vertex_text_rotation = angle,
        output_size=(2000,2000),
        hide = 20,
        output=output1
    )
    state.draw(
        pos=pos,
        edge_color=gt.prop_to_size(g.ep.gravity, power=1, log=True),
        ecmap=(matplotlib.cm.inferno, .6),
        eorder=g.ep.weight,
        edge_pen_width=gt.prop_to_size(g.ep.weight, 1, 4, power=1, log=True),
        edge_gradient=[],
        output_size=(2000,2000),
        hide = 20,
        output=output2
    )


# %%
def stats_num_blocks(s_samples):
    num_blocks = []
    for s in s_samples:
        num_blocks.append(s.get_levels()[0].get_nonempty_B())
    num_blocks = np.array(num_blocks)
    return (num_blocks.mean(), num_blocks.std())


# %%
def find_significant_blocks(g, s_samples, thr_frac=0.95):
    bs = []
    for s in s_samples:
        bs.append(s.get_bs())
    pmode = gt.PartitionModeState(bs, nested=True, converge=True)
    pv = pmode.get_marginal(g)
    g.vertex_properties['pv'] = pv

    thr_cnt = len(s_samples) * thr_frac
    vp_aff = g.new_vp("int")
    g.vp['aff'] = vp_aff
    for v in g.vertices():
        pvec = np.array(pv[v])
        #print(f'{pvec =}')
        vp_aff[v] = pvec.argmax() if pvec.max() >= thr_cnt else -1


# %%
def print_block_member(g, jsonfile, texfile, aff, name):
    aff_dict = {}
    for v in g.vertices():
        if aff[v] not in aff_dict:
            aff_dict[aff[v]] = [name[v]]
        else:
            aff_dict[aff[v]].append(name[v])
    with open(jsonfile, 'w') as f:
        json.dump(aff_dict, f, indent=2)

    aff_list_sorted = sorted(aff_dict.items(), key=lambda x: len(x[1]), reverse=True)
    with open(texfile, 'w') as f:
        for cnt, val in enumerate(aff_list_sorted ):
            f.write(f"{cnt} & ")
            for node_name in val[1]:
                f.write(f"{node_name}, ")
            f.write("\\\n")


# %%
def print_block_nodewise(g, outfile, aff, name):
    node_dict = {}
    for v in g.vertices():
        node_dict[g.vertex_index[v]] = {"name":name[v], "block":aff[v]}
    with open(outfile, 'w') as f:
        json.dump(node_dict, f, indent=2)


# %%
def print_block_tree(g, outfile, aff, name):
    with open(outfile, 'w') as f:
        cnt_node = np.zeros(int(aff.a.max())+1)
        for v in g.vertices():
            if aff[v] >= 0:
                cnt_node[aff[v]] += 1
                f.write(f'1:{aff[v]+1}:{int(cnt_node[aff[v]])} 1.0 "{name[v]}" {g.vertex_index[v]+1}\n')


# %%
def print_block_tree_extend(g, outfile, aff, name):
    with open(outfile, 'w') as f:
        cnt_node = np.zeros(int(aff.a.max())+1)
        for v in g.vertices():
            if aff[v] >= 0:
                cnt_node[aff[v]] += 1
                role_label = 1 if g.vp.out_strength[v] > g.vp.in_strength[v] else 2
                f.write(f'1:{role_label}:{aff[v]+1}:{int(cnt_node[aff[v]])}:1 1.0 "{name[v]}" {g.vertex_index[v]+1}\n')


# %%
def print_significant_blocks(g, year, out_dir = "output"):
    print_block_nodewise(g, f"{out_dir}/block_nodewise_{year}.json", g.vp.aff, g.vp.name)
    print_block_member(g, f"{out_dir}/block_member_{year}.json", f"{out_dir}/block_member_{year}.tex", g.vp.aff, g.vp.name)
    print_block_tree(g, f"{out_dir}/block_tree_{year}.tree", g.vp.aff, g.vp.name)
    print_block_tree_extend(g, f"{out_dir}/block_tree_extend_{year}.tree", g.vp.aff, g.vp.name)


# %%
def read_aff(g, year, out_dir = "output"):
    with open(f"{out_dir}/block_nodewise_{year}.json", 'r') as f:
        js = json.load(f)
        vp_aff = g.new_vp("int")
        g.vp['aff'] = vp_aff
        for i in js:
            v = g.vertex(i)
            vp_aff[v] = js[i]["block"]


# %%
def set_plot_para():
    matplotlib.rcdefaults()
    plt.rcParams['font.family'] ='sans-serif'
    plt.rcParams["figure.subplot.left"] = 0.22 # should be common in an article
    plt.rcParams["figure.subplot.right"] = 0.90 # should be common in an article # This option for legends to include
    plt.rcParams["figure.subplot.bottom"] = 0.20 # should be common in an article
    plt.rcParams["figure.subplot.top"] = 0.95 # should be common in an article
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['axes.labelsize'] = 30
    plt.rcParams['font.size'] = 22
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams["legend.loc"] = "best"         # 凡例の位置、"best"でいい感じのところ
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.fontsize"] = 15


# %%
def plot_timeseries(ts_stats, years, output):
    set_plot_para()
    years = np.array(years)
    mean = np.array([x[0] for x in ts_stats])
    std = np.array([x[1] for x in ts_stats])
    plt.ylim(3,11)
    plt.yticks([4, 6, 8, 10])
    plt.xlim(1965, 2014)
    plt.xticks([1970,1990,2010])
    plt.xlabel("year")
    plt.ylabel("number of blocks", fontsize = 27)
    plt.errorbar(years, mean, yerr=std)
    plt.savefig(output)


# %%
def print_block_member_single_sample(g, year, state, name, out_dir = "output"):
    levels = state.get_levels()
    s = levels[0]
    aff = s.get_blocks()

    aff_dict = {}
    for v in g.vertices():
        if aff[v] not in aff_dict:
            aff_dict[aff[v]] = [name[v]]
        else:
            aff_dict[aff[v]].append(name[v])
    aff_list_sorted = sorted(aff_dict.items(), key=lambda x: len(x[1]), reverse=True)
    with open(f"{out_dir}/block_member_{year}_single.tex", 'w') as f:
        for cnt, val in enumerate(aff_list_sorted):
            f.write(f"{cnt} & ")
            for node_name in val[1]:
                f.write(f"{node_name}, ")
            f.write("\\\n")


# %%
if __name__ == '__main__':
    
    wait = 10
    num_samp = 100

    start = 1970
    end = 2013
    step = 1

    threshold = 0.95
    lognorm = True
    gravity = True

    np.random.seed(777)
    gt.seed_rng(777)

    ts_stats = []
    years = range(start, end+1, step)

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    for year in years:
        print(year)
        g = read_graph(year)
        state, s_samples = sample_blocks(g, wait, num_samp, lognorm, gravity)

        draw_nested_with_node_names(state, f"{out_dir}/nested_blocks_{year}_labeled.png",  f"{out_dir}/nested_blocks_{year}.png") # visualization of the last-run result
        print_block_member_single_sample(g, year, state, g.vp.name)

        ts_stats.append(stats_num_blocks(s_samples))
        find_significant_blocks(g, s_samples, threshold)
        print_significant_blocks(g, year, out_dir)

    plot_timeseries(ts_stats, years, f"{out_dir}/num_blocks_samp{num_samp}.pdf")
