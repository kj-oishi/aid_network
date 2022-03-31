# %%
# +
# #!/usr/bin/env python
# -

# %%
import graph_tool.all as gt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import aid_graph
import block_interaction
from cycler import cycler


# %%
def make_partitions(g):
    bg = block_interaction.make_block_graph(g)
    partitions = []
    for bv in bg.vertices():
        member_names = []
        for m in bg.vp.member[bv]:
            member_names.append(m["name"])
        partitions.append(member_names)
    partitions = sorted(partitions, key = lambda x: len(x), reverse = True)
    return partitions


# %%
def read_regions():
    df = pd.read_csv('data/actors.csv', sep=',')
    actor_dict = {row[0]:row[1] for index, row in df.iterrows()}
    regions = list(df['region'].unique())
    return regions, actor_dict


# %%
def read_LDC(year):
    ldc = pd.read_csv('data/LDC.csv', sep=',')
    ldcy = ldc[ldc["year"] == year]
    LDC = set(ldcy["LDC"])
    return LDC


# %%
def plot_regions(g, partitions, regions, year, actor_dict):
    p_freq_list =[]
    for p_id in range(len(partitions)):
        freq_dict = {x:0 for x in regions}
        for a in partitions[p_id]:
            freq_dict[actor_dict[a]] += 1
        #freq_list = [freq_dict[r]/len(partitions[p_id]) for r in regions]
        freq_list = [freq_dict[r] for r in regions]
        p_freq_list.append(freq_list)
    dataset = pd.DataFrame(p_freq_list, columns = regions, index = range(len(partitions)))

    com_reg = np.array(dataset).T
    width = 0.8
    B = len(partitions)
    plt.xticks(list(range(B)))
    plt.xlabel("block ID")
    plt.ylabel("number of nodes")
    #plt.ylim(0,150)
    labels = np.arange(B)
    plt.bar(labels, com_reg[0], label = dataset.columns[0])
    for b in range(1,len(dataset.columns)):
        plt.bar(labels, com_reg[b], width, bottom = com_reg[:b].sum(axis = 0), label = dataset.columns[b])
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"output/regions_{year}.pdf", bbox_inches='tight')
    plt.clf()


# %%
def plot_LDCs(g, partitions, LDC, year):

    B = len(partitions)
    ldc_num = np.zeros(B)
    ldc_frac = np.zeros(B)
    for i, p in enumerate(partitions):
        for a in p:
            if a in LDC:
                ldc_num[i] += 1
        ldc_frac[i] = ldc_num[i]/len(p) if len(p)>0 else 0
    #ldc_frac = ldc_num/ldc_num.sum()

    width = 0.8
    labels = np.arange(B)

    plt.xticks(list(range(B)))
    plt.xlabel("Block ID")
    plt.ylabel("number of LDC", fontsize = 27)
    plt.bar(labels, ldc_num, label = "number of LDC")
    plt.savefig(f"output/ldc_num_{year}.pdf")
    plt.clf()

    plt.xticks(list(range(B)))
    plt.xlabel("Block ID")
    plt.ylim(0, 1)
    plt.ylabel("fraction of LDC", fontsize = 27)
    plt.bar(labels, ldc_frac, label = "fraction of LDC")
    plt.savefig(f"output/ldc_frac_{year}.pdf")
    plt.clf()


# %%
regions, actor_dict = read_regions()

aid_graph.set_plot_para()
plt.rcParams['axes.prop_cycle']  = cycler(color=['#0078B0', '#FF7D26', '#0B9E3B', '#DE2A2D','#9369B9',
                                                 '#8F564D','#E878BE','#7E7E7E','#BDBA3B','#00BDCD', "black"])
start = 1970
end = 2013
step = 10

for year in range(start, end+1, step):
    LDC = read_LDC(year)
    g = aid_graph.read_graph(year)
    aid_graph.read_aff(g, year)
    partitions = make_partitions(g)
    plot_LDCs(g, partitions, LDC, year)
    plot_regions(g, partitions, regions, year, actor_dict)
