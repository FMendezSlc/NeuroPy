import numpy as np
import neo
import matplotlib.pyplot as plt
import pandas as pd
from quantities import Hz, s, ms
import elephant as elp
import seaborn as sns
import networkx as nx
import os
from scipy import stats


def build_block(data_file):
    """(plaintextfile_path) -> neo.core.block.Block
    Thie function reads a plain text file (data_file) with the data exported (per waveform) from Plexon Offline Sorter after sorting and returns a neo.Block with spiketrains ordered in any number of 10 minute segments.

    For practicality and Plexon management of channels names, units and channels have been ignored in the block structure."""

    raw_data = pd.read_csv(data_file, sep=',', header=0, usecols=[0, 1, 2])
    ord_times = raw_data.groupby(['Channel Name', 'Unit'])['Timestamp']
    new_block = neo.Block()
    chx = neo.ChannelIndex(index=None, name='MEA_60')
    new_block.channel_indexes.append(chx)
    # Next line will not work properly if last spike happens
    # exactly at the end of the recording
    num_segments = range(int(raw_data['Timestamp'].max() // 600 + 1))
    for ind in num_segments:
        seg = neo.Segment(name='segment {}'.format(ind), index=ind)
        new_block.segments.append(seg)

    for name, group in ord_times:
        time_stamps = ord_times.get_group(name).values
        inter = 600  # Number of seconds in 10 minutes
        first_seg = neo.SpikeTrain(time_stamps[time_stamps < inter], units='sec', t_stop=inter)
        new_block.segments[0].spiketrains.append(first_seg)
        new_unit = neo.Unit(name=name)
        sptrs = [first_seg]
        for seg in num_segments[1:-1]:
            seg_train = neo.SpikeTrain(time_stamps[(time_stamps > seg * inter) & (time_stamps < ((seg + 1) * inter))],
                                       units='sec', t_start=(seg * inter), t_stop=((seg + 1) * inter))
            new_block.segments[seg].spiketrains.append(seg_train)
            sptrs.append(seg_train)
        last_seg = neo.SpikeTrain(time_stamps[time_stamps > (num_segments[-1] * inter)], units='sec',
                                  t_start=(num_segments[-1]) * inter, t_stop=((num_segments[-1] + 1) * inter))
        new_block.segments[num_segments[-1]].spiketrains.append(last_seg)
        sptrs.append(last_seg)
        new_unit.spiketrains = sptrs
        chx.units.append(new_unit)
    return new_block


def sttc(spiketrain_1, spiketrain_2, dt=0.01 * s):
    ''' Calculates the Spike Time Tiling Coefficient as described in (Cutts & Eglen, 2014) following Cutts' implementation in C (https://github.com/CCutts/Detecting_pairwise_correlations_in_spike_trains/blob/master/spike_time_tiling_coefficient.c)'''

    def run_P(spiketrain_1, spiketrain_2, N1, N2, dt):
        '''Check every spike in train 1 to see if there's a spike in train 2 within dt.'''
        Nab = 0
        j = 0
        for i in range(N1):
            while (j < N2):  # don't need to search all j each iteration
                if np.abs(spiketrain_1[i] - spiketrain_2[j]) <= dt:
                    Nab = Nab + 1
                    break
                elif spiketrain_2[j] > spiketrain_1[i]:
                    break
                else:
                    j = j + 1
        return Nab

    def run_T(spiketrain, N, dt):
        ''' Calculate the proportion of the total recording time 'tiled' by sipkes.'''
        time_A = 2 * N * dt  # maxium possible time

        if N == 1:  # for just one spike in train
            if spiketrain[0] - spiketrain.t_start < dt:
                time_A = time_A - dt + sipketrain[0] - spiketrain.t_start
            elif spiketrain[0] + dt > spiketrain.t_stop:
                time_A = time_A - dt - spiketrain[0] + spiketrain.t_stop

        else:  # if more than one spike in train
            i = 0
            while (i < (N - 1)):
                diff = spiketrain[i + 1] - spiketrain[i]

                if diff < (2 * dt):  # subtract overlap
                    time_A = time_A - 2 * dt + diff
                i += 1
                # check if spikes are within dt of the start and/or end
                # if so just need to subract overlap of first and/or last spike
            if (spiketrain[0] - spiketrain.t_start) < dt:
                time_A = time_A + spiketrain[0] - dt - spiketrain.t_start

            if (spiketrain.t_stop - spiketrain[N - 1]) < dt:
                time_A = time_A - spiketrain[-1] - dt + spiketrain.t_stop

            T = (time_A / (spiketrain.t_stop - spiketrain.t_start)).item()
            return T

    N1 = len(spiketrain_1)
    N2 = len(spiketrain_2)

    if N1 == 0 or N2 == 0:
        index = np.nan
    else:
        TA = run_T(spiketrain_1, N1, dt)
        TB = run_T(spiketrain_2, N2, dt)
        PA = run_P(spiketrain_1, spiketrain_2, N1, N2, dt)
        PA = PA / N1
        PB = run_P(spiketrain_2, spiketrain_1, N2, N1, dt)
        PB = PB / N2
        index = 0.5 * (PA - TB) / (1 - TB * PA) + 0.5 * (PB - TA) / (1 - TA * PB)
    return index


def sttc_g(sttc_file):
    ''' [csv -> nx.Graph]
    Builds an empty nx.Graph and fills it with edges from a csv read as a pandas DataFrame. '''

    sttc_df = pd.read_csv(sttc_file).drop(columns=['Unnamed: 0'])
    sttc_df['STTC_weight'] = sttc_df['STTC_weight']
    nodes = [ii for ii in sttc_df['Node_2'].unique()]
    nodes.append(sttc_df['Node_1'].iloc[-1])
    sttc_graph = nx.Graph()
    sttc_graph.add_nodes_from(nodes)
    median = np.median(abs(sttc_df['STTC_weight']))
    mad = np.median(abs(sttc_df['STTC_weight'] - median)) / 0.6745
    thr = mad * 2
    for i, nrow in sttc_df.iterrows():
        if nrow[4] < (median + thr) and nrow[4] > (median - thr):
            pass
        else:
            sttc_graph.add_edge(nrow[2], nrow[3], weight=round(nrow[4], 2))
    print(thr)

    return sttc_graph


os.chdir('/Users/felipeantoniomendezsalcido/Desktop/MEAs/spike_times')

list_files
data_file = list_files[20]
data_file
name_keys = data_file.split(sep='_')
data_block = build_block(data_file)
len(data_block.list_units)
# STTC calculations, remeber you did it under the diagonal
# That is (N * N-1) % 2
#%%
STTC_dic = {'Date': [], 'Gen_type': [], 'Sex': [], 'Node_1': [], 'Node_2': [], 'STTC_weight': []}
#names_right = {'Node_1' : [], 'Node_2' : []}
for jj in range(len(data_block.list_units)):
    unit1 = data_block.list_units[jj]
    for kk in range(jj):
        unit2 = data_block.list_units[kk]
        STTC_dic['Date'].append(name_keys[3])
        STTC_dic['Gen_type'].append(name_keys[1])
        STTC_dic['Sex'].append(name_keys[2])
        unit1_name = 'Ch{},U{}'.format(unit1.name[0][-2:], unit1.name[1])
        STTC_dic['Node_1'].append(unit1_name)
        unit2_name = 'Ch{},U{}'.format(unit2.name[0][-2:], unit2.name[1])
        STTC_dic['Node_2'].append(unit2_name)
        STTC_dic['STTC_weight'].append(sttc(unit1.spiketrains[0], unit2.spiketrains[0]))
STTC_df = pd.DataFrame(STTC_dic)
STTC_df.to_csv(data_file.split('.')[0] + 'STTC')
#%%

#%%
mat_size = len(data_block.list_units)
sttc_mat = np.empty((mat_size, mat_size))
sttc_mat[:] = np.nan
count = 0
for ii in range(mat_size):
    for jj in range(ii):
        sttc_mat[ii, jj] = STTC_df['STTC_weight'][count]
        count += 1
#%%

# Plot heatmap of adjacency matrix
#%%
names_ag = [ii for ii in STTC_df['Node_2'].unique()]
STTC_df['Node_1'].iloc[-1]
names_ag.append(STTC_df['Node_1'].iloc[-1])
sttc_round = np.around(sttc_mat, decimals=2)

mat_fig = plt.figure(figsize=(14, 14))
sns.heatmap(sttc_round, center=0, cmap='seismic', cbar_kws={
            'shrink': .8}, yticklabels=names_ag, xticklabels=names_ag, square=True)
plt.title(data_file)
plt.ylabel('Units')
plt.xlabel('Units')
plt.show()
mat_fig.savefig(data_file.split('.')[0] + '_fig.svg')
#%%

os.chdir('/Users/felipeantoniomendezsalcido/Desktop/MEAs/STTC_data')
os.listdir()
sttc_list = os.listdir()[:-2]
sttc_list
pd.read_csv(sttc_list[0])
float(nx.info(sttc_graph).split('\n')[4].split(' ')[-1])
round(sum([ii for ii in nx.get_edge_attributes(sttc_graph, 'weight').values() if ii > 0]), 2)

# Populate a DataFrame with pooled data of general stats from all graphs
#%%
graph_dic = {'Date': [], 'Gen_type': [], 'Sex': [],
             'Nx_Degree': [], 'Edges': [], 'Nx_Density': [], 'Total_Weight': [], 'Av_degree': [], 'Pos_w': [], 'Neg_w': [], 'Av_weight': [], 'Av_node_strength': []}

for ii in sttc_list:
    name_keys = ii.split(sep='_')
    graph_dic['Date'].append(name_keys[3])
    graph_dic['Gen_type'].append(name_keys[1])
    graph_dic['Sex'].append(name_keys[2])
    sttc_df = pd.read_csv(ii).drop(columns=['Unnamed: 0'])
    sttc_df['STTC_weight'] = sttc_df['STTC_weight'].round(2)
    sttc_graph = sttc_g(ii)

    graph_dic['Nx_Degree'].append(int(nx.info(sttc_graph).split('\n')[2].split(' ')[-1]))
    graph_dic['Edges'].append(int(nx.info(sttc_graph).split('\n')[3].split(' ')[-1]))
    graph_dic['Nx_Density'].append(nx.density(sttc_graph))
    graph_dic['Av_degree'].append(float(nx.info(sttc_graph).split('\n')[4].split(' ')[-1]))
    graph_dic['Total_Weight'].append(
        round(sum(nx.get_edge_attributes(sttc_graph, 'weight').values()), 2))
    graph_dic['Pos_w'].append(
        round(sum([ii for ii in nx.get_edge_attributes(sttc_graph, 'weight').values() if ii > 0]), 2))
    graph_dic['Neg_w'].append(
        round(sum([ii for ii in nx.get_edge_attributes(sttc_graph, 'weight').values() if ii < 0]), 2))
    graph_dic['Av_weight'].append(round(sum(nx.get_edge_attributes(
        sttc_graph, 'weight').values()), 2) / int(nx.info(sttc_graph).split('\n')[3].split(' ')[-1]))
    node_strength = dict(nx.degree(sttc_graph, weight='weight'))
    graph_dic['Av_node_strength'].append(round(np.mean([ii for ii in node_strength.values()]), 2))

nx_df = pd.DataFrame(graph_dic)
new_ord = ['Date', 'Sex', 'Gen_type', 'Nx_Degree', 'Nx_Density', 'Edges',
           'Av_degree', 'Av_node_strength', 'Total_Weight', 'Av_weight', 'Pos_w', 'Neg_w']
nx_df = nx_df[new_ord]
#%%

nx_df
aa = dict(nx.degree(sttc_graph, weight='weight'))
aa = round(np.mean([ii for ii in aa.values()]), 2)
aa
#%%
# Nx Density
with sns.axes_style('whitegrid'):
    plt.figure(figsize=(12, 12))
    sns.stripplot(x='Gen_type', y='Nx_Density', data=nx_df, jitter=0.05,
                  edgecolor='gray', linewidth=1, size=6.5, palette='deep', order=['WT', 'KO'])
    sns.boxplot(x='Gen_type', y='Nx_Density', data=nx_df, order=[
                'WT', 'KO'], saturation=1, width=0.3, palette='deep', linewidth=2.5)
    sns.despine()
    plt.xlabel('Genotipo', fontsize=16)
    plt.ylabel('Densidad de Conexión', fontsize=16)
    plt.tick_params(labelsize=14)
#%%
ko_density = nx_df[nx_df['Gen_type'] == 'KO']['Nx_Density']
wt_density = nx_df[nx_df['Gen_type'] == 'WT']['Nx_Density']
stats.ttest_ind(ko_density, wt_density)
stats.normaltest(wt_density)

#%%
# Av Degree
with sns.axes_style('whitegrid'):
    plt.figure(figsize=(12, 12))
    sns.stripplot(x='Gen_type', y='Av_degree', data=nx_df, jitter=0.05,
                  edgecolor='gray', linewidth=1, size=6.5, palette='deep', order=['WT', 'KO'])
    sns.boxplot(x='Gen_type', y='Av_degree', data=nx_df, order=[
                'WT', 'KO'], saturation=1, width=0.3, palette='deep', linewidth=2.5)
    sns.despine()
    plt.xlabel('Genotipo', fontsize=16)
    plt.ylabel('Grado Promedio por Nodo', fontsize=16)
    plt.tick_params(labelsize=14)
#%%
#%%
# Total Weight
with sns.axes_style('whitegrid'):
    plt.figure(figsize=(12, 12))
    sns.stripplot(x='Gen_type', y='Total_Weight', data=nx_df, jitter=0.05,
                  edgecolor='gray', linewidth=1, size=6.5, palette='deep', order=['WT', 'KO'])
    sns.boxplot(x='Gen_type', y='Total_Weight', data=nx_df, order=[
                'WT', 'KO'], saturation=1, width=0.3, palette='deep', linewidth=2.5)
    sns.despine()
    plt.xlabel('Genotipo', fontsize=16)
    plt.ylabel('Fuerza Total', fontsize=16)
    plt.tick_params(labelsize=14)
#%%
#%%
# Positive Weights
with sns.axes_style('whitegrid'):
    plt.figure(figsize=(12, 12))
    sns.stripplot(x='Gen_type', y='Pos_w', data=nx_df, jitter=0.05,
                  edgecolor='gray', linewidth=1, size=6.5, palette='deep', order=['WT', 'KO'])
    sns.boxplot(x='Gen_type', y='Pos_w', data=nx_df, order=[
                'WT', 'KO'], saturation=1, width=0.3, palette='deep', linewidth=2.5)
    sns.despine()
    plt.xlabel('Genotipo', fontsize=16)
    plt.ylabel('Promedio Fuerzas Positivas', fontsize=16)
    plt.tick_params(labelsize=14)
#%%
#%%
# Av Weights
with sns.axes_style('whitegrid'):
    plt.figure(figsize=(12, 12))
    sns.stripplot(x='Gen_type', y='Av_weight', data=nx_df, jitter=0.05,
                  edgecolor='gray', linewidth=1, size=6.5, palette='deep', order=['WT', 'KO'])
    sns.boxplot(x='Gen_type', y='Av_weight', data=nx_df, order=[
                'WT', 'KO'], saturation=1, width=0.3, palette='deep', linewidth=2.5)
    sns.despine()
    plt.xlabel('Genotipo', fontsize=16)
    plt.ylabel('STTC Promedio', fontsize=16)
    plt.tick_params(labelsize=14)
#%%
#%%
# Av Node Strength
with sns.axes_style('whitegrid'):
    plt.figure(figsize=(12, 12))
    sns.stripplot(x='Gen_type', y='Av_node_strength', data=nx_df, jitter=0.05,
                  edgecolor='gray', linewidth=1, size=6.5, palette='deep', order=['WT', 'KO'])
    sns.boxplot(x='Gen_type', y='Av_node_strength', data=nx_df, order=[
                'WT', 'KO'], saturation=1, width=0.3, palette='deep', linewidth=2.5)
    sns.despine()
    plt.xlabel('Genotipo', fontsize=16)
    plt.ylabel('Fuerza por nodo promedio', fontsize=16)
    plt.tick_params(labelsize=14)
#%%
ko_nstrength = nx_df[nx_df['Gen_type'] == 'KO']['Av_node_strength']
wt_nstrength = nx_df[nx_df['Gen_type'] == 'WT']['Av_node_strength']
stats.ttest_ind(ko_nstrength, wt_nstrength)
stats.normaltest(wt_nstrength)
stats.normaltest(ko_nstrength)
ko_nstrength.max()

tests_wt = sttc_list[11]
ko_g = sttc_g(sttc_list[19])
nx.info(ko_g)
wt_g = sttc_g(sttc_list[19])

abs(sttc_df['STTC_weight']).median() + np.median(abs(sttc_df['STTC_weight'].round(2) -
                                                     np.median(sttc_df['STTC_weight'].round(2)))) / 0.674 * 2
sttc_df[sttc_df['STTC_weight'] >= 0.02]['STTC_weight'].sum()
sttc_df = pd.read_csv(tests_wt).drop(columns=['Unnamed: 0'])
sttc_df['STTC_weight'] = sttc_df['STTC_weight'].round(2)
sttc_graph = nx.Graph()
sttc_df['STTC_weight'].std()
mad
post_th = sttc_df[sttc_df['STTC_weight'] >= 0]['STTC_weight'].quantile(0.75)
post_th
neg_th = sttc_df[sttc_df['STTC_weight'] <= 0]['STTC_weight'].quantile(0.25)
neg_th
for i, nrow in sttc_df.iterrows():
    if nrow[4] < post_th and nrow[4] > neg_th:
        # if nrow[4] == 0:
        pass
    else:
        sttc_graph.add_edge(nrow[2], nrow[3], weight=nrow[4])

nx.info(sttc_graph)
nx.draw_circular(sttc_graph)
round(sum(nx.get_edge_attributes(sttc_graph, 'weight').values()), 2)
round(sum(nx.get_edge_attributes(sttc_graph, 'weight').values()), 2) / \
    int(nx.info(sttc_graph).split('\n')[3].split(' ')[-1])

nx_df['Av_weight'].mad()
help(nx_df.mad())
nx.info(wt_g)

wt_df = pd.read_csv(sttc_list[19]).drop(columns=['Unnamed: 0'])
wt_mean = wt_df['STTC_weight'].mean()
wt_df['STTC_weight'].median()
wt_std = wt_df['STTC_weight'].std()
wt_mean + (2 * wt_std)
wt_std
wt_mean
degs = {}
for n in wt_g.nodes():
    deg = wt_g.degree(n)
    if deg not in degs:
        degs[deg] = 0
    degs[deg] += 1
items = sorted(degs . items())
#%%
fig = plt . figure()
ax = fig . add_subplot(111)
ax.hist([v for (k, v) in items])
#plt.xscale( 'linear')
#plt.yscale( 'linear')
plt. title("Degree Distribution ")
#%%
[wt_g.degree(ii) for ii in wt_g.nodes()]

degs.keys()

[v for k, v in ii[2].items() for ii in list(wt_g.edges(data=True))]
[v for k, v in ii[2].items() for ii in list(wt_g.edges(data=True))]
weights = []
for ii in list(wt_g.edges(data=True)):
    weights.append([v for k, v in ii[2].items()])

len(weights)
np.mean(weights)
np.std(weights)
np.median(weights)
plt.hist(weights, bins=10)
weights


[k for (k, v) in items]
[v for (k, v) in items]
#%%
plt.figure()
a = np.logspace(0, 100)
plt.plot(a, a)
plt.xscale('linear')
plt.yscale('log')
#%%

# Plotting graph
#%%
c_map = []

for node in sttc_graph:
    dg = ['Ch23', 'Ch34', 'Ch14', 'Ch35', 'Ch26', 'Ch12', 'Ch13', 'Ch24', 'Ch25', 'Ch16', 'Ch17']
    if node.split(',')[0] in dg:
        c_map.append('g')
    else:
        c_map.append('b')
#%%
#%%
plt.figure(figsize=(14, 14))
nx.draw_circular(sttc_graph, with_labels=False, node_color=c_map,
                 node_size=350, alpha=.9, linewidths=.5, style='dotted')
plt.show()
#%%

#%%
plt.figure(figsize=(16, 16))
plt.hist([ii[1] for ii in list(nx.degree(sttc_graph))])
#%%
[ii for ii in list(nx.nodes(sttc_g))]
list(nx.degree(sttc_graph))

for node in sttc_graph:
    print(node)

nx.info(sttc_graph).split('\n')
