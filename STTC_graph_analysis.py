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
from scipy import polyfit


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
        first_seg = neo.SpikeTrain(
            time_stamps[time_stamps < inter], units='sec', t_start=0,  t_stop=inter)
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


def sttc(spiketrain_1, spiketrain_2, dt=0.005 * s):
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


def sttc_g(sttc_file, threshold=2, all_nodes=True, exclude_shadows=True):
    ''' [csv -> nx.Graph]
    Builds an empty nx.Graph and fills it with edges from a csv containing STTC values read as a pandas DataFrame. '''

    sttc_df = pd.read_csv(sttc_file).drop(columns=['Unnamed: 0'])
    sttc_df['STTC_weight'] = sttc_df['STTC_weight']
    nodes = [ii for ii in sttc_df['Node_2'].unique()]
    nodes.append(sttc_df['Node_1'].iloc[-1])
    sttc_graph = nx.Graph()
    if all_nodes == True:
        sttc_graph.add_nodes_from(nodes)
    median = np.median(sttc_df['STTC_weight'])
    mad = np.median(abs(sttc_df['STTC_weight'] - median)) / 0.6745
    thr = mad * threshold
    for i, nrow in sttc_df.iterrows():
        if nrow[4] < (median + thr) and nrow[4] > (median - thr):
            pass
        else:
            sttc_graph.add_edge(nrow[2], nrow[3], weight=round(nrow[4], 2))
    print(thr)

    return sttc_graph


os.chdir('/Users/felipeantoniomendezsalcido/Desktop/MEAs/STTC_bootstrapped')
spiketrains = os.listdir()[1:]
spiketrains[9]
sttc_mat = pd.read_csv(spiketrains[10]).drop(columns=['Unnamed: 0'])
sttc_mat
boots_mat = pd.read_csv(spiketrains[9]).drop(columns=['Unnamed: 0'])
boots_mat

#%%
real_vs_boots = plt.figure(figsize=(20, 20))
sns.distplot(sttc_mat['STTC_weight'])
sns.distplot(boots_mat['STTCboot_mean'])
plt.title('Real vs Bootstrapped STTC values')
plt.xlabel('STTC')
plt.ylabel('Normalized Density')
plt.text(.2, 1, 'bootstrapped mean +/- 3 std = 0.0121\nreal STTC mean + mad = 0.0183', fontsize=14)
real_vs_boots.savefig('real_vs_boots.svg')
#%%

np.std(boots_mat['STTCboot_mean'], ddof=1)
np.mean(boots_mat['STTCboot_mean']) + 3 * (np.std(boots_mat['STTCboot_mean'], ddof=1))

np.median(abs(sttc_mat['STTC_weight'] - median)) / 0.6745
median = np.median(sttc_mat['STTC_weight'])
np.median(sttc_mat['STTC_weight']) + median
test_block.list_units[26].spiketrains[0].size
test_train1 = test_block.list_units[2].spiketrains[0]
test_train2 = test_block.list_units[7].spiketrains[0]

a_test = segmented_train(test_train1)
b_test = segmented_train(test_train2)
a_test[-1]
sttc(a_test, b_test)
sttc(test_train1, test_train2)


def segmented_train(spiketrain, t_start=(150 * s), t_stop=(450 * s)):
    for index, item in enumerate(spiketrain):
        if item >= t_start:
            if item < t_stop:
                start_index = index
                break
            else:
                print('Empty Train')
                break
    for index, item in enumerate(spiketrain):
        if item > t_stop:
            stop_index = index - 1
            break
    segmented_train = neo.SpikeTrain(
        spiketrain[start_index:stop_index], t_start=t_start, t_stop=t_stop)
    return segmented_train


def surrogated_corr(spiketrain1, spiketrain2, boot_samples=5):
    boots_sttc = []
    for ii in range(boot_samples):
        shift = np.random.random(1)
        surrogated = elp.spike_train_surrogates.dither_spike_train(spiketrain2, shift=shift * s)[0]
        boots_sttc.append(sttc(spiketrain1, surrogated))
    return boots_sttc


segmented_train(test_train1, t_start=t_start, t_stop=t_stop)
surr = np.random.choice(surrogated_corr(test_train1, test_train2), size=(1000, 10))
surr
debug_block = build_block(spiketrains[0])
segmented_train(debug_block.list_units[26].spiketrains[0], t_start=(120 * s), t_stop=(240 * s))


for index, item in enumerate(debug_block.list_units[26].spiketrains[0]):
    print(index, item)

surr
np.mean(np.mean(surr, axis=1))
np.mean(np.std(surr, axis=1))
boot_mean + 3 * boot_std

spiketrains

np.random.random(5)
sttc(test_train1, test_train2)

# Create bootstrapped adjacency matrices
spiketrains
files_count = 1
for ii in spiketrains:
    print('Working on {}'.format(ii))
    new_block = build_block(ii)

    STTCboots_dic = {'Node_1': [], 'Node_2': [], 'STTCboot_mean': [], 'STTCboot_std': []}

    for jj in range(len(new_block.list_units)):
        unit1 = new_block.list_units[jj]
        for kk in range(jj):
            unit2 = new_block.list_units[kk]
            print('On trains: {}, {}'.format(unit1.name, unit2.name))
            unit1_name = 'Ch{},U{}'.format(unit1.name[0][-2:], unit1.name[1])
            STTCboots_dic['Node_1'].append(unit1_name)
            unit2_name = 'Ch{},U{}'.format(unit2.name[0][-2:], unit2.name[1])
            STTCboots_dic['Node_2'].append(unit2_name)
            surr = np.random.choice(surrogated_corr(
                unit1.spiketrains[0], unit2.spiketrains[0]), size=(1000, 5))
            boot_mean = np.mean(np.mean(surr, axis=1))
            boot_std = np.mean(np.std(surr, axis=1))
            print(boot_mean, boot_std)
            STTCboots_dic['STTCboot_mean'].append(boot_mean)
            STTCboots_dic['STTCboot_std'].append(boot_std)
    STTCboots_df = pd.DataFrame(STTCboots_dic)
    STTCboots_df.to_csv(ii.rsplit('_', 2)[0] + '_STTCboots')
    print('Done with {}/27 files'.format(files_count))
    files_count += 1

spiketrains[0].list_units

list_files
data_file = list_files[20]
data_file
name_keys = data_file.split(sep='_')
test_block = build_block()
data_block.list_units[0].name
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

# Transform STTC values to Adjacency Matrix
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
    sttc_graph = sttc_g(ii, threshold=2)

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
print(round(np.mean(nx_df[nx_df['Gen_type'] == 'KO']['Neg_w']), 2),
      round((stats.sem(nx_df[nx_df['Gen_type'] == 'KO']['Neg_w'])), 2))
deg = nx_df[nx_df['Gen_type'] == 'WT']['Nx_Degree']
den = nx_df[nx_df['Gen_type'] == 'WT']['Nx_Density']
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
#%%
with sns.axes_style('whitegrid'):
    sns.lmplot(x='Nx_Degree', y='Nx_Density', hue='Gen_type', data=nx_df, palette='deep', size=8)
    sns.despine()
#%%
den
plt.plot(deg, den)
den = np.arange(0, 10)
deg = np.arange(0, 10)
ko_density = nx_df[nx_df['Gen_type'] == 'KO']['Nx_Density']
wt_density = nx_df[nx_df['Gen_type'] == 'WT']['Nx_Density']
stats.ttest_ind(ko_density, wt_density)
stats.normaltest(ko_density)

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
with sns.axes_style('whitegrid'):
    sns.lmplot(x='Nx_Degree', y='Edges', hue='Gen_type',
               data=nx_df, palette='deep', size=8, legend_out=False)
    sns.despine()
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

with sns.axes_style('whitegrid'):
    sns.lmplot(x='Nx_Degree', y='Total_Weight', hue='Gen_type',
               data=nx_df, palette='deep', size=8, legend_out=False)
    sns.despine()


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
ko_avweight = nx_df[nx_df['Gen_type'] == 'KO']['Av_weight']
wt_avweight = nx_df[nx_df['Gen_type'] == 'WT']['Av_weight']
stats.ttest_ind(ko_avweight, wt_avweight)

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

with sns.axes_style('whitegrid'):
    sns.lmplot(x='Nx_Degree', y='Av_node_strength', hue='Gen_type',
               data=nx_df, palette='deep', size=8, legend_out=False)
    sns.despine()


#%%
with sns.axes_style('whitegrid'):
    edges_num = plt.figure(figsize=(8, 8))
    sns.stripplot(x='Gen_type', y='Edges', data=nx_df, jitter=0.05,
                  edgecolor='gray', linewidth=1, size=6.5, palette='deep', order=['WT', 'KO'])
    sns.boxplot(x='Gen_type', y='Edges', data=nx_df, order=[
                'WT', 'KO'], saturation=1, width=0.3, palette='deep', linewidth=2.5)
    sns.despine()
    plt.xlabel('Genotipo', fontsize=16)
    plt.ylabel('Número de Ejes', fontsize=16)
    plt.tick_params(labelsize=14)
#%%


ko_edges = nx_df[nx_df['Gen_type'] == 'KO']['Edges']
wt_edges = nx_df[nx_df['Gen_type'] == 'WT']['Edges']
stats.ttest_ind(wt_edges, ko_edges)

nx.info(sttc_graph)
nx.draw_circular(sttc_graph)
round(sum(nx.get_edge_attributes(sttc_graph, 'weight').values()), 2)
round(sum(nx.get_edge_attributes(sttc_graph, 'weight').values()), 2) / \
    int(nx.info(sttc_graph).split('\n')[3].split(' ')[-1])

nx_df['Av_weight'].mad()
help(nx_df.mad())
nx.info(wt_g)

ttest = sttc_list[12]
ttest_g = sttc_g(ttest, threshold=1, all_nodes=False)
nx.info(ttest_g)
ttest_g.degre()
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
MAD_test = plt.figure(figsize=(10, 10))
nx.draw_circular(ttest_g, with_labels=False,
                 node_size=350, alpha=.9, linewidths=.5, style='dotted')
plt.text(-1.1, -1.1, 'Nodos (N): 50\nEjes: 429\nConectividad: {}\nGrado promedio (k): 17.16'.format(
    round(nx.density(ttest_g), 2)), fontsize=16)
plt.show()
# MAD_test.savefig('MAD1.png')
#%%


#%%
plt.figure(figsize=(16, 16))
plt.hist([ii[1] for ii in list(nx.degree(sttc_graph))])
#%%

sttc_list
ko_66 = sttc_list[10]
wt_66 = sttc_list[19]
wt_66
ko_66
ko66_g = sttc_g(ko_66)
nx.info(ko66_g)
ko_degs = [jj for jj in [int(ii) for ii in dict(nx.degree(ko66_g)).values()]]

wt_g = sttc_g(wt_66)
wt_degs = [jj for jj in [int(ii) for ii in dict(nx.degree(wt_g)).values()]]

hist_val, hist_bins = np.histogram((wt_degs + ko_degs), bins='auto')
wt_values, wt_h = np.histogram(wt_degs, bins=hist_bins)
ko_values, ko_h = np.histogram(ko_degs, bins=hist_bins)

wt_values = wt_values / len(wt_g.nodes())
ko_values = ko_values / len(ko66_g.nodes())
wt_values
log_wt = np.log(wt_values + 1)
logko = np.log(ko_values + 1)
wt_values
ko_values
log_wt
hist_bins[:-1]

w_slope, w_intercept, wr_value, p_value, std_err = stats.linregress(
    np.log(hist_bins[:-1] + 1), np.log(wt_values + 1))
xfid = np.linspace(0, 3.7)     # This is just a set of x to plot the straight
k_slope, k_intercept, kr_value, p_value, std_err = stats.linregress(
    np.log(hist_bins[:-1] + 1), np.log(ko_values + 1))

wr_value**2
kr_value**2
nx.degree(wt_g)
# Draw graphs
#%%
ko_circ = plt.figure(figsize=(8, 8))
nx.draw_circular(ko66_g, node_color='g', node_size=100, style='dotted', width=.7)
plt.title('KO', fontsize=14)
plt.text(-1.1, -1.1, 'Densidad: {}'.format(
    round(nx.density(ko66_g), 2)) + '\n' + nx.info(ko66_g), fontsize=12)
#%%
ko_circ.savefig('KO graph.svg')

#%%
wt_circ = plt.figure(figsize=(8, 8))
nx.draw_circular(wt_g, node_color='b', node_size=100, style='dotted', width=.7)
plt.title('WT', fontsize=14)
plt.text(-1.1, -1.1, 'Densidad: {}'.format(
    round(nx.density(wt_g), 2)) + '\n' + nx.info(wt_g), fontsize=12)
#%%
wt_circ.savefig('WT graph.svg')

# Plot log log to approxiamte a power law fit
#%%
pow_law = plt.figure(figsize=(10, 6))
plt.plot(np.log(hist_bins[:-1] + 1), log_wt, 'bo', label='WT')
plt.plot(np.log(hist_bins[:-1] + 1), logko, 'go', label='KO')
plt.plot(xfid, xfid * w_slope + w_intercept, 'b-',
         label='$\mathit{R}^2$' + ' > {}'.format(round(wr_value**2, 1)))
plt.plot(xfid, xfid * k_slope + k_intercept, 'g-',
         label='$\mathit{R}^2$' + ' > {}'.format(round(kr_value**2, 1)))
plt.xlabel('log(k)', fontsize=14)
plt.ylabel('log(Proporción de Nodos)', fontsize=14)
plt.legend()
#%%
pow_law.savefig('power law approximation.svg')

hist_bins
hist_val

pow_dict = {'bins': hist_bins[1:], 'wt_values': wt_values, 'ko_values': ko_values}
pow_df = pd.DataFrame(pow_dict)
#%%
with sns.axes_style('whitegrid'):
    degree_hist = plt.figure(figsize=(6, 6))
    sns.distplot(wt_degs, bins=hist_bins, kde=False, label='WT')
    sns.distplot(ko_degs, bins=hist_bins, color='g', kde=False, label='KO')
    plt.xlabel('k', fontsize=14)
    plt.ylabel('Número de Nodos', fontsize=14)
    plt.legend(fontsize=14)

#%%
degree_hist.savefig('Degree Histogram.svg')

# Weight distribution Plot
raw_ko = pd.read_csv(ko_66).drop(columns=['Unnamed: 0'])
ko_median = np.median(raw_ko['STTC_weight'])
ko_mad = np.median(abs(raw_ko['STTC_weight'] - median)) / 0.6745
ko_thr = ko_mad * 3
ko_values = []
for i, nrow in sttc_df.iterrows():
    if nrow[4] < (ko_median + ko_thr) and nrow[4] > (ko_median - ko_thr):
        pass
    else:
        ko_values.append(round(abs(nrow[4]), 2))
raw_wt = pd.read_csv(wt_66).drop(columns=['Unnamed: 0'])


mad_values

#%%
with sns.axes_style('whitegrid'):
    weigths_hits = plt.figure(figsize=(8, 8))
    sns.distplot(abs(raw_wt['STTC_weight']), norm_hist=True, label='WT')
    sns.distplot(abs(raw_ko['STTC_weight']), norm_hist=True, kde=True, color='g', label='KO')
    plt.xlabel('STTC', fontsize=14)
    plt.ylabel('Frecuencia', fontsize=14)
    plt.legend(fontsize=14)
#%%
weigths_hits.savefig('Raw_hist.svg')

#%%
with sns.axes_style('whitegrid'):
    log_sttc = plt.figure(figsize=(8, 8))
    sns.distplot(np.log(abs(raw_wt['STTC_weight'])), norm_hist=True, label='WT')
    sns.distplot(np.log(abs(raw_ko['STTC_weight'])), norm_hist=True, color='g', label='KO')
    plt.xlabel('log(STTC)', fontsize=14)
    plt.ylabel('Densidad Normalizada', fontsize=14)
    plt.legend(fontsize=14)
#%%
log_sttc.savefig('logweights_hist.svg')
