import numpy as np
import neo
import matplotlib.pyplot as plt
import pandas as pd
from quantities import Hz, s, ms
import elephant as elp
import seaborn as sns
import networkx as netx
import os


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


ff = '/Users/felipeantoniomendezsalcido/Desktop/MEAs/spike_times/CA3_KO_male_170218_merged_sp.txt'
ex_exp = build_block(
    '/Users/felipeantoniomendezsalcido/Desktop/MEAs/spike_times/CA3_KO_male_170218_merged_sp.txt')

graph_edges = []
for ii in ex_exp.list_units:


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


os.chdir('/Users/felipeantoniomendezsalcido/Desktop/MEAs/spike_times')
# Exploratory dataframe for an experiment
list_files = os.listdir()
list_files = list_files[1:]
len(list_files)
list_files

stats_dic = {'Date': [], 'Gen_type': [], 'Sex': [],
             'Channel': [], 'Unit': [], 'FR_Bs': [], 'CV2': []}

for ii in list_files:
    data_file = os.getcwd() + '/' + ii
    name_keys = ii.split(sep='_')
    data_block = build_block(ii)
    for unit in data_block.list_units:
        stats_dic['Date'].append(name_keys[3])
        stats_dic['Gen_type'].append(name_keys[1])
        stats_dic['Sex'].append(name_keys[2])
        stats_dic['FR_Bs'].append(elp.statistics.mean_firing_rate(unit.spiketrains[0]).item())
        stats_dic['Channel'].append(unit.name[0])
        stats_dic['Unit'].append(unit.name[1])
        intervals = elp.statistics.isi(unit.spiketrains[0])
        stats_dic['CV2'].append(elp.statistics.cv(intervals))

build_df = pd.DataFrame(stats_dic)
new_ord = ['Date', 'Sex', 'Gen_type', 'Channel', 'Unit', 'FR_Bs', 'CV2']
ordered_df = build_df[new_ord]

ordered_df
ordered_df
os.chdir('/Users/felipeantoniomendezsalcido/Desktop/MEAs')
ordered_df.to_csv('/Users/felipeantoniomendezsalcido/Desktop/MEAs/pooled_data', index=False)


# Build the crosscorrelation matrix according to Eggermont
binned_trains = [BinnedSpikeTrain(ii, binsize=0.001 * s) for ii in ex_exp.segments[0].spiketrains]
few_trains = binned_trains[0:9]

egg_mat_size = len(few_trains)
egg_mat = np.zeros((egg_mat_size, egg_mat_size), float)

for ii in range(egg_mat_size):
    print('Im on it, Neuron {}'.format(ii))
    for jj in range(egg_mat_size):
        cch_try = sp_cor.cch(few_trains[ii], few_trains[jj], cross_corr_coef=True)
        egg_mat[ii, jj] = max(cch_try)

# Pearson's Correlation Matrix
cc_mat = corrcoef(binned_trains)
np.fill_diagonal(cc_mat, 0)

plt.figure(figsize=(10, 10))
sns.heatmap(cc_mat, square=True, cmap='Spectral_r')
plt.ylabel('Cells')
plt.xlabel('Cells')
plt.show()

# Raster Plots
plt.figure(figsize=(30, 30))
plt.eventplot(n_block.segments[0].spiketrains, color='k', linelengths=0.5)
plt.xlabel('Seconds')
plt.ylabel('Cell')
plt.yticks(range(len(n_block.segments[0].spiketrains)))
plt.show()

# Network Spikes
time_hit = est.time_histogram(n_block.segments[0].spiketrains, binsize=0.005 * sec)
plt.figure(figsize=(18, 5))
plt.plot(time_hit, color='k')
plt.ylabel('# active cells')
plt.xlabel('Seconds')
plt.show()

# for ii in list_files:
list_files
data_file = list_files[20]
data_file
name_keys = data_file.split(sep='_')
data_block = build_block(data_file)
len(data_block.list_units)
# STTC calculations, remeber you did it under the diagonal
# That is N*N - N % 2
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
#%%
len(data_block.list_units)
len(STTC_dic['STTC_weight'])
STTC_df.to_csv(data_file.split('.')[0] + 'STTC')

mat_size = len(data_block.list_units)
#%%
sttc_mat = np.empty((mat_size, mat_size))
sttc_mat[:] = np.nan
count = 0
for ii in range(mat_size):
    for jj in range(ii):
        sttc_mat[ii, jj] = STTC_df['STTC_weight'][count]
        count += 1
count
#%%
STTC_df['Node_2'].unique()
STTC_df['Node_1'].unique()
names_ag = [ii for ii in STTC_df['Node_2'].unique()]
STTC_df['Node_1'].iloc[-1]
names_ag.append(STTC_df['Node_1'].iloc[-1])
len(names_ag)
sttc_round = np.around(sttc_mat, decimals=2)
#%%
mat_fig = plt.figure(figsize=(14, 14))
sns.heatmap(sttc_round, center=0, cmap='seismic', cbar_kws={
            'shrink': .8}, yticklabels=names_ag, xticklabels=names_ag, square=True)
plt.title(data_file)
plt.ylabel('Units')
plt.xlabel('Units')
plt.show()
#%%
mat_fig.savefig(data_file.split('.')[0] + '_fig.svg')
data_file.split('.')[0]
# Stupid demonstration of half matrix creation
# Remeber this shit man, you don't want to do it yet again
bs = []
bs_c = 0
for ii in range(10):
    for jj in range(ii):
        bs.append(bs_c)
        bs_c += 1
bs

len(names_df['Node_2.0'].unique())
mat_mean = np.nanmean(sttc_mat)
mat_std = np.nanstd(sttc_mat)

trld = mat_mean + (1 * mat_std)
trld
sttc_round = np.around(sttc_mat, decimals=2)
len(sttc_round[sttc_round > 0])


plt.figure(figsize=(10, 10))
sns.heatmap(sttc_round, center=0, cmap='seismic', cbar_kws={'shrink': .8})
plt.show()

plt.figure(figsize=(10, 10))
sns.distplot(sttc_mat[~np.isnan(sttc_mat)])
plt.show()


sttc_mat = np.reshape(sttc_mat, (63, 63))
np.fill_diagonal(sttc_mat, 0)
eg_5ms = plt.figure(figsize=(12, 12))
sns.heatmap(sttc_mat, cmap='seismic', center=0, square=True, cbar_kws={'shrink': 0.80})
plt.ylabel('Cells')
plt.xlabel('Cells')
plt.show()
eg_5ms.savefig('example_experiment_5ms')

sns.heatmap()


len(ex_exp.segments[0].spiketrains)
sttc_mat_1ms = np.empty((63, 63))
for ii in range(63):
    for jj in range(63):
        sttc_mat_1ms[ii, jj] = sttc(ex_exp.segments[0].spiketrains[ii],
                                    ex_exp.segments[0].spiketrains[jj], dt=0.001 * s)

ddt = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1]  # windows to test
len(ddt)
trains_test = [ii for ii in ex_exp.segments[0].spiketrains[0:10]]  # trains to test

sttc_test = []

for ii in ddt:
    dt = ii * s
    sttc_val = []
    for jj in trains_test:
        sttc_val.append(sttc(trains_test[1], jj, dt=dt))
        sttc_test.append(sttc_val)
sttc_test
df = pd.DataFrame(sttc_test)
df = df.drop_duplicates()
df.index = ddt
window_test = plt.figure(figsize=(13, 10))
for col in df:
    plt.semilogx(df[col], 'o-')
plt.xticks(ddt)
plt.xlabel('Synchronicity window in log10[1 Seg]')
plt.ylabel('Spike Time Tiling Coefficient')
plt.show()

window_test.savefig('Synchronicity Window test')
