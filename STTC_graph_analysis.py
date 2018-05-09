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
