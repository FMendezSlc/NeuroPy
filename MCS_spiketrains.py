import numpy as np
import neo
import matplotlib.pyplot as plt
import pandas as pd
from quantities import Hz, s, ms
from elephant.conversion import BinnedSpikeTrain
from elephant import spike_train_generation as sp_gen
import seaborn as sns
import elephant.statistics as est
import elephant.spike_train_correlation as sp_cor
import os


def build_block(data_file):
    """(plaintextfile_path) -> neo.core.block.Block
    Thie function reads a plain text file (data_file) with the data exported (per waveform) from Plexon Offline Sorter after sorting and returns a neo.Block with spiketrains ordered in any number of 10 minute segments.

    For practicality and Plexon management of channels names, units and channels have been ignored in the block structure."""

    raw_data = pd.read_csv(data_file, sep=',', header=0, usecols=[0, 1, 2])
    ord_times = raw_data.groupby(['Channel', 'Unit'])['Timestamp']
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


ex_exp = build_block(
    '/Users/felipeantoniomendezsalcido/Desktop/MEAs/data_files/CA3_WT_fem_091117_merged_sp.txt')
ex_exp.segments[0].spiketrains[0][-1]
# Working function to calculate the Spike Time Tiling Coefficient
# According to Cutts and Eglen (2014)
# Original function written in C (github/CCutts)


def sttc(spiketrain_1, spiketrain_2, dt=0.005 * s):
    ''' Calculates the Spike Time Tiling Coefficient as described in (Cutss & Eglen, 2014) following Cutts' implementation in C (https://github.com/CCutts/Detecting_pairwise_correlations_in_spike_trains/blob/master/spike_time_tiling_coefficient.c)'''

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

            T = (time_A / spiketrain.t_stop).item()
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


os.chdir('/Users/felipeantoniomendezsalcido/Desktop/MEAs/data_files')
# Exploratory dataframe for an experiment
list_files = os.listdir()
stats_dic = {'Date': [], 'Gen_type': [], 'Sex': [], 'FR_Bs': [], 'FR_TBS': [],
             'FR_TBS-30': [], 'FR_TBS-60': [], 'CV2': [], 'Fano_fact': []}
for ii in list_files:
    data_file = os.getcwd() + '/' + ii
    name_keys = ii.split(sep='_')
    data_block = build_block(data_file)

    for unit in data_block.list_units:
        stats_dic['Date'].append(name_keys[3])
        stats_dic['Gen_type'].append(name_keys[1])
        stats_dic['Sex'].append(name_keys[2])
        stats_dic['FR_Bs'].append(est.mean_firing_rate(unit.spiketrains[0]).item())
        stats_dic['FR_TBS'].append(est.mean_firing_rate(unit.spiketrains[1]).item())
        stats_dic['FR_TBS-30'].append(est.mean_firing_rate(unit.spiketrains[2]).item())
        stats_dic['FR_TBS-60'].append(est.mean_firing_rate(unit.spiketrains[3]).item())
        intervals = est.isi(unit.spiketrains[0])
        stats_dic['CV2'].append(est.cv(intervals))
        stats_dic['Fano_fact'].append(est.fanofactor(unit.spiketrains))


build_df = pd.DataFrame(stats_dic)
new_ord = ['Date', 'Sex', 'Gen_type', 'CV2', 'FR_Bs',
           'FR_TBS', 'FR_TBS-30', 'FR_TBS-60', 'Fano_fact']
ordered_df = build_df[new_ord]
ordered_df['Fano_fact']

ordered_df['FR_Bs'][0].
plt.figure(figsize=(10, 10))
sns.pairplot(data=ordered_df, hue='Gen_type', vars=['FR_Bs', 'CV2', 'Fano_fact'])
plt.show()

plt.figure(figsize=(10, 10))
sns.stripplot(x='Gen_type', y='FR_Bs', hue='Sex', data=ordered_df, jitter=True)
plt.show()


# Now, characterize them
# What are the most useful statistics to describe a spike train?
# Firing rate, brust/regular, cv, network spikes, spikes in burst, burst rate

sns.stripplot(exp_df['Firing_Rate'])
plt.xticks(range(21))
plt.show()

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

egg_mat
bin_mat = binned_trains.to_array()
bin_mat.shape
lim = bin_mat.shape[1]
lim
sim_mat = np.zeros(shape=(lim, lim))
for i in range(lim):
    for j in range(lim):
        a = bin_mat[:, i]
        b = bin_mat[:, j]
        n_a = np.linalg.norm(a)
        n_b = np.linalg.norm(b)
        sim = (np.dot(a, b) / (np.dot(n_a, n_b)))
        sim_mat[i, j] = sim

plt.figure(figsize=(15, 10))
sns.heatmap(sim_mat)
plt.show()

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
# plt.eventplot(exp_block.segments[1].spiketrains, color='k')
# plt.eventplot(exp_block.segments[2].spiketrains, color='k')
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
