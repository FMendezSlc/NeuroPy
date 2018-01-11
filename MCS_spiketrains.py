import numpy as np
import scipy as sp
import neo
import matplotlib.pyplot as plt
import pandas as pd
from quantities import Hz, s, ms
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef
import seaborn as sb
import elephant.statistics as est
import elephant.spike_train_correlation as sp_cor
import os as os


cd '/Users/felipeantoniomendezsalcido/Desktop/MEAs/data_files'
ls
data_file = os.listdir()[2]
raw_data = pd.read_csv(data_file, sep=',', header=0, usecols=[0, 1, 2])
ord_times = raw_data.groupby(['Channel', 'Unit'])['Timestamp']


def build_block(data_file):
    """(plaintextfile_path) -> neo.core.block.Block
    Thie function reads a plain text file (data_file) with the data exported (per waveform) from Plexon Offline Sorter after sorting and returns a neo.Block with spiketrains ordered in any number of 10 minute segments.

    For practicality and Plexon management of channels names, units and channels have been ignored in the block structure."""

    raw_data = pd.read_csv(data_file, sep=',', header=0, usecols=[0, 1, 2])
    ord_times = raw_data.groupby(['Channel', 'Unit'])['Timestamp']
    new_block = neo.Block()
    num_segments = range(int(raw_data['Timestamp'].max() // 600 + 1))
    for ind in num_segments:
        seg = neo.Segment(name='segment {}'.format(ind), index=ind)
        new_block.segments.append(seg)

    for name, group in ord_times:
        st = ord_times.get_group(name).values
        inter = 600
        first_seg = neo.SpikeTrain(st[st < inter], units='sec', t_stop=inter)
        new_block.segments[0].spiketrains.append(first_seg)
        seg_count = 1

        for seg in num_segments:
            seg_count += 1
            if seg_count < num_segments[-1]:
                seg_train = neo.SpikeTrain(st[(st > (seg_count - 1 * inter)) & (st < (seg_count * inter))],
                                           units='sec', t_start=(seg_count - 1 * inter), t_stop=(seg_count * inter))
                new_block.segments[seg_count].spiketrains.append(seg_train)

        last_seg = neo.SpikeTrain(st[st > (num_segments[-1] * inter)], units='sec',
                                  t_start=(num_segments[-1]) * inter, t_stop=((num_segments[-1] + 1) * inter))
        new_block.segments[num_segments[-1]].spiketrains.append(last_seg)
    return new_block


n_block = build_block(data_file)

# So we got the spike trains all tidy and nice in a block by segments
# Now, characterize them
# What are the most useful statistics to describe a spike train?
# Firing rate, brust/regular, cv, network spikes, spikes in burst, burst rate

# Build a DataFrame for that data

stat_dic = {'F_Rate': (), 'CV': (), }
for i in n_block.segments[0].spikestrains:
    get_rate = est.mean_firing_rate(i)
    get_cv = est.cv(i)

train = n_block.segments[0].spiketrains
len(train)
isis = est.isi(train)


# Calculate and build the crosscorrelation matrix
binned_trains = BinnedSpikeTrain(n_block.segments[0].spiketrains, binsize=10 * s)

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
sb.heatmap(sim_mat)
plt.show()

cc_mat = corrcoef(binned_trains)
np.fill_diagonal(cc_mat, 0)


plt.figure(figsize=(10, 10))
sb.heatmap(cc_mat, square=True, cmap='Spectral_r')
plt.ylabel('Cells')
plt.xlabel('Cells')
plt.show()


# Raster Plots
plt.figure(figsize=(30, 30))
plt.eventplot(n_block.segments[0].spiketrains, color='k', linelengths=0.5)
#plt.eventplot(exp_block.segments[1].spiketrains, color='k')
#plt.eventplot(exp_block.segments[2].spiketrains, color='k')


plt.xlabel('Seconds')
plt.ylabel('Cell')
plt.yticks(range(42))
plt.show()


plt.figure(figsize=(30, 2))
plt.eventplot(train, color='k', linelengths=0.3)
plt.show()

# Network Spikes
time_hit = est.time_histogram(n_block.segments[0].spiketrains, binsize=0.005 * sec)


plt.figure(figsize=(18, 5))
plt.plot(time_hit, color='k')
plt.ylabel('# active cells')
plt.xlabel('Seconds')
plt.show()
