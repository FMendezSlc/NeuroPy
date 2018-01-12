import numpy as np
import neo
import matplotlib.pyplot as plt
import pandas as pd
from quantities import Hz, s, ms
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef
import seaborn as sns
import elephant.statistics as est
import elephant.spike_train_correlation as sp_cor


def build_block(data_file):
    """(plaintextfile_path) -> neo.core.block.Block
    Thie function reads a plain text file (data_file) with the data exported (per waveform) from Plexon Offline Sorter after sorting and returns a neo.Block with spiketrains ordered in any number of 10 minute segments.

    For practicality and Plexon management of channels names, units and channels have been ignored in the block structure."""

    raw_data = pd.read_csv(data_file, sep=',', header=0, usecols=[0, 1, 2])
    ord_times = raw_data.groupby(['Channel', 'Unit'])['Timestamp']
    new_block = neo.Block()
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

        for seg in num_segments[1:-1]:
            seg_train = neo.SpikeTrain(time_stamps[(time_stamps > seg * inter) & (time_stamps < ((seg + 1) * inter))],
                                       units='sec', t_start=(seg * inter), t_stop=((seg + 1) * inter))
            new_block.segments[seg].spiketrains.append(seg_train)

        last_seg = neo.SpikeTrain(time_stamps[time_stamps > (num_segments[-1] * inter)], units='sec',
                                  t_start=(num_segments[-1]) * inter, t_stop=((num_segments[-1] + 1) * inter))
        new_block.segments[num_segments[-1]].spiketrains.append(last_seg)
    return new_block


data_file = '/Users/felipeantoniomendezsalcido/Desktop/MEAs/data_files/CA3_KO_fem_21117_merged_SP.txt'

n_block = build_block(data_file)
# So we got the spike trains all tidy and nice in a block by segments
# Now, characterize them
# What are the most useful statistics to describe a spike train?
# Firing rate, brust/regular, cv, network spikes, spikes in burst, burst rate

# Build a DataFrame for that data
num_spt = len(n_block.segments[0].spiketrains)
exp_dic = {'Firing_Rate': np.zeros(num_spt), 'CV2': np.zeros(num_spt), 'LVar': np.zeros(num_spt)}

ind_i = 0

for ii in n_block.segments[0].spiketrains:
    exp_dic['Firing_Rate'][ind_i] = est.mean_firing_rate(ii)
    ii_isis = est.isi(ii)
    if len(ii_isis) > 1:
        exp_dic['CV2'][ind_i] = est.cv(ii_isis)
        exp_dic['LVar'][ind_i] = est.lv(ii_isis)
    else:
        print('Not enough spikes in train {}'.format(ind_i))
    ind_i += 1

exp_df = pd.DataFrame(exp_dic)
exp_df.describe()


sns.stripplot(exp_df['Firing_Rate'])
plt.xticks(range(21))
plt.show()
est.lv(est.isi(n_block.segments[0].spiketrains[0]))

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
#plt.eventplot(exp_block.segments[1].spiketrains, color='k')
#plt.eventplot(exp_block.segments[2].spiketrains, color='k')
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
