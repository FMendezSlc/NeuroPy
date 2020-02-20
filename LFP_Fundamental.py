import neo
import numpy as np
import scipy as sp
import scipy.signal as sp_sig
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import fftpack

file_eg = neo.io.AxonIO('/Volumes/KINGSTON/adrian 1.abf')  # load file

sf_Hz = file_eg.get_signal_sampling_rate()  # get sampling frequency

an_sig = file_eg.get_analogsignal_chunk()[:, 0]  # a way to get the
# remember there's 2 active signals in the file, raw and integral
# you just use the first one
# analog
an_sig.shape
plt.plot(an_sig)
clean_rec = an_sig[4850000:]  # clean the record, up to that point
# it's just noise
tags = file_eg.get_event_timestamps()[0] / 2 - 4850000  # clean it
tags = tags.flatten()  # got to do this to make it work.
# type(tags)
# tags
plt.plot(clean_rec)
a_min = 60 * sf_Hz  # what's a minute but a bunch of samples?
# define the segments (not neo.segments) by [NaS]
basal = clean_rec[int(tags[1] - a_min):int(tags[1])]
NaS_1 = clean_rec[int(tags[1]):int(tags[1] + a_min)]
NaS_1rec = clean_rec[int(tags[2] - a_min):int(tags[2])]
NaS_2 = clean_rec[clean_rec[int(tags[2]):int(tags[2] + a_min)]]
NaS_2rec = clean_rec[int(tags[3] - a_min):int(tags[3])]
NaS_3 = clean_rec[int(tags[3]):int(tags[3] + a_min)]
NaS_3rec = clean_rec[int(tags[4] - a_min):int(tags[4])]
NaS_4 = clean_rec[int(tags[4]):int(tags[4] + a_min)]
NaS_4rec = clean_rec[int(tags[4] - a_min):int(tags[4])]
(len(bsl) / sf_Hz) / 60
# Verification plot of SoI
(int(tags[0]) + 75000) - int(tags[0])
a_min = 60 * sf_Hz
all_SoI = [basal, NaS_1, NaS_2, NaS_3, NaS_4]

#%%
plt.figure(figsize=(16, 8))
plt.plot()
#%%
wd = 4 * sf_Hz  # windows for Welch periodogram
f, px = sp_sig.welch(NaS_1[:], sf_Hz, nperseg=wd)
#%%
plt.figure(figsize=(16, 8))
psd_ax = plt.subplot(2, 1, 1)
plt.semilogy(f, px)
plt.xlim(0, 100)
psd_ax.set_xticks(ticks=range(0, 110, 10))
spec_ax = plt.subplot(2, 1, 2)
plt.specgram(NaS_1, Fs=sf_Hz, cmap='viridis', scale='dB')
plt.colorbar(pad=.01, fraction=0.01)
plt.ylim(0, 100)
#%%
# Normal lfp power analysis by bands, the usual stuff
f_res = f[1] - f[0]
total_power = simps(px, dx=f_res)  # all the power
delta = (1, 4)
theta = (4, 12)
beta = (15, 30)
gamma = (30, 100)

bands_dic = {'delta': (1, 4), 'theta': (4, 12), 'beta': (15, 30), 'gamma': (30, 100)}

spectral_dic = {'total_pw': [], 'delta_abs': [], 'theta_abs': [], 'beta_abs': [],
                'gamma_abs': [], 'delta_rel': [], 'theta_rel': [], 'beta_rel': [], 'gamma_rel': [], 'period': []}

spectral_dic['total_pw'] = total_power
spectral_dic['period'] =

for key in bands_dic:
    limits = bands_dic[key]
    band = np.logical_and(f >= limits[0], f <= limits[1])
    spectral_dic[key + '_abs'] = simps(px[band], dx=f_res)
    spectral_dic[key + '_rel'] = round(spectral_dic[key + '_abs'] / spectral_dic['total_pw'], 3)
spectral_dic

hf_Nas = NaS_2  # [int(len(NaS_3) / 2):]
no_win = int((len(hf_Nas)) // 5000)
(1813 / sf_Hz)
sf_Hz
no_win
len(bsl) / 2

#%%
plt.plot(f_X[np.logical_and(f_X > 0, f_X <= 100)], np.abs(
    red_psd[np.logical_and(f_X > 0, f_X <= 100)]))
plt.yscale('log')
#%%
[red_psd, blah] = sp_sig.periodogram(hf_Nas[lims[0]:lims[1]], fs=sf_Hz)

red_roi = red_psd[np.logical_and(red_psd >= 0, red_psd <= 200)]
red_roi
red_psd
fund_fqs = np.empty((len(red_roi), no_win))

np.append(fund_fqs, [np.arange(no_win)], axis=0)
num_windows = no_win
lims = np.array([0, 5000])
for ii in range(no_win):
    [f_X, X] = sp_sig.periodogram(hf_Nas[lims[0]:lims[1]], fs=sf_Hz)
    red_roi = X[np.logical_and(f_X >= 0, f_X <= 200)].T
    fund_fqs[:, ii] = red_roi
    lims += 5000

#%%
fufqs_fig = plt.figure(figsize=(10, 10))
plt.imshow(np.corrcoef(fund_fqs), origin='lower', extent=[0, 200, 0, 200], cmap='RdGy_r')
plt.colorbar(shrink=.8, aspect=10)
plt.ylabel('Frequency(Hz)')
plt.xlabel('Frequency(Hz)')
plt.xlim(0, 100)
plt.ylim(0, 100)
#%%
