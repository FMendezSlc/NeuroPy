import neo
import numpy as np
import scipy as sp
import scipy.signal as sp_sig
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import fftpack

file_eg = neo.io.AxonIO('/Volumes/KINGSTON/adrian 1.abf')

sf_Hz = file_eg.get_signal_sampling_rate()

an_sig = file_eg.get_analogsignal_chunk()

tags = file_eg.get_event_timestamps()[0] / 2 - 4850000
tags = tags.flatten()  # got to do this to make it work.
type(tags)


clean_rec = an_sig[4850000:, :]
bsl = clean_rec[0:int(tags[0])]
NaS_1 = clean_rec[int(tags[0]):int(tags[1])]
NaS_2 = clean_rec[int(tags[1]):int(tags[2])]
NaS_3 = clean_rec[int(tags[2]):int(tags[3])]
NaS_4 = clean_rec[int(tags[3]):int(tags[4])]
(len(bsl) / sf_Hz) / 60
#%%
plt.figure(figsize=(16, 8))
plt.plot(NaS_1)
#%%
wd = 5 * sf_Hz
type(wd)
[f, px] = sp_sig.welch(NaS_1[:, 0], sf_Hz, nperseg=wd)
#%%
plt.figure(figsize=(16, 8))
plt.subplot(2, 1, 1)
plt.plot(f[f < 100], px[f < 100])
plt.subplot(2, 1, 2)
plt.specgram(NaS_1[:, 0], Fs=sf_Hz, cmap='viridis', scale='dB')
plt.colorbar()
plt.ylim(0, 100)
#%%

f_res = f[1] - f[0]
total_power = simps(px, dx=freq_res)
delta = (1, 4)
theta = (4, 12)
beta = (15, 30)
gamma = (30, 100)

bands_dic = {'delta': (1, 4), 'theta': (4, 12), 'beta': (15, 30), 'gamma': (30, 100)}

spectral_dic = {'total_pw': [], 'delta_abs': [], 'theta_abs': [], 'beta_abs': [],
                'gamma_abs': [], 'delta_rel': [], 'theta_rel': [], 'beta_rel': [], 'gamma_rel': []}

spectral_dic['total_pw'] = total_power

for key in bands_dic:
    limits = bands_dic[key]
    band = np.logical_and(f >= limits[0], f <= limits[1])
    spectral_dic[key + '_abs'] = simps(px[band], dx=freq_res)
    spectral_dic[key + '_rel'] = round(spectral_dic[key + '_abs'] / spectral_dic['total_pw'], 3)
spectral_dic

no_win = int((len(hf_Nas)) // 5000)
(1813 / sf_Hz)
4 * sf_Hz
len(bsl) / 2
hf_Nas = NaS_4[int(len(NaS_4) / 2):]
#%%
plt.plot(f_X[np.logical_and(f_X > 0, f_X <= 100)], np.abs(
    red_psd[np.logical_and(f_X > 0, f_X <= 100)]))
plt.yscale('log')
#%%
red_roi = red_psd[np.logical_and(f_X >= 0, f_X <= 100)]
fund_fqs = np.empty((len(red_roi), no_win))

np.append(fund_fqs, [np.arange(no_win)], axis=0)
num_windows = no_win
lims = np.array([0, 5000])
for ii in range(num_windows):
    [f_X, X] = sp_sig.periodogram(hf_Nas[lims[0]:lims[1], 0], fs=sf_Hz)
    red_roi = X[np.logical_and(f_X >= 0, f_X <= 300)].T
    fund_fqs[:, ii] = red_roi
    lims += 5000

#%%
fufqs_fig = plt.figure(figsize=(10, 10))
plt.imshow(np.corrcoef(fund_fqs), origin='lower', extent=[0, 300, 0, 300], cmap='RdGy_r')
plt.colorbar(shrink=.8, aspect=10)
plt.ylabel('Frequency(Hz)')
plt.xlabel('Frequency(Hz)')
#plt.xlim(0, 100)
#plt.ylim(0, 100)
#%%
