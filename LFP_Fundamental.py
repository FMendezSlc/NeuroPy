import neo
import numpy as np
import scipy as sp
import scipy.signal as sp_sig
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import fftpack

file_name = '/Volumes/KINGSTON/adrian 1.abf'  # .abf files only
file_eg = neo.io.AxonIO(file_name)  # load file

sf_Hz = file_eg.get_signal_sampling_rate()  # get sampling frequency

an_sig = file_eg.get_analogsignal_chunk()[:, 0]  # a way to get the
# remember there's 2 active signals in the file, raw and integral
# you just use the first one; analog signal
# an_sig.shape
plt.plot(an_sig)

clean_rec = an_sig[4850000:]  # clean the record, up to that point
# it's just noise
#%%
plt.figure(figsize=(16, 8))
plt.plot(clean_rec)
#%%
# now, let's get the tags. tags tell you the segments of interest
tags = file_eg.get_event_timestamps()[0] / 2 - 4850000  # clean it
tags = tags.flatten()  # got to do this to make it work.
# type(tags)
# tags

a_min = 60 * sf_Hz  # what's a minute but a bunch of samples?


def get_segments(num_seg=4, tags=tags, rec_sig=clean_rec):
    '''[int, event_timestamps, array-like -> dictionary] Get dictionary which keys contain one minute of recorded signal around a tag (epoch)'''
    use_tags = tags[0:num_seg]
    seg_dict = {}

    for a_seg in range(num_seg):
        seg_dict['SoI_{}_pre'.format(a_seg)] = clean_rec[int(tags[a_seg] - a_min):int(tags[a_seg])]
        seg_dict['SoI_{}_post'.format(a_seg)] = clean_rec[int(tags[a_seg]):int(tags[a_seg] + a_min)]

    seg_dict['SoI_lePilon'] = clean_rec[int(
        tags[num_seg] + (20 * a_min)):int(tags[num_seg] + (21 * a_min))]

    return seg_dict


SoI = get_segments(4, tags, clean_rec)

# Normal lfp power analysis by bands, the usual stuff

bands_dic = {'delta': (1, 4), 'theta': (4, 12), 'beta': (
    15, 30), 'gamma_slow': (30, 60), 'gamma_fast': (60, 100)}

spectral_dic = {'total_pw': [], 'delta_abs': [], 'theta_abs': [], 'beta_abs': [],
                'gamma_fast_abs': [], 'gamma_slow_abs': [], 'delta_rel': [], 'theta_rel': [], 'beta_rel': [], 'gamma_slow_rel': [], 'gamma_fast_rel': [], 'period': []}


wd = 4 * sf_Hz  # windows for Welch periodogram in samples

for key in SoI:  # plot it like a plot
    f, px = sp_sig.welch(SoI[key], sf_Hz, nperseg=wd)  # welch periodogram
    f_res = f[1] - f[0]  # frequency resolution
    total_power = simps(px, dx=f_res)  # all the power
    spectral_dic['total_pw'].append(total_power)
    spectral_dic['period'].append(key)

    for band_key in bands_dic:
        limits = bands_dic[band_key]
        band = np.logical_and(f >= limits[0], f <= limits[1])
        band_abs = round(simps(px[band], dx=f_res), 5)
        spectral_dic[band_key + '_abs'].append(band_abs)
        band_rel = round(band_abs / total_power, 5)
        spectral_dic[band_key + '_rel'].append(band_rel)
    # # Now, fundamental frequencies analysis
    num_win = int((len(SoI[key])) // 5000)  # 4 second windows
    lims = np.array([0, 5000])  # limits for the moving window
    red_psd, _ = sp_sig.periodogram(SoI[key][lims[0]:lims[1]], fs=sf_Hz)
    red_roi = red_psd[np.logical_and(red_psd >= 0, red_psd <= 200)]
    fund_fqs = np.empty((len(red_roi), no_win))

    for ii in range(no_win):
        [f_X, X] = sp_sig.periodogram(SoI[key][lims[0]:lims[1]], fs=sf_Hz)
        red_roi = X[np.logical_and(f_X >= 0, f_X <= 200)]
        fund_fqs[:, ii] = red_roi
        lims += 5000
        len(red_roi)

    spec_fig, (psd_ax, spec_ax, red_ax) = plt.subplots(
        3, 1, gridspec_kw={'height_ratios': [1, 1, 2]}, figsize=(7.3, 10))

    psd_ax.set_title('Periodograma (Welch)')
    psd_ax.semilogy(f, px)
    psd_ax.set_ylabel('PSD (V^2/Hz)')
    psd_ax.set_xlim(0, 100)
    psd_ax.set_xticks(ticks=range(0, 105, 5), minor=True)
    psd_ax.set_xlabel('Frecuencia [Hz]')
    psd_ax.set_xticks(ticks=range(0, 110, 10))

    spec_ax.set_title('Espectrograma')
    spectogram = spec_ax.specgram(SoI[key], Fs=sf_Hz, cmap='viridis', scale='dB')
    spec_ax.set_ylabel('Frecuencia (Hz)')
    plt.colorbar(spectogram[-1], pad=.01, fraction=0.03, ax=spec_ax)
    spec_ax.set_ylim(0, 100)

    red_corr = red_ax.imshow(np.corrcoef(fund_fqs), origin='lower',
                             extent=[0, 200, 0, 200], cmap='RdGy_r')
    plt.colorbar(red_corr, pad=.07, fraction=0.1, ax=red_ax)
    red_ax.set_title('Frecuencias Fundamentales')
    red_ax.set_ylabel('Frecuencia (Hz)')
    red_ax.set_xlabel('Frecuencia (Hz)')
    # red_ax.set_xlim(0, 100) # this is to make a zoom-like figure
    # red_ax.set_ylim(0, 100) # same, gotta use both
    plt.tight_layout()
    plt.show()
    # to save the figs uncomment the following line and add an appropiate
    # directory
    # spec_fig.savefig(file_name.split('.')[0]+'_{}_spectral_fig.png'.format(key))
    # to get a vector image change file notation to .svg
#%%
spec_data = pd.DataFrame(spectral_dic)
# to save the DataFrame with power meassures
spec_data.to_csv('file_name.split('.')[0]' + 'data.csv')
# uncomment and give a proper directory
