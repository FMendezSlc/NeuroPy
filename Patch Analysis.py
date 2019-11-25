import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

Patch_data = pd.read_excel(
    '/Users/felipeantoniomendezsalcido/Desktop/Data/Patch Analysis.xlsx', sheet_name = None)

Patch_data.keys()
Patch_data['sp_Peaks']


def ecdf(raw_data):
    '''[np.array -> tuple]
    Equivalent to R's ecdf(). Credit to Kripanshu Bhargava from Codementor'''
    cdfx = np.unique(raw_data)
    x_values = np.linspace(start=min(cdfx), stop=max(cdfx), num=len(cdfx))
    size_data = raw_data.size
    y_values = []
    for i in x_values:
        # all the values in raw data less than the ith value in x_values
        temp = raw_data[raw_data <= i]
        # fraction of that value with respect to the size of the x_values
        value = temp.size / size_data

        y_values.append(value)

    return (x_values, y_values)

# Gap free Analysis
# Remeber amp is in pA, int is instantaneous fq in Hz
spEPSC = Patch_data['sp_Peaks']
spInFq = Patch_data['sp_Int']
spInFq.columns == spEPSC.columns
spMeans = {'Cell' : [], 'Genotype' : [], 'Mean (pA)' : [], 'Mean_IsI' : []}

for column in spEPSC:
    spMeans['Cell'].append(column)
    spMeans['Genotype'].append(column.split('_')[1])
    spMeans['Mean (pA)'].append(abs(spEPSC[column].mean()))
    spMeans['Mean_IsI'].append(spInFq[column].mean())

tidy_spMeans = pd.DataFrame(spMeans)
tidy_spMeans

WT_epsp = np.array([])
KO_epsp = np.array([])
WT_isi = np.array([])
KO_isi = np.array([])

for column in spEPSC:
    if 'WT' in column:
        WT_epsp = np.append(WT_epsp, spEPSC[column].dropna().abs())
        WT_isi = np.append(WT_isi, spInFq[column].dropna())
    elif 'KO' in column:
        KO_epsp = np.append(KO_epsp, spEPSC[column].dropna().abs())
        KO_isi = np.append(KO_isi, spInFq.dropna())
KO_ecdf = ecdf(WT_epsp.round(2))
WT_ecdf = ecdf(KO_epsp.round(2))
KO_isicd = ecdf(KO_isi.round(2))
WT_isicd = ecdf(WT_isi.round(2))

len(KO_epsp)

sns.barplot(x = 'Genotype', y = 'Mean_IsI', data = tidy_spMeans, ci = 68, capsize = .1)

sns.distplot(spEPSC.iloc[:, 0].dropna(), kde = False)
WT_epsp

WT_ecdf = ecdf(np.abs(np.array(WT_epsp)))
KO_ecdf = ecdf(np.abs(np.array(KO_epsp)))

KS = stats.ks_2samp(WT_isi, KO_isi)
KS

Wts = tidy_spMeans['Mean (pA)'][tidy_spMeans['Genotype'] == 'WT']
Kos = tidy_spMeans['Mean (pA)'][tidy_spMeans['Genotype'] == 'KO']

stats.ttest_ind(Wts, Kos)
stats.shapiro(Wts)
#%%
ecdf_epsc, axs = plt.subplots(1, 2, figsize=(7, 4))
axs[0].plot(WT_ecdf[0], WT_ecdf[1], color='b', label='WT', linewidth=3)
axs[0].plot(KO_ecdf[0], KO_ecdf[1], color='g', label='KO', linewidth=3)
axs[0].set(ylabel  = 'Cum. P.', xlabel='Amplitude (pA)')

sns.despine(ax = axs[0])
ins = axs[0].inset_axes([.7, .2, .2, .5])
sns.barplot(x = 'Genotype', y = 'Mean (pA)', data = tidy_spMeans, ci = 68, errwidth = 1.5, capsize = .1, palette = ['b', 'g'], ax = ins)
sns.despine(ax = ins)
ins.set(xlabel='', ylabel='Mean (pA)')

axs[1].plot(WT_isicd[0], WT_isicd[1], color='b', label='WT', linewidth=3)
axs[1].plot(KO_isicd[0], KO_isicd[1], color='g', label='KO', linewidth=3)
axs[1].set(ylabel  = '', xlabel='Int. Fq. (Hz)')

sns.despine(ax = axs[1])
ins = axs[1].inset_axes([.7, .2, .2, .5])
sns.barplot(x = 'Genotype', y = 'Mean_IsI', data = tidy_spMeans, ci = 68, errwidth = 1.5, capsize = .1, palette = ['b', 'g'], ax = ins)
sns.despine(ax = ins)
ins.set(xlabel='', ylabel='Mean Int Fq (Hz)')
#%%
ecdf_epsc.savefig('/Users/felipeantoniomendezsalcido/Desktop/PDCB pics/gap_free.png', dpi = 300)
#%%
Patch_data.keys()
spInFq = Patch_data['sp_Int']
spInFq.max()
cols = []
for column in spInFq.filter(like='KO'):
    cols.append(spInFq[column].dropna())
    merged = pd.concat(cols, ignore_index=True)
cols
all_wt = np.abs(merged.round(2))
all_ko = np.abs(merged.round(2))

KO_instfq = ecdf(all_ko)
WT_instfq = ecdf(all_wt)

#%%
ecdf_infq = plt.figure(figsize=(10, 8))
plt.plot(WT_instfq[0], WT_instfq[1], color='b', label='WT', linewidth=3)
plt.plot(KO_instfq[0], KO_instfq[1], color='g', label='KO', linewidth=3)
plt.title('ECDF spEPSC Inst. Fq.')
plt.legend()
plt.ylabel('P. Cum.')
plt.xlabel('Frequency (Hz)')
#%%
ecdf_infq.savefig('ecdf_instfq.svg')
stats.ks_2samp(all_ko, all_wt)

passive_df = Patch_data['Passive']
passive_df
passive_CA3 = passive_df[passive_df['Cell Type'] == 'CA3_Pyr']
passive_CA3.columns
#%%
pass_fig, axes = plt.subplots(3, 2, figsize=(4, 5))
sns.boxplot('Gen_type', 'Vm (mV)', data=passive_CA3, order=[
            'WT', 'KO'], palette=['b', 'g'], width=0.3, ax=axes[0, 0], boxprops=dict(alpha=.7))
axes[0,0].set_xlabel('')
sns.boxplot('Gen_type', 'Rn (Mohms)', data=passive_CA3, order=[
            'WT', 'KO'], palette=['b', 'g'], width=0.3, ax=axes[0, 1], boxprops=dict(alpha=.7))
axes[0,1].set_xlabel('')
sns.boxplot('Gen_type', 'Tau (ms)', data=passive_CA3, order=[
            'WT', 'KO'], palette=['b', 'g'], width=0.3, ax=axes[1, 0], boxprops=dict(alpha=.7))
axes[1,0].set_xlabel('')
sns.boxplot('Gen_type', 'Cm (pF)', data=passive_CA3, order=[
            'WT', 'KO'], palette=['b', 'g'], width=0.3, ax=axes[1, 1], boxprops=dict(alpha=.7))
axes[1,1].set_xlabel('')
sns.boxplot('Gen_type', 'Rheobase (pA)', data=passive_CA3, order=[
            'WT', 'KO'], palette=['b', 'g'], width=0.3, ax=axes[2, 0], boxprops=dict(alpha=.7))
axes[2,0].set_xlabel('')
sns.boxplot('Gen_type', 'Sag (mV)', data=passive_CA3, order=[
            'WT', 'KO'], palette=['b', 'g'], width=0.3, ax=axes[2, 1], boxprops=dict(alpha=.7))
axes[2,1].set_xlabel('')
plt.tight_layout()
#%%
pass_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/PDCB pics/paasive.png', dpi = 300)

passive_CA1 = passive_df[passive_df['Cell Type'] == 'CA1_Pyr']
#%%
passCA1_fig, axes = plt.subplots(3, 2, figsize=(10, 10))
sns.boxplot('Gen_type', 'Vm (mV)', data=passive_CA1, order=[
            'WT', 'KO'], palette='deep', width=0.3, ax=axes[0, 0])
sns.boxplot('Gen_type', 'Rn (Mohms)', data=passive_CA1, order=[
            'WT', 'KO'], palette='deep', width=0.3, ax=axes[0, 1])
sns.boxplot('Gen_type', 'Tau (ms)', data=passive_CA1, order=[
            'WT', 'KO'], palette='deep', width=0.3, ax=axes[1, 0])
sns.boxplot('Gen_type', 'Cm (pF)', data=passive_CA1, order=[
            'WT', 'KO'], palette='deep', width=0.3, ax=axes[1, 1])
sns.boxplot('Gen_type', 'Rheobase (pA)', data=passive_CA1, order=[
            'WT', 'KO'], palette='deep', width=0.3, ax=axes[2, 0])
sns.boxplot('Gen_type', 'Sag (mV)', data=passive_CA1, order=[
            'WT', 'KO'], palette='deep', width=0.3, ax=axes[2, 1])
#%%
passCA1_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/MEAs/MEA_figures/paasive_CA1.svg')

IV_df = Patch_data['IV']
IV_df = IV_df.drop(IV_df[IV_df['Current (pA)'] > 300].index)
CA3_mv = IV_df.filter(like='CA3')
CA3_mv
CA3_mv_means = []
CA3_mv_error = []
for index, row in CA3_mv.filter(like='WT').iterrows():
    CA3_mv_means.append(row.mean())
    CA3_mv_error.append(row.sem())
CA3KO_means = CA3_mv_means
CAKO_errors = CA3_mv_error
CA3WT_means = CA3_mv_means
CAWT_errors = CA3_mv_error

#%%
iv_fig = plt.figure(figsize=(10, 10))
ax = iv_fig.add_subplot(1, 1, 1)
ax.errorbar(x=IV_df['Current (pA)'], y=CA3WT_means, yerr=np.nan_to_num(
    np.array(CAWT_errors)), fmt='sb-', label='WT = 6', capsize = 3)
ax.errorbar(x=IV_df['Current (pA)'], y=CA3KO_means, yerr=np.nan_to_num(
    np.array(CAKO_errors)), fmt='go-', label='KO = 7', capsize = 3)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xlim(-300, 350)
ax.legend(loc=4)
ax.set_xlabel('Current (pA)')
ax.xaxis.set_label_coords(0.06,0.52)
ax.set_ylabel('Vm (mV)')
ax.yaxis.set_label_coords(0.5, .87)
#%%
iv_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/PDCB pics/IV_CA3', dpi = 300)

CA1_mv = IV_df.filter(like='CA1')
CA1_mv
CA1_mv_means = []
CA1_mv_error = []
for index, row in CA1_mv.filter(like='KO').iterrows():
    CA1_mv_means.append(row.mean())
    CA1_mv_error.append(row.sem())
CA1KO_means = CA1_mv_means
CA1KO_errors = CA1_mv_error

#%%
ivCA1_fig = plt.figure(figsize=(10, 10))
ax_ca1 = ivCA1_fig.add_subplot(1, 1, 1)
ax_ca1.errorbar(x=IV_df['Current (pA)'], y=CA1WT_means, yerr=np.nan_to_num(
    np.array(CA1WT_errors)), fmt='sb-', label='WT = 10')
ax_ca1.errorbar(x=IV_df['Current (pA)'], y=CA1KO_means, yerr=np.nan_to_num(
    np.array(CA1KO_errors)), fmt='go-', label='KO = 10')
ax_ca1.spines['left'].set_position('center')
ax_ca1.spines['bottom'].set_position('center')
ax_ca1.spines['right'].set_color('none')
ax_ca1.spines['top'].set_color('none')
ax_ca1.set_xlim(-500, 500)
ax_ca1.legend(loc=4)
ax_ca1.set_xlabel('Current (pA)')
ax_ca1.xaxis.set_label_coords(0.06,0.52)
ax_ca1.set_ylabel('Vm (mV)')
ax_ca1.yaxis.set_label_coords(0.5, .9)
#%%
ivCA1_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/MEAs/MEA_figures/IV_CA1.svg')

Firing_df = Patch_data['Firing']
Firing_df
#IV_df = IV_df.drop(IV_df[IV_df['Current (pA)'] > 300].index)
CA3_fire = Firing_df.filter(like='CA1')
CA3_fire
CA3_fire_means = []
CA3_fire_error = []
for index, row in CA3_fire.filter(like='KO').iterrows():
    CA3_fire_means.append(row.mean())
    CA3_fire_error.append(row.sem())
CA3KO_means = np.array(CA3_fire_means)/2
CAKO_errors = np.array(CA3_fire_error)/2
CA3WT_means = np.array(CA3_fire_means)/2
CAWT_errors = np.array(CA3_fire_error)/2
CA3KO_means
#%%
fire_fig = plt.figure(figsize=(4, 4))
ax = fire_fig.add_subplot(1, 1, 1)
ax.errorbar(x= Firing_df['Current (pA)'], y=CA3WT_means, yerr=np.nan_to_num(
    np.array(CAWT_errors)), fmt='sb-', label='WT = 6', capsize = 3)
ax.errorbar(x=Firing_df['Current (pA)'], y=CA3KO_means, yerr=np.nan_to_num(
    np.array(CAKO_errors)), fmt='go-', label='KO = 7', capsize = 3)
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#ax.set_xlim(-0, 400)
ax.legend(loc=4)
ax.set_xlabel('Current (pA)')
#ax.xaxis.set_label_coords(0.06,0.52)
ax.set_ylabel('#AP')
plt.tight_layout()
#%%
fire_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/PDCB pics/IO_CA1.png', dpi = 300)
CA3WT_means

fepsp_df = pd.read_excel('/Users/felipeantoniomendezsalcido/Desktop/Field Potentials.xlsx', sheet_name = None)

fepsp_df
LTP = fepsp_df['LTP']
LTP
inout_means = []
inout_error = []

for index, row in In_Out.iterrows():
    inout_means.append(row.mean())
    inout_error.append(row.sem())

i = 0
ii = 4
step_mean = []
step_error = []
for h in range(len(LTP['021018_WT'])):
        step_mean.append(pd.Series(LTP['021018_WT'][i:ii]).mean())
        step_error.append(pd.Series(LTP['021018_WT'][i:ii]).sem())
        i +=4
        ii+=4

time = np.arange(-30, 90)
len(time)
basal = pd.Series(np.abs(step_mean[0:31])).mean()

ltp_points = np.abs((pd.Series(step_mean).dropna()))/basal
ltp_err = np.abs((pd.Series(step_error).dropna()))
#%%
ltp_fig = plt.figure(figsize=(12,6))
ax_inout = ltp_fig.add_subplot(1,1,1)
ax_inout.errorbar(x = time, y = ltp_points, yerr = ltp_err, fmt = 'ko-')
ax_inout.set_xlabel('Tiempo (Seg)')
ax_inout.set_ylabel('% Basal')
#%%

ltp_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/MEAs/MEA_figures/LTP.svg')

InOut = fepsp_df['InOut']

Iout = InOut['081018_WT']
i = 0
ii = 4
in_out_mean = []
in_out_error = []
for h in range(len(Iout)):
        in_out_mean.append(Iout[i:ii].mean())
        in_out_error.append(Iout[i:ii].sem())
        i +=4
        ii+=4

out_put = pd.Series(in_out_mean).dropna()
out_err = pd.Series(in_out_error).dropna()
len(out_put)
curent_in = np.arange(0, 575, 25)
#%%
inout_fig = plt.figure(figsize=(10,10))
ax_if = inout_fig.add_subplot(1,1,1)
ax_if.errorbar(x = curent_in, y = np.abs(out_put), yerr = out_err, fmt = 'ko-')
ax_if.set_xlabel('Corriente (pA)')
ax_if.set_ylabel('Pendiente (mV/ms)')
#%%
inout_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/MEAs/MEA_figures/InOut.svg')
