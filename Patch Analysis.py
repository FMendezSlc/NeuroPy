import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

Patch_data = pd.read_excel(
    '/Users/felipeantoniomendezsalcido/Desktop/Patch Analysis.xlsx', sheet_name=None)

Patch_data.keys()
Patch_data['sp_Peaks']


def ecdf(raw_data):
    '''[np.array -> tuple]
    Equivalent to R's ecdf(). Credit to Kripanshu Bhargava from Codementor'''
    cdfx = np.sort(raw_data.unique())
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


spEPSC = Patch_data['sp_Peaks']

spEPSC.filter(like='WT')
cols = []
means = []
for column in spEPSC.filter(like='KO'):
    cols.append(spEPSC[column].dropna())
    means.append(spEPSC[column].mean())
    merged = pd.concat(cols, ignore_index=True)
means_wt = means
means_ko = means
means_ko
means_df = pd.DataFrame([means_wt, means_ko], ['WT', 'KO']).T
means_df
cols
all_wt = np.abs(merged.round(2))
all_ko = np.abs(merged_ko.round(2))
all_wt
KO_ecdf = ecdf(all_ko)
WT_ecdf = ecdf(all_wt)

#%%
ecdf_epsc = plt.figure(figsize=(10, 8))
plt.plot(WT_ecdf[0], WT_ecdf[1], color='b', label='WT', linewidth=3)
plt.plot(KO_ecdf[0], KO_ecdf[1], color='g', label='KO', linewidth=3)
plt.axes([.6, .2, .3, .3])
plt.axis('off')
sns.boxplot(y=['WT', 'KO'], data=means_df)
plt.title('ECDF spEPSC')
plt.legend()
plt.ylabel('P. Cum.')
plt.xlabel('Amplitude (pA)')
#%%
os.chdir('/Users/felipeantoniomendezsalcido/Desktop')
ecdf_epsc.savefig('ecdf_epsc.svg')
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
pass_fig, axes = plt.subplots(3, 2, figsize=(10, 10))
sns.boxplot('Gen_type', 'Vm (mV)', data=passive_CA3, order=[
            'WT', 'KO'], palette='deep', width=0.3, ax=axes[0, 0])
sns.boxplot('Gen_type', 'Rn (Mohms)', data=passive_CA3, order=[
            'WT', 'KO'], palette='deep', width=0.3, ax=axes[0, 1])
sns.boxplot('Gen_type', 'Tau (ms)', data=passive_CA3, order=[
            'WT', 'KO'], palette='deep', width=0.3, ax=axes[1, 0])
sns.boxplot('Gen_type', 'Cm (pF)', data=passive_CA3, order=[
            'WT', 'KO'], palette='deep', width=0.3, ax=axes[1, 1])
sns.boxplot('Gen_type', 'Rheobase (pA)', data=passive_CA3, order=[
            'WT', 'KO'], palette='deep', width=0.3, ax=axes[2, 0])
sns.boxplot('Gen_type', 'Sag (mV)', data=passive_CA3, order=[
            'WT', 'KO'], palette='deep', width=0.3, ax=axes[2, 1])
#%%
pass_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/MEAs/MEA_figures/paasive.svg')

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
CA3_mv = IV_df.filter(like='CA3')
CA3_mv
CA3_mv_means = []
CA3_mv_error = []
for index, row in CA3_mv.filter(like='KO').iterrows():
    CA3_mv_means.append(row.mean())
    CA3_mv_error.append(row.sem())
CA3KO_means = CA3_mv_means
CAKO_errors = CA3_mv_error

#%%
iv_fig = plt.figure(figsize=(10, 10))
ax = iv_fig.add_subplot(1, 1, 1)
ax.errorbar(x=IV_df['Current (pA)'], y=CA3WT_means, yerr=np.nan_to_num(
    np.array(CAWT_errors)), fmt='sb-', label='WT')
ax.errorbar(x=IV_df['Current (pA)'], y=CA3KO_means, yerr=np.nan_to_num(
    np.array(CAKO_errors)), fmt='go-', label='KO')
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xlim(-500, 500)
ax.legend(loc=4)
#%%
iv_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/MEAs/MEA_figures/IV_CA3')
