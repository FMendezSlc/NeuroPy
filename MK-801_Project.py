import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

mk_raw = pd.read_csv('/Users/felipeantoniomendezsalcido/Desktop/Data/MK_Protocol_Log.csv')

mk_wt_df = mk_raw.groupby(['Genotype', 'Treatment', 'Day P( )']).mean().reset_index()

mk_wt_sem = mk_raw.groupby(['Genotype', 'Treatment', 'Day P( )']).sem().reset_index()
days = mk_wt_df['Day P( )'].unique()

KO_Sl = mk_wt_df[(mk_wt_df['Genotype'] == 'KO') & (mk_wt_df['Treatment'] == 'Saline')]['Weight (g)']
KO_Sl_err = mk_wt_sem[(mk_wt_sem['Genotype'] == 'KO') & (
    mk_wt_sem['Treatment'] == 'Saline')]['Weight (g)']
KO_Mk = mk_wt_df[(mk_wt_df['Genotype'] == 'KO') & (mk_wt_df['Treatment'] == 'MK-801')]['Weight (g)']
KO_Mk_err = mk_wt_sem[(mk_wt_sem['Genotype'] == 'KO') & (
    mk_wt_sem['Treatment'] == 'MK-801')]['Weight (g)']
WT_Sl = mk_wt_df[(mk_wt_df['Genotype'] == 'WT') & (mk_wt_df['Treatment'] == 'Saline')]['Weight (g)']
len(WT_Sl)
WT_Mk_err
WT_Sl_err = mk_wt_sem[(mk_wt_sem['Genotype'] == 'WT') & (
    mk_wt_sem['Treatment'] == 'Saline')]['Weight (g)']
WT_Mk = mk_wt_df[(mk_wt_df['Genotype'] == 'WT') & (mk_wt_df['Treatment'] == 'MK-801')]['Weight (g)']
WT_Mk_err = mk_wt_sem[(mk_wt_sem['Genotype'] == 'WT') & (
    mk_wt_sem['Treatment'] == 'MK-801')]['Weight (g)']

#%%
weigth_fig = plt.figure()
plt.errorbar(x=days, y=KO_Sl, yerr=KO_Sl_err, fmt='o-',
             color='forestgreen', capsize=2, label='KO-Saline')
plt.errorbar(x=days, y=KO_Mk, yerr=KO_Mk_err, fmt='v--',
             color='forestgreen', capsize=2, label='KO-MK-801')
plt.errorbar(x=days[0:11], y=WT_Sl, yerr=WT_Sl_err, fmt='o-',
             color='royalblue', capsize=2, label='WT-Saline')
plt.errorbar(x=days[0:11], y=WT_Mk, yerr=WT_Mk_err, fmt='v--',
             color='royalblue', capsize=2, label='WT-MK-801')
plt.xticks(range(5, 19))
plt.xlabel('Postnatal Day')
plt.ylabel('Weigth (g)')
plt.axvline(9, color='k', linestyle=':')
plt.legend()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#%%
