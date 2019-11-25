import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import pingouin as pg

# Training sessions analysis
raw_HB = pd.read_csv('/Users/felipeantoniomendezsalcido/Desktop/HW_training_2.csv')
raw_HB
HB_group
HB_group = raw_HB.groupby(['Session', 'ID']).sum()
HB_WT = HB_group.filter(like='WT', axis=0)
HB_KO = HB_group.filter(like='KO', axis=0)

WT_means = []
WT_error = []

for Session, new_df in HB_WT.groupby(level=0):
    WT_means.append(new_df['Effective_time'].mean())
    WT_error.append(new_df['Effective_time'].sem())

KO_means = []
KO_error = []

for Session, new_df in HB_KO.groupby(level=0):
    KO_means.append(new_df['Effective_time'].mean())
    KO_error.append(new_df['Effective_time'].sem())


#%%
trainig_fig = plt.figure(figsize=(8, 6))
tr_plot = trainig_fig.add_subplot(1, 1, 1)
tr_plot.errorbar(x=np.arange(1, 7), y=WT_means, yerr=WT_error, fmt='bo-', label='WT = 3')
tr_plot.errorbar(x=np.arange(1, 7), y=KO_means, yerr=KO_error, fmt='go-', label='KO = 4')
tr_plot.axhline(y=60, color='k', linestyle='--', label='Performance threshold')
tr_plot.set_xlabel('Session')
tr_plot.set_ylabel('Cummulative Time (Seg)')
tr_plot.legend()
plt.show()
#%%
trainig_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/Training')

# Hebb-Williams Tests
HB4 = pd.read_csv('/Users/felipeantoniomendezsalcido/Desktop/H-W_tests.csv')
HB4
HB4['ID'].unique()
# Total errors Analysis
total_err = HB4.groupby(['ID']).sum()
total_err

err_WT = total_err.filter(like='WT', axis=0)
err_KO = total_err.filter(like='KO', axis=0)

WTer_means = []
WTer_error = []

for Session, new_df in err_WT.groupby(level=0):
    WTer_means.append(new_df['Error Count'].mean())
    WTer_error.append(new_df['Error Count'].sem())

KOer_means = []
KOer_error = []

for Session, new_df in err_KO.groupby(level=0):
    KOer_means.append(new_df['Error Count'].mean())
    KOer_error.append(new_df['Error Count'].sem())


#%%
pb4_fig = plt.figure(figsize=(12, 8))
pb4_plot = pb4_fig.add_subplot(1, 1, 1)
pb4_plot.errorbar(x=np.arange(1, 4), y=WTer_means, yerr=WTer_error, fmt='bo-', label='WT = 4')
pb4_plot.errorbar(x=np.arange(1, 4), y=KOer_means, yerr=KOer_error, fmt='go-', label='KO = 3')
pb4_plot.set_xlabel('Session')
pb4_plot.set_ylabel('Total Errors')
pb4_plot.legend()
#%%

# Kesner's coefficients analysis
coef_df = HB4.groupby(['Session', 'ID'])
cof_dic = {'Session': [], 'ID': [], 'E1-5': [], 'E6-10': []}
for ii in coef_df.groups.keys():
    cof_dic['Session'].append(ii[0])
    cof_dic['ID'].append(ii[1])
    cof_dic['E1-5'].append(HB4['Error Count'].loc[coef_df.groups[ii].values[0:5]].sum())
    cof_dic['E6-10'].append(HB4['Error Count'].loc[coef_df.groups[ii].values[5:]].sum())

cof_df = pd.DataFrame(cof_dic)
K_cof = {'ID': [], 'WD': [], 'BD': []}
for ii in cof_df['ID'].unique():
    K_cof['ID'].append(ii)
    A = (cof_df['E1-5'][(cof_df['ID'] == ii) & (cof_df['Session'] == 1)].values)
    B = (cof_df['E6-10'][(cof_df['ID'] == ii) & (cof_df['Session'] == 1)].values)
    C = (cof_df['E1-5'][(cof_df['ID'] == ii) & (cof_df['Session'] == 2)].values)
    D = (cof_df['E6-10'][(cof_df['ID'] == ii) & (cof_df['Session'] == 2)].values)
    E = (cof_df['E1-5'][(cof_df['ID'] == ii) & (cof_df['Session'] == 3)].values)
    WD = (((A - B) + (C - D)) / 2).item()
    BD = (((B - C) + (D - E)) / 2).item()
    K_cof['WD'].append(WD)
    K_cof['BD'].append(BD)

Ks_df = pd.DataFrame(K_cof)
Ks_df['Gen_type'] = ['KO', 'KO', 'WT', 'WT', 'WT', 'WT', 'KO']
Ks_df

#%%
K_fig, axes = plt.subplots(1, 2, figsize=(10, 10))
sns.barplot(x='Gen_type', y='WD', data=Ks_df, palette='deep', order=['WT', 'KO'], ax=axes[0])
sns.barplot(x='Gen_type', y='BD', data=Ks_df, palette='deep', order=['WT', 'KO'], ax=axes[1])
#%%

cof_df[['E1-5', 'E6-10']][(cof_df['ID'] == 'KO_1') & (cof_df['Session'] == 1)].values

multi_err = cof_df.set_index(['ID', 'Session']).sort_index()

names = ['WT_0', 'WT_1', 'WT_2', 'WT_3', 'KO_1', 'KO_2', 'KO_3']

err_t = []
err_t
for ii in names:
    a = multi_err.loc[(ii, 1)].values
    b = multi_err.loc[(ii, 2)].values
    c = multi_err.loc[(ii, 3)].values
    err_t.append(np.concatenate((a, b, c)))
err_t

sns.barplot(x='Session', y='Time elapsed', hue='Gene_type', data=HB4, palette='deep')

for ii in list(HB4.groupby(['ID', 'Session']).groups.keys()):
    print(ii[0].split('_')[0])

by_tr = HB4.groupby(['Gene_type', 'Session', 'Trial']).std()
sns.factorplot(x='Trial', y='Error Count', hue='Gene_type',
               col='Session', data=by_tr.reset_index(), palette='deep')
HB4[HB4['Gene_type'] == 'WT']

mice = [0, 1, 2, 3, 4, 5]

np.random.choice(mice, size=(6), replace=False)

#All in all (individuals, mazes, trials)
HB4


All_points = sns.catplot(x='Trial', y='Error Count', col='Maze', hue='Gene_type',
                         data=HB4, col_wrap=3, palette=sns.color_palette(['green', 'blue']), dodge=True)
poitn_plotHW = sns.catplot(x='Trial', y='Error Count', col='Maze', hue='Gene_type',
                           data=HB4, col_wrap=3, kind='point', palette=sns.color_palette(['green', 'blue']))
All_points.savefig('/Users/felipeantoniomendezsalcido/Desktop/all_pints_HW.png')
poitn_plotHW.savefig('/Users/felipeantoniomendezsalcido/Desktop/point_plot_HW.png')

a = range(0, 6)
np.

np.log10((.1, .2, .5, .8, 1))


# Hebb-Williams-Kesner Group
# The good groups

HWK_data = pd.read_csv('/Users/felipeantoniomendezsalcido/Desktop/Data/HWK_1.csv')

HWK_data[HWK_data['Task Phase'] == 'Training_1']
# get just the testing phases
HWK_errors = HWK_data.drop(HWK_data[HWK_data['Task Phase'].str.contains(
    'Training')].index)  # remove training trials

HWK_errors
# Way better to get blocks
for ii in HWK_errors['Task Phase'].unique():
    HWK_errors.loc[(HWK_errors['Task Phase'] == ii) & (
        HWK_errors['Trial'] <= 5), 'Block'] = f"D{ii.split('_')[-1]}_1"
    HWK_errors.loc[(HWK_errors['Task Phase'] == ii) & (
        HWK_errors['Trial'] > 5), 'Block'] = f"D{ii.split('_')[-1]}_2"

HWK_err_gpd = HWK_errors.groupby(['Gene_type', 'Sex', 'ID', 'Task Phase', 'Block'], as_index=False)[
    'Error Count'].agg({'Error Sum': 'sum', 'Error mean': 'mean'})  # looks like it worked
HWK_err_gpd
# now a proper group vairable
HWK_err_gpd['Group'] = HWK_err_gpd['Sex'].astype(str).str[0] + HWK_err_gpd['Gene_type']  # nice!!

HWK_err_gpd['Subject'] = HWK_err_gpd['Sex'].astype(str).str[0] + HWK_err_gpd['ID']
HWK_err_gpd
HWK_errors.to_csv('/Users/felipeantoniomendezsalcido/Desktop/HWK_tidy.csv')

HWK_lat_gpd
HWK_lat_gpd = HWK_errors.groupby(['Gene_type', 'Sex', 'ID', 'Task Phase', 'Block'], as_index=False)[
    'Time elapsed'].agg({'Latency mean': 'mean'})
HWK_lat_gpd['Group'] = HWK_err_gpd['Sex'].astype(str).str[0] + HWK_lat_gpd['Gene_type']  # nice!!

HWK_lat_gpd['Subject'] = HWK_err_gpd['Sex'].astype(str).str[0] + HWK_lat_gpd['ID']

# let's see if this works in a point plot
#%%
HW_figure,  (err_ax, lat_ax) = plt.subplots(nrows=2, figsize = (7, 5))
sns.pointplot(x='Block', y='Error mean', hue='Group', hue_order=['FKO', 'MKO', 'FWT', 'MWT'], data=HWK_err_gpd, kind='point', scale = 0.8, ci=68, capsize=.1, palette=[
                   'g', 'g', 'b', 'b'], linestyles=['--', '-', '--', '-'], markers=['v', 'o', 'v', 'o'], errwidth=2, dodge=True, ax = err_ax)
sns.despine()
err_ax.axvline(5, 0, 1, color = 'k', ls = '--')
err_ax.text(0, 1, '*', fontsize = 14)
leg_handles = err_ax.get_legend_handles_labels()[0]
err_ax.legend(leg_handles, ['Fem -/-', 'Male -/-', 'Fem +/+', 'Male +/+'], loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox = True, shadow=False)
err_ax.set_ylabel('Mean Errors')
err_ax.set_xlabel('')

sns.pointplot(x='Block', y='Latency mean', hue='Group', hue_order=['FKO', 'MKO', 'FWT', 'MWT'], data=HWK_lat_gpd, kind='point', scale = 0.8, ci=68, capsize=.1, palette=[
                       'g', 'g', 'b', 'b'], linestyles=['--', '-', '--', '-'], markers=['v', 'o', 'v', 'o'], errwidth=2, dodge=True, ax = lat_ax)
sns.despine()
lat_ax.axvline(5, 0, 1, color = 'k', ls = '--')
lat_ax.text(0, 8, '*', fontsize = 14)
leg_handles = lat_ax.get_legend_handles_labels()[0]
lat_ax.legend_.remove()
lat_ax.set_ylabel('Mean Latency (s)')
lat_ax.set_xlabel('Trial Block')
#%%
HW_figure.savefig('/Users/felipeantoniomendezsalcido/Desktop/PDCB pics/HWK_modified.png', dpi= 300)
# OK all that worked
HWK_err_gpd
HWK_short = HWK_err_gpd.drop(HWK_err_gpd[HWK_err_gpd['Task Phase'].isin(['Testing_4', 'Testing_5'])].index)
HWK_short
HWK_short_lat = HWK_lat_gpd.drop(HWK_lat_gpd[HWK_lat_gpd['Task Phase'].isin(['Testing_4', 'Testing_5'])].index)
#%%
HW_figure,  (err_ax, lat_ax) = plt.subplots(nrows=2, figsize = (7, 5))
sns.pointplot(x='Block', y='Error mean', hue='Group', hue_order=['FKO', 'MKO', 'FWT', 'MWT'], data=HWK_short, kind='point', scale = 0.8, ci=68, capsize=.1, palette=[
                   'g', 'g', 'b', 'b'], linestyles=['--', '-', '--', '-'], markers=['v', 'o', 'v', 'o'], errwidth=2, dodge=True, ax = err_ax)
sns.despine()
plt.axvline(3, 0, 1)
leg_handles = err_ax.get_legend_handles_labels()[0]
err_ax.legend(leg_handles, ['Fem -/-', 'Male -/-', 'Fem +/+', 'Male +/+'], loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox=True, shadow=True)
err_ax.set_ylabel('Mean Errors')
err_ax.set_xlabel('')

sns.pointplot(x='Block', y='Latency mean', hue='Group', hue_order=['FKO', 'MKO', 'FWT', 'MWT'], data=HWK_short_lat, kind='point', scale = 0.8, ci=68, capsize=.1, palette=[
                       'g', 'g', 'b', 'b'], linestyles=['--', '-', '--', '-'], markers=['v', 'o', 'v', 'o'], errwidth=2, dodge=True, ax = lat_ax)
sns.despine()
leg_handles = lat_ax.get_legend_handles_labels()[0]
lat_ax.legend_.remove()
lat_ax.set_ylabel('Mean Latency (s)')
lat_ax.set_xlabel('Trial Block')
#%%
# Now pingouin ANOVA
# pg.friedman(data = HWK_err_gpd, dv = 'Error mean', within = 'Block', subject = 'Subject')
# pg.rm_anova(data=HWK_err_gpd, dv='Error mean', within=['Block', 'Group'],  subject='Subject', detailed = True)
pg.mixed_anova(data=HWK_short, dv='Error mean', within='Block',
               between='Group', subject='Subject')


bonf = pg.pairwise_ttests(data=HWK_short, dv='Error mean', within='Block', between='Group', subject='Subject', alpha=0.05, padjust='holm', return_desc=True)

bonf

#
# Kesner indexes with mean erros per block
subjects = list(HWK_errors['Subject'].unique())
index_dict = {'Sex': [], 'Genotype': [], 'Group': [],
              'Subject': [], 'Acquisition': [], 'Retrival': []}

for ii in subjects:
    index_dict['Group'].append(HWK_errors['Group'][HWK_errors['Subject'] == ii].unique().item())
    index_dict['Sex'].append(HWK_errors['Sex'][HWK_errors['Subject'] == ii].unique().item())
    index_dict['Genotype'].append(
        HWK_errors['Gene_type'][HWK_errors['Subject'] == ii].unique().item())
    index_dict['Subject'].append(ii)

    d1a = HWK_errors['Error mean'][(HWK_errors['Subject'] == ii) &
                                   (HWK_errors['Block'] == 'D1_1')].item()
    d1b = HWK_errors['Error mean'][(HWK_errors['Subject'] == ii) &
                                   (HWK_errors['Block'] == 'D1_2')].item()
    d2a = HWK_errors['Error mean'][(HWK_errors['Subject'] == ii) &
                                   (HWK_errors['Block'] == 'D2_1')].item()
    d2b = HWK_errors['Error mean'][(HWK_errors['Subject'] == ii) &
                                   (HWK_errors['Block'] == 'D2_2')].item()
    d3a = HWK_errors['Error mean'][(HWK_errors['Subject'] == ii) &
                                   (HWK_errors['Block'] == 'D3_1')].item()

    index_dict['Acquisition'].append(((d1a - d1b) + (d2a - d2b)) / 2)
    index_dict['Retrival'].append(((d1b - d2a) + (d2b - d3a)) / 2)

kesner_index = pd.DataFrame(index_dict)
kesner_index
pg.anova(dv='Acquisition', between=['Genotype', 'Sex'],
         data=kesner_index, export_filename='kesneraov_acq')
acq_tk = pg.pairwise_tukey(dv='Acquisition', between='Group', data=kesner_index)
pg.anova(dv='Retrival', between=['Genotype', 'Sex'],
         data=kesner_index, export_filename='kesneraov_ret')

acq_tk
#  Kesner indexes figure
#%%
kesner_ind = plt.figure(figsize=(9, 5))
plt.subplot(1, 2, 1)
acq_ax = sns.barplot(x='Group', y='Acquisition', data=kesner_index,
                     ci=68, capsize=.3, palette=['g', 'g', 'b', 'b'])
plt.xticks(range(0, 4), ['Fem_KO', 'Male_KO', 'Fem_WT', 'Male_WT'])
acq_ax.annotate('*', xy=(0.5, .93), xytext=(0.5, .91), xycoords='axes fraction', fontsize=18, ha='center',
                va='bottom', fontweight='bold', arrowprops=dict(arrowstyle='-[, widthB=6, lengthB=.1', lw=2, color='black'))
acq_ax.annotate('**', xy=(0.25, .83), xytext=(0.25, .81), xycoords='axes fraction', fontsize=18, ha='center',
                va='bottom', fontweight='bold', arrowprops=dict(arrowstyle='-[, widthB=2, lengthB=.1', lw=2, color='black'))
plt.yticks(range(0, 8))
plt.ylim(0, 8)
plt.ylabel('Acquisition Index')
plt.subplot(1, 2, 2)
sns.barplot(x='Group', y='Retrival', data=kesner_index,
            ci=68, capsize=.3, palette=['g', 'g', 'b', 'b'])
plt.yticks(range(0, 7))
plt.ylim(-0.3, 6.5)
plt.ylabel('Retrival Index')
plt.xticks(range(0, 4), ['Fem_KO', 'Male_KO', 'Fem_WT', 'Male_WT'])
plt.tight_layout()
#%%
kesner_ind.savefig('/Users/felipeantoniomendezsalcido/Desktop/HWK_ind.png')

HWK_errors.groupby(['Gene_type', 'Sex', 'Task Phase', 'Block'])['Error Sum'].mean()
type(Err_sum_mean)

Err_sum_sem = HWK_errors['Error Sum'].groupby(['Gene_type', 'Sex', 'Task Phase', 'Block']).sem()


# Fig Analisis de errores: Suma total por bloque
#%%
Total_err = plt.figure(figsize=(10, 6))
plt.errorbar(x=np.arange(1, 7), y=Err_sum_mean['KO', 'Female'].values,
             yerr=Err_sum_sem['KO', 'Female'].values, fmt='go--', capsize=3, label='KO_Fem')
plt.errorbar(x=np.arange(1, 7), y=Err_sum_mean['KO', 'Male'].values,
             yerr=Err_sum_sem['KO', 'Male'].values, fmt='go-', capsize=3, label='KO_Male')
plt.errorbar(x=np.arange(1, 7), y=Err_sum_mean['WT', 'Female'].values,
             yerr=Err_sum_sem['WT', 'Female'].values, fmt='bo--', capsize=3, label='WT_Fem')
plt.errorbar(x=np.arange(1, 7), y=Err_sum_mean['WT', 'Male'].values,
             yerr=Err_sum_sem['WT', 'Male'].values, fmt='bo-', capsize=3, label='WT_Male')
plt.ylabel('Errores Totales')
plt.xlabel('Bloque')
plt.legend()
plt.xticks(np.arange(1, 7), ['D1_1', 'D1_2', 'D2_1', 'D2_2', 'D3_1', 'D3_2'])
plt.title('Errores Totales por Bloque', fontsize=12)
#%%
Total_err.savefig('/Users/felipeantoniomendezsalcido/Desktop/Errores_total.png')

Err_mean_mean = HWK_errors['Error mean'].groupby(['Gene_type', 'Sex', 'Task Phase', 'Block']).mean()


Err_mean_sem = HWK_errors['Error mean'].groupby(['Gene_type', 'Sex', 'Task Phase', 'Block']).sem()


#%%
Mean_err = plt.figure(figsize=(10, 6))
plt.errorbar(x=np.arange(1, 7), y=Err_mean_mean['KO', 'Female'].values,
             yerr=Err_mean_sem['KO', 'Female'].values, fmt='go--', capsize=3, label='KO_Fem')
plt.errorbar(x=np.arange(1, 7), y=Err_mean_mean['KO', 'Male'].values,
             yerr=Err_mean_sem['KO', 'Male'].values, fmt='go-', capsize=3, label='KO_Male')
plt.errorbar(x=np.arange(1, 7), y=Err_mean_mean['WT', 'Female'].values,
             yerr=Err_mean_sem['WT', 'Female'].values, fmt='bo--', capsize=3, label='WT_Fem')
plt.errorbar(x=np.arange(1, 7), y=Err_mean_mean['WT', 'Male'].values,
             yerr=Err_mean_sem['WT', 'Male'].values, fmt='bo-', capsize=3, label='WT_Male')
plt.ylabel('Errores Promedio')
plt.xlabel('Bloque')
plt.legend()
plt.xticks(np.arange(1, 7), ['D1_1', 'D1_2', 'D2_1', 'D2_2', 'D3_1', 'D3_2'])
plt.title('Promedio de Errores por Bloque', fontsize=12)
#%%

Mean_err.savefig('/Users/felipeantoniomendezsalcido/Desktop/Errores_promedio.png')

HWK_data
# Latency Analysis
HWK_latency = HWK_data.groupby(['Gene_type', 'Sex', 'ID', 'Task Phase', 'Block'])[
    'Time elapsed'].agg({'Latency': 'mean'})


HWK_latency = HWK_latency.drop(['Training_1', 'Training_2', 'Training_3'], level='Task Phase')

HWK_latency.reset_index()

Latency_mean = HWK_latency['Latency'].groupby(['Gene_type', 'Sex', 'Task Phase', 'Block']).mean
Latency_sem = HWK_latency['Latency'].groupby(['Gene_type', 'Sex', 'Task Phase', 'Block']).sem()

#%% Latency plot
Latency_fig = plt.figure(figsize=(10, 6))
plt.errorbar(x=np.arange(1, 7), y=Latency_mean['KO', 'Female'].values,
             yerr=Latency_sem['KO', 'Female'].values, fmt='go--', capsize=3, label='KO_Fem')
plt.errorbar(x=np.arange(1, 7), y=Latency_mean['KO', 'Male'].values,
             yerr=Latency_sem['KO', 'Male'].values, fmt='go-', capsize=3, label='KO_Male')
plt.errorbar(x=np.arange(1, 7), y=Latency_mean['WT', 'Female'].values,
             yerr=Latency_sem['WT', 'Female'].values, fmt='bo--', capsize=3, label='WT_Fem')
plt.errorbar(x=np.arange(1, 7), y=Latency_mean['WT', 'Male'].values,
             yerr=Latency_sem['WT', 'Male'].values, fmt='bo-', capsize=3, label='WT_Male')
plt.ylabel('Latencia')
plt.xlabel('Bloque')
plt.legend()
plt.xticks(np.arange(1, 7), ['D1_1', 'D1_2', 'D2_1', 'D2_2', 'D3_1', 'D3_2'])
plt.title('Latencia por Bloque', fontsize=12)
#%%

Latency_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/Latencias.png')

# Verification HWK_data[(HWK_data['ID'] == 'KO_0') & (HWK_data['Sex'] == 'Female')]

# Latency and error count Correlation
#%%
Lat_err = plt.figure(figsize=(10, 6))
sns.scatterplot(x='Error Count', y='Time elapsed', hue='Gene_type',
                data=HWK_data, palette=['g', 'b'], x_jitter=.9)
plt.title('Correlación Latencia-Errores')
plt.xlabel('Errores por ensayo')
plt.ylabel('Latencia (S)')
#%%
Lat_err.savefig('/Users/felipeantoniomendezsalcido/Desktop/Latencia_error_correlacion.png')

# Open Field Analysis

OF_data = pd.read_csv('/Users/felipeantoniomendezsalcido/Desktop/Data/OF_summary_report.csv')

OF_data['% Center'] = OF_data['Time in Zone (%) - Zone 6'] + OF_data['Time in Zone (%) - Zone 7'] + \
    OF_data['Time in Zone (%) - Zone 10'] + OF_data['Time in Zone (%) - Zone 11']

OF_data['% Periphery'] = OF_data['Time in Zone (%) - Zone 1'] + OF_data['Time in Zone (%) - Zone 2'] + OF_data['Time in Zone (%) - Zone 3'] + OF_data['Time in Zone (%) - Zone 4'] + OF_data['Time in Zone (%) - Zone 5'] + \
    OF_data['Time in Zone (%) - Zone 8'] + OF_data['Time in Zone (%) - Zone 13'] + OF_data['Time in Zone (%) - Zone 14'] + \
    OF_data['Time in Zone (%) - Zone 15'] + OF_data['Time in Zone (%) - Zone 16']
OF_data['% Total'] = OF_data['% Center'] + OF_data['% Periphery']
OF_data['Subject Group'] = OF_data['Gender'].astype(str).str[0] + OF_data['Subject Group']

OF_data
# Open Field Statistics
pg.anova(dv='Total Distance', between=['Gender', 'Genotype'],
         data=OF_data, detailed=True, export_filename='OFaov_distance')
OFdist_tk = pg.pairwise_tukey(
    dv='Total Distance', between='Subject Group', data=OF_data, alpha=0.05)
OFdist_tk
pg.anova(dv='Zone Transition Number', between=[
         'Gender', 'Genotype'], data=OF_data, detailed=True, export_filename='OFaov_crosses')
OFcross_tk = pg.pairwise_tukey(dv='Zone Transition Number',
                               between='Subject Group', data=OF_data, alpha=0.05)
OFcross_tk
pg.anova(dv='% Center', between=['Gender', 'Genotype'],
         data=OF_data, detailed=True, export_filename='OFaov_center')
OFcenter_tk = pg.pairwise_tukey(dv='% Center', between='Subject Group', data=OF_data, alpha=0.05)
OFcenter_tk
pg.anova(dv='% Periphery', between=['Gender', 'Genotype'],
         data=OF_data, detailed=True, export_filename='OFaov_periphery')
OFper_tk = pg.pairwise_tukey(dv='% Periphery', between='Subject Group', data=OF_data, alpha=0.05)
OFper_tk
#%%
OF_plot = plt.figure(figsize=(8, 8))
dist_ax = plt.subplot(2, 2, 1)
sns.barplot(x='Gender', y='Total Distance', hue='Genotype', data=OF_data,
            palette=['forestgreen', 'royalblue'], ci=68, capsize=.1)
dist_ax.annotate('*', xy=(0.5, .93), xytext=(0.5, .91), xycoords='axes fraction', fontsize=18, ha='center',
                 va='bottom', fontweight='bold', arrowprops=dict(arrowstyle='-[, widthB=4.5, lengthB=.1', lw=2, color='black'))
dist_ax.annotate('*', xy=(.75, .84), xytext=(.75, .82), xycoords='axes fraction', fontsize=18, ha='center',
                 va='bottom', fontweight='bold', arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=.1', lw=2, color='black'))
plt.ylabel('Total Distance (cm)')
plt.ylim(0, 7000)
plt.yticks(range(0, 7000, 1000))
dist_ax.get_legend().remove()
cross_ax = plt.subplot(2, 2, 2)
sns.barplot(x='Gender', y='Zone Transition Number', hue='Genotype', data=OF_data,
            palette=['forestgreen', 'royalblue'], ci=68, capsize=.1)
plt.ylabel('Crosses')
center_ax = plt.subplot(2, 2, 3)
sns.barplot(x='Gender', y='% Center', hue='Genotype', data=OF_data,
            palette=['forestgreen', 'royalblue'], ci=68, capsize=.1)
plt.ylabel('Time in Arena Center (%)')
center_ax.annotate('*', xy=(0.5, .93), xytext=(0.5, .91), xycoords='axes fraction', fontsize=18, ha='center',
                   va='bottom', fontweight='bold', arrowprops=dict(arrowstyle='-[, widthB=4.5, lengthB=.1', lw=2, color='black'))
center_ax.annotate('***', xy=(0.25, .83), xytext=(0.25, .81), xycoords='axes fraction', fontsize=18, ha='center',
                   va='bottom', fontweight='bold', arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=.1', lw=2, color='black'))
center_ax.annotate('**', xy=(0.75, .83), xytext=(0.75, .81), xycoords='axes fraction', fontsize=18, ha='center',
                   va='bottom', fontweight='bold', arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=.1', lw=2, color='black'))
center_ax.annotate('***', xy=(0.5, .75), xytext=(0.5, .73), xycoords='axes fraction', fontsize=18, ha='center',
                   va='bottom', fontweight='bold', arrowprops=dict(arrowstyle='-[, widthB=2, lengthB=.1', lw=2, color='black'))
plt.ylim(0, 25)
plt.yticks(range(0, 25, 5))
center_ax.get_legend().remove()
peri_ax = plt.subplot(2, 2, 4)
sns.barplot(x='Gender', y='% Periphery', hue='Genotype', data=OF_data,
            palette=['forestgreen', 'royalblue'], ci=68, capsize=.1)
peri_ax.annotate('*', xy=(0.75, .85), xytext=(0.75, .83), xycoords='axes fraction', fontsize=18, ha='center',
                 va='bottom', fontweight='bold', arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=.1', lw=2, color='black'))
peri_ax.annotate('**', xy=(0.25, .85), xytext=(0.25, .83), xycoords='axes fraction', fontsize=18, ha='center',
                 va='bottom', fontweight='bold', arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=.1', lw=2, color='black'))
peri_ax.annotate('***', xy=(0.5, .93), xytext=(0.5, .91), xycoords='axes fraction', fontsize=18, ha='center',
                 va='bottom', fontweight='bold', arrowprops=dict(arrowstyle='-[, widthB=2, lengthB=.1', lw=2, color='black'))
plt.ylim(0, 100)
plt.yticks(range(0, 130, 20))
plt.ylabel('Time in Arena Periphery (%)')
peri_ax.get_legend().remove()
plt.tight_layout()
#%%
OF_plot.savefig('/Users/felipeantoniomendezsalcido/Desktop/OF_analysis.png')
OF_data

# Area Timm Analysis

timm_df = pd.read_csv('/Users/felipeantoniomendezsalcido/Desktop/Data/Timm Area.csv')
timm_df.columns
timm_df['Group'] = timm_df['Subject'].astype('str').str[0:3]
timm_df['Mean Area'][timm_df['Genotype'] == 'WT']
pg.ttest(x=timm_df['Mean Area'][timm_df['Genotype'] == 'WT'],
         y=timm_df['Mean Area'][timm_df['Genotype'] == 'KO'], paired=False)

#%%
timm_fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(7, 3))
timm_point = sns.pointplot(x='Level', y='Mean Area', hue='Genotype', data=timm_df, palette=[
                           'b', 'g'], capsize=.05, scale = .7, errorwidth=.05, ci=68, ax=a0, label=['a', 'b'])
a0.set_ylabel(r'Mean Area ($\mu$m$^2$)')
a0.set_xlabel('from Bregma (mm)')
a0.invert_xaxis()
sns.despine()
timm_point.get_legend().remove()
timm_fig.legend(loc='upper right', bbox_to_anchor=(.29, .93), ncol=1)
timm_total = sns.barplot(x='Genotype', y='Mean Area', data=timm_df,
                         palette=['b', 'g'], ax=a1, ci=68, capsize=0.05, errwidth = 1.5)
a1.set_ylabel(r'Mean Area ($\mu$m$^2$)')
a1.annotate('***', xy=(0.5, .98), xytext=(0.5, .96), xycoords='axes fraction', fontsize=18, ha='center',
            va='bottom', fontweight='bold', arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=.2', lw=1, color='black'))
#a1.set_ylim(0, 20000)
plt.tight_layout()
#%%
timm_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/Timm_fig.png', dpi = 300)
