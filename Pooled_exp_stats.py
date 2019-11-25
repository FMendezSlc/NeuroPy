import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import pingouin as pg

pooled_units = pd.read_csv('/Users/felipeantoniomendezsalcido/Desktop/MEAs/general_data_basal copy.csv')

pooled_units['FR_Bs'] = pooled_units['FR_Bs'].round(2)

pooled_units

exp_all = pooled_units.groupby(['Date', 'Gen_type', 'Sex']).agg(
    {'CV2': 'count', 'FR_Bs': 'mean'}).reset_index()
exp_all = exp_all.rename(columns={'CV2': 'Cell_Count'})
exp_all
pooled_units

#%%
gen_stats_fig = plt.figure(constrained_layout= True, figsize= (6, 4))
gs = gen_stats_fig.add_gridspec(2, 3)
f3_ax1 = gen_stats_fig.add_subplot(gs[0, 0])
f3_ax1.set_title('gs[0, 0]')
strip = sns.stripplot(x='Gen_type', y='Cell_Count', data=exp_all, jitter=0.05,
                      edgecolor='gray', linewidth=1, size=1, palette=['b', 'g'], ax = f3_ax1, marker = 'o')
boxp = sns.boxplot(x='Gen_type', y='Cell_Count', data=exp_all,
                   saturation=1, width=0.2, palette= ['b', 'g'], linewidth=1.5, ax = f3_ax1)
sns.despine()
plt.xlabel('')
plt.ylabel('Units')
strip.tick_params(labelsize=12)
plt.title('Units per Rec.', fontsize=12)
f3_ax2 = gen_stats_fig.add_subplot(gs[0, 1])
f3_ax2.set_title('gs[0, 1]')
strip_2 = sns.stripplot(x='Gen_type', y='Channel Count', data=cells_chanl,
                      palette=['b', 'g'], jitter=0.1, edgecolor='gray', size=1, linewidth=1, ax = f3_ax2)
sns.boxplot(x='Gen_type', y='Channel Count', data=cells_chanl,
            saturation=1, width=.2, palette= ['b', 'g'], linewidth=1.5, ax = f3_ax2)
sns.despine()
plt.xlabel('')
plt.ylabel('Active Channels')
strip .tick_params(labelsize=12)
plt.title('Active Channels per Rec.', fontsize=12)

f3_ax3 = gen_stats_fig.add_subplot(gs[0, 2])
f3_ax3.set_title('gs[0, 2]')
strip_3 = sns.stripplot(x='Gen_type', y='FR_Bs', data=mean_FR,
                      palette= ['b', 'g'], jitter=0.1, edgecolor='gray', size=1, linewidth=1, ax = f3_ax3)
sns.boxplot(x='Gen_type', y='FR_Bs', data=mean_FR,
            saturation=1, width=.2, palette=['b', 'g'], linewidth=1.5, fliersize = 1, ax = f3_ax3)
sns.despine()
plt.xlabel('')
plt.ylabel('Firing Rate(Hz)')
strip .tick_params(labelsize=12)
plt.title('Mean Firing Rate', fontsize=12)

f3_ax4 = gen_stats_fig.add_subplot(gs[1, :])
f3_ax4.set_title('gs[1, :]')
sns.regplot(x='Channel Count', y='Unit', data=units_again[units_again['Gen_type']  == 'KO'], color = 'g', ax = f3_ax4, scatter_kws={'s':15}, line_kws={'linewidth':2})
sns.regplot(x='Channel Count', y='Unit', data=units_again[units_again['Gen_type']  == 'WT'], color = 'b', ax = f3_ax4, scatter_kws={'s':15}, line_kws={'linewidth':2})
plt.tick_params(labelsize=10)
plt.xlabel('Active Channels', fontsize=12)
plt.ylabel('Units', fontsize=12)
plt.legend(labels = ['WT', 'KO'])
plt.title('Number of Units per Active Channels', fontsize = 10)
#%%
gen_stats_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/PDCB pics/general_sts.png', dpi= 300)
#%%
# Cell count by experiment by genotype
with sns.axes_style('whitegrid'):
    cell_count_plot = plt.figure(figsize=(12, 12))
    strip = sns.stripplot(x='Gen_type', y='Cell_Count', data=exp_all, jitter=0.05,
                          edgecolor='gray', linewidth=1, size=6.5, palette='deep')
    boxp = sns.boxplot(x='Gen_type', y='Cell_Count', data=exp_all,
                       saturation=1, width=0.3, palette='deep', linewidth=2.5)
    strip.legend(labels=['WT = 761', 'KO = 572'], fontsize=14)
    sns.despine()
    plt.xlabel('Genotipo', fontsize=16)
    plt.ylabel('Unidades', fontsize=16)
    strip.tick_params(labelsize=14)
    plt.title('Número de Unidades', fontsize=18)
#%%
exp_all[exp_all['Gen_type'] == 'WT']['Cell_Count'].sum()


os.getcwd()
os.chdir('/Users/felipeantoniomendezsalcido/Desktop/MEAs/MEA_figures')
cell_count_plot.savefig('Units_per_Exp.svg')

# Channels with Units (Active) per experiment
cells_chanl = pooled_units.groupby(['Date', 'Gen_type']).agg({'Channel': 'unique'}).reset_index()
cells_chanl['Gen_type']
chan_count = []
chan_count
for ii in cells_chanl['Channel']:
    chan_count.append(len(ii))
cells_chanl
cells_chanl['Channel Count'] = chan_count
# %%
with sns.axes_style('whitegrid'):
    channels_count = plt.figure(figsize=(12, 12))
    strip = sns.stripplot(x='Gen_type', y='Channel Count', data=cells_chanl,
                          palette='deep', jitter=0.1, edgecolor='gray', size=6.5, linewidth=1)
    sns.boxplot(x='Gen_type', y='Channel Count', data=cells_chanl,
                saturation=1, width=.3, palette='deep', linewidth=2.5)
    sns.despine()
    plt.xlabel('Genotipo', fontsize=16)
    plt.ylabel('Canales Activos', fontsize=16)
    strip .tick_params(labelsize=14)
    plt.title('Número de Canales Activos', fontsize=18)
    channels_count.savefig('Canales_Activos.svg')
#%%
# Mean Firing Rate
cells_chanl['Channel Count'][cells_chanl['Gen_type'] == 'WT']
sp.stats.ttest_ind(cells_chanl['Channel Count'][cells_chanl['Gen_type'] == 'KO'], cells_chanl['Channel Count'][cells_chanl['Gen_type'] == 'WT'])

mean_FR = pooled_units.groupby(['Date', 'Gen_type']).agg(
    {'FR_Bs': 'mean'}).reset_index()
#%%
with sns.axes_style('whitegrid'):
    mean_FRplot = plt.figure(figsize=(12, 12))
    strip = sns.stripplot(x='Gen_type', y='FR_Bs', data=mean_FR,
                          palette='deep', jitter=0.1, edgecolor='gray', size=6.5, linewidth=1)
    sns.boxplot(x='Gen_type', y='FR_Bs', data=mean_FR,
                saturation=1, width=.3, palette='deep', linewidth=2.5)
    sns.despine()
    plt.xlabel('Genotipo', fontsize=16)
    plt.ylabel('Frecuencia de Disparo Promedio (Hz)', fontsize=16)
    strip .tick_params(labelsize=14)
    plt.title('Frecuencia de Disparo', fontsize=18)
mean_FRplot.savefig('Frecuencia de Disparo.svg')
#%%

# Linear regression units per channel
units_per_chnl = pooled_units.groupby(['Date', 'Gen_type']).agg({'Channel': 'unique'}).reset_index()
chan_count = []
for ii in cells_chanl['Channel']:
    chan_count.append(len(ii))
units_per_chnl['Channel Count'] = chan_count
units_again = pooled_units.groupby(['Date', 'Gen_type']).agg({'Unit': 'count'}).reset_index()
units_again['Channel Count'] = units_per_chnl['Channel Count']
units_again

with sns.axes_style('whitegrid'):
    lm_plot = sns.lmplot(x='Channel Count', y='Unit', data=units_again,
                         hue='Gen_type', palette='deep', size=7, legend_out=False, aspect=1.5)
    plt.tick_params(labelsize=14)
    plt.xlabel('Canales Activos', fontsize=16)
    plt.ylabel('Número de Unidades', fontsize=16)
    plt.legend(loc=4)
    new_title = 'Unidades/Canal'
    leg = lm_plot.axes.flat[0].get_legend()
    leg.set_title(new_title)
    new_labels = ['WT = 1.79', 'KO = 1.72']
    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    lm_plot.savefig('Unidades por canal.svg')
#%%
