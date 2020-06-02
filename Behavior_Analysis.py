import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pingouin as pg

si_raw = pd.read_csv('/Users/labc02/Documents/PDCB_data/Behavior/Social_Interaction_data.csv')

si_samp = si_raw[si_raw['Phase'] == 'Sample']

samp_ent = pd.melt(si_samp, id_vars=['Subject', 'Group'], value_vars=['Entries Obj/New Cons', 'Entries Conspecific'], var_name='Side', value_name='Entries')
samp_time = pd.melt(si_samp, id_vars=['Subject', 'Group'], value_vars=['Time Object/New Cons Chamber', 'Time Conspecific Chamber'], var_name='Side', value_name='Time')
samp_ent

#%%
[ent_fig, axs]=plt.subplots(nrows=1, ncols=2, figsize=(7,3.5), dpi= 600)
sns.barplot(x='Group', y='Entries', hue='Side', data=samp_ent, ci=68, palette=['grey', 'white'], edgecolor='k', capsize=.1, ax=axs[0])
axs[0].get_legend().set_visible(False)
sns.boxplot(x='Group', y='Time', hue='Side', data=samp_time, palette=['grey', 'white'], ax=axs[1])
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles=handles, labels=['Obj', 'Cons'])
plt.tight_layout()
#%%
# Socialbility Index: (Time_Cons - Time_Obj) / (Time_Cons + Time_Obj)
si_samp['Soc_Idx'] =(si_samp.loc[:,] - si_samp.iloc[:,-3])/si_samp.iloc[:,-1]

si_samp['Soc_Idx'] =  si_samp['Soc_Idx'].round(2)

si_samp
si_samp_ent = si_samp.loc[:,['Group', 'Entries Obj/New Cons', 'Entries Conspecific']]

si_samp_ent= pd.melt(si_samp_ent, id_vars=['Group'], value_vars=['Entries Obj/New Cons', 'Entries Conspecific'], var_name = 'Side', value_name = 'Entries')

for ii,jj in enumerate(si_samp_ent['Side']):
    if 'Obj' in jj:
        si_samp_ent.iloc[ii, 1] = 'Object'
    else:
        si_samp_ent.iloc[ii, 1] = 'Conspecific'

si_samp_ent
sns.boxplot(x = 'Group', y = 'Entries', data = si_samp_ent, hue = 'Side', showmeans = True, meanprops={"marker":"+","markeredgecolor":"k"}, palette= ['dimgray', 'white'])
# Figure: Sample Phase
#%%
samp_fig = plt.figure()
