import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pingouin as pg

si_raw = pd.read_csv('/Users/labc02/Documents/PDCB_data/Behavior/Social_Interaction_data.csv')

def detec_outlier(df, var_name, var_group):
    '''[DataFrame, str, str -> DataFrame]
    Outlier detection based on absolute deviaton from the median.
    Returns a copy of the original DataFrame without the indexes deemed as outliers'''
    clean_df = df.copy()
    for group in df[var_group].unique():
        mad = df[var_name][df[var_group] == group].mad()
        median = df[var_name][df[var_group] == group].median()
        th_up = median+(3*mad)
        th_down = median-(3*mad)
        outliers = df[(df[var_group] == group) & (~df[var_name].between(th_down, th_up))].index
        clean_df.drop(outliers, inplace = True)
    return clean_df
#Get just sample phase data
si_samp = si_raw[si_raw['Phase'] == 'Sample']
si_samp = detec_outlier(si_samp, 'Total Exploration', 'Group')

samp_time = pd.melt(si_samp, id_vars=['Subject', 'Group'], value_vars=['Time Object/New Cons Chamber', 'Time Conspecific Chamber'], var_name='Side', value_name='Time')

vars_time = ['Total Exploration', 'Time Conspecific Chamber', 'Time Object/New Cons Chamber']
# Test for normality
for var_ in vars_time:
    print(f'Normality test (Shapiro), {var_}')
    print(pg.normality(si_samp, dv=var_, group='Group'))
#Get test phase data
test_df = si_raw[si_raw['Phase'] == 'Test']
test_df = detec_outlier(test_df, 'Total Exploration', 'Group')

test_time = pd.melt(test_df, id_vars=['Subject', 'Group'], value_vars=['Time Object/New Cons Chamber', 'Time Conspecific Chamber'], var_name='Side', value_name='Time')
#Test for normality
for var_ in vars_time:
    print(f'Normality test (Shapiro), {var_}')
    print(pg.normality(test_df, dv=var_, group='Group'))

# Within group rm ANOVA to determine side preference
#Prep df with only Sample and Test phases
rm_df = si_raw[si_raw['Phase']!='Habituation']
rm_df = pd.melt(rm_df, id_vars=['Subject', 'Group', 'Phase'], value_vars=['Time Object/New Cons Chamber', 'Time Conspecific Chamber'], var_name='Side', value_name='Time')
# Run ANOVA
samp_anova = pg.anova(data=rm_df[rm_df['Phase']=='Sample'], dv='Time', between=['Side', 'Group'])
#Save to csv
samp_anova.to_csv('/Users/labc02/Documents/PDCB_data/Behavior/Stats/sample_time_anova.csv')
# post hoc test, pairwise_ttests, holm-bonf correction
samp_posthoc = pg.pairwise_ttests(data=rm_df[rm_df['Phase']=='Sample'], dv='Time', between=['Group', 'Side'], padjust= 'holm')
samp_posthoc.to_csv('/Users/labc02/Documents/PDCB_data/Behavior/Stats/sample_time_posthoc.csv')
samp_posthoc[['Group', 'p-corr']]
#%%
choice_fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
plt.suptitle('Social Choice')
sns.boxplot(x='Group', y='Total Exploration', data=si_samp, ax=axs[0], palette=['forestgreen', 'forestgreen', 'royalblue', 'royalblue'], showmeans=True, meanprops={'marker':'+', 'markeredgecolor':'k'}, width=0.5)
axs[0].set_ylabel('Total Time Exploring (s)')

sns.boxplot(x='Group', y='Time', hue='Side', data=samp_time, palette=['dimgray', 'white'], showmeans=True, meanprops={'marker':'+', 'markeredgecolor':'k'}, width=0.8, ax=axs[1])
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles=handles, labels=['Obj', 'Cons'])
axs[1].set_ylabel('Time in Chamber (s)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

novel_fig, axs_ = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
plt.suptitle('Social Novelty Preference')
sns.boxplot(x='Group', y='Total Exploration', data=test_df, ax=axs_[0], palette=['forestgreen', 'forestgreen', 'royalblue', 'royalblue'], showmeans=True, meanprops={'marker':'+', 'markeredgecolor':'k'}, width=0.5)
axs_[0].set_ylabel('Total Time Exploring (s)')

sns.boxplot(x='Group', y='Time', hue='Side', data=test_time, palette=['dimgray', 'white'], showmeans=True, meanprops={'marker':'+', 'markeredgecolor':'k'}, width=0.8, ax=axs_[1])
handles, labels = axs_[1].get_legend_handles_labels()
axs_[1].legend(handles=handles, labels=['New', 'Fam'])
axs_[1].set_ylabel('Time in Chamber (s)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#%%


#%%
[inter_fig, axs]=plt.subplots(nrows=1, ncols=2, figsize=(7,3.5), dpi= 600)
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
