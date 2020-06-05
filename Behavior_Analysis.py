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
ph_val = zip(['***', '***', '***', '*'], [6,55,102,155], [120, 100, 120, 120])
for ii, jj, kk in ph_val:
    print(ii, jj, kk)

test_anova = pg.anova(data=rm_df[rm_df['Phase']=='Test'], dv='Time', between=['Side', 'Group'])
test_anova.to_csv('/Users/labc02/Documents/PDCB_data/Behavior/Stats/test_time_anova.csv')
test_posthoc = pg.pairwise_ttests(data=rm_df[rm_df['Phase']=='Test'], dv='Time', between=['Group', 'Side'], padjust= 'holm')
test_posthoc
test_posthoc.to_csv('/Users/labc02/Documents/PDCB_data/Behavior/Stats/test_time_posthoc.csv')

#%%
choice_fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
plt.suptitle('Social Choice')
sns.boxplot(x='Group', y='Total Exploration', data=si_samp, ax=axs[0], palette=['forestgreen', 'forestgreen', 'royalblue', 'royalblue'], showmeans=True, meanprops={'marker':'+', 'markeredgecolor':'k'}, width=0.5)
axs[0].set_ylabel('Total Time Exploring (s)')

sns.boxplot(x='Group', y='Time', hue='Side', data=samp_time, palette=['dimgray', 'white'], showmeans=True, meanprops={'marker':'+', 'markeredgecolor':'k'}, width=0.8, ax=axs[1])
handles, labels = axs[1].get_legend_handles_labels()
#axs[1].set_ylim(25, 225)
axs[1].legend(handles=handles, labels=['Obj', 'Cons'], frameon=False)
axs[1].set_ylabel('Time in Chamber (s)')
ph_val = zip(['***', '***', '***', '*'], [6,55,102,155], [120, 90, 120, 120])
for text, x, y in ph_val:
    axs[1].annotate(s=text, xy=(x,y), xycoords='axes points', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#%%
choice_fig.savefig('/Users/labc02/Documents/PDCB_data/Behavior/Figures/si_choice.png', dpi = 600)
#%%
novel_fig, axs_ = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
plt.suptitle('Social Novelty Preference')
sns.boxplot(x='Group', y='Total Exploration', data=test_df, ax=axs_[0], palette=['forestgreen', 'forestgreen', 'royalblue', 'royalblue'], showmeans=True, meanprops={'marker':'+', 'markeredgecolor':'k'}, width=0.5)
axs_[0].set_ylabel('Total Time Exploring (s)')

sns.boxplot(x='Group', y='Time', hue='Side', data=test_time, palette=['dimgray', 'white'], showmeans=True, meanprops={'marker':'+', 'markeredgecolor':'k'}, width=0.8, ax=axs_[1])
handles, labels = axs_[1].get_legend_handles_labels()
axs_[1].legend(handles=handles, labels=['New', 'Fam'], frameon=False)
axs_[1].set_ylabel('Time in Chamber (s)')
axs_[1].annotate(s='**', xy=(9,180), xycoords='axes points', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#%%
novel_fig.savefig('/Users/labc02/Documents/PDCB_data/Behavior/Figures/si_novelty.png', dpi = 600)

# Indexes: Sociability, Social Recognition and Novelty Preference
#Sociability
si_samp['Sociability']=((si_samp['Time Conspecific Chamber']-si_samp['Time Object/New Cons Chamber'])/si_samp['Total Exploration']).round(2)
soc_anova = pg.anova(data=si_samp, dv='Sociability', between='Group')
soc_ph= pg.pairwise_ttests(data=si_samp, dv='Sociability', between='Group', padjust='holm')
soc_ph
#Novelty Preference
test_df['Novelty']=((test_df['Time Object/New Cons Chamber']-test_df['Time Conspecific Chamber'])/test_df['Total Exploration']).round(2)
test_df['Novelty']
nov_anova = pg.anova(data=test_df, dv='Novelty', between='Group')
nov_anova
nov_ph= pg.pairwise_ttests(data=test_df, dv='Novelty', between='Group', padjust='holm')
nov_ph
#Social Recognition
samp_sub = set(si_samp['Subject'])
tst_sub = set(test_df['Subject'])
sub_set = samp_sub.intersection(tst_sub)
si_samp[si_samp['Subject'].isin(sub_set)]
srm_df = si_samp[['Subject', 'Group', 'Time Conspecific Chamber']][si_samp['Subject'].isin(sub_set)]
srm_df['Time New Cons'] = test_df['Time Object/New Cons Chamber'][test_df['Subject'].isin(sub_set)].values
srm_df
srm_df['SRM']=(srm_df['Time New Cons']-srm_df['Time Conspecific Chamber'])/(srm_df['Time New Cons']+srm_df['Time Conspecific Chamber'])
srm_nova= pg.anova(srm_df, dv='SRM', between='Group')
srm_ph = pg.pairwise_ttests(srm_df, dv='SRM', between='Group', padjust='holm')
srm_ph
#%%
idx_fig, idx_ax = plt.subplots(nrows=1, ncols=3, figsize=(7,4))
sns.boxplot(data=si_samp, x='Group', y='Sociability', palette=['forestgreen', 'forestgreen', 'royalblue', 'royalblue'], width=.5, showmeans=True, meanprops={'marker':'+', 'markeredgecolor':'k'}, ax=idx_ax[0])
idx_ax[0].annotate(s='**', xy=(25, 70), xycoords='axes points', xytext=(-4.5, -15), textcoords='offset points', arrowprops={'arrowstyle':'-[', 'color':'k'})
idx_ax[0].annotate(s='***', xy=(60, 70), xycoords='axes points', xytext=(-7, -15), textcoords='offset points', arrowprops={'arrowstyle':'-[', 'color':'k'})
props = {'connectionstyle':'bar','arrowstyle':'-[',\
                 'shrinkA':2,'shrinkB':2,'linewidth':1, 'color':'k'}
idx_ax[0].annotate('**', xy=(0.65, .2), xytext=(0.65, .13), xycoords='axes fraction', ha='center',
                va='bottom', arrowprops=dict(arrowstyle='-[, widthB=2.5, lengthB=.1', lw=1, color='k'))

sns.boxplot(data=test_df, x='Group', y='Novelty', palette=['forestgreen', 'forestgreen', 'royalblue', 'royalblue'], width=.5, showmeans=True, meanprops={'marker':'+', 'markeredgecolor':'k'}, ax=idx_ax[1])
idx_ax[1].set_ylabel('Novelty Preference')
idx_ax[1].annotate('*', xy=(0.35, .06), xytext=(0.35, .001), xycoords='axes fraction', ha='center',
                va='bottom', arrowprops=dict(arrowstyle='-[, widthB=2.2, lengthB=.1', lw=1, color='k'))

sns.boxplot(data=srm_df, x='Group', y='SRM', palette=['forestgreen', 'forestgreen', 'royalblue', 'royalblue'], width=.5, showmeans=True, meanprops={'marker':'+', 'markeredgecolor':'k'}, ax=idx_ax[2])
idx_ax[2].annotate('**', xy=(0.25, .8), xytext=(0.25, .80), xycoords='axes fraction', ha='center',
                va='bottom', arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=.1', lw=1, color='k'))
plt.tight_layout()
#%%
idx_fig.savefig('/Users/labc02/Documents/PDCB_data/Behavior/Figures/social_idx.png', dpi = 600)
