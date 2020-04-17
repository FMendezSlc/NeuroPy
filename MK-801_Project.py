# TODO activate survivalist env
import matplotlib.pyplot as plt
import pandas as pd
import lifelines as lils

mk_raw = pd.read_csv('/Users/felipeantoniomendezsalcido/Desktop/Data/MK_Protocol_Log.csv')

mk_means = mk_raw.groupby(['Genotype', 'Treatment', 'Day P( )']).mean().reset_index()

mk_sem = mk_raw.groupby(['Genotype', 'Treatment', 'Day P( )']).sem().reset_index()

genotypes = mk_means['Genotype'].unique()
treatments = mk_means['Treatment'].unique()

mean_groups = {}
for ii in genotypes:
    for jj in treatments:
        mean_groups['{}_{}'.format(ii, jj)] = mk_means[(mk_means['Genotype'] == ii) & (mk_means['Treatment'] == jj)]['Weight (g)']
# DAMN was that Pythonic
# let's do it again
sem_groups = {}
for ii in genotypes:
    for jj in treatments:
        sem_groups['{}_{}'.format(ii, jj)] = mk_sem[(mk_sem['Genotype'] == ii) & (mk_sem['Treatment'] == jj)]['Weight (g)']

days_wt = range(5, 5+len(mean_groups['WT_Saline']))
days_ko = range(5, 5+len(mean_groups['KO_Saline']))
#%%
weigth_fig = plt.figure()
wt_ax = plt.subplot(111)

plt.errorbar(x=days_ko ,y=mean_groups['KO_Saline'], yerr=sem_groups['KO_Saline'], fmt='o-',
             color='forestgreen', capsize=2, label='KO-Saline')
plt.errorbar(x=days_ko, y=mean_groups['KO_MK-801'], yerr=sem_groups['KO_MK-801'], fmt='v--',
             color='forestgreen', capsize=2, label='KO-MK-801')
plt.errorbar(x=days_wt, y=mean_groups['WT_Saline'], yerr=sem_groups['WT_Saline'], fmt='o-',
             color='royalblue', capsize=2, label='WT-Saline')
plt.errorbar(x=days_wt, y=mean_groups['WT_MK-801'], yerr=sem_groups['WT_MK-801'], fmt='v--',
             color='royalblue', capsize=2, label='WT-MK-801')
#plt.xticks(range(5, 23))
plt.xlabel('Postnatal Day')
plt.ylabel('Weigth (g)')
plt.axvline(9, color='k', linestyle=':')
plt.legend()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#%%
weigth_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/Data/Figures/MK-weight.png')

surv_df = pd.read_csv('/Users/felipeantoniomendezsalcido/Desktop/Data/MK_protocol_survival.csv')
surv_df
durantion, event = lils.utils.datetimes_to_durations(surv_df['Birth'], surv_df['Death'], dayfirst=True)

surv_df['Group'] = [ii.split('_')[0] for ii in surv_df['ID']]
surv_df['Duration'] = durantion
surv_df['Events'] = event

TnE = {}
for ii in surv_df['Group'].unique():
    TnE['{}_T'.format(ii)] = surv_df['Duration'][surv_df['Group'] == ii]
    TnE['{}_E'.format(ii)] = surv_df['Events'][surv_df['Group'] == ii]
TnE
kmf = lils.KaplanMeierFitter()

#%%
surv_fig = plt.figure()
kmf.fit(durations= TnE['KOT_T'], event_observed = TnE['KOT_E'], label = 'KO-MK-801')
kmf.plot(color = 'forestgreen', linestyle='-')
kmf.fit(durations= TnE['KOS_T'], event_observed = TnE['KOS_E'], label = 'KO-MK-Saline')
kmf.plot(color = 'forestgreen', linestyle='--')
kmf.fit(durations= TnE['WTT_T'], event_observed = TnE['WTT_E'], label = 'WT-MK-801')
kmf.plot(color = 'royalblue', linestyle='-')
kmf.fit(durations= TnE['WTS_T'], event_observed = TnE['WTS_E'], label = 'WT-MK-Saline')
kmf.plot(color = 'royalblue', linestyle='--')
plt.title('Kaplan-Meier Survival Estimate')
plt.xlabel('Days')
plt.ylabel('Survival Probability')
ax = plt.gca()
ax.annotate(r'$p ≤ 0.01$', xy=(0.86, .5), xytext=(.88, .5), xycoords='axes fraction', fontsize=12, ha='left',
                va='center', fontweight='normal', arrowprops=dict(arrowstyle='-[, widthB=5.5,lengthB=0.5', lw=1, color='black'))
plt.axvline(5, color = 'k', linestyle= ':')
plt.axvline(9, color = 'k', linestyle= ':')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(loc= 3,fancybox = True, ncol=1)
#%5
surv_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/Data/Figures/MK-kmf.png')

logrank_ = lils.statistics.logrank_test(TnE['KOT_T'], TnE['WTT_T'], TnE['KOT_E'], TnE['WTT_E'])

logrank_.print_summary()
