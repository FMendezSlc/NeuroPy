import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

nest_raw = pd.read_csv('/Users/labc02/Documents/PDCB_data/MK-project/Nesting Data.csv')

nest_1mg = nest_raw[nest_raw['Dose (mg/Kg)'] == 1.0]
nest_1mg

#Variable is ordinal; Kruskall-Wallis
#Check homoscedasticity
pg.homoscedasticity(data=nest_1mg, dv='Nesting Score', group='Genotype')
nest_kw = pg.kruskal(data=nest_1mg, dv='Nesting Score', between='Genotype')
nest_kw
#%%
nest_fig = plt.figure(figsize=(4,3))
sns.boxplot(x='Tx', y='Nesting Score', hue='Genotype', data= nest_1mg, palette=['forestgreen', 'royalblue'], showmeans=True, meanprops={'marker':'+', 'markeredgecolor':'k'}, width=.5)
plt.legend(frameon=False, loc = 'lower right')
plt.xlabel('Treatment')
plt.tight_layout()
#%%
nest_fig.savefig('/Users/labc02/Documents/PDCB_data/MK-project/Figures/nesting_fig.png', dpi=600)

burrow_raw = pd.read_csv('/Users/labc02/Documents/PDCB_data/MK-project/Burrowing.csv')
burrow_raw
burrow_raw['Group'] = burrow_raw['Genotype']+'_'+burrow_raw['Tx']
for tx in burrow_raw['Tx'].unique():
        print(tx)
        print(pg.normality(data=burrow_raw, dv='% Test (12 h)', group='Genotype'))
# Check homoscedasticity
pg.homoscedasticity(data=burrow_raw, dv='% Test (12 h)', group='Group')

burr_kw = pg.kruskal(data=burrow_raw, dv='% Test (12 h)', between='Group')
burr_kw
#%%
burr_fig = plt.figure(figsize=(4,4))
sns.boxplot(x='Tx', y='% Test (12 h)', hue='Genotype', data= burrow_raw, palette=['forestgreen', 'royalblue'], showmeans=True, meanprops={'marker':'+', 'markeredgecolor':'k'}, width=.5)
plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Treatment')
plt.ylabel('Weight Burrowed at 12h (%)')
plt.tight_layout()
#%%
burr_fig.savefig('/Users/labc02/Documents/PDCB_data/MK-project/Figures/burrowing_fig.png', dpi=600)
