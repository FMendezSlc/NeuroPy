import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

si_raw = pd.read_csv('/Users/felipeantoniomendezsalcido/Desktop/Data/Social_Interaction_data.csv')

si_samp = si_raw[si_raw['Phase'] == 'Sample']
# Socialbility Index: (Time_Cons - Time_Obj) / (Time_Cons + Time_Obj)
si_samp['Soc_Idx'] =(si_samp.iloc[:,-2] - si_samp.iloc[:,-3])/si_samp.iloc[:,-1]

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
