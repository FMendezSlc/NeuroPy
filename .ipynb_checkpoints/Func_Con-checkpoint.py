import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import scipy as sp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import networkx as nx

# Little function for the Lorenz Curve.
# Credit to https://gist.github.com/CMCDragonkai
def lorenz(arr):
    # this divides the prefix sum by the total sum
    # this ensures all the values are between 0 and 1.0
    scaled_prefix_sum = arr.cumsum() / arr.sum()
    # this prepends the 0 value (because 0% of all people have 0% of all wealth)
    return np.insert(scaled_prefix_sum, 0, 0)

def ecdf(raw_data):
    '''[np.array -> tuple]
    Equivalent to R's ecdf(). Credit to Kripanshu Bhargava from Codementor'''
    cdfx = np.unique(raw_data)
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

# FIRST: STTCs REPRODUCES LOG-NORMAL DISTRIBUTION OF CA3 CONNECTIVITY
# (Ikegaya, cerebral cortex 2012).
eg_WT = pd.read_csv('/Users/felipeantoniomendezsalcido/Desktop/MEAs/STTC_basal/CA3_WT_male_02117_merged_sp-02STTC').drop(columns=['Unnamed: 0'])

loren_arr = lorenz(np.sort(np.array(eg_WT['STTC_weight'][eg_WT['STTC_weight'] > 0])))

#%%
lgnl_fig = plt.figure(figsize= (10,8))
ax = lgnl_fig.add_subplot(1,1,1)
sns.distplot(eg_WT['STTC_weight'][eg_WT['STTC_weight'] > 0], kde = False)
plt.yscale('log')
plt.xlabel('STTC', fontsize = 16)
plt.ylabel('Frequency (log)', fontsize = 16)
sns.despine()
in_ax = inset_axes(ax, width = '25%', height = 2, loc = 1)
sns.distplot(np.log(eg_WT['STTC_weight'][eg_WT['STTC_weight'] > 0]))
plt.ylabel('Density')
plt.xlabel('STTC (log)')
in_ax2 = inset_axes(ax, width = '25%', height = 2, loc = 4, bbox_to_anchor=(0,.15,1,1), bbox_transform=ax.transAxes)
plt.plot(np.linspace(0.0, 1.0, loren_arr.size), loren_arr)
plt.yticks([0.0, 0.50, 1.00])
plt.axhline(0.50, color = 'k')
plt.axvline(0.50, color = 'k')
plt.title('Lorenz Curve')
plt.ylabel('Cumulative')
plt.xlabel('STTC ordered')
#%%
lgnl_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/Lognormal Nature of CA3 connectivity.png')

# NOW, REPRODUCE PREVIOUS RESULTS
tidy_df = pd.read_csv('/Users/felipeantoniomendezsalcido/Desktop/MEAs/tidy_STTC.csv')

#%%
plt.figure(figsize=(12,12))
sns.distplot(np.log(tidy_df['STTC_weight'][tidy_df['Gen_type'] == 'WT']).dropna())
#%%
# Analyzing only positive values
All_pos = tidy_df['STTC_weight'][tidy_df['STTC_weight'] > 0]

All_WT = tidy_df['STTC_weight'][(tidy_df['Gen_type'] == 'WT') & (tidy_df['STTC_weight'] > 0)]

All_KO = tidy_df['STTC_weight'][(tidy_df['Gen_type']  == 'KO') & (tidy_df['STTC_weight'] > 0)]


# Determining groups and global threshold

WT_pos_th = (3*sm.robust.scale.mad(All_WT[All_WT > 0])+All_WT.median()).round(3)
KO_pos_th = (3*sm.robust.scale.mad(All_KO[All_KO > 0])+All_KO.median()).round(3)
Global_th = 3*sm.robust.scale.mad(All_pos)+ All_pos.median()
Global_th.round(3)

# ECDF time
cdf_wt = np.sort(np.unique(All_WT[All_WT >= Global_th.round(3)].values))
cdf_ko = np.sort(np.unique(All_KO[All_KO >= Global_th.round(3)].values))

wt_values = np.linspace(start=min(cdf_wt), stop=max(cdf_wt), num=len(cdf_wt))

ko_values = np.linspace(start=min(cdf_ko), stop=max(cdf_ko), num=len(cdf_ko))

wt_size = All_WT[All_WT >= Global_th.round(3)].size
ko_size = All_KO[All_KO >= Global_th.round(3)].size
wt_z = []
ko_z = []

for ii in wt_values:
    temp = All_WT[All_WT >= Global_th.round(3)][All_WT[All_WT >= Global_th.round(3)]< ii]
    fn_x = temp.size / wt_size
    wt_z.append(fn_x)
for ii in ko_values:
    temp = All_KO[All_KO >= Global_th.round(3)][All_KO[All_KO >= Global_th.round(3)]< ii]
    fn_x = temp.size / ko_size
    ko_z.append(fn_x)

# EDGES WEIGHTS DISTRIBUTIONS
#%%
agg_sig, (a0, a1) =  plt.subplots(1,2, gridspec_kw = {'width_ratios':[2, 1.5]}, figsize=(8,4))
sns.distplot(All_WT[All_WT >= Global_th.round(3)], kde = False, norm_hist = False, color = 'royalblue', hist_kws = {'histtype': 'bar', 'linewidth': 3, 'alpha':0.2}, ax = a0)
sns.distplot(All_WT[All_WT >= Global_th.round(3)], kde = False, norm_hist = False, color = 'royalblue', hist_kws = {'histtype': 'step', 'linewidth': 3, 'alpha' : 1}, ax= a0, label = 'WT')
sns.distplot(All_KO[All_KO >= Global_th.round(3)], kde = False, norm_hist = False, color = 'green', hist_kws = {'histtype': 'step', 'linewidth': 3, 'alpha' : 1}, ax = a0, label = 'KO')
sns.distplot(All_KO[All_KO >= Global_th.round(3)], kde = False, norm_hist = False, color = 'forestgreen', hist_kws = {'histtype': 'bar', 'linewidth': 4, 'alpha': 0.2}, ax = a0)
a0.set_yscale('log')
a0.set_ylabel('Frequency (log)', fontsize = 14)
a0.set_xlabel('STTC', fontsize = 14)
sns.despine()
a0.legend()
a1.plot(wt_values, wt_z, color = 'b')
a1.plot(ko_values, ko_z, color = 'g')
a1.set_xlabel('STTC', fontsize = 14)
a1.set_ylabel('Cumulative Probability', fontsize = 14)


in_a1 = inset_axes(a1, width = 1, height = 2, loc = 7)
plt.bar((.5), (All_WT[All_WT >= Global_th.round(3)].mean()), yerr= (All_WT[All_WT >= Global_th.round(3)].sem()), capsize = 3, color ='b', width = 0.5)
plt.bar((1.5), (All_KO[All_KO >= Global_th.round(3)].mean()), yerr = All_KO[All_KO >= Global_th.round(3)].sem(), capsize = 3, color = 'g', width = 0.5)
plt.xticks((.5, 1.5), ['WT', 'KO'])
plt.xlim(0,2)
plt.ylim(0, 0.2)
plt.ylabel('STTC')
in_a1.annotate('***', xy=(0.5, .8), xytext=(0.5, .85), xycoords='axes fraction',fontsize= 12, ha='center', va='bottom', fontweight = 'bold', arrowprops=dict(arrowstyle='-[, widthB=2, lengthB=0.2', lw=1))
##%%
agg_sig.savefig('/Users/felipeantoniomendezsalcido/Desktop/STTC analysis.png', dpi = 300)
help(inset_axes)
# Statistical test
sp.stats.mannwhitneyu(All_WT[All_WT >= Global_th.round(3)], All_KO[All_KO >= Global_th.round(3)], alternative = 'two-sided')

# Graph Analisys With the Global threshold
Gl_th = Global_th.round(3)

graph_df = tidy_df.drop(columns=['Unnamed: 0', 'Sex', 'STTC_boots', 'STTC_TBS', 'STTC_PostTBS'])

g_names = graph_df['Date'].unique()
g_names

 # DEGREES DISTRIBUTION ANALYSIS
wt_degs = []
ko_degs = []
wt_wedegs = []
ko_wedegs = []

for ii in g_names:
    print(ii)
    ex_g =  graph_df[['Gen_type', 'Node_1', 'Node_2', 'STTC_weight']][(graph_df['Date'] == ii) & (graph_df['STTC_weight'] >= Gl_th)]
    ex_gr = nx.from_pandas_edgelist(df = ex_g, source = 'Node_1', target = 'Node_2',  edge_attr = True)
    if ex_g['Gen_type'].unique()  == 'WT':
        wt_degs += list(dict(nx.degree(ex_gr)).values())
        wt_wedegs +=list(dict(nx.degree(ex_gr, weight = 'STTC_weight')).values())
    elif ex_g['Gen_type'].unique()  == 'KO':
        ko_degs += list(dict(nx.degree(ex_gr)).values())
        ko_wedegs += list(dict(nx.degree(ex_gr, weight = 'STTC_weight')).values())
len(wt_degs), len(wt_wedegs)



degrees_df = pd.DataFrame(data = [wt_degs, ko_degs, wt_wedegs, ko_wedegs])
degrees_df = degrees_df.T
degrees_df.columns = ['WT_uw', 'KO_uw', 'WT_w', 'KO_w']

hist, bins = np.histogram(degrees_df['KO_uw'].dropna(), bins = 'auto', density = True)
hist, bins
np.cumsum(hist)
center = (bins[:-1] + bins[1:]) / 2
len(center)
np.insert(bins, 0, 0)
1-np.cumsum(hist)
len(1-np.cumsum(hist)), len(bins)
len(hist), len(bins)

# FUNCTIONS TO FIT DISTRIBUTIONS
def powerlaw(x, g):
    return x**-g
def myexp(x, g):
    return np.exp(-g*x)
def trunc_power(x, g, k):
    return x**-g * np.exp(-x/k)
# DEGREES DISTRIBUTION FIGURE
S_fits = {}
#%%
degrees_fig = plt.figure(figsize= (7,7))
deg_1 = plt.subplot(2,1,1)
sns.distplot(degrees_df['WT_uw'].dropna(), kde = False, color = 'royalblue', hist_kws = {'histtype': 'step', 'linewidth': 3, 'alpha' : 1}, bins = 'auto', norm_hist = False, label = 'WT')
sns.distplot(degrees_df['WT_uw'].dropna(), kde = False, color = 'royalblue', hist_kws = {'histtype': 'bar', 'linewidth': 3, 'alpha':0.2}, bins = 'auto', norm_hist = False)
sns.distplot(degrees_df['KO_uw'].dropna(), kde = False, bins = 'auto', norm_hist = False, color = 'green', hist_kws = {'histtype': 'step', 'linewidth': 3, 'alpha' : 1}, label = 'KO')
sns.distplot(degrees_df['KO_uw'].dropna(), kde = False, bins = 'auto', norm_hist = False, color = 'forestgreen', hist_kws = {'histtype': 'bar', 'linewidth': 4, 'alpha': 0.2})
plt.xlabel('Node degree (k)', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.legend(loc = (.1,.75))
inset_axes(deg_1, width = '35%', height = 1.7)
wu_ecdf = ecdf(degrees_df['WT_uw'].dropna())
wu_cCDf = 1-np.array(wu_ecdf[1])
plt.plot(wu_ecdf[0], wu_cCDf, color = 'royalblue', marker = 'o', mfc = 'none', linestyle = '')
ku_ecdf = ecdf(degrees_df['KO_uw'].dropna())
ku_cCDf = 1-np.array(ku_ecdf[1])
plt.plot(ku_ecdf[0], ku_cCDf, color = 'green', marker = 'o', mfc = 'none', linestyle = '')
sp.stats.ks_2samp(degrees_df['WT_uw'].dropna(), degrees_df['KO_uw'].dropna())
plt.text(1, .002, r'$p$ ≤ 0.05')
popt,pcov =  sp.optimize.curve_fit(powerlaw, wu_ecdf[0], wu_cCDf)
plt.plot(wu_ecdf[0], powerlaw(wu_ecdf[0], popt), 'b:')
S_fits['power_wt'] = np.sqrt(np.sum(((wu_cCDf)-powerlaw(wu_ecdf[0], popt))**2)/(len(wu_cCDf)-1))
popt,pcov =  sp.optimize.curve_fit(myexp, wu_ecdf[0], wu_cCDf)
plt.plot(wu_ecdf[0], myexp(wu_ecdf[0], popt), 'b--')
S_fits['exp_wt'] = np.sqrt(np.sum(((wu_cCDf)-myexp(wu_ecdf[0], popt))**2)/(len(wu_cCDf)-1))
popt,pcov =  sp.optimize.curve_fit(trunc_power, wu_ecdf[0], wu_cCDf)
plt.plot(wu_ecdf[0], trunc_power(wu_ecdf[0], popt[0], popt[1]), 'b-')
print(popt)
S_fits['trpower_wt'] = np.sqrt(np.sum(((wu_cCDf)-trunc_power(wu_ecdf[0], popt[0], popt[1]))**2)/(len(wu_cCDf)-1))
popt,pcov =  sp.optimize.curve_fit(powerlaw, ku_ecdf[0], ku_cCDf)
plt.plot(ku_ecdf[0], powerlaw(ku_ecdf[0], popt), 'g:')
popt,pcov =  sp.optimize.curve_fit(myexp, ku_ecdf[0], ku_cCDf)
S_fits['power_ko'] = np.sqrt(np.sum(((ku_cCDf)-powerlaw(ku_ecdf[0], popt))**2)/(len(ku_cCDf)-1))
plt.plot(ku_ecdf[0], myexp(ku_ecdf[0], popt), 'g--')
S_fits['exp_ko'] = np.sqrt(np.sum(((ku_cCDf)-myexp(ku_ecdf[0], popt))**2)/(len(ku_cCDf)-1))
popt,pcov =  sp.optimize.curve_fit(trunc_power, ku_ecdf[0], ku_cCDf)
plt.plot(ku_ecdf[0], trunc_power(ku_ecdf[0], popt[0], popt[1]), 'g-')
S_fits['trpower_ko'] = np.sqrt(np.sum(((ku_cCDf)-trunc_power(ku_ecdf[0], popt[0], popt[1]))**2)/(len(ku_cCDf)-1))
print(popt)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('log(k)', fontsize = 10)
plt.ylabel('log P(degree > k)', fontsize = 10)


deg_2 = plt.subplot(2,1,2)
sns.distplot(degrees_df['WT_w'].dropna(), kde = False, color = 'royalblue', hist_kws = {'histtype': 'step', 'linewidth': 3, 'alpha' : 1}, bins = 'auto', norm_hist = False)
sns.distplot(degrees_df['WT_w'].dropna(), kde = False, color = 'royalblue', hist_kws = {'histtype': 'bar', 'linewidth': 3, 'alpha':0.2}, bins = 'auto', norm_hist = False)
sns.distplot(degrees_df['KO_w'].dropna(), kde = False, bins = 'auto', norm_hist = False, color = 'green', hist_kws = {'histtype': 'step', 'linewidth': 3, 'alpha' : 1})
sns.distplot(degrees_df['KO_w'].dropna(), kde = False, bins = 'auto', norm_hist = False, color = 'forestgreen', hist_kws = {'histtype': 'bar', 'linewidth': 4, 'alpha': 0.2})
plt.xlabel('Node Strength (s)', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
sp.stats.ks_2samp(degrees_df['WT_w'].dropna(), degrees_df['KO_w'].dropna())
inset_axes(deg_2, width = '35%', height = 1.7)
w_ecdf = ecdf(degrees_df['WT_w'].dropna().round(1))
w_cCDf = 1-np.array(w_ecdf[1])
plt.plot(w_ecdf[0], w_cCDf, color = 'royalblue', marker = 'o', mfc = 'none', linestyle = '')
k_ecdf = ecdf(degrees_df['KO_w'].dropna().round(1))
k_cCDf = 1-np.array(k_ecdf[1])
plt.plot(k_ecdf[0], k_cCDf, color = 'green', marker = 'o', mfc = 'none', linestyle = '')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('log(s)', fontsize = 10)
plt.ylabel('log P(Str > s)', fontsize = 10)
#%%
degrees_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/Nodes_distribution.png', dpi = 300)

S_fits

# Curve fitting
def powerlaw(x, g):
    return x**-g
def myexp(x, g):
    return np.exp(-g*x)
def trunc_power(x, g, k):
    return x**-g * np.exp(-x/k)
popt,pcov =  sp.optimize.curve_fit(trunc_power, wu_ecdf[0], wu_cCDf)
popt
#%%
plt.figure()
plt.plot(wu_ecdf[0], wu_cCDf)
plt.plot(wu_ecdf[0], trunc_power(wu_ecdf[0], -0.26, 3.6))
#%%
# Working figure
#%%
# wu_hist, wu_bins = np.histogram(degrees_df['WT_uw'].dropna(), bins = 'auto', density = True)
# wu_p = 1-(np.cumsum(wu_hist))
# ku_hist, ku_bins = np.histogram(degrees_df['KO_uw'].dropna(), bins = 'auto', density = True)
# ku_p = 1-(np.cumsum(ku_hist))
plt.figure(figsize=(7, 7))
# plt.plot((wu_bins[:-1] + wu_bins[1:]) / 2, wu_p, color = 'royalblue', marker = '*', linestyle = '')
# w_slope, w_intercept, wr_value, p_value, std_err = sp.stats.linregress(np.log((wu_bins[:-1] + wu_bins[1:]) / 2)[2:], np.log(wu_p)[2:])
# plt.plot(((wu_bins[:-1] + wu_bins[1:]) / 2)[1:-2], ((wu_bins[:-1] + wu_bins[1:]) / 2)[1:-2]**w_slope* np.exp(w_intercept), 'b--')
# plt.plot((ku_bins[:-1] + ku_bins[1:]) / 2, ku_p, color = 'green', marker = '*', linestyle = '')
# k_slope, k_intercept, kr_value, p_value, std_err = sp.stats.linregress(np.log((ku_bins[:-1] + ku_bins[1:]) / 2)[:], np.log(ku_p)[:])
# k_slope, k_intercept, kr_value, p_value, std_err
# plt.plot(((ku_bins[:-1] + ku_bins[1:]) / 2)[:], ((ku_bins[:-1] + ku_bins[1:]) / 2)[:]**k_slope * np.exp(k_intercept), 'g--')
for ii in g_names:
    ex_g =  graph_df[['Gen_type', 'Node_1', 'Node_2', 'STTC_weight']][(graph_df['Date'] == ii) & (graph_df['STTC_weight'] >= Gl_th)]
    ex_gr = nx.from_pandas_edgelist(df = ex_g, source = 'Node_1', target = 'Node_2',  edge_attr = True)
    if ex_g['Gen_type'].unique()  == 'WT':
        print(ii)
        wt_degs += list(dict(nx.degree(ex_gr)).values())
        hist, bins = np.histogram(wt_degs, bins = 'auto', density = True)
        wu_p = 1-(np.cumsum(hist))
        plt.plot((bins[:-1] + bins[1:]) / 2, wu_p, color = 'royalblue', marker = '*', linestyle = '', markersize = 5)
    elif ex_g['Gen_type'].unique()  == 'KO':
        print(ii)
        ko_degs += list(dict(nx.degree(ex_gr)).values())
        hist, bins = np.histogram(ko_degs, bins = 'auto', density = True)
        ku_p = 1-(np.cumsum(hist))
        plt.plot((bins[:-1] + bins[1:]) / 2, ku_p, color = 'green', marker = '*', linestyle = '', markersize = 5)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('log(k)', fontsize = 12)
plt.ylabel('log P(degree > k)', fontsize = 12)
#%%
plt.hist(wt_degs)
plt.hist(ko_degs)
# So first no. nodes, no. edges, av. degree, density, connectivity, clustering, degree centrality, av. shortest path, global efficiency
# Might need an obnibus tests, most likely ANOVA
g_dic = {'Date': [], 'Gen_type': [], 'No. Nodes': [], 'No. Edges': [], 'Density': [], 'Av. Degree': [], 'Node Conn': [], 'Clustering': [], 'Clustering_W': [], 'Char Path' : [], 'Char Path_w' : [], 'Global Eff' : [], 'Major C': [], 'Degree Cent' : [], 'Sigma' : [], 'Omega' : []}

for ii in g_names:
    print(ii)
    g_df = graph_df[graph_df['Date'] == ii]
    g_th = g_df[['Node_1', 'Node_2', 'STTC_weight']][g_df['STTC_weight'] >= Gl_th]
    g_th['STTC_weight'] = g_th['STTC_weight'].round(3)
    g_th['STTC_rec'] = 1/g_th['STTC_weight'].round(3)
    temp_g = nx.from_pandas_edgelist(df = g_th, source = 'Node_1', target = 'Node_2',  edge_attr = True)

    if nx.is_empty(temp_g) == True:
        print(ii, 'is empty!!!!')

    else:
        g_dic['Date'].append(g_df['Date'].unique()[0])
        g_dic['Gen_type'].append(g_df['Gen_type'].unique()[0])
        g_dic['No. Nodes'].append(nx.number_of_nodes(temp_g))
        g_dic['No. Edges'].append(nx.number_of_edges(temp_g))
        g_dic['Density'].append(nx.density(temp_g))
        g_dic['Av. Degree'].append(round(sum(dict(temp_g.degree()).values())/float(nx.number_of_nodes(temp_g)), 2))
        g_dic['Degree Cent'].append(round(sum(dict(nx.degree_centrality(temp_g)).values())/float(nx.number_of_nodes(temp_g)), 2))
        g_dic['Node Conn'].append(nx.node_connectivity(temp_g))
        g_dic['Clustering'].append(nx.average_clustering(temp_g))
        g_dic['Clustering_W'].append(nx.average_clustering(temp_g, weight = 'STTC_rec'))
        if nx.is_connected(temp_g) == True:
            g_dic['Char Path'].append(nx.average_shortest_path_length(temp_g))
            g_dic['Char Path_w'].append(nx.average_shortest_path_length(temp_g, weight = 'STTC_rec'))
            g_dic['Global Eff'].append(nx.global_efficiency(temp_g))
            g_dic['Major C'].append(nx.number_of_nodes(temp_g)/nx.number_of_nodes(temp_g)*100)
            g_dic['Sigma'].append(nx.(temp_g))
            g_dic['Omega'].append(nx.omega(temp_g))
        else:
            print(ii, 'is not connected')
            Gc = max(nx.connected_component_subgraphs(temp_g), key=len)
            g_dic['Char Path'].append(nx.average_shortest_path_length(Gc))
            g_dic['Char Path_w'].append(nx.average_shortest_path_length(Gc, weight = 'STTC_rec'))
            g_dic['Global Eff'].append(nx.global_efficiency(Gc))
            g_dic['Major C'].append(nx.number_of_nodes(Gc)/nx.number_of_nodes(temp_g)*100)
            g_dic['Sigma'].append(nx.sigma(Gc))
            g_dic['Omega'].append(nx.omega(Gc))

g_df = pd.DataFrame(g_dic)
g_df = g_df.drop(24) # outlier, only 10 nodes

# PROPORTION OF CONNECTED AND FRAGMENTED GRPAHS
wt_con = g_df['Gen_type'][(g_df['Major C']== 100) & (g_df['Gen_type'] == 'WT')].count()
wt_disc = g_df['Gen_type'][(g_df['Major C']< 100) & (g_df['Gen_type'] == 'WT')].count()
ko_con = g_df['Gen_type'][(g_df['Major C']== 100) & (g_df['Gen_type'] == 'KO')].count()
ko_disc = g_df['Gen_type'][(g_df['Major C']< 100) & (g_df['Gen_type'] == 'KO')].count()

cont_tab = np.array([[wt_con, wt_disc], [ko_con, ko_disc]])
sp.stats.fisher_exact(cont_tab)

disconnected_Nxs = (g_df['Gen_type'][(g_df['Major C']< 100) & (g_df['Gen_type'] == 'WT')].count()/g_df['Gen_type'][g_df['Gen_type'] == 'WT'].count()).round(2), (g_df['Gen_type'][(g_df['Major C']< 100) & (g_df['Gen_type'] == 'KO')].count()/g_df['Gen_type'][g_df['Gen_type'] == 'KO'].count()).round(2)

connected_Nxs = (g_df['Gen_type'][(g_df['Major C']== 100) & (g_df['Gen_type'] == 'WT')].count()/g_df['Gen_type'][g_df['Gen_type'] == 'WT'].count()).round(2), (g_df['Gen_type'][(g_df['Major C']== 100) & (g_df['Gen_type'] == 'KO')].count()/g_df['Gen_type'][g_df['Gen_type'] == 'KO'].count()).round(2)


rep_graphs = ['151117', '161117', '61217', '22117', '140218(b)', '140218']

# CONNECTED/DISCONNECTED FIGURE
#%%
frag_prop = plt.figure(figsize = (12,6))
fr_ax = plt.subplot2grid((2, 4), (0, 0), rowspan = 2)
plt.bar((1, 1.5), disconnected_Nxs, color = ['cornflowerblue', 'darkseagreen'], width = .3, edgecolor = ['b', 'g'], linewidth = 4, label = 'Disconnected')
plt.bar((1, 1.5), connected_Nxs, bottom = disconnected_Nxs, color = ['b', 'green'], width = .3, edgecolor = ['b', 'g'], linewidth = 4, label = 'Connected')
plt.xticks((1,1.5), ['WT', 'KO'], fontsize = 12)
plt.xlim(0.5, 2)
plt.ylabel('Proportion', fontsize = (12))
fr_ax.spines['top'].set_visible(False)
fr_ax.spines['right'].set_visible(False)
fr_ax.annotate('*', xy=(0.5, 1.01), xytext=(0.5, 1.03), xycoords='axes fraction',fontsize= 18, ha='center', va='bottom', fontweight = 'bold', arrowprops=dict(arrowstyle='-[, widthB=2.5, lengthB=0.5', lw=1.5))
fr_ax.annotate('C', xy=(.26, .85), xytext=(.3, .87), xycoords='axes fraction',fontsize= 12, color = 'white', fontweight ='bold')
fr_ax.annotate('DC', xy=(.26, .2), xytext=(.27, .35), xycoords='axes fraction',fontsize= 12, color = 'white', fontweight = 'bold')
fr_ax.annotate('C', xy=(.26, .85), xytext=(.64, .87), xycoords='axes fraction',fontsize= 12, color = 'white', fontweight ='bold')
fr_ax.annotate('DC', xy=(.26, .2), xytext=(.61, .35), xycoords='axes fraction',fontsize= 12, color = 'white', fontweight = 'bold')
other_ax = plt.subplot2grid((2,2), (0, 1), rowspan = 2)
_r = 1
_wc = 1
_kc = 1
for ii in rep_graphs:
    g_df = graph_df[graph_df['Date'] == ii]
    g_th = g_df[['Node_1', 'Node_2', 'STTC_weight']][g_df['STTC_weight'] >= Gl_th]
    g_th['STTC_weight'] = g_th['STTC_weight'].round(3)
    temp_g = nx.from_pandas_edgelist(df = g_th, source = 'Node_1', target = 'Node_2',  edge_attr = True)

    if g_df['Gen_type'].unique() == 'KO':
        _r = 1
        kx = plt.subplot2grid((2, 4), (_r, _kc))
        nx.draw_networkx(temp_g, node_size = 15, node_color = 'darkseagreen', linewidths = .5, width = .4, ax = kx, with_labels = False)
        kx.collections[0].set_edgecolor("k")
        plt.xticks([])
        plt.yticks([])
        _kc += 1
    elif g_df['Gen_type'].unique() == 'WT':
        _r = 0
        wx =  plt.subplot2grid((2,4), (_r, _wc))
        nx.draw_networkx(temp_g, node_size = 15, node_color = 'dodgerblue', linewidths = .5, width = .4 , ax = wx, with_labels = False)
        wx.collections[0].set_edgecolor("k")
        plt.xticks([])
        plt.yticks([])
        _wc += 1
#%%
frag_prop.savefig('/Users/felipeantoniomendezsalcido/Desktop/Fragmentation_fig.png')

# Clustering, Weighted Cluestring, Charactheristic Path (W) and Global Eff
#%%
g_paramsFig = plt.figure(figsize=(12, 8))
dens_ax = plt.subplot(2,3,1)
sns.barplot('Gen_type', 'Density', data = g_df, ci = 68, order = ['WT', 'KO'], palette= ['royalblue', 'forestgreen'], capsize = .2)
p_stat = sp.stats.ttest_ind(g_df['Density'][g_df['Gen_type'] == 'WT'], g_df['Density'][g_df['Gen_type'] == 'KO'])[1]
dens_ax.annotate(r'$p = {}$'.format(p_stat.round(3)), xy=(0.5, .93), xytext=(0.5, .93), xycoords='axes fraction',fontsize= 14, ha='center', va='bottom')
sns.despine()
plt.title('Density', fontsize = 14)
plt.ylabel('')
plt.xlabel('')
clus_ax = plt.subplot(2, 3, 2)
sns.barplot('Gen_type', 'Clustering', data = g_df, ci = 68, order = ['WT', 'KO'], palette= ['royalblue', 'forestgreen'], capsize = .2)
p_stat = sp.stats.ttest_ind(g_df['Clustering'][g_df['Gen_type'] == 'WT'], g_df['Clustering'][g_df['Gen_type'] == 'KO'])[1]
clus_ax.annotate(r'$p = {}$'.format(p_stat.round(3)), xy=(0.5, .93), xytext=(0.5, .93), xycoords='axes fraction',fontsize= 14, ha='center', va='bottom')
sns.despine()
plt.title('Clustering', fontsize = 14)
plt.ylabel('')
plt.xlabel('')
clusW_ax = plt.subplot(2,3,3)
sns.barplot('Gen_type', 'Clustering_W', data = g_df, ci = 68, order = ['WT', 'KO'], palette= ['royalblue', 'forestgreen'], capsize = .2)
p_stat = sp.stats.ttest_ind(g_df['Clustering_W'][g_df['Gen_type'] == 'WT'], g_df['Clustering_W'][g_df['Gen_type'] == 'KO'], equal_var = True)[1]
clusW_ax.annotate(r'$p = {}$'.format(p_stat.round(3)), xy=(0.5, .95), xytext=(0.5, .93), xycoords='axes fraction',fontsize= 14, ha='center', va='bottom')
sns.despine()
plt.title('Clustering (Weigthed)', fontsize = 14)
plt.ylabel('')
plt.xlabel('')
chpth_ax = plt.subplot(2,3,4)
sns.barplot('Gen_type', 'Char Path', data = g_df, ci = 68, order = ['WT', 'KO'], palette= ['royalblue', 'forestgreen'], capsize = .2)
p_stat = sp.stats.ttest_ind(g_df['Char Path'][g_df['Gen_type'] == 'WT'], g_df['Char Path'][g_df['Gen_type'] == 'KO'], equal_var = True)[1]
chpth_ax.annotate(r'$p = {}$'.format(p_stat.round(3)), xy=(0.7, .95), xytext=(0.5, .93), xycoords='axes fraction',fontsize= 14, ha='center', va='bottom')
sns.despine()
plt.title('Charc. Path', fontsize = 14)
plt.ylabel('')
plt.xlabel('')
chpthW_ax = plt.subplot(2,3,5)
sns.barplot('Gen_type', 'Char Path_w', data = g_df, ci = 68, order = ['WT', 'KO'], palette= ['royalblue', 'forestgreen'], capsize = .2)
p_stat = sp.stats.ttest_ind(g_df['Char Path_w'][g_df['Gen_type'] == 'WT'], g_df['Char Path_w'][g_df['Gen_type'] == 'KO'], equal_var = True)[1]
chpthW_ax.annotate(r'$p = {}$'.format(p_stat.round(3)), xy=(0.7, .95), xytext=(0.5, .93), xycoords='axes fraction',fontsize= 14, ha='center', va='bottom')
sns.despine()
plt.title('Charc. Path (Weighted)', fontsize = 14)
plt.ylabel('')
plt.xlabel('')
gloeff_ax = plt.subplot(2, 3, 6)
sns.barplot('Gen_type', 'Global Eff', data = g_df, ci = 68, order = ['WT', 'KO'], palette= ['royalblue', 'forestgreen'], capsize = .2)
p_stat = sp.stats.ttest_ind(g_df['Global Eff'][g_df['Gen_type'] == 'WT'], g_df['Global Eff'][g_df['Gen_type'] == 'KO'], equal_var = True)[1]
gloeff_ax.annotate(r'$p = {}$'.format(p_stat.round(3)), xy=(0.7, .95), xytext=(0.5, .93), xycoords='axes fraction',fontsize= 14, ha='center', va='bottom')
sns.despine()
plt.title('Global Efficiency', fontsize = 14)
plt.ylabel('')
plt.xlabel('')
#%%
g_paramsFig.savefig('/Users/felipeantoniomendezsalcido/Desktop/Graph_paramsFig2.png')


# GRAPH  CATALOG; ALL INDIVIDUAL GRAPHS
#%%
all_gFig = plt.figure(figsize= (14,14))

_r = 0
_c = 0
for ii in g_names:
    g_df = graph_df[graph_df['Date'] == ii]
    g_th = g_df[['Node_1', 'Node_2', 'STTC_weight']][g_df['STTC_weight'] >= Gl_th]
    g_th['STTC_weight'] = g_th['STTC_weight'].round(3)
    temp_g = nx.from_pandas_edgelist(df = g_th, source = 'Node_1', target = 'Node_2',  edge_attr = True)

    if g_df['Gen_type'].unique() == 'KO':
        kx = plt.subplot2grid((7, 4), (_r, _c))
        nx.draw_networkx(temp_g, node_size = 15, node_color = 'darkseagreen', linewidths = .5, width = .4, ax = kx, with_labels = False)
        kx.collections[0].set_edgecolor("k")
        plt.title(ii+g_df['Gen_type'].unique())
        plt.xticks([])
        plt.yticks([])
        _r += 1

    elif g_df['Gen_type'].unique() == 'WT':
        wx =  plt.subplot2grid((7,4), (_r, _c))
        nx.draw_networkx(temp_g, node_size = 15, node_color = 'dodgerblue', linewidths = .5, width = .4 , ax = wx, with_labels = False)
        wx.collections[0].set_edgecolor("k")
        plt.title(ii+g_df['Gen_type'].unique())
        plt.xticks([])
        plt.yticks([])
        _r += 1
    if _r == 7:
        _r = 0
        _c += 1

#%%
all_gFig.savefig('/Users/felipeantoniomendezsalcido/Desktop/All_graphs.png')

############################################################################

# ANALYSIS WITH INDIVIDUAL THRESHOLD

# ECDF time
cdf_wt = np.sort(np.unique(All_WT[All_WT >= WT_pos_th].values))
cdf_ko = np.sort(np.unique(All_KO[All_KO >= KO_pos_th].values))

wt_values = np.linspace(start=min(cdf_wt), stop=max(cdf_wt), num=len(cdf_wt))

ko_values = np.linspace(start=min(cdf_ko), stop=max(cdf_ko), num=len(cdf_ko))

wt_size = All_WT[All_WT >= WT_pos_th].size
ko_size = All_KO[All_KO >= KO_pos_th].size
wt_z = []
ko_z = []

for ii in wt_values:
    temp = All_WT[All_WT >= WT_pos_th][All_WT[All_WT >= WT_pos_th]< ii]
    fn_x = temp.size / wt_size
    wt_z.append(fn_x)
for ii in ko_values:
    temp = All_KO[All_KO >= KO_pos_th][All_KO[All_KO >= KO_pos_th]< ii]
    fn_x = temp.size / ko_size
    ko_z.append(fn_x)

# EDGES WEIGHTS DISTRIBUTIONS
#%%
agg_sig, (a0, a1) =  plt.subplots(1,2, gridspec_kw = {'width_ratios':[2, 1.5]}, figsize=(12,6))
sns.distplot(All_WT[All_WT >= WT_pos_th], kde = False, norm_hist = False, color = 'royalblue', hist_kws = {'histtype': 'bar', 'linewidth': 3, 'alpha':0.2}, ax = a0)
sns.distplot(All_WT[All_WT >= WT_pos_th], kde = False, norm_hist = False, color = 'royalblue', hist_kws = {'histtype': 'step', 'linewidth': 3, 'alpha' : 1}, ax= a0)
sns.distplot(All_KO[All_KO >= KO_pos_th], kde = False, norm_hist = False, color = 'green', hist_kws = {'histtype': 'step', 'linewidth': 3, 'alpha' : 1}, ax = a0)
sns.distplot(All_KO[All_KO >= KO_pos_th], kde = False, norm_hist = False, color = 'forestgreen', hist_kws = {'histtype': 'bar', 'linewidth': 4, 'alpha': 0.2}, ax = a0)
a0.set_yscale('log')
a0.set_ylabel('Frequency (log)', fontsize = 14)
a0.set_xlabel('STTC', fontsize = 14)
sns.despine()

a1.plot(wt_values, wt_z, color = 'b')
a1.plot(ko_values, ko_z, color = 'g')
a1.set_xlabel('STTC', fontsize = 14)
a1.set_ylabel('Cumulative Probability', fontsize = 14)
in_a1 = inset_axes(a1, width = '20%', height = 2, loc = 4, bbox_to_anchor=(0.25, 0.05,1,1), bbox_transform=ax.transAxes)
plt.errorbar((.5), (All_WT[All_WT >= WT_pos_th].mean()), yerr= (All_WT[All_WT >= WT_pos_th].sem()), fmt = 'bo', capsize = 3, markersize = 7.5)
plt.errorbar((1.5), (All_KO[All_KO >= KO_pos_th].mean()), yerr = All_KO[All_KO >= KO_pos_th].sem(), fmt = 'go', capsize = 3, markersize = 7.5)
plt.xticks((.5, 1.5), ['WT', 'KO'])
plt.xlim(0,2)
plt.ylim(0.1, 0.15)
plt.ylabel('STTC')
plt.text(.35,.13, '***')
##%%
agg_sig.savefig('/Users/felipeantoniomendezsalcido/Desktop/STTC_analysis_ind.png')
# Statistical test
sp.stats.mannwhitneyu(All_WT[All_WT >= WT_pos_th], All_KO[All_KO >= KO_pos_th], alternative = 'two-sided')

# Graph Analisys With the Global threshold
Gl_th = Global_th.round(3)

graph_df = tidy_df.drop(columns=['Unnamed: 0', 'Sex', 'STTC_boots', 'STTC_TBS', 'STTC_PostTBS'])

g_names = graph_df['Date'].unique()

 # DEGREES DISTRIBUTION ANALYSIS
wt_degs = []
ko_degs = []
wt_wedegs = []
ko_wedegs = []

graph_df['Gen_type'][graph_df['Date']==g_names[0]].unique()=='KO'

for ii in g_names:
    print(ii)
    if graph_df['Gen_type'][graph_df['Date']== ii].unique()=='WT':
        ex_g =  graph_df[['Gen_type', 'Node_1', 'Node_2', 'STTC_weight']][(graph_df['Date'] == ii) & (graph_df['STTC_weight'] >= WT_pos_th)]
        ex_gr = nx.from_pandas_edgelist(df = ex_g, source = 'Node_1', target = 'Node_2',  edge_attr = True)
        wt_degs += list(dict(nx.degree(ex_gr)).values())
        wt_wedegs +=list(dict(nx.degree(ex_gr, weight = 'STTC_weight')).values())
    elif graph_df['Gen_type'][graph_df['Date']== ii].unique()=='KO':
        ex_g =  graph_df[['Gen_type', 'Node_1', 'Node_2', 'STTC_weight']][(graph_df['Date'] == ii) & (graph_df['STTC_weight'] >= KO_pos_th)]
        ex_gr = nx.from_pandas_edgelist(df = ex_g, source = 'Node_1', target = 'Node_2',  edge_attr = True)
        wt_degs += list(dict(nx.degree(ex_gr)).values())
        wt_wedegs +=list(dict(nx.degree(ex_gr, weight = 'STTC_weight')).values())
        ko_degs += list(dict(nx.degree(ex_gr)).values())
        ko_wedegs += list(dict(nx.degree(ex_gr, weight = 'STTC_weight')).values())
len(ko_degs), len(ko_wedegs)



degrees_df = pd.DataFrame(data = [wt_degs, ko_degs, wt_wedegs, ko_wedegs])
degrees_df = degrees_df.T
degrees_df.columns = ['WT_uw', 'KO_uw', 'WT_w', 'KO_w']


# DEGREES DISTRIBUTION FIGURE
#%%
degrees_fig = plt.figure(figsize= (11,10))
deg_1 = plt.subplot(2,1,1)
sns.distplot(degrees_df['WT_uw'].dropna(), kde = False, color = 'royalblue', hist_kws = {'histtype': 'step', 'linewidth': 3, 'alpha' : 1}, bins = 'auto', norm_hist = True, label = 'WT')
sns.distplot(degrees_df['WT_uw'].dropna(), kde = False, color = 'royalblue', hist_kws = {'histtype': 'bar', 'linewidth': 3, 'alpha':0.2}, bins = 'auto', norm_hist = True)
sns.distplot(degrees_df['KO_uw'].dropna(), kde = False, bins = 'auto', norm_hist = True, color = 'green', hist_kws = {'histtype': 'step', 'linewidth': 3, 'alpha' : 1}, label = 'KO')
sns.distplot(degrees_df['KO_uw'].dropna(), kde = False, bins = 'auto', norm_hist = True, color = 'forestgreen', hist_kws = {'histtype': 'bar', 'linewidth': 4, 'alpha': 0.2})
plt.xlabel('Node degree (k)', fontsize = 14)
plt.ylabel('P(k)', fontsize = 14)
plt.legend(loc = (.11,.83))
inset_axes(deg_1, width = '35%', height = 2.3)
wu_hist, wu_bins = np.histogram(degrees_df['WT_uw'].dropna(), bins = 'auto', density = True)
wu_p = 1-(np.cumsum(wu_hist))
ku_hist, ku_bins = np.histogram(degrees_df['KO_uw'].dropna(), bins = 'auto', density = True)
ku_p = 1-(np.cumsum(ku_hist))

plt.plot((wu_bins[:-1] + wu_bins[1:]) / 2, wu_p, color = 'royalblue', marker = '*', linestyle = '')
w_slope, w_intercept, wr_value, p_value, std_err = sp.stats.linregress(np.log((wu_bins[:-1] + wu_bins[1:]) / 2)[2:], np.log(wu_p)[2:])
plt.plot(((wu_bins[:-1] + wu_bins[1:]) / 2)[1:-2], ((wu_bins[:-1] + wu_bins[1:]) / 2)[1:-2]**w_slope* np.exp(w_intercept), 'b--')

plt.plot((ku_bins[:-1] + ku_bins[1:]) / 2, ku_p, color = 'green', marker = '*', linestyle = '')
k_slope, k_intercept, kr_value, p_value, std_err = sp.stats.linregress(np.log((ku_bins[:-1] + ku_bins[1:]) / 2)[:], np.log(ku_p)[:])
k_slope, k_intercept, kr_value, p_value, std_err
plt.plot(((ku_bins[:-1] + ku_bins[1:]) / 2)[:], ((ku_bins[:-1] + ku_bins[1:]) / 2)[:]**k_slope * np.exp(k_intercept), 'g--')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('log(k)', fontsize = 12)
plt.ylabel('log P(degree > k)', fontsize = 12)

deg_2 = plt.subplot(2,1,2)
sns.distplot(degrees_df['WT_w'].dropna(), kde = False, color = 'royalblue', hist_kws = {'histtype': 'step', 'linewidth': 3, 'alpha' : 1}, bins = 'auto', norm_hist = True)
sns.distplot(degrees_df['WT_w'].dropna(), kde = False, color = 'royalblue', hist_kws = {'histtype': 'bar', 'linewidth': 3, 'alpha':0.2}, bins = 'auto', norm_hist = True)
sns.distplot(degrees_df['KO_w'].dropna(), kde = False, bins = 'auto', norm_hist = True, color = 'green', hist_kws = {'histtype': 'step', 'linewidth': 3, 'alpha' : 1})
sns.distplot(degrees_df['KO_w'].dropna(), kde = False, bins = 'auto', norm_hist = True, color = 'forestgreen', hist_kws = {'histtype': 'bar', 'linewidth': 4, 'alpha': 0.2})
plt.xlabel('Node Strength (s)', fontsize = 14)
plt.ylabel('Density', fontsize = 14)
inset_axes(deg_2, width = '35%', height = 2.3)
ww_hist, ww_bins = np.histogram(degrees_df['WT_w'].dropna(), bins = 'auto', density = True)
ww_p = 6-(np.cumsum(ww_hist))
kw_hist, kw_bins = np.histogram(degrees_df['KO_w'].dropna(), bins = 'auto', density = True)
kw_p = 6-(np.cumsum(kw_hist))
plt.plot((ww_bins[:-1] + ww_bins[1:]) / 2, ww_p, color = 'royalblue', marker = '*', linestyle = '')
ww_hist, kw_bins
w_slope, w_intercept, wr_value, p_value, std_err = sp.stats.linregress(np.log((ww_bins[:-1] + ww_bins[1:]) / 2)[3:], np.log(ww_p)[3:])
plt.plot(((ww_bins[:-1] + ww_bins[1:]) / 2)[2:], ((ww_bins[:-1] + ww_bins[1:]) / 2)[2:]**w_slope * np.exp(w_intercept), 'b--')

plt.plot((kw_bins[:-1] + kw_bins[1:]) / 2, kw_p, color = 'green', marker = '*', linestyle = '')
k_slope, k_intercept, kr_value, p_value, std_err = sp.stats.linregress(np.log((kw_bins[:-1] + kw_bins[1:]) / 2)[2:], np.log(kw_p)[2:])
plt.plot(((kw_bins[:-1] + kw_bins[1:]) / 2)[:], ((kw_bins[:-1] + kw_bins[1:]) / 2)[:]**k_slope * np.exp(k_intercept), 'g--')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('log(s)', fontsize = 12)
plt.ylabel('log P(Strength > s)', fontsize = 12)
#%%
degrees_fig.savefig('/Users/felipeantoniomendezsalcido/Desktop/Nodes_distribution_ind.png')

# So first no. nodes, no. edges, av. degree, density, connectivity, clustering, degree centrality, av. shortest path, global efficiency
# Might need an obnibus tests, most likely ANOVA
g_dic = {'Date': [], 'Gen_type': [], 'No. Nodes': [], 'No. Edges': [], 'Density': [], 'Av. Degree': [], 'Node Conn': [], 'Clustering': [], 'Clustering_W': [], 'Char Path' : [], 'Char Path_w' : [], 'Global Eff' : [], 'Major C': [], 'Degree Cent' : []}

for ii in g_names:
    print(ii)
    if graph_df['Gen_type'][graph_df['Date']== ii].unique()=='WT':
        g_df = graph_df[graph_df['Date'] == ii]
        g_th = g_df[['Node_1', 'Node_2', 'STTC_weight']][g_df['STTC_weight'] >= WT_pos_th]
        g_th['STTC_weight'] = g_th['STTC_weight'].round(3)
        g_th['STTC_rec'] = 1/g_th['STTC_weight'].round(3)
        temp_g = nx.from_pandas_edgelist(df = g_th, source = 'Node_1', target = 'Node_2',  edge_attr = True)
    elif graph_df['Gen_type'][graph_df['Date']== ii].unique()=='KO':
        g_df = graph_df[graph_df['Date'] == ii]
        g_th = g_df[['Node_1', 'Node_2', 'STTC_weight']][g_df['STTC_weight'] >= KO_pos_th]
        g_th['STTC_weight'] = g_th['STTC_weight'].round(3)
        g_th['STTC_rec'] = 1/g_th['STTC_weight'].round(3)
        temp_g = nx.from_pandas_edgelist(df = g_th, source = 'Node_1', target = 'Node_2',  edge_attr = True)
    if nx.is_empty(temp_g) == True:
        print(ii, 'is empty!!!!')

    else:
        g_dic['Date'].append(g_df['Date'].unique()[0])
        g_dic['Gen_type'].append(g_df['Gen_type'].unique()[0])
        g_dic['No. Nodes'].append(nx.number_of_nodes(temp_g))
        g_dic['No. Edges'].append(nx.number_of_edges(temp_g))
        g_dic['Density'].append(nx.density(temp_g))
        g_dic['Av. Degree'].append(round(sum(dict(temp_g.degree()).values())/float(nx.number_of_nodes(temp_g)), 2))
        g_dic['Degree Cent'].append(round(sum(dict(nx.degree_centrality(temp_g)).values())/float(nx.number_of_nodes(temp_g)), 2))
        g_dic['Node Conn'].append(nx.node_connectivity(temp_g))
        g_dic['Clustering'].append(nx.average_clustering(temp_g))
        g_dic['Clustering_W'].append(nx.average_clustering(temp_g, weight = 'STTC_rec'))
        if nx.is_connected(temp_g) == True:
            g_dic['Char Path'].append(nx.average_shortest_path_length(temp_g))
            g_dic['Char Path_w'].append(nx.average_shortest_path_length(temp_g, weight = 'STTC_rec'))
            g_dic['Global Eff'].append(nx.global_efficiency(temp_g))
            g_dic['Major C'].append(nx.number_of_nodes(temp_g)/nx.number_of_nodes(temp_g)*100)
        else:
            print(ii, 'is not connected')
            Gc = max(nx.connected_component_subgraphs(temp_g), key=len)
            g_dic['Char Path'].append(nx.average_shortest_path_length(Gc))
            g_dic['Char Path_w'].append(nx.average_shortest_path_length(Gc, weight = 'STTC_rec'))
            g_dic['Global Eff'].append(nx.global_efficiency(Gc))
            g_dic['Major C'].append(nx.number_of_nodes(Gc)/nx.number_of_nodes(temp_g)*100)

g_df = pd.DataFrame(g_dic)
g_df
g_df = g_df.drop(24) # outlier, only 10 nodes

# PROPORTION OF CONNECTED AND FRAGMENTED GRPAHS
wt_con = g_df['Gen_type'][(g_df['Major C']== 100) & (g_df['Gen_type'] == 'WT')].count()
wt_disc = g_df['Gen_type'][(g_df['Major C']< 100) & (g_df['Gen_type'] == 'WT')].count()
ko_con = g_df['Gen_type'][(g_df['Major C']== 100) & (g_df['Gen_type'] == 'KO')].count()
ko_disc = g_df['Gen_type'][(g_df['Major C']< 100) & (g_df['Gen_type'] == 'KO')].count()

wt_con, wt_disc, ko_con, ko_disc

cont_tab = np.array([[wt_con, wt_disc], [ko_con, ko_disc]])
sp.stats.chi2_contingency(cont_tab)

disconnected_Nxs = (g_df['Gen_type'][(g_df['Major C']< 100) & (g_df['Gen_type'] == 'WT')].count()/g_df['Gen_type'][g_df['Gen_type'] == 'WT'].count()).round(2), (g_df['Gen_type'][(g_df['Major C']< 100) & (g_df['Gen_type'] == 'KO')].count()/g_df['Gen_type'][g_df['Gen_type'] == 'KO'].count()).round(2)

connected_Nxs = (g_df['Gen_type'][(g_df['Major C']== 100) & (g_df['Gen_type'] == 'WT')].count()/g_df['Gen_type'][g_df['Gen_type'] == 'WT'].count()).round(2), (g_df['Gen_type'][(g_df['Major C']== 100) & (g_df['Gen_type'] == 'KO')].count()/g_df['Gen_type'][g_df['Gen_type'] == 'KO'].count()).round(2)


rep_graphs = ['151117', '161117', '61217', '22117', '140218(b)', '140218']

# CONNECTED/DISCONNECTED FIGURE
#%%
frag_prop = plt.figure(figsize = (12,6))
fr_ax = plt.subplot2grid((2, 4), (0, 0), rowspan = 2)
plt.bar((1, 1.5), disconnected_Nxs, color = ['cornflowerblue', 'darkseagreen'], width = .3, edgecolor = ['b', 'g'], linewidth = 4, label = 'Disconnected')
plt.bar((1, 1.5), connected_Nxs, bottom = disconnected_Nxs, color = ['b', 'green'], width = .3, edgecolor = ['b', 'g'], linewidth = 4, label = 'Connected')
plt.xticks((1,1.5), ['WT', 'KO'], fontsize = 12)
plt.xlim(0.5, 2)
plt.ylabel('Proportion', fontsize = (12))
fr_ax.spines['top'].set_visible(False)
fr_ax.spines['right'].set_visible(False)
fr_ax.annotate('ns', xy=(0.5, 1.01), xytext=(0.5, 1.03), xycoords='axes fraction',fontsize= 18, ha='center', va='bottom', arrowprops=dict(arrowstyle='-[, widthB=2.5, lengthB=0.5', lw=1.5))
fr_ax.annotate('C', xy=(.26, .85), xytext=(.3, .87), xycoords='axes fraction',fontsize= 12, color = 'white', fontweight ='bold')
fr_ax.annotate('DC', xy=(.26, .2), xytext=(.27, .35), xycoords='axes fraction',fontsize= 12, color = 'white', fontweight = 'bold')
fr_ax.annotate('C', xy=(.26, .85), xytext=(.64, .87), xycoords='axes fraction',fontsize= 12, color = 'white', fontweight ='bold')
fr_ax.annotate('DC', xy=(.26, .2), xytext=(.61, .35), xycoords='axes fraction',fontsize= 12, color = 'white', fontweight = 'bold')
other_ax = plt.subplot2grid((2,2), (0, 1), rowspan = 2)
_r = 1
_wc = 1
_kc = 1
for ii in rep_graphs:
    g_df = graph_df[graph_df['Date'] == ii]
    g_th = g_df[['Node_1', 'Node_2', 'STTC_weight']][g_df['STTC_weight'] >= Gl_th]
    g_th['STTC_weight'] = g_th['STTC_weight'].round(3)
    temp_g = nx.from_pandas_edgelist(df = g_th, source = 'Node_1', target = 'Node_2',  edge_attr = True)

    if g_df['Gen_type'].unique() == 'KO':
        _r = 1
        kx = plt.subplot2grid((2, 4), (_r, _kc))
        nx.draw_networkx(temp_g, node_size = 15, node_color = 'darkseagreen', linewidths = .5, width = .4, ax = kx, with_labels = False)
        kx.collections[0].set_edgecolor("k")
        plt.xticks([])
        plt.yticks([])
        _kc += 1
    elif g_df['Gen_type'].unique() == 'WT':
        _r = 0
        wx =  plt.subplot2grid((2,4), (_r, _wc))
        nx.draw_networkx(temp_g, node_size = 15, node_color = 'dodgerblue', linewidths = .5, width = .4 , ax = wx, with_labels = False)
        wx.collections[0].set_edgecolor("k")
        plt.xticks([])
        plt.yticks([])
        _wc += 1
#%%
frag_prop.savefig('/Users/felipeantoniomendezsalcido/Desktop/Fragmentation_ind_fig.png')

# Clustering, Weighted Cluestring, Charactheristic Path (W) and Global Eff
#%%
g_paramsFig = plt.figure(figsize=(12, 8))
dens_ax = plt.subplot(2,3,1)
sns.barplot('Gen_type', 'Density', data = g_df, ci = 68, order = ['WT', 'KO'], palette= ['royalblue', 'forestgreen'], capsize = .2)
p_stat = sp.stats.ttest_ind(g_df['Density'][g_df['Gen_type'] == 'WT'], g_df['Density'][g_df['Gen_type'] == 'KO'])[1]
dens_ax.annotate(r'$p = {}$'.format(p_stat.round(3)), xy=(0.5, .93), xytext=(0.5, .93), xycoords='axes fraction',fontsize= 14, ha='center', va='bottom')
sns.despine()
plt.title('Density', fontsize = 14)
plt.ylabel('')
plt.xlabel('')
clus_ax = plt.subplot(2, 3, 2)
sns.barplot('Gen_type', 'Clustering', data = g_df, ci = 68, order = ['WT', 'KO'], palette= ['royalblue', 'forestgreen'], capsize = .2)
p_stat = sp.stats.ttest_ind(g_df['Clustering'][g_df['Gen_type'] == 'WT'], g_df['Clustering'][g_df['Gen_type'] == 'KO'])[1]
clus_ax.annotate(r'$p = {}$'.format(p_stat.round(3)), xy=(0.5, .93), xytext=(0.5, .93), xycoords='axes fraction',fontsize= 14, ha='center', va='bottom')
sns.despine()
plt.title('Clustering', fontsize = 14)
plt.ylabel('')
plt.xlabel('')
clusW_ax = plt.subplot(2,3,3)
sns.barplot('Gen_type', 'Clustering_W', data = g_df, ci = 68, order = ['WT', 'KO'], palette= ['royalblue', 'forestgreen'], capsize = .2)
p_stat = sp.stats.ttest_ind(g_df['Clustering_W'][g_df['Gen_type'] == 'WT'], g_df['Clustering_W'][g_df['Gen_type'] == 'KO'], equal_var = True)[1]
clusW_ax.annotate(r'$p = {}$'.format(p_stat.round(3)), xy=(0.5, .95), xytext=(0.5, .93), xycoords='axes fraction',fontsize= 14, ha='center', va='bottom')
sns.despine()
plt.title('Clustering (Weigthed)', fontsize = 14)
plt.ylabel('')
plt.xlabel('')
chpth_ax = plt.subplot(2,3,4)
sns.barplot('Gen_type', 'Char Path', data = g_df, ci = 68, order = ['WT', 'KO'], palette= ['royalblue', 'forestgreen'], capsize = .2)
p_stat = sp.stats.ttest_ind(g_df['Char Path'][g_df['Gen_type'] == 'WT'], g_df['Char Path'][g_df['Gen_type'] == 'KO'], equal_var = True)[1]
chpth_ax.annotate(r'$p = {}$'.format(p_stat.round(3)), xy=(0.7, .95), xytext=(0.5, .93), xycoords='axes fraction',fontsize= 14, ha='center', va='bottom')
sns.despine()
plt.title('Charc. Path', fontsize = 14)
plt.ylabel('')
plt.xlabel('')
chpthW_ax = plt.subplot(2,3,5)
sns.barplot('Gen_type', 'Char Path_w', data = g_df, ci = 68, order = ['WT', 'KO'], palette= ['royalblue', 'forestgreen'], capsize = .2)
p_stat = sp.stats.ttest_ind(g_df['Char Path_w'][g_df['Gen_type'] == 'WT'], g_df['Char Path_w'][g_df['Gen_type'] == 'KO'], equal_var = True)[1]
chpthW_ax.annotate(r'$p = {}$'.format(p_stat.round(3)), xy=(0.7, .95), xytext=(0.5, .93), xycoords='axes fraction',fontsize= 14, ha='center', va='bottom')
sns.despine()
plt.title('Charc. Path (Weighted)', fontsize = 14)
plt.ylabel('')
plt.xlabel('')
gloeff_ax = plt.subplot(2, 3, 6)
sns.barplot('Gen_type', 'Global Eff', data = g_df, ci = 68, order = ['WT', 'KO'], palette= ['royalblue', 'forestgreen'], capsize = .2)
p_stat = sp.stats.ttest_ind(g_df['Global Eff'][g_df['Gen_type'] == 'WT'], g_df['Global Eff'][g_df['Gen_type'] == 'KO'], equal_var = True)[1]
gloeff_ax.annotate(r'$p = {}$'.format(p_stat.round(3)), xy=(0.7, .95), xytext=(0.5, .93), xycoords='axes fraction',fontsize= 14, ha='center', va='bottom')
sns.despine()
plt.title('Global Efficiency', fontsize = 14)
plt.ylabel('')
plt.xlabel('')
#%%
g_paramsFig.savefig('/Users/felipeantoniomendezsalcido/Desktop/Graph_paramsFig2_ind.png')


# GRAPH  CATALOG; ALL INDIVIDUAL GRAPHS
#%%
all_gFig = plt.figure(figsize= (14,14))

_r = 0
_c = 0
for ii in g_names:
    g_df = graph_df[graph_df['Date'] == ii]
    g_th = g_df[['Node_1', 'Node_2', 'STTC_weight']][g_df['STTC_weight'] >= Gl_th]
    g_th['STTC_weight'] = g_th['STTC_weight'].round(3)
    temp_g = nx.from_pandas_edgelist(df = g_th, source = 'Node_1', target = 'Node_2',  edge_attr = True)

    if g_df['Gen_type'].unique() == 'KO':
        kx = plt.subplot2grid((7, 4), (_r, _c))
        nx.draw_networkx(temp_g, node_size = 15, node_color = 'darkseagreen', linewidths = .5, width = .4, ax = kx, with_labels = False)
        kx.collections[0].set_edgecolor("k")
        plt.title(ii+g_df['Gen_type'].unique())
        plt.xticks([])
        plt.yticks([])
        _r += 1

    elif g_df['Gen_type'].unique() == 'WT':
        wx =  plt.subplot2grid((7,4), (_r, _c))
        nx.draw_networkx(temp_g, node_size = 15, node_color = 'dodgerblue', linewidths = .5, width = .4 , ax = wx, with_labels = False)
        wx.collections[0].set_edgecolor("k")
        plt.title(ii+g_df['Gen_type'].unique())
        plt.xticks([])
        plt.yticks([])
        _r += 1
    if _r == 7:
        _r = 0
        _c += 1

#%%
all_gFig.savefig('/Users/felipeantoniomendezsalcido/Desktop/All_graphs_ind.png')
