# Import new and previous libraries, dataframe, variables, model, and IRF function.
from NonParametricIRF_Data import *
from Functions_Required import *
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

##################################################################################
############################# kNN Forecasting & GIRF #############################
##################################################################################
shock = 1
y = pd.concat([epu, cpu, macro_data], axis=1)
df = y.copy()

mod = transformation_logdiff(df[trend])

df[trend] = mod.logdiff()
# p=1; pd.concat([df, sm.tsa.tsatools.lagmat(df[['Unemployment_Rate', 'Treasurey3Months']], maxlag=p, use_pandas=True).iloc[p:]], axis = 1)

# Forecasting
# Horizon "in the middle"
H = 40

def histoiOmega(macro_condition):
    if macro_condition == "great_recession":
        histoi = df.loc['2008-11-01':'2009-10-01'].mean()
        omega = pd.concat([df.loc[:'2008-10-01'], df.loc['2009-11-01':]])
    elif macro_condition == "recession":
        omega = df.loc[y.loc[y['Unemployment_Rate'] >= y['Unemployment_Rate'].mean()].index]
        histoi = omega.mean()
    elif macro_condition == "expansion":
        omega = df.loc[y.loc[y['Unemployment_Rate'] < y['Unemployment_Rate'].mean()].index]
        histoi = omega.mean()
    elif macro_condition == "inflationary":
        omega = df.loc[y.loc[df['Growth_PriceIndex_PCE']>0.0025].index]
        histoi = omega.mean()
    elif macro_condition == "LowCPU":
        omega = df.loc[y.loc[y['cpu_index'] < y['cpu_index'].mean()].index]
        histoi = omega.mean()
    elif macro_condition == "HighCPU":
        omega = df.loc[y.loc[y['cpu_index'] >= y['cpu_index'].mean()].index]
        histoi = omega.mean()
    elif macro_condition == "LowEPU":
        omega = df.loc[y.loc[y['epu_index'] < y['epu_index'].mean()].index]
        histoi = omega.mean()
    elif macro_condition == "HighEPU":
        omega = df.loc[y.loc[y['epu_index'] >= y['epu_index'].mean()].index]
        histoi = omega.mean()
    elif macro_condition == "High EPU - Recession":
        omega = df.loc[y.loc[
            (y['epu_index'] >= y['epu_index'].mean()) & (y['Unemployment_Rate'] >= y['Unemployment_Rate'].mean())
        ].index]
        histoi = omega.mean()
    elif macro_condition == "High EPU - Expansion":
        omega = df.loc[y.loc[
            (y['epu_index'] >= y['epu_index'].mean()) & (y['Unemployment_Rate'] < y['Unemployment_Rate'].mean())
        ].index]
        histoi = omega.mean()
    elif macro_condition == "Low EPU - Recession":
        omega = df.loc[y.loc[
            (y['epu_index'] < y['epu_index'].mean()) & (y['Unemployment_Rate'] >= y['Unemployment_Rate'].mean())
        ].index]
        histoi = omega.mean()
    elif macro_condition == "Low EPU - Expansion":
        omega = df.loc[y.loc[
            (y['epu_index'] < y['epu_index'].mean()) & (y['Unemployment_Rate'] < y['Unemployment_Rate'].mean())
        ].index]
        histoi = omega.mean()
    else:
        omega = df.iloc[:-(H+1)]
        #histoi = df.iloc[-1]
        histoi = omega.mean()
        print("Default history and omega.")
    return (histoi, omega)

interest = "Low EPU - Expansion"
(histoi, omega) = histoiOmega(interest)

# plt.figure(figsize = (12,10))
# # plt.plot(y[['epu_index']], color = 'black', linewidth = 2)
# plt.plot(y[['Unemployment_Rate']], color = 'black', linewidth = 2)
# plt.xticks(fontsize = 25)
# plt.yticks(fontsize = 25)
# for i in omega.index:
#     plt.axvspan(i, i+pd.DateOffset(months=1), color="silver")
# plt.show()

df = df.dropna()
omega = omega.dropna()
omega = omega.loc[:y.index[-1] - pd.DateOffset(months=H)]
omega_mean = omega.mean()
omega_std = omega.std()
omega = (omega - omega_mean)/omega_std
histoi = (histoi - omega_mean)/omega_std
T = omega.shape[0]

knn = NearestNeighbors(n_neighbors=T-H, metric='euclidean')
knn.fit(omega)
dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
# Estimate y_T
y_f = np.matmul(df.loc[omega.iloc[ind].index].T, weig).to_frame().T
# y_f = np.matmul(y.loc[omega.iloc[ind].index].T, weig).to_frame().T

u = df.loc[omega.iloc[ind].index] - y_f.values.squeeze()
u_mean = u.mul(weig, axis = 0)
sigma_u = np.matmul((u - u_mean).T, (u - u_mean).mul(weig, axis = 0)) / (1 - np.sum(weig**2))

# u = y.loc[omega.iloc[ind].index] - y_f.values.squeeze()
# u_mean = u.mul(weig, axis = 0)
# sigma_u = np.matmul((u - u_mean).T, (u - u_mean).mul(weig, axis = 0)) / (1 - np.sum(weig**2))

# Define the shock
# shock = 1
# Cholesky decomposition
B_mat = np.transpose(np.linalg.cholesky(sigma_u))
# The desired shock
delta = B_mat[shock]

for h in range(1,H+1):
    y_f.loc[h] = np.matmul(df.loc[omega.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
# dataplot(y_f)

# Estimate y_T_delta
y_f_delta = pd.DataFrame(columns=y_f.columns)
y_f_delta.loc[0] = y_f.loc[0] + delta

histoi_delta = (y_f.iloc[0] + delta - omega_mean.values)/omega_std.values
# histoi_delta = pd.concat([histoi_delta, histoi], axis=0)[:-df.shape[1]]

dist, ind = knn.kneighbors(histoi_delta.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

for h in range(1,H+1):
    y_f_delta.loc[h] = np.matmul(df.loc[omega.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
# dataplot(y_f_delta)

girf = y_f_delta - y_f
girf[trend] = mod.inv_logdiff_girf(girf[trend])
# dataplot(girf*(50/girf.iloc[0,shock]))
# dataplot(girf)

# y_f.plot(
#     subplots=True, layout=(2,4), color = 'blue',
#     ax=y_f_delta.plot(
#         subplots=True, layout=(2,4), color = 'red'
#     )
# )
# plt.show()

# Confidence Intervals
R=50
sim_girf = []

# Perform simulations
for r in range(0,R):
    omega_resamp = omega.sample(n=T, replace=True).sort_index()
    omega_resamp_mean = df.loc[omega_resamp.index].mean()
    omega_resamp_sd = df.loc[omega_resamp.index].std()
    # Estimate y_T
    knn.fit(omega_resamp)
    dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    # Estimate y_T
    y_f_resamp = np.matmul(df.loc[omega_resamp.iloc[ind].index].T, weig).to_frame().T
    for h in range(1,H+1):
        y_f_resamp.loc[h] = np.matmul(df.loc[omega_resamp.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
    
    y_f_delta_resamp = pd.DataFrame(columns=y_f_resamp.columns)
    y_f_delta_resamp.loc[0] = y_f_resamp.loc[0] + delta

    histoi_delta_resamp = (y_f_resamp.loc[0] - omega_resamp_mean)/omega_resamp_sd
    histoi_delta_resamp = pd.concat([histoi_delta, histoi], axis=0)[:-df.shape[1]]
    dist, ind = knn.kneighbors(histoi_delta_resamp.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    for h in range(1,H+1):
        y_f_delta_resamp.loc[h] = np.matmul(df.loc[omega_resamp.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
    
    girf_resamp = y_f_delta_resamp - y_f_resamp
    girf_resamp[trend] = mod.inv_logdiff_girf(girf_resamp[trend])
    sim_girf.append(girf_resamp)
    print('loop: '+str(r))
# End of loop, and now the sim_list_df has each of the resampled dataframes

conf = 0.90
# Define the multi-index dataframe for each horizon and CI for each column
girf_complete = pd.DataFrame(
    columns = y.columns,
    index = pd.MultiIndex(
        levels=[range(0,H+1),['lower','GIRF','upper']],
        codes=[[x//3 for x in range(0,(H+1)*3)],[0,1,2]*(H+1)], names=('Horizon', 'CI')
    )
)

for h in range(0,H+1):
    for col in y.columns:
        girf_complete[col][h,'lower'] = 2*girf[col][h] - np.quantile([each_df[col][h] for each_df in sim_girf], conf+(1-conf)/2)
        girf_complete[col][h,'GIRF'] = girf[col][h]
        girf_complete[col][h,'upper'] = 2*girf[col][h] - np.quantile([each_df[col][h] for each_df in sim_girf], (1-conf)/2)
# girf_complete
girf_complete = girf_complete.astype('float')

girf_complete = girf_complete.unstack()
multi_index_col = [(girf_complete.columns[i], girf_complete.columns[i+1], girf_complete.columns[i+2]) for i in range(0,24,3)]
# Plot
# girfplot(df, girf_complete*(50/girf.iloc[0,shock]), multi_index_col, shock)
#

girf_complete = girf_complete*(50/girf.iloc[0,shock])

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# plt.figure(figsize = (10,25))
# gs1 = gridspec.GridSpec(2, 4)
# gs1.update(wspace=0.025, hspace=0.2) # set the spacing between axes.
# c=0
# for i in range(8):
#     ax1 = plt.subplot(gs1[i])
#     # plt.axis('on')
#     ax1.plot(girf_complete[multi_index_col[c][1]])
#     ax1.set_title(y.columns[c])
#     ax1.tick_params(axis="y",direction="in", pad=-20)
#     c += 1
# plt.tight_layout()
# plt.show()
# , color = 'r'
plt.figure(figsize = (25,10))
gs1 = gridspec.GridSpec(2, 4)
gs1.update(wspace=0.025, hspace=0.2) # set the spacing between axes. 
c=0
for i in range(8):
    ax1 = plt.subplot(gs1[i])
    # plt.axis('on')
    ax1.plot(girf_complete[multi_index_col[c][1]], color='black')
    ax1.fill_between(
        np.arange(H+1),
        girf_complete[multi_index_col[c][0]],
        girf_complete[multi_index_col[c][2]],
        color = 'lightgrey'
    )
    ax1.set_title(y.columns[c], size = 20)
    ax1.tick_params(axis="y",direction="in", pad=-20, labelsize=20)
    c += 1
plt.suptitle(
    y.columns[shock] + " shock in " + interest + " periods",
    fontsize=20
)
plt.tight_layout()
plt.show()




girf_complete_high = girf_complete.copy()

plt.figure(figsize = (25,10))
gs1 = gridspec.GridSpec(2, 4)
gs1.update(wspace=0.025, hspace=0.2) # set the spacing between axes. 
c=0
for i in range(8):
    ax1 = plt.subplot(gs1[i])
    # plt.axis('on')
    ax1.plot(girf_complete[multi_index_col[c][1]])
    ax1.plot(girf_complete_high[multi_index_col[c][1]], color = 'r')
    ax1.fill_between(
        np.arange(H+1),
        girf_complete[multi_index_col[c][0]],
        girf_complete[multi_index_col[c][2]],
        alpha = 0.5
    )
    ax1.fill_between(
        np.arange(H+1),
        girf_complete_high[multi_index_col[c][0]],
        girf_complete_high[multi_index_col[c][2]],
        alpha = 0.5, color = 'r'
    )
    ax1.set_title(y.columns[c], size = 20)
    ax1.tick_params(axis="y",direction="in", pad=-20, labelsize=20)
    c += 1
plt.suptitle(y.columns[shock] + " shock", fontsize=20)
plt.tight_layout()
plt.show()