# Import new and previous libraries, dataframe, variables, model, and IRF function.
from NonParametricIRF_Data import *
from Functions_Required import *
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

##################################################################################
############################# kNN Forecasting & GIRF #############################
##################################################################################
shock = 1
y = pd.concat([epu, cpu, macro_data], axis=1)
delta_y = y.copy()

mod = transformation_logdiff(delta_y[trend])

delta_y[trend] = mod.logdiff()
# p=1; pd.concat([delta_y, sm.tsa.tsatools.lagmat(delta_y[['Unemployment_Rate', 'Treasurey3Months']], maxlag=p, use_pandas=True).iloc[p:]], axis = 1)

# Forecasting
# Horizon "in the middle"
H = 40

def histoiOmega(macro_condition):
    # if macro_condition == "great_recession":
    #     histoi = delta_y.loc['2008-11-01':'2009-10-01'].mean()
    #     omega = pd.concat([delta_y.loc[:'2008-10-01'], delta_y.loc['2009-11-01':]])
    # elif macro_condition == "recession":
    #     omega = delta_y.loc[y.loc[y['Unemployment_Rate'] >= y['Unemployment_Rate'].mean()].index]
    #     histoi = omega.mean()
    # elif macro_condition == "expansion":
    #     omega = delta_y.loc[y.loc[y['Unemployment_Rate'] < y['Unemployment_Rate'].mean()].index]
    #     histoi = omega.mean()
    # elif macro_condition == "inflationary":
    #     omega = delta_y.loc[y.loc[delta_y['Growth_PriceIndex_PCE']>0.0025].index]
    #     histoi = omega.mean()
    # elif macro_condition == "LowCPU":
    #     omega = delta_y.loc[y.loc[y['cpu_index'] < y['cpu_index'].mean()].index]
    #     histoi = omega.mean()
    # elif macro_condition == "HighCPU":
    #     omega = delta_y.loc[y.loc[y['cpu_index'] >= y['cpu_index'].mean()].index]
    #     histoi = omega.mean()
    # elif macro_condition == "LowEPU":
    #     omega = delta_y.loc[y.loc[y['epu_index'] < y['epu_index'].mean()].index]
    #     histoi = omega.mean()
    # elif macro_condition == "HighEPU":
    #     omega = delta_y.loc[y.loc[y['epu_index'] >= y['epu_index'].mean()].index]
    #     histoi = omega.mean()
    if macro_condition == "High EPU - Recession":
        omega = delta_y.loc[y.loc[
            (y['epu_index'] >= y['epu_index'].mean()) & (y['Unemployment_Rate'] >= y['Unemployment_Rate'].mean())
        ].index]
        histoi = omega.mean()
    elif macro_condition == "High EPU - Expansion":
        omega = delta_y.loc[y.loc[
            (y['epu_index'] >= y['epu_index'].mean()) & (y['Unemployment_Rate'] < y['Unemployment_Rate'].mean())
        ].index]
        histoi = omega.mean()
    elif macro_condition == "Low EPU - Recession":
        omega = delta_y.loc[y.loc[
            (y['epu_index'] < y['epu_index'].mean()) & (y['Unemployment_Rate'] >= y['Unemployment_Rate'].mean())
        ].index]
        histoi = omega.mean()
    elif macro_condition == "Low EPU - Expansion":
        omega = delta_y.loc[y.loc[
            (y['epu_index'] < y['epu_index'].mean()) & (y['Unemployment_Rate'] < y['Unemployment_Rate'].mean())
        ].index]
        histoi = omega.mean()
    else:
        omega = delta_y.iloc[:-(H+1)]
        #histoi = delta_y.iloc[-1]
        histoi = omega.mean()
        print("Default history and omega.")
    return (histoi, omega)

interest = [
    "High EPU - Recession",
    "High EPU - Expansion",
    "Low EPU - Recession",
    "Low EPU - Expansion"][2]
(histoi, omega) = histoiOmega(interest+"1231")

# plt.figure(figsize = (25,8))
# # plt.plot(y[['epu_index']], color = 'black', linewidth = 2)
# plt.plot(y[['Unemployment_Rate']], color = 'black', linewidth = 2)
# plt.xticks(fontsize = 40)
# plt.yticks(fontsize = 40)
# for i in omega.index:
#     plt.axvspan(i, i+pd.DateOffset(months=1), color="silver")
# plt.show()

# delta_y = delta_y.dropna()
omega = omega.dropna()
omega = omega.loc[:y.index[-1] - pd.DateOffset(months=H)]
omega_mean = omega.mean()
omega_std = omega.std()
omega_scaled = (omega - omega_mean)/omega_std
histoi = (histoi - omega_mean)/omega_std
T = omega_scaled.shape[0]

knn = NearestNeighbors(n_neighbors=T, metric='euclidean')
knn.fit(omega_scaled)
dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
# Estimate y_T
y_f = np.matmul(delta_y.loc[omega_scaled.iloc[ind].index].T, weig).to_frame().T
# y_f = np.matmul(y.loc[omega_scaled.iloc[ind].index].T, weig).to_frame().T
for h in range(1,H+1):
    y_f.loc[h] = np.matmul(delta_y.loc[omega_scaled.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
# dataplot(y_f)

# omega_hat = pd.DataFrame(index=omega.index, columns=omega.columns)

# for t in omega.index:
#     knn.fit(omega_scaled.drop(t))
#     dist, ind = knn.kneighbors(omega_scaled.loc[t].to_numpy().reshape(1,-1))
#     dist = dist[0,:]; ind = ind[0,:]
#     weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
#     omega_hat.loc[t] = np.matmul(omega.loc[omega_scaled.iloc[ind].index].T, weig).to_frame().T

# u = omega - omega_hat

u = omega - y_f.loc[0].values.squeeze()
# u_mean = u.mul(weig, axis = 0)
u_mean = u.mean()
sigma_u = np.matmul((u - u_mean).T, (u - u_mean).mul(weig, axis = 0)) / (1 - np.sum(weig**2))
# u.sort_index().plot(subplots = True, layout = (2,4))

# Define the shock
# shock = 1
# Cholesky decomposition
B_mat = np.transpose(np.linalg.cholesky(sigma_u))
# The desired shock
# B_mat = np.transpose(np.linalg.cholesky(u.cov()*((T-1)/(T-8-1))))
delta = B_mat[shock]

# Estimate y_T_delta
y_f_delta = pd.DataFrame(columns=y_f.columns)
y_f_delta.loc[0] = y_f.loc[0] + delta

histoi_delta = (y_f.iloc[0] + delta - omega_mean.values)/omega_std.values
# histoi_delta = pd.concat([histoi_delta, histoi], axis=0)[:-delta_y.shape[1]]

dist, ind = knn.kneighbors(histoi_delta.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

for h in range(1,H+1):
    y_f_delta.loc[h] = np.matmul(delta_y.loc[omega_scaled.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
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
R=500
sim_girf = []

# Perform simulations
for r in range(0,R):
    omega_scaled_resamp = omega_scaled.sample(n=T, replace=True).sort_index()
    omega_scaled_resamp_mean = delta_y.loc[omega_scaled_resamp.index].mean()
    omega_scaled_resamp_sd = delta_y.loc[omega_scaled_resamp.index].std()
    # Estimate y_T
    knn.fit(omega_scaled_resamp)
    dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    # Estimate y_T
    y_f_resamp = np.matmul(delta_y.loc[omega_scaled_resamp.iloc[ind].index].T, weig).to_frame().T
    for h in range(1,H+1):
        y_f_resamp.loc[h] = np.matmul(delta_y.loc[omega_scaled_resamp.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
    
    y_f_delta_resamp = pd.DataFrame(columns=y_f_resamp.columns)
    y_f_delta_resamp.loc[0] = y_f_resamp.loc[0] + delta

    histoi_delta_resamp = (y_f_resamp.loc[0] - omega_scaled_resamp_mean)/omega_scaled_resamp_sd
    histoi_delta_resamp = pd.concat([histoi_delta, histoi], axis=0)[:-delta_y.shape[1]]
    dist, ind = knn.kneighbors(histoi_delta_resamp.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    for h in range(1,H+1):
        y_f_delta_resamp.loc[h] = np.matmul(delta_y.loc[omega_scaled_resamp.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
    
    girf_resamp = y_f_delta_resamp - y_f_resamp
    girf_resamp[trend] = mod.inv_logdiff_girf(girf_resamp[trend])
    sim_girf.append(girf_resamp)
    print('loop: '+str(r))
# End of loop, and now the sim_list_delta_y has each of the resampled dataframes

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
        girf_complete[col][h,'lower'] = 2*girf[col][h] - np.quantile([each_delta_y[col][h] for each_delta_y in sim_girf], conf+(1-conf)/2)
        girf_complete[col][h,'GIRF'] = girf[col][h]
        girf_complete[col][h,'upper'] = 2*girf[col][h] - np.quantile([each_delta_y[col][h] for each_delta_y in sim_girf], (1-conf)/2)
# girf_complete
girf_complete = girf_complete.astype('float')

girf_complete = girf_complete.unstack()
multi_index_col = [(girf_complete.columns[i], girf_complete.columns[i+1], girf_complete.columns[i+2]) for i in range(0,24,3)]
# Plot
# girfplot(delta_y, girf_complete*(50/girf.iloc[0,shock]), multi_index_col, shock)
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
plt.figure(figsize = (40,10))
gs1 = gridspec.GridSpec(2, 4)
gs1.update(wspace=0.2, hspace=0.5) # set the spacing between axes. 
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
    ax1.set_title(y.columns[c], size = 40)
    ax1.tick_params(axis="y",direction="in", pad=-20, labelsize=40)
    ax1.tick_params(axis="x",direction="in", pad=-20, labelsize=40)
    c += 1
plt.suptitle(
    y.columns[shock] + " shock in " + interest + " periods",
    fontsize=40
)
plt.tight_layout()
plt.show()




# girf_complete_high = girf_complete.copy()

# plt.figure(figsize = (25,10))
# gs1 = gridspec.GridSpec(2, 4)
# gs1.update(wspace=0.025, hspace=0.2) # set the spacing between axes. 
# c=0
# for i in range(8):
#     ax1 = plt.subplot(gs1[i])
#     # plt.axis('on')
#     ax1.plot(girf_complete[multi_index_col[c][1]])
#     ax1.plot(girf_complete_high[multi_index_col[c][1]], color = 'r')
#     ax1.fill_between(
#         np.arange(H+1),
#         girf_complete[multi_index_col[c][0]],
#         girf_complete[multi_index_col[c][2]],
#         alpha = 0.5
#     )
#     ax1.fill_between(
#         np.arange(H+1),
#         girf_complete_high[multi_index_col[c][0]],
#         girf_complete_high[multi_index_col[c][2]],
#         alpha = 0.5, color = 'r'
#     )
#     ax1.set_title(y.columns[c], size = 20)
#     ax1.tick_params(axis="y",direction="in", pad=-20, labelsize=20)
#     c += 1
# plt.suptitle(y.columns[shock] + " shock", fontsize=20)
# plt.tight_layout()
# plt.show()