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

df = pd.concat([epu, cpu, macro_data_mod], axis=1)

# Retrieve the standardized dataset

df_std = (df - df.mean())/df.std()
# p=2; df_std = sm.tsa.tsatools.lagmat(df_std, maxlag=p, use_pandas=True).iloc[p:]
y = pd.concat([epu, cpu, macro_data], axis=1)

# Forecasting
# Horizon "in the middle"
H = 40

def histoiOmega(macro_condition):
    if macro_condition == "general":
        histoi = df_std.iloc[-(H+1):].mean()
        omega = df_std.iloc[:-(H+1)]
    elif macro_condition == "great_recession":
        histoi = df_std.loc['2008-11-01':'2009-10-01'].mean()
        omega = pd.concat([df_std.loc[:'2008-10-01'], df_std.loc['2009-11-01':]])
    elif macro_condition == "recessionary":
        histoi = df_std.loc[y.loc[y['Unemployment_Rate'] >= 5.5].index].mean()
        omega = df_std.loc[y.loc[y['Unemployment_Rate'] >= 5.5].index]
    elif macro_condition == "booming":
        histoi = df_std.loc[y.loc[y['Unemployment_Rate'] < 5.5].index].mean()
        omega = df_std.loc[y.loc[y['Unemployment_Rate'] < 5.5].index]
    elif macro_condition == "LowCPU":
        histoi = df_std.loc[y.loc[y['cpu_index'] < 100].index].mean()
        omega = df_std.loc[y.loc[y['cpu_index'] < 100].index]
    elif macro_condition == "HighCPU":
        histoi = df_std.loc[y.loc[y['cpu_index'] >= 100].index].mean()
        omega = df_std.loc[y.loc[y['cpu_index'] >= 100].index]
    else:
        histoi = df_std.iloc[-1]
        omega = df_std.iloc[:-(H+1)]
    return (histoi, omega)

(histoi, omega) = histoiOmega("HighCPU")

df = df.dropna()
df_std = df_std.dropna()
omega = omega.dropna()
omega = omega.loc[:y.index[-1] - pd.DateOffset(months=H)]

T = omega.shape[0]

knn = NearestNeighbors(n_neighbors=50, metric='euclidean')
knn.fit(omega)
dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
# Estimate y_T
y_f = np.matmul(y.loc[omega.iloc[ind].index].T, weig).to_frame().T

u = y.loc[omega.iloc[ind].index] - y_f.values.squeeze()
u_mean = u.mul(weig, axis = 0).mean()
sigma_u = np.matmul((u - u_mean).T, (u - u_mean).mul(weig, axis = 0)) / (1 - np.sum(weig**2))

# y_f.plot(
#     subplots=True, layout=(2,4), color = 'blue',
#     ax=y_f_delta.plot(
#         subplots=True, layout=(2,4), color = 'red'
#     )
# )
# plt.show()

for h in range(1,H+1):
    y_f.loc[h] = np.matmul(y.loc[omega.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
# dataplot(y_f)

# Cholesky decomposition
B_mat = np.linalg.cholesky(sigma_u)
# Note that sigma_u = residual_cov*((T-1)/(T-Kp-1))
# The desired shock
shock = 1
delta = B_mat[:,shock]

# Estimate y_T_delta
y_f_delta = pd.DataFrame(columns=y_f.columns)
y_f_delta.loc[0] = y_f.loc[0] + delta
# histoi_delta = (y_f_delta.loc[0] - y.mean())/y.std()

df_star = pd.concat([y.iloc[-2:-1], y_f_delta])
df_star[['Industrial_Production','PriceIndex_Producer','PriceIndex_PCE', 'Emission_CO2']] = np.log(df_star[[
    'Industrial_Production','PriceIndex_Producer','PriceIndex_PCE','Emission_CO2'
]]).diff().dropna()
df_star = (df_star - y.mean())/y.std()
histoi_delta = df_star.iloc[-1]

histoi_delta = pd.concat([histoi_delta, histoi], axis=0)[:-df.shape[1]]

knn.fit(omega)
dist, ind = knn.kneighbors(histoi_delta.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

for h in range(1,H+1):
    y_f_delta.loc[h] = np.matmul(y.loc[omega.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
# dataplot(y_f_delta)

girf = (y_f_delta - y_f)*(50/delta[shock])
dataplot(girf)

# Confidence Intervals
R=50
sim_girf = []

# Perform simulations
for r in range(0,R):
    omega_resamp = omega.sample(n=T, replace=True).sort_index()
    # Estimate y_T
    knn.fit(omega_resamp)
    dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    # Estimate y_T
    y_f_resamp = np.matmul(y.loc[omega_resamp.iloc[ind].index].T, weig).to_frame().T
    for h in range(1,H+1):
        y_f_resamp.loc[h] = np.matmul(y.loc[omega.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
    y_f_delta_resamp = pd.DataFrame(columns=y_f_resamp.columns)
    y_f_delta_resamp.loc[0] = y_f_resamp.loc[0] + delta
    histoi_delta = (y_f_delta.loc[0] - y.mean())/y.std()
    histoi_delta = pd.concat([histoi_delta, histoi], axis=0)[:-df.shape[1]]
    knn.fit(omega_resamp)
    dist, ind = knn.kneighbors(histoi_delta.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    for h in range(1,H+1):
        y_f_delta_resamp.loc[h] = np.matmul(y.loc[omega_resamp.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
    # dataplot(y_f_delta)
    sim_girf.append((y_f_delta_resamp - y_f_resamp)*(50/delta[shock]))
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
girfplot(df, girf_complete, multi_index_col, shock)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.figure(figsize = (10,25))
gs1 = gridspec.GridSpec(2, 4)
gs1.update(wspace=0.025, hspace=0.2) # set the spacing between axes. 
c=0
for i in range(8):
    ax1 = plt.subplot(gs1[i])
    # plt.axis('on')
    ax1.plot(girf_complete[multi_index_col[c][1]])
    ax1.set_title(y.columns[c])
    ax1.tick_params(axis="y",direction="in", pad=-20)
    c += 1
plt.tight_layout()
plt.show()

c=0
plt.figure(figsize = (10,25))
gs1 = gridspec.GridSpec(2, 4)
gs1.update(wspace=0.025, hspace=0.2) # set the spacing between axes. 
c=0
for i in range(8):
    ax1 = plt.subplot(gs1[i])
    # plt.axis('on')
    ax1.plot(girf_complete[multi_index_col[c][0]])
    ax1.plot(girf_complete[multi_index_col[c][1]])
    ax1.plot(girf_complete[multi_index_col[c][2]])
    ax1.title.set_text(df.columns[shock] + ">" + df.columns[c])
    ax1.tick_params(axis="y",direction="in", pad=-20)
    c += 1
plt.tight_layout()
plt.show()