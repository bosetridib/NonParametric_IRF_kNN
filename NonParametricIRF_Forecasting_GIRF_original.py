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

trend = 0

df = pd.concat([epu, cpu, macro_data], axis=1) if trend == 1 else pd.concat([epu, cpu, macro_data_mod], axis=1)
df = df.dropna()

# Retrieve the standardized dataset
df_std = (df - df.mean())/df.std()
df_std = sm.tsa.tsatools.lagmat(df_std, maxlag=3, use_pandas=True)
y = df.copy()

# Forecasting
# Horizon "in the middle"
H = 40
omega = df_std.iloc[:-1]
# omega = df_std.copy()
histoi = df_std.iloc[-1]

T = omega.shape[0]
k = omega.shape[0]
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')

# Estimate y_T
dist = np.array([euclidean(omega.loc[i], histoi) for i in omega.index])
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
# Estimated (NOT forecasted) the period of interest T
y_f = np.matmul(y.loc[omega.index].T, weig).to_frame().T

u = y.loc[omega.index] - y_f.values.squeeze()
u_mean = u.mul(weig, axis = 0).mean()
sigma_u = np.matmul((u - u_mean).T, (u - u_mean).mul(weig, axis = 0)) / (1 - np.sum(weig**2))

# y.plot(
#     subplots=True, layout=(2,4), color = 'blue',
#     ax=u.plot(
#         subplots=True, layout=(2,4), color = 'red'
#     )
# )
# plt.show()

for h in range(1,H+1):
    y_f.loc[h] = np.matmul(y.loc[omega.index[h:]].T, weig[:-h]).values
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
histoi_delta = (y_f_delta.loc[0] - df.mean())/df.std()
histoi_delta = pd.concat([histoi_delta, df_std.iloc[-1]], axis=0)[:-8]

dist = np.array([euclidean(omega.loc[i], histoi_delta) for i in omega.index])
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

for h in range(1,H+1):
    y_f_delta.loc[h] = np.matmul(y.loc[omega.index[h:]].T, weig[:-h]).values
# dataplot(y_f_delta)

girf = y_f_delta - y_f
dataplot(girf)
dataplot(np.exp(girf.cumsum()))

# Confidence Intervals
R=1000
sim_girf = []

# Perform simulations
for r in range(0,R):
    omega_resamp = omega.sample(n=T, replace=True).sort_index()
    # Estimate y_T
    dist = np.array([euclidean(omega_resamp.iloc[i], histoi) for i in range(0,omega_resamp.shape[0])])
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    # Estimated (NOT forecasted) the period of interest T
    y_f_resamp = y_f.loc[0].to_frame().T
    for h in range(1,H+1):
        y_f_resamp.loc[h] = np.matmul(y.loc[omega_resamp.index[h:]].T, weig[:-h]).values
    y_f_delta_resamp = y_f_delta.loc[0].to_frame().T
    dist = np.array([euclidean(omega_resamp.iloc[i], histoi_delta) for i in range(0,omega_resamp.shape[0])])
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    for h in range(1,H+1):
        y_f_delta_resamp.loc[h] = np.matmul(y.loc[omega_resamp.index[h:]].T, weig[:-h]).values
    sim_girf.append(y_f_delta_resamp - y_f_resamp)
# End of loop, and now the sim_list_df has each of the resampled dataframes

conf = 0.90
# Define the multi-index dataframe for each horizon and CI for each column
girf_complete = pd.DataFrame(
    columns = omega.columns,
    index = pd.MultiIndex(
        levels=[range(0,H+1),['lower','GIRF','upper']],
        codes=[[x//3 for x in range(0,(H+1)*3)],[0,1,2]*(H+1)], names=('Horizon', 'CI')
    )
)

for h in range(0,H+1):
    for col in omega.columns:
        girf_complete[col][h,'lower'] = 2*girf[col][h] - np.quantile([each_df[col][h] for each_df in sim_girf], conf+(1-conf)/2)
        girf_complete[col][h,'GIRF'] = girf[col][h]
        girf_complete[col][h,'upper'] = 2*girf[col][h] - np.quantile([each_df[col][h] for each_df in sim_girf], (1-conf)/2)
# girf_complete
girf_complete = girf_complete.astype('float')
girf_complete.iloc[:,[2,4,5,6]] = np.exp(girf_complete.iloc[:,[2,4,5,6]].cumsum())
girf_complete = girf_complete.rename(columns={
    'Growth_Industrial_Production':'Industrial_Production',
    'Growth_PriceIndex_Producer':'PriceIndex_Producer',
    'Growth_PriceIndex_PCE':'PriceIndex_PCE',
    'Growth_Emission_CO2':'Emission_CO2'
})

girf_complete = girf_complete.unstack()
multi_index_col = [(girf_complete.columns[i], girf_complete.columns[i+1], girf_complete.columns[i+2]) for i in range(0,24,3)]
# Plot
girfplot(df, girf_complete, multi_index_col, shock)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.figure(figsize = (10,20))
gs1 = gridspec.GridSpec(2, 4)
gs1.update(wspace=0.025, hspace=0.2) # set the spacing between axes. 
c=0
for i in range(8):
    ax1 = plt.subplot(gs1[i])
    # plt.axis('on')
    ax1.plot(girf_complete[multi_index_col[c][1]])
    c += 1
plt.tight_layout()
plt.show()

c=0
plt.figure(figsize = (10,20))
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
    c += 1
plt.tight_layout()
plt.show()