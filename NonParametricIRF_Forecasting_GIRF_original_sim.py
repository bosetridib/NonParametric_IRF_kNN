# Import new and previous libraries, dataframe, variables, model, and IRF function.
from NonParametricIRF_Data import *
from Functions_Required import *
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

##################################################################################
############################# kNN Forecasting & GIRF #############################
##################################################################################
shock = 1

y_sim = pd.read_csv('sim.csv')
y_sim = y_sim.iloc[:,1:]
y_sim.index = cpu.index

df = y_sim.copy()
df[trend] = np.log(df[trend])
df = df.dropna()
# VAR analysis
model_var = sm.tsa.VAR(y_sim)
results_var = model_var.fit(6)

irf = results_var.ma_rep(40)
irfplot(irf*50,df,1)
# irf_cumsum = irf.copy()
# for k in range(0,8):
#     irf_cumsum[:,2,k] = irf_cumsum[:,2,k].cumsum()
#     irf_cumsum[:,4,k] = irf_cumsum[:,4,k].cumsum()
#     irf_cumsum[:,5,k] = irf_cumsum[:,5,k].cumsum()
#     irf_cumsum[:,6,k] = irf_cumsum[:,6,k].cumsum()
# irfplot(irf_cumsum*50,df,1)


delta_y = y_sim.copy()

mod = transformation_logdiff(delta_y[trend])

delta_y[trend] = mod.logdiff()
# p=1; pd.concat([delta_y, sm.tsa.tsatools.lagmat(delta_y[['Unemployment_Rate', 'Treasurey3Months']], maxlag=p, use_pandas=True).iloc[p:]], axis = 1)

# Forecasting
# Horizon "in the middle"
H = 40

omega = delta_y.iloc[:-(H+1)]
#histoi = delta_y.iloc[-1]
histoi = omega.mean()

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
omega = omega.loc[:y_sim.index[-1] - pd.DateOffset(months=H)]
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
# B_mat = np.linalg.cholesky(u.cov()*((T-1)/(T-8-1)))
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
dataplot(girf*(50/girf.iloc[0,shock]))
# dataplot(girf)