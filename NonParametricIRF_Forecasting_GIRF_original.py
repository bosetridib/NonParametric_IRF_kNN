# Import new and previous libraries, dataframe, variables, model, and IRF function.
from NonParametricIRF_Data import *
from Functions_Required import *
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

##################################################################################
############################# kNN Forecasting & GIRF #############################
##################################################################################

df_std = (df - df.mean())/df.std()

# Forecasting
# Horizon "in the middle"
H = 40
k = df_std.shape[0] - H

# Since p=6 in the Gavriilidis, Kanzig, Stock (2023) paper, we select
# the histories at t-1...t-6;
p = 6; n_var = df.shape[1]

# The history of interest
omega = sm.tsa.lagmat(df_std, maxlag=p, use_pandas=True)
# omega = df_std.copy()

# Remove the 0's due to lag.
omega = omega.iloc[p:]
omega.plot(); plt.show()
histoi = omega.index[-1]

knn = NearestNeighbors(n_neighbors=k, metric='euclidean')

# Estimated omega
omega_hat = pd.DataFrame(index=omega_scaled.index, columns=omega_scaled.columns)

for t in omega_scaled.index:
    knn.fit(omega_scaled.drop(t))
    dist, ind = knn.kneighbors(omega_scaled.loc[t].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    indices = omega_scaled.drop(t).iloc[ind].index
    omega_hat.loc[t] = np.matmul(omega_scaled.loc[indices].T, weig)

# Compare the fitted values
# dataplot(omega_hat)
# dataplot(results_var.fittedvalues)

# The residuals
u = pd.DataFrame(index=omega_scaled.index, columns=omega_scaled.columns)

for t in u.index:
    knn.fit(omega_scaled.drop(t))
    dist, ind = knn.kneighbors(omega_scaled.loc[t].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    indices = omega_scaled.drop(t).iloc[ind].index
    u.loc[t] = np.matmul((omega_scaled-omega_hat).loc[indices].T, weig)

