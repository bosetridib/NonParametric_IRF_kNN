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

dist = np.array([euclidean(omega.loc[i], histoi_delta) for i in omega.index])
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

for h in range(1,H+1):
    y_f_delta.loc[h] = np.matmul(y.loc[omega.index[h:]].T, weig[:-h]).values
# dataplot(y_f_delta)

girf = y_f_delta - y_f
dataplot(girf)
dataplot(np.exp(girf.cumsum()))

# Confidence Intervals
R=100
sim_girf = []

# Perform simulations
for r in range(0,R):
    omega_resamp = omega.sample(n=T, replace=True).sort_index()
    # Estimate y_T
    dist = np.array([euclidean(omega_resamp.iloc[i], histoi) for i in range(0,omega_resamp.shape[0])])
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    # Estimated (NOT forecasted) the period of interest T
    y_f_resamp = np.matmul(y.loc[omega_resamp.index].T, weig).to_frame().T
    sim_girf.append(y_f_resamp)
# End of loop, and now the sim_list_df has each of the resampled dataframes
