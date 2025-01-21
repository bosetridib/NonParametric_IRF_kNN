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
dist = (dist - dist.min())/(dist.max() - dist.min())
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
# Estimated (NOT forecasted) the period of interest T
y_f = np.matmul(y.loc[omega.index].T, weig).to_frame().T

u = y.loc[omega.index] - y_f.values.squeeze()
u = u.multiply(weig, axis = 0)
sigma_u = np.matmul((u - u.mean()).T , (u - u.mean()).multiply(weig, axis = 0)) / (1 - np.sum(weig**2))

for h in range(1,H+1):
    dist = np.array([euclidean(omega.loc[h], histoi) for h in omega.index[h:]])
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    y_f.loc[h] = np.matmul(omega.iloc[h:].T, weig).values

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

omega_star = pd.concat([y.loc[omega.index], y_f_delta], axis=0)

for h in range(1,H+1):
    dist = np.array([euclidean(omega_star.loc[h], y_f_delta.iloc[0]) for h in omega_star.index[h:]])
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    y_f_delta.loc[h] = np.matmul(omega_star.iloc[h:].T, weig).values
# dataplot(y_f_delta)

girf = y_f_delta - y_f
dataplot(girf)

# Confidence Intervals
R=100
sim_list_df = []

for r in range(0,R):
    # For each resampled omega, we will store different
    # dataframes of the IRFs
    omega_resampled = omega.sample(n=T, replace=True).sort_index()
    # Bootstrapped Forecast
    dist = np.array([euclidean(omega_resampled.loc[i], histoi) for i in omega_resampled.index])
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    # Estimated (NOT forecasted) the period of interest T
    y_f_star = np.matmul(omega_resampled.T, weig).to_frame().T

    for i in range(1,H+1):
        dist = np.array([euclidean(omega.loc[i], histoi) for i in omega.index[i:]])
        dist = (dist - dist.min())/(dist.max() - dist.min())
        weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
        y_f_star.loc[i] = np.matmul(omega.iloc[i:].T, weig).values
    # dataplot(y_f)
    # Estimate y_T_delta
    y_f_delta_star = pd.DataFrame(columns=y_f.columns)
    y_f_delta_star.loc[0] = y_f_star.loc[0] + delta

    omega_star = pd.concat([omega_resampled, y_f_delta_star], axis=0)

    for i in range(1,H+1):
        dist = np.array([euclidean(omega_star.loc[i], y_f_delta.iloc[0]) for i in omega_star.index[i:]])
        dist = (dist - dist.min())/(dist.max() - dist.min())
        weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
        y_f_delta_star.loc[i] = np.matmul(omega_star.iloc[i:].T, weig).values
    # dataplot(y_f_delta)
    # Store the GIRFs in the list
    sim_list_df.append(y_f_delta_star - y_f_star)
# End of loop, and now the sim_list_df has each of the resampled dataframes