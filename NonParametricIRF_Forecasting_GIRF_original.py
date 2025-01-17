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

# Forecasting
# Horizon "in the middle"
H = 40
omega = df_std.iloc[:-1]
omega = df_std.copy()
histoi = df_std.iloc[-1]

T = omega.shape[0]
k = omega.shape[0]
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')

# Estimate y_T
dist = np.array([euclidean(omega.loc[i], histoi) for i in omega.index])
dist = (dist - dist.min())/(dist.max() - dist.min())
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
# Estimated (NOT forecasted) the period of interest T
y_f = np.matmul(omega.T, weig).to_frame().T

u = omega - y_f.values.squeeze()
u = u.multiply(weig, axis = 0)
sigma_u = np.matmul((u - u.mean()).T , (u - u.mean()).multiply(weig, axis = 0)) / (1 - np.sum(weig**2))

for i in range(1,H+1):
    dist = np.array([euclidean(omega.loc[i], histoi) for i in omega.index[i:]])
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    y_f.loc[i] = np.matmul(omega.iloc[i:].T, weig).values

dataplot(y_f)

# Cholesky decomposition
B_mat = np.linalg.cholesky(sigma_u)
# Note that sigma_u = residual_cov*((T-1)/(T-Kp-1))
# The desired shock
shock = 2
delta = B_mat[:,shock]

# Estimate y_T_delta
y_f_delta = pd.DataFrame(columns=y_f.columns)
y_f_delta.loc[0] = y_f.loc[0] + delta

omega_star = pd.concat([omega, y_f_delta], axis=0)

for i in range(1,H+1):
    dist = np.array([euclidean(omega_star.loc[i], omega_star) for i in omega_star.index[i:]])
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    y_f_delta.loc[i] = np.matmul(omega.iloc[i:].T, weig).values

dist = np.array([euclidean(omega.loc[i], omega_star) for i in omega.index])
dist = (dist - dist.min())/(dist.max() - dist.min())
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
# Estimated the period of interest T
y_f_delta = np.matmul(omega.T, weig).to_frame().T

y_f_delta - y_f
delta

girf = pd.DataFrame(columns=y_f.columns)