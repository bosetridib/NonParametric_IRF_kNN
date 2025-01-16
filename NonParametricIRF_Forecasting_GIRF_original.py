# Import new and previous libraries, dataframe, variables, model, and IRF function.
from NonParametricIRF_Data import *
from Functions_Required import *
from sklearn.neighbors import NearestNeighbors
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
histoi = df_std.iloc[-1]

k = omega.shape[0]
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')

# Estimate y_T
knn.fit(omega)

dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
dist = (dist - dist.min())/(dist.max() - dist.min())
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
# Estimated (NOT forecasted) the period of interest T
y_f = np.matmul(omega.iloc[ind].T, weig).to_frame().T

