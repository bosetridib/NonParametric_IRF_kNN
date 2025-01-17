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
histoi = df_std.iloc[-1]

k = omega.shape[0]
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')

# Estimate y_T
dist = np.array([euclidean(omega.loc[i], histoi) for i in omega.index])
dist = (dist - dist.min())/(dist.max() - dist.min())
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
# Estimated (NOT forecasted) the period of interest T
y_f = np.matmul(omega.T, weig).to_frame().T
y_f = np.matmul(df.iloc[:-1].T, weig).to_frame().T

u = df.iloc[:-1] - y_f.values.squeeze()
u = u.multiply(weig, axis = 0)
u.mean()

# df.iloc[:-1].plot(
#     subplots=True, layout=(2,4), color = 'blue',
#     ax=u.plot(
#         subplots=True, layout=(2,4), color = 'red'
#     )
# )
# plt.show()

