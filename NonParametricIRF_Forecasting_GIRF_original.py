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
k = df_std.shape[0] - H

omega = df_std.iloc[:-1]
histoi = df_std.index[-1]

knn = NearestNeighbors(n_neighbors=k, metric='euclidean')

