# Import new and previous libraries, dataframe, variables, model, and IRF function.
from NonParametricIRF_Data import *
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Setting y
y = df.copy()
y = df_mod.copy()
# dataplot(y)

# Var analysis
model_var = sm.tsa.VAR(y)
results_var = model_var.fit(6)

# Usual and orthogonal IRFs
# irf = results_var.ma_rep(40)
# irfplot(irf,df,0)
# irf = results_var.orth_ma_rep(40)
# irfplot(irf,df,2)

# Cumulative irf
# irf_cumul = results_var.orth_ma_rep(40).cumsum(axis = 0)
# irfplot(irf_cumul,df,0)

###############################################################################
############################# Estimation with kNN #############################
###############################################################################

# Prepping the data:
# Min-max scaling
# y_normalized = (y - y.min())/((y.max() - y.min()))
# Standardization
# y_normalized = (y - y.mean())/y.std()

# Robust scaling
from sklearn.preprocessing import RobustScaler
robust_transformer = RobustScaler()
robust_transformer.fit(y)
y_normalized = pd.DataFrame(
    robust_transformer.transform(y),
    columns=y.columns, index=y.index
)
# dataplot(y_normalized)

# Generating the residuals u_t by estimating y_t
from sklearn.neighbors import NearestNeighbors
T = y.shape[0]
k = round(np.sqrt(T), ndigits=None)
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
# knn = NearestNeighbors(
#     n_neighbors=k, metric='mahalanobis',
#     metric_params = {'VI': np.linalg.inv(y_normalized.cov())}
# )

y_hat = pd.DataFrame(index=y.index, columns=y.columns)
# If lagged=0, then lags and leads are both considered. If lagged!=0,
# then only lags are considered, not leads.
lagged = 0

for t in (y.index if (lagged == 0) else y.index[k:]):
    knn.fit(
        y_normalized.drop(t) if (lagged == 0) else y_normalized.loc[:t - pd.DateOffset(months=1)]
    )
    dist, ind = knn.kneighbors(y_normalized.loc[t].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    y_hat.loc[t] = np.matmul(y.drop(t).iloc[ind].T, weig)

# Compare the fitted values
# dataplot(y_hat)
# dataplot(results_var.fittedvalues)

# The residuals
u = y - y_hat
# dataplot(u)
# Compare the residuals to simple VAR
# dataplot(results_var.resid)

# RMSE
u = u.dropna()
N = u.shape[0] - u.shape[1]
(np.sum(u**2)/N)**0.5