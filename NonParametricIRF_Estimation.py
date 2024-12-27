# Import new and previous libraries, dataframe, variables, model, and the functions.
from NonParametricIRF_Data import *
import statsmodels.api as sm
from Functions_Required import *
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Set the period of the of interest
histoi = df.index.date[-1]

# Setting y
# y = df.copy()
# y = df_mod.copy()
# The structural model we consider would have the Temperature Anomaly, CPU index,
# Industrial Production, Unemployment Rate, Producer's Price Index, Treasurey Bill 3 months market rate
y = df_mod.iloc[:,[0,2,3,4,5,7]]
# dataplot(y)
# y = y[y.columns[1:8]]

# VAR analysis
model_var = sm.tsa.VAR(y)
results_var = model_var.fit(6)
# results_var.irf(40).plot(); plt.show()
# results_var.irf(40).plot_cum_effects(); plt.show()
# Usual and orthogonal IRFs (use 0:temperature, 2:cpu_index )
# irf = results_var.ma_rep(40)
# irfplot(irf,df,2)
# irfplot(irf.cumsum(axis = 0),df,2)
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

robust_transformer.fit(df_mod)
omega = pd.DataFrame(
    robust_transformer.transform(df_mod),
    columns=df_mod.columns, index=df_mod.index
)
# dataplot(omega)

robust_transformer.fit(y)
y_normalized = pd.DataFrame(
    robust_transformer.transform(y),
    columns=y.columns, index=y.index
)
# dataplot(y_normalized)

# Generating the residuals u_t by estimating y_t
T = y.shape[0]
k = round(np.sqrt(T), ndigits=None)
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
# For mahalanobis distance, use the function as
# knn = NearestNeighbors(
#     n_neighbors=k, metric='mahalanobis',
#     metric_params = {'VI': np.linalg.inv(y_normalized.cov())}
# )

# Estimated y
y_hat = pd.DataFrame(index=y.index, columns=y.columns)
# If lagged=0, then lags and leads are both considered. If lagged!=0,
# then only lags are considered, not leads.

for t in y.index:
    knn.fit(y_normalized.drop(t))
    dist, ind = knn.kneighbors(y_normalized.loc[t].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    indices = y_normalized.drop(t).iloc[ind].index
    y_hat.loc[t] = np.matmul(y.loc[indices].T, weig)

# Compare the fitted values
# dataplot(y_hat)
# dataplot(results_var.fittedvalues)

# The residuals
u = pd.DataFrame(index=y.index, columns=y.columns)

for t in u.index:
    knn.fit(y_normalized.drop(t))
    dist, ind = knn.kneighbors(y_normalized.loc[t].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    indices = y_normalized.drop(t).iloc[ind].index
    u.loc[t] = np.matmul((y-y_hat).loc[indices].T, weig)

# dataplot(u)
# Compare the residuals to simple VAR
# dataplot(results_var.resid)

# y.plot(
#     subplots=True, layout=(2,3), color = 'blue',
#     ax=u.plot(
#         subplots=True, layout=(2,3), color = 'red'
#     )
# )
# plt.show()

# Send everything here to the Forecasting_GIRF.py file