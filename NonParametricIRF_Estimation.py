# Import new and previous libraries, dataframe, variables, model, and the functions.
from NonParametricIRF_Data import *
import statsmodels.api as sm
from Functions_Required import *
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Setting y
# The structural model we consider would have the Temperature Anomaly, CPU index,
# Industrial Production, Unemployment Rate, Producer's Price Index, Treasurey
# Bill 3 months market rate, represented by following variables of interest
# y = df.copy()
y = df_mod.copy()
# voi = [0,2,3,4,5,7]
# y = df_mod.iloc[:,voi]

# dataplot(y)

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
omega = df_mod.copy()
# omega = omega.loc[omega['Unemployment_Rate'] >= 6]

histoi = omega.index.date[-1]

# Robust scaling
from sklearn.preprocessing import RobustScaler
robust_transformer = RobustScaler()

robust_transformer.fit(df_mod)
df_mod_scaled = pd.DataFrame(
    robust_transformer.transform(df_mod),
    columns=df_mod.columns, index=df_mod.index
)

robust_transformer.fit(omega)
omega_scaled = pd.DataFrame(
    robust_transformer.transform(omega),
    columns=omega.columns, index=omega.index
)
# dataplot(omega_scaled)
omega_mutated = omega_scaled.loc[str(histoi)]
omega_scaled = omega_scaled.loc[:histoi - pd.DateOffset(months = 1)]
omega = omega.loc[:histoi - pd.DateOffset(months = 1)]
# dataplot(omega)

# Generating the residuals u_t by estimating omega_t
T = omega.shape[0]+1
k = round(np.sqrt(T), ndigits=None)
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
# For mahalanobis distance, use the function as
# knn = NearestNeighbors(
#     n_neighbors=k, metric='mahalanobis',
#     metric_params = {'VI': np.linalg.inv(y_normalized.cov())}
# )

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

# dataplot(u)
# Compare the residuals to simple VAR
# dataplot(results_var.resid)

# omega_scaled.plot(
#     subplots=True, layout=(2,4), color = 'blue',
#     ax=u.plot(
#         subplots=True, layout=(2,4), color = 'red'
#     )
# )
# plt.show()

# Send everything here to the Forecasting_GIRF.py file