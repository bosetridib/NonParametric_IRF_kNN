# Import new and previous libraries, dataframe, variables, model, and IRF function.
from NonParametricIRF_Estimation import *
import warnings
warnings.filterwarnings('ignore')


##################################################################################
############################# kNN Forecasting & GIRF #############################
##################################################################################

# Retrieve shocks through Cholesky decomposition
# Error is allowed to vary with each horizon.
B_mat = np.linalg.cholesky(u.cov()*((T-1)/(T-8-1)))
# Note that sigma_u = residual_cov*((T-1)/(T-Kp-1))

# Horizon "in the middle"
H = 40

# The desired shock
delta = B_mat[:,0]

# Old method
# Since p=6 in the Gavriilidis, Kanzig, Stock (2023) paper, we select
# the histories at t-1...t-6;
p = 6; n_var = y.shape[1]

y_normalized_plags = sm.tsa.lagmat(y_normalized, maxlag=p, use_pandas=True)
# and remove the 0's due to lag.
y_normalized_plags = y_normalized_plags.iloc[p:]


# Supposing the history of interest is the recent month
myoi = str(y.index.date[-1])
omega = y_normalized_plags.iloc[-1]
X_train = y_normalized_plags.iloc[:-1]

# Manual way for h=0, 1 to H=40
from scipy.spatial.distance import euclidean
dist = np.array([euclidean(i, omega.to_numpy()) for i in X_train.to_numpy()])
dist = (dist - np.min(dist))/(np.max(dist) - np.min(dist))
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
y_f = np.matmul(y.iloc[p:].drop([myoi]).T, weig).to_frame().T

for h in range(1,H+1):
    dist = np.array([euclidean(i, omega.to_numpy()) for i in X_train.iloc[:-h].to_numpy()])
    dist = (dist - np.min(dist))/(np.max(dist) - np.min(dist))
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    y_f.loc[h] = np.matmul(y.iloc[p+h:].drop([myoi]).T, weig).values

# The forecasts are
# y_f.plot(subplots = True, layout = (2,4)); plt.show()
# y_f.cumsum().plot(subplots = True, layout = (2,4)); plt.show()

girf = pd.DataFrame(delta.reshape(-1,n_var), columns=y_f.columns)
# Updated history
omega_star = pd.concat(
    [
        (y_f.iloc[0] + delta - y.min())/(y.max() - y.min()),
        omega.iloc[:-n_var]
    ],
    axis=0
)

for h in range(1,H+1):
    dist = np.array([euclidean(i, omega_star.to_numpy()) for i in X_train.iloc[h:].to_numpy()])
    dist = (dist - np.min(dist))/(np.max(dist) - np.min(dist))
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    # weig = (1/dist)/sum(1/dist)
    girf.loc[h] = np.matmul(y.iloc[p+h:].drop(myoi).T, weig).values
    girf.loc[h] = girf.loc[h] - y_f.loc[h]

girf = pd.DataFrame(robust_transformer.inverse_transform(girf), columns=girf.columns)
dataplot(girf)




# New method

# Supposing the history of interest is the recent month
myoi = str(y.index.date[-1])
omega = y_normalized.loc[myoi]
X_train = y_normalized.drop(myoi)

knn.fit(X_train)

# Collected the nearest neighbour vectors for confidence intervals.
ci_df_y = pd.DataFrame(columns=y.columns)
ci_df_y_lead1 = pd.DataFrame(columns=y.columns)

dist, ind = knn.kneighbors(y_normalized.iloc[-1].to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
dist = (dist - dist.min())/(dist.max() - dist.min())
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
y_f = np.matmul(y_normalized.iloc[ind+1].T, weig).to_frame().T

ci_df_y = pd.concat([ci_df_y, y_normalized.iloc[ind]], axis=0)
ci_df_y_lead1 = pd.concat([ci_df_y_lead1, y_normalized.iloc[ind+1]], axis=0)

for h in range(1,H+1):
    knn.fit(pd.concat([X_train,y_f], axis=0).iloc[:-1])
    dist, ind = knn.kneighbors(y_f.iloc[-1].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    y_f.loc[h] = np.matmul(pd.concat([X_train, y_f], axis=0).iloc[ind+1].T, weig).values
    ci_df_y = pd.concat([ci_df_y, pd.concat([X_train, y_f], axis=0).iloc[ind]], axis=0)
    ci_df_y_lead1 = pd.concat([ci_df_y_lead1, pd.concat([X_train, y_f], axis=0).iloc[ind+1]], axis=0)


# Select the unique values of the confidence interval dataframe
ci_df_y = ci_df_y.loc[~ci_df_y.index.duplicated()]
ci_df_y_lead1 = ci_df_y_lead1.loc[~ci_df_y_lead1.index.duplicated()]

# y_f = pd.DataFrame(robust_transformer.inverse_transform(y_f), columns=girf.columns)
# dataplot(y_f)
# y_f = y_f*(y.max() - y.min()) + y.min()
# pd.concat([y_hat,y_f], axis=0).plot(subplots=True, layout=(2,4)); plt.show()

# y_fvar = pd.DataFrame(results_var.forecast(y.iloc[-6:].to_numpy(), steps = 40), columns=y.columns)
# dataplot(y_fvar)
# y_fvar.cumsum().plot(subplots=True, layout=(2,4)); plt.show()

delta = B_mat[:,0]

# GIRFs
# Updated history
omega_star = y_normalized.loc[myoi] + delta
knn.fit(X_train)
dist, ind = knn.kneighbors(omega_star.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
dist = (dist - dist.min())/(dist.max() - dist.min())
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
girf = np.matmul(pd.concat([X_train , y_normalized.iloc[-1].to_frame().T], axis=0).iloc[ind+1].T, weig).to_frame().T

# The confidence interval dataframe for GIRF
ci_df_girf = pd.DataFrame(columns=y.columns)
ci_df_girf = pd.concat([ci_df_girf, y_normalized.iloc[ind]], axis=0)
ci_df_girf_lead1 = pd.DataFrame(columns=y.columns)
ci_df_girf_lead1 = pd.concat([ci_df_girf, y_normalized.iloc[ind+1]], axis=0)

for h in range(1,H+1):
    knn.fit(pd.concat([X_train, girf], axis=0).iloc[:-1])
    dist, ind = knn.kneighbors(girf.iloc[-1].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    girf.loc[h] = np.matmul(pd.concat([X_train, girf], axis=0).iloc[ind+1].T, weig).values
    ci_df_girf = pd.concat([ci_df_girf, pd.concat([X_train, girf], axis=0).iloc[ind]], axis=0)
    ci_df_girf_lead1 = pd.concat([ci_df_girf_lead1, pd.concat([X_train, girf], axis=0).iloc[ind+1]], axis=0)


girf = girf - y_f
girf_cumul = girf.cumsum(axis=0)

# Removing the duplicates in the confidence interval GIRF dataframe
ci_df_girf = ci_df_girf.loc[~ci_df_girf.index.duplicated()]
ci_df_girf_lead1 = ci_df_girf_lead1.loc[~ci_df_girf_lead1.index.duplicated()]

# Set R: the number of simulations
R = 100

# GIRF_star and y_f_star at H=0
girf_star_df = pd.DataFrame(columns=girf.columns)
y_f_star_df = pd.DataFrame(columns=y_f.columns)

for i in range(0,R):
    X_train_ci = ci_df_y.sample(n = T, replace=True)
    # Sort the dataframe with respect to date and horizon, separately
    mask = np.array([True if type(i) != int else False for i in X_train_ci.reset_index()['index']])
    X_train_ci = pd.concat([X_train_ci.loc[mask].sort_index(), X_train_ci.loc[~mask].sort_index()], axis=0)
    # X_train_ci_lead1 = ci_df.index.get_indexer(X_train_ci.loc[mask].index.unique())
    knn.fit(X_train_ci)
    # Bootstrapped forecast
    dist, ind = knn.kneighbors(y_normalized.iloc[-1].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    y_f_star_df = pd.concat([y_f_star_df, np.matmul(X_train_ci.iloc[ind].T, weig).to_frame().T])
    # Bootstrapped GIRF
    X_train_ci = ci_df_girf.sample(n = T-k, replace=True)
    mask = np.array([True if type(i) != int else False for i in X_train_ci.reset_index()['index']])
    X_train_ci = pd.concat([X_train_ci.loc[mask].sort_index(), X_train_ci.loc[~mask].sort_index()], axis=0)
    knn.fit(X_train_ci)
    dist, ind = knn.kneighbors(omega_star.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    girf_star_df = pd.concat([girf_star_df, np.matmul(X_train_ci.iloc[ind].T, weig).to_frame().T])

girf_star_df = girf_star_df - y_f_star_df
# Confidence level
conf = 0.90

# Multi-Index GIRF
girf_complete = pd.concat([2*girf.iloc[0] - girf_star_df.quantile(conf+(1-conf)/2), girf.iloc[0], 2*girf.iloc[0] - girf_star_df.quantile((1-conf)/2)], axis=1).T

for h in range(1,H+1):
    girf_star_df = pd.DataFrame(columns=girf.columns)
    y_f_star_df = pd.DataFrame(columns=y_f.columns)
    for i in range(0,R):
        X_train_ci = ci_df_y.sample(n = T, replace=True)
        # Sort the dataframe with respect to date and horizon, separately
        mask = np.array([True if type(i) != int else False for i in X_train_ci.reset_index()['index']])
        X_train_ci = pd.concat([X_train_ci.loc[mask].sort_index(), X_train_ci.loc[~mask].sort_index()], axis=0)
        knn.fit(X_train_ci)
        # Bootstrapped forecast
        dist, ind = knn.kneighbors(y_f.iloc[h-1].to_numpy().reshape(1,-1))
        dist = dist[0,:]; ind = ind[0,:]
        dist = (dist - dist.min())/(dist.max() - dist.min())
        weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
        y_f_star_df = pd.concat([y_f_star_df, np.matmul(X_train_ci.iloc[ind].T, weig).to_frame().T])
        # Bootstrapped GIRF
        X_train_ci = ci_df_girf.sample(n = T, replace=True)
        # Sort the dataframe with respect to date and horizon, separately
        mask = np.array([True if type(i) != int else False for i in X_train_ci.reset_index()['index']])
        X_train_ci = pd.concat([X_train_ci.loc[mask].sort_index(), X_train_ci.loc[~mask].sort_index()], axis=0)
        knn.fit(X_train_ci)
        dist, ind = knn.kneighbors((girf+y_f).iloc[h-1].to_numpy().reshape(1,-1))
        dist = dist[0,:]; ind = ind[0,:]
        dist = (dist - dist.min())/(dist.max() - dist.min())
        weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
        girf_star_df = pd.concat([girf_star_df, np.matmul(X_train_ci.iloc[ind].T, weig).to_frame().T])
    girf_star_df = girf_star_df - y_f_star_df
    girf_complete = pd.concat([
        girf_complete, pd.concat([2*girf.iloc[h] - girf_star_df.quantile(conf+(1-conf)/2), girf.iloc[h], 2*girf.iloc[h] - girf_star_df.quantile((1-conf)/2)], axis=1).T
    ])

girf_complete.index = pd.MultiIndex(levels=[range(0,H+1),['lower','GIRF','upper']], codes=[[x//3 for x in range(0,41*3)],[0,1,2]*41], names=('Horizon', 'CI'))
girf_complete

girf_complete.unstack(level=1)
girf_complete[[y.columns[4]]].unstack().plot(); plt.show()

girf = pd.DataFrame(robust_transformer.inverse_transform(girf), columns=girf.columns)
dataplot(girf)
girf_cumul = pd.DataFrame(robust_transformer.inverse_transform(girf_cumul), columns=girf.columns)
dataplot(girf_cumul)