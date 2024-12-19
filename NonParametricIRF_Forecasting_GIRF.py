# Import new and previous libraries, dataframe, variables, model, and IRF function.
from NonParametricIRF_Estimation import *
import warnings
warnings.filterwarnings('ignore')

##################################################################################
############################# kNN Forecasting & GIRF #############################
##################################################################################

# Retrieve shocks through Cholesky decomposition
B_mat = np.linalg.cholesky(u.cov()*((T-1)/(T-8-1)))
# Note that sigma_u = residual_cov*((T-1)/(T-Kp-1))

# Horizon "in the middle"
H = 40

# The desired shock
delta = B_mat[:,0]

# New method

# Set the period of the of interest
histoi = y.index.date[-1]

# Set the history of interest upto and NOT including the period (T-1)
omega = y_normalized.loc[:histoi - pd.DateOffset(months = 1)]
# Period of interest's realized data at T
omega_mutated = y_normalized.loc[str(histoi)]
T = omega.shape[0] + 1

# Fit the knn upto and NOT including the history of interest
knn.fit(omega)

# Find the nearest neighbours and their distance from period of interest.
dist, ind = knn.kneighbors(omega_mutated.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
dist = (dist - dist.min())/(dist.max() - dist.min())
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

# Map the lead indices of the nearest neighbours
lead_index = omega.iloc[ind].index + pd.DateOffset(months=1)
omega_lead = pd.concat([omega, omega_mutated.to_frame().T], axis=0).loc[lead_index]
y_f = np.matmul(omega_lead.T, weig).to_frame().T
# This is the forecast for the period of interest, ie. E(y_T) at h=0

# The following would be the forecast of the y_T+1,+2,...H
# The principle is same as before. Fit the data upto and NOT including the final period. Find
# nearest neighbour of the final period and estimate the forecast. Update the data with the
# forecasts, and repeat.
for h in range(1,H+1):
    omega_updated = pd.concat([omega,y_f], axis=0)
    X_train = pd.concat([omega,y_f], axis=0)
    knn.fit(omega_updated)
    dist, ind = knn.kneighbors(X_train.iloc[-1].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    lead_index = np.array(
        [i+1 if type(i) == int else i + pd.DateOffset(months=1) for i in omega_updated.iloc[ind].index]
    )
    X_train_lead = X_train.loc[lead_index]
    y_f.loc[h] = np.matmul(X_train_lead.T, weig).values

# Select the unique values of the confidence interval dataframe

# y_f = pd.DataFrame(robust_transformer.inverse_transform(y_f), columns=girf.columns)
# dataplot(y_f)
# y_fvar = pd.DataFrame(results_var.forecast(y.iloc[-6:].to_numpy(), steps = 40), columns=y.columns)
# dataplot(y_fvar)
# y_fvar.cumsum().plot(subplots=True, layout=(2,4)); plt.show()

# GIRFs
omega = y_normalized.loc[:histoi - pd.DateOffset(months = 1)]
X_train = y_normalized.loc[:histoi]
# The X_train contains observations upto and including the period T.

# The period of interest with shock
omega_star = y_normalized.loc[str(histoi)] + delta

# Fit the knn upto and NOT including the history of interest
knn.fit(omega)

# Find the nearest neighbours and their distance from period of interest.
dist, ind = knn.kneighbors(omega_star.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
dist = (dist - dist.min())/(dist.max() - dist.min())
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

# Map the lead indices of the nearest neighbours
lead_index = omega.iloc[ind].index + pd.DateOffset(months=1)
X_train_lead = X_train.loc[lead_index]
y_f_delta = np.matmul(X_train_lead.T, weig).to_frame().T
# This is the forecast for the period of interest with shock, ie. E(y_T, delta) at h=0

# The following would be the forecast of the y_T+1,+ 2,...H with shock
# The principle is same as in forecasts.
for h in range(1,H+1):
    omega_updated = pd.concat([omega,y_f_delta], axis=0).iloc[:-1]
    X_train = pd.concat([omega,y_f_delta], axis=0)
    knn.fit(omega_updated)
    dist, ind = knn.kneighbors(X_train.iloc[-1].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    lead_index = np.array([i+1 if type(i) == int else i + pd.DateOffset(months=1) for i in omega_updated.iloc[ind].index])
    X_train_lead = X_train.loc[lead_index]
    y_f_delta.loc[h] = np.matmul(X_train_lead.T, weig).values

# dataplot(y_f_delta - y_f)
# Set R: the number of simulations
R = 100

# GIRF_star and y_f_star at H=0
y_f_delta_star_df = pd.DataFrame(columns=y_f_delta.columns)
y_f_star_df = pd.DataFrame(columns=y_f.columns)

for i in range(0,R):
    X_train_ci = X_train.sample(n = T, replace=True).sort_index()
    X_train_ci_lead1 = y_normalized.loc[X_train_ci.index + pd.DateOffset(months=1)]
    knn.fit(X_train_ci)
    # Bootstrapped forecast
    dist, ind = knn.kneighbors(y_normalized.iloc[-1].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    y_f_star_df = pd.concat([y_f_star_df, np.matmul(X_train_ci_lead1.iloc[ind].T, weig).to_frame().T])
    # Bootstrapped GIRF
    X_train_ci = X_train.sample(n = T, replace=True).sort_index()
    X_train_ci_lead1 = y_normalized.loc[X_train_ci.index + pd.DateOffset(months=1)]
    knn.fit(X_train_ci)
    dist, ind = knn.kneighbors(omega_star.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    y_f_delta_star_df = pd.concat([y_f_delta_star_df, np.matmul(X_train_ci_lead1.iloc[ind].T, weig).to_frame().T])

girf_star_df = y_f_delta_star_df - y_f_star_df

# Confidence level
conf = 0.90

# Multi-Index GIRF
girf_complete = pd.concat([2*(y_f_delta.iloc[0] - y_f.iloc[0]) - girf_star_df.quantile(conf+(1-conf)/2), y_f_delta.iloc[0] - y_f.iloc[0], 2*(y_f_delta.iloc[0] - y_f.iloc[0]) - girf_star_df.quantile((1-conf)/2)], axis=1).T

for h in range(1,H+1):
    y_f_delta_star_df = pd.DataFrame(columns=y_f_delta.columns)
    y_f_star_df = pd.DataFrame(columns=y_f.columns)
    for i in range(0,R):
        X_train_ci = X_train.sample(n = T, replace=True).sort_index()
        X_train_ci_lead1 = y_normalized.loc[X_train_ci.index + pd.DateOffset(months=1)]
        knn.fit(X_train_ci)
        # Bootstrapped forecast
        dist, ind = knn.kneighbors(y_f.iloc[h-1].to_numpy().reshape(1,-1))
        dist = dist[0,:]; ind = ind[0,:]
        dist = (dist - dist.min())/(dist.max() - dist.min())
        weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
        y_f_star_df = pd.concat([y_f_star_df, np.matmul(X_train_ci_lead1.iloc[ind].T, weig).to_frame().T])
        # Bootstrapped GIRF
        X_train_ci = X_train.sample(n = T, replace=True).sort_index()
        X_train_ci_lead1 = y_normalized.loc[X_train_ci.index + pd.DateOffset(months=1)]
        knn.fit(X_train_ci)
        dist, ind = knn.kneighbors(y_f_delta.iloc[h-1].to_numpy().reshape(1,-1))
        dist = dist[0,:]; ind = ind[0,:]
        dist = (dist - dist.min())/(dist.max() - dist.min())
        weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
        y_f_delta_star_df = pd.concat([girf_star_df, np.matmul(X_train_ci_lead1.iloc[ind].T, weig).to_frame().T])
    girf_star_df = y_f_delta_star_df - y_f_star_df
    girf_complete = pd.concat([
        girf_complete, pd.concat([2*(y_f_delta.iloc[h-1] - y_f.iloc[h-1]) - girf_star_df.quantile(conf+(1-conf)/2), y_f_delta.iloc[h-1] - y_f.iloc[h-1], 2*(y_f_delta.iloc[h-1] - y_f.iloc[h-1]) - girf_star_df.quantile((1-conf)/2)], axis=1).T
    ])

girf_complete.index = pd.MultiIndex(levels=[range(0,H+1),['lower','GIRF','upper']], codes=[[x//3 for x in range(0,41*3)],[0,1,2]*41], names=('Horizon', 'CI'))
girf_complete

girf = y_f_delta - y_f
girf_cumul = girf.cumsum(axis=0)

girf_complete[[y.columns[2]]].unstack().plot(); plt.show()
girf[[y.columns[2]]].plot(); plt.show()

girf_complete[[y.columns[4]]].unstack().plot(); plt.show()
girf[[y.columns[4]]].plot(); plt.show()

girf_complete[[y.columns[3]]].unstack().cumsum().plot(); plt.show()
girf[[y.columns[3]]].plot(); plt.show()

girf = pd.DataFrame(robust_transformer.inverse_transform(girf), columns=girf.columns)
dataplot(girf)
girf_cumul = pd.DataFrame(robust_transformer.inverse_transform(girf_cumul), columns=girf.columns)
dataplot(girf_cumul)