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

# Supposing the history of interest is the recent month
histoi = str(y.index.date[-1])
omega = y_normalized.loc[histoi]
X_train = y_normalized.loc[:histoi]

# Initiated during the estimation.
knn.fit(X_train)
# Find the nearest neighbours and their distance from omega.
dist, ind = knn.kneighbors(omega.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
dist = (dist - dist.min())/(dist.max() - dist.min())
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
# Map the lead index
y_f = np.matmul(y_normalized.iloc[ind+1].T, weig).to_frame().T

for h in range(1,H+1):
    knn.fit(pd.concat([X_train,y_f], axis=0).iloc[:-1])
    dist, ind = knn.kneighbors(y_f.iloc[-1].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    y_f.loc[h] = np.matmul(pd.concat([X_train, y_f], axis=0).iloc[ind+1].T, weig).values

# Select the unique values of the confidence interval dataframe

# y_f = pd.DataFrame(robust_transformer.inverse_transform(y_f), columns=girf.columns)
# dataplot(y_f)
# y_f = y_f*(y.max() - y.min()) + y.min()
# pd.concat([y_hat,y_f], axis=0).plot(subplots=True, layout=(2,4)); plt.show()

# y_fvar = pd.DataFrame(results_var.forecast(y.iloc[-6:].to_numpy(), steps = 40), columns=y.columns)
# dataplot(y_fvar)
# y_fvar.cumsum().plot(subplots=True, layout=(2,4)); plt.show()

# GIRFs
# Updated history
omega_star = y_normalized.loc[histoi] + delta
knn.fit(X_train)
dist, ind = knn.kneighbors(omega_star.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
dist = (dist - dist.min())/(dist.max() - dist.min())
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
girf = np.matmul(pd.concat([X_train , y_normalized.iloc[-1].to_frame().T], axis=0).iloc[ind+1].T, weig).to_frame().T

for h in range(1,H+1):
    knn.fit(pd.concat([X_train, girf], axis=0).iloc[:-1])
    dist, ind = knn.kneighbors(girf.iloc[-1].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    girf.loc[h] = np.matmul(pd.concat([X_train, girf], axis=0).iloc[ind+1].T, weig).values

# Set R: the number of simulations
R = 100

# GIRF_star and y_f_star at H=0
girf_star_df = pd.DataFrame(columns=girf.columns)
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
    girf_star_df = pd.concat([girf_star_df, np.matmul(X_train_ci_lead1.iloc[ind].T, weig).to_frame().T])

girf_star_df = girf_star_df - y_f_star_df

# Confidence level
conf = 0.90

# Multi-Index GIRF
girf_complete = pd.concat([2*(girf.iloc[0] - y_f.iloc[0]) - girf_star_df.quantile(conf+(1-conf)/2), girf.iloc[0] - y_f.iloc[0], 2*(girf.iloc[0] - y_f.iloc[0]) - girf_star_df.quantile((1-conf)/2)], axis=1).T

for h in range(1,H+1):
    girf_star_df = pd.DataFrame(columns=girf.columns)
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
        dist, ind = knn.kneighbors(girf.iloc[h-1].to_numpy().reshape(1,-1))
        dist = dist[0,:]; ind = ind[0,:]
        dist = (dist - dist.min())/(dist.max() - dist.min())
        weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
        girf_star_df = pd.concat([girf_star_df, np.matmul(X_train_ci_lead1.iloc[ind].T, weig).to_frame().T])
    girf_star_df = girf_star_df - y_f_star_df
    girf_complete = pd.concat([
        girf_complete, pd.concat([2*(girf.iloc[h-1] - y_f.iloc[h-1]) - girf_star_df.quantile(conf+(1-conf)/2), girf.iloc[h-1] - y_f.iloc[h-1], 2*(girf.iloc[h-1] - y_f.iloc[h-1]) - girf_star_df.quantile((1-conf)/2)], axis=1).T
    ])

girf_complete.index = pd.MultiIndex(levels=[range(0,H+1),['lower','GIRF','upper']], codes=[[x//3 for x in range(0,41*3)],[0,1,2]*41], names=('Horizon', 'CI'))
girf_complete

girf_complete[[y.columns[2]]].unstack().plot(); plt.show()
girf[[y.columns[2]]].plot(); plt.show()

girf_complete[[y.columns[4]]].unstack().plot(); plt.show()
girf[[y.columns[4]]].plot(); plt.show()

girf_complete[[y.columns[3]]].unstack().cumsum().plot(); plt.show()
girf[[y.columns[3]]].plot(); plt.show()

girf = girf - y_f
girf_cumul = girf.cumsum(axis=0)

girf = pd.DataFrame(robust_transformer.inverse_transform(girf), columns=girf.columns)
dataplot(girf)
girf_cumul = pd.DataFrame(robust_transformer.inverse_transform(girf_cumul), columns=girf.columns)
dataplot(girf_cumul)