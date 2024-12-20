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

# Estimated (NOT forecasted) the period of interest T
y_f = np.matmul(omega.iloc[ind].T, weig).to_frame().T
# This is the estimate for the period of interest, ie. E(y_T) at h=0

# The following would be the forecast of the y_T+1,+2,...H
# The principle is to fit the data upto and NOT including the final period, find the
# nearest neighbour of the final period and estimate the forecast. Update the data with the
# forecasts, and repeat.
for h in range(1,H+1):
    omega_updated = pd.concat([omega,y_f], axis=0).iloc[:-1]
    knn.fit(omega_updated)
    dist, ind = knn.kneighbors(y_f.iloc[-1].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    lead_index = np.array(
        [i+1 if type(i)==int else 0 if i==omega.index[-1] else i + pd.DateOffset(months=1) for i in omega_updated.iloc[ind].index]
    )
    omega_lead = pd.concat([omega,y_f], axis=0).loc[lead_index]
    y_f.loc[h] = np.matmul(omega_lead.T, weig).values

# y_f = pd.DataFrame(robust_transformer.inverse_transform(y_f), columns=girf.columns)
# dataplot(y_f)
# y_fvar = pd.DataFrame(results_var.forecast(y.iloc[-6:].to_numpy(), steps = 40), columns=y.columns)
# dataplot(y_fvar)
# y_fvar.cumsum().plot(subplots=True, layout=(2,4)); plt.show()

# GIRFs

# The period of interest with shock
omega_star = y_normalized.loc[str(histoi)] + delta

# Fit the knn upto and NOT including the history of interest
knn.fit(omega)

# Find the nearest neighbours and their distance from period of interest.
dist, ind = knn.kneighbors(omega_star.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
dist = (dist - dist.min())/(dist.max() - dist.min())
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

# The estimated period of interest with the shock. Note that since the period of
# interest is being considered as the current period, at period T and h=0, there's
# no forecasting, only estimation.
y_f_delta = np.matmul(omega.iloc[ind].T, weig).to_frame().T
# This is the estimation for the period of interest with shock, ie. E(y_T, delta) at h=0

# The following would be the forecast of the y_T+1,+ 2,...H with shock
# The principle is same as in forecasts.
for h in range(1,H+1):
    omega_updated = pd.concat([omega,y_f_delta], axis=0).iloc[:-1]
    knn.fit(omega_updated)
    dist, ind = knn.kneighbors(y_f_delta.iloc[-1].to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    lead_index = np.array(
        [i+1 if type(i)==int else 0 if i==omega.index[-1] else i + pd.DateOffset(months=1) for i in omega_updated.iloc[ind].index]
    )
    omega_lead = pd.concat([omega,y_f_delta], axis=0).loc[lead_index]
    y_f_delta.loc[h] = np.matmul(omega_lead.T, weig).values

girf = y_f_delta - y_f
# The raw IRF (without CI)
# dataplot(y_f_delta - y_f)

# The confidence interval
# Set R: the number of simulations
R = 100
# The following list will collect the simulated dataframes of the
# GIRF for each resampling
sim_list_df = []

for i in range(0,R):
    # For each resampled omega, we will store different
    # dataframes of the IRFs
    omega_resampled = omega.sample(n=T, replace=True).sort_index()
    # Bootstrapped Forecast
    knn.fit(omega_resampled)
    # Find the nearest neighbours and their distance from period of interest.
    dist, ind = knn.kneighbors(omega_mutated.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    # Again, estimate (NOT forecast) at period of interest T
    y_f_star = np.matmul(omega_resampled.iloc[ind].T, weig).to_frame().T
    # Forecast for periods h=1,...,H
    for h in range(1,H+1):
        omega_updated = pd.concat([omega_resampled,y_f_star], axis=0).iloc[:-1]
        knn.fit(omega_updated)
        dist, ind = knn.kneighbors(y_f_star.iloc[-1].to_numpy().reshape(1,-1))
        dist = dist[0,:]; ind = ind[0,:]
        dist = (dist - dist.min())/(dist.max() - dist.min())
        weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
        lead_index = np.array(
            [i+1 if type(i)==int else 0 if i==omega_resampled.index[-1] else i + pd.DateOffset(months=1) for i in omega_updated.iloc[ind].index]
        )
        # Pick the lead index from the original (not resampled) dataframe
        omega_lead = pd.concat([omega,y_f_star], axis=0).loc[lead_index]
        y_f_star.loc[h] = np.matmul(omega_lead.T, weig).values
    # Bootstrapped Forecast with shock
    knn.fit(omega_resampled)
    # Find the nearest neighbours and their distance from period of interest.
    dist, ind = knn.kneighbors(omega_star.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    dist = (dist - dist.min())/(dist.max() - dist.min())
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    y_f_delta_star = np.matmul(omega_resampled.iloc[ind].T, weig).to_frame().T
    # Forecast of the y_T+1,+ 2,...H with shock
    for h in range(1,H+1):
        omega_updated = pd.concat([omega_resampled,y_f_delta_star], axis=0).iloc[:-1]
        knn.fit(omega_updated)
        dist, ind = knn.kneighbors(y_f_delta_star.iloc[-1].to_numpy().reshape(1,-1))
        dist = dist[0,:]; ind = ind[0,:]
        dist = (dist - dist.min())/(dist.max() - dist.min())
        weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
        lead_index = np.array(
            [i+1 if type(i)==int else 0 if i==omega.index[-1] else i + pd.DateOffset(months=1) for i in omega_updated.iloc[ind].index]
        )
        # Same as in Bootstrapped forecast
        omega_lead = pd.concat([omega,y_f_delta_star], axis=0).loc[lead_index]
        y_f_delta_star.loc[h] = np.matmul(omega_lead.T, weig).values
    # Store the GIRFs in the list
    sim_list_df.append(y_f_delta_star - y_f_star)
# End of loop, and now the sim_list_df has each of the resampled dataframes

# Confidence level
conf = 0.90
# Define the multi-index dataframe for each horizon and CI for each column
girf_complete = pd.DataFrame(
    columns = omega.columns,
    index = pd.MultiIndex(
        levels=[range(0,H+1),['lower','GIRF','upper']],
        codes=[[x//3 for x in range(0,41*3)],[0,1,2]*(H+1)], names=('Horizon', 'CI')
    )
)

for h in range(0,H+1):
    for col in omega.columns:
        girf_complete[col][h,'lower'] = 2*girf[col][h] - np.quantile([each_df[col][h] for each_df in sim_list_df], conf+(1-conf)/2)
        girf_complete[col][h,'GIRF'] = girf[col][h]
        girf_complete[col][h,'upper'] = 2*girf[col][h] - np.quantile([each_df[col][h] for each_df in sim_list_df], (1-conf)/2)

girf_complete
# GIRF CI at h=0

girf = y_f_delta - y_f
girf_cumul = girf.cumsum(axis=0)

girf_complete[[y.columns[2]]].unstack().plot(); plt.show()
girf[[y.columns[2]]].plot(); plt.show()

girf_complete[[y.columns[4]]].unstack().plot(); plt.show()
girf[[y.columns[4]]].plot(); plt.show()

girf_complete[[y.columns[3]]].unstack().cumsum().plot(); plt.show()
np.exp(girf[[y.columns[3]]].cumsum()).plot(); plt.show()

girf_complete = pd.DataFrame(robust_transformer.inverse_transform(girf_complete), columns=girf_complete.columns, index=girf_complete.index)
dataplot(girf)
girf_cumul = pd.DataFrame(robust_transformer.inverse_transform(girf_cumul), columns=girf.columns)
dataplot(girf_cumul)