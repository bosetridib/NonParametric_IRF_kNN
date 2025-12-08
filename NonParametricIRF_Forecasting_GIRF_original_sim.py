# Import new and previous libraries, dataframe, variables, model, and IRF function.
from NonParametricIRF_Data import *
from Functions_Required import *
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

##################################################################################
########################### Generate TVP-SVAR datasets ###########################
##################################################################################

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Function to generate random walk
def random_walk(T, scl):
    #e_t = np.random.normal(size=T, scale=scl)
    e_t = np.random.normal(size=T, scale=scl)
    rw_t = np.zeros(T)
    rw_t[0] = e_t[0]
    for i in range(1,T):
        rw_t[i] = rw_t[i-1] + e_t[i]
    return rw_t

# Function to generate Y_tvp
def tvp_simulate(n_obs = 200, n_var = 3, n_lags = 4, intercept = 1):
    # number of observations, variables, and lags

    # Define Y_TVP: we also initialize it with the minimum number
    # of observations required ( = number of lags) to begin generating
    # the further observations. We slice them out in the end.
    y_sim_tvp = pd.DataFrame(
        data=[np.random.normal(size=n_var) for _ in range(n_lags)],
        columns=['y'+str(c_num) for c_num in range(1,n_var+1)]
    )

    # Define epsilon - the vector of error terms.
    epsilon_sim = np.array([np.random.normal(size=n_obs) for _ in range(n_var)])

    # Define B matrix for all coefficients of B_t (slope coefficients)
    # and c_t (intercept). Note that the total number of elements
    # would be n_var*(1 + (n_var*n_lags)).
    # B_mat = np.array([random_walk(n_obs, (0.05/100)**0.5) for _ in range(n_var*(1 + (n_var*n_lags)))])
    B_mat = np.array([np.zeros(n_obs) for _ in range(n_var*(1 + (n_var*n_lags)))])
    # Fill the B_mat_0 with some initial values
    B_mat[:,0] = np.random.randn(n_var*(1 + (n_var*n_lags)))*((0.05/100)**0.5)
    # Also, check the stability condition by forming the companion matrix at t=0
    # stuck counter: if the eigenvalues do not satisfy the stability condition
    # for too long, we return 'stuck' to avoid infinite loop.
    stuck = 0
    # No need to understand the following if loop, just see the subsequent codes.
    if max(
        abs(np.linalg.eigvals(np.concatenate(
            (B_mat[:,0].reshape(n_var,(n_var*n_lags)+1)[:,1:],
             np.concatenate((
                 np.eye(n_var*(n_lags-1)),
                 np.zeros((n_var*(n_lags-1),n_var))
             ), axis = 1)), axis = 0
        )))
    ) > 0.95 : return 'stuck'
    # For the subsequent periods, we confirm that the eienvalues of B's are less than 1
    t = 1
    while t < n_obs:
        B_mat[:,t] = B_mat[:,t-1] + np.random.randn(n_var*(1 + (n_var*n_lags)))*((0.05/100)**0.5)
        # Check the stability condition by forming the companion matrix
        Phi_mat = B_mat[:,t].reshape(n_var,(n_var*n_lags)+1)
        # remove intercept
        Phi_mat = Phi_mat[:,1:]
        # Companion matrix
        comp_mat = np.concatenate((np.eye(n_var*(n_lags-1)),np.zeros((n_var*(n_lags-1),n_var))), axis = 1)
        comp_mat = np.concatenate((Phi_mat, comp_mat), axis = 0)
        # Check eigenvalues
        if max(abs(np.linalg.eigvals(comp_mat))) > 0.95: t -= 1
        t += 1
        stuck += 1
        if stuck > 10000: return 'stuck'
    # Define alpha matrix for A_t matrix (shocks)
    alpha_t = np.array([random_walk(n_obs, (0.5/100)**0.5) for _ in range(np.int64((n_var*(n_var - 1))/2))])

    for t in range(n_obs):
        # We form the matrices/vectors at time t.
        # For X_t' matrix, we first collect the
        # vector [y_t-1, ..., y_t-p] for the vector
        # [1, y_t-1, ..., y_t-p] for p being the
        # number of lags (n_lags).
        y_t_lags = y_sim_tvp.iloc[::-1].iloc[:n_lags]
        # Define X_t' matrix using the Kronecker product.
        X_t = np.kron(
            np.identity(n_var),
            np.append(intercept,y_t_lags)
        )
        # Define B_t matrix at time t
        B_t = B_mat[:,t]
        # Define the shock matrix A_t at time t
        A_t = np.eye(n_var)
        A_t[np.tril_indices(n_var, k=-1)] = alpha_t[:,t]
        # We add the observation at time t
        y_sim_tvp.loc[n_lags + t] = np.matmul(X_t, B_t.T) + np.matmul(np.linalg.inv(A_t), epsilon_sim[:,t])
    # Slice the initialized values out
    y_sim_tvp = y_sim_tvp.iloc[n_lags:].reset_index(drop=True)
    y_sim_tvp.index += 1
    # Check the stability condition
    # Return a dictionary of all elements
    return {
        'data': y_sim_tvp,
        'B_mat': B_mat,
        'alpha_t': alpha_t,
        'n_obs': n_obs,
        'n_var': n_var,
        'n_lags': n_lags
    }

test_sim = tvp_simulate(200, 4, 4)
# np.mean(test_sim['B_mat'][2,:])
# plt.plot(test_sim['B_mat'][7,:]);plt.show()
# dataplot(test_sim['data'])

def tvp_irf(sim_elements, impulse = 0):
    if sim_elements == 'stuck':
        return 'stuck'
    # Collect the basic variables.
    n_obs = sim_elements['n_obs']
    n_var = sim_elements['n_var']
    n_lags = sim_elements['n_lags']
    # Collect the alpha matrix to capture the shocks
    alpha_t = sim_elements['alpha_t']
    # Collect the coefficient matrices to capture the shocks
    B_mat = sim_elements['B_mat']

    # Fix the lower triangular matrix
    A_t = np.eye(n_var)
    A_t[np.tril_indices(n_var, k=-1)] = alpha_t[:,n_obs-1]

    # Collect the coefficient matrices to form the companion matrix
    Phi_mat = B_mat[:,n_obs-1]
    Phi_mat = Phi_mat.reshape(n_var,(n_var*n_lags)+1)
    Phi_mat = Phi_mat[:,1:]

    # Companion matrix
    comp_mat = np.concatenate((np.eye(n_var*(n_lags-1)),np.zeros((n_var*(n_lags-1),n_var))), axis = 1)
    comp_mat = np.concatenate((Phi_mat, comp_mat), axis = 0)
    
    J = np.concatenate((np.eye(n_var),np.zeros((n_var,n_var*(n_lags-1)))), axis = 1)

    Phi_i = [np.matmul(np.matmul(J,np.linalg.matrix_power(comp_mat,_)), J.T) for _ in range(0,11)]
    Theta = [np.matmul(_,np.linalg.inv(A_t)) for _ in Phi_i]

    irf_tvp = pd.DataFrame([_[:,impulse] for _ in Theta], columns=sim_elements['data'].columns)

    return irf_tvp

def knn_irf(sim_elements, impulse=0):
    if sim_elements == 'stuck':
        return 'stuck'
    delta_y = sim_elements['data'].copy()
    n_lags = sim_elements['n_lags']
    n_var = sim_elements['n_var']
    omega = pd.concat([delta_y, sm.tsa.tsatools.lagmat(delta_y, maxlag=n_lags, use_pandas=True).iloc[n_lags:]], axis = 1)
    omega = omega.dropna()

    omega_mean = omega.mean()
    omega_std = omega.std()
    omega_scaled = (omega - omega_mean)/omega_std
    histoi = omega.iloc[-10:].mean()
    omega_scaled = omega_scaled.iloc[:-10]
    histoi = (histoi - omega_mean)/omega_std
    T = omega_scaled.shape[0]

    knn = NearestNeighbors(n_neighbors=T, metric='euclidean')
    knn.fit(omega_scaled)
    dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

    # Estimate y_T
    y_f = np.matmul(delta_y.loc[omega_scaled.iloc[ind].index].T, weig).to_frame().T
    # y_f = np.matmul(y.loc[omega_scaled.iloc[ind].index].T, weig).to_frame().T
    for h in range(1,10+1):
        y_f.loc[h] = np.matmul(delta_y.loc[omega_scaled.iloc[ind].index + h].T, weig).values
    # dataplot(y_f)
    u = delta_y - y_f.loc[0].values.squeeze()
    u = u.iloc[:T]
    # u_mean = u.mul(weig, axis = 0)
    ################IMPORTANT CORRECTION HERE##################
    u_mean = u.mean()
    sigma_u = np.matmul((u - u_mean).T, (u - u_mean).mul(weig, axis = 0)) / (1 - np.sum(weig**2))
    # Cholesky decomposition
    B_mat = np.transpose(np.linalg.cholesky(sigma_u))
    # The desired shock
    # B_mat = np.transpose(np.linalg.cholesky(u.cov()*((T-1)/(T-8-1))))
    delta = B_mat[impulse]

    # Estimate y_T_delta
    y_f_delta = pd.DataFrame(columns=y_f.columns)
    y_f_delta.loc[0] = y_f.loc[0] + delta

    histoi_delta = (y_f.iloc[0] + delta - omega_mean[:n_var])/omega_std[:n_var]
    histoi_delta = pd.concat([histoi_delta, histoi], axis=0)[:-n_var]

    dist, ind = knn.kneighbors(histoi_delta.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

    for h in range(1,10+1):
        y_f_delta.loc[h] = np.matmul(delta_y.loc[omega_scaled.iloc[ind].index + h].T, weig).values
    # dataplot(y_f_delta)

    girf = y_f_delta - y_f
    return girf

sim_T = tvp_simulate(1000, 4, 1)
dataplot(tvp_irf(sim_T) - knn_irf(sim_T))

# Bgin simulations

n_sim = 50

bias = []

for n_obs in [_*200 for _ in range(1,6)]:
    for n_var in range(2,5):
        for n_lags in range(1,5):
            counter = 0
            while counter < n_sim:
                sim = tvp_simulate(n_obs, n_var, n_lags, intercept=1)
                if sim == 'stuck':
                    counter -= 1
                else:
                    try:
                        bias.append(knn_irf(sim).T - tvp_irf(sim).T)
                        if bias[-1].isnull().values.any():
                            bias.pop()
                            counter -= 1
                    except:
                        counter -= 1
                        pass
                print(str(n_obs) + ',' + str(n_var) + ',' +str(n_lags) + ',' +str(counter))
                counter += 1
#End

import pickle
# Saving objects:
with open('objs.pkl', 'wb') as f:
    pickle.dump(bias, f)

# Getting back the objects:
import pickle
with open('objs.pkl', 'rb') as f:
    bias = pickle.load(f)

bias_avg = [np.absolute(_).mean() for _ in bias]
bias_avg = sum(bias_avg)/len(bias_avg)
bias_avg.plot(); plt.show()

rmse_avg = [np.absolute(_).mean()**2 for _ in bias]
rmse_avg = ((sum(rmse_avg)/len(rmse_avg)))**0.5
rmse_avg.plot(); plt.show()

# n_var: 2,3,4: n_obs: 200,400,600,800,1000: n_lags:1,2,3,4
# Separate bias list by n_var
bias_avg_var = [[_ for _ in bias if len(_.index) == count] for count in range(2,5)]
bias_avg_var = [[np.absolute(b).mean() for b in _] for _ in bias_avg_var]
bias_avg_var = [sum(_)/len(_) for _ in bias_avg_var]
bias_avg_var = pd.DataFrame([_ for _ in bias_avg_var], index=[_ for _ in range(2,5)]).T
bias_avg_var.plot();plt.show()

rmse_avg_var = [[_ for _ in bias if len(_.index) == count] for count in range(2,5)]
rmse_avg_var = [[np.absolute(b).mean()**2 for b in _] for _ in rmse_avg_var]
rmse_avg_var = [(sum(_)/len(_))**0.5 for _ in rmse_avg_var]
rmse_avg_var = pd.DataFrame([_ for _ in rmse_avg_var], index=[_ for _ in range(2,5)]).T
rmse_avg_var.plot();plt.show()

# Separate bias list by n_obs
# n_obs: 200 = n_var:2,3,4 & 200 obs used in n_lags: 1,2,3,4; and so on
# so, each n_obs is repeated 
bias_avg_T = [bias[(_*(n_sim*3*4)):(_+1)*(n_sim*3*4)] for _ in range(5)]
bias_avg_T = [[np.absolute(b).mean() for b in _] for _ in bias_avg_T]
bias_avg_T = [sum(_)/len(_) for _ in bias_avg_T]
bias_avg_T = pd.DataFrame([_ for _ in bias_avg_T], index=[_*200 for _ in range(1,6)]).T
bias_avg_T.plot(); plt.show()

rmse_avg_T = [bias[(_*(n_sim*3*4)):(_+1)*(n_sim*3*4)] for _ in range(5)]
rmse_avg_T = [[(b**2).mean() for b in _] for _ in rmse_avg_T]
rmse_avg_T = [(sum(_)/len(_))**0.5 for _ in rmse_avg_T]
rmse_avg_T = pd.DataFrame([_ for _ in rmse_avg_T], index=[_*200 for _ in range(1,6)]).T
rmse_avg_T.plot(); plt.show()



def knn_irf_1(sim_elements, impulse=0):
    if sim_elements == 'stuck':
        return 'stuck'
    delta_y = sim_elements['data'].copy()
    n_var = sim_elements['n_var']
    omega = delta_y.copy()
    omega = omega.dropna()

    omega_mean = omega.mean()
    omega_std = omega.std()
    omega_scaled = (omega - omega_mean)/omega_std
    histoi = omega.iloc[-10:].mean()
    omega_scaled = omega_scaled.iloc[:-10]
    histoi = (histoi - omega_mean)/omega_std
    T = omega_scaled.shape[0]

    knn = NearestNeighbors(n_neighbors=T, metric='euclidean')
    knn.fit(omega_scaled)
    dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

    # Estimate y_T
    y_f = np.matmul(delta_y.loc[omega_scaled.iloc[ind].index].T, weig).to_frame().T
    # y_f = np.matmul(y.loc[omega_scaled.iloc[ind].index].T, weig).to_frame().T
    for h in range(1,10+1):
        y_f.loc[h] = np.matmul(delta_y.loc[omega_scaled.iloc[ind].index + h].T, weig).values
    # dataplot(y_f)
    u = delta_y - y_f.loc[0].values.squeeze()
    u = u.iloc[:T]
    # u_mean = u.mul(weig, axis = 0)
    ################IMPORTANT CORRECTION HERE##################
    u_mean = u.mean()
    sigma_u = np.matmul((u - u_mean).T, (u - u_mean).mul(weig, axis = 0)) / (1 - np.sum(weig**2))
    # Cholesky decomposition
    B_mat = np.transpose(np.linalg.cholesky(sigma_u))
    # The desired shock
    # B_mat = np.transpose(np.linalg.cholesky(u.cov()*((T-1)/(T-8-1))))
    delta = B_mat[impulse]

    # Estimate y_T_delta
    y_f_delta = pd.DataFrame(columns=y_f.columns)
    y_f_delta.loc[0] = y_f.loc[0] + delta

    histoi_delta = (y_f.iloc[0] + delta - omega_mean)/omega_std

    dist, ind = knn.kneighbors(histoi_delta.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

    for h in range(1,10+1):
        y_f_delta.loc[h] = np.matmul(delta_y.loc[omega_scaled.iloc[ind].index + h].T, weig).values
    # dataplot(y_f_delta)

    girf = y_f_delta - y_f
    return girf


n_sim = 50
bias_1 = []

for n_obs in [_*200 for _ in range(1,6)]:
    for n_var in range(2,5):
        for n_lags in range(1,5):
            counter = 0
            while counter < n_sim:
                sim = tvp_simulate(n_obs, n_var, n_lags, intercept=1)
                if sim == 'stuck':
                    counter -= 1
                else:
                    try:
                        bias_1.append(knn_irf(sim).T - tvp_irf(sim).T)
                        if bias[-1].isnull().values.any():
                            bias.pop()
                            counter -= 1
                    except:
                        counter -= 1
                        pass
                print(str(n_obs) + ',' + str(n_var) + ',' +str(n_lags) + ',' +str(counter))
                counter += 1
#End

import pickle
# Saving objects:
with open('objs_1.pkl', 'wb') as f:
    pickle.dump(bias_1, f)
# Getting back the objects:

import pickle
with open('objs_1.pkl', 'rb') as f:
    bias_1 = pickle.load(f)
# End of code

bias_avg_1 = [np.absolute(_).mean() for _ in bias_1]
bias_avg_1 = sum(bias_avg_1)/len(bias_avg_1)
bias_avg_1.plot(); plt.show()

rmse_avg_1 = [np.absolute(_).mean()**2 for _ in bias_1]
rmse_avg_1 = ((sum(rmse_avg_1)/len(rmse_avg_1)))**0.5
rmse_avg_1.plot(); plt.show()


# n_var: 2,3,4: n_obs: 200,400,600,800,1000: n_lags:1,2,3,4
# Separate bias list by n_var
bias_avg_var = [[_ for _ in bias if len(_.index) == count] for count in range(2,5)]
bias_avg_var = [[np.absolute(b).mean() for b in _] for _ in bias_avg_var]
bias_avg_var = [sum(_)/len(_) for _ in bias_avg_var]
bias_avg_var = pd.DataFrame([_ for _ in bias_avg_var], index=[_ for _ in range(2,5)]).T
bias_avg_var.plot();plt.show()

rmse_avg_var = [[_ for _ in bias if len(_.index) == count] for count in range(2,5)]
rmse_avg_var = [[np.absolute(b).mean()**2 for b in _] for _ in rmse_avg_var]
rmse_avg_var = [(sum(_)/len(_))**0.5 for _ in rmse_avg_var]
rmse_avg_var = pd.DataFrame([_ for _ in rmse_avg_var], index=[_ for _ in range(2,5)]).T
rmse_avg_var.plot();plt.show()