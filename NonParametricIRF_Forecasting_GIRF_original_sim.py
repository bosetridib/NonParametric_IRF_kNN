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
    B_mat = np.array([random_walk(n_obs, 0.05/100) for _ in range(n_var*(1 + (n_var*n_lags)))])

    # Define alpha matrix for A_t matrix (shocks)
    alpha_t = np.array([random_walk(n_obs, 0.5/100) for _ in range(np.int64((n_var*(n_var - 1))/2))])

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
    # Return a dictionary of all elements
    return {
        'data': y_sim_tvp,
        'B_mat': B_mat,
        'alpha_t': alpha_t,
        'n_obs': n_obs,
        'n_var': n_var,
        'n_lags': n_lags
    }

def tvp_irf(sim_elements, impulse):
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

    Phi_i = [np.matmul(np.matmul(J,np.linalg.matrix_power(comp_mat,_)), J.T) for _ in range(0,41)]
    Theta = [np.matmul(_,np.linalg.inv(A_t)) for _ in Phi_i]

    irf_tvp = pd.DataFrame([_[:,impulse] for _ in Theta], columns=sim_elements['data'].columns)

    return irf_tvp

def knn_irf(data, impulse):
    omega = data.copy()
    omega_mean = omega.mean()
    omega_std = omega.std()
    omega_scaled = (omega - omega_mean)/omega_std
    histoi = omega.iloc[-40:].mean()
    omega_scaled = omega_scaled.iloc[:-40]
    histoi = (histoi - omega_mean)/omega_std
    T = omega_scaled.shape[0]

    knn = NearestNeighbors(n_neighbors=T, metric='euclidean')
    knn.fit(omega_scaled)
    dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

    # Estimate y_T
    y_f = np.matmul(omega.loc[omega_scaled.iloc[ind].index].T, weig).to_frame().T
    # y_f = np.matmul(y.loc[omega_scaled.iloc[ind].index].T, weig).to_frame().T
    for h in range(1,40+1):
        y_f.loc[h] = np.matmul(omega.loc[omega_scaled.iloc[ind].index + h].T, weig).values
    # dataplot(y_f)
    u = omega - y_f.loc[0].values.squeeze()
    # u_mean = u.mul(weig, axis = 0)
    u = u.iloc[:-40]
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

    histoi_delta = (y_f.iloc[0] + delta - omega_mean.values)/omega_std.values
    # histoi_delta = pd.concat([histoi_delta, histoi], axis=0)[:-omega.shape[1]]

    dist, ind = knn.kneighbors(histoi_delta.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

    for h in range(1,40+1):
        y_f_delta.loc[h] = np.matmul(omega.loc[omega_scaled.iloc[ind].index + h].T, weig).values
    # dataplot(y_f_delta)

    girf = y_f_delta - y_f
    return girf

# Bgin simulations
# n_obs = 400
# n_var = 4
# n_lags = 4

n_sim = 50
impulse = 0

bias = []

for n_obs in [_*200 for _ in range(1,6)]:
    for n_var in range(3,11):
        for n_lags in [_*2 for _ in range(1,7)]:
            for _ in range(n_sim):
                sim = tvp_simulate(n_obs, n_var, n_lags)
                bias.append(knn_irf(sim['data'], impulse))
                print(str(n_obs) + ',' + str(n_var) + ',' +str(n_lags) + ',' +str(_))
#End

bias_avg = [np.absolute(_).mean(axis=1) for _ in bias]
bias_avg = sum(bias_avg)/len(bias_avg)
bias_avg = pd.DataFrame([_ for _ in bias_avg], index=[_ for _ in range(0,41)])
bias_avg.plot()

rmse_avg = [(_**2).mean(axis=1) for _ in bias]
rmse_avg = ((sum(rmse_avg)/len(rmse_avg)))**0.5
rmse_avg = pd.DataFrame([_ for _ in rmse_avg], index=[_ for _ in range(0,41)])
rmse_avg.plot()


# n_sim * 6 lags * 8 vars = 1040
bias_avg_T = [bias[(_*(n_sim*6*8)):(_+1)*(n_sim*6*8)] for _ in range(5)]
bias_avg_T = [[np.absolute(b).mean(axis=1) for b in _] for _ in bias_avg_T]
bias_avg_T = [sum(_)/len(_) for _ in bias_avg_T]
bias_avg_T = pd.DataFrame([_ for _ in bias_avg_T], index=[_*200 for _ in range(1,6)]).T
bias_avg_T.plot()

rmse_avg_T = [bias[(_*(n_sim*6*8)):(_+1)*(n_sim*6*8)] for _ in range(5)]
rmse_avg_T = [[(b**2).mean(axis=1) for b in _] for _ in rmse_avg_T]
rmse_avg_T = [(sum(_)/len(_))**0.5 for _ in rmse_avg_T]
rmse_avg_T = pd.DataFrame([_ for _ in rmse_avg_T], index=[_*200 for _ in range(1,6)]).T
rmse_avg_T.plot()

# n_sim * 5 lags = 130
bias_avg_var = [[_ for _ in bias if len(_.columns) == count] for count in range(3,11)]
bias_avg_var = [[np.absolute(b).mean(axis=1) for b in _] for _ in bias_avg_var]
bias_avg_var = [sum(_)/len(_) for _ in bias_avg_var]
bias_avg_var = pd.DataFrame([_ for _ in bias_avg_var], index=[_ for _ in range(3,11)]).T
bias_avg_var.plot();plt.show()

rmse_avg_var = [[_ for _ in bias if len(_.columns) == count] for count in range(3,11)]
rmse_avg_var = [[(b**2).mean(axis=1) for b in _] for _ in rmse_avg_var]
rmse_avg_var = [(sum(_)/len(_))**0.5 for _ in rmse_avg_var]
rmse_avg_var = pd.DataFrame([_ for _ in rmse_avg_var], index=[_ for _ in range(3,11)]).T
rmse_avg_var.plot();plt.show()

import pickle
# Saving objects:
with open('objs.pkl', 'wb') as f:
    pickle.dump(bias, f)
# Getting back the objects:
import pickle
with open('objs.pkl', 'rb') as f:
    bias = pickle.load(f)
