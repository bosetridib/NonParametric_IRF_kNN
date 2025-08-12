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
import matplotlib.pyplot as plt

# Function to generate random walk
def random_walk(T, scl):
    e_t = np.random.normal(size=T, scale=scl)
    rw_t = np.zeros(T)
    rw_t[0] = e_t[0]
    for i in range(1,T):
        rw_t[i] = rw_t[i-1] + e_t[i]
    return rw_t

# Function to generate Y_tvp
def y_tvp(n_obs = 200, n_var = 3, n_lags = 4):
    intercept = 1
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
    return {'data': y_sim_tvp, 'B_mat': B_mat, 'alpha_t': alpha_t}

n_obs = 400
n_var = 3
n_lags = 4

y_sim_tvp = y_tvp(n_obs,n_var,n_lags)
data = y_sim_tvp['data']
# y_sim_tvp_data.plot(subplots=True);plt.show()
alpha_t = y_sim_tvp['alpha_t']
B_mat = y_sim_tvp['B_mat']

Phi_mat = B_mat[:,n_obs-1]
Phi_mat = Phi_mat.reshape(n_var,(n_var*n_lags)+1)
Phi_mat