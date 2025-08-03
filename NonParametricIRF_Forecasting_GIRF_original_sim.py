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

# Random walk generator
def random_walk(n):
    e_t = np.random.normal(size=n, scale=np.random.randint(low=1, high=5))
    rw_t = np.zeros(n)
    rw_t[0] = e_t[0]
    for i in range(1,n):
        rw_t[i] = rw_t[i-1] + e_t[i]
    return rw_t

n_obs = 10
n_var = 3
n_lags = 4
intercpt = False

if intercpt == True:
    B_mat = np.array([random_walk(n_obs) for _ in range((n_var+1)*n_lags)])
else:
    B_mat = np.array([random_walk(n_obs) for _ in range(n_var*n_lags)])

alpha_t = np.array([random_walk(n_obs) for _ in range(np.int64((n_var*(n_var - 1))/2))])

# Define Y_TVP
y_sim = pd.DataFrame(index=range(n_obs), columns=['y'+str(c_num) for c_num in range(1,n_var+1)])
# Initialize Y
for i in range(n_lags):
    y_sim.iloc[i] = np.random.normal(size=n_var)

y_sim.iloc[4]
np.reshape(B_mat[:,0], (n_var+1,n_var))
B_mat[:,0].shape
