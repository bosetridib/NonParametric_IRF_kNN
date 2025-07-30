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

# Function to generate time-varying parameters
def random_walk(n):
    e_t = np.random.normal(size=n, scale=np.random.randint(low=1, high=5))
    rw_t = np.zeros(n)
    rw_t[0] = e_t[0]
    for i in range(1,n):
        rw_t[i] = rw_t[i-1]+ e_t[i]
    
    return rw_t

n_obs = 10
B_t = np.matrix([random_walk(n_obs) for _ in range(10)])
A_t = np.tril([random_walk(n_obs) for _ in range(5)])
A_t = np.eye(5)