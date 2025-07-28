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
from statsmodels.tsa.api import VAR

# Function to generate time-varying parameters
def generate_time_varying_parameters(n_obs, n_vars, seed=42):
    np.random.seed(seed)
    # Create a matrix to hold the parameters
    parameters = np.zeros((n_obs, n_vars, n_vars))
    # Initialize with stable values and small random changes over time
    for t in range(n_obs):
        parameters[t] = np.identity(n_vars) * 0.8
        # Apply small random fluctuations
        parameters[t] += np.random.normal(0, 0.01, (n_vars, n_vars))
    return parameters

# Generate synthetic data
def generate_data(parameters, n_obs, n_vars, seed=42):
    np.random.seed(seed)
    y = np.zeros((n_obs, n_vars))
    for t in range(1, n_obs):
        y[t] = y[t-1] @ parameters[t] + np.random.normal(0, 0.1, n_vars)
    return y

# Main function to simulate a TVP-VAR process
def simulate_tvp_var(n_obs=100, n_vars=2):
    parameters = generate_time_varying_parameters(n_obs, n_vars)
    data = generate_data(parameters, n_obs, n_vars)
    return data, parameters

# Generate and plot the data
n_obs = 100
n_vars = 3
data, parameters = simulate_tvp_var(n_obs, n_vars)

# Convert to DataFrame for easier handling
data_df = pd.DataFrame(data, columns=[f'Var{i}' for i in range(n_vars)])

# Plot the data
data_df.plot(title='Time-Varying Parameters VAR Simulation')
plt.show()