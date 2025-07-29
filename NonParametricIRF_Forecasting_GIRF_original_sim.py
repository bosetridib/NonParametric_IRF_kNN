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
