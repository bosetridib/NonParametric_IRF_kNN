# Code for 'Nonparametric Impulse Response analysis in changing Macroeconomic conditions'
# Data from Gavriilidis, Känzig, and Stock (2023)
# Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

cpu = pd.read_csv(
    'https://www.policyuncertainty.com/media/CPU%20index.csv',
    skiprows=4,
    usecols=['date', 'cpu_index'], index_col='date'
)
cpu.index = pd.to_datetime(cpu.index, format="%b-%y")
# From the following plot, it seems that the data ends
# on June-19
# cpu.loc[:'2019-06-01'].plot(grid=True); plt.show()
# Hence, we get the date range of monthly data from
# Apr-87 to Jun-19
cpu = cpu.loc['1987-04-01':'2019-06-01']

epu = pd.read_excel('https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx', skipfooter=1)
epu.set_index(
    pd.to_datetime(
        epu.Year.astype(str) + '-' + epu.Month.astype(str),
        format="%Y-%m"
    ), inplace=True
)
epu.drop(
    ['Year', 'Month', 'Three_Component_Index'],
    axis=1,
    inplace=True
)
epu = epu.loc['1987-04-01':'2019-06-01']
epu.columns = ['epu_index']

macro_data = DataReader(
    ['INDPRO', 'UNRATE', 'PPIACO', 'PCEPI', 'TB3MS'],
    'fred',
    start='1987-04-01',  end='2019-06-01'
)
macro_data.columns = [
    'Industrial_Production',
    'Unemployment_Rate',
    'PriceIndex_Producer',
    'PriceIndex_PCE',
    'Treasurey3Months'
]

# Chow-Lin interpolation

X = DataReader(
    ['CPIENGSL', 'INDPRO', 'PPIACO'],
    'fred',
    start='1987-04-01',  end='2019-06-01'
)

Y = DataReader(
    'EMISSCO2TOTVTTTOUSA',
    'fred',
    start='1987-04-01',  end='2019-06-01'
)
Y = Y.asfreq(pd.infer_freq(Y.index))

from tsdisagg import disaggregate_series

Emission_CO2 = disaggregate_series(
    low_freq_df=Y,
    high_freq_df=X.assign(intercept=1),
    method="chow-lin",
    agg_func="first"
)
# optimizer_kwargs={"method": "L-BFGS-B"}
# Emission_CO2.plot(); plt.show()
macro_data.insert(4, 'Emission_CO2', Emission_CO2/10)

# Define the trend variables
trend = macro_data.columns[[0,2,3,4]]