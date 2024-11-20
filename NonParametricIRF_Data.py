# Code for 'Nonparametric Impulse Response analysis in changing Macroeconomic conditions'

# Import required packages
import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader

# Data collection:

# Source: Gavriilidis, K. (2021). Measuring Climate Policy Uncertainty. Available at SSRN: https://ssrn.com/abstract=3847388"
cpu = pd.read_csv(
    'https://www.policyuncertainty.com/media/CPU%20index.csv',
    skiprows=4,
    usecols=['date', 'cpu_index'], index_col='date'
)
cpu.index = pd.to_datetime(cpu.index, format="%b-%y")

# Source: 'Measuring Economic Policy Uncertainty' by Scott Baker, Nicholas Bloom and Steven J. Davis at www.PolicyUncertainty.com.
epu = pd.read_excel('https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx', skipfooter=1)
epu.set_index(
    pd.to_datetime(
        epu.Year.astype(str) + '-' + epu.Month.astype(str),
        format="%Y-%m"
    ), inplace=True
)
epu.drop(['Year', 'Month', 'Three_Component_Index'], axis=1, inplace=True)

# See Macroeconomic impact of climate change Bilal Kanzig
# For licencing contact: https://berkeleyearth.org/data/
temperature = pd.read_fwf(
    'https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Regional/TAVG/united-states-TAVG-Trend.txt',
    skiprows=70, header=None, usecols = [0,1,2], names = ['Year','Month','TempAnomaly']
)
temperature.set_index(
    pd.to_datetime(
        temperature.Year.astype(str) + '-' + temperature.Month.astype(str),
        format="%Y-%m"
    ), inplace=True
)
temperature.drop(['Year', 'Month'], axis=1, inplace=True)
# temperature.plot(); plt.show()

# Slicing the EPU data, temperature based on CPU dates
epu = epu.loc[cpu.index[0]:cpu.index[-1]]
epu.columns = ['epu_index']
temperature = temperature.loc[cpu.index[0]:cpu.index[-1]]

# References:
# 1. Board of Governors of the Federal Reserve System (US), Industrial Production: Total Index [INDPRO], retrieved from FRED,
# Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/INDPRO, November 19, 2024.
# 2. U.S. Bureau of Labor Statistics, Unemployment Rate [UNRATE], retrieved from FRED,
# Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/UNRATE, November 19, 2024.
# 3. U.S. Bureau of Labor Statistics, Producer Price Index by Commodity: All Commodities [PPIACO], retrieved from FRED,
# Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PPIACO, November 19, 2024.
# 4. U.S. Bureau of Economic Analysis, Personal Consumption Expenditures: Chain-type Price Index [PCEPI], retrieved from FRED,
# Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PCEPI, November 19, 2024.
# 5. Board of Governors of the Federal Reserve System (US), 3-Month Treasury Bill Secondary Market Rate, Discount Basis [TB3MS],
# retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/TB3MS, November 19, 2024.

macro_data = DataReader(
    ['INDPRO', 'UNRATE', 'PPIACO', 'PCEPI', 'TB3MS'],
    'fred',
    start=cpu.index[0],  end=cpu.index[-1]
)
macro_data.columns = [
    'Industrial_Production',
    'Unemployment_Rate',
    'PriceIndex_Producer',
    'PriceIndex_PCE',
    'Treasurey3Months'
]

# Dataframe for the raw data.
df = pd.concat([temperature, epu, cpu, macro_data], axis=1)

# Dataframe for the modified data.
df_mod = df.copy()
df_mod[['Industrial_Production', 'PriceIndex_Producer', 'PriceIndex_PCE']] = np.log(df[['Industrial_Production', 'PriceIndex_Producer', 'PriceIndex_PCE']]).diff()
df_mod = df_mod.rename(columns={
    'Industrial_Production':'D_log_Industrial_Production',
    'PriceIndex_Producer':'D_log_PriceIndex_Producer',
    'PriceIndex_PCE':'D_log_PriceIndex_PCE'
})

df = df.dropna()
df_mod = df_mod.dropna()

# Comparison with 'The Macroeconomic Effects of Climate Policy Uncertainty' by Gavriilidis, Kanzig, and Stock (2023)
# df = df.loc[:'2019-06-01']
# Financial Crisis and Trump year
# df_fin = df.drop(macro_data.loc['2007-12-01':'2009-06-01'].index)
# df_trump = df.drop(macro_data.loc['2017-02-01':].index)

# Send everything here to the Estimation.py file