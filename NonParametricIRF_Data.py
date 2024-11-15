# Code for Nonparametric Impulse Response analysis in changing Macroeconomic conditions

# Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader

# Data collection:

# Reference: Gavriilidis, K. (2021)
cpu = pd.read_csv(
    'https://www.policyuncertainty.com/media/CPU%20index.csv',
    skiprows=4,
    usecols=['date', 'cpu_index'], index_col='date'
)
cpu.index = pd.to_datetime(cpu.index, format="%b-%y")

epu = pd.read_excel('https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx', skipfooter=1)
epu.set_index(
    pd.to_datetime(
        epu.Year.astype(str) + '-' + epu.Month.astype(str),
        format="%Y-%m"
    ), inplace=True
)
epu.drop(['Year', 'Month', 'Three_Component_Index'], axis=1, inplace=True)

# Slicing the EPU data based on CPU dates
epu = epu.loc[cpu.index[0]:cpu.index[-1]]
epu.columns = ['epu_index']

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
temperature = temperature.loc[cpu.index[0]:cpu.index[-1]]
# temperature.plot(); plt.show()

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

df = pd.concat([temperature, epu, cpu, macro_data], axis=1)
df_mod = df.copy()
df_mod[['Industrial_Production', 'PriceIndex_Producer', 'PriceIndex_PCE']] = np.log(df[['Industrial_Production', 'PriceIndex_Producer', 'PriceIndex_PCE']]).diff()
df_mod = df_mod.rename(columns={
    'Industrial_Production':'D_log_Industrial_Production',
    'PriceIndex_Producer':'D_log_PriceIndex_Producer',
    'PriceIndex_PCE':'D_log_PriceIndex_PCE'
})
df = df.dropna()
df_mod = df_mod.dropna()
# df.plot(subplots=True, layout=(2,4)); plt.show()
# df = pd.concat([epu, cpu, macro_data], axis=1)
# df = df.loc[:'2019-06-01']

# Financial Crisis and Trump year
# df_fin = df.drop(macro_data.loc['2007-12-01':'2009-06-01'].index)
# df_trump = df.drop(macro_data.loc['2017-02-01':].index)

# Define the irf plotting function
def irfplot(irf,df,c):
    fig, ax = plt.subplots(2,4)
    k = 0
    for i in range(2):
        for j in range(4):
            if (k>irf.shape[2]-1): continue
            ax[i,j].plot(irf[:,k,c])
            ax[i,j].grid(True)
            ax[i,j].axhline(y=0, color = 'k')
            ax[i,j].title.set_text(df.columns[c] + ">" + df.columns[k])
            k = k + 1
    fig.show()

# Subplots
def dataplot(data):
    data.plot(subplots=True, layout=(2,4))
    plt.show()
# Send everything here to the Estimation py file