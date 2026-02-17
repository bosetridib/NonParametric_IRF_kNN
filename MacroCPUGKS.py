# Replication of The macroeconomic Effects of CPU
# by Gavriilidis, Känzig, and Stock (2023)

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
    ['Year', 'Month'],
    axis=1,
    inplace=True
)
epu = epu.sort_index().loc['1987-04-01':'2019-06-01']
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

df = pd.concat([epu, cpu, macro_data], axis=1)

# df.plot(subplots=True, layout=(2,4)); plt.show()

# Var analysis
model_var = sm.tsa.VAR(df)

results_var = model_var.fit(6)
results_var.summary()
u_t = results_var.resid
print(results_var.test_normality().summary())
import statsmodels
print(statsmodels.stats.diagnostic.normal_ad(u_t))
# np.linalg.cholesky(results_var.resid_corr).tolist()

results_var.irf(40).plot(orth = True, impulse = 'cpu_index')
plt.show()

import localprojections as lp
irf = lp.TimeSeriesLP(
    data = df,
    Y = df.columns.to_list(),
    response=df.columns.to_list(),
    horizon=40, lags=6
)
lp.IRFPlot(
    irf = irf,
    response=df.columns.to_list(),
    shock=['cpu_index'],
    n_rows=2,n_columns=4
)

results_var.irf(40).plot(orth = True, impulse = 'epu_index')
plt.show()

# Controlling for EPU
df_new = pd.concat([cpu, macro_data], axis=1)

model_var_new = sm.tsa.VAR(df_new)

results_var_new = model_var_new.fit(6)
results_var_new.summary()

results_var_new.irf(40).plot(orth = True, impulse = 'cpu_index')
plt.show()

# Stock Price Index: M1109BUSM293NNBR, House Price Index: CSUSHPISA, Consumer Sentiment: UMCSENT
macro_data_stock = DataReader(
    ['INDPRO', 'UNRATE', 'PPIACO', 'PCEPI', 'M1109BUSM293NNBR'],
    'fred',
    start='1987-04-01',  end='2019-06-01'
)
macro_data_house = DataReader(
    ['INDPRO', 'UNRATE', 'PPIACO', 'PCEPI', 'CSUSHPISA'],
    'fred',
    start='1987-04-01',  end='2019-06-01'
)
macro_data_cs = DataReader(
    ['INDPRO', 'UNRATE', 'PPIACO', 'PCEPI', 'UMCSENT'],
    'fred',
    start='1987-04-01',  end='2019-06-01'
)

macro_data_stock.insert(4, 'Emission_CO2', Emission_CO2/10)
macro_data_house.insert(4, 'Emission_CO2', Emission_CO2/10)
macro_data_cs.insert(4, 'Emission_CO2', Emission_CO2/10)

df_stock = pd.concat([epu, cpu, macro_data_stock], axis=1)
df_house = pd.concat([epu, cpu, macro_data_house], axis=1)
df_cs = pd.concat([epu, cpu, macro_data_cs], axis=1)

model_var_stock = sm.tsa.VAR(df_stock)
model_var_house = sm.tsa.VAR(df_house)
model_var_cs = sm.tsa.VAR(df_cs)

results_var_stock = model_var_stock.fit(6)
results_var_house = model_var_house.fit(6)
results_var_cs = model_var_cs.fit(6)

results_var_stock.irf(40).plot(orth = True, impulse = 'cpu_index')
plt.show()
results_var_house.irf(40).plot(orth = True, impulse = 'cpu_index')
plt.show()
results_var_cs.irf(40).plot(orth = True, impulse = 'cpu_index')
plt.show()

# Financial Crisis and Trump year
df_fin = df.drop(macro_data.loc['2007-12-01':'2009-06-01'].index)
df_trump = df.drop(macro_data.loc['2017-02-01':].index)

model_var_fin = sm.tsa.VAR(df_fin)
model_var_trump = sm.tsa.VAR(df_trump)

results_var_fin = model_var_fin.fit(6)
results_var_trump = model_var_trump.fit(6)

results_var_fin.irf(40).plot(orth = True, impulse = 'cpu_index')
plt.show()
results_var_trump.irf(40).plot(orth = True, impulse = 'cpu_index')
plt.show()

results_var.irf(40).plot()

irf = results_var.irf(40).irfs
fig, ax = plt.subplots(2,4)
k = 0
for i in range(2):
    for j in range(4):
        ax[i,j].plot(irf[:,k,1])
        ax[i,j].grid(True)
        ax[i,j].axhline(y=0, color = 'k')
        k = k + 1
fig.show()

def irfplot(irf,c):
    fig, ax = plt.subplots(2,4)
    k = 0
    for i in range(2):
        for j in range(4):
            ax[i,j].plot(irf[:,k,c])
            ax[i,j].grid(True)
            ax[i,j].axhline(y=0, color = 'k')
            k = k + 1
    fig.show()

irfplot(irf,0)

