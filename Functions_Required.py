# Import required libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    plt.tight_layout()
    fig.show()

def girfplot(df_mod, girf_complete, multi_index_col, shock):
    fig, ax = plt.subplots(2,4)
    c = 0
    for i in range(2):
        for j in range(4):
            ax[i,j].plot(girf_complete[multi_index_col[c][0]], color = 'black')
            ax[i,j].plot(girf_complete[multi_index_col[c][1]])
            ax[i,j].plot(girf_complete[multi_index_col[c][2]], color = 'black')
            ax[i,j].grid(True)
            ax[i,j].axhline(y=0, color = 'k')
            ax[i,j].title.set_text(df_mod.columns[shock] + ">" + df_mod.columns[c])
            c += 1
    plt.tight_layout()
    fig.show()

# Subplots
def dataplot(data):
    data.plot(subplots=True, layout=(2,4), figsize = (10,5))
    plt.show()

# RMSE function
def rmse(u):
    u = u.dropna()
    N = u.shape[0] - u.shape[1]
    return (np.sum(u**2)/N)**0.5
# Send it to estimation


# Detrender and inverse detrender class
class transformation_logdiff:
    # The functions below assumes pandas and numpy
    # are already imported
    def __init__(self, dataframe):
        self.init_val = dataframe.iloc[0]
        self.dataframe = dataframe
    def logdiff(self):
        self.data_transformed = np.log(self.dataframe).diff()
        return(self.data_transformed.dropna())
    # Only meant to retrieve the original dataset
    def inv_logdiff(self):
        inv_transformation = np.exp(self.data_transformed.cumsum())*self.init_val
        inv_transformation.iloc[0] = self.init_val
        return(inv_transformation)
    # Meant for GIRF inverse transform
    def inv_logdiff_girf(self, X):
        inv_transformation = X.cumsum()
        return(inv_transformation)
    # Class ends
# Send to the GIRF file

############### Rough Work ###############

# import pandas as pd
# import numpy as np
# from pandas_datareader.data import DataReader
# import matplotlib.pyplot as plt

# # Source: Gavriilidis, K. (2021). Measuring Climate Policy Uncertainty. Available at SSRN: https://ssrn.com/abstract=3847388"
# cpu = pd.read_csv(
#     'https://www.policyuncertainty.com/media/CPU%20index.csv',
#     skiprows=4,
#     usecols=['date', 'cpu_index'], index_col='date'
# )
# cpu.index = pd.to_datetime(cpu.index, format="%b-%y")

# # Source: 'Measuring Economic Policy Uncertainty' by Scott Baker, Nicholas Bloom and Steven J. Davis at www.PolicyUncertainty.com.
# epu = pd.read_excel('https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx', skipfooter=1)
# epu.set_index(
#     pd.to_datetime(
#         epu.Year.astype(str) + '-' + epu.Month.astype(str),
#         format="%Y-%m"
#     ), inplace=True
# )
# epu = epu.drop(['Year', 'Month', 'Three_Component_Index'], axis=1)

# # See Macroeconomic impact of climate change Bilal Kanzig
# # For licencing contact: https://berkeleyearth.org/data/
# temperature = pd.read_fwf(
#     'https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Complete_TAVG_complete.txt',
#     skiprows = 35, header=None, usecols = [0,1,2], names = ['Year','Month','TempAnomaly']
# )

# temperature = temperature.set_index(
#     pd.to_datetime(
#         temperature.Year.astype(str) + '-' + temperature.Month.astype(str),
#         format="%Y-%m"
#     )
# )
# temperature = temperature.drop(['Year', 'Month'], axis=1)
# # temperature.pct_change().plot(); plt.show()

# # Slicing the EPU data, temperature based on CPU dates
# epu = epu.loc[cpu.index[0]:cpu.index[-1]]
# epu.columns = ['epu_index']
# temperature = temperature.loc[cpu.index[0]:cpu.index[-1]]

# # References:
# # 1. Board of Governors of the Federal Reserve System (US), Industrial Production: Total Index [INDPRO], retrieved from FRED,
# # Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/INDPRO, November 19, 2024.
# # 2. U.S. Bureau of Labor Statistics, Unemployment Rate [UNRATE], retrieved from FRED,
# # Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/UNRATE, November 19, 2024.
# # 3. U.S. Bureau of Labor Statistics, Producer Price Index by Commodity: All Commodities [PPIACO], retrieved from FRED,
# # Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PPIACO, November 19, 2024.
# # 4. U.S. Bureau of Economic Analysis, Personal Consumption Expenditures: Chain-type Price Index [PCEPI], retrieved from FRED,
# # Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PCEPI, November 19, 2024.
# # 5. Board of Governors of the Federal Reserve System (US), 3-Month Treasury Bill Secondary Market Rate, Discount Basis [TB3MS],
# # retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/TB3MS, November 19, 2024.

# macro_data = DataReader(
#     ['INDPRO', 'UNRATE', 'PPIACO', 'PCEPI', 'TB3MS'],
#     'fred',
#     start=cpu.index[0],  end=cpu.index[-1]
# )
# macro_data.columns = [
#     'Industrial_Production',
#     'Unemployment_Rate',
#     'PriceIndex_Producer',
#     'PriceIndex_PCE',
#     'Treasurey3Months'
# ]

# # Dataframe for the raw data.
# df = pd.concat([temperature, epu, cpu, macro_data], axis=1)

# # Dataframe for the modified data.
# df_mod = df.copy()
# # df_mod[['TempAnomaly','Industrial_Production','PriceIndex_Producer','PriceIndex_PCE']] = df_mod[[
# #     'TempAnomaly','Industrial_Production','PriceIndex_Producer','PriceIndex_PCE'
# # ]].pct_change()

# df_mod[['TempAnomaly']] = df_mod[['TempAnomaly']].pct_change()
# df_mod[['Industrial_Production','PriceIndex_Producer','PriceIndex_PCE']] = np.log(df_mod[[
#     'Industrial_Production','PriceIndex_Producer','PriceIndex_PCE'
# ]]).diff()

# df_mod = df_mod.rename(columns={
#     'TempAnomaly':'Growth_TempAnomaly',
#     'Industrial_Production':'Growth_Industrial_Production',
#     'PriceIndex_Producer':'Growth_PriceIndex_Producer',
#     'PriceIndex_PCE':'Growth_PriceIndex_PCE'
# })

# df = df.dropna()
# df_mod = df_mod.dropna()

# # Comparison with 'The Macroeconomic Effects of Climate Policy Uncertainty' by Gavriilidis, Kanzig, and Stock (2023)
# # df = df[df.columns[1:]].loc[:'2019-06-01']; df_mod = df_mod[df_mod.columns[1:]].loc[:'2019-06-01']
# # Note that emission is excluded since in their paper they mapped it through the industrial production data, which
# # was the reason that emission always copied the behavior of industrial production. Omitting that variable didn't
# # change anything.
# # Financial Crisis and Trump year
# # df_fin = df.drop(macro_data.loc['2007-12-01':'2009-06-01'].index)
# # df_trump = df.drop(macro_data.loc['2017-02-01':].index)