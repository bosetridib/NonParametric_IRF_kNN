# Import new and previous libraries, dataframe, variables, model, and IRF function.
from NonParametricIRF_Data import *
from Functions_Required import *
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

##################################################################################
############################# kNN Forecasting & GIRF #############################
##################################################################################

df = pd.concat([epu, cpu, macro_data_mod], axis=1)
# p=2; df = sm.tsa.tsatools.lagmat(df, maxlag=p, use_pandas=True).iloc[p:]

# Retrieve the standardized dataset

y = pd.concat([epu, cpu, macro_data], axis=1)

# Forecasting
# Horizon "in the middle"
H = 40

def histoiOmega(macro_condition):
    if macro_condition == "general":
        histoi = df.iloc[-(H+1):].mean()
        omega = df.iloc[:-(H+1)]
    elif macro_condition == "great_recession":
        histoi = df.loc['2008-11-01':'2009-10-01'].mean()
        omega = pd.concat([df.loc[:'2008-10-01'], df.loc['2009-11-01':]])
    elif macro_condition == "recessionary":
        omega = df.loc[y.loc[y['Unemployment_Rate'] >= 5.5].index]
        histoi = omega.iloc[-1]
    elif macro_condition == "booming":
        omega = df.loc[y.loc[y['Unemployment_Rate'] < 5.5].index]
        histoi = omega.iloc[-1]
    elif macro_condition == "inflationary":
        omega = df.loc[y.loc[df['Growth_PriceIndex_PCE']>0.0025].index]
        histoi = omega.iloc[-1]
    elif macro_condition == "LowCPU":
        omega = df.loc[y.loc[y['cpu_index'] < 100].index]
        histoi = omega.iloc[-1]
    elif macro_condition == "HighCPU":
        omega = df.loc[y.loc[y['cpu_index'] >= 100].index]
        histoi = omega.iloc[-1]
    elif macro_condition == "LowEPU":
        omega = df.loc[y.loc[y['epu_index'] < 100].index]
        histoi = omega.iloc[-1]
    elif macro_condition == "HighEPU":
        omega = df.loc[y.loc[y['epu_index'] >= 100].index]
        histoi = omega.iloc[-1]
    else:
        histoi = df.iloc[-1]
        omega = df.iloc[:-(H+1)]
        print("Default history and omega.")
    return (histoi, omega)

interest = "general."
(histoi, omega) = histoiOmega(interest)

df = df.dropna()
omega = omega.dropna()
omega = omega.loc[:y.index[-1] - pd.DateOffset(months=H)]
omega_mean = omega.mean()
omega_std = omega.std()
omega = (omega - omega_mean)/omega_std
histoi = (histoi - omega_mean)/omega_std
T = omega.shape[0]

knn = NearestNeighbors(n_neighbors=T-H, metric='euclidean')
knn.fit(omega)
dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
# Estimate y_T
y_f = np.matmul(y.loc[omega.iloc[ind].index].T, weig).to_frame().T
df_omega = np.matmul(df.loc[omega.iloc[ind].index].T, weig).to_frame().T

# Define the shock
shock = 1

u_y = y.loc[omega.iloc[ind].index] - y_f.iloc[0].values.squeeze()
u_mean = u_y.mul(weig, axis = 0)
sigma_u = np.matmul((u_y - u_mean).T, (u_y - u_mean).mul(weig, axis = 0)) / (1 - np.sum(weig**2))
delta_y = np.linalg.cholesky(sigma_u)[:,shock]

u = df.loc[omega.iloc[ind].index] - df_omega.values.squeeze()
u_mean = u.mul(weig, axis = 0)
sigma_u = np.matmul((u - u_mean).T, (u - u_mean).mul(weig, axis = 0)) / (1 - np.sum(weig**2))

# Cholesky decomposition
# B_mat = np.linalg.cholesky(sigma_u)
# The desired shock
delta = np.linalg.cholesky(sigma_u)[:,shock]

for h in range(1,H+1):
    y_f.loc[h] = np.matmul(y.loc[omega.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
# dataplot(y_f)

# Estimate y_T_delta
y_f_delta = pd.DataFrame(columns=y_f.columns)
y_f_delta.loc[0] = y_f.loc[0] + delta_y

histoi_delta = (df_omega.iloc[0] + delta - omega_mean.values)/omega_std.values

# histoi_delta = pd.concat([histoi_delta, histoi], axis=0)[:-df.shape[1]]

knn.fit(omega)
dist, ind = knn.kneighbors(histoi_delta.to_numpy().reshape(1,-1))
dist = dist[0,:]; ind = ind[0,:]
weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

for h in range(1,H+1):
    y_f_delta.loc[h] = np.matmul(y.loc[omega.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
# dataplot(y_f_delta)

girf = y_f_delta - y_f
dataplot(girf*(50/delta[shock]))
# dataplot(girf)

# y_f.plot(
#     subplots=True, layout=(2,4), color = 'blue',
#     ax=y_f_delta.plot(
#         subplots=True, layout=(2,4), color = 'red'
#     )
# )
# plt.show()

# Confidence Intervals
R=500
sim_girf = []

# Perform simulations
for r in range(0,R):
    omega_resamp = omega.sample(n=T, replace=True).sort_index()
    # Estimate y_T
    knn.fit(omega_resamp)
    dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    # Estimate y_T
    y_f_resamp = np.matmul(y.loc[omega_resamp.iloc[ind].index].T, weig).to_frame().T
    for h in range(1,H+1):
        y_f_resamp.loc[h] = np.matmul(y.loc[omega_resamp.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
    
    y_f_delta_resamp = pd.DataFrame(columns=y_f_resamp.columns)
    y_f_delta_resamp.loc[0] = y_f_resamp.loc[0] + delta

    df_star = pd.concat([y.loc[omega_resamp.iloc[-3:-1].index], y_f_delta_resamp])
    df_star[['Industrial_Production','PriceIndex_Producer','PriceIndex_PCE', 'Emission_CO2']] = np.log(df_star[[
        'Industrial_Production','PriceIndex_Producer','PriceIndex_PCE','Emission_CO2'
    ]]).diff().dropna()
    df_star = (df_star - y.mean())/y.std()
    histoi_delta = df_star.iloc[-1]
    # histoi_delta = (y_f_delta.loc[0] - y.mean())/y.std()
    histoi_delta = pd.concat([histoi_delta, histoi], axis=0)[:-df.shape[1]]
    knn.fit(omega_resamp)
    dist, ind = knn.kneighbors(histoi_delta.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
    for h in range(1,H+1):
        y_f_delta_resamp.loc[h] = np.matmul(y.loc[omega_resamp.iloc[ind].index + pd.DateOffset(months=h)].T, weig).values
    # dataplot(y_f_delta)
    sim_girf.append(y_f_delta_resamp - y_f_resamp)
    print('loop: '+str(r))
# End of loop, and now the sim_list_df has each of the resampled dataframes

conf = 0.90
# Define the multi-index dataframe for each horizon and CI for each column
girf_complete = pd.DataFrame(
    columns = y.columns,
    index = pd.MultiIndex(
        levels=[range(0,H+1),['lower','GIRF','upper']],
        codes=[[x//3 for x in range(0,(H+1)*3)],[0,1,2]*(H+1)], names=('Horizon', 'CI')
    )
)

for h in range(0,H+1):
    for col in y.columns:
        girf_complete[col][h,'lower'] = 2*girf[col][h] - np.quantile([each_df[col][h] for each_df in sim_girf], conf+(1-conf)/2)
        girf_complete[col][h,'GIRF'] = girf[col][h]
        girf_complete[col][h,'upper'] = 2*girf[col][h] - np.quantile([each_df[col][h] for each_df in sim_girf], (1-conf)/2)
# girf_complete
girf_complete = girf_complete.astype('float')

girf_complete = girf_complete.unstack()
multi_index_col = [(girf_complete.columns[i], girf_complete.columns[i+1], girf_complete.columns[i+2]) for i in range(0,24,3)]
# Plot
girfplot(df, girf_complete*(50/delta[shock]), multi_index_col, shock)
#
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.figure(figsize = (10,25))
gs1 = gridspec.GridSpec(2, 4)
gs1.update(wspace=0.025, hspace=0.2) # set the spacing between axes.

c=0
for i in range(8):
    ax1 = plt.subplot(gs1[i])
    # plt.axis('on')
    ax1.plot(girf_complete[multi_index_col[c][1]]*(50/delta[shock]))
    ax1.set_title(y.columns[c])
    ax1.tick_params(axis="y",direction="in", pad=-20)
    c += 1
plt.tight_layout()
plt.show()

girf_complete = girf_complete*(50/delta[shock])
c=0
plt.figure(figsize = (10,25))
gs1 = gridspec.GridSpec(2, 4)
gs1.update(wspace=0.025, hspace=0.2) # set the spacing between axes. 
c=0
for i in range(8):
    ax1 = plt.subplot(gs1[i])
    # plt.axis('on')
    ax1.plot(girf_complete[multi_index_col[c][0]], color = 'black')
    ax1.plot(girf_complete[multi_index_col[c][1]])
    ax1.plot(girf_complete[multi_index_col[c][2]], color = 'black')
    ax1.title.set_text(y.columns[shock] + ">" + y.columns[c])
    ax1.tick_params(axis="y",direction="in", pad=-20)
    c += 1
plt.tight_layout()
plt.show()