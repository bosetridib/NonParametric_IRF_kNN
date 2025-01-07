# Import required libraries
import matplotlib.pyplot as plt
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
            ax[i,j].plot(girf_complete[multi_index_col[c][0]])
            ax[i,j].plot(girf_complete[multi_index_col[c][1]])
            ax[i,j].plot(girf_complete[multi_index_col[c][2]])
            ax[i,j].grid(True)
            ax[i,j].axhline(y=0, color = 'k')
            ax[i,j].title.set_text(df_mod.columns[shock] + ">" + df_mod.columns[c])
            c += 1
    plt.tight_layout()
    fig.show()

# Subplots
def dataplot(data):
    data.plot(subplots=True, layout=(2,4))
    plt.show()

# RMSE function
def rmse(u):
    u = u.dropna()
    N = u.shape[0] - u.shape[1]
    return (np.sum(u**2)/N)**0.5
# Send it to estimation

# # Old method
# # Since p=6 in the Gavriilidis, Kanzig, Stock (2023) paper, we select
# # the histories at t-1...t-6;
# p = 6; n_var = y.shape[1]

# y_normalized_plags = sm.tsa.lagmat(y_normalized, maxlag=p, use_pandas=True)
# # and remove the 0's due to lag.
# y_normalized_plags = y_normalized_plags.iloc[p:]


# # Supposing the history of interest is the recent month
# myoi = str(y.index.date[-1])
# omega = y_normalized_plags.iloc[-1]
# X_train = y_normalized_plags.iloc[:-1]

# # Manual way for h=0, 1 to H=40
# from scipy.spatial.distance import euclidean
# dist = np.array([euclidean(i, omega.to_numpy()) for i in X_train.to_numpy()])
# dist = (dist - np.min(dist))/(np.max(dist) - np.min(dist))
# weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
# y_f = np.matmul(y.iloc[p:].drop([myoi]).T, weig).to_frame().T

# for h in range(1,H+1):
#     dist = np.array([euclidean(i, omega.to_numpy()) for i in X_train.iloc[:-h].to_numpy()])
#     dist = (dist - np.min(dist))/(np.max(dist) - np.min(dist))
#     weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
#     y_f.loc[h] = np.matmul(y.iloc[p+h:].drop([myoi]).T, weig).values

# # The forecasts are
# # y_f.plot(subplots = True, layout = (2,4)); plt.show()
# # y_f.cumsum().plot(subplots = True, layout = (2,4)); plt.show()

# girf = pd.DataFrame(delta.reshape(-1,n_var), columns=y_f.columns)
# # Updated history
# omega_star = pd.concat(
#     [
#         (y_f.iloc[0] + delta - y.min())/(y.max() - y.min()),
#         omega.iloc[:-n_var]
#     ],
#     axis=0
# )

# for h in range(1,H+1):
#     dist = np.array([euclidean(i, omega_star.to_numpy()) for i in X_train.iloc[h:].to_numpy()])
#     dist = (dist - np.min(dist))/(np.max(dist) - np.min(dist))
#     weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
#     # weig = (1/dist)/sum(1/dist)
#     girf.loc[h] = np.matmul(y.iloc[p+h:].drop(myoi).T, weig).values
#     girf.loc[h] = girf.loc[h] - y_f.loc[h]

# girf = pd.DataFrame(robust_transformer.inverse_transform(girf), columns=girf.columns)
# dataplot(girf)