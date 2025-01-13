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