# Import new and previous libraries, dataframe, variables, model, and IRF function.
# from NonParametricIRF_Data import *
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
from sklearn.neighbors import NearestNeighbors
from scipy.stats import invwishart, invgamma
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Function to generate random walk
def random_walk(T, scl):
    #e_t = np.random.normal(size=T, scale=scl)
    e_t = np.random.normal(size=T, scale=scl)
    rw_t = np.zeros(T)
    rw_t[0] = e_t[0]
    for i in range(1,T):
        rw_t[i] = rw_t[i-1] + e_t[i]
    return rw_t

# Function to generate Y_tvp
def tvp_simulate(n_obs = 200, n_var = 2, n_lags = 2, intercept = 1):
    # number of observations, variables, and lags

    # Define Y_TVP: we also initialize it with the minimum number
    # of observations required ( = number of lags) to begin generating
    # the further observations. We slice them out in the end.
    y_sim_tvp = pd.DataFrame(
        data=[np.random.normal(size=n_var) for _ in range(n_lags)],
        columns=['y'+str(c_num) for c_num in range(1,n_var+1)]
    )
    # Define variances of B_t and A_t
    var_alpha = 1
    var_beta = 1
    c = 4
    # Define epsilon - the vector of error terms.
    epsilon_sim = np.array([np.random.normal(size=n_obs) for _ in range(n_var)])

    # Define B matrix for all coefficients of B_t (slope coefficients)
    # and c_t (intercept). Note that the total number of elements
    # would be n_var*(1 + (n_var*n_lags)).
    # B_mat = np.array([random_walk(n_obs, (0.05/100)**0.5) for _ in range(n_var*(1 + (n_var*n_lags)))])
    B_mat = np.array([np.zeros(n_obs) for _ in range(n_var*(1 + (n_var*n_lags)))])
    # Fill the B_mat_0 with some initial values
    B_mat[:,0] = np.random.randn(n_var*(1 + (n_var*n_lags)))*(c/n_obs)
    # Also, check the stability condition by forming the companion matrix at t=0
    # stuck counter: if the eigenvalues do not satisfy the stability condition
    # for too long, we return 'stuck' to avoid infinite loop.
    stuck = 0
    # No need to understand the following if loop, just see the subsequent codes.
    if max(
        abs(np.linalg.eigvals(np.concatenate(
            (B_mat[:,0].reshape(n_var,(n_var*n_lags)+1)[:,1:],
             np.concatenate((
                 np.eye(n_var*(n_lags-1)),
                 np.zeros((n_var*(n_lags-1),n_var))
             ), axis = 1)), axis = 0
        )))
    ) > 0.99 : return 'stuck'
    # For the subsequent periods, we confirm that the eienvalues of B's are less than 1
    # var_beta = 0.05/100
    t = 1
    while t < n_obs:
        B_mat[:,t] = B_mat[:,t-1] + (np.random.randn(n_var*(1 + (n_var*n_lags))))*(c/n_obs)
        # Check the stability condition by forming the companion matrix
        Phi_mat = B_mat[:,t].reshape(n_var,(n_var*n_lags)+1)
        # remove intercept
        Phi_mat = Phi_mat[:,1:]
        # Companion matrix
        comp_mat = np.concatenate((np.eye(n_var*(n_lags-1)),np.zeros((n_var*(n_lags-1),n_var))), axis = 1)
        comp_mat = np.concatenate((Phi_mat, comp_mat), axis = 0)
        # Check eigenvalues
        if max(abs(np.linalg.eigvals(comp_mat))) > 0.99:
            t -= 1
            stuck += 1
        t += 1
        if stuck > 10000: return 'stuck'
    # Define alpha matrix for A_t matrix (shocks)
    # var_alpha = 0.5/100
    alpha_t = np.array([random_walk(n_obs, var_alpha**0.5) for _ in range(np.int64((n_var*(n_var - 1))/2))])*(c/n_obs)

    for t in range(n_obs):
        # We form the matrices/vectors at time t.
        # For X_t' matrix, we first collect the
        # vector [y_t-1, ..., y_t-p] for the vector
        # [1, y_t-1, ..., y_t-p] for p being the
        # number of lags (n_lags).
        y_t_lags = y_sim_tvp.iloc[::-1].iloc[:n_lags]
        # Define X_t' matrix using the Kronecker product.
        X_t = np.kron(
            np.identity(n_var),
            np.append(intercept,y_t_lags)
        )
        # Define B_t matrix at time t
        B_t = B_mat[:,t]
        # Define the shock matrix A_t at time t
        A_t = np.eye(n_var)
        A_t[np.tril_indices(n_var, k=-1)] = alpha_t[:,t]
        # We add the observation at time t
        y_sim_tvp.loc[n_lags + t] = np.matmul(X_t, B_t.T) + np.matmul(np.linalg.inv(A_t), epsilon_sim[:,t])
    # Slice the initialized values out
    y_sim_tvp = y_sim_tvp.iloc[n_lags:].reset_index(drop=True)
    y_sim_tvp.index += 1
    # Check the stability condition
    # Return a dictionary of all elements
    return {
        'data': y_sim_tvp,
        'B_mat': B_mat,
        'alpha_t': alpha_t,
        'n_obs': n_obs,
        'n_var': n_var,
        'n_lags': n_lags
    }

# Following class and estimations are based on statsmodels and  Nakajima (2011)
class TVPVAR_beta(sm.tsa.statespace.MLEModel):
    # Steps 2-3 are best done in the class "constructor", i.e. the __init__ method
    def __init__(self, y, lags=1):
        # Create a matrix with [y_t' : y_{t-1}'] for t = 2, ..., T
        augmented = sm.tsa.lagmat(y, maxlag= lags, trim='both', original='in', use_pandas=True)
        # Separate into y_t and z_t = [1 : y_{t-1}']
        p = y.shape[1]
        y_t = augmented.iloc[:, :p]
        z_t = sm.add_constant(augmented.iloc[:, p:])

        # Recall that the length of the state vector is p * (p + 1)
        k_states = p * (1 + p*lags)
        super().__init__(y_t, exog=z_t, k_states=k_states)

        # Note that the state space system matrices default to contain zeros,
        # so we don't need to explicitly set c_t = d_t = 0.

        # Construct the design matrix Z_t
        # Notes:
        # -> self.k_endog = p is the dimension of the observed vector
        # -> self.k_states = p + (p**2)*self.lags is the dimension of the observed vector
        # -> self.nobs = T is the number of observations in y_t
        self['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        for i in range(self.k_endog):
            start = i * (self.k_endog * lags + 1)
            end = start + self.k_endog * lags + 1
            self['design', i, start:end, :] = z_t.T

        # Construct the transition matrix T = I
        self['transition'] = np.eye(k_states)

        # Construct the selection matrix R = I
        self['selection'] = np.eye(k_states)

        self['obs_cov'] = np.zeros((p, p, self.nobs))

        # Step 3: Initialize the state vector as alpha_1 ~ N(0, 5I)
        self.ssm.initialize('known', stationary_cov= np.eye(k_states))

    # Step 4. Create a method that we can call to update H and Q
    def update_variances(self, a_t, state_cov_diag):
        T = self.nobs
        p = self.endog.shape[1]
        
        for t in range(T):
            # 1. Reconstruct A_t: Lower triangular with unit diagonal (Nakajima p. 125)
            At = np.eye(p)
            # Example for a 3-variable system: filling off-diagonals a_21, a_31, a_32
            # At[5] = a_t[0, t]
            # At[6] = a_t[1, t]
            # At[5, 6] = a_t[2, t]
            # (Generic logic to map a_t vector to lower triangle goes here)
            At[np.tril_indices(p, k=-1)] = a_t[:, t]
            # 2. Reconstruct Sigma_t: Diagonal matrix of exp(h_t) (Nakajima eq 8)
            Sigmat = np.eye(p)
            # 3. Compute Reduced-form H_t = inv(At) @ Sigmat @ inv(At).T
            InvAt = np.linalg.inv(At)
            self['obs_cov', :, :, t] = InvAt @ Sigmat @ InvAt.T

        # self['obs_cov'] = obs_cov   # H matrix
        self['state_cov'] = state_cov_diag

    # Finally, it can be convenient to define human-readable names for
    # each element of the state vector. These will be available in output
    @property
    def state_names(self):
        state_names = np.empty((self.k_endog, self.k_endog + 1), dtype=object)
        for i in range(self.k_endog):
            endog_name = self.endog_names[i]
            state_names[i] = (
                ['intercept.%s' % endog_name] +
                ['L1.%s->%s' % (other_name, endog_name) for other_name in self.endog_names])
        return state_names.ravel().tolist()

class TVPVAR_alpha(sm.tsa.statespace.MLEModel):
    def __init__(self, residuals):
        # p is the number of variables in the VAR
        p = residuals.shape[1]
        # Number of off-diagonal elements in the lower-triangular matrix A_t
        k_a = p * (p - 1) // 2

        super().__init__(residuals, k_states=k_a)
        
        # 1. Transition equation: Random walk for a_t (Nakajima eq 8)
        self['transition'] = np.eye(k_a)
        self['selection'] = np.eye(k_a)
        
        # 2. Design matrix Z_t: Built from residuals as shown in Nakajima p.126
        # This matrix links the residuals to the structural states a_t
        self['design'] = np.zeros((p, self.k_states, self.nobs))
        self.update_design(residuals)

        self.ssm.initialize('known', stationary_cov= np.eye(k_a))

    def update_design(self, residuals):
        """Fills the Z_t matrix with negative residuals to reflect recursive form."""
        p = self.endog.shape[1]
        for t in range(self.nobs):
            z_t = np.zeros((p, self.k_states))
            idx = 0
            for i in range(1, p):
                # Equation i depends on residuals of variables 0 to i-1
                z_t[i, idx:idx+i] = -residuals[t, 0:i]
                idx += i
            self['design', :, :, t] = z_t

    def update_system(self, H_a, Q_a):
        """Updates structural variances (Sigma_t) and state innovation covariance."""
        # Observation covariance H_t is diagonal: diag(exp(h_t))
        self['obs_cov'] = np.diag(H_a)
        # State covariance for the random walk of a_t
        self['state_cov'] = np.diag(Q_a)

def tvp_bayesian_estimation(sim_elements, niter = 11000, nburn = 1000):
    mod = TVPVAR_beta(sim_elements['data'], sim_elements['n_lags'])
    mod_a = TVPVAR_alpha(np.zeros_like(mod.endog))
    # initial_obs_cov = np.cov(sim_T['data'])
    alpha_t_init = np.array([random_walk(mod.nobs, 0.01**0.5) for _ in range(mod_a.k_states)])

    initial_state_cov_diag = 0.01 * np.eye(mod.k_states)
    # Update H and Q
    # mod.update_variances(alpha_t_init, initial_state_cov_diag)

    # Perform Kalman filtering and smoothing
    # (the [] is just an empty list that in some models might contain
    # additional parameters. Here, we don't have any additional parameters
    # so we just pass an empty list)
    # initial_res = mod.smooth([])

    # Gibbs sampler setup
    niter = niter
    nburn = nburn

    # 1. Create storage arrays
    store_states = np.zeros((niter + 1, mod.nobs, mod.k_states))
    store_obs_cov = np.zeros((niter + 1, mod_a.k_states,  mod.nobs))
    store_state_cov = np.zeros((niter + 1, mod.k_states, mod.k_states))

    store_states_a = np.zeros((niter + 1, mod_a.nobs, mod_a.k_states))
    store_obs_cov_a = np.zeros((niter + 1, mod_a.k_endog))
    store_state_cov_a = np.zeros((niter + 1, mod_a.k_states))

    # 2. Put in the initial values
    store_obs_cov[0] = alpha_t_init
    store_state_cov[0] = initial_state_cov_diag
    # mod.update_variances(store_obs_cov[0], store_state_cov[0])
    store_obs_cov_a[0] = mod_a.k_endog*[0.01]
    store_state_cov_a[0] = mod_a.k_states*[0.01]

    # 3. Construct posterior samplers
    sim = mod.simulation_smoother(method='kfs')
    sim_a = mod_a.simulation_smoother(method='kfs')

    # 2. Define priors (Nakajima p. 132 suggests v0=4, S0=0.01 for diffuse prior)
    v0_a = 4.0
    S0_a = 0.01
    # --- 1. SET PRIOR VALUES (Based on Nakajima) ---
    # Innovation Variances (Tight for alpha, Diffuse for a)
    v0_alpha, S0_alpha = 20.0, 0.01
    v0_a, S0_a = 4.0, 0.01
    # Constant Structural Variances (Diffuse prior for Sigma diagonal)
    # v0_sigma=0.01, S0_sigma=0.01 represents a very flat/diffuse prior
    v0_sigma, S0_sigma = 0.01, 0.01

    for i in range(niter):
        mod.update_variances(store_obs_cov[i], store_state_cov[i])
        # Sample Beta
        sim.simulate()
        store_states[i + 1] = sim.simulated_state.T
        fitted = np.matmul(mod['design'].transpose(2, 0, 1), store_states[i + 1][..., None])[..., 0]
        resid = mod.endog - fitted
        # np.diff computes alpha_{t+1} - alpha_t
        innovations = np.diff(store_states[i + 1], axis=0) # Result shape: (k_states, T-1)
        # 2. Compute the sum of squared innovations (Sum of outer products)
        # This corresponds to Nakajima eq. 5 (Scale matrix update)
        sum_sq_innovations = innovations.T @ innovations
        # 3. Define posterior parameters (Nakajima p. 125, eq. 5)
        # prior_nu and prior_S are your chosen hyperparameters (e.g., T+3 and Identity)
        post_df = 15 + (mod.nobs - 1)
        post_scale = 0.01 * np.eye(mod.k_states) + sum_sq_innovations
        # 4. Draw the new SIGMA_beta (state_cov) matrix
        # This generates a (k_states x k_states) matrix
        # current_Q_alpha = invwishart.rvs(df=post_df, scale=post_scale)
        # 5. Update the model instance for the next iteration
        # This ensures the KFS smoother uses the updated state_cov
        # Sample Sigma_Beta
        store_state_cov[i + 1] = invwishart.rvs(df=post_df, scale=post_scale)
        # Sample alpha
        mod_a.update_design(resid)
        mod_a.update_system(store_obs_cov_a[i], store_state_cov_a[i])
        sim_a.simulate()
        store_states_a[i + 1] = sim_a.simulated_state.T
        # Sample Sigma_alpha
        # 1. Calculate innovations for structural relations
        # current_a shape: (k_a, T) where k_a = p*(p-1)/2
        diff_a = np.diff(store_states_a[i + 1], axis=0) # Shape: (k_a, T-1)
        # 3. Sample each diagonal element from Inverse-Gamma
        # Nakajima Step 4 (p. 135) / Step 6 (p. 127) logic
        current_sigma_a_diag = np.zeros(mod_a.k_states)
        for j in range(mod_a.k_states):
            # Sum of squared differences for the j-th structural relation
            ss_a_j = np.sum(diff_a[j, :]**2)
            # Calculate posterior parameters
            post_v_a = v0_a + (mod_a.nobs - 1) / 2.0
            post_S_a = (S0_a + ss_a_j) / 2.0
            # Sample from IG
            current_sigma_a_diag[j] = invgamma.rvs(post_v_a, scale=post_S_a)
        # 4. Update the structural relations model instance
        store_state_cov_a[i + 1] = current_sigma_a_diag
        store_obs_cov[i + 1] = store_states_a[i + 1].T
        # --- 2. INITIALIZE STRUCTURAL IMPACT MODEL (mod_a) ---
        # T: number of observations, p: number of endog variables
        # k_a = p * (p - 1) // 2
        # Initialize the diagonal obs_cov with an identity matrix or sample covariance
        current_sigma_diag = np.zeros(mod_a.k_endog)
        # --- 3. SAMPLING THE DIAGONAL OBS_COV IN MCMC LOOP ---
        # (Step A: Update and sample alpha_t and a_t as shown in previous turns)
        # STEP B: Sample the constant diagonal obs_cov (Sigma)
        # 1. Calculate structural shocks: e_t = A_t * residuals_t
        # structural_shocks = calculate_structural_shocks(current_residuals, current_a)
        structural_shocks = np.zeros_like(resid)
        for _ in range(mod_a.nobs):
            # 1. Reconstruct A_t: Lower triangular with unit diagonal (Nakajima p. 125)
            At = np.eye(mod_a.k_endog)
            # (Generic logic to map a_t vector to lower triangle goes here)
            At[np.tril_indices(mod_a.k_endog, k=-1)] = store_states_a[i + 1][_, :]
            # 2. Reconstruct Sigma_t: Diagonal matrix of exp(h_t) (Nakajima eq 8)
            Sigmat = np.eye(mod_a.k_endog)
            # 3. Compute Reduced-form H_t = inv(At) @ Sigmat @ inv(At).T
            structural_shocks[_] = At @ resid[_, :]
        # 2. Draw each diagonal element sigma_i^2 from Inverse-Gamma
        for j in range(mod_a.k_endog):
            # Sum of squared structural shocks for variable j
            ss_e_j = np.sum(structural_shocks[:, j]**2)
            # Calculate posterior parameters (Nakajima Section III.C logic)
            post_v_sigma = (v0_sigma + mod_a.nobs) / 2.0
            post_S_sigma = (S0_sigma + ss_e_j) / 2.0
            current_sigma_diag[j] = invgamma.rvs(post_v_sigma, scale=post_S_sigma)
        store_obs_cov_a[i + 1] = current_sigma_diag
    # Return the results as a dictionary similar to the simulate
    return {
        'data' : sim_elements['data'],
        'B_mat': np.mean(store_states[nburn + 1:], axis=0).T,
        'alpha_t': np.mean(store_states_a[nburn + 1:], axis=0).T,
        'B_mat_complete': store_states[nburn + 1:],
        'alpha_t_complete': store_states_a[nburn + 1:],
        'n_obs': mod.nobs,
        'n_var': mod.k_endog,
        'n_lags': sim_elements['n_lags'],
        'iterations' : niter - nburn - 1
        }

def tvp_irf(sim_elements, t=20, impulse = 0):
    if sim_elements == 'stuck':
        return 'stuck'
    if t <= 0:
        return 'Value of t should be greater than 0'
    # Collect the basic variables.
    n_var = sim_elements['n_var']
    n_lags = sim_elements['n_lags']
    # Collect the alpha matrix to capture the shocks
    alpha_t = sim_elements['alpha_t']
    # Collect the coefficient matrices to capture the shocks
    B_mat = sim_elements['B_mat']

    # Fix the lower triangular matrix
    A_t = np.eye(n_var)
    A_t[np.tril_indices(n_var, k=-1)] = alpha_t[:,t-1]

    # Collect the coefficient matrices to form the companion matrix
    Phi_mat = B_mat[:,t-1]
    Phi_mat = Phi_mat.reshape(n_var,(n_var*n_lags)+1)
    Phi_mat = Phi_mat[:,1:]

    # Companion matrix
    comp_mat = np.concatenate((np.eye(n_var*(n_lags-1)),np.zeros((n_var*(n_lags-1),n_var))), axis = 1)
    comp_mat = np.concatenate((Phi_mat, comp_mat), axis = 0)
    
    J = np.concatenate((np.eye(n_var),np.zeros((n_var,n_var*(n_lags-1)))), axis = 1)

    Phi_i = [np.matmul(np.matmul(J,np.linalg.matrix_power(comp_mat,_)), J.T) for _ in range(0,11)]
    Theta = [np.matmul(_,np.linalg.inv(A_t)) for _ in Phi_i]

    irf_tvp = pd.DataFrame([_[:,impulse] for _ in Theta], columns=sim_elements['data'].columns)

    return irf_tvp
# sim_elements = tvp_simulate(200, 3, 2); impulse=0

def tvp_irf_bayesian(tvp_estimates, t=20, impulse=0):
    H=10
    sim_tvp_irf = [tvp_irf({
        'data' : tvp_estimates['data'],
        'B_mat': tvp_estimates['B_mat_complete'][_].T,
        'alpha_t': tvp_estimates['alpha_t_complete'][_].T,
        'n_var': tvp_estimates['n_var'],
        'n_lags': tvp_estimates['n_lags']
    }, t=t-1, impulse=impulse) for _ in range(tvp_estimates['iterations'])]
    tvp_irf_complete = pd.DataFrame(
        columns = tvp_estimates['data'].columns,
        index = pd.MultiIndex(
            levels=[range(0,H+1),['lower','TVP_IRF','upper']],
            codes=[[x//3 for x in range(0,(H+1)*3)],[0,1,2]*(H+1)], names=('Horizon', 'CI')
        )
    )
    tvp_irf_complete = tvp_irf_complete.unstack()
    for col in range(tvp_estimates['data'].shape[1]):
        tvp_irf_complete[(tvp_estimates['data'].columns[col],'lower')] = np.quantile(sim_tvp_irf, 0.05, axis=0)[:,col]
        tvp_irf_complete[(tvp_estimates['data'].columns[col],'TVP_IRF')] = np.quantile(sim_tvp_irf, 0.5, axis=0)[:,col]
        tvp_irf_complete[(tvp_estimates['data'].columns[col],'upper')] = np.quantile(sim_tvp_irf, 0.95, axis=0)[:,col]
    # tvp_irf_complete
    tvp_irf_complete = tvp_irf_complete.astype('float')

    return tvp_irf_complete

def knn_irf(sim_elements, t=20, impulse=0):
    if sim_elements == 'stuck':
        return 'stuck'
    if t <= 0:
        return 'Value of t should be greater than 0'
    delta_y = sim_elements['data'].copy()
    n_lags = sim_elements['n_lags']
    n_var = sim_elements['n_var']
    omega = pd.concat([delta_y, sm.tsa.tsatools.lagmat(delta_y, maxlag=n_lags, use_pandas=True).iloc[n_lags:]], axis = 1)
    omega = omega.dropna()

    omega_mean = omega.mean()
    omega_std = omega.std()
    omega_scaled = (omega - omega_mean)/omega_std
    
    # We subset the data to the last t observations, and we keep the last observation as the history to find the neighbors.
    omega_scaled = omega_scaled.iloc[:t]
    histoi = omega_scaled.iloc[-1]
    omega_scaled = omega_scaled.iloc[:-11]
    # histoi = (histoi - omega_mean)/omega_std
    T = omega_scaled.shape[0]

    knn = NearestNeighbors(n_neighbors=T, metric='euclidean')
    knn.fit(omega_scaled)
    dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

    # Estimate y_T
    y_f = np.matmul(delta_y.loc[omega_scaled.iloc[ind].index].T, weig).to_frame().T
    # y_f = np.matmul(y.loc[omega_scaled.iloc[ind].index].T, weig).to_frame().T
    for h in range(1,10+1):
        y_f.loc[h] = np.matmul(delta_y.loc[omega_scaled.iloc[ind].index + h].T, weig).values
    # dataplot(y_f)
    u = delta_y - y_f.loc[0].values.squeeze()
    u = u.iloc[:T]
    # u_mean = u.mul(weig, axis = 0)
    ################IMPORTANT CORRECTION HERE##################
    u_mean = u.mean()
    sigma_u = np.matmul((u - u_mean).T, (u - u_mean).mul(weig, axis = 0)) / (1 - np.sum(weig**2))
    # Cholesky decomposition
    B_mat = np.transpose(np.linalg.cholesky(sigma_u))
    # The desired shock
    # B_mat = np.transpose(np.linalg.cholesky(u.cov()*((T-1)/(T-8-1))))
    delta = B_mat[impulse]

    # Estimate y_T_delta
    y_f_delta = pd.DataFrame(columns=y_f.columns)
    y_f_delta.loc[0] = y_f.loc[0] + delta

    histoi_delta = (y_f.iloc[0] + delta - omega_mean[:n_var])/omega_std[:n_var]
    histoi_delta = pd.concat([histoi_delta, histoi], axis=0)[:-n_var]

    dist, ind = knn.kneighbors(histoi_delta.to_numpy().reshape(1,-1))
    dist = dist[0,:]; ind = ind[0,:]
    weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))

    for h in range(1,10+1):
        y_f_delta.loc[h] = np.matmul(delta_y.loc[omega_scaled.iloc[ind].index + h].T, weig).values
    # dataplot(y_f_delta)

    girf = y_f_delta - y_f

    # Confidence intervals
    R=50
    sim_girf = []

    # Perform bootstrap resampling
    H = 10
    for r in range(0,R):
        omega_scaled_resamp = omega_scaled.sample(n=T, replace=True).sort_index()
        #omega_scaled_resamp_mean = delta_y.loc[omega_scaled_resamp.index].mean()
        #omega_scaled_resamp_sd = delta_y.loc[omega_scaled_resamp.index].std()
        # Estimate y_T
        knn.fit(omega_scaled_resamp)
        dist, ind = knn.kneighbors(histoi.to_numpy().reshape(1,-1))
        dist = dist[0,:]; ind = ind[0,:]
        weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
        # Estimate y_T
        y_f_resamp = np.matmul(delta_y.loc[omega_scaled_resamp.iloc[ind].index].T, weig).to_frame().T
        for h in range(1,H+1):
            y_f_resamp.loc[h] = np.matmul(delta_y.loc[omega_scaled_resamp.iloc[ind].index + h].T, weig).values

        y_f_delta_resamp = pd.DataFrame(columns=y_f_resamp.columns)
        y_f_delta_resamp.loc[0] = y_f_resamp.loc[0] + delta

        histoi_delta_resamp = (y_f_resamp.loc[0] + delta - omega_mean[:n_var])/omega_std[:n_var]
        histoi_delta_resamp = pd.concat([histoi_delta_resamp.squeeze(), histoi.T[:-delta_y.shape[1]].squeeze()], axis=0)
        dist, ind = knn.kneighbors(histoi_delta_resamp.to_numpy().reshape(1,-1))
        dist = dist[0,:]; ind = ind[0,:]
        weig = np.exp(-dist**2)/np.sum(np.exp(-dist**2))
        for h in range(1,H+1):
            y_f_delta_resamp.loc[h] = np.matmul(delta_y.loc[omega_scaled_resamp.iloc[ind].index + h].T, weig).values
        
        girf_resamp = y_f_delta_resamp - y_f_resamp
        sim_girf.append(girf_resamp)
    # End of loop, and now the sim_list_delta_y has each of the resampled dataframes

    girf_complete = pd.DataFrame(
        columns = y_f.columns,
        index = pd.MultiIndex(
            levels=[range(0,H+1),['lower','GIRF','upper']],
            codes=[[x//3 for x in range(0,(H+1)*3)],[0,1,2]*(H+1)], names=('Horizon', 'CI')
        )
    )
    girf_complete = girf_complete.unstack()
    for col in y_f.columns:
        girf_complete[(col,'lower')] = [np.quantile([each_delta_y[col][h] for each_delta_y in sim_girf], 0.05) for h in range(0,H+1)]
        girf_complete[(col,'GIRF')] = [np.quantile([each_delta_y[col][h] for each_delta_y in sim_girf], 0.5) for h in range(0,H+1)]
        girf_complete[(col,'upper')] = [np.quantile([each_delta_y[col][h] for each_delta_y in sim_girf], 0.95) for h in range(0,H+1)]
    # girf_complete
    girf_complete = girf_complete.astype('float')

    return {'girf': girf, 'girf_complete': girf_complete}

sim_T = tvp_simulate(200, 3, 1)
tvp_irf(sim_T)
tvp_est = tvp_bayesian_estimation(sim_T, niter=1100, nburn=100)
tvp_irf_bayesian(tvp_est, t=80, impulse=0)


sim_T['B_mat']
tvp_est['B_mat']
tvp_irf(tvp_est, impulse=0, t=1)
knn_irf(sim_T)['girf']
tvp_irf(sim_T) - knn_irf(sim_T)['girf']
kirf = knn_irf(sim_T)
girf_lower = pd.DataFrame([kirf['girf_complete'].loc[_,'lower'] for _ in range(0,11)]).reset_index(drop=True).T
girf = pd.DataFrame([kirf['girf_complete'].loc[_,'GIRF'] for _ in range(0,11)]).reset_index(drop=True).T
girf_upper = pd.DataFrame([kirf['girf_complete'].loc[_,'upper'] for _ in range(0,11)]).reset_index(drop=True).T

knn_irf(sim_T)['girf_complete'].loc[1,'GIRF']


# Begin simulations

n_sim = 25
bias = []

for n_obs in [_*200 for _ in range(1,6)]:
    n_var = 3; n_lags = 1
    counter = 0
    while counter < n_sim:
        sim = tvp_simulate(n_obs, n_var, n_lags, intercept=1)
        # TVP-VAR Bayesian estimation
        tvp_est_sim = tvp_bayesian_estimation(sim, niter=1100, nburn=100)

        if sim == 'stuck':
            counter -= 1
        else:
            try:
                for t in range(20, n_obs - 20, 20):
                    # KNN IRF estimation
                    knn_irf_sim = knn_irf(sim, t)
                    girf_lwr = knn_irf_sim['girf_complete'].loc[:,(slice(None),'lower')].T.set_index(sim['data'].columns)
                    girf = knn_irf_sim['girf_complete'].loc[:,(slice(None),'GIRF')].T.set_index(sim['data'].columns)
                    girf_upr = knn_irf_sim['girf_complete'].loc[:,(slice(None),'upper')].T.set_index(sim['data'].columns)
                    # TVP-VAR IRF estimation
                    tvp_irf_sim = tvp_irf_bayesian(tvp_est_sim, t)
                    tvp_irf_lwr = tvp_irf_sim.loc[:,(slice(None),'lower')].T.set_index(sim['data'].columns)
                    tvp_irf_mid = tvp_irf_sim.loc[:,(slice(None),'TVP_IRF')].T.set_index(sim['data'].columns)
                    tvp_irf_upr = tvp_irf_sim.loc[:,(slice(None),'upper')].T.set_index(sim['data'].columns)
                    bias.append({
                        'bias_knn' : girf - tvp_irf(sim).T,
                        'bias_tvp' : tvp_irf_mid - tvp_irf(sim).T,
                        'tvp_irf' : tvp_irf(sim).T,
                        'T': n_obs, 't' : t,
                        'ci_l' : girf_lwr, 'ci_u' : girf_upr,
                        'tvp_ci_l' : tvp_irf_lwr, 'tvp_ci_u' : tvp_irf_upr
                    })
                    # There are NaN values generated sometimes randomly, so we remove those simulations
                    if bias[-1]['bias_knn'].isnull().values.any() or bias[-1]['bias_tvp'].isnull().values.any():
                        bias.pop()
                        print('pop')
                        counter -= 1
                    # Print the progress
                    print(str(n_obs) + ',' + str(t) + ',' + str(counter))
            except:
                counter -= 1
                print('error')
                pass
        # print(str(n_obs) + ',' + str(t) + ',' + str(counter))
        counter += 1
#End

n_obs = 200; n_var = 3; n_lags = 1; intercept = 1


import pickle
# Saving objects:
with open('objs.pkl', 'wb') as f:
    pickle.dump(bias, f)


# Getting back the objects:
import pickle
with open('objs.pkl', 'rb') as f:
    bias = pickle.load(f)


bias_avg = [np.absolute(_['bias']).mean() for _ in bias]
bias_avg = sum(bias_avg)/len(bias_avg)
bias_avg_tvp = [np.absolute(_['bias_tvp']).mean() for _ in bias]
bias_avg_tvp = sum(bias_avg_tvp)/len(bias_avg_tvp)

bias_avg.plot(); plt.show()
bias_avg_tvp.plot(); plt.show()

rmse_avg = [np.absolute(_['bias']).mean()**2 for _ in bias]
rmse_avg = ((sum(rmse_avg)/len(rmse_avg)))**0.5
rmse_avg_tvp = [np.absolute(_['bias_tvp']).mean()**2 for _ in bias]
rmse_avg_tvp = ((sum(rmse_avg_tvp)/len(rmse_avg_tvp)))**0.5

rmse_avg.plot(); plt.show()
rmse_avg_tvp.plot(); plt.show()

# n_var: 2,3,4: n_obs: 200,400,600,800,1000: n_lags:1,2,3,4
# Separate bias list by n_var
bias_avg_var = [[_ for _ in bias if _['k'] == count] for count in range(2,5)]
bias_avg_var = [[np.absolute(b['bias']).mean() for b in _] for _ in bias_avg_var]
bias_avg_var = [sum(_)/len(_) for _ in bias_avg_var]
bias_avg_var = pd.DataFrame([_ for _ in bias_avg_var], index=[_ for _ in range(2,5)]).T
bias_avg_var.plot();plt.show()

rmse_avg_var = [[_ for _ in bias if _['k'] == count] for count in range(2,5)]
rmse_avg_var = [[np.absolute(b['bias']).mean()**2 for b in _] for _ in rmse_avg_var]
rmse_avg_var = [(sum(_)/len(_))**0.5 for _ in rmse_avg_var]
rmse_avg_var = pd.DataFrame([_ for _ in rmse_avg_var], index=[_ for _ in range(2,5)]).T
rmse_avg_var.plot();plt.show()

# Separate bias list by n_obs
# n_obs: 200 = n_var:2,3,4 & 200 obs used in n_lags: 1,2,3,4; and so on
# so, each n_obs is repeated
bias_avg_T = [[_ for _ in bias if _['T'] == count*200] for count in range(1,4)]
bias_avg_T = [[np.absolute(b['bias']).mean() for b in _] for _ in bias_avg_T]
bias_avg_T = [sum(_)/len(_) for _ in bias_avg_T]
bias_avg_T = pd.DataFrame([_ for _ in bias_avg_T], index=[_*200 for _ in range(1,4)]).T
bias_avg_T_tvp = [[_ for _ in bias if _['T'] == count*200] for count in range(1,4)]
bias_avg_T_tvp = [[np.absolute(b['bias_tvp']).mean() for b in _] for _ in bias_avg_T_tvp]
bias_avg_T_tvp = [sum(_)/len(_) for _ in bias_avg_T_tvp]
bias_avg_T_tvp = pd.DataFrame([_ for _ in bias_avg_T_tvp], index=[_*200 for _ in range(1,4)]).T

bias_avg_T.plot(); plt.show()
bias_avg_T_tvp.plot(); plt.show()

rmse_avg_T = [[_ for _ in bias if _['T'] == count*200] for count in range(1,4)]
rmse_avg_T = [[np.absolute(b['bias']).mean()**2 for b in _] for _ in rmse_avg_T]
rmse_avg_T = [(sum(_)/len(_))**0.5 for _ in rmse_avg_T]
rmse_avg_T = pd.DataFrame([_ for _ in rmse_avg_T], index=[_*200 for _ in range(1,4)]).T
rmse_avg_T_tvp = [[_ for _ in bias if _['T'] == count*200] for count in range(1,4)]
rmse_avg_T_tvp = [[np.absolute(b['bias_tvp']).mean()**2 for b in _] for _ in rmse_avg_T_tvp]
rmse_avg_T_tvp = [(sum(_)/len(_))**0.5 for _ in rmse_avg_T_tvp]
rmse_avg_T_tvp = pd.DataFrame([_ for _ in rmse_avg_T_tvp], index=[_*200 for _ in range(1,4)]).T

rmse_avg_T.plot(); plt.show()
rmse_avg_T_tvp.plot(); plt.show()

# Separate bias list by n_lags
bias_avg_p = [[_ for _ in bias if _['p'] == count] for count in range(1,5)]
bias_avg_p = [[np.absolute(b['bias']).mean() for b in _] for _ in bias_avg_p]
bias_avg_p = [sum(_)/len(_) for _ in bias_avg_p]
bias_avg_p = pd.DataFrame([_ for _ in bias_avg_p], index=[_ for _ in range(1,5)]).T
bias_avg_p.plot(); plt.show()

rmse_avg_p = [[_ for _ in bias if _['p'] == count] for count in range(1,5)]
rmse_avg_p = [[np.absolute(b['bias']).mean()**2 for b in _] for _ in rmse_avg_p]
rmse_avg_p = [(sum(_)/len(_))**0.5 for _ in rmse_avg_p]
rmse_avg_p = pd.DataFrame([_ for _ in rmse_avg_p], index=[_ for _ in range(1,5)]).T
rmse_avg_p.plot(); plt.show()


tab = pd.DataFrame(columns=range(0,11), index=pd.MultiIndex.from_product(
    [range(2,5), range(1,5)], names=['Variables', 'Lags']
))
tab_bias = tab.copy()
tab_rmse = tab.copy()
tab_CR = tab.copy()

tab_CR_T = pd.DataFrame(columns=range(0,11), index=[_*200 for _ in range(1,4)])

tab_bias_tvp = tab.copy()
tab_rmse_tvp = tab.copy()
tab_CR_tvp = tab.copy()

tab_CR_T_tvp = pd.DataFrame(columns=range(0,11), index=[_*200 for _ in range(1,4)])

for n_var in range(2,5):
    for n_lags in range(1,5):
        for h in range(0,11):
            bias_h = [np.absolute(b['bias'][h]).mean() for b in bias if (b['k'] == n_var) & (b['p'] == n_lags)]
            bias_h = sum(bias_h)/len(bias_h)
            tab_bias.loc[(n_var, n_lags), h] = bias_h

            rmse_h = [np.absolute(b['bias'][h]).mean()**2 for b in bias if (b['k'] == n_var) & (b['p'] == n_lags)]
            rmse_h = (sum(rmse_h)/len(rmse_h))**0.5
            tab_rmse.loc[(n_var, n_lags), h] = rmse_h

            coverage_h = [(b['ci_l'][h].values < b['tvp_irf'][h].values) & (b['tvp_irf'][h].values <= b['ci_u'][h].values) for b in bias if (b['k'] == n_var) & (b['p'] == n_lags)]
            tab_CR.loc[(n_var, n_lags), h] = np.mean(coverage_h)

for n_obs in [_*200 for _ in range(1,4)]:
    for h in range(0,11):
        coverage_h_T = [np.mean((b['ci_l'][h].values < b['tvp_irf'][h].values) & (b['tvp_irf'][h].values <= b['ci_u'][h].values)) for b in bias if (b['T'] == n_obs)]
        tab_CR_T.loc[n_obs, h] = np.mean(coverage_h_T)
        coverage_h_T_tvp = [np.mean((b['tvp_ci_l'][h].values < b['tvp_irf'][h].values) & (b['tvp_irf'][h].values <= b['tvp_ci_u'][h].values)) for b in bias if (b['T'] == n_obs)]
        tab_CR_T_tvp.loc[n_obs, h] = np.mean(coverage_h_T_tvp)


print(tab_bias[[0,2,8]].to_latex(float_format="%.2f"))
print(tab_rmse[[0,2,8]].to_latex(float_format="%.2f"))
print(tab_CR[[2,4,8]].to_latex(float_format="%.2f"))
print(pd.concat(
    [tab_bias[[0,2,8]],
     tab_rmse[[0,2,8]],
     tab_CR[[2,4,8]]],
     keys=['Bias', 'RMSE', 'Coverage Rate'], axis = 1
).to_latex(float_format="%.2f"))
print(tab_bias[[0,2,8]].to_latex(float_format="%.2f"))
print(tab_rmse[[0,2,8]].to_latex(float_format="%.2f"))
print(tab_CR[[2,4,8]].to_latex(float_format="%.2f"))

print(tab_CR_T[[2,4,8]].to_latex(float_format="%.2f"))
# Exit