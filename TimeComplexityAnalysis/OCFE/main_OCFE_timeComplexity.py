# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:31:47 2025

@author: lorenzo
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz
from scipy_dae.integrate import solve_dae
import matplotlib.pyplot as plt
import timeit
from time import time

### Function Definitions

def chebdif(N, M):
    """
    Computes the differentiation matrices D1, D2, ..., DM on Chebyshev nodes.
    
    Parameters:
        N (int): Size of differentiation matrix.
        M (int): Number of derivatives required (0 < M <= N-1).
        
    Returns:
        x (ndarray): Chebyshev points (1D array of length N).
        DM (ndarray): Differentiation matrices stored in a 3D array of shape (M, N, N)
                      such that DM[i] is the (i+1)-th derivative matrix.
                      
    Note:
        The function uses a flipping trick to build the intermediate matrix DX.
        A final mirror (flip) on each DM is applied to match MATLAB’s ordering.
    """
    # Identity matrix and boolean mask for the diagonal.
    I = np.eye(N)
    L = I.astype(bool)
    
    # Indices for the flipping trick.
    n1 = int(np.floor(N / 2))
    n2 = int(np.ceil(N / 2))
    
    # Theta vector and Chebyshev nodes.
    k = np.arange(N)
    th = k * np.pi / (N - 1)
    seq = np.arange(N-1, 1-N-1, -2)   # Equivalent to MATLAB: [N-1:-2:1-N]'
    x = np.sin(np.pi * seq / (2 * (N - 1)))
    
    # Build matrix T.
    T = np.tile(th / 2, (N, 1))
    
    # Compute DX using the trigonometric identity.
    DX = 2 * np.sin(T.T + T) * np.sin(T.T - T)
    
    # Apply the flipping trick.
    top = DX[:n1, :]
    bottom = -np.flip(DX[:n2, :])
    DX = np.vstack((top, bottom))
    
    # Set the diagonal of DX to 1.
    np.fill_diagonal(DX, 1)
    
    # Construct the Toeplitz matrix C (converted to float for divisions).
    c = (-1) ** k
    C = toeplitz(c).astype(np.float64)
    C[0, :] *= 2
    C[-1, :] *= 2
    C[:, 0] /= 2
    C[:, -1] /= 2
    
    # Compute matrix Z = 1/(x(k)-x(j)) with zeros on the diagonal.
    Z = 1.0 / DX
    np.fill_diagonal(Z, 0)
    
    # Initialize D and allocate DM.
    D = np.eye(N)
    DM = np.empty((M, N, N), dtype=np.float64)
    
    # Compute differentiation matrices iteratively.
    for ell in range(1, M + 1):
        D = ell * Z * (C * np.diag(D)[:, None] - D)
        np.fill_diagonal(D, -np.sum(D, axis=1))
        DM[ell - 1, :, :] = D
    
    x = np.flip(x) # flip Chebyshev points vector for coherence
    
    return x, DM

def assembleDAEsys(ne, n, f, param):
    """
    Assemble the DAE system matrices for a spectral-collocation discretization.
    
    Parameters:
        ne (int): number of finite elements.
        n  (int): number of collocation points per finite element (used in chebdif).
        f  (float): a factor used to compute the finite element lengths.
        param (dict): dictionary containing parameters; must include:
                      'R'  : the total domain length.
                      'D'  : the diffusivity (or similar coefficient).
    
    Returns:
        M (ndarray): Mass matrix of size [ne*(n-2+1)+1] x [ne*(n-2+1)+1].
        K (ndarray): Stiffness matrix (same size as M).
        b (ndarray): Right‐hand side input vector.
    
    """
    # Get Chebyshev collocation points and derivative matrices.
    xi, DM = chebdif(n, 2)
    S = DM[0, :, :]  # first derivative matrix
    T = DM[1, :, :]  # second derivative matrix

    # Parameters.
    R = param['R']
    D = param['D']

    # Finite element lengths: in MATLAB,
    #   h = [repmat(f*R/(ne-1), [ne-1,1]); (1-f)*R];
    h = np.concatenate((np.full(ne - 1, f * R / (ne - 1)),
                        np.array([(1 - f) * R]))
                       )
    # Cumulative sum to get right-end positions, and left endpoints.
    re = np.cumsum(h)
    ri = np.concatenate(([0], re[:-1]))
    
    # Define a helper function for the location of discretization nodes.
    # Here j is the finite element index (0-indexed) and idx are indices into xi.
    def r_func(j, idx):
        return (h[j] / 2) * xi[idx] + (ri[j] + re[j]) / 2

    # Number of inner collocation points per finite element.
    ncp = n - 2  # (in MATLAB, ncp = n-2)

    # Total number of degrees of freedom:
    dim = ne * (ncp + 1) + 1

    # Initialize stiffness and mass matrices.
    K = np.zeros((dim, dim))
    M = np.zeros((dim, dim))

    # 1st boundary condition (r = 0, no flux):
    # MATLAB: K(1,1:n) = (2*D/h(1))*S(1,:);
    # In Python: row 0, columns 0 to n-1.
    K[0, 0:n] = (2 * D / h[0]) * S[0, :]

    # Loop over finite elements.
    for j in range(ne):
        # Inner collocation point indices
        # In Python, these are indices 1,..., ncp.
        ii = np.arange(1, ncp + 1)  # length ncp

        # Row indices in K for the inner collocation points of element j:
        row_inds = j * (ncp + 1) + ii

        # Column indices in the local block for element j:
        col_inds = j * (ncp + 1) + np.arange(0, n)

        # Assemble the PDE evaluated at the inner collocation points.
        term1 = D * ((2 / h[j]) ** 2) * T[ii, :]
        term2 = (2 * D / r_func(j, ii)[:, None]) * (2 / h[j]) * S[ii, :]
        K[np.ix_(row_inds, col_inds)] = term1 + term2

        # For all but the last element, impose continuity conditions.
        if j != ne - 1:
            # The continuity equation is imposed at the node common to finite elements j and j+1.
            # MATLAB: row index = (jj-1)*(ncp+1)+ncp+2; convert: j*(ncp+1) + ncp+1.
            row_idx = j * (ncp + 1) + ncp + 1

            # (a) First part:
            # which in Python becomes:
            col_range1 = np.arange(j * (ncp + 1), j * (ncp + 1) + n - 1)
            # In Python, S[n-1, 0:n-1]
            K[row_idx, col_range1] = (2 * D / h[j]) * S[n - 1, 0:n - 1]

            # (b) Second part:
            col_idx = j * (ncp + 1) + (n - 1)
            # In Python, S(n,n) -> S[n-1, n-1] and S(1,1) -> S[0,0].
            K[row_idx, col_idx] = (2 * D / h[j]) * S[n - 1, n - 1] - (2 * D / h[j + 1]) * S[0, 0]

            # (c) Third part:
            # Convert: For finite element j+1 (MATLAB jj = j+1), the columns become
            # from: 1 + (ncp+1)*(j+1)  to: (ncp+1)*(j+1) + n - 1.
            col_range2 = np.arange(1 + (ncp + 1) * (j + 1), (ncp + 1) * (j + 1) + n)
            K[row_idx, col_range2] = -(2 * D / h[j + 1]) * S[0, 1:n]

        # Mass matrix: at inner collocation points, assign identity.
        M[np.ix_(row_inds, row_inds)] = np.eye(len(ii))

    # Second boundary condition (r = R) at the last row.
    col_start = (ne - 1) * (ncp + 1)
    col_end = col_start + n  # Python slice will cover indices col_start to col_end-1.
    K[-1, col_start:col_end] = -(2 * D / h[ne - 1]) * S[n - 1, :]

    # Input vector b.
    b = np.zeros(dim)
    b[-1] = 1

    return M, K, b

def stateFcn(x, u, t, param):
    """
    Solves the DAE system using scipy_dae's solve_dae to handle the singular mass matrix.
    
    Parameters:
        x     : Initial state vector (1D array of length 2*N)
        u     : Input scalar (e.g. current)
        t     : 1D array of time points for integration
        param : Dictionary of parameters.
    
    Returns:
        x_sol : 2D array containing the state evolution.
                Shape is (n_states, len(t)); each column is the state vector at a time point.
    """
    # Assemble DAE system matrices for negative and positive electrodes.
    Mn, Kn, bn = assembleDAEsys(param['ne'], param['n'], param['f'], param['neg'])
    Mp, Kp, bp = assembleDAEsys(param['ne'], param['n'], param['f'], param['pos'])
    
    # Total number of discretization points.
    N = param['ne'] * (param['n'] - 1) + 1
    
    # Build global stiffness matrix K (size 2N x 2N)
    K = np.zeros((2 * N, 2 * N))
    K[0:N, 0:N] = Kn
    K[N:2*N, N:2*N] = Kp

    # Build global mass matrix M (size 2N x 2N)
    M = np.zeros((2 * N, 2 * N))
    M[0:N, 0:N] = Mn
    M[N:2*N, N:2*N] = Mp

    # Extract additional parameters.
    dt      = param['dt']
    F_const = param['F']
    A       = param['H'] * param['W']
    an      = 3.0 * param['neg']['eps'] / param['neg']['R']
    ap      = 3.0 * param['pos']['eps'] / param['pos']['R']
    Ln      = param['neg']['L']
    Lp      = param['pos']['L']
    
    # Define input functions.
    def un(t_val):
        return -(u / A / F_const / an / Ln) if t_val >= 0 else 0.0
    def up(t_val):
        return  (u / A / F_const / ap / Lp) if t_val >= 0 else 0.0

    # Define the DAE residual: F(t,x,x_dot) = M*x_dot - (K*x + forcing(t)) = 0.
    def F_dae(t_val, x_val, xdot_val):
        forcing = np.concatenate((bn * un(t_val), bp * up(t_val)))
        return M.dot(xdot_val) - (K.dot(x_val) + forcing)
    
    # Time span.
    t_span = (t[0], t[-1])
    
    # Compute initial derivative guess yp0 by solving M*yp0 = K*x + forcing at t[0].
    forcing0 = np.concatenate((bn * un(t[0]), bp * up(t[0])))
    f0 = K.dot(x) + forcing0  # right-hand side at t[0]
    yp0, residuals, rank, s = np.linalg.lstsq(M, f0, rcond=None)
    
    # Solve the DAE using scipy_dae's solve_dae.
    sol = solve_dae(F_dae, t_span, x, yp0, atol=1e-2, rtol=1e-2, method="Radau", t_eval=t, max_step=10)
    
    # Return the solution array (each column corresponds to the state at a time point).
    return sol.y

def plot_results(x, ct_avg, err, opt_fcn):
    """
    Creates a figure with three overlaid axes to plot:
      - Computational time vs. state vector size (red, left y-axis)
      - Error vs. state vector size (blue, middle y-axis)
      - Optimality function vs. state vector size (black, right y-axis)
      
    Parameters:
        x       : 1D array of state vector sizes (e.g. x = 2*N)
        ct_avg  : 1D array of computational times (msec)
        err     : 1D array of error values
        opt_fcn : 1D array of optimality function values (e.g., normalized ct_avg + normalized err)
    """
    # Figure size (cm)
    figureWidth_cm = 20   # Width in centimeters
    figureHeight_cm = 13  # Height in centimeters
    # Convert centimeters to inches (1 inch = 2.54 cm)
    figureWidth_in = figureWidth_cm / 2.54
    figureHeight_in = figureHeight_cm / 2.54

    # Create the figure
    fig = plt.figure(figsize=(figureWidth_in, figureHeight_in))
    
    # Axes position in centimeters: [left, bottom, width, height]
    axesPosition_cm = np.array([6, 3, 8, 8])
    yWidth_cm = 1.8  # y-axes spacing in cm

    # Convert axes position to normalized coordinates (relative to figure dimensions in cm)
    norm_left_h1 = axesPosition_cm[0] / figureWidth_cm
    norm_bottom   = axesPosition_cm[1] / figureHeight_cm
    norm_width_h1 = axesPosition_cm[2] / figureWidth_cm
    norm_height   = axesPosition_cm[3] / figureHeight_cm

    # Compute x-axis limits from x data
    xLimit = [np.min(x), np.max(x)]
    diff_x = xLimit[1] - xLimit[0]
    # xOffset (in cm): MATLAB computes xOffset = -yWidth * diff(xLimit) / axesPosition(3)
    xOffset_cm = - yWidth_cm * diff_x / axesPosition_cm[2]

    # --- Create Axes h1 ---
    # h1: original axes position
    h1_rect = [norm_left_h1, norm_bottom, norm_width_h1, norm_height]
    h1 = fig.add_axes(h1_rect, facecolor='w')
    h1.tick_params(axis='x', colors='k')
    h1.tick_params(axis='y', colors='r')
    h1.set_xlim(xLimit)
    h1.set_ylim([np.min(ct_avg), np.max(ct_avg) * 1.1])
    
    # --- Create Axes h2 ---
    axesPosition_h2_cm = axesPosition_cm + np.array([-yWidth_cm, 0, yWidth_cm, 0])
    norm_left_h2 = axesPosition_h2_cm[0] / figureWidth_cm
    norm_width_h2 = axesPosition_h2_cm[2] / figureWidth_cm
    h2_rect = [norm_left_h2, norm_bottom, norm_width_h2, norm_height]
    h2 = fig.add_axes(h2_rect, facecolor='none')
    h2.tick_params(axis='x', colors='k')
    h2.tick_params(axis='y', colors='b')
    h2.set_xlim([xLimit[0] + xOffset_cm, xLimit[1]])
    h2.set_ylim([np.min(err), np.max(err) * 1.1])
    h2.set_xticks([])  # remove x-tick labels
    
    # --- Create Axes h3 ---
    axesPosition_h3_cm = axesPosition_cm + np.array([-2*yWidth_cm, 0, 2*yWidth_cm, 0])
    norm_left_h3 = axesPosition_h3_cm[0] / figureWidth_cm
    norm_width_h3 = axesPosition_h3_cm[2] / figureWidth_cm
    h3_rect = [norm_left_h3, norm_bottom, norm_width_h3, norm_height]
    h3 = fig.add_axes(h3_rect, facecolor='none')
    h3.tick_params(axis='x', colors='k')
    h3.tick_params(axis='y', colors='k')
    h3.set_xlim([xLimit[0] + 2*xOffset_cm, xLimit[1]])
    h3.set_ylim([0, 1.1])
    h3.set_xticks([])

    # Add labels.
    h1.set_xlabel('State Vector Size (n)')
    h1.set_ylabel('Comp. Time (msec)', color='r')
    h2.set_ylabel('Avg. Normalized Error (-)', color='b')
    h3.set_ylabel('Optimality Fcn (-)', color='k')
    
    # Plot the data.
    h1.plot(x, ct_avg, 'r-', linewidth=1.25)
    h2.plot(x, err, 'b-', linewidth=1.25)
    h3.plot(x, opt_fcn, 'k--', linewidth=1.25)
    
    plt.title('OCFE')
    
    plt.show()

## MAIN

if __name__ == '__main__':
    
    print('OCFE time complexity')

    # Load the MATLAB .mat file (assumed to be "pybamm_sol_data.mat")
    directory = r"C:/Users/alexa/OneDrive - Politecnico di Milano/Desktop/PHD_python/SPM_pimionet/SPM_PIMIONet_new/TimeComplexityAnalysis_SPM/OCFE"
    filepath = os.path.join(directory, "pybamm_sol_data.mat")
    plant = loadmat(filepath)
    
    # Plant simulation settings:
    t = np.squeeze(plant['t'])
    dt = np.mean(np.diff(t))
    
    # Extract additional plant data:
    rn = np.squeeze(plant['rn'][:, 0, 0])
    cn_true = np.squeeze(plant['cn'][:, 0, :])
    rp = np.squeeze(plant['rp'][:, 0, 0])
    cp_true = np.squeeze(plant['cp'][:, 0, :])
    
    # Assemble the parameters in a dictionary
    param = {}
    param['dt'] = dt
    
    # SP Model parameters (same as those used in PyBaMM):
    param['neg'] = {}
    param['pos'] = {}
    
    param['neg']['D'] = 3.9e-14        # Negative electrode diffusivity (m^2/s)
    param['pos']['D'] = 1.0e-13        # Positive electrode diffusivity (m^2/s)
    param['neg']['cmax'] = 2.4983e4    # Maximum concentration in negative electrode (mol/m^3)
    param['pos']['cmax'] = 5.1218e4    # Maximum concentration in positive electrode (mol/m^3)
    param['neg']['R'] = 1e-5           # Negative particle radius (m)
    param['pos']['R'] = 1e-5           # Positive particle radius (m)
    param['neg']['eps'] = 0.6          # Negative electrode active material volume fraction
    param['pos']['eps'] = 0.5          # Positive electrode active material volume fraction
    param['neg']['L'] = 0.0001         # Negative electrode thickness (m)
    param['pos']['L'] = 0.0001         # Positive electrode thickness (m)
    
    param['F'] = 96485.33212           # Faraday's constant (C/mol)
    param['T'] = 298.15                # Cell temperature (K)
    param['R'] = 8.314472              # Universal Gas Constant (J/mol/K)
    param['ce0'] = 1000.0              # Initial concentration in electrolyte (mol/m^3)
    param['W'] = 0.207                 # Electrode width (m)
    param['H'] = 0.137                 # Electrode height (m)
    param['I1C'] = 0.680616            # 1C Rate Current (A)
    
    # Define the finite element vector:
    ne_vect = np.arange(1, 9)
    # ne_vect = np.arange(2, 5)
    
    # Preallocate lists/arrays:
    r_vect = [None] * len(ne_vect)            # List to hold r_vect for each ne value
    N_arr = np.zeros(len(ne_vect), dtype=int) # Total number of state variables for each run
    err_arr = np.zeros(len(ne_vect))          # Average error for each run
    ct_avg = np.zeros(len(ne_vect))           # Average computational time (msec)
    n_it = 5                                 # Number of iterations for timing
    
    for jj, ne_val in enumerate(ne_vect):
        
        
        
        # Set OCFE-specific parameters
        param['ne'] = ne_val      # number of finite elements (fe)
        param['n'] = 6            # total number of collocation points per FE
        param['f'] = 0.96         # shaping factor
    
        # Total number of state variables:
        N_arr[jj] = param['ne'] * (param['n'] - 1) + 1
        
        print('Number of states: {}'.format(param['ne'] * (param['n'] - 1) + 1))
    
        
        # Assume plant['I'] exists; extract u (ensure proper squeezing if needed)
        u = np.squeeze(plant['I'])
        
        # Initial conditions:
        cn0 = np.mean(plant['cn'][:, 0, 0])
        cp0 = np.mean(plant['cp'][:, 0, 0])

    
        # Estimated average computational time (in msec):
        ct = []
        
        for _ in range(n_it):
            
            x = np.concatenate((np.full(N_arr[jj], cn0), np.full(N_arr[jj], cp0))).reshape(-1,1)
            time_step = np.array([0, t[1]]) # --> t[1] is teh impsoed time step
            
            t0 = time() 
            
            for j in range(1,len(t)):

                x_p = stateFcn(x[:, -1], u[0], time_step, param)[:,-1:] # [0,:] is the k-th step, [:,1] is the k+1 -th step
                x = np.hstack((x, x_p))
            
            
            ct.append((time() - t0) / len(t) * 1e3)  # time needed for 1 time step propagation
            
        ct_avg[jj] = np.mean(ct)
        
        # Compute Chebyshev nodes (and flip them):
        xi, _ = chebdif(param['n'], 2)
        
        # Finite element lengths:
        R_val = param['neg']['R']   # NB: Rn = Rp in this case
        h = np.concatenate((np.full(param['ne'] - 1, param['f'] * R_val / (param['ne'] - 1)),
                            np.array([(1 - param['f']) * R_val])))
        
        # Compute right-end and left-end positions:
        re_vec = np.cumsum(h)
        ri_vec = np.concatenate(([0], re_vec[:-1]))
        
        # Define the node location function:
        def r_func(j, idx):
            # j: finite element index (0-indexed)
            # idx: indices into xi (an array)
            return (h[j] / 2) * xi[idx] + (ri_vec[j] + re_vec[j]) / 2
    
        # Assemble the vector of discretization nodes r_vect for the current FE partition:
        r_list = [r_func(i, np.arange(0, param['n'] - 1)) for i in range(param['ne'])]
        r_vect_jj = np.concatenate(r_list)
        # Append the final node:
        final_node = r_func(param['ne'] - 1, np.array([param['n'] - 1]))[0]
        r_vect_jj = np.concatenate((r_vect_jj, np.array([final_node])))
        r_vect[jj] = r_vect_jj
    
        # Interpolate the true solution from PyBaMM to compare with the OCFE method.
        # For the negative electrode concentration:
        cn_true_int_list = []
        for ti in range(len(t)):
            interp_func = interp1d(rn, cn_true[:, ti], kind='linear', fill_value="extrapolate")
            cn_true_int_list.append(interp_func(r_vect_jj))
        cn_true_int = np.column_stack(cn_true_int_list)
        
        # For the positive electrode concentration:
        cp_true_int_list = []
        for ti in range(len(t)):
            interp_func = interp1d(rp, cp_true[:, ti], kind='linear', fill_value="extrapolate")
            cp_true_int_list.append(interp_func(r_vect_jj))
        cp_true_int = np.column_stack(cp_true_int_list)
        
        # Extract estimated concentration evolution from OCFE:
        cn_est = x[0:N_arr[jj], :]
        cp_est = x[N_arr[jj]:, :]
        
        # Calculate average error:
        en = np.mean(np.linalg.norm((cn_true_int - cn_est) / param['neg']['cmax'], axis=1))
        ep = np.mean(np.linalg.norm((cp_true_int - cp_est) / param['pos']['cmax'], axis=1))
        err_arr[jj] = (en + ep)/2

# Normalize ct_avg and err_arr to the [0, 1] range.
ct_n = (ct_avg - np.min(ct_avg)) / (np.max(ct_avg) - np.min(ct_avg))
err_n = (err_arr - np.min(err_arr)) / (np.max(err_arr) - np.min(err_arr))

# Compute the optimality function (as sum of normalized ct_avg and err_arr).
opt_fcn = ct_n + err_n

# Compute x as 2 times the number of state variables for each case.
x_plot = 2 * N_arr

# Call the plotting function.
plot_results(x_plot, ct_avg, err_arr, opt_fcn)

# # Save the array
np.save("OCFE_tuple2.npy", (x_plot, ct_avg, err_arr, opt_fcn))

