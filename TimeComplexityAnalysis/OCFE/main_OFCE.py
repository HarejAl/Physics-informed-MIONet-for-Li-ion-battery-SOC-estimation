"""
Created on Tue Feb 25 16:02:10 2025

@author: lorenzo
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz
from scipy_dae.integrate import solve_dae
import matplotlib.pyplot as plt

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

def plot_comparison(true_sol, est_sol, t, r_vect, electrode_name="Electrode"):
    """
    Plots heatmaps of the true concentration, estimated concentration, and their error.
    
    Parameters:
        true_sol      : 2D ndarray (n_space x n_time) for the true/interpolated concentration.
        est_sol       : 2D ndarray (n_space x n_time) for the estimated concentration.
        t             : 1D array of time points.
        r_vect        : 1D array of spatial node locations.
        electrode_name: String for labeling (e.g., "Positive Electrode" or "Negative Electrode").
    """
    
    # Compute the error between true and estimated solutions.
    error = true_sol - est_sol
    
    # Create a figure with three subplots.
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot true concentration heatmap.
    im0 = axs[0].imshow(true_sol, aspect='auto', origin='lower',
                          extent=[t[0], t[-1], r_vect[0], r_vect[-1]], cmap='RdBu')
    axs[0].set_title(f"{electrode_name} True Concentration")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position (m)")
    fig.colorbar(im0, ax=axs[0])
    
    # Plot estimated concentration heatmap.
    im1 = axs[1].imshow(est_sol, aspect='auto', origin='lower',
                          extent=[t[0], t[-1], r_vect[0], r_vect[-1]], cmap='RdBu')
    axs[1].set_title(f"{electrode_name} Estimated Concentration")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Position (m)")
    fig.colorbar(im1, ax=axs[1])
    
    # Plot the error heatmap.
    im2 = axs[2].imshow(error, aspect='auto', origin='lower',
                          extent=[t[0], t[-1], r_vect[0], r_vect[-1]], cmap='RdBu')
    axs[2].set_title(f"{electrode_name} Concentration Error")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Position (m)")
    fig.colorbar(im2, ax=axs[2])
    
    plt.tight_layout()
    plt.show()

## MAIN
if __name__ == '__main__':

    # Load the MATLAB .mat file (assumed to be "pybamm_sol_data.mat")
    directory = r"C:/Users/alexa/OneDrive - Politecnico di Milano/Desktop/PHD_python/SPM_pimionet/SPM_PIMIONet_new/TimeComplexityAnalysis_SPM/OCFE"
    filepath = os.path.join(directory, "pybamm_sol_data.mat")
    plant = loadmat(filepath)
    
    # Plant simulation settings:
    T_end = 3000
    t = np.linspace(0,T_end, int(T_end/1+1))
    dt = np.mean(np.diff(t))
    
    # Extract additional plant data:
    # rn = np.squeeze(plant['rn'][:, 0, 0])
    # cn_true = np.squeeze(plant['cn'][:, 0, :])
    # rp = np.squeeze(plant['rp'][:, 0, 0])
    # cp_true = np.squeeze(plant['cp'][:, 0, :])
    
    import pybamm
    param1 = pybamm.ParameterValues("Mohtat2020")
    var_pts = {
    "x_n": 10,  # negative electrode
    "x_s": 10,  # separator
    "x_p": 10,  # positive electrode
    "r_n": 100,  # negative particle
    "r_p": 100,  # positive particle
    }
    
    batt_model = pybamm.lithium_ion.SPM()
    param1.set_initial_stoichiometries(1)
    rn = rp = np.linspace(0,1,100)

    ## SOLUTION
    sim = pybamm.Simulation(batt_model, var_pts=var_pts, parameter_values=param1) #  , experiment = experiment
    solution = sim.solve(t)
    time = solution["Time [s]"].entries # time_pb is automatically cutted at the cut off value 
    cp_true = np.mean(solution["Positive particle concentration [mol.m-3]"].entries,axis=1)
    cn_true = np.mean(solution["Negative particle concentration [mol.m-3]"].entries,axis=1)
    
    # Assemble the parameters in a dictionary
    param = {}
    param['dt'] = dt
    
    # SP Model parameters (same as those used in PyBaMM):
    param['neg'] = {}
    param['pos'] = {}
    
    Diff_neg = param1['Negative electrode diffusivity [m2.s-1]']
    Diff_pos = param1['Positive electrode diffusivity [m2.s-1]']
    
    param['neg']['D'] = Diff_neg(0.5,298.15).value    # Negative electrode diffusivity (m^2/s)
    param['pos']['D'] = Diff_pos(0.5,298.15).value     # Positive electrode diffusivity (m^2/s)
    param['neg']['cmax'] = param1['Maximum concentration in negative electrode [mol.m-3]']  # Maximum concentration in negative electrode (mol/m^3)
    param['pos']['cmax'] = param1['Maximum concentration in positive electrode [mol.m-3]']    # Maximum concentration in positive electrode (mol/m^3)
    param['neg']['R'] = param1['Negative particle radius [m]']         # Negative particle radius (m)
    param['pos']['R'] = param1['Positive particle radius [m]']        # Positive particle radius (m)
    param['neg']['eps'] = param1['Negative electrode active material volume fraction']        # Negative electrode active material volume fraction
    param['pos']['eps'] = param1['Positive electrode active material volume fraction']        # Positive electrode active material volume fraction
    param['neg']['L'] = param1['Negative electrode thickness [m]']         # Negative electrode thickness (m)
    param['pos']['L'] = param1['Positive electrode thickness [m]']        # Positive electrode thickness (m)
    
    param['F'] = param1['Faraday constant [C.mol-1]']          # Faraday's constant (C/mol)
    param['T'] = 298.15                # Cell temperature (K)
    param['R'] = 8.314472              # Universal Gas Constant (J/mol/K)
    param['ce0'] = param1['Initial concentration in electrolyte [mol.m-3]']           # Initial concentration in electrolyte (mol/m^3)
    param['W'] = param1['Electrode width [m]']              # Electrode width (m)
    param['H'] = param1['Electrode height [m]']             # Electrode height (m)
    param['I1C'] = param1['Nominal cell capacity [A.h]']          # 1C Rate Current (A)
    
    # Set OCFE-specific parameters
    param['ne'] = 4      # number of finite elements (fe)
    param['n'] = 6       # total number of collocation points per FE
    param['f'] = 0.96    # shaping factor
    
    # Total number of state variables:
    N_arr = param['ne'] * (param['n'] - 1) + 1  # --> Two enighbour elements share the boundary state

    # Allocate state variable vector x of size 2*N x len(t)
    x = np.zeros((2 * N_arr, len(t)))
    
    # Assume plant['I'] exists; extract u (ensure proper squeezing if needed)
    u = np.array([param['I1C']])
    
    # Initial conditions:
    cn0 = np.mean(cn_true[:,0])
    cp0 = np.mean(cp_true[:,0])
    x = np.concatenate((np.full(N_arr, cn0),np.full(N_arr, cp0))).reshape(-1,1)
    

    time_step = np.array([0, t[1]]) # --> t[1] is teh impsoed time step
    
    for j in range(1,len(t)):
        x_p = stateFcn(x[:, -1], u[0], time_step, param)[:,-1:] # [0,:] is the k-th step, [:,1] is the k+1 -th step
        x = np.hstack((x, x_p))
    
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
    r_vect = np.concatenate((r_vect_jj, np.array([final_node])))

    # Interpolate the true solution from PyBaMM to compare with the OCFE method.
    # For the negative electrode concentration:
    cn_true_int_list = []
    for ti in range(len(t)):
        interp_func = interp1d(rn, cn_true[:, ti], kind='linear', fill_value="extrapolate")
        cn_true_int_list.append(interp_func(r_vect))
        
    cn_true_int = np.column_stack(cn_true_int_list)/param['neg']['cmax'] 
    
    # For the positive electrode concentration:
    cp_true_int_list = []
    for ti in range(len(t)):
        interp_func = interp1d(rp, cp_true[:, ti], kind='linear', fill_value="extrapolate")
        cp_true_int_list.append(interp_func(r_vect))            
    
    cp_true_int = np.column_stack(cp_true_int_list)/param['pos']['cmax']
    
    # Extract estimated concentration evolution from OCFE:
    cn_est = x[:N_arr, :]/param['neg']['cmax']
    cp_est = x[N_arr:, :]/param['pos']['cmax']
    
    # Plot comparison for the positive electrode concentration:
    plot_comparison(cp_true_int, cp_est, t, r_vect, electrode_name="Positive Electrode")
    
    # Optionally, to compare the negative electrode concentration profiles:
    plot_comparison(cn_true_int, cn_est, t, r_vect, electrode_name="Negative Electrode")
    
    diff_n = (cn_true_int - cn_est) # they are already normalized / param['neg']['cmax']
    en = np.mean(diff_n ** 2)
    
    print('Neg error {} (-)'.format(en))
    
    diff_p = (cp_true_int - cp_est) # / param['pos']['cmax']
    ep = np.mean(diff_p ** 2)
    print('Pos error {} (-)'.format(ep))
    
    en = np.mean(np.linalg.norm((cn_true_int - cn_est), axis = 1))
    print('Neg error {} (-)'.format(en))
    
    ep = np.mean(np.linalg.norm((cp_true_int - cp_est), axis=1))
    print('Pos error {} (-)'.format(ep))
