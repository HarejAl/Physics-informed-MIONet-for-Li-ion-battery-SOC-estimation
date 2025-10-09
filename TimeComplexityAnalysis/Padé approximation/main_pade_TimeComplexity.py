import os
import pickle
import numpy as np
import sympy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae
from scipy.interpolate import interp1d
from time import time

def pade_state_space_realization(param, Gpade):
    """
    Compute a state-space realization (A, B, C, D) from the Padé approximant Gpade.
    """
    # Separate the symbolic expression into numerator and denominator
    num_s, den_s = sp.fraction(Gpade)
    
    D_sym, R_sym, s_sym = sp.symbols('D R s')
    num_coef = sp.Poly(num_s, s_sym).all_coeffs()
    num_coef = list(reversed(num_coef))
    den_coef = sp.Poly(den_s, s_sym).all_coeffs()
    den_coef = list(reversed(den_coef))
    
    # Normalize by dividing by the last element of the denominator coefficients.
    den_last = den_coef[1]
    num_coef_norm = [sp.simplify(c/den_last) for c in num_coef]
    den_coef_norm = [sp.simplify(c/den_last) for c in den_coef]
    
    subs_dict = {D_sym:  param['D'], R_sym:  param['R']}
    num_coeffs = [float(co.subs(subs_dict).evalf()) for co in num_coef_norm]
    den_coeffs = [float(co.subs(subs_dict).evalf()) for co in den_coef_norm]
    
    n = len(num_coeffs)
    
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    C = np.zeros((2, n))
    D = np.zeros((2, 1))
    
    A[:-1,1:] = np.eye(np.size(A[:-1,1:],1))
    A[-1,1:]  = -np.array(den_coeffs[1:-1])/np.array(den_coeffs[-1])
    B[-1]     = 1/np.array(den_coeffs[-1])
    C[0,:]    = np.array(num_coeffs)
    C[1,:]    = np.multiply(-np.array(num_coeffs[0]), np.append(-np.array(den_coeffs[1:-1]), -np.array(den_coeffs[-1])))
    
    return A, B, C, D

def stateFcn(x_state, u, t, param, Gpade):
    """
    Solves the DAE system using scipy_dae's solve_dae to handle the singular mass matrix.
    
    Parameters:
        x_state : Initial state vector (1D array) for the system.
        u       : Input scalar (current)
        t       : 1D array of time points for integration
        param   : Dictionary of parameters.
        Gpade   : The Padé approximant transfer function, expressed symbolically in s.
    
    Returns:
        x_state : 2D array containing the state evolution.
        c       : 2D array containing the computed output.
    """
    # Assemble DAE system matrices for negative and positive electrodes.
    An, Bn, Cn, Dn = pade_state_space_realization(param['neg'], Gpade)
    Ap, Bp, Cp, Dp = pade_state_space_realization(param['pos'], Gpade)
    
    dt      = param['dt']
    F_const = param['F']
    A_val   = param['H'] * param['W']
    an      = 3.0 * param['neg']['eps'] / param['neg']['R']
    ap      = 3.0 * param['pos']['eps'] / param['pos']['R']
    Ln      = param['neg']['L']
    Lp      = param['pos']['L']
    
    t_span = (t[0], t[-1])
    
    def un(t_val):
        return (u / A_val / F_const / an / Ln) if t_val >= 0 else 0.0
    def up(t_val):
        return -(u / A_val / F_const / ap / Lp) if t_val >= 0 else 0.0

    def ode_n(t_val, x_val, xdot_val):
        return An @ x_val + Bn.flatten() * un(t_val) - xdot_val
    
    yn0  = x_state[:int(np.size(x_state,0)/2)]
    yn10 = np.zeros(int(np.size(x_state,0)/2))
    sol_n = solve_dae(ode_n, t_span, yn0, yn10, atol=1e-2, rtol=1e-2, method="Radau", t_eval=t, max_step=10)
    xn    = sol_n.y
    cn    = np.matmul(Cn, xn)

    def ode_p(t_val, x_val, xdot_val):
        return Ap @ x_val + Bp.flatten() * up(t_val) - xdot_val

    yp0  = x_state[int(np.size(x_state,0)/2):]
    yp10 = np.zeros(int(np.size(x_state,0)/2))
    sol_p = solve_dae(ode_p, t_span, yp0, yp10, atol=1e-2, rtol=1e-2, method="Radau", t_eval=t, max_step=10)
    xp    = sol_p.y
    cp    = np.matmul(Cp, xp)

    x_state = np.vstack((xn, xp))
    c = np.vstack((cn, cp))
    
    return x_state, c

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
    h1.set_xlabel('TF order')
    h1.set_ylabel('Comp. Time (msec)', color='r')
    h2.set_ylabel('Avg. Normalized Error (-)', color='b')
    h3.set_ylabel('Optimality Fcn (-)', color='k')
    
    # Plot the data.
    h1.plot(x, ct_avg, 'r-', linewidth=1.25)
    h2.plot(x, err, 'b-', linewidth=1.25)
    h3.plot(x, opt_fcn, 'k--', linewidth=1.25)
    
    plt.title('PADE')
    
    plt.show()


## MAIN
if __name__ == '__main__':

    # Load the MATLAB .mat file
    directory = r"C:/Users/alexa/OneDrive - Politecnico di Milano/Desktop/PHD_python/SPM_pimionet/SPM_PIMIONet_new/TimeComplexityAnalysis_SPM/Padé approximation"
    filepath = os.path.join(directory, "pybamm_sol_data.mat")
    plant = loadmat(filepath)
    
    t = np.squeeze(plant['t'])
    dt = np.mean(np.diff(t))
    
    rn = np.squeeze(plant['rn'][:, 0, 0])
    cn_true = np.squeeze(plant['cn'][:, 0, :])
    rp = np.squeeze(plant['rp'][:, 0, 0])
    cp_true = np.squeeze(plant['cp'][:, 0, :])
    
    param = {}
    param['dt'] = dt
    param['neg'] = {}
    param['pos'] = {}
    
    param['neg']['D'] = 3.9e-14
    param['pos']['D'] = 1.0e-13
    param['neg']['cmax'] = 2.4983e4
    param['pos']['cmax'] = 5.1218e4
    param['neg']['R'] = 1e-5
    param['pos']['R'] = 1e-5
    param['neg']['eps'] = 0.6
    param['pos']['eps'] = 0.5
    param['neg']['L'] = 0.0001
    param['pos']['L'] = 0.0001
    
    param['F'] = 96485.33212
    param['T'] = 298.15
    param['R'] = 8.314472
    param['ce0'] = 1000.0
    param['W'] = 0.207
    param['H'] = 0.137
    param['I1C'] = 0.680616
    
    param['x0'] = 0.18322346594476283
    param['x100'] = 0.9493038520217835
    param['y0'] = 0.9610241419672079
    param['y100'] = 0.5126064431891388
    

    xnew_neg = np.linspace(0, param['neg']['R'], 100)
    cn_true_int_list = [
        interp1d(rn, cn_true[:, i], kind='linear', fill_value='extrapolate')(xnew_neg)
        for i in range(len(t))
    ]
    cn_true_int = np.column_stack(cn_true_int_list)
    
    xnew_pos = np.linspace(0, param['pos']['R'], 100)
    cp_true_int_list = [
        interp1d(rp, cp_true[:, i], kind='linear', fill_value='extrapolate')(xnew_pos)
        for i in range(len(t))]
    cp_true_int = np.column_stack(cp_true_int_list)
    
    # Declare symbolic variables (do not overwrite these later)
    D_sym, R_sym, s_sym, x = sp.symbols('D R s x')
    
    # Define the original transfer function Gtrue_x
    Gtrue_x = (R_sym/D_sym) / (1 - x * sp.coth(x))
    
    # Set the Padé approximant order (for example, testing orders 1 and 2)
    N_list = np.append(np.array(1), np.arange(5,45,5))
    N_list.astype(np.int64)
    i = 0
    
    n_it = 4
    
    with open('Gpade_list.pkl', 'rb') as file:
        Gpade_s = pickle.load(file)
        
    ct_avg = []
    err_arr = []

    cn0 = np.mean(plant['cn'][:, 0, 0])
    cp0 = np.mean(plant['cp'][:, 0, 0])
    c0 = np.array([cn0, cn0, cp0, cp0])
    
    for n_order in N_list:
        
        print(n_order)
        
        # Compute state-space realizations (for both negative and positive electrodes)
        An, Bn, Cn, Dn = pade_state_space_realization(param['neg'], Gpade_s[i])
        Ap, Bp, Cp, Dp = pade_state_space_realization(param['pos'], Gpade_s[i])
        
        # Rename the numeric state vector as x_state (instead of x) to avoid conflicts.
        x_state = np.zeros((2 * An.shape[0], len(t)))
        y = np.zeros((1, len(t)))
        u = np.squeeze(plant['I'])
        
        time_step = np.array([0, t[1]]) # --> t[1] is teh impsoed time step
        
        ct = []
        
        for _ in range(n_it):
            
            t0 = time()
            
            # Set first column of c to zeros.
            c= np.array([[0, 0, 0, 0]]).T

            x10 = np.linalg.lstsq(Cn, c[0:2, 0], rcond=None)
            x20 = np.linalg.lstsq(Cp, c[2:4, 0], rcond=None)
            x = np.concatenate((x10[0],x20[0]), axis=0)
                    
            for j in range(1,len(t)):
                x_p, c_p = stateFcn(x, u[0], time_step, param, Gpade_s[i]) # [0,:] is the k-th step, [:,1] is the k+1 -th step
                c = np.hstack((c, c_p[:,-1:]))
                x = x_p[:,-1]
            
            ct.append((time() - t0)/len(t) *1e3)
        
        ct_avg.append(np.mean(ct))
        # c += c0[:, None]  # add initial concentration to computed profile

        i = i+1
        
        # Pre-allocate arrays for averages and estimated voltage.
        # (Assuming t is a 1D numpy array.)
        cn_avg = np.zeros_like(t)
        cp_avg = np.zeros_like(t)
        
        # Loop over time indices (MATLAB indices 1:length(t) become Python 0:len(t))
        for k in range(len(t)):

            cn_avg[k] = (3 / (param['neg']['R']**3)) * np.dot(cn_true_int[:, k], rn**2) * np.mean(np.diff(rn))
            cp_avg[k] = (3 / (param['pos']['R']**3)) * np.dot(cp_true_int[:, k], rp**2) * np.mean(np.diff(rp))

        # Neg surface
        err = (cn_true_int[-1, :] - (c[0, :] + c0[0])) / param['neg']['cmax']
        en_surf = np.linalg.norm(err)
        
        # Pos surface
        err = (cp_true_int[-1, :] - (c[2, :] + c0[2])) / param['pos']['cmax']
        ep_surf = np.linalg.norm(err)
        
        # Neg avg
        err = (cn_avg - (c[1, :] + c0[1])) / param['neg']['cmax']
        en_avg = np.linalg.norm(err)
        
        # Pos avg
        err = (cp_avg - (c[3, :] + c0[3])) / param['pos']['cmax']
        ep_avg = np.linalg.norm(err)
        
        # Average 
        error_avg = np.mean((en_avg, ep_avg, en_surf, ep_surf))
        
        err_arr.append(error_avg)
        
        
    # Normalize ct_avg and err_arr to the [0, 1] range.
    ct_n = (ct_avg - np.min(ct_avg)) / (np.max(ct_avg) - np.min(ct_avg))
    err_n = (err_arr - np.min(err_arr)) / (np.max(err_arr) - np.min(err_arr))
    
    # Compute the optimality function (as sum of normalized ct_avg and err_arr).
    opt_fcn = ct_n + err_n
    
    # Compute x as 2 times the number of state variables for each case.
    x_plot = N_list
    
    # Call the plotting function.
    plot_results(x_plot, ct_avg, err_arr, opt_fcn)
    
    # # Save the array
    np.save("PADE_tuple2.npy", (x_plot, ct_avg, err_arr, opt_fcn))
        
