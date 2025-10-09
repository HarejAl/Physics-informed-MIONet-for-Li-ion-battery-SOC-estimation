"""
Created on Wed Feb 26 11:40:21 2025

@author: lorenzo
"""

import os
import numpy as np
import sympy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae
from scipy.interpolate import interp1d

# Custom function to compute the [n/n] Padé approximant using the standard formulation.
def pade_approximant(Gtrue_x, n):
    
    # We need the series of f(x)=x^2*Gtrue_x to order 2*n (i.e. 2n+1 terms)
    n = 2*n
    order = (2 * n + 1)
    
    # Multiply by x^2 to cancel the pole at x=0: f(x)=x^2*Gtrue_x
    f = sp.simplify(x**2 * Gtrue_x)

    # Obtain the Taylor series expansion of f up to order 2*n+1 and remove the big-O term.
    series_poly = sp.series(f, x, 0, order).removeO()
    
    # Convert the series to a polynomial in x
    poly   = sp.Poly(series_poly, x)
    coeffs = poly.all_coeffs()
    # The coefficients come in descending order, so reverse them
    coeffs = list(reversed(coeffs))
    # Ensure we have exactly 2*n+1 coefficients (i.e. from x^0 to x^(2n))
    # while len(coeffs) < 2*n + 1:
    #     coeffs.append(0)
    
    # coeffs is a list: c_0, c_1, ..., c_{2n}
    # Solve for b_1,..., b_n from:
    # for i = 1,..., n:  sum_{j=1}^n b_j * c_{n+i-j} = -c_{n+i}

    A = sp.Matrix([[coeffs[n + i - j] for j in range(1, n+1)] for i in range(1, n+1)])
    b_vec = sp.Matrix([-coeffs[n + i] for i in range(1, n+1)])
    sol = sp.linsolve((A, b_vec))
    b_sol = list(sol)[0]
    # b_0 is 1
    b_coeffs = [sp.Integer(1)] + list(b_sol)
    
    # Instead of nested loops, build a (n+1)x(n+1) lower triangular Toeplitz matrix C,
    # where C[k, j] = coeffs[k - j] if j <= k else 0.
    C = sp.Matrix([[coeffs[k - j] if j <= k else 0 for j in range(n+1)] for k in range(n+1)])
    b_vec_full = sp.Matrix(b_coeffs)
    
    # Compute numerator coefficients using matrix multiplication
    a_coeffs_vec = C * b_vec_full
    a_coeffs = [sp.simplify(a_coeffs_vec[i]) for i in range(n+1)]
    
    # Build the numerator and denominator polynomials
    P_poly = sum(a_coeffs[i] * x**i for i in range(n+1))
    Q_poly = sum(b_coeffs[i] * x**i for i in range(n+1))
    
    return sp.simplify(P_poly), sp.simplify(Q_poly)


def plot_pade_approx(Gtrue, Gapprox, param, order):
    """
    Plots the frequency response (magnitude in dB and phase in degrees) of
    the true transfer function Gtrue and its Padé approximant Gapprox.
    
    Parameters:
      Gtrue   : sympy expression for the true transfer function in variable s.
      Gapprox : sympy expression for the Padé approximant in variable s.
      param   : dictionary with at least the keys 'D' and 'R'.
                (For example, you can pass param['neg'] from your parameter dictionary.)
      order   : order of the Padé approximant (used only for legend text).
    """
    # Create a logarithmically spaced frequency array (rad/s)
    omega = np.logspace(-5, 3, num=50)
    
    # We assume that param is a dictionary with keys 'D' and 'R'. 
    # (In your main script you may use param['neg'] if needed.)
    D_val = param['D']
    R_val = param['R']
    
    # In the MATLAB code a variable m is substituted as 1/param.D.
    # Our expressions don't include m, but if needed, add: m_val = 1/D_val
    subs_dict = {sp.symbols('D'): D_val, sp.symbols('R'): R_val}
    
    # For the true transfer function, we need to substitute x = R*sqrt(s/D).
    # (This is consistent with your earlier conversion.)
    s = sp.symbols('s')
    x = sp.symbols('x')
    Gtrue_s = sp.simplify(Gtrue.subs(x, R_val*sp.sqrt(s/D_val)))
    
    # Substitute D and R values into the symbolic expressions.
    Gtrue_s_num = sp.simplify(Gtrue_s.subs(subs_dict))
    Gapprox_s_num = sp.simplify(Gapprox.subs(subs_dict))
    
    # Create numerical functions for evaluation (using numpy as backend)
    Gtrue_func = sp.lambdify(s, Gtrue_s_num, 'numpy')
    Gapprox_func = sp.lambdify(s, Gapprox_s_num, 'numpy')
    
    # Evaluate both transfer functions along the jω axis: s = 1j*omega
    s_vals = 1j * omega
    G_true_jw = Gtrue_func(s_vals)
    G_approx_jw = Gapprox_func(s_vals)
    
    # Compute magnitude in dB and phase in degrees.
    G_true_mag = 20 * np.log10(np.abs(G_true_jw))
    G_true_phase = np.degrees(np.angle(G_true_jw))
    
    G_approx_mag = 20 * np.log10(np.abs(G_approx_jw))
    G_approx_phase = np.degrees(np.angle(G_approx_jw))
    
    # Construct the legend name for the Padé approximant
    order = int(order)
    if order == 1:
        name = f"{order}-st order Padè approx."
    elif order == 2:
        name = f"{order}-nd order Padè approx."
    elif order == 3:
        name = f"{order}-rd order Padè approx."
    else:
        name = f"{order}-th order Padè approx."
    
    # Plot the magnitude and phase responses
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # Magnitude plot (top)
    axs[0].semilogx(omega, G_true_mag, label='True', linewidth=1)
    axs[0].semilogx(omega, G_approx_mag, label=name, linewidth=1)
    axs[0].set_ylabel('Magnitude (dB)')
    axs[0].set_title('Frequency Response of G(s) = C_s(s)/U(s)')
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[0].legend(loc='best')
    
    # Phase plot (bottom)
    axs[1].semilogx(omega, G_true_phase, label='True', linewidth=1)
    axs[1].semilogx(omega, G_approx_phase, label=name, linewidth=1)
    axs[1].set_xlabel('Frequency (rad/s)')
    axs[1].set_ylabel('Phase (deg)')
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

def pade_state_space_realization(param, Gpade):
    
    """
    Compute a state-space realization (A, B, C, D) from the Padé approximant Gpade.
    
    Parameters
    ----------
    param : dict
        Dictionary with at least keys 'D' and 'R'. (These are used for substitution.)
    Gpade : sympy expression
        The Padé approximant transfer function, expressed symbolically in s.
    
    Returns
    -------
    A, B, C, D : numpy arrays
        State-space matrices.
    """
    
    # Separate the symbolic expression into numerator and denominator
    num_s, den_s = sp.fraction(Gpade_s)

    # Convert the numerator and denominator into polynomial objects in 's'
    D, R, s = sp.symbols('D R s')
    num_coef = sp.Poly(num_s, s).all_coeffs()
    num_coef = list(reversed(num_coef))
    den_coef = sp.Poly(den_s, s).all_coeffs()
    den_coef = list(reversed(den_coef))

    # Normalize by dividing by the last element of the denominator coefficients.
    den_last = den_coef[1]
    num_coef_norm = [sp.simplify(c/den_last) for c in num_coef]
    den_coef_norm = [sp.simplify(c/den_last) for c in den_coef]

    # Extract coefficients and convert them to floats (or another appropriate numeric type)
    subs_dict = {D:  param['D'], R:  param['R']}
    num_coeffs = [float(co.subs(subs_dict).evalf()) for co in num_coef_norm]
    den_coeffs = [float(co.subs(subs_dict).evalf()) for co in den_coef_norm]
    
    n = len(num_coeffs)
    
    # Create numeric state-space matrices:
    # A1: n x n, B1: n x 1, C1: 2 x n, D1: 2 x 1.
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
    
def stateFcn(x, u, t, param, Gpade):
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
    An, Bn, Cn, Dn = pade_state_space_realization(param['neg'], Gpade_s)
    Ap, Bp, Cp, Dp = pade_state_space_realization(param['pos'], Gpade_s)
    
    # Extract additional parameters.
    dt      = param['dt']
    F_const = param['F']
    A       = param['H'] * param['W']
    an      = 3.0 * param['neg']['eps'] / param['neg']['R']
    ap      = 3.0 * param['pos']['eps'] / param['pos']['R']
    Ln      = param['neg']['L']
    Lp      = param['pos']['L']
    
    # Time span.
    t_span = (t[0], t[-1])
    
    # Define input functions.
    def un(t_val):
        return (u / A / F_const / an / Ln) if t_val >= 0 else 0.0
    def up(t_val):
        return -(u / A / F_const / ap / Lp) if t_val >= 0 else 0.0

    # ODE for the negative electrode:
    #    dxn/dt = An*xn + Bn*un(t)
    def ode_n(t_val, x_val, xdot_val):
        return An @ x_val + Bn.flatten() * un(t_val) - xdot_val
    
    yn0  = x[:int(np.size(x,0)/2)]
    yn10 = np.zeros(int(np.size(x,0)/2))
    sol_n = solve_dae(ode_n, t_span, yn0, yn10, atol=1e-2, rtol=1e-2, method="Radau", t_eval=t, max_step=10)
    xn    = sol_n.y
    cn    = np.matmul(Cn,xn)

    # ODE for the positive electrode:
    #    dxp/dt = Ap*xp + Bp*up(t)
    def ode_p(t_val, x_val, xdot_val):
        return Ap @ x_val + Bp.flatten() * up(t_val) - xdot_val

    yp0  = x[int(np.size(x,0)/2):]
    yp10 = np.zeros(int(np.size(x,0)/2))
    sol_p = solve_dae(ode_p, t_span, yp0, yp10, atol=1e-2, rtol=1e-2, method="Radau", t_eval=t, max_step=10)
    xp    = sol_p.y
    cp    = np.matmul(Cp,xp)

    x = np.vstack((xn, xp))
    c = np.vstack((cn, cp))
    
    # Return the solution array (each column corresponds to the state at a time point).
    return x, c

def ocp_n(sto):
    """
    Open-circuit potential for the negative electrode.
    
    Parameters
    ----------
    sto : float or np.array
        Stoichiometry value.
        
    Returns
    -------
    Un : float or np.array
        The negative electrode open-circuit potential.
    """
    Un = (0.194 +
          1.5 * np.exp(-120.0 * sto) +
          0.0351 * np.tanh((sto - 0.286) / 0.083) -
          0.0045 * np.tanh((sto - 0.849) / 0.119) -
          0.035  * np.tanh((sto - 0.9233) / 0.05) -
          0.0147 * np.tanh((sto - 0.5) / 0.034) -
          0.102  * np.tanh((sto - 0.194) / 0.142) -
          0.022  * np.tanh((sto - 0.9) / 0.0164) -
          0.011  * np.tanh((sto - 0.124) / 0.0226) +
          0.0155 * np.tanh((sto - 0.105) / 0.029))
    return Un

def ocp_p(sto):
    """
    Open-circuit potential for the positive electrode.
    
    Parameters
    ----------
    sto : float or np.array
        Stoichiometry value.
        
    Returns
    -------
    Up : float or np.array
        The positive electrode open-circuit potential.
    """
    stretch = 1.062
    sto = stretch * sto
    Up = (2.16216 +
          0.07645 * np.tanh(30.834 - 54.4806 * sto) +
          2.1581  * np.tanh(52.294 - 50.294 * sto) -
          0.14169 * np.tanh(11.0923 - 19.8543 * sto) +
          0.2051  * np.tanh(1.4684 - 5.4888 * sto) +
          0.2531  * np.tanh((-sto + 0.56478) / 0.1316) -
          0.02167 * np.tanh((sto - 0.525) / 0.006))
    return Up

def exchange_current_density_n(cs, ce, T, param):
    """
    Computes the exchange current density for the negative electrode.
    
    Parameters
    ----------
    cs : float
        Negative surface concentration.
    ce : float
        Electrolyte concentration.
    T : float
        Temperature (K).
    param : dict
        Dictionary containing parameters. Expected keys:
          - "R": universal gas constant (or electrode-specific constant)
          - "neg": dictionary with key "cmax" (maximum concentration for negative electrode)
    
    Returns
    -------
    jn0 : float
        Negative electrode exchange current density.
    """
    mref = 2e-5      # A/m2 (m3/mol)^1.5, includes reference concentrations
    E_r  = 37480     # Activation energy (J/mol)
    R_const = param["R"]
    Tref = 298.15
    arrhenius = np.exp(E_r / R_const * (1/Tref - 1/T))
    jn0 = mref * arrhenius * np.sqrt(ce) * np.sqrt(cs) * np.sqrt(param["neg"]["cmax"] - cs)
    return jn0

def exchange_current_density_p(cs, ce, T, param):
    """
    Computes the exchange current density for the positive electrode.
    
    Parameters
    ----------
    cs : float
        Positive surface concentration.
    ce : float
        Electrolyte concentration.
    T : float
        Temperature (K).
    param : dict
        Dictionary containing parameters. Expected keys:
          - "R": universal gas constant (or electrode-specific constant)
          - "pos": dictionary with key "cmax" (maximum concentration for positive electrode)
    
    Returns
    -------
    jp0 : float
        Positive electrode exchange current density.
    """
    mref = 6e-7      # A/m2 (m3/mol)^1.5, includes reference concentrations
    E_r  = 39570     # Activation energy (J/mol)
    R_const = param["R"]
    Tref = 298.15
    arrhenius = np.exp(E_r / R_const * (1/Tref - 1/T))
    jp0 = mref * arrhenius * np.sqrt(ce) * np.sqrt(cs) * np.sqrt(param["pos"]["cmax"] - cs)
    return jp0

def measFcn(x, u, T, param):
    """
    Measurement function that computes the cell voltage.
    
    Parameters
    ----------
    x : array_like
        State vector. Expected to contain at least:
          x[0] : negative surface concentration (cs_n)
          x[2] : positive surface concentration (cs_p)
    u : float
        Input current.
    T : float
        Temperature (K).
    param : dict
        Dictionary containing model parameters. Expected keys:
          "R", "F", "ce0", "W", "H", "x0", "x100", "y0", "y100",
          and sub-dictionaries "neg" and "pos" with keys:
              "eps", "R", "L", "cmax"

    Returns
    -------
    y : float
        The computed cell voltage.
    """
    # Extract surface concentrations (MATLAB: x(1) and x(3))
    cs_n = x[0]
    cs_p = x[2]
    
    # Compute scaling factor
    RT_F = param["R"] * T / param["F"]
    ce = param["ce0"]  # assumed constant
    
    # Compute exchange current densities
    j0_n = exchange_current_density_n(cs_n, ce, T, param)
    j0_p = exchange_current_density_p(cs_p, ce, T, param)
    
    # Geometry and reaction area factors
    A  = param["W"] * param["H"]
    an = 3.0 * param["neg"]["eps"] / param["neg"]["R"]
    Ln = param["neg"]["L"]
    ap = 3.0 * param["pos"]["eps"] / param["pos"]["R"]
    Lp = param["pos"]["L"]
    
    # Reaction current densities
    j_n = (u / A) / (an * Ln)
    j_p = -(u / A) / (ap * Lp)
    
    # Overpotentials via inverse hyperbolic sine (asinh)
    eta_n = 2 * RT_F * np.arcsinh(j_n / (2 * j0_n))
    eta_p = 2 * RT_F * np.arcsinh(j_p / (2 * j0_p))
    
    # Compute potentials
    phi_s_n = 0
    phi_e = - eta_n - ocp_n(cs_n / param["neg"]["cmax"])
    phi_s_p = eta_p + phi_e + ocp_p(cs_p / param["pos"]["cmax"])
    
    # Cell voltage
    V = phi_s_p - phi_s_n
    return V

## MAIN
if __name__ == '__main__':

    # Load the MATLAB .mat file (assumed to be "pybamm_sol_data.mat")
    directory = r"C:/Users/alexa/OneDrive - Politecnico di Milano/Desktop/PHD_python/SPM_pimionet/SPM_PIMIONet_new/TimeComplexityAnalysis_SPM/Padé approximation"
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
    
    param['x0'] = 0.18322346594476283  # 0%   SOC negative cathode stochiometry
    param['x100'] = 0.9493038520217835 # 100% SOC negative cathode stochiometry
    param['y0'] = 0.9610241419672079   # 0%   SOC positive cathode stochiometry
    param['y100'] = 0.5126064431891388 # 100% SOC positive cathode stochiometry
    
    # SOC calculation functions:
    soc_p = lambda x: (x - param['y100']) / (param['y0'] - param['y100'])
    soc_n = lambda x: (x - param['x0']) / (param['x100'] - param['x0'])
    
    # Interpolate true solution (from PyBaMM) to allow comparison:
    xnew_neg = np.linspace(0, param['neg']['R'], 100)  # new spatial grid (100 points)
    # For each time index, interpolate the corresponding column of cn_true
    cn_true_int_list = [
        interp1d(rn, cn_true[:, i], kind='linear', fill_value='extrapolate')(xnew_neg)
        for i in range(len(t))
    ]
    # Stack the interpolated columns into a 2D array with shape (100, len(t))
    cn_true_int = np.column_stack(cn_true_int_list)
    
    xnew_pos = np.linspace(0, param['pos']['R'], 100)  # new spatial grid for positive electrode
    cp_true_int_list = [
        interp1d(rp, cp_true[:, i], kind='linear', fill_value='extrapolate')(xnew_pos)
        for i in range(len(t))
    ]
    cp_true_int = np.column_stack(cp_true_int_list)
    
    # Declare symbols
    D, R, s, x = sp.symbols('D R s x')
    
    # Define the original transfer function Gtrue_x
    Gtrue_x = (R/D) / (1 - x * sp.coth(x))
    
    # Set the Padé approximant order (param.n = 10)
    n = 1
    
    # Compute the Padé approximant for f(x) as a [n/n] approximant.
    P, Q = pade_approximant(Gtrue_x, n)
    
    # Recall that f(x)=x^2 * Gtrue_x, so the approximant for Gtrue_x is (P/Q)/x^2.
    Gpade_x = sp.simplify((P / Q) / x**2)
    
    # Now substitute x = R*sqrt(s/D) to obtain the rational approximant in s.
    Gpade_s = sp.simplify(Gpade_x.subs(x, R * sp.sqrt(s/D)))
    Gtrue_s = sp.simplify(Gtrue_x.subs(x, R * sp.sqrt(s/D)))

    ## Plotting the TFs 
    # plot_pade_approx(Gtrue_s, Gpade_s, param['neg'], order=n)
    # plot_pade_approx(Gtrue_s, Gpade_s, param['pos'], order=n)

    An, Bn, Cn, Dn = pade_state_space_realization(param['neg'], Gpade_s)
    Ap, Bp, Cp, Dp = pade_state_space_realization(param['pos'], Gpade_s)
    
    # Assume t is a 1D NumPy array.
    x = np.zeros((2 * An.shape[0], len(t)))  # 2*(number of rows of An)-by-len(t) array.
    y = np.zeros((1, len(t)))           # 1-by-len(t) array.
    u = np.squeeze(plant['I'])         # Assuming plant.I is defined appropriately.
    
    # Set initial value:
    # In MATLAB, plant.cn(:,1,1) means "all rows, first column, first page"
    # Python uses 0-based indexing, so:
    cn0 = np.mean(plant['cn'][:, 0, 0])
    cp0 = np.mean(plant['cp'][:, 0, 0])
    c0 = np.array([
        cn0, # Initial c on surf
        cn0, # Initial avg c
        cp0, # Initial c on surf
        cp0 # Initial avg c
    ])

    # Set first column of c to zeros.
    c= np.array([[0, 0, 0, 0]]).T
    
    """
    c = [c_neg_surface,
         c_neg_avg,
         c_pos_surface,
         c_pos_avg]
    """
    
    # For x, we solve two linear systems:
    # then stack the solutions vertically.
    x10 = np.linalg.lstsq(Cn, c[0:2, 0], rcond=None)
    x20 = np.linalg.lstsq(Cp, c[2:4, 0], rcond=None)
    x = np.concatenate((x10[0],x20[0]), axis=0).reshape(-1,1)
    
    time_step = np.array([0, t[1]]) # --> t[1] is teh impsoed time step
    
    for j in range(1,len(t)):
        x_p, c_p = stateFcn(x[:, -1], u[0], time_step, param, Gpade_s) # --> stateFcn returns: (x_p, c_p): x_p = [x_p(t=0), x_p(t=Dt)], c_p = [c_p(t=0), c_p(t=Dt)]
        c = np.hstack((c, c_p[:,-1:]))
        x = x_p[:,-1:]
    
    # x, c = stateFcn(x[:, 0], u[0], t, param, Gpade_s)
    
    # Pre-allocate arrays for averages and estimated voltage.
    # (Assuming t is a 1D numpy array.)
    cn_avg = np.zeros_like(t)
    cp_avg = np.zeros_like(t)
    y_est   = np.zeros_like(t)  # estimated voltage
    
    # Loop over time indices (MATLAB indices 1:length(t) become Python 0:len(t))
    for k in range(len(t)):
        # For the negative electrode: 
        #   cn_avg(k) = (3/(param.neg.R^3))*dot(cn_true_int(:,k), rn.^2)*mean(diff(rn))
        cn_avg[k] = (3 / (param['neg']['R']**3)) * np.dot(cn_true_int[:, k], rn**2) * np.mean(np.diff(rn))
        # For the positive electrode:
        cp_avg[k] = (3 / (param['pos']['R']**3)) * np.dot(cp_true_int[:, k], rp**2) * np.mean(np.diff(rp))
        # Estimated voltage: MATLAB: y(k) = measFcn(c(:,k)+c0, u(1),298.15,param)
        # Here we assume u is an array; use its first element.
        y_est[k] = measFcn(c[:, k] + c0, u[0], 298.15, param)

    # Use k = len(t)
    k = len(t)
    
    # --------------------
    # Figure 1: 2x2 tiled layout
    # MATLAB figure size [19 14] cm -> (19/2.54, 14/2.54) inches.
    fig1, axs = plt.subplots(2, 2, figsize=(19/2.54, 14/2.54))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    width = 1.5  # used to scale linewidth
    
    # Subplot 1: Negative particle Li+ surface concentration
    ax = axs[0, 0]
    ax.plot(t/60, cn_true_int[-1, :], linestyle='-', linewidth=width*2, color=(0.7,0.7,0.7))
    ax.plot(t/60, c[0, :] + c0[0], linestyle='--', linewidth=width, color=(0.1,0.1,0.1))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Amplitude (mol/m^3)")
    ax.set_title("Negative particle Li^+ surface concentration")
    error = (cn_true_int[-1, :] - (c[0, :] + c0[0]) ) / cn_true_int[-1, :] * 100
    ax2 = ax.twinx()
    ax2.plot(t/60, error)
    ax2.set_ylabel("Error (%)")
    ax.autoscale(tight=True)
    err = (cn_true_int[-1, :] - (c[0, :] + c0[0])) / param['neg']['cmax']
    en_surf = np.linalg.norm(err)
    print('Negative surface Fr. norm error {}'.format(en_surf))
    
    # Subplot 2: Positive particle Li+ surface concentration
    ax = axs[0, 1]
    ax.plot(t/60, cp_true_int[-1, :], linestyle='-', linewidth=width*2, color=(0.7,0.7,0.7))
    ax.plot(t/60, c[2, :] + c0[2], linestyle='--', linewidth=width, color=(0.1,0.1,0.1))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Amplitude (mol/m^3)")
    ax.set_title("Positive particle Li^+ surface concentration")
    error = (cp_true_int[-1, :] - (c[2, :] + c0[2]) ) / cp_true_int[-1, :] * 100
    ax2 = ax.twinx()
    ax2.plot(t/60, error)
    ax2.set_ylabel("Error (%)")
    ax.autoscale(tight=True)
    
    err = (cp_true_int[-1, :] - (c[2, :] + c0[2])) / param['pos']['cmax']
    ep_surf = np.linalg.norm(err)
    print('Positive surface Fr. norm error {}'.format(ep_surf))
    
    # Subplot 3: Negative particle Li+ average concentration
    ax = axs[1, 0]
    ax.plot(t/60, cn_avg, linestyle='-', linewidth=width*2, color=(0.7,0.7,0.7))
    ax.plot(t/60, c[1, :] + c0[1], linestyle='--', linewidth=width, color=(0.1,0.1,0.1))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Amplitude (mol/m^3)")
    ax.set_title("Negative particle Li^+ average concentration")
    error = (cn_avg - (c[1, :] + c0[1]) ) / cn_avg * 100
    ax2 = ax.twinx()
    ax2.plot(t/60, error)
    ax2.set_ylabel("Error (%)")
    ax.autoscale(tight=True)
    
    err = (cn_avg - (c[1, :] + c0[1])) / param['neg']['cmax']
    en_avg = np.linalg.norm(err)
    print('Negative average Fr. norm error {}'.format(en_avg))
    
    # Subplot 4: Positive particle Li+ average concentration
    ax = axs[1, 1]
    ax.plot(t/60, cp_avg, linestyle='-', linewidth=width*2, color=(0.7,0.7,0.7), label='true')
    ax.plot(t/60, c[3, :] + c0[3], linestyle='--', linewidth=width, color=(0.1,0.1,0.1), label='estimated')
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Amplitude (mol/m^3)")
    ax.set_title("Positive particle Li^+ average concentration")
    ax.autoscale(tight=True)
    error = (cp_avg - (c[3, :] + c0[3]) ) / cp_avg * 100
    ax2 = ax.twinx()
    ax2.plot(t/60, error)
    ax2.set_ylabel("Error (%)")
    # Place legend in the "north" (top) of the tiled layout.
    # Here, we add a common legend to the entire figure.
    handles, labels = ax.get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center', ncol=2)
    
    err = (cp_avg - (c[3, :] + c0[3])) / param['pos']['cmax']
    ep_avg = np.linalg.norm(err)
    print('Positive avergae Fr. norm error {}'.format(ep_avg))
    
    # --------------------
    # Figure 2: Terminal Voltage
    fig2, ax2 = plt.subplots(figsize=(9/2.54, 7/2.54))
    ax2.plot(t/60, np.squeeze(plant['V']), linestyle='-', linewidth=width*2, color=(0.7,0.7,0.7), label='true')
    ax2.plot(t/60, y_est[:k], linestyle='--', linewidth=width, color=(0.1,0.1,0.1), label='estimated')
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Amplitude (V)")
    ax2.set_title("Terminal Voltage")
    ax2.autoscale(tight=True)
    ax2.legend(loc='upper right', ncol=1)
    
    plt.show()
    
    error = np.mean((en_avg, ep_avg, en_surf, ep_surf))
    print('Average error {}'.format(error))
    

