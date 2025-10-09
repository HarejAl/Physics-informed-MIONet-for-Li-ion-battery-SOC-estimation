"""
Battery simulation and observer utility library
-----------------------------------------------

This module contains helper classes and methods used by the main simulation script
for modelling, solving, and post-processing lithium-ion battery systems. It provides:

1. **MIONet class (torch.nn.Module)**
   - Neural network architecture for the physics-informed MIONet surrogate model.
   - Forward pass combines two branch networks (initial concentration, current) 
     and one trunk network (time/space coordinates) via element-wise multiplication 
     and summation.

2. **PINN class**
   - Wraps a trained MIONet for convenient predictions during observer simulations.
   - Handles interpolation if the number of spatial discretization points `N` in 
     the current simulation differs from the one used during training.
   - Provides `predict()` method to advance electrode surface concentrations given:
       * `Cs_0`  – previous surface concentration distribution
       * `dT`    – time step
       * `I`     – applied current

3. **plant class**
   - Represents the "plant" (reference battery model) solved via PyBaMM.
   - Provides current profile generators:
       * `UDDS(t_eval, CR)` – Urban Dynamometer Driving Schedule profile.
       * `CC(t_eval, CR)`   – Constant-current profile.
   - Provides electrochemical utilities:
       * `SOC_test()` – computes electrode stoichiometries at SOC = 0% and 100% 
         using PyBaMM's ElectrodeSOHSolver.
       * `ref_solution()` – solves the plant DFN model (with or without lumped 
         thermal model) for a given initial SOC and current profile.
       * `SOC_calculator()` – computes pack SOC from simulated electrode concentrations
         by integrating over particle radius and averaging both electrodes.

4. **observer_mionet class**
   - Helper for MIONet-based observers.
   - Loads an OCP–SOC curve (from `soc_ocp_MOH.npy`) to compute terminal voltage 
     from electrode stoichiometries and applied current.
   - Includes kinetic and transport parameters from the PyBaMM parameter set to 
     compute overpotentials and approximate voltage corrections.
   - Provides:
       * `Voltage(y, i_cell)` – maps state vector `y` + current to terminal voltage.
       * `SOC_test()` – same as in `plant` (electrode stoichiometries at SOC limits).
       * `SOC_calculator()` – computes SOC from electrode concentration vectors.

Usage
-----
In the main script, this module is imported as `utils` and used to:
- Initialise the plant with given parameters.
- Generate current profiles.
- Compute reference solutions for the DFN model.
- Instantiate MIONet observers with correct voltage mapping.
- Convert simulated concentrations to SOC for comparison and error analysis.

If run as `__main__`, the script will:
- Initialise a `plant` object with Mohtat2020 parameters.
- Generate a chosen current profile.
- Solve the DFN model (optionally with thermal model enabled).
- Plot the predicted voltage and SOC over time.

Dependencies:
- numpy, scipy, torch, pybamm, matplotlib
"""

import numpy as np
import pybamm 
import scipy
import torch 
import torch.nn as nn 
from scipy.interpolate import CubicSpline
pybamm.set_logging_level("ERROR")

def j0_pos_func(ce,cs,cs_pos_max,T):

    E_r  = 39570
     
    R    = 8.314462618
    F    = 96485.33212
    mref = 1 * 10 ** (-11) * F  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    Tref = 298.15 
    arrhenius =  np.exp(E_r/ R * (1/Tref - 1/T))
     
    jp0 = mref * arrhenius * np.sqrt(ce) * np.sqrt(np.max(cs, 0)) * np.sqrt(np.max(cs_pos_max - cs, 0))
    
    return jp0 
 
def j0_neg_func(ce,cs,cs_neg_max,T):
    
    E_r  = 5000
     
    R    = 8.314462618
    F    = 96485.33212
    m_ref = 1 * 10 ** (-11) * F  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    Tref = 298.15 
    arrhenius =  np.exp(E_r/ R * (1/Tref - 1/T))
     
    return m_ref * arrhenius * ce**0.5 * np.sqrt(np.max(cs, 0)) * np.sqrt(np.max(cs_neg_max - cs, 0))

class MIONet(nn.Module): 
        
    def forward(self, coordinate, input1, input2): # Process each input stream 
        
        out1 = self.branch1(input1) 
        out2 = self.branch2(input2)
        out_trunk = self.trunk(coordinate) 
        # Concatenate outputs from both branches 
        dot_product = torch.mul(torch.mul(out1,out2),out_trunk)
        output = torch.sum(dot_product,1) 
        return output

class PINN():
    def __init__(self,NN_model, n):
        self.network = NN_model
        self.sensor_pts = 31 ## Depends on the trained model (b1 in training.py) --> = 31 for saved models
        self.N = n
        self.t = torch.tensor(np.ones([self.N,1]),dtype=torch.float32)*0.1 # In the saved model the time was scaled by 10. If you retrain a model then remove it since in the latest version it has been removed
        self.ksi = torch.tensor(np.linspace(0,1,self.N)[:,None],dtype=torch.float32)
        
    def predict(self, Cs_0, dT, I):
        
        if self.N != self.sensor_pts:
            Cs_0 = np.interp(np.linspace(0,1,self.sensor_pts), np.linspace(0,1,self.N), Cs_0)
        
        cs_0 =torch.tensor(Cs_0,dtype=torch.float32)
        i = torch.tensor(np.array([[I]]),dtype=torch.float32)
        
        t_ksi = torch.cat((self.t*dT,self.ksi),1)
        y = self.network(t_ksi,cs_0, i)
        
        return y.detach().numpy()

class plant():
    
    def __init__(self, param, N):
        self.param = param
        self.C = param['Nominal cell capacity [A.h]']
        self.N = N
        x0, x100, y0, y100 = self.SOC_test()
        self.x0 = x0
        self.x100 = x100
        self.y0 = y0
        self.y100 = y100

    def UDDS(self, t_eval, CR):
        
        """
        1 UDDS profile lasts for 1400 secs 
        N is the time i want to repeat the DST profile 
        """
        
        UDDS = scipy.io.loadmat('UDDS_profile.mat')
        t = UDDS['t'][:,0]
        c = UDDS['C'][:,0]
        
        T_end = t_eval[-1]
        N = T_end // 1370 + 1
        N = int(N)
    
        for i in range(N-1):
            c = np.concatenate((c,UDDS['C'][:,0]), axis = 0)
        
        t_aux = np.linspace(0,len(c)-1,len(c))   
        curr = np.interp(t_eval,t_aux,c)
        
        current_string = 'UDDS'
    
        return curr*CR*self.C, current_string

    def CC(self,t_eval, CR):
        curr = np.ones((t_eval.shape))*CR*self.C
        return curr, 'CC'

    
    def SOC_test(self):
        
        """
        This method needs to determine the stoichiometries of Li atoms in electrodes at soc = 0 and soc = 100%
        """

        # Determine stochiometries limits (using electrode state of health model)
        Vmin = self.param["Lower voltage cut-off [V]"] 
        Vmax = self.param["Upper voltage cut-off [V]"]
        self.param.set_initial_stoichiometries(1)
    
        # print('V_max {}; V_min {}'.format(Vmax,Vmin))
    
        parameter_values = pybamm.LithiumIonParameters()
        
        Q_n  = self.param.evaluate(parameter_values.n.Q_init)
        Q_p  = self.param.evaluate(parameter_values.p.Q_init)
        Q_Li = self.param.evaluate(parameter_values.Q_Li_particles_init)
    
        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(self.param, parameter_values)
        inputs = {"V_min": Vmin, "V_max": Vmax, "Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        esoh_sol = esoh_solver.solve(inputs)
    
        x_100 = esoh_sol["x_100"]
        y_100 = esoh_sol["y_100"]
        x_0   = esoh_sol["x_0"]
        y_0   = esoh_sol["y_0"]
        
        # x_0 = 0.0065
        # x_100 = 0.84
        # y_0 = 0.9651
        # y_100 = 0.0065
    
        
        return x_0, x_100, y_0, y_100
    
    def ref_solution(self, soc_0, current, t_eval, thermal_model = False):
        
        """
        This method computes the reference solution given initial soc and current
        """
        if thermal_model:
            batt_model = pybamm.lithium_ion.DFN(
                        options={
                            "particle": "Fickian diffusion",
                            "cell geometry": "arbitrary",
                            "thermal": "lumped",
                            "particle mechanics": "swelling only",
                        }
                        )
 
        else: 
            batt_model = pybamm.lithium_ion.DFN()
        
        self.param.set_initial_stoichiometries(soc_0)
        
        var = pybamm.standard_spatial_vars
        var_pts = {
            var.x_n: 10,
            var.x_s: 10,
            var.x_p: 10,
            var.r_n: self.N,
            var.r_p: self.N,
        }
        
        drive_cycle_power = np.column_stack([t_eval, current])
        experiment = pybamm.Experiment([pybamm.step.current(drive_cycle_power)])
        del drive_cycle_power
        
        sim = pybamm.Simulation(batt_model, parameter_values=self.param, var_pts=var_pts , experiment = experiment)
        solution = sim.solve(t_eval)
        
        ## Results
        time = solution["Time [min]"].entries # time_pb is automatically cutted at the cut off value 
        cs_pos_ref =  np.mean(solution["Positive particle concentration"](t_eval),axis=1)
        cs_neg_ref =  np.mean(solution["Negative particle concentration"](t_eval),axis=1)
        Temp = solution["X-averaged cell temperature [K]"](t_eval)
        voltage_ref = solution["Voltage [V]"](t_eval)
        
        y_ref = np.concatenate((cs_pos_ref,cs_neg_ref),axis=0)
        
        return t_eval/60, y_ref, voltage_ref, Temp
    
    def SOC_calculator(self,y):
        """
        This method computes the soc as the average soc of both electrodes
        Inputs: normalised concentration vector: y = [c_pos, c_neg].T
        
        Returns: SOC 
        """
        POS = y[:self.N]
        NEG = y[self.N:]
        
        x0, x100, y0, y100 = self.SOC_test()
                
        r = np.linspace(0,1,len(POS[:,0]))[:,None]; dr = r[1]; dr_2 = dr/2
        argument = np.ones(POS.shape)*(r-dr_2)**2*dr
        
        ## To compute the average concentration inside a spherical particle it is necessary to solve the integral INT[c(r)*dr]0,R / Volume

        integral_pos = 3*argument*POS
        pos_avg = np.sum(integral_pos,axis=0)
        
        integral_neg = 3*argument*NEG
        neg_avg = np.sum(integral_neg,axis=0)
        
        SOC_pos = (pos_avg - y0)/(y100-y0)
        SOC_neg = (neg_avg - x0)/(x100-x0)
        
        SOC = (SOC_pos + SOC_neg)/2  # Global SOC as the average of both electrodes
        
        return SOC*100

class observer_mionet():
    """
    There are some methods usefull for mioneet observer
    """
    
    def __init__(self, param, N):
        self.param = param
        self.C = param['Nominal cell capacity [A.h]']
        self.N = N
        x0, x100, y0, y100 = self.SOC_test()
        self.x0 = x0
        self.x100 = x100
        self.y0 = y0
        self.y100 = y100
        
        graphite_ocp_Enertech_Ai2020_data = pybamm.parameters.process_1D_data(
            "graphite_ocp_Enertech_Ai2020.csv")

        name, (x, y) = graphite_ocp_Enertech_Ai2020_data
        self.ocp_neg = CubicSpline(x[0], y)

        lico2_ocp_Ai2020_data = pybamm.parameters.process_1D_data(
            "lico2_ocp_Ai2020.csv")

        name, (x, y) = lico2_ocp_Ai2020_data
        self.ocp_pos= CubicSpline(x[0], y)

    
    def Voltage(self, y,I):
        i_cell = I/0.08149800001137754
        
        if np.ndim(y)>1:
            POS = y[:self.N, :]
            NEG = y[-self.N:, :]
        else:
            POS = y[:self.N]
            NEG = y[-self.N:]

        R = self.param['Ideal gas constant [J.K-1.mol-1]']
        F = self.param['Faraday constant [C.mol-1]']     

        cs_max_pos = self.param['Maximum concentration in positive electrode [mol.m-3]']
        cs_max_neg = self.param['Maximum concentration in negative electrode [mol.m-3]']
        
        c_e = self.param['Initial concentration in electrolyte [mol.m-3]']
        T = self.param['Ambient temperature [K]']

        
        cs_p_surf = POS[-1]*cs_max_pos; cs_n_surf = NEG[-1]*cs_max_neg
    
        j0_p = j0_pos_func(c_e,cs_p_surf,cs_max_pos,T)
        j0_n = j0_neg_func(c_e,cs_n_surf,cs_max_neg,T)
        
        epsilon_n = self.param['Negative electrode active material volume fraction']
        Rs_n = self.param['Negative particle radius [m]']
        a_n = 3*epsilon_n/Rs_n
        L_n = self.param['Negative electrode thickness [m]']
        
        epsilon_p = self.param['Positive electrode active material volume fraction']
        Rs_p = self.param['Positive particle radius [m]']
        a_p = 3*epsilon_p/Rs_p
        L_p = self.param['Positive electrode thickness [m]']
        
        W=self.param['Electrode width [m]'] # Electrode height [m]
        H=self.param['Electrode height [m]']
        
        j_n = i_cell / (L_n* a_n)
        j_p = -i_cell / (L_p * a_p)
        
        eta_neg = 2*R*T/F * np.arcsinh(j_n / (2 * j0_n))  
        eta_pos = 2*R*T/F * np.arcsinh(j_p / (2 * j0_p))
        
     

        OCP = self.ocp_pos(POS[-1]) - self.ocp_neg(NEG[-1])
        
        U = OCP  - eta_neg + eta_pos - i_cell/1800  # The last term represents an approximated voltage error between SPM and DFN --> A trial and error term that sligthly improves the performance of the observer

        
        return U
    
    def SOC_test(self):
        
        """
        This method needs to determine the stoichiometries of Li atoms in electrodes at soc = 0 and soc = 100%
        """

        # Determine stochiometries limits (using electrode state of health model)
        Vmin = self.param["Lower voltage cut-off [V]"] 
        Vmax = self.param["Upper voltage cut-off [V]"]
        self.param.set_initial_stoichiometries(1)
    
        # print('V_max {}; V_min {}'.format(Vmax,Vmin))
    
        parameter_values = pybamm.LithiumIonParameters()
        
        Q_n  = self.param.evaluate(parameter_values.n.Q_init)
        Q_p  = self.param.evaluate(parameter_values.p.Q_init)
        Q_Li = self.param.evaluate(parameter_values.Q_Li_particles_init)
    
        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(self.param, parameter_values)
        inputs = {"V_min": Vmin, "V_max": Vmax, "Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        esoh_sol = esoh_solver.solve(inputs)
    
        x_100 = esoh_sol["x_100"]
        y_100 = esoh_sol["y_100"]
        x_0   = esoh_sol["x_0"]
        y_0   = esoh_sol["y_0"]
        
        # x_0 = 0.0065
        # x_100 = 0.84
        # y_0 = 0.9651
        # y_100 = 0.0065
        
        return x_0, x_100, y_0, y_100
    
    def SOC_calculator(self,y):
        """
        This method computes the soc as the average soc of both electrodes
        Inputs: normalised concentration vector: y = [c_pos, c_neg].T
        
        Returns: SOC 
        """
        POS = y[:self.N]
        NEG = y[self.N:]
        
        x0, x100, y0, y100 = self.SOC_test()
                
        r = np.linspace(0,1,len(POS[:,0]))[:,None]; dr = r[1]; dr_2 = dr/2
        argument = np.ones(POS.shape)*(r-dr_2)**2*dr
        
        ## To compute the average concentration inside a spherical particle it is necessary to solve the integral INT[c(r)*dr]0,R / Volume

        integral_pos = 3*argument*POS
        pos_avg = np.sum(integral_pos,axis=0)
        
        integral_neg = 3*argument*NEG
        neg_avg = np.sum(integral_neg,axis=0)
        
        SOC_pos = (pos_avg - y0)/(y100-y0)
        SOC_neg = (neg_avg - x0)/(x100-x0)
        
        SOC = (SOC_pos + SOC_neg)/2  # Global SOC as the average of both electrodes
        
        return SOC*100

if __name__ == "__main__":
    param = pybamm.ParameterValues("Ai2020")
    N = 20
    lib = plant(param, N)
    
    soc_0 = 1
    CR = 0.5
    T_end = 3900/abs(CR)

    ## Setting current profile
    t_eval = np.linspace(0, T_end , int(T_end)+1)
    
    ## Uncomment the current profile you desire
    current, current_string = lib.CC(t_eval, CR)
    # current, current_string = lib.UDDS(t_eval, CR)
    
    Time, cs, Voltage, Temp = lib.ref_solution(soc_0, current, t_eval, thermal_model = False)
    obs = observer_mionet(param, N)
    voltage_custom = []
    
    for j in range(len(Time)):
        Vt = obs.Voltage(cs[:,j], current[j])
        voltage_custom.append(Vt)
        
    soc = lib.SOC_calculator(cs)
    
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    
    # First subplot: Temperature
    axs[0].plot(Time, Voltage, label = 'REF')
    axs[0].plot(Time, voltage_custom, label = 'CUSTOM')
    axs[0].set_ylabel('Voltage (V)')
    axs[0].set_title('Voltage prediction')
    axs[0].grid(True)
    axs[0].legend()
    
    # Second subplot: SOC
    axs[1].plot(Time, soc, color='orange')
    axs[1].set_xlabel('Time (min)')
    axs[1].set_ylabel('SOC (%)')
    axs[1].set_title('SOC prediction')
    axs[1].grid(True)
    
    # Second subplot: surface concentration 
    axs[2].plot(Time, cs[N-1,:], label = 'POS')
    axs[2].plot(Time, cs[-1,:], label = 'NEG')
    axs[2].set_xlabel('Time (min)')
    axs[2].set_ylabel('cs ()')
    axs[2].set_title('Surface concentration')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
    
    epsilon_n = param['Negative electrode active material volume fraction']
    Rs_n = param['Negative particle radius [m]']
    a_n = 3*epsilon_n/Rs_n
    L_n = param['Negative electrode thickness [m]']
    csn_max = param['Maximum concentration in negative electrode [mol.m-3]']
    
    epsilon_p = param['Positive electrode active material volume fraction']
    Rs_p = param['Positive particle radius [m]']
    a_p = 3*epsilon_p/Rs_p
    L_p = param['Positive electrode thickness [m]']
    csp_max = param['Maximum concentration in positive electrode [mol.m-3]']
    
    tot = cs[N-1,:]*L_p*epsilon_p*csp_max + cs[-1,:]*L_n*epsilon_n*csn_max
    
    axs[2].plot(Time, tot/np.max(tot), '--', label = 'tot')
    
    # pos = cs[:N,:]; neg = cs[N:,:]
    # r = np.linspace(0,1,N); dr = r[1]; dr_2 = dr/2
    # argument = np.ones(N)*(r-dr_2)**2*dr
    
    # ## To compute the average concentration inside a spherical particle it is necessary to solve the integral INT[c(r)*dr]0,R / Volume

    # pos_avg = 3*argument@pos
    # # pos_avg = np.sum(integral_pos,axis=0)
    
    # neg_avg = 3*argument@neg
    # # neg_avg = np.sum(integral_neg,axis=0)
    # tot = pos_avg + neg_avg
    
    # plt.plot(Time, tot)
    