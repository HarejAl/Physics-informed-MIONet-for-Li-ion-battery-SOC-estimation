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
pybamm.set_logging_level("ERROR")

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
        
        return x_0, x_100, y_0, y_100
    
    def ref_solution(self, soc_0, current, t_eval, thermal_model = False):
        
        """
        This method computes the reference solution given initial soc and current
        """
        if thermal_model:
            batt_model = pybamm.lithium_ion.DFN(
                options = {'thermal' : 'lumped'})  
        else: 
            batt_model = pybamm.lithium_ion.DFN()
        
        self.param.set_initial_stoichiometries(soc_0)
     
        var_pts = {
        "x_n": 10,  # negative electrode
        "x_s": 10,  # separator
        "x_p": 10,  # positive electrode
        "r_n": self.N,  # negative particle
        "r_p": self.N,  # positive particle
        }
        
        drive_cycle_power = np.column_stack([t_eval, current])
        experiment = pybamm.Experiment([pybamm.step.current(drive_cycle_power)])
        del drive_cycle_power
        
        sim = pybamm.Simulation(batt_model, parameter_values=self.param, var_pts=var_pts , experiment = experiment)
        solution = sim.solve()
        
        ## Results
        time = solution["Time [min]"].entries # time_pb is automatically cutted at the cut off value 
        cs_pos_ref =  np.mean(solution["Positive particle concentration"].entries[:,:,:],axis=1)
        cs_neg_ref =  np.mean(solution["Negative particle concentration"].entries[:,:,:],axis=1)
        Temp = solution["X-averaged cell temperature [K]"].entries
        voltage_ref = solution["Voltage [V]"].entries
        
        y_ref = np.concatenate((cs_pos_ref,cs_neg_ref),axis=0)
        
        return time, y_ref, voltage_ref, Temp
    
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
        (soc_aux, ocp_aux) = np.load('soc_ocp_MOH.npy') # OCP curve in function of SOC --> Used to compute the OCP of electrodes for a give stoichiometry
        self.soc_aux = soc_aux 
        self.ocp_aux = ocp_aux
    
    def Voltage(self, y,i_cell):
        POS = y[:self.N]
        NEG = y[-self.N:]
        
        pos_avg = np.mean(POS[-5:]) # SOC on surface of particles: Avergaed on last 5 points to filter some process uncertainities --> OCP
        neg_avg = np.mean(NEG[-5:])
        
        SOC_pos = (pos_avg - self.y0)/(self.y100-self.y0)
        SOC_neg = (neg_avg - self.x0)/(self.x100-self.x0)
        
        soc = (SOC_pos + SOC_neg)/2

        R = self.param['Ideal gas constant [J.K-1.mol-1]']
        F = self.param['Faraday constant [C.mol-1]']     
        
        j0_pos_func = self.param['Positive electrode exchange-current density [A.m-2]']
        j0_neg_func = self.param['Negative electrode exchange-current density [A.m-2]']
        
        cs_max_pos = self.param['Maximum concentration in positive electrode [mol.m-3]']
        cs_max_neg = self.param['Maximum concentration in negative electrode [mol.m-3]']
        
        c_e = self.param['Initial concentration in electrolyte [mol.m-3]']
        T = self.param['Ambient temperature [K]']
        # T = 273 + 25

        ## Just to avoid som numerical issues
        POS[POS >= 1-5e-2] = 1-5e-2
        POS[POS <= 5e-2] = 5e-2
        
        NEG[NEG >= 1-5e-2] = 1-5e-2
        NEG[NEG <= 5e-2] = 5e-2
        
        cs_p_surf = POS[-1]*cs_max_pos; cs_n_surf = NEG[-1]*cs_max_neg
    
        j0_p = j0_pos_func(c_e,cs_p_surf,cs_max_pos,T).value
        j0_n = j0_neg_func(c_e,cs_n_surf,cs_max_neg,T).value
        
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
        
        j_n = i_cell / (L_n* a_n*W*H)
        j_p = -i_cell / (L_p * a_p*W*H)
        
        eta_neg = 2*R*T/F * np.arcsinh(j_n / (2 * j0_n))  
        eta_pos = 2*R*T/F * np.arcsinh(j_p / (2 * j0_p))
        
        OCP = np.interp(soc,self.soc_aux,self.ocp_aux)
       
        OCP_neg = self.param['Negative electrode OCP [V]']
        OCP_pos = self.param['Positive electrode OCP [V]']
        
        DuDt_func = self.param['Negative electrode OCP entropic change [V.K-1]']
        
        sto = POS[-1]
        Up = OCP_pos(POS[-1])# + DuDt_func(POS[-1])*(T-298)
        
        sto = NEG[-1]
        Un = OCP_neg(NEG[-1]) #+ DuDt_func(NEG[-1])*(T-298)
        
        OCP = Up - Un
        
        U = OCP  - eta_neg + eta_pos - i_cell/400  # The last term represents an approximated voltage error between SPM and DFN --> A trial and error term that sligthly improves the performance of the observer

        
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
    param = pybamm.ParameterValues("Prada2013")
    lib = plant(param, N = 20)
    
    soc_0 = 1
    CR = 1
    T_end = 3500/abs(CR)

    ## Setting current profile
    t_eval = np.linspace(0, T_end , int(T_end)+1)
    
    ## Uncomment the current profile you desire
    current, current_string = lib.CC(t_eval, CR)
    # current, current_string = lib.UDDS(t_eval, CR)
    
    Time, cs, Voltage, Temp = lib.ref_solution(soc_0, current, t_eval, thermal_model = False)
    soc = lib.SOC_calculator(cs)
    
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # First subplot: Temperature
    axs[0].plot(Time, Voltage)
    axs[0].set_ylabel('Voltage (V)')
    axs[0].set_title('Voltage prediction')
    axs[0].grid(True)
    
    # Second subplot: SOC
    axs[1].plot(Time, soc, color='orange')
    axs[1].set_xlabel('Time (min)')
    axs[1].set_ylabel('SOC (%)')
    axs[1].set_title('SOC prediction')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    
    