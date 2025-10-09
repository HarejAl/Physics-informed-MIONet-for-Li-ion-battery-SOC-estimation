"""
Comparison with experimental data for 'Ai2020' parameters set
"""

import numpy as np
import matplotlib.pyplot as plt
import pybamm
## Set 
font_lab = {'family':'serif','color':'black','size':12} # Font for labels
font_tit = {'family':'serif','color':'black','size':14} # Font for labels

def SOC_test():
    
    """
    This method needs to determine the stoichiometries of Li atoms in electrodes at soc = 0 and soc = 100%
    """

    # Determine stochiometries limits (using electrode state of health model)
    Vmin = param["Lower voltage cut-off [V]"]
    Vmax = param["Upper voltage cut-off [V]"]
    param.set_initial_stoichiometries(1)

    # print('V_max {}; V_min {}'.format(Vmax,Vmin))

    parameter_values = pybamm.LithiumIonParameters()
    
    Q_n  = param.evaluate(parameter_values.n.Q_init)
    Q_p  = param.evaluate(parameter_values.p.Q_init)
    Q_Li = param.evaluate(parameter_values.Q_Li_particles_init)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param, parameter_values)
    inputs = {"V_min": Vmin, "V_max": Vmax, "Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
    esoh_sol = esoh_solver.solve(inputs)

    x_100 = esoh_sol["x_100"]
    y_100 = esoh_sol["y_100"]
    x_0   = esoh_sol["x_0"]
    y_0   = esoh_sol["y_0"]
    
    return x_0, x_100, y_0, y_100

def SOC_calculator(POS, NEG):
    """
    This method computes the soc as the average soc of both electrodes
    Inputs: normalised concentration vector: y = [c_pos, c_neg].T
    
    Returns: SOC 
    """
    x0, x100, y0, y100 = SOC_test()
            
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

model = pybamm.lithium_ion.SPM()

# update parameters, making C-rate and input
param = pybamm.ParameterValues("Ai2020")
capacity = param["Nominal cell capacity [A.h]"]
param.update({"Current function [A]": capacity*1.05 * pybamm.InputParameter("C-rate")})

# define the simulation
sim = pybamm.Simulation(
    model,
    parameter_values=param,
)

# solve for different C-rates
Crates = [0.5, 1, 2]
solutions = []
for Crate in Crates:
    print(f"{Crate} C")
    sol = sim.solve(t_eval=np.linspace(0,3650/Crate, 1001), inputs={"C-rate": Crate})
    solutions.append(sol)


# unpack solutions
solution05C, solution1C, solution2C = solutions

# load experimental results
import pandas as pd

data_loader = pybamm.DataLoader()


data_V_05C = pd.read_csv(
    data_loader.get_data("0.5C_discharge_U.txt"), delimiter="\s+", header=None
)
data_V_1C = pd.read_csv(
    data_loader.get_data("1C_discharge_U.txt"), delimiter="\s+", header=None
)
data_V_2C = pd.read_csv(
    data_loader.get_data("2C_discharge_U.txt"), delimiter="\s+", header=None
)


t_all2C = solution2C["Time [h]"].entries
V_n2C = solution2C["Voltage [V]"].entries
I2C = solution2C["Current [A]"].entries

t_all1C = solution1C["Time [h]"].entries
V_n1C = solution1C["Voltage [V]"].entries
I1C = solution1C["Current [A]"].entries

t_all05C = solution05C["Time [h]"].entries
V_n05C = solution05C["Voltage [V]"].entries
I05C = solution05C["Current [A]"].entries


f, ax1 = plt.subplots(1, 1, figsize=(14, 4))

ax1.plot(t_all2C, V_n2C, "r-", label="Simulation")
ax1.plot(
    data_V_2C.values[::30, 0] / 3600,
    data_V_2C.values[::30, 1],
    "ro",
    markerfacecolor="none",
    label="Experiment",
)
ax1.plot(t_all05C, V_n05C, "g-")
ax1.plot(
    data_V_05C.values[::100, 0] / 3600,
    data_V_05C.values[::100, 1],
    "go",
    markerfacecolor="none",
)
ax1.plot(t_all1C, V_n1C, "b-")
ax1.plot(
    data_V_1C.values[::50, 0] / 3600,
    data_V_1C.values[::50, 1],
    "bo",
    markerfacecolor="none",
)
ax1.legend()
ax1.set_xlabel("Time [h]")
ax1.set_ylabel("Voltage [V]")
ax1.text(0.1, 3.2, r"2 C", {"color": "r", "fontsize": 14})
ax1.text(1.1, 3.2, r"1 C", {"color": "b", "fontsize": 14})
ax1.text(1.6, 3.2, r"0.5 C", {"color": "g", "fontsize": 14})

f.tight_layout()
f.show()

