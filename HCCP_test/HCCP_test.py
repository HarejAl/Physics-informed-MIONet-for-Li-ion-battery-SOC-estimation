#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:47:15 2024

@author: gabriel
"""

import pybamm
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# Set model options
options={
            "particle": "Fickian diffusion",
            "cell geometry": "arbitrary",
            "thermal": "isothermal",
            "particle mechanics": "swelling only",
        }

# Define model
model = pybamm.lithium_ion.DFN()

# Set model parameters (Enertech cell)
# param = pybamm.ParameterValues("Marquis2019")
# param = pybamm.ParameterValues("Mohtat2020")
param = pybamm.ParameterValues("Ai2020")
# param.update({"Nominal cell capacity [A.h]": 2.45})
# param['Nominal cell capacity [A.h]'] = 1

# temp = 25 + 273.15;
# param.update({"Ambient temperature [K]": temp})
# param.update({"Initial temperature [K]": temp})

# Calculate stoichiometries at 100% SOC
param.set_initial_stoichiometries(1)

# Determine stochiometries limits (using electrode state of health model)
Vmin = param["Lower voltage cut-off [V]"]
Vmax = param["Upper voltage cut-off [V]"]

print('V_max {}; V_min {}'.format(Vmax,Vmin))

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

# Update the mesh
# var = pybamm.standard_spatial_vars
# var_pts = {
#     var.x_n: 50,
#     var.x_s: 50,
#     var.x_p: 50,
#     var.r_n: 21,
#     var.r_p: 21,
# }

# Define extra model variables
model.variables["SoCn"] = ((model.variables["Negative electrode stoichiometry"]-x_0)/(x_100 - x_0)) * 100
model.variables["SoCp"] = ((model.variables["Positive electrode stoichiometry"]-y_0)/(y_100-y_0)) * 100

# Define HPPC experiment
Npulses = 64; ## 77 for Marquis
s = pybamm.step.string
hppc_experiment = pybamm.Experiment(
    
    [
     (   
        s("Rest for 0.2 hour"),
        s("Discharge at 1C for 1 minutes"),
     )
    ]*Npulses
    
    +

    [
            s("Rest for 0.2 hour"),
            
        ],
    
    # [
    #  (   
    #     s("Rest for 0.1 hour"),
    #     s("Discharge at 0.1C for 1 minutes"),
    #  )
    # ]*30, 
    
    period="1 second",

    termination="2.4V"
)

pybamm.set_logging_level("NOTICE")

sim = pybamm.Simulation(model, 
                        parameter_values=param, 
                        experiment=hppc_experiment)

##### THERE are some mistakes
sim.solve()
sol = sim.solution

# relaxation_experiment =  pybamm.Experiment(
    
#     [
#         s("Rest for 1 hour"),
        
#     ],
    
#     period="1 second",
# )

# sim1 = pybamm.Simulation(model, 
#                         parameter_values=param, 
#                         experiment=relaxation_experiment, 
#                         )
# sim1.solve(starting_solution=sol)
# sol1 = sim1.solution


# SoCn     = sim1.solution["SoCn"].entries
# SoCp     = sim1.solution["SoCp"].entries

# Voltage = sim1.solution["Voltage [V]"].entries
# Current = sim1.solution["Current [A]"].entries
# Time    = sim1.solution["Time [h]"].entries

SoCn     = sim.solution["SoCn"].entries
SoCp     = sim.solution["SoCp"].entries

Voltage = sim.solution["Voltage [V]"].entries
Current = sim.solution["Current [A]"].entries
Time    = sim.solution["Time [h]"].entries

# comparing voltage components for Doyle-Fuller-Newman model and Single Particle Model
cm = 1/2.54 # centimeters in inches
h = 12.0
w = h*2
fig, axs = plt.subplots(2,1, figsize=(w*cm, h*cm)) 

axs[0].plot(Time, Current, 'k')
axs[1].plot(Time, Voltage, 'k')

# Add labels with LaTeX font size
axs[0].set_xlabel(r'$t\ (h)$', fontsize=12)
axs[0].set_ylabel(r'$i\ (A)$', fontsize=12)

axs[1].set_xlabel(r'$t\ (h)$', fontsize=12)
axs[1].set_ylabel(r'$v\ (V)$', fontsize=12)

# Change tick font size to LaTeX style
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    # Set tight x-axis boundaries
    ax.set_xlim([Time[0], Time[-1]])

# Adjust layout
fig.tight_layout()

# Show the plot
plt.show()

fig2, axs2 = plt.subplots(2,1, figsize=(w*cm, h*cm)) 

axs2[0].plot(Time, SoCn, 'k')
axs2[1].plot(Time, SoCp, 'k')

# Add labels with LaTeX font size
axs2[0].set_xlabel(r'$t\ (h)$', fontsize=12)
axs2[0].set_ylabel(r'$SoC_n\ (\%)$', fontsize=12)

axs2[1].set_xlabel(r'$t\ (h)$', fontsize=12)
axs2[1].set_ylabel(r'$SoC_p\ (\%)$', fontsize=12)

# Change tick font size to LaTeX style
for ax in axs2:
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    # Set tight x-axis boundaries
    ax.set_xlim([Time[0], Time[-1]])
    
# Adjust layout
fig2.tight_layout()

# Show the plot
plt.show()

folder_path = "C:/Users/Lorenzo/OneDrive - Politecnico di Milano/PhD1/Ricerca Lithium-Ion Batteries/PyBaMM/New Research (ecTM battery cell model)/ECM parameters estimation/hppc_test_data"
base_filename = "hppc_test_data_dfn_Ai.mat"
filename = base_filename

sol.save_data(
    filename,
    ["Current [A]", "Time [s]", "Voltage [V]"],
    to_format="matlab",
    short_names={"Current [A]": "current", "Time [s]": "time", "Voltage [V]": "voltage"},
)
