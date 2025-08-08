"""
This script compares the solution of a trained MIONet to a PyBaMM reference solution under a constant current profile. It is intended for evaluating the performance of MIONet models.

Key points:
- The MIONet predicts the system's state recursively, step by step, as it was trained for short time intervals. This is well-suited for applications such as observers, where the space formulation is essential.
- While MIONet evaluation may appear slower than a single PyBaMM simulation over the entire time horizon, this is because MIONet advances recursively (mimicking real observer operation), whereas PyBaMM computes the solution for the full interval in one call. 
- If PyBaMM were used recursively in the same way as MIONet, it would be significantly slower.

This evaluation approach reflects the intended deployment scenario of MIONet in embedded model-based observers.
"""

import pybamm
import numpy as np
import matplotlib.pyplot as plt
import utils 

plt.close('all')

TL = True # True if you want to predict concentrations using transfer learning models
N = 20 # Number of discretization points along the radial coordinate

# --- Parameter set and constants ---
param = pybamm.ParameterValues("Mohtat2020")

if TL :
    param = pybamm.ParameterValues("Prada2013")
    
param.set_initial_stoichiometries(1)
C = param['Nominal cell capacity [A.h]']      # [A.h]
CR = 1                                  # You can change this to any scaling

t_end = 3500/abs(CR)   # [s]
dT = 2

# Set constant current
I_app = C * CR    # [A]
t_eval = np.linspace(0, t_end, int(t_end/dT)+1)
current = np.ones_like(t_eval) * I_app

# Setup experiment: current for t_end seconds
drive_cycle_power = np.column_stack([t_eval, current])
experiment = pybamm.Experiment([pybamm.step.current(drive_cycle_power)])
del drive_cycle_power

var_pts = {
"x_n": 10,  # negative electrode
"x_s": 10,  # separator
"x_p": 10,  # positive electrode
"r_n": N,  # negative particle
"r_p": N,  # positive particle
}

# SPM model and simulation
model = pybamm.lithium_ion.SPM()
sim = pybamm.Simulation(model, parameter_values=param, var_pts = var_pts, experiment = experiment)
solution = sim.solve(t_eval)

# Extract results
time = solution["Time [s]"].entries
r = np.linspace(0,1,N)

# Negative and positive particle concentrations
cs_pos =  np.mean(solution["Positive particle concentration"].entries[:,:,:],axis=1)
cs_neg =  np.mean(solution["Negative particle concentration"].entries[:,:,:],axis=1)

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(12, 5), sharey=True)
i,j = 0,0
pcm0 = axs[i,j].pcolormesh(time/60, r, cs_neg, cmap='viridis')
axs[i,j].set_title("Negative Particle Concentration: REFERENCE")
axs[i,j].set_xlabel("Time (min)")
axs[i,j].set_ylabel("Particle Radius (-)")
fig.colorbar(pcm0, ax=axs[i,j], label="Concentration [-]")

i,j = 0,1
pcm1 = axs[i,j].pcolormesh(time/60, r, cs_pos, cmap='viridis')
axs[i,j].set_title("Positive Particle Concentration: REFERENCE")
axs[i,j].set_xlabel("Time (min)")
fig.colorbar(pcm1, ax=axs[i,j], label="Concentration [-]")

plt.tight_layout()
plt.show()

##################### MIONet prediction #####################

import torch 
import torch.nn as nn 

class MIONet(nn.Module): 
        
    def forward(self, coordinate, cs_0, current): # Process each input stream 
        out1 = self.branch1(cs_0) 
        out2 = self.branch2(current) 
        out_trunk = self.trunk(coordinate) 
        dot_product = torch.mul(torch.mul(out1,out2),out_trunk)
        output = torch.sum(dot_product,1)
        return output
    
net = torch.load('pos_saved_model.pt',map_location=torch.device('cpu'),weights_only=False)
if TL == 1:
    net = torch.load('pos_saved_model_tl.pt',map_location=torch.device('cpu'),weights_only=False)
model_p = utils.PINN(net, N) 

net = torch.load('neg_saved_model.pt',map_location=torch.device('cpu'),weights_only=False)
if TL == 1:
    net = torch.load('neg_saved_model_tl.pt',map_location=torch.device('cpu'),weights_only=False)
model_n = utils.PINN(net, N)

def F(cs_k, dT, I):
    x_pos = cs_k[:N] - cs_k[0] ## Remove the constant part of the concentration profile
    x_neg = cs_k[N:] - cs_k[N]
    x_pos_pred = model_p.predict(x_pos,dT,I) + cs_k[0]
    x_neg_pred = model_n.predict(x_neg,dT,I) + cs_k[N]
    cs_k1 = np.concatenate((x_pos_pred,x_neg_pred))
    return cs_k1

from tqdm import tqdm

c = np.zeros((2*N,))
c[:N] = cs_pos[0,0]
c[-N:] =cs_neg[0,0] ## Setting initial values
y = []; y.append(c)

for j in tqdm(range(len(time)-1), desc="Recursive MIONet prediction"):
    
    dT = time[j+1]-time[j]

    c_p = F(c, dT, current[j])
    y.append(c_p)
    c = c_p

y = np.array(y).T
y_pos = y[:N,:]
y_neg = y[-N:,:]

i,j = 1,0
pcm2 = axs[i,j].pcolormesh(time/60, r, y_neg, cmap='viridis')
axs[i,j].set_title("Negative Particle Concentration: MIONet")
axs[i,j].set_xlabel("Time (min)")
axs[i,j].set_ylabel("Particle Radius (-)")
fig.colorbar(pcm2, ax=axs[i,j], label="Concentration [-]")

i,j = 1,1
pcm3 = axs[i,j].pcolormesh(time/60, r, y_pos, cmap='viridis')
axs[i,j].set_title("Positive Particle Concentration: MIONet")
axs[i,j].set_xlabel("Time (min)")
fig.colorbar(pcm3, ax=axs[i,j], label="Concentration [-]")

plt.tight_layout()

## MIONet's prediction error
if y_neg.shape == cs_neg.shape:
    # error on anode
    rmse = np.sqrt(np.mean((cs_neg - y_neg) ** 2))
    print(f"Neg error {rmse:.1e} (rmse)")

    # error on cathode
    rmse = np.sqrt(np.mean((cs_pos - y_pos) ** 2))
    print(f"Pos error {rmse:.1e} (rmse)")