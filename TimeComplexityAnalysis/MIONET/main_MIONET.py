"""
Code for plotting the sensor_ptsponse of the trained electrochimical model
Both electrodes are involved
"""

import numpy as np 
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt
import scipy
import pybamm
from time import time

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

def true_solution(t_eval,current, N):
    
    param = pybamm.ParameterValues("Mohtat2020")
    param.set_initial_stoichiometries(1)
    batt_model = pybamm.lithium_ion.SPM()
    
    var_pts = {
    "x_n": 10,  # negative electrode
    "x_s": 10,  # separator
    "x_p": 10,  # positive electrode
    "r_n": N,  # negative particle
    "r_p": N,  # positive particle
    }
    
    # Defining the experiment
    drive_cycle_power = np.column_stack([t_eval, current])
    experiment = pybamm.Experiment([pybamm.step.current(drive_cycle_power)])
    del drive_cycle_power
    
    sim = pybamm.Simulation(batt_model, parameter_values=param, var_pts=var_pts, experiment = experiment)
    solution = sim.solve(t_eval)
    
    ## Results
    time_pb = solution["Time [min]"].entries # time_pb is automatically cutted at the cut off value 
    
    cs_pos_true =  solution["Positive particle concentration"].entries[:,0,:]
    cs_neg_true =  solution["Negative particle concentration"].entries[:,0,:]
    
    cs_pos_true -= cs_pos_true[0,0]
    cs_neg_true -= cs_neg_true[0,0]
    
    # y_true = np.concatenate((cs_pos_true,cs_neg_true),axis=0)
    return cs_pos_true, cs_neg_true

class MIONet(nn.Module): 
        
    def forward(self, coordinate, input1, input2): # Process each input stream 
        out1 = self.branch1(input1) 
        out2 = self.branch2(input2)
        out_trunk = self.trunk(coordinate) 
        # Concatenate outputs from both branches 
        dot_product = torch.mul(torch.mul(out1,out2),out_trunk)
        output = torch.sum(dot_product,1) #+self.output_lay.bias

        return torch.unsqueeze(output,1)

class PINN():
    def __init__(self,NN_model, N, dT):
        self.network = NN_model
        t = torch.tensor(np.ones([N,1])*dT,dtype=torch.float32)*0.100
        ksi = torch.tensor(np.linspace(0,1,N)[:,None],dtype=torch.float32)
        self.t_ksi = torch.cat((t,ksi),1) #*torch.tensor(np.array([[0.1,1]]),dtype=torch.float32)
        
    def predict(self, Cs_0, I):
        
        cs_0 =torch.tensor(Cs_0,dtype=torch.float32)
        i = torch.tensor(np.array([[I]]),dtype=torch.float32)
        y = self.network(self.t_ksi,cs_0, i)
        
        return y.detach().numpy()

class SPM_mionet():
    def __init__(self, N):
            self.net_p = torch.load('best_pi_loss_pos_13.pth',map_location=torch.device('cpu'),weights_only=False)
            self.net_n = torch.load('best_pi_loss_neg_14.pth',map_location=torch.device('cpu'),weights_only=False)
            self.sensor_pts = 31
            self.PINN = PINN
            

            self.r_sensor = np.linspace(0,1,self.sensor_pts)[:,None]
            self.N = N
            self.r = np.linspace(0,1,N)[:,None]
            self.param = pybamm.ParameterValues("Mohtat2020")


    def f_nn(self,cs_k,I):
        x_pos = np.interp(self.r_sensor[:,0], self.r[:,0], cs_k[:self.N] - cs_k[0].copy())
        x_neg = np.interp(self.r_sensor[:,0], self.r[:,0], cs_k[self.N:] - cs_k[self.N].copy())
        x_pos_1 = self.model_pos.predict(x_pos,I) + cs_k[0].copy()
        x_neg_1 = self.model_neg.predict(x_neg,I) + cs_k[self.N].copy()
        cs_k1 = np.concatenate((x_pos_1,x_neg_1))
        return cs_k1
    
    def simulation(self, t_eval, current):
        
        self.dT = t_eval[1]-t_eval[0]
        self.model_pos = self.PINN(self.net_p, self.N, self.dT)
        self.model_neg = self.PINN(self.net_n, self.N, self.dT)
        iterations = len(t_eval)
        y = np.zeros((2*self.N,1))
        
        for j in range(1,iterations):
            I = current[j-1]
            cs_p = self.f_nn(y[:,-1].copy(), I)
            
            y = np.concatenate((y,cs_p),axis=1)
            
        y_pos = y[:self.N,:] #+ y_true_pos[0,0]
        y_neg = y[self.N:,:] #+ y_true_neg[0,0]
    
        return y_pos, y_neg


T_end = 200
t_eval = np.linspace(0,T_end, T_end//2+1)
current = np.ones((len(t_eval)))*5

N = 31

y_true_pos, y_true_neg = true_solution(t_eval, current, N)

sim_class = SPM_mionet(N)

t0 = time()
y_pos, y_neg = sim_class.simulation(t_eval, current)
Time_needed = ((time() - t0) / len(t_eval))


### error 1
diff_n = (y_true_neg - y_neg) # they are already normalized / param['neg']['cmax']
en = np.mean(diff_n ** 2)

print('Neg error {} (-)'.format(en))

diff_p = (y_true_pos - y_pos) # / param['pos']['cmax']
ep = np.mean(diff_p ** 2)
print('Pos error {} (-)'.format(ep))


### error 2
en = np.mean(np.linalg.norm((y_true_neg - y_neg) / 1, axis=1))
ep = np.mean(np.linalg.norm((y_true_pos - y_pos) / 1, axis=1)) # --> cs_max = 1 (normalized concentrations)
print('Neg error {} (Fr. norm)'.format(en))
print('Pos error {} (Fr. norm)'.format(ep))
error = ((en + ep)/2)*100
         
whole_solution=1
fixed_time_solution=1
voltage=1
plot_soc = 1

Time = t_eval; r = np.linspace(0,1,N)


if whole_solution==1:

    plot_comparison(y_true_pos, y_pos, Time, r, electrode_name="Positive")

if whole_solution==1:

    plot_comparison(y_true_neg, y_neg, Time, r, electrode_name="Negative")

### Comparing the two solutions at index idx:
if fixed_time_solution==1:
    
    fig, ax = plt.subplots(1,1,layout='constrained',figsize=(6,4))
    
    k = 0
    for i in [0, 0.1 ,0.3, 0.6, 0.95]:
        
        idx = int(i*len(Time))
        
        c = y_pos[:,idx]
        c_true = y_true_pos[:,idx]
        
        ax.plot(r,c,'b',label='POS predicted')
        ax.plot(r,c_true,'c--',label='POS true')
        
        c = y_neg[:,idx]
        c_true = y_true_neg[:,idx]
        
        ax.plot(r,c,'r',label='NEG predicted')
        ax.plot(r,c_true,'m--',label='NEG true')
        
        if k == 0:
            ax.legend()
            k+=1
    
    # ax.title('Solution @ {}s'.format(Time[idx]))
    ax.set_title('Solution in time')
    ax.set_xlabel('R [-]')
    ax.set_ylabel('Concentration [-]')
    # ax.legend()
    plt.grid()

