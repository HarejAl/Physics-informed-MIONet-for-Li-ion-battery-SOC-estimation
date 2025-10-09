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
        t = torch.tensor(np.ones([N,1])*dT,dtype=torch.float32)*0.1001
        ksi = torch.tensor(np.linspace(0,1,N)[:,None],dtype=torch.float32)
        self.t_ksi = torch.cat((t,ksi),1) #*torch.tensor(np.array([[0.1,1]]),dtype=torch.float32)
        
    def predict(self, Cs_0, I):
        
        cs_0 =torch.tensor(Cs_0,dtype=torch.float32)
        i = torch.tensor(np.array([[I]]),dtype=torch.float32)
        y = self.network(self.t_ksi,cs_0, i)
        
        return y.detach().numpy()

class SPM_mionet():
    def __init__(self, N):
            self.net_p = torch.load('pos_saved_model.pt',map_location=torch.device('cpu'),weights_only=False)
            self.net_n = torch.load('neg_saved_model.pt',map_location=torch.device('cpu'),weights_only=False)

            self.sensor_pts = 31
            self.PINN = PINN

            self.r_sensor = np.linspace(0,1,self.sensor_pts)[:,None]
            self.N = N
            self.r = np.linspace(0,1,N)[:,None]
            self.param = pybamm.ParameterValues("Mohtat2020")


    def f_nn(self,cs_k,I):
        
        if self.N != self.sensor_pts:
            x_pos = np.interp(self.r_sensor[:,0], self.r[:,0], cs_k[:self.N] - cs_k[0].copy())
            x_neg = np.interp(self.r_sensor[:,0], self.r[:,0], cs_k[self.N:] - cs_k[self.N].copy())
            
        else: 
            x_pos = cs_k[:self.N] - cs_k[0].copy()
            x_neg = cs_k[self.N:] - cs_k[self.N].copy()
            
            
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

def plot_results(x, ct_avg, err, opt_fcn):
    """
    Creates a figure with three overlaid axes to plot:
      - Computational time vs. state vector size (red, left y-axis)
      - Error vs. state vector size (blue, middle y-axis)
      - Optimality function vs. state vector size (black, right y-axis)
      
    Parameters:
        x       : 1D array of state vector sizes = N_arr
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
    h2.set_yscale('log')
    
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
    
    plt.title('MIONET')
    plt.show()

print('MIONET time complexity')

T_end = 200
t_eval = np.linspace(0,T_end, T_end//2+1)
current = np.ones((len(t_eval)))*5

N_list = np.linspace(6,41,8,dtype=int) # states in each electrode
ct_avg = []
err_arr = []

just_plotting = 0

if just_plotting != 1:

    for N in N_list:
        
        y_pos_true, y_neg_true = true_solution(t_eval, current, N)
        
        n_rep = 20 # Each simualtion is repeated n_rep times --> to average the times
        ct = []
        print(N)
    
        sim_class = SPM_mionet(N)
        for _ in range(n_rep):
            t0 = time()
            y_pos, y_neg = sim_class.simulation(t_eval, current)
            ct.append((time() - t0) / len(t_eval))
        
        ct_avg.append(np.mean(ct)*1e3)
        
        en = np.mean(np.linalg.norm((y_neg_true - y_neg), axis=1))
        ep = np.mean(np.linalg.norm((y_pos_true - y_pos), axis=1)) 
        err_arr.append((en + ep)/2)
        
        # err_arr.append(ep)
        
else: 
    
    (N_list, ct_avg, err_arr, opt_fcn) = np.load("Mionet_tuple.npy")
        


# Normalize ct_avg and err_arr to the [0, 1] range.
ct_n = (ct_avg - np.min(ct_avg)) / (np.max(ct_avg) - np.min(ct_avg))
err_n = (err_arr - np.min(err_arr)) / (np.max(err_arr) - np.min(err_arr))

# Compute the optimality function (as sum of normalized ct_avg and err_arr).
opt_fcn = ct_n + err_n

# # Save the array
np.save("Mionet_tuple14.npy", (N_list*2, ct_avg, err_arr, opt_fcn))

# Call the plotting function.
plot_results(N_list, ct_avg, err_arr, opt_fcn)


