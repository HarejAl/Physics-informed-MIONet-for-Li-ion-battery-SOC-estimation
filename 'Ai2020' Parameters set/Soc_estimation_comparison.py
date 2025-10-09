"""
Battery plant simulation and SOC estimation comparison (MIONet vs. ECM) with experimental data
---------------------------------------------------------------------
The two observers are compared based on a plant that 

"""

import numpy as np
import matplotlib.pyplot as plt
import pybamm
import torch 
import torch.nn as nn 
import scipy
import utils_ai

## Set logging level to suppress informational messages
pybamm.set_logging_level("ERROR")

class MIONet(nn.Module): 
        
    def forward(self, coordinate, cs_0, current): # Process each input stream 
        out1 = self.branch1(cs_0) 
        out2 = self.branch2(current) 
        out_trunk = self.trunk(coordinate) 
        dot_product = torch.mul(torch.mul(out1,out2),out_trunk)
        output = torch.sum(dot_product,1)
        return output

## Setting seed for reproducibility
np.random.seed(42)  

## Set 
font_lab = {'family':'serif','color':'black','size':12} # Font for labels
font_tit = {'family':'serif','color':'black','size':14} # Font for labels

T_amb = 25

param = pybamm.ParameterValues("Ai2020")
param['Ambient temperature [K]'] = 273.12 + T_amb
param['Initial temperature [K]'] = param['Ambient temperature [K]'] 
C = param['Nominal cell capacity [A.h]']*1.05 # Capacity [Ah]

#### Setting common parameters 
# T_end = 3614 #36878 #7308 3614
dT = 2 # Time step --> It can be setup up to 2.5 sec
N = 31 # Number of states in each electrode
r = np.linspace(0,1,N) #n is the number of points for the discretization in pybamm

plant_utils = utils_ai.plant(param, N) ## Initializing the class plant

## Initializing the dictionary for solution
D = {
     'plant':{},
     'mionet':{},
     'ecm':{}
     }

## Parameters to be modified
soc_0 = 1
CR = 0.5

if CR == 0.5:
    T_end = 7308
elif CR == 1:
    T_end = 3614
elif CR == 0.1:
    T_end = 36574

# T_end = 3000## Setting current profile
# T_end = 3700/CR
t_eval = np.linspace(0, T_end , int(T_end/dT)+1)
current = np.ones_like(t_eval)*C*CR
# current, current_string = plant_utils.CC(t_eval, CR = CR)
current_string = 'CC'
# current, current_string = lib.DST(t_eval, CR)
# current, current_string = plant_utils.UDDS(t_eval, CR); soc_0 = 0.8
# Uncomment the desired current profile
print('Comparison between ECM and Mionet for '+current_string+' at {} CR for total time {:.2f} hr'.format(CR, T_end/3600))

## REFERENCE solution with pybamm

# Use Time instead of t_eval in case the simulation is triggered earlier --> pybamm does it automatically iin case the min value of voltage is reached
noise_std = 1e-3
use_experimental_data = False

if use_experimental_data:
    print('With experimental data')
    noise_std *= 10 # Increase R in the filter
    # 
    import pandas as pd
    
    data_loader = pybamm.DataLoader()
    
    if CR == 2:
        
        data = pd.read_csv(
            data_loader.get_data("2C_discharge_U.txt"), delimiter="\s+", header=None)
    
    if CR == 0.5:
            
        data = np.load("CC_05.npy")            
    
    if CR == 1:
        data = np.load("CC_1.npy") 
    
    Time = data[::dT,0]/60
    Voltage_ref = data[::dT,1]
    soc_exp = data[::dT,2]

    
    # if T_end < 3614/CR: #7308 for 0.5cr
    # idx0 = np.where(abs(soc_exp - soc_0*100) < 1e-1)[0][0]
    # idx1= np.where(Time == (Time[idx0] + T_end+dT))[0][0]
    
    # Time = Time[idx0:idx1]/60
    # soc_exp = 1 - Time/Time[-1]
    # soc_exp*= 100
    # Voltage_ref = Voltage_ref[idx0:idx1]
    # soc_exp = soc_exp[idx0:idx1]
        
    Voltage_plant = Voltage_ref
    
    # fig, ax = plt.subplots(1,1, figsize=(8,6))
    # ax.plot(Time/60,Voltage_plant,'o','navy',linewidth = 1, label = 'Experiment')
    # ax.plot(Time/60,Voltage_sim,'navy',linewidth = 1, label = 'PyBamm')
    
    # ax.set_ylabel('Voltage [V]',fontdict = font_lab)
    # ax.set_xlabel('Time [m]',fontdict = font_lab)
    # ax.set_title('Voltage plant',fontdict = font_tit)
    # ax.grid(True, linestyle='--', alpha=0.7)
    # ax.legend()
    # plt.show()

    
    D['plant']['soc'] = soc_exp
    
    # fig, axs = plt.subplots(1,3)
    # j = 0
    # axs[j].plot(Time, Voltage_plant, 'o', markerfacecolor='none',label = 'Plant')
    # axs[j].plot(Time_sim, Voltage_sim, label = 'Sim')
    # axs[j].set_ylabel('Voltage [V]',fontdict = font_lab)
    # axs[j].set_xlabel('Time [m]',fontdict = font_lab)
    # axs[j].set_title('Voltage',fontdict = font_tit)
    # axs[j].grid(True, linestyle='--', alpha=0.7)
    # axs[j].legend()
    # j = 1
    # axs[j].plot(Time, soc_exp, 'o', markerfacecolor='none', label = 'Plant')
    # axs[j].plot(Time_sim, soc_sim, label = 'Sim')
    # axs[j].set_ylabel('SOC [%]',fontdict = font_lab)
    # axs[j].set_xlabel('Time [m]',fontdict = font_lab)
    # axs[j].set_title('SOC',fontdict = font_tit)
    # axs[j].grid(True, linestyle='--', alpha=0.7)
    # axs[j].legend()
    # j = 2
    # axs[j].plot(soc_exp, Voltage_plant, 'o', markerfacecolor='none', label = 'Plant')
    # axs[j].plot(soc_sim, Voltage_sim, label = 'Sim')
    # axs[j].set_ylabel('Voltage [V]',fontdict = font_lab)
    # axs[j].set_xlabel('SOC [%]',fontdict = font_lab)
    # axs[j].set_title('SOC-Voltage',fontdict = font_tit)
    # axs[j].grid(True, linestyle='--', alpha=0.7)
    # axs[j].legend()
    # plt.show()
    
else: 
    # Time, y, Voltage_sim, Temp = plant_utils.ref_solution(soc_0, current, t_eval, thermal_model = False)
    # Voltage_plant = Voltage_sim + np.random.normal(0,noise_std,(len(Voltage_sim))) 
    # D['plant']['soc sim'] = plant_utils.SOC_calculator(y)
    # D['plant']['soc'] = D['plant']['soc sim']
    # soc1 = plant_utils.SOC_calculator(y)
    # D['plant']['concentration'] = y
    
    if CR == 0.5:
        data = np.load("CC_05_sim.npy")            
    
    if CR == 1:
        data = np.load("CC_1_sim.npy") 
        
    if CR == 2:
        data = np.load("CC_2_sim.npy") 
    
    Time = data[::dT,0]
    Voltage_plant = data[::dT,1]
    soc = data[::dT,2]
    D['plant']['soc'] = soc


D['plant']['Voltage'] = Voltage_plant
D['plant']['time'] = Time
D['plant']['current'] = current
D['plant']['current_string'] = current_string
D['plant']['dT'] = dT
D['plant']['measurment_noise'] = noise_std

# current += np.random.normal(0,noise_std,(len(current))) # Adding noise to the current profile --> Mimic realistic scenario

plot = False
if plot:
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(t_eval/60,Voltage_plant,'navy',linewidth = 1)
    ax.set_ylabel('Voltage [V]',fontdict = font_lab)
    ax.set_xlabel('Time [m]',fontdict = font_lab)
    ax.set_title('Voltage plant',fontdict = font_tit)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
plot = False
if plot:
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(t_eval/60,current/C,'navy',linewidth = 1)
    ax.set_ylabel('i [x CR]',fontdict = font_lab)
    ax.set_xlabel('Time [m]',fontdict = font_lab)
    ax.set_title(current_string,fontdict = font_tit)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.show()

##################################################################################################
##################################### 1 SOLUTION with MIONET #####################################
net = torch.load('pos_saved_model_AI.pt',map_location=torch.device('cpu'),weights_only=False)
model_p = utils_ai.PINN(net, N) 

net = torch.load('neg_saved_model_AI.pt',map_location=torch.device('cpu'),weights_only=False)
model_n = utils_ai.PINN(net, N)

obs_utils = utils_ai.observer_mionet(param, N)

# Setting up the Unscented Kalman Filter (UKF)
# In this implementation, the state vector contains all concentrations, concatenated:
# x = [cs_pos1, cs_pos2, ..., cs_posN, cs_neg1, cs_neg2, ..., cs_negN] --> Number of states in the observer = 2*N
from filterpy.kalman import JulierSigmaPoints 
from filterpy.kalman import UnscentedKalmanFilter as UKF

# We use Julier Sigma Points. kappa=0 gives the standard unscented filter (Filterpy documentation)
sigmas = JulierSigmaPoints(n = 2*N, kappa = 0, sqrt_method=scipy.linalg.sqrtm)

## Define process and measurement models
def F(cs_k,dt,I):
    x_pos = cs_k[:N] - cs_k[0]
    x_neg = cs_k[N:] - cs_k[N]
    x_pos_1 = model_p.predict(x_pos,dt,I) + cs_k[0]
    x_neg_1 = model_n.predict(x_neg,dt,I) + cs_k[N]
    cs_k1 = np.concatenate((x_pos_1,x_neg_1))
    return cs_k1

def H(cs_k,I):
    U = obs_utils.Voltage(cs_k[:],I)
    cs_conservation = np.mean(cs_k[:N])*L_p*epsilon_p*csp_max + np.mean(cs_k[-N:])*L_n*epsilon_n*csn_max
    return np.array([U, cs_conservation])

ukf = UKF(dim_x=2*N, dim_z = 2, dt = dT, fx = F, hx = H, points=sigmas)

## Initializing the sta vector --> stoichiometries are initialized with an initial SOC of 50%
ukf.x = np.ones((2*N))
soc_0_aux = 0.5 # Initial soc guess 
ukf.x[:N] *= 0.43 + (1-soc_0_aux)*(0.97 - 0.43)    
ukf.x[N:] *= soc_0_aux*0.84   

## Initializing covariance matrices --> Part of Kalman filter tuning
ukf.P *= 1e-3
ukf.R *= (noise_std*2)**2
ukf.R[-1,-1] = 1e-4
ukf.Q *= (1e-4)**2 #2e-4 for sim 

D['mionet']['Q'] = ukf.Q
D['mionet']['R'] = ukf.R

epsilon_n = param['Negative electrode active material volume fraction']
L_n = param['Negative electrode thickness [m]']
csn_max = param['Maximum concentration in negative electrode [mol.m-3]']

epsilon_p = param['Positive electrode active material volume fraction']
L_p = param['Positive electrode thickness [m]']
csp_max = param['Maximum concentration in positive electrode [mol.m-3]']

Voltage_mn = []
P = []; y = []
curr = C * CR

from tqdm import tqdm

for j in tqdm(range(len(Time)), desc="UKF (MIONet) estimation: "):

    # Measurement
    z = np.array([Voltage_plant[j], 2.0580594058001718])
    
    # Update step
    if j > 0:
        ukf.update(z = z, I = curr)

    ## Saving values
    P.append(np.diag(ukf.P*10))
    y.append(ukf.x)
    Voltage_mn.append(ukf.hx(ukf.x,I = curr)[0])
    
    ## Propagate step
    ukf.predict(I = curr)

## Storing saved variables in the dictionary
P = np.array(P).T; y = np.array(y).T

D['mionet']['soc'] = obs_utils.SOC_calculator(y)
D['mionet']['concentration'] = y

# CONFIDENCE intervals
conf = np.sqrt(P)*2 # +- 2 sigma interval
conf[N:] *= -1
D['mionet']['soc_upper'] = obs_utils.SOC_calculator(y + conf)
D['mionet']['soc_bottom'] = obs_utils.SOC_calculator(y - conf)
D['mionet']['Voltage'] = Voltage_mn

##### SOC error
idx = int(120/dT) ## Steady state error: we neglect the transient
err = (D['plant']['soc'] - D['mionet']['soc'])**2
err = np.sqrt(np.mean(err[idx:]))

print('MIONET SOC estimation error {:.2e} (RMSE)'.format(err))

##################################################################################################
##################################### 1 SOLUTION with ECM ########################################
##### LOADING ECM PARAMETERS
parameters = scipy.io.loadmat('ecm_parameters_ai.mat')
R0_aux = parameters['R0'].T[:,0]
soc_aux = parameters['SOC'].T[:,0]

Em_aux = parameters['Em'].T[:,0]
tau_x = parameters['taux'].T
R_x = parameters['Rx'].T
soc_aux1, Em_aux = np.load('soc_ocp_AI.npy') # OCP curve in function of SOC --> Used to compute the OCP of electrodes for a give stoichiometry

plot = False
if plot:
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(soc_exp,Voltage_plant,label = 'Plant')
    ax.plot(soc_aux1*100, Em_aux,label = 'ecm')
    ax.set_ylabel('Voltage [V]',fontdict = font_lab)
    ax.set_xlabel('SOC [%]',fontdict = font_lab)
    ax.set_title('Voltage plant',fontdict = font_tit)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_xlim([soc_exp[0]-5, soc_exp[-1]+5])
    plt.show()

##### SETTING PARAMETERS
Q_batt = C*3600  # Ah -> As  
# q_soc = (1e-2)**2
# q_Vn  = (2e-4)**2
# P0_vn = 1e-6
# P0_soc = 1e-8

# P0_soc = 5e-2
# P0_Vn = 1e-1
f = 0.1
q_soc = (0.5e-2)**2*f
q_Vn  = (10e-4)**2*f

P0_soc = 5e-2
P0_Vn = 1e-8

def generateA(SOC_T):
    
    A = np.zeros([4,4])
    
    # Interpolating ECM parameters 
    Tau1 = np.interp(SOC_T,soc_aux,tau_x[:,0])
    Tau2 = np.interp(SOC_T,soc_aux,tau_x[:,1])
    Tau3 = np.interp(SOC_T,soc_aux,tau_x[:,2])
    
    A[0,0] = 1
    A[1,1] = np.e**(-dT/Tau1)
    A[2,2] = np.e**(-dT/Tau2)
    A[3,3] = np.e**(-dT/Tau3)
    
    return A

def generateB(SOC):
    
    B = np.zeros([4,1])
    # Interpolating ECM parameters 
    Tau1 = np.interp(SOC,soc_aux,tau_x[:,0])
    Tau2 = np.interp(SOC,soc_aux,tau_x[:,1])
    Tau3 = np.interp(SOC,soc_aux,tau_x[:,2])
    
    R1 = np.interp(SOC,soc_aux,R_x[:,0])
    R2 = np.interp(SOC,soc_aux,R_x[:,1])
    R3 = np.interp(SOC,soc_aux,R_x[:,2])
    
    B[0,0] = -dT/Q_batt
    B[1,0] = (1-np.e**(-dT/Tau1))*R1
    B[2,0] = (1-np.e**(-dT/Tau2))*R2
    B[3,0] = (1-np.e**(-dT/Tau3))*R3
    
    return B
  
from filterpy.kalman import UnscentedKalmanFilter as UKF

def f_nn(x,Dt,I):

    SOC = x[0]
    
    x = x[:,None]
    
    A = generateA(SOC)
    B = generateB(SOC)
    
    x_out = A@x + B*I

    return x_out[:,0]
    
def h_nn(x, I):
    
    SOC = x[0]
    V1 = x[1]
    V2 = x[2]
    V3 = x[3]
    
    R0 = np.interp(SOC,soc_aux,R0_aux)
    Em = np.interp(SOC,soc_aux1,Em_aux)
    
    Vt = Em - R0*I - V1 - V2 - V3 
    
    return np.array([Vt])
    
# Different choices for sigma points
from filterpy.kalman import JulierSigmaPoints 
sigmas = JulierSigmaPoints(n=4, kappa=0)
       
## UKF
ukf = UKF(dim_x=4,dim_z = 1,dt = dT,fx = f_nn,hx = h_nn, points=sigmas)
ukf.x = np.array([0.5,0,0,0])

ukf.P *= P0_Vn
ukf.P[0,0] = P0_soc
ukf.R *= (noise_std)**2
ukf.Q = np.diag([q_soc,q_Vn,q_Vn,q_Vn])


D['ecm']['Q'] = ukf.Q
D['ecm']['R'] = ukf.R

soc_ecm = []
Voltage_ecm = []
P = []

for j in tqdm(range(len(Time)), desc="UKF (ECM) estimation: "):
    
    z = Voltage_plant[j]
    ukf.update(z, I = curr)
    soc_ecm.append(ukf.x[0]*100)
    Voltage_ecm.append(ukf.hx(ukf.x,curr)[0])
    P.append(ukf.P[0,0])
    
    ukf.predict(I = curr)
    
soc_ecm = np.array(soc_ecm)

D['ecm']['soc'] = soc_ecm
D['ecm']['soc_upper'] = soc_ecm + (2*np.sqrt(P))*100 # +- 2 sigma interval
D['ecm']['soc_bottom'] = soc_ecm - (2*np.sqrt(P))*100
D['ecm']['Voltage'] = Voltage_ecm

##### SOC error
err = (D['ecm']['soc'] - D['plant']['soc'])**2
err = np.sqrt(np.mean(err[idx:]))

print('ECM SOC estimation error {:.2e} (RMSE)'.format(err))

####################################################################
############################# PLOTTING #############################
plot_voltage = 1
plot_soc = 1

############# PLOT 1 VOLTAGE comparison 

if plot_voltage==1:
    font_lab = {'family':'serif','color':'black','size':12}
    font_tit = {'family':'serif','color':'black','size':14}
    
    # error1 = abs((Voltage_plant - Voltage_mn)/Voltage_plant*100  ) 
    # error2 = abs((Voltage_plant - Voltage_ecm)/Voltage_plant*100 )
    
    fig, (ax1) = plt.subplots(1,1, figsize = (8,4), layout = 'constrained')
    ax1.plot(Time, Voltage_plant,'o', markerfacecolor='none', markeredgecolor='k', markersize=10, label = 'Plant')
    ax1.plot(Time, Voltage_mn,'b',linewidth=1, label='Prediction MIONet')
    ax1.plot(Time, Voltage_ecm,'r',linewidth=1, label='Prediction ECM')
    ax1.set_xlabel("Time [min]",fontdict=font_lab)
    ax1.set_ylabel("voltage [V]",fontdict=font_lab)
    ax1.set_title('Voltage comparison',fontdict = font_tit)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

if plot_soc==1:
    font_lab = {'family':'serif','color':'black','size':14}
    font_tit = {'family':'serif','color':'black','size':18}
    
    fig = plt.figure(figsize=(16, 6), layout='constrained')
    ax = fig.subplot_mosaic([['soc','soc_error']])
    
    ax["soc"].plot(Time, D['plant']['soc'],'k',linewidth = 1, label = 'Reference')
    ax['soc'].plot(Time,D['mionet']['soc'],'b',linewidth = 2,label = 'Prediction MIONet')
    ax['soc'].plot(Time,D['ecm']['soc'],'r',linewidth = 2,label = 'Prediction ECM')
    ax['soc'].fill_between(Time, D['mionet']['soc_bottom'], D['mionet']['soc_upper'], color = 'b',alpha=0.2)
    ax['soc'].fill_between(Time, D['ecm']['soc_bottom'], D['ecm']['soc_upper'], color = 'r',alpha=0.2)
    ax['soc'].legend()
    ax['soc'].grid(True, linestyle='--', alpha=0.7)
    ax['soc'].set_title('SOC',fontdict = font_tit)
    ax['soc'].set_xlabel('Time [min]',fontdict = font_lab)
    ax['soc'].set_ylabel('SOC [%]',fontdict = font_lab)
    ax['soc'].legend(fontsize = 14)
    
    ## Plot Current profile
    ax_curr= ax['soc'].inset_axes([0, -0.4, 1, 0.25]) 
    ax_curr.plot(Time,current[:len(Time)]/C,'navy',linewidth = 1)
    ax_curr.set_ylabel('i [x CR]',fontdict = font_lab)
    ax_curr.set_xlabel(current_string,fontdict = font_lab)
    ax_curr.set_xticklabels([])
    ax_curr.grid(True, linestyle='--', alpha=0.7)
    
    # # Create a zoomed-in region --> Optionally, you must set coordinates
    # x1, x2, y1, y2 = 30, 40, 45, 55  # Zoom region coordinates: X_left, X_rigth, Y_bottom, Y_upper
    # axins = ax['soc'].inset_axes([0.01, 0.01, 0.45, 0.45])  # [left, bottom, width, height]
    # axins.plot(Time, D['plant']['soc'],'k',linewidth = 1)
    # axins.plot(Time,D['mionet']['soc'],'b',linewidth = 1)
    # axins.plot(Time,D['ecm']['soc'],'r',linewidth = 1)
    # axins.fill_between(Time, D['mionet']['soc_bottom'], D['mionet']['soc_upper'], color = 'b',alpha=0.2)
    # axins.fill_between(Time, D['ecm']['soc_bottom'], D['ecm']['soc_upper'], color = 'r',alpha=0.2)
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])
    # # Add lines connecting the inset to the main plot
    # ax['soc'].indicate_inset_zoom(axins, edgecolor='red', linewidth=2, linestyle='--')
    
    error_nn = D['mionet']['soc'] - D['plant']['soc']
    error_ecm = D['ecm']['soc'] - D['plant']['soc']

    ax['soc_error'].plot(Time, error_nn,'b',linewidth = 1.5,label='Mionet')
    ax['soc_error'].plot(Time, error_ecm,'r',linewidth = 1.5,label='ECM')
        
    ax['soc_error'].plot(Time, np.zeros(len(Time)), '--', color = 'grey', alpha=0.8,linewidth = 1.5)
    ax['soc_error'].set_title('Error',fontdict = font_tit)
    ax['soc_error'].grid(True, linestyle='--', alpha=0.7)
    ax['soc_error'].set_xlabel('Time [min]',fontdict = font_lab)
    ax['soc_error'].set_ylabel('$\Delta$ SOC [%]',fontdict = font_lab)
    ax['soc_error'].legend(fontsize = 14)
    ax['soc_error'].set_ylim([-5,5])
    ax['soc_error'].fill_between(Time, D['mionet']['soc_bottom'] - D['plant']['soc'], D['mionet']['soc_upper'] - D['plant']['soc'], color = 'b', alpha=0.2)
    ax['soc_error'].plot(Time, D['mionet']['soc_bottom'] - D['plant']['soc'], '--', color = 'blue', alpha=0.7,linewidth = 1.5)
    ax['soc_error'].plot(Time, D['mionet']['soc_upper'] - D['plant']['soc'], '--', color = 'blue', alpha=0.7,linewidth = 1.5)
    
    ax['soc_error'].fill_between(Time, D['ecm']['soc_bottom'] - D['plant']['soc'], D['ecm']['soc_upper'] - D['plant']['soc'], color = 'r',alpha=0.2)
    ax['soc_error'].plot(Time, D['ecm']['soc_bottom'] - D['plant']['soc'], '--', color = 'red', alpha=0.7,linewidth = 1.5)
    ax['soc_error'].plot(Time, D['ecm']['soc_upper'] - D['plant']['soc'], '--', color = 'red', alpha=0.7,linewidth = 1.5)
    
    # fig.savefig(current_string+'.pdf', format='pdf')

    fig.align_ylabels()  # same as fig.align_xlabels(); fig.align_ylabels()
    plt.show()
    
plot_concentrations = 12

if plot_concentrations == 1:
    
    y_mn = D['mionet']['concentration']
    
    
    ## CONCENTRATION ON SURFACE
    fig, ax = plt.subplots(1,1,layout='constrained',figsize=(8,4))
    ax.plot(Time,y_mn[N-1,:],'c--',label = 'MIONet pos')
    
    if not use_experimental_data:
        y_pl = D['plant']['concentration']
        ax.plot(Time,y_pl[N-1,:],'b',label = 'PLANT pos')
        ax.plot(Time,y_pl[-1,:],'r',label = 'PLANT neg')
    
    ax.plot(Time,y_mn[-1,:],'m--',label = 'MIONet neg')
    
    ax.set_title('Concentrations on surface')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Concentrations [-]')
    ax.legend()

############ Saving the data
save = True
import pickle
version = 5

if save == True and use_experimental_data:
    
    s = str(CR)
    cr = s.replace('.', '') 
    
    file_name = current_string + '_'+ cr +'_'+'exp'+'_'+str(version)

    with open(file_name, 'wb') as f:
        pickle.dump(D, f)
        
    print(f'saved file {file_name}')
    
elif save == True:
    
    s = str(CR)
    cr = s.replace('.', '') 
    
    file_name = current_string + '_'+ cr +'_'+'sim'+'_'+str(version)

    with open(file_name, 'wb') as f:
        pickle.dump(D, f)
        
    print(f'saved file {file_name}')