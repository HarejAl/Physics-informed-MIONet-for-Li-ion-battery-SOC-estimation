"""
Battery plant simulation and SOC estimation benchmark (MIONet vs. ECM)
---------------------------------------------------------------------

Overview
- Simulates a lithium-ion battery "plant" with PyBaMM using parameters from Mohtat2020.
- Solves the plant for a user-selected current profile (CC, DST, or UDDS).
- Treats the PyBaMM solution as ground truth (with optional sensor noise).
- Runs two observers on the same input/output data to estimate State of Charge (SOC):
    1) Physics-informed observer based on MIONet (one network per electrode) + UKF.
    2) Equivalent Circuit Model (ECM) + UKF.
- Compares SOC and terminal voltage predictions, uncertainty bands, and RMSE.
- Optionally saves all results to disk.

Dependencies
- numpy, matplotlib, scipy, torch, pybamm, filterpy, tqdm
- local `utils` module providing:
  * plant(...)            -> plant utilities (profiles, reference solver, SOC calculator, etc.)
  * observer_mionet(...)  -> MIONet-based observer utilities (voltage mapping, SOC calculator)
  * PINN(...)             -> wrapper around saved MIONet (positive/negative) .pt files

Inputs & configuration (edit here)
- `T_amb`: ambient temperature [°C] for the plant.
- `thermal_model`: include a lumped thermal model in the plant (True/False).
- `soc_0`: initial SOC (%) for the plant (used in ground-truth simulation).
- `CR`: C-rate magnitude for selected profile (negative for discharge in CC profile as coded).
- Current profile selection (uncomment ONE):
    * `plant_utils.CC(...)`
    * `plant_utils.DST(...)`
    * `plant_utils.UDDS(...)`
- Time grid: `T_end` total time [s], `dT` time step [s].
- Discretization: `N` nodes per electrode for MIONet state dimension (= 2*N total).
- Noise: `noise_std` for terminal-voltage measurement noise.
- File paths:
    * MIONet weights: `pos_saved_model.pt`, `neg_saved_model.pt`
    * ECM parameters: `ECM_parameters_3.mat`
- Plot toggles: `plot_voltage`, `plot_soc`.
- Save toggle: set `save = 1` and define `folder_path` to export results.

Workflow
1) Plant setup
   - Load Mohtat2020 parameter set via PyBaMM, set ambient/initial temperature.
   - Choose current profile and build time vector `t_eval`.
   - Solve the plant with `plant_utils.ref_solution(...)` to obtain time, states, voltage, temperature.
   - Add Gaussian noise to plant terminal voltage (to emulate measurements).

2) MIONet observer (electrochemical surrogate + UKF)
   - Load positive/negative MIONet models from `.pt` files and wrap with `utils.PINN`.
   - Create an observer interface `obs_utils = utils.observer_mionet(...)` at nominal temperature.
   - Define UKF state as concatenated electrode surface stoichiometries (length = 2*N).
   - Process model `F`: advances both electrodes via MIONet surrogates.
   - Measurement model `H`: maps states + current to terminal voltage via `obs_utils.Voltage(...)`.
   - Tune UKF covariances (P, Q, R) and run update/predict loop over all time steps.
   - Compute SOC from states, keep 2σ confidence bounds, and compute SOC RMSE.

3) ECM observer (second-order RC + ohmic + UKF)
   - Load SOC-dependent ECM parameters (R0, Em, Rx, taux) from `ECM_parameters_3.mat`.
   - Define state vector x = [SOC, V1, V2, V3]; build A(SOC), B(SOC) for discrete-time model.
   - Process model `f_nn`: SOC integration + RC branch dynamics.
   - Measurement model `h_nn`: terminal voltage from OCV(Em) minus ohmic and RC drops.
   - Tune UKF covariances and run the same update/predict loop.
   - Store SOC, 2σ bounds, voltage, and compute SOC RMSE.

4) Visualization
   - Voltage comparison plot: plant (reference) vs. MIONet vs. ECM.
   - SOC plot with confidence bands; inset shows current profile (in C-rate units).
   - SOC error plot (prediction – reference) with uncertainty envelopes.

5) Output
   - Dictionary `D` collects plant/observer time series, uncertainties, and metadata.
   - Optional pickle save to `folder_path` (file name encodes profile, C-rate, and temperature).

Notes
- Units: time is plotted in minutes; current inset shows i/C (i in A, normalized by nominal capacity C in Ah).
- UKF sigma points use Julier scheme (kappa = 0).
- Initial observer states are set around mid-stoichiometry; covariances are illustrative and may be retuned.
- If the PyBaMM solver terminates early (e.g., voltage limits), use returned `Time` instead of `t_eval`.
- Set random seeds for reproducibility (`numpy` only here; add torch seeds if training is included elsewhere).
"""

import numpy as np
import matplotlib.pyplot as plt
import pybamm
import torch 
import torch.nn as nn 
import scipy
import utils

class MIONet(nn.Module): 
        
    def forward(self, coordinate, cs_0, current): # Process each input stream 
        out1 = self.branch1(cs_0) 
        out2 = self.branch2(current) 
        out_trunk = self.trunk(coordinate) 
        dot_product = torch.mul(torch.mul(out1,out2),out_trunk)
        output = torch.sum(dot_product,1)
        return output

## Set logging level to suppress informational messages
pybamm.set_logging_level("ERROR")

## Setting seed for reproducibility
np.random.seed(42)  

## Set 
font_lab = {'family':'serif','color':'black','size':12} # Font for labels
font_tit = {'family':'serif','color':'black','size':14} # Font for labels

T_amb = 25

param = pybamm.ParameterValues("Prada2013")
param['Ambient temperature [K]'] = 273.12 + T_amb
param['Initial temperature [K]'] = param['Ambient temperature [K]'] 
C = param['Nominal cell capacity [A.h]'] # Capacity [Ah]

#### Setting common parameters 
T_end = 2400
dT = 2 # Time step --> It can be setup up to 2.5 sec
N = 15 # Number of states in each electrode
r = np.linspace(0,1,N) #n is the number of points for the discretization in pybamm

plant_utils = utils.plant(param, N) ## Initializing the class plant

## Initializing the dictionary for solution
D = {
     'plant':{},
     'mionet':{},
     'ecm':{}
     }

## Parameters to be modified
soc_0 = 0.8
CR = 3
thermal_model = False

## Setting current profile
t_eval = np.linspace(0, T_end , int(T_end/dT)+1)

# current, current_string = plant_utils.CC(t_eval, CR = -CR); soc_0 = 0.1
# current, current_string = lib.DST(t_eval, CR)
current, current_string = plant_utils.UDDS(t_eval, CR); soc_0 = 0.8
# Uncomment the desired current profile
print('Comparison between ECM and Mionet for '+current_string+' at {} CR for total time {:.2f} hr'.format(CR, T_end/3600))

## REFERENCE solution with pybamm
Time, y_ref, Voltage_ref, Temp = plant_utils.ref_solution(soc_0, current, t_eval, thermal_model)
# Use Time instead of t_eval in case the simulation is triggered earlier --> pybamm does it automatically iin case the min value of voltage is reached
noise_std = 1e-3
Voltage_plant = Voltage_ref + np.random.normal(0,noise_std,(len(Voltage_ref))) 

D['plant']['soc'] = plant_utils.SOC_calculator(y_ref)
D['plant']['Voltage'] = Voltage_plant
D['plant']['time'] = Time
D['plant']['current'] = current
D['plant']['current_string'] = current_string
D['plant']['dT'] = dT
D['plant']['measurment_noise'] = noise_std

current += np.random.normal(0,noise_std,(len(current))) # Adding noise to the current profile --> Mimic realistic scenario

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
net = torch.load('pos_saved_model.pt',map_location=torch.device('cpu'),weights_only=False)
model_p = utils.PINN(net, N) 

net = torch.load('neg_saved_model.pt',map_location=torch.device('cpu'),weights_only=False)
model_n = utils.PINN(net, N)

## Define again the param set with nominal temperature!
# param = pybamm.ParameterValues("Mohtat2020")
obs_utils = utils.observer_mionet(param, N)

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
    U = obs_utils.Voltage(cs_k[:],I,)
    return np.array([U])

ukf = UKF(dim_x=2*N, dim_z = 1, dt = dT, fx = F, hx = H, points=sigmas)

## Initializing the sta vector --> stoichiometries are initialized with an initial SOC of 50%
ukf.x = np.ones((2*N))
ukf.x[:N] *= 0.5*0.89   #(0.89 is the max stochiometry in the cathode)
ukf.x[N:] *= 0.5*0.83   #(0.83 is the max stochiometry in the anode)

## Initializing covariance matrices --> Part of Kalman filter tuning
ukf.P *= 1e-3
ukf.R *= (noise_std)**2
ukf.Q *= (1e-5)**2

first_time = True
Voltage_mn = []
p_10 = len(Time) // 10; k = 0
P = []; y = []

from tqdm import tqdm

for j in tqdm(range(len(Time)), desc="UKF (MIONet) estimation: "):

    # Measurement
    z = np.array([Voltage_plant[j]])
    
    # Update step
    if j > 0:
        ukf.update(z = z, I = current[j])

    ## Saving values
    P.append(np.diag(ukf.P))
    y.append(ukf.x)
    Voltage_mn.append(ukf.hx(ukf.x,I = current[j])[0])
    
    ## Propagate step
    ukf.predict(I = current[j])

## Storing saved variables in the dictionary
P = np.array(P).T; y = np.array(y).T

D['mionet']['soc'] = plant_utils.SOC_calculator(y)
D['mionet']['concentration'] = y

# CONFIDENCE intervals
conf = np.sqrt(P)*2 # +- 2 sigma interval
conf[N:] *= -1
D['mionet']['soc_upper'] = obs_utils.SOC_calculator(y + conf)
D['mionet']['soc_bottom'] = obs_utils.SOC_calculator(y - conf)
D['mionet']['Voltage'] = Voltage_mn

##### SOC error
idx = int(600/dT) ## Steady state error: we neglect the first 10 min transient
err = (D['plant']['soc'] - D['mionet']['soc'])**2
err = np.sqrt(np.mean(err[idx:]))

print('MIONET SOC estimation error {:.2e} (RMSE)'.format(err))

##################################################################################################
##################################### 1 SOLUTION with ECM ########################################
##### LOADING ECM PARAMETERS
parameters = scipy.io.loadmat('ECM_parameters_3.mat')
R0_aux = parameters['R0'].T[:,0]
soc_aux = parameters['SOC'].T[:,0]
soc1 = np.linspace(0.1,0,len(soc_aux))/100
soc_aux += soc1
Em_aux = parameters['Em'].T[:,0]
tau_x = parameters['taux'].T
R_x = parameters['Rx'].T

##### SETTING PARAMETERS
Q_batt = C*3600   # Ah -> As  
q_soc = (1e-2)**2
q_Vn  = (2e-4)**2
P0_vn = 1e-6
P0_soc = 1e-8

P0_soc = 5e-2
P0_Vn = 1e-1
soc_0 = 0.5  # Initial value of SOC

V1 = 0; V2=0; V3 = 0

x = np.array([[soc_0,V1,V2,V2]]).T

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
    Em = np.interp(SOC,soc_aux,Em_aux)
    
    Vt = Em - R0*I - V1 - V2 - V3 
    
    return np.array([Vt])
    
# Different choices for sigma points
from filterpy.kalman import JulierSigmaPoints 
sigmas = JulierSigmaPoints(n=4, kappa=0)
       
## UKF
ukf = UKF(dim_x=4,dim_z = 1,dt = 1,fx = f_nn,hx = h_nn, points=sigmas)
ukf.x = np.array([soc_0,0,0,0])

ukf.P *= P0_vn
ukf.P[0,0] = P0_soc
ukf.R *= noise_std**2
ukf.Q = np.diag([q_soc,q_Vn,q_Vn,q_Vn])

soc_ecm = []
Voltage_ecm = []
P = []

for j in tqdm(range(len(Time)), desc="UKF (ECM) estimation: "):
    
    z = Voltage_plant[j]
    ukf.update(z, I = current[j])
    soc_ecm.append(ukf.x[0]*100)
    Voltage_ecm.append(ukf.hx(ukf.x,current[j])[0])
    P.append(ukf.P[0,0])
    
    ukf.predict(I = current[j])
    
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
    
    error1 = abs((Voltage_plant - Voltage_mn)/Voltage_plant*100  ) 
    error2 = abs((Voltage_plant - Voltage_ecm)/Voltage_plant*100 )
    
    fig, (ax1) = plt.subplots(1,1, figsize = (8,4), layout = 'constrained')
    ax1.plot(Time, Voltage_plant,'k--',linewidth=1, label='Reference')
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

############ Saving the data
save = False

if save == True:
    import pickle
    
    file_name = current_string + '_'+str(CR) +'_'+str(T_amb)

    with open(file_name, 'wb') as f:
        pickle.dump(D, f)
        
    print(f'saved file {file_name}')