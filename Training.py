"""
Physics-informed MIONet Training Script for Li-ion Battery Single Particle Model (SPM)
===================================================================

This script demonstrates how to train a Physics-Informed Multiple Input Operator Network (MIONet)
to approximate the solution of the Single Particle Model (SPM) for Li-ion battery electrochemistry.

Key Features:
-------------
- Trains the MIONet to predict the evolution of the negative electrode (anode) concentration profile.
- The model learns to solve the physical laws behind the model (PDE, BC, IC)
- Data is generated on-the-fly using synthetic sampling and pybamm-based simulation for validation.
- The training pipeline includes Adam/LBFGS optimizers, linear learning rate decay, and early stopping.
- Modular structure: supports customization for positive electrode training by adjusting parameter values and boundary conditions.

Instructions:
-------------
- To train a MIONet for the anode (negative electrode), run this script as-is.
- For the positive electrode, modify the relevant battery parameters and change the sign in the boundary condition loss accordingly.

"""

import torch 
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
import math
from torch.optim.lr_scheduler import LinearLR
import pybamm
# Set logging level to suppress informational messages
pybamm.set_logging_level("ERROR")

class GRF:
    """
    Samples random functions using a Gaussian Random Field (GRF) 
    with a given length scale and number of points.

    Args:
        T (float): Range of input domain.
        length_scale (float): Length scale for the RBF kernel.
        N (int): Number of sample points.
    """
    def __init__(self, T, length_scale, N):
        from sklearn.gaussian_process.kernels import RBF
        self.N = N
        self.x = np.linspace(0, T, num=N)[:, None]
        kernel = RBF(length_scale=length_scale)
        self.K = kernel(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, size):
        """Returns random function samples (shape: [size, N]) from the GRF."""
        u = np.random.randn(self.N, size)
        return np.dot(self.L, u).T

    def eval_batch(self, features, xs):
        """
        Interpolates sampled GRF features at new input locations xs.
        Shifts results so minimum is zero and scales.
        """
        interp_values = np.vstack([np.interp(xs, np.ravel(self.x), y).T for y in features])
        min_vals = interp_values[:, 0]
        shifted = interp_values - min_vals.reshape(len(features[:, 0]), 1)
        return shifted / 10

class loss_class():
    """
    Tracks and plots different loss components during MIONet training.
    """

    def __init__(self):
        self.loss_tot = []
        self.loss_val = []
        self.loss_pde = []
        self.loss_ic = []
        self.loss_bc = []
        self.training_time = []
        self.evaluation = []
        self.best_val = []
        
        
    def update(self, tot_loss, val_loss, pde_loss, ic_loss, bc_loss):
        
        self.loss_tot.append(tot_loss)
        self.loss_val.append(val_loss)
        self.loss_pde.append(pde_loss)
        self.loss_ic.append(ic_loss)
        self.loss_bc.append(bc_loss)
        
    def plotting(self):
        epo = range(len(self.loss_tot))
        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(6,4))
        ax1.set_yscale('log')
        ax1.plot(epo,self.loss_tot,'b',label='TOTAL loss')
        ax1.plot(epo,self.loss_val,'r',label='VALIDATION loss')
        ax1.set_title('LOSS')
        ax1.legend()
        ax1.grid(visible=True)
        
        ax2.set_yscale('log')
        ax2.plot(epo,self.loss_pde,'b',label='PDE loss')
        ax2.plot(epo,self.loss_ic,'r',label='IC loss')
        ax2.plot(epo,self.loss_bc,'g',label='BC loss')
        ax2.set_title('LOSS')
        ax2.legend()
        ax2.grid(visible=True)

class MIONet(nn.Module): 
    def __init__(self, branch1_dim, branch2_dim, trunk_dim, hidden_dim, number_of_hidden_layers): 
        super(MIONet, self).__init__()
        
        activation_fn=nn.Tanh()

        layer_width = branch1_dim
        layers1 = []
        
        for n_layer in range(number_of_hidden_layers):
            layers1.append(nn.Linear(layer_width,hidden_dim))
            layers1.append(activation_fn)
            layer_width = hidden_dim
        
        self.branch1 = nn.Sequential(*layers1)
        
        layer_width = branch2_dim
        layers2 = []
        for n_layer in range(number_of_hidden_layers):
            layers2.append(nn.Linear(layer_width,hidden_dim))
            layers2.append(activation_fn)
            layer_width = hidden_dim
        
        self.branch2 = nn.Sequential(*layers2)
        
        layer_width = trunk_dim
        layers3 = []
        layers3.append(nn.Linear(layer_width,hidden_dim))
        layer_width = hidden_dim
        for n_layer in range(1,number_of_hidden_layers):
           layers3.append(nn.Linear(layer_width,hidden_dim))
           layers3.append(activation_fn)
       
        self.trunk = nn.Sequential(*layers3) 

        for layer in self.branch1:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
                
        for layer in self.branch2:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
        
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
        
        
    def forward(self, coordinate, cs_0, current): # Process each input stream 
        out1 = self.branch1(cs_0) 
        out2 = self.branch2(current) 
        out_trunk = self.trunk(coordinate) 
        dot_product = torch.mul(torch.mul(out1,out2),out_trunk)
        output = torch.sum(dot_product,2)
        return torch.unsqueeze(output,2)

def create_loader(tensor, num_batches, shuffle=True):
    dataset_size = len(tensor)
    batch_size = dataset_size // num_batches

    # To handle cases where the dataset isn't perfectly divisible
    if dataset_size % num_batches != 0:
        batch_size += 1

    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader

class PINN():
    def __init__(self,NN_model, lr = 1e-4):
        self.network = NN_model
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
    def makeNetwork(self, t, ksi, cs_0, curr):
        t_ksi = torch.cat((t,ksi),2)
        return self.network(t_ksi, cs_0, curr)    
    
    def loss_pde(self, X):
        
        t = X[:,:,:1].requires_grad_()
        ksi = X[:,:,1:2].requires_grad_()
        cs_0 = X[:,:1,2:2+b1]
        i = X[:,:1,2+b1:2+b1+b2]
        
        y = self.makeNetwork(t,ksi,cs_0,i)
        y_t = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y),  create_graph=True)[0]
        y_ksi = torch.autograd.grad(y, ksi, grad_outputs=torch.ones_like(y),  create_graph=True)[0]
        y_ksi_ksi = torch.autograd.grad(y_ksi, ksi, grad_outputs=torch.ones_like(y),  create_graph=True)[0]

        loss = (2*y_ksi/ksi + y_ksi_ksi) - y_t*Rs**2/Ds
        return torch.mean(loss**2)
    
    def loss_ic(self, XX):
        t = XX[:,:,:1]
        ksi = XX[:,:,1:2]
        cs_0 = XX[:,:1,2:2+b1]
        i = XX[:,:1,2+b1:2+b1+b2]
        y_true = XX[:,:,-1:]

        y = self.makeNetwork(t,ksi,cs_0,i)
        
        loss = y_true - y
        
        return torch.mean(loss**2)
    
    def loss_bc1(self, X):
   
        t = X[:,:,:1]
        ksi = X[:,:,-1:].requires_grad_()
        cs_0 = X[:,:1,2:2+b1]
        i = X[:,:1,2+b1:2+b1+b2]
        
        y = self.makeNetwork(t,ksi,cs_0,i)
        y_ksi = torch.autograd.grad(y, ksi, grad_outputs=torch.ones_like(y),  create_graph=True)[0]
        return torch.mean(y_ksi**2)
    
    def loss_bc2(self, X):
        t = X[:,:,:1]
        ksi = X[:,:,1:2].requires_grad_()
        cs_0 = X[:,:1,2:2+b1]
        i = X[:,:1,2+b1:2+b1+b2]
        
        y = self.makeNetwork(t,ksi,cs_0,i)
        y_ksi = torch.autograd.grad(y, ksi, grad_outputs=torch.ones_like(y),  create_graph=True)[0]
        loss = y_ksi + sign*i/j0_aux
        return torch.mean(loss**2)
    
    def loss_dd(self, XX):
        t = XX[:,:,:1]
        ksi = XX[:,:,1:2]
        cs_0 = XX[:,:,2:2+b1]
        i = XX[:,:,2+b1:2+b1+b2]
        y_true = XX[:,:,-1:]

        y = self.makeNetwork(t,ksi,cs_0,i)
        
        loss = y_true - y
        
        return torch.mean(loss**2)
    
    def prepare_data(self, data_loader, num_batches):
        
        X_pde = data_loader.data_pde()
        X_ic = data_loader.data_ic()
        X_bc = data_loader.data_bc()
        X_val = data_loader.data_val() # Validation data are prepared in a slgithly different manner

        # Create DataLoaders
        loader_pde = create_loader(X_pde, num_batches)
        loader_ic = create_loader(X_ic, num_batches)
        loader_bc = create_loader(X_bc, num_batches)
        loader_val = np.array_split(X_val, num_batches, axis=1)
        
        return loader_pde, loader_ic, loader_bc, loader_val
        
    def train(self, num_epochs, data_gen):
        lossTracker = loss_class()
        self.network.train()
        best_val_loss = float('inf')
        best_val_loss_epoch = 0
        change_opt = True
        epoch_change = num_epochs
        scheduler = LinearLR(self.optimizer,
                             start_factor=1.0,  # start at 100% of 0.01
                            end_factor=0.01,    # end at 0 (zero learning rate)
                            total_iters=num_epochs    # over 100 epochs
                            )
        
        num_batches = 10
        loader_pde, loader_ic, loader_bc, loader_val = self.prepare_data(data_gen, num_batches)
        
        ## Prep the DSs
        def closure():
            self.optimizer.zero_grad()
            loss_pde = self.loss_pde(X_pde)*W_pde
            loss_ic = self.loss_ic(X_ic)*W_ic
            loss_bc1 = self.loss_bc1(X_bc)*W_bc
            loss_bc2 = self.loss_bc2(X_bc)*W_bc
            loss = loss_pde + loss_ic + loss_bc1 + loss_bc2
            loss.backward()
            
            PDE_loss.append(loss_pde.item())
            IC_loss.append(loss_ic.item())
            BC_loss.append(loss_bc1.item()+loss_bc2.item())
            
            return loss 
        from tqdm import tqdm
        for epoch in tqdm(range(num_epochs), desc = 'Training : '):
            
            epoch_loss = 0
            val_loss = 0
            
            PDE_loss = []; BC_loss = []; IC_loss = []
            
            for batch_pde, batch_ic, batch_bc, X_val in zip(loader_pde, loader_ic, loader_bc, loader_val):

                X_pde = batch_pde[0]
                X_ic = batch_ic[0]
                X_bc = batch_bc[0]
                loss = self.optimizer.step(closure)                
                epoch_loss += loss.item()
                with torch.no_grad():  # Disable gradient tracking
                    val_loss += self.loss_dd(X_val).item()

            epoch_loss /= num_batches
            val_loss /= num_batches
            pde_loss = np.mean(np.array(PDE_loss))
            ic_loss = np.mean(np.array(BC_loss))
            bc_loss = np.mean(np.array(IC_loss))
            
            lossTracker.update(epoch_loss, val_loss, pde_loss,
                                ic_loss, bc_loss)
            
            if epoch%20==0 or epoch<10:
                print('epoch {}, loss = {:10.2e}, val. = {:10.2e}'.format(epoch,loss,val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = epoch
                torch.save(self.network, name_nn)

        
            if (epoch - best_val_loss_epoch > 100 and epoch > int(0.6*num_epochs)) or epoch > 0.8*num_epochs and change_opt:
                change_opt = False
                self.optimizer = torch.optim.LBFGS(self.network.parameters(),
                                    lr = 1e-2,
                                    max_iter=100,
                                    tolerance_grad=1e-6,
                                    tolerance_change=1e-9,
                                    line_search_fn = "strong_wolfe")
                print(f"Optimizer change at iteration {epoch}.")
                epoch_change = epoch
                
            if math.isnan(loss.item()):
                print(f"NaN detected at iteration {epoch}, stopping loop.") # Sometimes Nan appears when using LBFGS with a too high lr
                break
            
            if epoch - best_val_loss_epoch > 50 and not change_opt and epoch - epoch_change > 50:
                print(f"Early stopping at iteration {epoch}, stopping loop.") # Early stopping
                break
            
            if epoch % 100 == 0:
                loader_pde, loader_ic, loader_bc, _ = self.prepare_data(data_gen, num_batches)
                ## Resample training data every 100 epochs --> Optionally
            
            scheduler.step()
               
        print('best val loss {:10.4e}'.format(best_val_loss))
            
        return lossTracker
         
    def evaluation(self, X_eval):
        
        loss_eval = self.loss_dd(X_eval).item()
        
        print('Evaluation loss on an unseen set of data: {:10.2e} (MSE)'.format(loss_eval))
        
        return loss_eval
        
    def predict(self, res, dT, Cs_0, Curr):
        
        T = np.ones([1,res,1])*dT
        Ksi = np.zeros((1,res,1))
        Ksi[0,:,0] = np.linspace(0,1,res)
        
        t = torch.tensor(T,dtype=torch.float32)
        ksi = torch.tensor(Ksi,dtype=torch.float32)
        cs_0 =torch.tensor(Cs_0,dtype=torch.float32)
        curr =torch.tensor(np.array([[Curr]]),dtype=torch.float32)

        y = self.makeNetwork(t,ksi,cs_0, curr)
        return y.detach().numpy()[0,:,:1]
    
class data():   
    
    """
        Generates and prepares synthetic datasets for training and validating the MIONet in battery diffusion-reaction problems.
    
        This class creates:
          - The dataset to enforce the PDE loss
          - The dataset to enforce the IC loss
          - The dataset to enforce the BC loss
          - Reference ("ground-truth") validation data using pybamm simulations for a 'data-driven' validation loss -> to give a more 
          tangible validation loss; it could be substituted with a PI validation loss (a different set of collocation points and input functions)
    
        Main methods:
          - data_pde(...):      Generates collocation points and input functions for enforcing the PDE.
          - data_ic(...):       Generates collocation points and input functions for enforcing the IC.
          - data_bc(...):       Generates collocation points and input functions for enforcing the BC.
          - data_val(...):      Produces validation data by simulating a reference SPM battery using pybamm.
    
        The class uses a GRF to randomly sample spatial profiles for input features (e.g., initial concentration).
    
        Args:
            N_pde (int): Number of PDE (collocation) batches.
            Ny_pde (int): Points per PDE batch.
            N_ic (int): Number of IC (initial condition) batches.
            Ny_ic (int): Points per IC batch.
            N_bc (int): Number of BC (boundary condition) batches.
            Ny_bc (int): Points per BC batch.


        """
    
    def __init__(self, N_pde, Ny_pde, N_ic, Ny_ic, N_bc, Ny_bc):
        self.space = GRF(T=1, length_scale=0.3, N=20)
        self.N_pde = N_pde
        self.Ny_pde = Ny_pde
        self.N_ic = N_ic
        self.Ny_ic = Ny_ic
        self.N_bc = N_bc
        self.Ny_bc = Ny_bc

    def data_pde(self, N_pde=None, Ny_pde=None):
        
        if N_pde == None or Ny_pde == None :
            N_pde = self.N_pde
            Ny_pde = self.Ny_pde
        
        
        X_aux = np.zeros([1, Ny_pde, 2 + b1 + b2])
        
        soboleng_pde = torch.quasirandom.SobolEngine(dimension=2)
        coll_points_pde =  soboleng_pde.draw(Ny_pde).detach().numpy()

            
        coll_points_pde[:,0]*=T_end
        coll_points_pde[coll_points_pde[:,1]<=1e-2, 1] = 1e-2 # It needs to avoid dividing by zero
        
        eval_pts = np.linspace(0,1,b1) #Position of sensors
        first_time = True
        
        I = np.random.normal(0,1,(N_pde))*C
        
        for idx in range(N_pde):
            
            features = self.space.random(1)
            c_0 = self.space.eval_batch(features,eval_pts)

            X_aux[0,:,:2]=coll_points_pde
            X_aux[0,:,2:2+b1]= c_0
            X_aux[0,:,-1]= I[idx]        
            
            if first_time:
                
                X_pde = X_aux.copy()
                first_time = False
            
            else:
                X_pde = np.concatenate((X_pde,X_aux),axis = 0)
    
        
        return torch.tensor(X_pde, dtype=DTYPE, device= DEVICE) 

    def data_ic(self, N_ic=None, Ny_ic=None):
        
        if N_ic == None or Ny_ic == None :
            N_ic = self.N_ic
            Ny_ic = self.Ny_ic
            
        X_aux = np.zeros([1, Ny_ic, 2 + b1 + b2 + 1])
        
        soboleng_ic = torch.quasirandom.SobolEngine(dimension=1)
        coll_points_ic =  soboleng_ic.draw(Ny_ic).detach().numpy()
        
        eval_pts = np.linspace(0,1,b1) #Position of sensors
        first_time = True

        I = np.random.normal(0,1,(N_ic))*C
        
        for idx in range(N_ic):
            
            features = self.space.random(1)
            c_0 = self.space.eval_batch(features,eval_pts)
            c0_x = self.space.eval_batch(features,coll_points_ic).T
            
            X_aux[0,:,1:2]=coll_points_ic
            X_aux[0,:,2:2+b1]= c_0
            X_aux[0,:,2+b1:2+b1+b2]= I[idx]
            X_aux[0,:,-1:]=c0_x
            
            if first_time:
                
                X_ic = X_aux.copy()
                first_time = False
            
            else:
                X_ic = np.concatenate((X_ic,X_aux),axis = 0)
    
        return torch.tensor(X_ic, dtype=DTYPE, device= DEVICE) 
    
    def data_bc(self, N_bc=None, Ny_bc=None):
        
        if N_bc == None or Ny_bc == None :
            N_bc = self.N_bc
            Ny_bc = self.Ny_bc
        
        X_aux = np.zeros([1, self.Ny_bc, 2 + b1 + b2 + 1])
        
        soboleng_bc = torch.quasirandom.SobolEngine(dimension=1)
        coll_points_bc =  soboleng_bc.draw(self.Ny_bc).detach().numpy()
        
        # coll_points_bc*=0.98
        # coll_points_bc+=0.02
        coll_points_bc*=T_end
        
        eval_pts = np.linspace(0,1,b1) #Position of sensors
        first_time = True
        
        # I = np.random.uniform(-2,2,(self.N_bc))*CR1
        I = np.random.normal(0,1,(self.N_bc))*C
        
        for idx in range(self.N_bc):
            
            features = self.space.random(1)
            u_0 = self.space.eval_batch(features,eval_pts)
            
            X_aux[0,:,:1]=coll_points_bc
            X_aux[0,:,1] = 1
            X_aux[0,:,-1] = 0
            X_aux[0,:,2:2+b1]= u_0
            X_aux[0,:,-2]= I[idx]
            
            if first_time:
                
                X_bc = X_aux.copy()
                first_time = False
            
            else:
                X_bc = np.concatenate((X_bc,X_aux),axis = 0)
    
        return torch.tensor(X_bc, dtype=DTYPE, device= DEVICE) 
    
    def data_val(self):
        """
        Idea is to use reference data for the validation: a DD validation
        
        A long pybamm simulation is split in T_tr slices: In each slice the first column is used as IC and the other points are used to evaluate the loss
        """
        import pybamm
        
        
        T_sim = 5000; dT = 0.1
        res = b1
        T_tr = 2.5 # Time used for training the nn model

        batt_model = pybamm.lithium_ion.SPM()
        # param = batt_model.default_parameter_values
        param.set_initial_stoichiometries(0.5)
        
        t_eval = np.linspace(0, T_sim , int(T_sim/dT)+1)
        current = np.ones([len(t_eval)])
        dt = 100*T_tr; k = 0
        for t in t_eval:
            if t%dt==0:
                II = np.random.normal(0,1)*C
            current[k]=II
            k += 1
        
        
        var_pts = {
        "x_n": 10,  # negative electrode
        "x_s": 10,  # separator
        "x_p": 10,  # positive electrode
        "r_n": res,  # negative particle
        "r_p": res,  # positive particle
        }

        # Defining the experiment
        drive_cycle_power = np.column_stack([t_eval, current])
        experiment = pybamm.Experiment([pybamm.step.current(drive_cycle_power)])
        del drive_cycle_power
        
        sim = pybamm.Simulation(batt_model, parameter_values=param, var_pts=var_pts, experiment = experiment)
        solution = sim.solve()
        
        ## Results
        time1 = solution["Time [s]"].entries
        y_true =  np.mean(solution[f"{Electrode} particle concentration"].entries, axis = 1)   
                
        X_dd = np.zeros([1,res*len(time1),2+b1+b2+1])
        r = np.linspace(0,1,res)
        k = 0
        
        for j in range(len(time1)):
            
            Time = time1[j]%(T_tr)
            
            if Time == 0:
                
                c0 = y_true[0,j].copy()
                cs_0 = np.interp(np.linspace(0,1,b1),r,y_true[:,j] - c0)[None,:]
                k+= 1
                
            X_dd[0,res*j:res*(j+1),0] = Time
            X_dd[0,res*j:res*(j+1),1] = r[-res:]
            X_dd[0,res*j:res*(j+1),2:2+b1] = cs_0
            X_dd[0,res*j:res*(j+1),2+b1:2+b1+b2] = current[j]
            X_dd[0,res*j:res*(j+1),-1] = y_true[-res:,j] - c0

        return torch.tensor(X_dd, dtype = DTYPE, device = DEVICE) 
    
###############################################################################

t = 2; b1 = 31; b2 = 1  ## Dimensions of the branch nets
T_end = 2.5             ## The MIONet will be trained within this time interval --> 2-3 times the time step imposed in the latter observer
W_pde, W_ic, W_bc  = 1e-2, 1e3, 10  ## PI losses weights

## Choose here to train the negative or positive MIONet
Electrode = 'Positive'

## Set here the ELECTROCHEMICAL parameters
param = pybamm.ParameterValues("Prada2013")
Ds = param[f'{Electrode} electrode diffusivity [m2.s-1]'] # Prada

# param = pybamm.ParameterValues("Mohtat2020")
# Diff_neg = param['Negative electrode diffusivity [m2.s-1]']; Ds=Diff_neg(0.5,298.15).value

F = param['Faraday constant [C.mol-1]'] 
epsilon = param[f'{Electrode.capitalize()} electrode active material volume fraction']
Rs = param[f'{Electrode.capitalize()} particle radius [m]']
L = param[f'{Electrode.capitalize()} electrode thickness [m]']
cs_max = param[f'Maximum concentration in {Electrode.lower()} electrode [mol.m-3]']
# Automatically assigns parameter values based on the selected electrode
a = 3*epsilon/Rs
W=param['Electrode width [m]'] # Electrode height [m]
H=param['Electrode height [m]']
C = param['Nominal cell capacity [A.h]']
j0_aux = W*H*Ds*cs_max*F*a*L/Rs ## Appears in BC1
sign = 1 if Electrode == 'Negative' else -1
DEVICE = torch.device('cuda')
DTYPE = torch.float32

## Decide if to make a new MIONet or continue with a trained one
New_MIONet = True

if New_MIONet:
    net = MIONet(branch1_dim=b1,branch2_dim=1, 
                  trunk_dim=t, hidden_dim=100, 
                  number_of_hidden_layers=7)
    name_nn = f'best_{Electrode.lower()}_model_prada.pt'
else: 
    net = torch.load('best_neg_model.pt', weights_only=False) ## Select here if you want retrain a model 
    name_nn = f'best_{Electrode.lower()}_model_retrained_prada.pt'

net.to(DEVICE)

## Prepare data
data_generator = data(N_pde=150,Ny_pde=10000, N_ic=1000, Ny_ic=100, N_bc=150, Ny_bc=1000)

## Train
model = PINN(net, lr=1e-4)  #Initial learning rate 
from time import time
t0 = time()
pi_training = model.train(1000, data_generator)
pi_training.plotting()

print('time needed: {:.2f} hr'.format((time()-t0)/3600))

## Evaluation on an unseen dataset
_eval = model.evaluation(data_generator.data_val())

## Save loss history
import pickle
import os
folder_path = 'C:/Users/alexa/OneDrive - Politecnico di Milano/Desktop/PHD_python/SPM_pimionet/SPM_PIMIONet_new/Mohtat/Training/Plotting'
file_path = os.path.join(folder_path, f'loss_history_{Electrode}_prada2.pkl')
with open(file_path, 'wb') as f:
    pickle.dump(pi_training.__dict__, f)
