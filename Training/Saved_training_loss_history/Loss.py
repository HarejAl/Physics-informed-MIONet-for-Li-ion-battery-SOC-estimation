
"""
PLOTTING training loss and some other figures for the papaer
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


# Font settings to match LaTeX document
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 12
})

cm = 1/1.64 

# Plotting
plot = 12

if plot == 1: 

    # Load the saved attributes:
    with open('loss_history_Positive_prada_TL.pkl', 'rb') as f:
        saved_attributes = pickle.load(f)
        
    num_epochs = 1000
    
    training_loss = np.array(saved_attributes['loss_tot'])[:num_epochs]
    validation_loss = np.array(saved_attributes['loss_val'])[:num_epochs]

    # training_loss = np.array(loss_loaded.loss_pde)
    # validation_loss = np.array(loss_loaded.loss_ic)

    # training_loss = training_loss[training_loss != np.array([None])]
    # validation_loss = validation_loss[validation_loss != np.array([None])]

    num_epochs = len(training_loss)
    epochs = np.arange(1, num_epochs+1)

    # Find minimum validation loss
    min_val_loss_epoch = np.argmin(validation_loss) + 1
    min_val_loss = validation_loss[min_val_loss_epoch - 1]

    # Define the epoch where optimizer changed
    Ep_optimizer = 700

    # Find minimum validation loss
    min_val_loss_epoch = np.argmin(validation_loss) + 1
    min_val_loss = validation_loss[min_val_loss_epoch - 1]
    
    plt.figure(figsize=(9*cm, 9*cm))
    plt.yscale('log')
    # Plotting training and validation loss before optimizer change
    plt.plot(epochs[:Ep_optimizer], training_loss[:Ep_optimizer], 'tab:blue', linewidth=2, label='Training Loss (Adam)')
    plt.plot(epochs[:Ep_optimizer], validation_loss[:Ep_optimizer], 'tab:cyan', linewidth=2, label='Validation Loss (Adam)')
    
    # Plotting training and validation loss after optimizer change
    plt.plot(epochs[Ep_optimizer-1:], training_loss[Ep_optimizer-1:], 'tab:red', linewidth=2, label='Training Loss (LBFGS)')
    plt.plot(epochs[Ep_optimizer-1:], validation_loss[Ep_optimizer-1:], 'tab:orange', linewidth=2, label='Validation Loss (LBFGS)')
    
    # Highlighting the epoch when optimizer changed
    plt.axvline(x=Ep_optimizer, color='gray', linestyle='--', linewidth=2)
    plt.text(Ep_optimizer+5, max(max(training_loss)*1e-5, max(validation_loss))*0.2, 'Optimizer Change', rotation=90, verticalalignment='center', fontsize=14, color='gray')
    
    # Highlighting minimum validation loss
    # plt.scatter(min_val_loss_epoch, min_val_loss, color='red', zorder=5)
    # plt.annotate(f'Min Validation Loss\nEpoch {min_val_loss_epoch}',
    #              xy=(min_val_loss_epoch, min_val_loss),
    #              xytext=(min_val_loss_epoch - 100, min_val_loss + 1e-6),
    #              arrowprops=dict(facecolor='black', arrowstyle='->'),
    #              fontsize=14)
    
    # Titles and labels
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    
    # Grid and legend
    # plt.grid(alpha=0.3)
    plt.legend(fontsize = 12, loc = 'best')
    
    # Styling
    plt.xlim(0, max(epochs) + 1)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()

plot = 1

if plot == 1: 

    # Load the saved attributes:
    with open('loss_history_pos_prada.pkl', 'rb') as f:
        saved_attributes = pickle.load(f)
        
    num_epochs = 1000
    
    training_loss = np.array(saved_attributes['loss_tot'])[:num_epochs]
    validation_loss = np.array(saved_attributes['loss_val'])[:num_epochs]
    
    with open('loss_history_pos_prada_TL.pkl', 'rb') as f:
        saved_attributes_tl = pickle.load(f)
            
    training_loss_tl = np.array(saved_attributes_tl['loss_tot'])
    validation_loss_tl = np.array(saved_attributes_tl['loss_val'])

    num_epochs = len(training_loss)
    epochs = np.arange(1, num_epochs+1)
    
    num_epochs_tl = len(training_loss_tl)
    epochs_tl = np.arange(1, num_epochs_tl+1)

    # Find minimum validation loss
    min_val_loss_epoch = np.argmin(validation_loss) + 1
    min_val_loss = validation_loss[min_val_loss_epoch - 1]

    # Define the epoch where optimizer changed
    Ep_optimizer = 801

    # Find minimum validation loss
    min_val_loss_epoch = np.argmin(validation_loss) + 1
    min_val_loss = validation_loss[min_val_loss_epoch - 1]
    
    fig, axs = plt.subplots(1, 2, figsize=(18*cm, 7*cm), sharey=True)  # 1x2 layout
    
    # --- Subplot 1: Training from Scratch ---
    axs[0].set_yscale('log')
    axs[0].plot(epochs, training_loss,  linewidth=2, label='Training Loss')
    axs[0].plot(epochs, validation_loss, linewidth=2, label='Validation Loss')
    axs[0].axvline(x=Ep_optimizer, color='gray', linestyle='--', label='Optimizer Change')
    axs[0].set_title('(a)', fontsize=14)
    axs[0].set_xlabel('Epochs', fontsize=12)
    axs[0].set_ylabel('Amplitude', fontsize=12)
    axs[0].legend(fontsize=10, loc='upper right')
    axs[0].set_xlim(1, max(epochs)+1)
    # axs[0].set_ylim(bottom=0)
    
    # --- Subplot 2: Transfer Learning ---
    axs[1].set_yscale('log')
    axs[1].plot(epochs_tl, training_loss_tl,  linewidth=2, label='Training Loss (TL)')
    axs[1].plot(epochs_tl, validation_loss_tl, linewidth=2, label='Validation Loss (TL)')
    axs[1].set_title('(b)', fontsize=14)
    axs[1].set_xlabel('Epochs', fontsize=12)
    axs[1].legend(fontsize=10, loc='upper right')
    axs[1].set_xlim(1, max(epochs_tl))
    
    # ✅ Add y-axis ticks and label on the right
    axs[1].yaxis.set_ticks_position('left')
    axs[1].yaxis.set_label_position('left')
    axs[1].set_ylabel('Amplitude', fontsize=12)


    # axs[1].set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()

plot = 1

if plot == 1:


    # Load the saved attributes
    with open('loss_history_neg_mohtat.pkl', 'rb') as f:
        saved_attributes = pickle.load(f)
    
    # Extract losses
    num_epochs = 880
    training_loss = np.array(saved_attributes['loss_tot'])[:num_epochs]
    validation_loss = np.array(saved_attributes['loss_val'])[:num_epochs]
    loss_pde = np.array(saved_attributes.get('loss_pde', []))[:num_epochs]
    loss_ic = np.array(saved_attributes.get('loss_ic', []))[:num_epochs]
    loss_bc = np.array(saved_attributes.get('loss_bc', []))[:num_epochs]
    
    # Define epochs and optimizer change epoch
    epochs = np.arange(1, num_epochs + 1)
    Ep_optimizer = 701
    
    # Create 2x1 subplots
    fig, axs = plt.subplots(1, 2, figsize=((18*cm, 8*cm)))
    
    # Subplot 1: Total training and validation loss
    axs[0].set_yscale('log')
    axs[0].plot(epochs, training_loss, 'tab:blue', label='Training Loss')
    # axs[0].plot(epochs[:Ep_optimizer], validation_loss[:Ep_optimizer], 'tab:blue.-', label='Validation Loss (Adam)')
    # axs[0].plot(epochs, training_loss, 'tab:orange', label='Training Loss')
    axs[0].plot(epochs, validation_loss, 'tab:orange', label='Validation Loss')
    axs[0].axvline(x=Ep_optimizer, color='gray', linestyle='--', label='Optimizer Change')
    axs[0].set_title('(a)', fontsize=14)
    axs[0].set_xlabel('Epochs', fontsize=12)
    axs[0].set_ylabel('Amplitude', fontsize=12)
    axs[0].legend(fontsize=10, loc= "lower left")
    # axs[0].set_ylim(1e-9, max(training_loss))
    axs[0].set_xlim(1, max(epochs)+1)
    # axs[0].grid(True)
    
    # Subplot 2: PDE, IC, and BC losses
    axs[1].set_yscale('log')
    axs[1].plot(epochs, loss_pde, label='PDE Loss')
    axs[1].plot(epochs, loss_ic, label='IC Loss')
    axs[1].plot(epochs, loss_bc, label='BC Loss')
    axs[1].axvline(x=Ep_optimizer, color='gray', linestyle='--', label='Optimizer Change')
    axs[1].set_title('(b)', fontsize=14)
    axs[1].set_xlabel('Epochs', fontsize=12)
    axs[1].set_ylabel('Amplitude', fontsize = 12)
    axs[1].legend(fontsize=10, loc='best')
    axs[1].set_xlim(1, max(epochs)+1)
    # axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    

#####################################
plot = 12
if plot == 1:
    # Create subplot figure (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(19*cm, 8*cm))
    
    class GRF():
        def __init__(self, T, length_scale, N):
            from sklearn.gaussian_process.kernels import RBF
            self.N = N
            self.x = np.linspace(0, T, num=N)[:, None]
            K = RBF(length_scale=length_scale)
            self.K = K(self.x)
            self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))
    
        def random(self, size):
            u = np.random.randn(self.N, size)
            return np.dot(self.L, u).T
        
        def eval_batch(self, features, xs):
            # tanh = 0.5+np.tanh(xs*10-5).T*0.5
            pippo = np.vstack([np.interp(xs, np.ravel(self.x), y).T for y in features])
            pippo_min = pippo[:,0]
            pluto = pippo - pippo_min.reshape(len(features[:,0]),1)
            return pluto/10
    
    space = GRF(T=1, length_scale=0.3, N=20)
    features = space.random(10)
    x = np.linspace(0,1,100)
    y = space.eval_batch(features,x)
    sensors = np.linspace(0,1,50)
    y_sensors = space.eval_batch(features,sensors)
    fs2 = 14
    
    for i in range(y.shape[0]):
        if i == 0:
            axes[0].plot(x, y[i], 'k', linewidth=1.2, alpha=0.7, label = 'Continous profile')
            axes[0].plot(sensors, y_sensors[i], '*', markersize = 5,color =  'k',alpha=0.7, label = 'Sensor points')
        else: 
            axes[0].plot(x, y[i], linewidth=1.2, alpha=0.7)
            axes[0].plot(sensors, y_sensors[i],'*', markersize = 5, alpha=0.7)
    
    axes[0].set_xlabel('Adimensional radial coordinate ($r$)', fontsize=fs2)
    axes[0].set_ylabel('Adimensional concentration ($c$)', fontsize=fs2)
    # axes[0].set_title('GRF sampled profiles')
    axes[0].grid(False)
    legend = axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=1, frameon=True, fontsize = 12)
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_edgecolor('black')
    
    # axes[0].spines['right'].set_visible(False)
    # axes[0].spines['top'].set_visible(False)
    
    from matplotlib.ticker import FormatStrFormatter
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    #####################################
    from scipy.stats import norm
    
    # Gaussian distribution parameters
    mu, sigma = 0, 1
    
    # Generate Gaussian distribution data for curve plotting
    x = np.linspace(-4, 4, 100)
    y = norm.pdf(x, mu, sigma)
    
    # Sample points from Gaussian distribution
    num_samples = 100
    sampled_points = np.random.normal(mu, sigma, num_samples)
    
    # Plotting in the middle subplot (axes[1])
    axes[1].plot(x, y, color='black', linewidth=1.5, label='Normal Distribution')
    axes[1].fill_between(x, y, color='lightgray', alpha=0.7)
    axes[1].plot(sampled_points, np.zeros_like(sampled_points), 'ro', markersize=6, label='Sampled Points')

    axes[1].set_xlabel('C-Rate', fontsize = fs2)
    axes[1].set_ylabel('Probability Density', fontsize = fs2)
    legend = axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=1, frameon=True, fontsize=12)
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_edgecolor('black')
    axes[1].grid(False)
    
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    # Remove unnecessary top/right spines from the middle plot
    # axes[1].spines['right'].set_visible(False)
    # axes[1].spines['top'].set_visible(False)
      
    #####################################
    import torch
    soboleng = torch.quasirandom.SobolEngine(dimension=2)
    points_pde =  soboleng.draw(250).detach().numpy()
    points_ic =  soboleng.draw(50).detach().numpy()
    points_ic[:,0] = 0 
    points_bc =  soboleng.draw(50).detach().numpy()
    points_bc[:,1] = 1
    
    axes[2].fill_between([0,1],[1,1], color='lightgray', alpha=0.7, label = 'Domain')
    axes[2].plot(points_pde[:,0], points_pde[:,1], 'ko', markersize=3, label='PDE')
    axes[2].plot(points_ic[:,0], points_ic[:,1], 'ro', markersize=3, label='IC')
    axes[2].plot(points_bc[:,0], points_bc[:,1], 'bo', markersize=3, label='BC')
    points_bc[:,1] = 0
    axes[2].plot(points_bc[:,0], points_bc[:,1], 'bo', markersize=3)
    
    
    axes[2].set_xlabel('Adimensional time ($t$)', fontsize = fs2)
    axes[2].set_ylabel('Adimensional radial coordinate ($r$)', fontsize = fs2)
    # axes[2].legend(frameon=True, fontsize=10, loc=(0.5,1))
    axes[2].grid(False)
    
    legend = axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=True, fontsize = 12)
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_edgecolor('black')
    
    # Remove unnecessary top/right spines from the middle plot
    # axes[2].spines['right'].set_visible(False)
    # axes[2].spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    



#######################################
plot = 12
if plot == 1: 
    # Number of discretized states
    n_states = 20
    r = np.linspace(0, 1, 500)
    c = 0.1 + r**2/6
    
    # Discrete radial points (interval edges)
    states = np.linspace(0, 1, n_states)
    c_discrete = np.interp(states, r, c)
    # Plotting
    fig, ax = plt.subplots( figsize=(4, 3))
    ax.plot(r, c, label='Continuous state c(r)', linewidth=2)
    ax.plot(states, c_discrete,'o',alpha = 0.6,
            label='Discretized states $\mathbf{c}$', markersize=6)
    
    # Labels and title
    ax.set_xlabel('$r$', fontsize=12)
    # ax.set_ylabel('Adimensional concentration ($c$)', fontsize=12)
    # ax.set_title('Discretization of Radial Concentration Profile', fontsize=14)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_ylim([0.1,0.45])
    
    # Grid and legend
    # ax.grid(True)
    ax.legend(fontsize=12)
    ax.set_yticks([])
    
    # Display plot
    plt.tight_layout()
    plt.show()
    
    
    
#######################################
plot = 12
if plot == 1: 
    
    # Example error data for each method (replace these with your real data)
    errors_method1 = np.random.rand(10) * 2  # ECM
    errors_method2 = np.random.rand(10) * 0.5  # MIONet
    
    # Compute averages
    avg_method1 = np.mean(errors_method1)
    avg_method2 = np.mean(errors_method2)
    
    # Prepare x-axis labels
    methods = ['ECM', 'MIONet']
    x_positions = [1, 2]
    
    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Scatter individual errors
    plt.scatter(np.full_like(errors_method1, x_positions[0]), errors_method1, alpha=0.7, color='blue', label='Individual errors' if 'Individual errors' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.scatter(np.full_like(errors_method2, x_positions[1]), errors_method2, alpha=0.7, color='green')
    
    # Plot average errors
    plt.scatter(x_positions, [avg_method1, avg_method2], color='red', marker='*', s=200, label='Average error')
    
    # Customize plot
    plt.xticks(x_positions, methods)
    plt.ylabel('Error [%]', fontsize=12)
    plt.title('Error comparison: ECM vs MIONet', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.show()
        
    
    







