import numpy as np
import matplotlib.pyplot as plt

def plot_results(x, ct_avg, err, opt_fcn, title):
    """
    Creates a figure with three overlaid axes to plot:
      - Computational time vs. state vector size (red, left y-axis)
      - Error vs. state vector size (blue, middle y-axis)
      - Optimality function vs. state vector size (black, right y-axis)
      
    Parameters:
        x       : 1D array of state vector sizes (e.g. x = 2*N)
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
    
    plt.title(title)
    
    plt.show()

def optimal_fnc(ct, err):
    # ct_n = (ct - np.min(ct)) / (np.max(ct) - np.min(ct))
    # err_n = (err - np.min(err)) / (np.max(err) - np.min(err))
    
    ct_n = (ct - ct_min) / (ct_max - ct_min)
    err_n = (err - err_min) / (err_max - err_min)
    
    # Compute the optimality function (as sum of normalized ct_avg and err_arr).
    opt_fcn = ct_n + err_n
    # opt_fcn = ct_avg + err_arr
    
    return opt_fcn

# Font settings to match LaTeX document
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 12
})
cm = 1/1.64 


"""
x: vector with states (for both electrodes)
ct: vector with average time for a single time step in ms
err: vector with average error (Fr. norm) dimensionless (nether in %)
opt_fcn: optimal function time + error

_1 OCFE
_2 MIONET 
_3 PADE

"""

(x_1, ct_1, err_1, opt_fcn_1) = np.load("OCFE_tuple2.npy")

(x_2, ct_2, err_2, opt_fcn_2) = np.load("Mionet_tuple2.npy")

(x_3, ct_3, err_3, opt_fcn_3) = np.load("PADE_tuple2.npy")

# x_3 = x_3[1:]
# ct_3 = ct_3[1:]
# err_3 = err_3[1:]
# opt_fcn_3 = opt_fcn_3[1:]

ct_min = np.min(np.hstack((ct_1, ct_2, ct_3)))
ct_max = np.max(np.hstack((ct_1, ct_2, ct_3)))
err_min = np.min(np.hstack((err_1, err_2, err_3)))
err_max = np.max(np.hstack((err_1, err_2, err_3)))

x_3 *= 2

plot_1 = 0
plot_2 = 1
plot_3 = 0

if plot_1 == 1:
    name = 'OCFE'
    plot_results(x_1, ct_1, err_1, opt_fcn_1, name)
    name = 'MIONET'
    plot_results(x_2, ct_2, err_2, opt_fcn_2, name)
    name = 'PADE'
    plot_results(x_3, ct_3, err_3, opt_fcn_3, name)
    

if plot_2 == 1:
    
    fs1 = 16; fs2 = 14

    # Plotting errors and times
    fig, (ax_err, ax_time) = plt.subplots(1, 2, figsize=(19*cm,8*cm), constrained_layout=True)
    
    # Error plot
    ax_err.plot(x_1, err_1, marker='o', linestyle='-', linewidth=1.5, markersize=6, label='OCFE', color='tab:blue')
    ax_err.plot(x_2, err_2, marker='s', linestyle='-', linewidth=1.5, markersize=6, label='MIONET', color='tab:red')
    ax_err.plot(x_3, err_3, marker='^', linestyle='-', linewidth=1.5, markersize=6, label='PADE', color='tab:green')
    
    ax_err.set_title('Prediction error', fontsize=fs1)
    ax_err.set_xlabel('States', fontsize=fs2)
    ax_err.set_ylabel('Amplitude (-)', fontsize=fs2)
    ax_err.set_yscale('log')
    ax_err.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    ax_err.legend(fontsize=12)
    
    # Time plot
    ax_time.plot(x_1, ct_1, marker='o', linestyle='-', linewidth=1.5, markersize=6, label='OCFE', color='tab:blue')
    ax_time.plot(x_2, ct_2, marker='s', linestyle='-', linewidth=1.5, markersize=6, label='MIONET', color='tab:red')
    ax_time.plot(x_3, ct_3, marker='^', linestyle='-', linewidth=1.5, markersize=6, label='PADE', color='tab:green')
    
    ax_time.set_title('Computational time', fontsize=fs1)
    ax_time.set_xlabel('States', fontsize=fs2)
    ax_time.set_ylabel('Amplitude (ms/step)', fontsize=fs2)
    ax_time.set_yscale('log')
    ax_time.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    ax_time.legend(fontsize=12)
    
    # Overall figure title
    # fig.suptitle('Performance Comparison', fontsize=18, weight='bold')
    
    plt.show()



if plot_3 == 1:
    
    opt_fcn_1 = optimal_fnc(ct_1, err_1)
    opt_fcn_2 = optimal_fnc(ct_2, err_2)
    opt_fcn_3 = optimal_fnc(ct_3, err_3)
    
    #### Plotting only the errors
    fig, ax = plt.subplots(1,1,layout='constrained',figsize=(6,4))
    ax.plot(x_1, opt_fcn_1, 'b',linewidth=0.5)
    ax.plot(x_1, opt_fcn_1,'o',color='b', label = 'OCFE') #,markerfacecolor='none'
    ax.plot(x_2, opt_fcn_2, 'r',linewidth=0.5)
    ax.plot(x_2, opt_fcn_2,'o',color='r', label = 'MIONET')
    ax.plot(x_3, opt_fcn_3, 'g',linewidth=0.5)
    ax.plot(x_3, opt_fcn_3,'o',color='g', label = 'PADE')
    ax.set_title('Optimal function')
    ax.set_xlabel('States')
    ax.set_ylabel('Optimal value')
    ax.legend()
    ax.grid()
    ax.set_yscale('log')
    

# ax.plot([y-1,y],[j,i],'o',color='b',markerfacecolor='none')
# ax.plot([y-1,y],[j,i],'k--',linewidth = 1.5, alpha = 0.5)

