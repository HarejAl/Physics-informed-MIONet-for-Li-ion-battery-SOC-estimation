
import numpy as np 
import matplotlib.pyplot as plt
import pickle

def compute_cwc(y_true, y_lower, y_upper, mu, gamma, eta):
    N = len(y_true)

    # Compute PICP
    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    picp = np.sum(in_interval) / N

    # Compute MPIW
    mpiw = np.mean(y_upper - y_lower)

    # Compute CWC
    if picp < mu:
        cwc = mpiw * (1 + gamma * np.exp(-eta * (picp - mu)))
    else:
        cwc = mpiw

    return cwc

# Font settings to match LaTeX document
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 12
})
cm = 1/1.64  # centimeters in inches

# Load the dictionary from the file
with open('charging_05_25', 'rb') as f:
    D = pickle.load(f)

Time = D['plant']['time']
Voltage_plant = D['plant']['Voltage'] 
Voltage_mn = D['mionet']['Voltage'] 
Voltage_ecm = D['ecm']['Voltage'] 
current = D['plant']['current']
current_string = D['plant']['current_string']

C = 5 #Nominal capacity for 'Mohtat' parameters set

##### SOC error
err = np.mean(np.linalg.norm((D['plant']['soc'] - D['mionet']['soc']) / 100))
# print('MIONET SOC estimation error {:.2e} (Fr. norm)'.format(err))

idx = int(300) ## Avoids the initial error caused by the unknown ICs
err = (D['plant']['soc'] - D['mionet']['soc'])**2
err = np.sqrt(np.mean(err[idx:]))

print('MIONET SOC estimation error {:.2e} (RMSE)'.format(err))

err = np.mean(np.linalg.norm((D['plant']['soc'] - D['ecm']['soc']) / 100))
# print('ECM SOC estimation error {:.2e} (Fr. norm)'.format(err))

err = (D['ecm']['soc'] - D['plant']['soc'])**2
err = np.sqrt(np.mean(err[idx:]))

print('ECM SOC estimation error {:.2e} (RMSE)'.format(err))

score_mionet = compute_cwc(D['plant']['soc'],  D['mionet']['soc_upper'], D['mionet']['soc_bottom'], mu = 0.95, gamma = 1, eta = 1)

score_ecm = compute_cwc(D['plant']['soc'], D['ecm']['soc_bottom'], D['ecm']['soc_upper'], mu = 0.95, gamma = 1, eta = 1)

print('MIONet precision score {:.2f}'.format(score_mionet))
print('ECM precision score {:.2f}'.format(score_ecm))

plot_voltage = False
plot_soc = True


############# PLOT 1 VOLTAGE comparison 

if plot_voltage:
    
    fs1 = 12
    fs2 = 14
    # font_lab = {'family':'serif','color':'black','size':12}
    # font_tit = {'family':'serif','color':'black','size':14}
    
    error1 = abs((Voltage_plant - Voltage_mn)/Voltage_plant*100  ) 
    error2 = abs((Voltage_plant - Voltage_ecm)/Voltage_plant*100 )
    
    ax1.plot(Time, Voltage_plant,'k--',linewidth=1, label='Reference')
    ax1.plot(Time, Voltage_mn,'b',linewidth=1, label='Prediction MIONet')
    ax1.plot(Time, Voltage_ecm,'r',linewidth=1, label='Prediction ECM')

    ax1.set_xlabel("Time (min)",fontsize = fs1)
    ax1.set_ylabel("voltage (V)",fontsize = fs1)
    ax1.set_title('Voltage comparison',fontsize = fs2)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize = fs1)


if plot_soc:
    fs1 = 14
    fs2 = 16
    
    fig = plt.figure(figsize=(19*cm, 10*cm), layout='constrained')
    ax = fig.subplot_mosaic([['soc','soc_error']])
    
    # ax2 = ax['soc'].twinx() 
   
    ax['soc'].plot(Time,D['mionet']['soc'],'b',linewidth = 2,label = 'MIONet')
    ax['soc'].plot(Time,D['ecm']['soc'],'r',linewidth = 2,label = 'ECM')
    
    ax['soc'].fill_between(Time, D['mionet']['soc_bottom'], D['mionet']['soc_upper'], color = 'b',alpha=0.2, label = r'MIONet $\pm 2\sigma $ interval')
    ax['soc'].fill_between(Time, D['ecm']['soc_bottom'], D['ecm']['soc_upper'], color = 'r',alpha=0.2, label = r'ECM $\pm 2\sigma$ interval')
    
    ax["soc"].plot(Time, D['plant']['soc'],'k',linewidth = 1, label = 'Reference')
    
    ax['soc'].legend()
    ax['soc'].grid(True, linestyle='--', alpha=0.7)
    
    ax['soc'].set_title('SOC',fontsize = fs2)
    ax['soc'].set_xlabel('Time (min)',fontsize = fs1)
    ax['soc'].set_ylabel('Amplitude (%)',fontsize = fs1)
    ax['soc'].legend(fontsize = 12)
    # ax['soc'].set_xlim([0,60])
    
    ax_curr= ax['soc'].inset_axes([0, -0.45, 1, 0.2]) 
    
    ax_curr.plot(Time,current[:len(Time)]/C,'navy',linewidth = 1)
    ax_curr.set_ylabel('i (CR)',fontsize = fs1)
    ax_curr.set_xlabel(current_string+' profile',fontsize = fs1)
    ax_curr.set_xticklabels([])
    ax_curr.grid(True, linestyle='--', alpha=0.7)

    
    # Specify the zoomed-in region
    # x1, x2, y1, y2 = 200, 300, 75, 79  # Zoom region coordinates
    
    # Create an inset of the zoomed region
    # axins = ax['soc'].inset_axes([0.01, 0.01, 0.45, 0.45])  # [left, bottom, width, height]
    # axins.plot(Time_pb, soc_ref,'k',linewidth = 1)
    # axins.plot(Time_pb,soc_mn,'r',linewidth = 1)
    # axins.plot(Time_pb,SOC_ukf,'b',linewidth = 1)
    # axins.fill_between(Time_pb, SOC_bottom, SOC_upper,color = 'c',alpha=0.5)
    
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])
    
    # # Add lines connecting the inset to the main plot
    # ax['soc'].indicate_inset_zoom(axins, edgecolor='red', linewidth=2, linestyle='--')

    # vmin = min(conf_soc_nn_up.min(),conf_soc_nn_b.min(),conf_soc_ecm_up.min(),conf_soc_ecm_b.min(),error_nn.min(),error_ecm.min())
    # vmax = max(conf_soc_nn_up.max(),conf_soc_nn_b.max(),conf_soc_ecm_up.max(),conf_soc_ecm_b.max(),error_nn.max(),error_ecm.max())
    # y_lim = max(abs(vmin),abs(vmax))
    
    error_nn = D['mionet']['soc'] - D['plant']['soc']
    error_ecm = D['ecm']['soc'] - D['plant']['soc']
    
    # if plot_avg_error==1:
    #     ax['soc_error'].plot(Time, error_nn,'b',linewidth = 1.5,label='Mionet ($\Delta$ avg: {:.2f})'.format(np.mean(abs(error_nn))))
    #     ax['soc_error'].plot(Time, error_ecm,'r',linewidth = 1.5,label='ECM ($\Delta$ avg: {:.2f})'.format(np.mean(abs(error_ecm))))
    # else:
        
    ax['soc_error'].plot(Time, error_nn,'b',linewidth = 1.5)
    ax['soc_error'].plot(Time, error_ecm,'r',linewidth = 1.5)
        
    ax['soc_error'].plot(Time, np.zeros(len(Time)), '--', color = 'k', alpha=0.8,linewidth = 1.5)
    ax['soc_error'].set_title('Error',fontsize = fs2)
    ax['soc_error'].grid(True, linestyle='--', alpha=0.7)
    ax['soc_error'].set_xlabel('Time (min)',fontsize = fs1)
    ax['soc_error'].set_ylabel('Amplitude (%)',fontsize = fs1)
    # ax['soc_error'].legend(fontsize = 12)
    # h = 1.5; ax['soc_error'].set_ylim(-h*y_lim,h*y_lim)
    ax['soc_error'].fill_between(Time, (D['mionet']['soc_bottom'] - D['plant']['soc']), (D['mionet']['soc_upper'] - D['plant']['soc']), color = 'b', alpha=0.2)
    # ax['soc_error'].plot(Time, D['mionet']['soc_bottom'] - D['plant']['soc'], '--', color = 'k', alpha=0.7,linewidth = 1.5)
    # ax['soc_error'].plot(Time, D['mionet']['soc_upper'] - D['plant']['soc'], '--', color = 'k', alpha=0.7,linewidth = 1.5)
    
    ax['soc_error'].fill_between(Time, D['ecm']['soc_bottom'] - D['plant']['soc'], D['ecm']['soc_upper'] - D['plant']['soc'], color = 'r',alpha=0.2)
    # ax['soc_error'].plot(Time, D['ecm']['soc_bottom'] - D['plant']['soc'], '--', color = 'red', alpha=0.7,linewidth = 1.5)
    # ax['soc_error'].plot(Time, D['ecm']['soc_upper'] - D['plant']['soc'], '--', color = 'red', alpha=0.7,linewidth = 1.5)
    ax['soc_error'].set_ylim([-2.5,2.5])

    plt.tight_layout()
    fig.align_ylabels()  # same as fig.align_xlabels(); fig.align_ylabels()
    # fig.align_titles()
    plt.show()
    
    