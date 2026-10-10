"""
SOC estimation benchmark: MIONet-based observer vs. ECM-based observer (Mohtat2020)
----------------------------------------------------------------------------------

1) Plant: the PyBaMM DFN (Mohtat2020) is solved for the selected current profile and treated
   as ground truth; Gaussian noise is added to the measured voltage and current.
2) MIONet observer (observers.MIONetObserver): MIONet process model + electrochemical
   measurement model (voltage + lithium balance) in a UKF. Settings: observer_config.json,
   produced by the pipeline in tuning/ (ohmic fit + NES tuning of the UKF weights).
3) ECM observer (observers.ECMObserver): 3-RC ECM (HCCP-identified parameters) in a UKF.
4) Metrics: SOC RMSE (t >= 600 s) and CWC; plots; optional save to Saved_simulations format.

Both observers start from SOC = 50 % (the true initial SOC is unknown to them).

Configuration: edit the block below. Examples of the paper cases (25 °C):
    charging_01_25 : PROFILE = 'CC',   CR = -0.1, T_END = 6 * 3600, SOC_0 = 0.25
    charging_1_25  : PROFILE = 'CC',   CR = -1,   T_END = 3600,     SOC_0 = 0.25 (stops at 4.2 V)
    UDDS_1_25      : PROFILE = 'UDDS', CR = 1,    T_END = 7200,     SOC_0 = 0.80
    UDDS_2_25      : PROFILE = 'UDDS', CR = 3,    T_END = 3600,     SOC_0 = 0.80
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import observers as ob
import utils

###################################### Configuration ######################################
T_AMB = 25             # ambient temperature [°C]
PROFILE = 'CC'         # 'CC' (CR < 0: charge) or 'UDDS'
CR = -0.5              # C-rate
T_END = 4500           # simulated time [s]
SOC_0 = 0.25           # initial SOC of the plant
THERMAL = False        # lumped thermal model in the plant (the archived paper cases are isothermal)
NOISE_STD = 1e-3       # std of voltage [V] and current [A] measurement noise
SEED = 42
PLOT = True
SAVE = False           # saves a dictionary in the format of Saved_simulations/
############################################################################################

np.random.seed(SEED)
param = utils.make_param(T_AMB)
C = param['Nominal cell capacity [A.h]']
plant = utils.plant(param, ob.N)

## Plant
t_eval = np.linspace(0, T_END, int(T_END / ob.DT) + 1)
current, current_string = plant.CC(t_eval, CR) if PROFILE == 'CC' else plant.UDDS(t_eval, CR)
print('MIONet vs ECM: %s at %g C, %.2f h, T = %d °C' % (current_string, CR, T_END / 3600, T_AMB))
Time, y_ref, Voltage_ref, Temp = plant.ref_solution(SOC_0, current, t_eval, THERMAL)
n = len(Time); current = current[:n]
soc_ref = plant.SOC_calculator(y_ref)

V_meas = Voltage_ref + np.random.normal(0, NOISE_STD, n)
I_meas = current + np.random.normal(0, NOISE_STD, n)

## Observers
mionet = ob.MIONetObserver(T_AMB).run(V_meas, I_meas)
ecm = ob.ECMObserver(C=C).run(V_meas, I_meas)

for name, res in (('MIONet', mionet), ('ECM', ecm)):
    m = ob.metrics(res, soc_ref)
    print('%-6s SOC RMSE %.3e %%  |  CWC %.3f %%  (PICP %.2f, MPIW %.3f %%)' % (name, m['rmse'], m['cwc'], m['picp'], m['mpiw']))

## Plots
if PLOT:
    font_lab = {'family': 'serif', 'color': 'black', 'size': 14}
    font_tit = {'family': 'serif', 'color': 'black', 'size': 18}
    fig = plt.figure(figsize=(16, 6), layout='constrained')
    ax = fig.subplot_mosaic([['soc', 'soc_error']])

    ax['soc'].plot(Time, soc_ref, 'k', linewidth=1, label='Reference')
    for res, c, lab in ((mionet, 'b', 'MIONet'), (ecm, 'r', 'ECM')):
        ax['soc'].plot(Time, res['soc'], c, linewidth=2, label='Prediction ' + lab)
        ax['soc'].fill_between(Time, res['soc_lower'], res['soc_upper'], color=c, alpha=0.2)
        ax['soc_error'].plot(Time, res['soc'] - soc_ref, c, linewidth=1.5, label=lab)
        ax['soc_error'].fill_between(Time, res['soc_lower'] - soc_ref, res['soc_upper'] - soc_ref, color=c, alpha=0.2)
    ax['soc'].set_title('SOC', fontdict=font_tit)
    ax['soc'].set_ylabel('SOC [%]', fontdict=font_lab)
    ax['soc_error'].plot(Time, np.zeros(n), '--', color='grey', linewidth=1.5)
    ax['soc_error'].set_title('Error', fontdict=font_tit)
    ax['soc_error'].set_ylabel(r'$\Delta$ SOC [%]', fontdict=font_lab)
    ax['soc_error'].set_ylim([-5, 5])
    for a in ax.values():
        a.set_xlabel('Time [min]', fontdict=font_lab); a.grid(True, linestyle='--', alpha=0.7); a.legend(fontsize=14)

    ax_curr = ax['soc'].inset_axes([0, -0.4, 1, 0.25])
    ax_curr.plot(Time, current / C, 'navy', linewidth=1)
    ax_curr.set_ylabel('i [x CR]', fontdict=font_lab)
    ax_curr.set_xlabel(current_string, fontdict=font_lab)
    ax_curr.set_xticklabels([])
    ax_curr.grid(True, linestyle='--', alpha=0.7)
    plt.show()

## Save (same structure as Saved_simulations/; here soc_lower / soc_upper are correctly ordered)
if SAVE:
    D = {'plant': dict(soc=soc_ref, Voltage=V_meas, time=Time, current=current, current_string=current_string,
                       dT=ob.DT, measurment_noise=NOISE_STD),
         'mionet': dict(soc=mionet['soc'], concentration=mionet['concentration'], soc_bottom=mionet['soc_lower'],
                        soc_upper=mionet['soc_upper'], Voltage=mionet['voltage']),
         'ecm': dict(soc=ecm['soc'], soc_bottom=ecm['soc_lower'], soc_upper=ecm['soc_upper'], Voltage=ecm['voltage'])}
    file_name = os.path.join(ob.HERE, '%s_%g_%d.pkl' % (current_string.replace(' ', '_'), CR, T_AMB))
    with open(file_name, 'wb') as f:
        pickle.dump(D, f)
    print('saved', file_name)
