"""
SOC estimation: MIONet-based vs ECM-based observer, Ai2020 parameter set (Enertech LCO-G cell)
-----------------------------------------------------------------------------------------------

Constant-current discharge from fully charged, at 0.5C, 1C or 2C, either on:
- measured data (USE_EXPERIMENT = True): voltage of the Enertech cell (PyBaMM Ai2020 data set,
  files CC_*.npy), SOC reference = linear ramp from 100 % to the cut-off. Both observers use the
  extra series resistance and voltage noise calibrated on the 2C discharge (tuning/); or
- a simulated plant (USE_EXPERIMENT = False): PyBaMM DFN with noise on voltage and current.

Observers (observers_ai.py): MIONet + UKF (settings: observer_config.json, from tuning/) and
3-RC ECM + UKF. Both start from SOC = 50 %. Metrics: SOC RMSE (t >= 600 s) and CWC.
"""

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import observers_ai as ob
import utils_ai

###################################### Configuration ######################################
CR = 1                 # 0.5, 1 or 2
USE_EXPERIMENT = True
NOISE_STD = 1e-3       # simulated plant: std of voltage [V] and current [A] noise
SEED = 42
PLOT = True
SAVE = False
############################################################################################

np.random.seed(SEED)
C = utils_ai.C_RATE_CAPACITY

if USE_EXPERIMENT:
    t, V_meas, soc_ref = utils_ai.load_experiment(CR, ob.DT)
    I_meas = np.full(len(V_meas), C * CR)
    fit = json.load(open(os.path.join(ob.HERE, 'tuning', 'results', 'measurement_fit.json')))
    kw = dict(sigma_v=fit['sigma_v_exp'], r_extra=fit['r_extra'])  # calibrated on the 2C discharge
    source = 'measured'
else:
    plant = utils_ai.plant(utils_ai.make_param(25), ob.N)
    t_eval = np.linspace(0, 3800 / CR, int(3800 / CR / ob.DT) + 1)
    current, _ = plant.CC(t_eval, CR)
    Time, y_ref, V_ref, _ = plant.ref_solution(1.0, current, t_eval)
    t = Time * 60; n = len(t)
    soc_ref = plant.SOC_calculator(y_ref)
    V_meas = V_ref + np.random.normal(0, NOISE_STD, n)
    I_meas = current[:n] + np.random.normal(0, NOISE_STD, n)
    kw = {}
    source = 'simulated'
Time = t / 60
print('MIONet vs ECM: %g C discharge, %s data, %.2f h' % (CR, source, t[-1] / 3600))

mionet = ob.MIONetObserver(25).run(V_meas, I_meas, **kw)
ecm = ob.ECMObserver().run(V_meas, I_meas, **kw)
for name, res in (('MIONet', mionet), ('ECM', ecm)):
    m = ob.metrics(res, soc_ref)
    print('%-6s SOC RMSE %.3f %%  |  CWC %.3f %%  (PICP %.2f, MPIW %.3f %%)' % (name, m['rmse'], m['cwc'], m['picp'], m['mpiw']))

if PLOT:
    fig = plt.figure(figsize=(16, 6), layout='constrained')
    ax = fig.subplot_mosaic([['voltage', 'soc', 'soc_error']])
    ax['voltage'].plot(Time, V_meas, 'k.', markersize=2, label='Plant (%s)' % source)
    ax['soc'].plot(Time, soc_ref, 'k', linewidth=1, label='Reference')
    for res, c, lab in ((mionet, 'b', 'MIONet'), (ecm, 'r', 'ECM')):
        ax['voltage'].plot(Time, res['voltage'], c, linewidth=1, label=lab)
        ax['soc'].plot(Time, res['soc'], c, linewidth=2, label=lab)
        ax['soc'].fill_between(Time, res['soc_lower'], res['soc_upper'], color=c, alpha=0.2)
        ax['soc_error'].plot(Time, res['soc'] - soc_ref, c, linewidth=1.5, label=lab)
        ax['soc_error'].fill_between(Time, res['soc_lower'] - soc_ref, res['soc_upper'] - soc_ref, color=c, alpha=0.2)
    ax['voltage'].set_ylabel('Voltage [V]'); ax['soc'].set_ylabel('SOC [%]'); ax['soc_error'].set_ylabel(r'$\Delta$ SOC [%]')
    ax['soc_error'].set_ylim([-5, 5])
    for a in ax.values():
        a.set_xlabel('Time [min]'); a.grid(True, linestyle='--', alpha=0.7); a.legend()
    plt.show()

if SAVE:
    D = {'plant': dict(soc=soc_ref, Voltage=V_meas, time=Time, current=I_meas, dT=ob.DT, source=source),
         'mionet': dict(soc=mionet['soc'], soc_bottom=mionet['soc_lower'], soc_upper=mionet['soc_upper'],
                        concentration=mionet['concentration'], Voltage=mionet['voltage']),
         'ecm': dict(soc=ecm['soc'], soc_bottom=ecm['soc_lower'], soc_upper=ecm['soc_upper'], Voltage=ecm['voltage'])}
    f = os.path.join(ob.HERE, 'CC_%s_%s.pkl' % (str(CR).replace('.', ''), source))
    pickle.dump(D, open(f, 'wb'))
    print('saved', f)
