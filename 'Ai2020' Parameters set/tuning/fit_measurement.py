"""
Step 1 (Ai2020): measurement model and calibration constants.

- k_ohm      : ohmic term of the observer's voltage map (U -= k_ohm * I), least squares on the true
               plant states of the TRAINING simulations (original script: 1 / (1800 * A_EFF)).
- li_target  : exact total lithium in the particles, from the parameter set
               (y_100 * L_p eps_p c_p,max + x_100 * L_n eps_n c_n,max), conserved by the plant.
- r_extra    : extra series resistance of the real cell, absent from the Ai2020 model (tabs,
               contacts): from the measured 2C discharge vs the DFN at the same SOC, 20-90 % SOC.
- sigma_v_exp: residual spread of that comparison, used as voltage noise std on measured data.
r_extra and sigma_v_exp use ONLY the 2C discharge, which is not part of Table 3 (0.5C, 1C).

Output: results/measurement_fit.json
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import observers_ai as ob  # noqa: E402
import utils_ai  # noqa: E402

o = ob.MIONetObserver(25, config=dict(k_ohm=0.0))
k_orig = 1 / (1800 * ob.A_EFF)

# k_ohm on the training simulations
num = den = 0.0
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "data", "train_*.npz"))):
    d = np.load(f)
    H0 = o.voltage(d["y_ref"].T, d["I_clean"])
    num += np.sum((H0 - d["V_clean"]) * d["I_clean"]); den += np.sum(d["I_clean"] ** 2)
    rows.append((os.path.basename(f)[:-4], H0, d["V_clean"], d["I_clean"]))
k_ohm = num / den
print("k_ohm = %.5f Ohm (original script %.5f Ohm)" % (k_ohm, k_orig))
for name, H0, V, I in rows:
    print("  %-26s U RMSE original %5.2f mV | fit %5.2f mV" % (
        name, 1e3 * np.sqrt(np.mean((H0 - k_orig * I - V) ** 2)), 1e3 * np.sqrt(np.mean((H0 - k_ohm * I - V) ** 2))))

# exact lithium total
li_target = float(o.y100 * o.li_p + o.x100 * o.li_n)
print("li_target (exact, from parameters) = %.5f" % li_target)

# extra series resistance and voltage noise of the real cell, from the 2C discharge only
_, V_exp, soc_exp = utils_ai.load_experiment(2, ob.DT)
sim = np.load(os.path.join(HERE, "data", "simtest_2C.npz"))
gap = V_exp - np.interp(-soc_exp, -sim["soc_ref"], sim["V_clean"])
m = (soc_exp > 20) & (soc_exp < 90)
I2 = 2 * utils_ai.C_RATE_CAPACITY
r_extra = float(-gap[m].mean() / I2)
sigma_v = float(np.std(gap[m] + r_extra * I2))
print("2C calibration: r_extra = %.1f mOhm, sigma_v_exp = %.1f mV" % (1e3 * r_extra, 1e3 * sigma_v))

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(dict(k_ohm=k_ohm, li_target=li_target, r_extra=r_extra, sigma_v_exp=sigma_v),
          open(os.path.join(HERE, "results", "measurement_fit.json"), "w"), indent=1)
