"""
Step 1: fits the measurement model on the TRAINING set (true plant states, noise-free signals).

- k_ohm: ohmic correction term of the voltage map  U = Up - Un - eta_n + eta_p - k_ohm * i
         (public release: 1/600 Ohm), least squares on the true concentrations.
- li_target: value of the particle-averaged lithium balance (conserved by the plant).
  The surface-node version used in the public release is printed for comparison: it is not
  conserved during current flow.

Output: results/measurement_fit.json
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import observers as ob  # noqa: E402

cfg = dict(k_ohm=0.0, li_mode="avg")
num = den = 0.0
rows, li_all = [], []
obs = {}
for f in sorted(glob.glob(os.path.join(HERE, "data", "train_*.npz"))):
    d = np.load(f); T = int(d["T_amb"])
    o = obs.setdefault(T, ob.MIONetObserver(T, config=cfg))
    Y, I, V = d["y_ref"].T, d["I_clean"], d["V_clean"]
    H0 = o.voltage(Y, I)
    num += np.sum((H0 - V) * I); den += np.sum(I ** 2)
    li_avg = o.lithium(Y)
    li_surf = Y[:, ob.N - 1] * o.li_p + Y[:, -1] * o.li_n
    li_all.append(li_avg)
    rows.append((os.path.basename(f)[:-4], H0, V, I, li_avg, li_surf))

k_ohm = num / den
li_target = float(np.mean(np.concatenate(li_all)))
print("k_ohm = %.5f Ohm (1/%.0f); public release 1/600" % (k_ohm, 1 / k_ohm))
print("%-28s %13s %13s %22s %22s" % ("scenario", "U RMSE 1/600", "U RMSE fit", "Li avg (min..max)", "Li surface (min..max)"))
for name, H0, V, I, la, ls in rows:
    print("%-28s %10.2f mV %10.2f mV   %.5f..%.5f   %.5f..%.5f" % (
        name, 1e3 * np.sqrt(np.mean((H0 - I / 600 - V) ** 2)), 1e3 * np.sqrt(np.mean((H0 - k_ohm * I - V) ** 2)),
        la.min(), la.max(), ls.min(), ls.max()))
print("li_target (avg) = %.5f; public surface target 0.94544" % li_target)

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(dict(k_ohm=k_ohm, li_target=li_target), open(os.path.join(HERE, "results", "measurement_fit.json"), "w"), indent=1)
