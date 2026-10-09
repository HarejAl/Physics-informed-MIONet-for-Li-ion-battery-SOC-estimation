"""
Step 1: retune the ohmic correction term of the measurement model.

U = OCP(surface) - eta_n + eta_p - k_ohm * i       (public: k_ohm = 1/600 Ohm)

k_ohm is fitted by least squares on the TRAINING set, evaluating the measurement model
on the true plant concentrations (y_ref) and the noise-free plant voltage / current,
so the fit only absorbs the model mismatch between the observer's voltage map and the DFN.

Also prints a diagnostic of the lithium-balance pseudo-measurement: how constant the
quantity used in the public script actually is along the true trajectories.
"""

import glob
import json
import os
import numpy as np
import fast_ukf as fu

files = sorted(glob.glob(os.path.join(fu.HERE, "data", "train_*.npz")))
num = den = 0.0
rows = []
for f in files:
    d = np.load(f)
    obs = fu.Observer(int(d["T_amb"]))
    Y, I, V = d["y_ref"].T, d["I_clean"], d["V_clean"]
    H0 = np.array([obs.voltage(Y[j:j + 1], I[j], 0.0)[0] for j in range(len(I))])
    num += np.sum((H0 - V) * I); den += np.sum(I ** 2)
    li = Y[:, fu.N - 1] * obs.li_p + Y[:, -1] * obs.li_n
    rows.append((os.path.basename(f)[:-4], H0, V, I, li))

k_fit = num / den
print("fitted k_ohm = %.5f Ohm  (1/%.0f)   public: 1/600 = %.5f" % (k_fit, 1 / k_fit, 1 / 600))
print("%-28s %12s %12s %22s" % ("scenario", "H RMSE 1/600", "H RMSE fit", "Li balance min..max"))
for name, H0, V, I, li in rows:
    e_pub = np.sqrt(np.mean((H0 - I / 600 - V) ** 2)) * 1e3
    e_fit = np.sqrt(np.mean((H0 - k_fit * I - V) ** 2)) * 1e3
    print("%-28s %9.2f mV %9.2f mV   %.4f..%.4f (target %.4f)" % (name, e_pub, e_fit, li.min(), li.max(), fu.LI_TARGET))

json.dump({"k_ohm": k_fit}, open(os.path.join(fu.HERE, "k_ohm.json"), "w"), indent=1)
