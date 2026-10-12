"""
Step 0 (Ai2020): data sets for tuning and testing the MIONet observer.

- train_*  : new DFN simulations (isothermal, 25 °C), NOT the Table 3 discharges. Used for the
             measurement fit and the NES tuning.
- simtest_*: DFN simulations of the Table 3 discharges (0.5C, 1C, 2C from fully charged to the
             cut-off), noise-free, with the true SOC. Noise is added by table3.py.
The measured discharges are read directly from ../CC_*.npy (utils_ai.load_experiment).

Output: data/<name>.npz
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import observers_ai as ob  # noqa: E402
import utils_ai  # noqa: E402

OUT = os.path.join(HERE, "data")
NOISE = 1e-3

# name: (profile, CR, soc_0, T_end [s])    CC: CR > 0 discharge, CR < 0 charge
TRAIN = {
    "train_cc_dis_03C_soc100": ("CC", 0.3, 1.00, 7200),
    "train_cc_dis_075C_soc90": ("CC", 0.75, 0.90, 3600),
    "train_cc_dis_15C_soc80": ("CC", 1.5, 0.80, 1500),
    "train_cc_chg_05C_soc20": ("CC", -0.5, 0.20, 4300),
    "train_udds_1C_soc80": ("UDDS", 1.0, 0.80, 2400),
}
SIMTEST = {"simtest_05C": 0.5, "simtest_1C": 1, "simtest_2C": 2}


def simulate(prof, CR, soc0, T_end):
    pl = utils_ai.plant(utils_ai.make_param(25), ob.N)
    t_eval = np.linspace(0, T_end, int(T_end / ob.DT) + 1)
    cur, _ = pl.CC(t_eval, CR) if prof == "CC" else pl.UDDS(t_eval, CR)
    _, y_ref, V_ref, _ = pl.ref_solution(soc0, cur, t_eval)
    n = len(V_ref)
    return dict(V_clean=V_ref, I_clean=cur[:n], y_ref=y_ref, soc_ref=pl.SOC_calculator(y_ref), T_amb=25)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for seed, (name, args) in enumerate(TRAIN.items()):
        rng = np.random.default_rng(100 + seed)
        d = simulate(*args); n = len(d["V_clean"])
        np.savez_compressed(os.path.join(OUT, name + ".npz"), V=d["V_clean"] + rng.normal(0, NOISE, n),
                            I=d["I_clean"] + rng.normal(0, NOISE, n), **d)
        print("saved", name, "(%d steps)" % n, flush=True)
    for name, CR in SIMTEST.items():
        d = simulate("CC", CR, 1.0, 3800 / CR); d.pop("y_ref")
        np.savez_compressed(os.path.join(OUT, name + ".npz"), **d)
        print("saved", name, "(%d steps, SOC %.1f -> %.1f %%)" % (len(d["V_clean"]), d["soc_ref"][0], d["soc_ref"][-1]), flush=True)
