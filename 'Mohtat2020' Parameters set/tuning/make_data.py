"""
Step 0: builds the data sets used to tune and test the MIONet observer.

- train_*  : new plant simulations (DFN + lumped thermal), NOT used in the paper. Used for the
             ohmic fit (fit_measurement.py) and the NES tuning (tune_nes.py).
- test_*   : the archived cases in ../Saved_simulations (data of Fig. 9 / Table 2), with the
             archived noisy voltage and a seeded current noise.
- plant25_*: the 25 °C paper cases re-simulated without noise (isothermal DFN, as the archived
             plant), so that seed_test.py can draw several noise realisations.

Output: data/<name>.npz
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import observers as ob  # noqa: E402
import utils  # noqa: E402

OUT = os.path.join(HERE, "data")
NOISE = 1e-3  # std of voltage [V] and current [A] measurement noise

# name: (profile, CR, soc_0, T_end [s], T_amb [°C])     CC: CR < 0 is charge
TRAIN = {
    "train_cc_chg_03C_soc40": ("CC", -0.3, 0.40, 3600, 25),
    "train_cc_chg_075C_soc10": ("CC", -0.75, 0.10, 2400, 25),
    "train_cc_dis_05C_soc90": ("CC", 0.5, 0.90, 3600, 25),
    "train_udds_15C_soc70": ("UDDS", 1.5, 0.70, 2400, 25),
    "train_udds_1C_soc60_35deg": ("UDDS", 1.0, 0.60, 3600, 35),
}
PLANT25 = {
    "charging_01_25": ("CC", -0.1, 0.25, 21600),
    "charging_02_25": ("CC", -0.2, 0.25, 7200),
    "charging_05_25": ("CC", -0.5, 0.25, 4500),
    "charging_1_25": ("CC", -1.0, 0.25, 3600),  # stops at the upper voltage cut-off
    "UDDS_1_25": ("UDDS", 1.0, 0.80, 7200),
    "UDDS_2_25": ("UDDS", 3.0, 0.80, 3600),
}


def simulate(prof, CR, soc0, T_end, T_amb, thermal):
    pl = utils.plant(utils.make_param(T_amb), ob.N)
    t_eval = np.linspace(0, T_end, int(T_end / ob.DT) + 1)
    cur, _ = pl.CC(t_eval, CR) if prof == "CC" else pl.UDDS(t_eval, CR)
    _, y_ref, V_ref, _ = pl.ref_solution(soc0, cur, t_eval, thermal_model=thermal)
    n = len(V_ref)
    return dict(V_clean=V_ref, I_clean=cur[:n], y_ref=y_ref, soc_ref=pl.SOC_calculator(y_ref), T_amb=T_amb)


def make_train():
    for seed, (name, (prof, CR, soc0, T_end, T_amb)) in enumerate(TRAIN.items()):
        rng = np.random.default_rng(100 + seed)
        d = simulate(prof, CR, soc0, T_end, T_amb, thermal=True)
        n = len(d["V_clean"])
        np.savez_compressed(os.path.join(OUT, name + ".npz"), V=d["V_clean"] + rng.normal(0, NOISE, n),
                 I=d["I_clean"] + rng.normal(0, NOISE, n), **d)
        print("saved", name, flush=True)


def make_test():
    src = os.path.join(os.path.dirname(HERE), "Saved_simulations")
    for k, name in enumerate(sorted(f for f in os.listdir(src) if f.startswith(("charging_", "UDDS_")))):
        D = pickle.load(open(os.path.join(src, name), "rb"))
        rng = np.random.default_rng(200 + k)
        soc = np.asarray(D["plant"]["soc"], float); n = len(soc)
        np.savez_compressed(os.path.join(OUT, "test_" + name + ".npz"), V=np.asarray(D["plant"]["Voltage"], float)[:n],
                 I=np.asarray(D["plant"]["current"], float)[:n] + rng.normal(0, NOISE, n), soc_ref=soc,
                 T_amb=int(name.split("_")[-1]))
        print("saved test", name, flush=True)


def make_plant25():
    src = os.path.join(os.path.dirname(HERE), "Saved_simulations")
    for name, (prof, CR, soc0, T_end) in PLANT25.items():
        d = simulate(prof, CR, soc0, T_end, 25, thermal=False)
        A = pickle.load(open(os.path.join(src, name), "rb"))
        n = min(len(d["V_clean"]), len(A["plant"]["Voltage"]))
        dv = np.sqrt(np.mean((d["V_clean"][:n] - np.asarray(A["plant"]["Voltage"])[:n]) ** 2))
        ds = np.max(np.abs(d["soc_ref"][:n] - np.asarray(A["plant"]["soc"])[:n]))
        d.pop("y_ref")  # not needed by seed_test.py
        np.savez_compressed(os.path.join(OUT, "plant25_" + name + ".npz"), **d)
        print("saved plant25 %s: vs archive V RMSE %.2f mV (archive noise 1 mV), max |dSOC| %.3f %%" % (name, 1e3 * dv, ds), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    make_test()
    make_plant25()
    make_train()
