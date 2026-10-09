"""
Builds the training and test sets used for retuning the MIONet observer.

Training (tuning only, NOT used in the paper): new plant simulations with the public
plant (DFN + lumped thermal), different C-rates / initial SOCs / temperatures.
Test: the archived Mohtat2020 cases in ../Saved_simulations (Fig. 9 / Table 2 data).

Output: data/<name>.npz with V (noisy voltage), I (noisy current), soc_ref, T_amb
(+ y_ref and V_clean for the training set, used by fit_ohmic.py).
"""

import os
import pickle
import numpy as np
import fast_ukf as fu
import utils

OUT = os.path.join(fu.HERE, "data")
os.makedirs(OUT, exist_ok=True)
NOISE = 1e-3

# name: (profile, CR, soc_0, T_end [s], T_amb)   CR < 0 is charge for CC
TRAIN = {
    "train_cc_chg_03C_soc40": ("CC", -0.3, 0.40, 3600, 25),
    "train_cc_chg_075C_soc10": ("CC", -0.75, 0.10, 2400, 25),
    "train_cc_dis_05C_soc90": ("CC", 0.5, 0.90, 3600, 25),
    "train_udds_15C_soc70": ("UDDS", 1.5, 0.70, 2400, 25),
    "train_udds_1C_soc60_35deg": ("UDDS", 1.0, 0.60, 3600, 35),
}


def make_train():
    os.chdir(fu.ROOT)  # utils.plant.UDDS loads UDDS_profile.mat from the cwd
    for seed, (name, (prof, CR, soc0, T_end, T_amb)) in enumerate(TRAIN.items()):
        f = os.path.join(OUT, name + ".npz")
        if os.path.exists(f):
            continue
        rng = np.random.default_rng(100 + seed)
        param = fu.make_param(T_amb)
        pl = utils.plant(param, fu.N)
        t_eval = np.linspace(0, T_end, int(T_end / fu.DT) + 1)
        cur, _ = pl.CC(t_eval, CR) if prof == "CC" else pl.UDDS(t_eval, CR)
        _, y_ref, V_ref, _ = pl.ref_solution(soc0, cur, t_eval, thermal_model=True)
        np.savez(f, V=V_ref + rng.normal(0, NOISE, len(V_ref)), I=cur + rng.normal(0, NOISE, len(cur)),
                 I_clean=cur, V_clean=V_ref, y_ref=y_ref, soc_ref=pl.SOC_calculator(y_ref), T_amb=T_amb)
        print("saved", name)


def make_test():
    src = os.path.join(fu.ROOT, "Saved_simulations")
    for k, name in enumerate(sorted(f for f in os.listdir(src) if f.startswith(("charging_", "UDDS_")))):
        D = pickle.load(open(os.path.join(src, name), "rb"))
        rng = np.random.default_rng(200 + k)
        I = np.asarray(D["plant"]["current"], float)
        soc = np.asarray(D["plant"]["soc"], float)
        np.savez(os.path.join(OUT, "test_" + name + ".npz"), V=np.asarray(D["plant"]["Voltage"], float)[: len(soc)],
                 I=I[: len(soc)] + rng.normal(0, NOISE, len(soc)), soc_ref=soc, T_amb=int(name.split("_")[-1]),
                 archive_mionet_rmse=np.sqrt(np.mean((np.asarray(D["mionet"]["soc"]) - soc)[300:] ** 2)),
                 archive_ecm_rmse=np.sqrt(np.mean((np.asarray(D["ecm"]["soc"]) - soc)[300:] ** 2)))
        print("saved test", name)


if __name__ == "__main__":
    make_test()
    make_train()
