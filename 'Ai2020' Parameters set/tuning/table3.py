"""
Step 3 (Ai2020): recreates Table 3 of the paper (SOC RMSE on the experimental discharges).

For each discharge (0.5C, 1C, 2C from fully charged):
- measured data (../CC_*.npy): constant current C * CR, SOC reference = linear ramp. Both observers
  use the extra series resistance and voltage noise calibrated on the 2C discharge
  (results/measurement_fit.json), so 2C is a calibration case and 0.5C / 1C are the test cases;
- simulated counterpart (data/simtest_*.npz): DFN with the true SOC, 3 noise seeds.
Paper Table 3 also lists 0.1C, for which no measured data are available (PyBaMM Ai2020 data:
0.5C, 1C, 2C).

Output: results/table3.{json,md}
"""

import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import observers_ai as ob  # noqa: E402
import utils_ai  # noqa: E402

PAPER = {0.1: (0.20, 0.95), 0.5: (0.35, 1.40), 1: (0.58, 2.10)}  # Table 3: MIONet, ECM
ARCHIVE = {0.5: "CC_05_exp_4", 1: "CC_1_exp_4"}
SEEDS = (0, 1, 2)
NOISE = 1e-3
ECM_CFG = os.path.join(os.path.dirname(HERE), "ecm_config.json")  # NES-tuned ECM (tune_nes.py --ecm)


def ecms():
    out = {"ECM": ob.ECMObserver()}
    if os.path.exists(ECM_CFG):
        cfg = json.load(open(ECM_CFG)); cfg.pop("source", None)
        out["ECM (NES)"] = ob.ECMObserver(**cfg)
    return out


def run_exp(CR):
    _, V, soc_ref = utils_ai.load_experiment(CR, ob.DT)
    I = np.full(len(V), utils_ai.C_RATE_CAPACITY * CR)
    fit = json.load(open(os.path.join(HERE, "results", "measurement_fit.json")))
    kw = dict(soc_0=0.5, sigma_v=fit["sigma_v_exp"], r_extra=fit["r_extra"])
    out = {"MIONet": ob.metrics(ob.MIONetObserver(25).run(V, I, **kw), soc_ref)}
    out.update({k: ob.metrics(e.run(V, I, **kw), soc_ref) for k, e in ecms().items()})
    return CR, "exp", out


def run_sim(args):
    CR, seed = args
    d = np.load(os.path.join(HERE, "data", "simtest_%s.npz" % {0.5: "05C", 1: "1C", 2: "2C"}[CR]))
    rng = np.random.default_rng(1000 * seed + 3); n = len(d["V_clean"])
    V, I = d["V_clean"] + rng.normal(0, NOISE, n), d["I_clean"] + rng.normal(0, NOISE, n)
    out = {"MIONet": ob.metrics(ob.MIONetObserver(25).run(V, I), d["soc_ref"])}
    out.update({k: ob.metrics(e.run(V, I), d["soc_ref"]) for k, e in ecms().items()})
    return CR, seed, out


def archive_rmse(CR):
    if CR not in ARCHIVE:
        return None
    D = pickle.load(open(os.path.join(os.path.dirname(HERE), "Saved_simulations", ARCHIVE[CR]), "rb"))
    s = np.asarray(D["plant"]["soc"], float)
    return tuple(float(np.sqrt(np.mean((np.asarray(D[k]["soc"], float)[:len(s)] - s)[ob.IDX_SS:] ** 2))) for k in ("mionet", "ecm"))


if __name__ == "__main__":
    CRS = (0.5, 1, 2)
    with Pool(len(CRS) * (1 + len(SEEDS))) as pool:
        exp = pool.map_async(run_exp, CRS)
        sim = pool.map_async(run_sim, [(c, s) for c in CRS for s in SEEDS])
        exp, sim = exp.get(), sim.get()
    res = {str(c): {"exp": e} for c, _, e in exp}
    for c, s, out in sim:
        res[str(c)].setdefault("sim", {})[s] = out
    json.dump(res, open(os.path.join(HERE, "results", "table3.json"), "w"), indent=1)

    f = lambda m: "%.2f (CWC %.2f, PICP %.2f)" % (m["rmse"], m["cwc"], m["picp"])
    obsv = list(res[str(CRS[0])]["exp"])
    md = ["Table 3 recreated: SOC RMSE [%] (t >= 600 s); CWC [%] and PICP over the whole run. 2C is the "
          "calibration/tuning case; 0.5C and 1C are test cases. 'ECM' = original settings, 'ECM (NES)' = tuned "
          "on the same objective as the MIONet.", "",
          "| C-rate | paper MIONet / ECM | archive MIONet / ECM | " + " | ".join("measured: " + k for k in obsv)
          + " | " + " | ".join("simulated (3 seeds): " + k for k in obsv) + " |", "|---|---|---|" + "---|" * (2 * len(obsv))]
    for c in (0.1,) + CRS:
        p = "%.2f / %.2f" % PAPER[c] if c in PAPER else "-"
        a = archive_rmse(c); a = "%.2f / %.2f" % a if a else "-"
        if str(c) not in res:
            md.append("| %g | %s | %s | no measured data |" % (c, p, a) + " |" * (2 * len(obsv) - 1)); continue
        e = res[str(c)]["exp"]
        sm = lambda k: "%.3f ± %.3f" % (np.mean([res[str(c)]["sim"][s][k]["rmse"] for s in SEEDS]),
                                         np.std([res[str(c)]["sim"][s][k]["rmse"] for s in SEEDS]))
        md.append("| %g%s | %s | %s | %s | %s |" % (c, " (calibration)" if c == 2 else "", p, a,
                                                 " | ".join(f(e[k]) for k in obsv), " | ".join(sm(k) for k in obsv)))
    open(os.path.join(HERE, "results", "table3.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
