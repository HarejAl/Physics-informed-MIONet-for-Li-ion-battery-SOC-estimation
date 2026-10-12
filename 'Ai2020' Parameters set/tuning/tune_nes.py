"""
Step 2 (Ai2020): SNES tuning of the UKF weights on the TRAINING set: log10 of P0 / q (independent
node terms), P0_soc / q_soc (SOC mode, see observers_ai.MIONetObserver.run), rV, rLi.
Objective over the 5 training simulations AND the measured 2C discharge (the calibration case of
fit_measurement.py; Table 3 uses 0.5C and 1C, which are never seen here):
    J = mean SOC RMSE [%] (t >= 600 s, capped at 20 %) + 5 * max(0, 0.95 - mean 2-sigma coverage, whole run)
The measured case makes the tuning account for the real cell's model mismatch.
With --ecm, the ECM weights (log10 of q_soc, q_V, P0_soc, P0_V, r) are tuned on the same objective,
so that both observers get the same treatment.

Usage:  python tune_nes.py [generations] [population] [workers] [--ecm]
Output: results/nes.json + ../observer_config.json   (--ecm: results/nes_ecm.json + ../ecm_config.json)
"""

import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import observers_ai as ob  # noqa: E402
import utils_ai  # noqa: E402

ECM = "--ecm" in sys.argv
if ECM:  # start point: original Ai2020 script settings
    START = dict(q_soc=0.1 * (0.5e-2) ** 2, q_V=0.1 * (1e-3) ** 2, P0_soc=5e-2, P0_V=1e-8, r=(1e-3) ** 2)
    BOUNDS = {"q_soc": (-12, -3), "q_V": (-14, -4), "P0_soc": (-4, 0), "P0_V": (-12, -2), "r": (-8, -2)}
else:
    START = dict(P0=1e-6, P0_soc=1e-1, q=1e-10, q_soc=1e-9, rV=(5e-3) ** 2, rLi=1e-4)
    BOUNDS = {"P0": (-9, -2), "P0_soc": (-4, 0), "q": (-14, -6), "q_soc": (-14, -5), "rV": (-8, -2), "rLi": (-10, 0)}
NAMES = list(BOUNDS)
COVERAGE_TARGET, COVERAGE_WEIGHT, RMSE_CAP = 0.95, 5.0, 20.0

_G = {}


def _init(cfg):
    _G["cfg"] = cfg
    _G["data"] = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(HERE, "data", "train_*.npz")))]
    fit = json.load(open(os.path.join(HERE, "results", "measurement_fit.json")))
    _, V, soc = utils_ai.load_experiment(2, ob.DT)
    _G["data"].append(dict(V=V, I=np.full(len(V), 2 * utils_ai.C_RATE_CAPACITY), soc_ref=soc,
                           kw=dict(sigma_v=fit["sigma_v_exp"], r_extra=fit["r_extra"])))
    _G["obs"] = None if ECM else ob.MIONetObserver(25, config=cfg)


def objective(theta):
    params = dict(zip(NAMES, 10.0 ** np.asarray(theta)))
    if ECM:
        o = ob.ECMObserver(**params)
    else:
        o = _G["obs"]; o.cfg = dict(_G["cfg"], **params)
    rmse, cover = [], []
    try:
        for d in _G["data"]:
            r = o.run(d["V"], d["I"], **d.get("kw", {}))
            rm = np.sqrt(np.mean((r["soc"] - d["soc_ref"])[ob.IDX_SS:] ** 2))
            rmse.append(min(rm, RMSE_CAP) if np.isfinite(rm) else RMSE_CAP)
            s = d["soc_ref"]
            cover.append(np.mean((s >= r["soc_lower"]) & (s <= r["soc_upper"])))
    except Exception:
        return 1e3
    return float(np.mean(rmse) + COVERAGE_WEIGHT * max(0.0, COVERAGE_TARGET - np.mean(cover)))


def snes(cfg, gens, pop, workers, rng, log):
    d = len(NAMES)
    lo = np.array([BOUNDS[n][0] for n in NAMES]); hi = np.array([BOUNDS[n][1] for n in NAMES])
    mu, sigma = np.log10([START[n] for n in NAMES]), np.ones(d)
    eta_s = (3 + np.log(d)) / (5 * np.sqrt(d))
    u = np.maximum(0, np.log(pop / 2 + 1) - np.log(np.arange(1, pop + 1))); u = u / u.sum() - 1 / pop
    best = (np.inf, None)
    with Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for g in range(gens):
            eps = rng.standard_normal((pop // 2, d)); eps = np.vstack((eps, -eps))
            X = np.clip(mu + sigma * eps, lo, hi)
            F = np.array(pool.map(objective, list(X)))
            order = np.argsort(F); uu = np.empty(pop); uu[order] = u
            mu = np.clip(mu + sigma * (uu @ eps), lo, hi)
            sigma = sigma * np.exp(eta_s / 2 * (uu @ (eps ** 2 - 1)))
            if F[order[0]] < best[0]:
                best = (F[order[0]], X[order[0]].copy())
            log.append(dict(gen=g, best_J=float(best[0]), median_J=float(np.median(F)), mu=mu.tolist(), sigma=sigma.tolist()))
            print("gen %2d  best J %.4f  median %.4f  mu %s  sigma %s" % (g, best[0], np.median(F), np.round(mu, 2), np.round(sigma, 2)), flush=True)
    return best


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gens = int(args[0]) if len(args) > 0 else 30
    pop = int(args[1]) if len(args) > 1 else 18
    workers = int(args[2]) if len(args) > 2 else 18
    fit = json.load(open(os.path.join(HERE, "results", "measurement_fit.json")))
    cfg = {} if ECM else dict(k_ohm=fit["k_ohm"], li_target=fit["li_target"], flux_projection=True)
    t0 = time.time(); log = []
    J, theta = snes(cfg, gens, pop, workers, np.random.default_rng(0), log)
    best = dict(cfg, **dict(zip(NAMES, (10.0 ** theta).tolist())))
    res, out = ("nes_ecm.json", "ecm_config.json") if ECM else ("nes.json", "observer_config.json")
    json.dump(dict(J=J, config=best, minutes=(time.time() - t0) / 60, history=log),
              open(os.path.join(HERE, "results", res), "w"), indent=1)
    json.dump(dict(best, source="tuning/results/%s (J = %.4f)" % (res, J)),
              open(os.path.join(os.path.dirname(HERE), out), "w"), indent=1)
    print("done", best, "J=%.4f" % J)
