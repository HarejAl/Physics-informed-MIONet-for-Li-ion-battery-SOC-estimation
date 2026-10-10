"""
Step 2: tunes the UKF weights of the MIONet observer with a Separable Natural Evolution
Strategy (SNES, Schaul et al., 2011) on the TRAINING set.

Search space: log10 of [P0, q, rV(, rLi)], with P = P0*I, Q = q*I, R = diag(rV, rLi).
Objective:
    J = mean SOC RMSE [%] (t >= 600 s, capped at 20 % per scenario)
        + max(0, 0.90 - mean 2-sigma coverage for t >= 600 s)
The coverage term stops NES from buying RMSE with an over-confident filter.

Usage:  python tune_nes.py <variant> [generations] [population] [workers]
    variant: li_avg (particle-averaged Li balance) | li_surface (public release) | no_li
Output:  results/nes_<variant>.json
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
import observers as ob  # noqa: E402

LI_SURFACE_TARGET = 0.9454372727901035  # target of the surface-node balance in the public release
VARIANTS = {"li_avg": "avg", "li_surface": "surface", "no_li": "none"}
PUBLIC = dict(P0=5e-3, q=(1e-4) ** 2, rV=(20e-3) ** 2, rLi=1e-3)  # start point: public release weights
BOUNDS = {"P0": (-6, -1), "q": (-14, -4), "rV": (-8, -2), "rLi": (-10, 0)}
COVERAGE_TARGET = 0.90
RMSE_CAP = 20.0

_G = {}


def base_config(variant):
    fit = json.load(open(os.path.join(HERE, "results", "measurement_fit.json")))
    mode = VARIANTS[variant]
    return dict(k_ohm=fit["k_ohm"], li_mode=mode, li_target=fit["li_target"] if mode == "avg" else LI_SURFACE_TARGET,
                rLi=1.0)


def _init(cfg):
    _G["cfg"] = cfg
    _G["data"] = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(HERE, "data", "train_*.npz")))]
    _G["obs"] = {T: ob.MIONetObserver(T, config=cfg) for T in {int(d["T_amb"]) for d in _G["data"]}}


def objective(theta, names):
    cfg = dict(_G["cfg"], **dict(zip(names, 10.0 ** np.asarray(theta))))
    rmse, cover = [], []
    try:
        for d in _G["data"]:
            o = _G["obs"][int(d["T_amb"])]; o.cfg = cfg
            r = o.run(d["V"], d["I"])
            e = (r["soc"] - d["soc_ref"])[ob.IDX_SS:]
            rm = np.sqrt(np.mean(e ** 2))
            rmse.append(min(rm, RMSE_CAP) if np.isfinite(rm) else RMSE_CAP)
            s = d["soc_ref"][ob.IDX_SS:]
            cover.append(np.mean((s >= r["soc_lower"][ob.IDX_SS:]) & (s <= r["soc_upper"][ob.IDX_SS:])))
    except Exception:
        return 1e3
    return float(np.mean(rmse) + max(0.0, COVERAGE_TARGET - np.mean(cover)))


def _obj(args):
    return objective(*args)


def snes(names, mu, sigma, gens, pop, workers, rng, log):
    d = len(mu)
    lo = np.array([BOUNDS[n][0] for n in names]); hi = np.array([BOUNDS[n][1] for n in names])
    eta_mu, eta_s = 1.0, (3 + np.log(d)) / (5 * np.sqrt(d))
    u = np.maximum(0, np.log(pop / 2 + 1) - np.log(np.arange(1, pop + 1)))
    u = u / u.sum() - 1 / pop  # fitness-shaping utilities
    best = (np.inf, None)
    with Pool(workers, initializer=_init, initargs=(_G["cfg"],)) as pool:
        for g in range(gens):
            eps = rng.standard_normal((pop // 2, d)); eps = np.vstack((eps, -eps))  # mirrored sampling
            X = np.clip(mu + sigma * eps, lo, hi)
            F = np.array(pool.map(_obj, [(x, names) for x in X]))
            order = np.argsort(F)
            uu = np.empty(pop); uu[order] = u
            mu = np.clip(mu + eta_mu * sigma * (uu @ eps), lo, hi)
            sigma = sigma * np.exp(eta_s / 2 * (uu @ (eps ** 2 - 1)))
            if F[order[0]] < best[0]:
                best = (F[order[0]], X[order[0]].copy())
            log.append(dict(gen=g, best_J=float(best[0]), gen_best_J=float(F[order[0]]), median_J=float(np.median(F)),
                            mu=mu.tolist(), sigma=sigma.tolist()))
            print("gen %2d  best J %.4f  gen-best %.4f  median %.4f  mu %s  sigma %s" % (
                g, best[0], F[order[0]], np.median(F), np.round(mu, 2), np.round(sigma, 2)), flush=True)
    return best


if __name__ == "__main__":
    variant = sys.argv[1]
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    pop = int(sys.argv[3]) if len(sys.argv) > 3 else 18
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 9
    cfg = base_config(variant); _G["cfg"] = cfg
    names = ["P0", "q", "rV"] + (["rLi"] if cfg["li_mode"] != "none" else [])

    t0 = time.time(); log = []
    J, theta = snes(names, np.log10([PUBLIC[n] for n in names]), np.ones(len(names)), gens, pop, workers,
                    np.random.default_rng(0), log)
    best = dict(cfg, **dict(zip(names, (10.0 ** theta).tolist())))
    json.dump(dict(variant=variant, J=J, config=best, minutes=(time.time() - t0) / 60, history=log),
              open(os.path.join(HERE, "results", "nes_%s.json" % variant), "w"), indent=1)
    print("done", variant, best, "J=%.4f" % J)
