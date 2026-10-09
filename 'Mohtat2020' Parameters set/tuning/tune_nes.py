"""
Step 2: tune the UKF weights (P0, Q, R) of the MIONet observer with a
Separable Natural Evolution Strategy (SNES, Schaul et al. 2011).

Search space: log10 of [P0, q, rV(, rLi)] with P = P0*I, Q = q*I, R = diag(rV, rLi).
Objective (TRAINING set only, see make_data.py):
    J = mean SOC RMSE [%] (t >= 600 s, capped at 20 % per scenario) + max(0, 0.90 - mean 2-sigma coverage)
The coverage term stops NES from buying RMSE with an over-confident filter.

Usage:  python tune_nes.py <variant> [generations] [population] [workers]
    variant: li_surface (public 2nd measurement) | no_li | li_avg
"""

import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import fast_ukf as fu

VARIANTS = {"li_surface": True, "no_li": False, "li_avg": "avg"}
BOUNDS = {"P0": (-6, -1), "q": (-14, -4), "rV": (-8, -2), "rLi": (-10, 0)}
COVERAGE_TARGET = 0.90

_G = {}


def _init(use_li, k_ohm):
    _G["use_li"], _G["k_ohm"] = use_li, k_ohm
    _G["data"] = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(fu.HERE, "data", "train_*.npz")))]
    _G["obs"] = {T: fu.Observer(T) for T in {int(d["T_amb"]) for d in _G["data"]}}


def evaluate(theta, names, data=None, obs=None, use_li=None, k_ohm=None):
    data = data or _G["data"]; obs = obs or _G["obs"]
    use_li = _G["use_li"] if use_li is None else use_li
    k_ohm = _G["k_ohm"] if k_ohm is None else k_ohm
    w = dict(zip(names, 10.0 ** np.asarray(theta)))
    w.setdefault("rLi", 1.0)
    out = []
    for d in data:
        o = obs[int(d["T_amb"])]
        Y, Pd = fu.run_ukf(o, d["V"], d["I"], w["P0"], w["q"], w["rV"], w["rLi"], k_ohm, use_li=use_li, sqrt="eigh")
        out.append(fu.metrics(o, Y, Pd, d["soc_ref"]))
    return out


RMSE_CAP = 20.0  # [%] per-scenario cap, so diverging candidates are still ranked (no flat landscape)


def objective(theta, names):
    try:
        m = evaluate(theta, names)
    except Exception:
        return 1e3
    rmse = np.mean([min(x["rmse"], RMSE_CAP) if np.isfinite(x["rmse"]) else RMSE_CAP for x in m])
    cov = np.mean([x["coverage"] for x in m])
    return float(rmse + max(0.0, COVERAGE_TARGET - cov))


def _obj(args):
    return objective(*args)


def snes(names, mu, sigma, gens, pop, workers, rng, log):
    d = len(mu)
    lo = np.array([BOUNDS[n][0] for n in names]); hi = np.array([BOUNDS[n][1] for n in names])
    eta_mu, eta_s = 1.0, (3 + np.log(d)) / (5 * np.sqrt(d))
    u = np.maximum(0, np.log(pop / 2 + 1) - np.log(np.arange(1, pop + 1)))
    u = u / u.sum() - 1 / pop  # fitness shaping utilities
    best = (np.inf, None)
    with Pool(workers, initializer=_init, initargs=(_G["use_li"], _G["k_ohm"])) as pool:
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
    pop = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    use_li = VARIANTS[variant]
    k_ohm = json.load(open(os.path.join(fu.HERE, "k_ohm.json")))["k_ohm"]
    names = ["P0", "q", "rV"] + (["rLi"] if use_li else [])
    _G["use_li"], _G["k_ohm"] = use_li, k_ohm

    mu0 = np.log10([fu.PUBLIC[n] for n in names])  # start from the public weights
    t0 = time.time(); log = []
    J, theta = snes(names, mu0, np.ones(len(names)), gens, pop, workers, np.random.default_rng(0), log)
    res = dict(variant=variant, k_ohm=k_ohm, names=names, best_J=J, best_log10=theta.tolist(),
               best=dict(zip(names, (10.0 ** theta).tolist())), minutes=(time.time() - t0) / 60, history=log)
    os.makedirs(os.path.join(fu.HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(fu.HERE, "results", "nes_%s.json" % variant), "w"), indent=1)
    print("done", variant, res["best"], "J=%.4f" % J)
