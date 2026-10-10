"""
Recreates Table 2 of the paper (SOC estimation, MIONet vs ECM) with the current repository.

Scenarios as in Table 2 (C-rate, duration), 25 °C, isothermal DFN plant (as the archived cases).
Initial SOC: 80 % for UDDS, 25 % for the charging cases. The plant stops at the voltage cut-off
if reached before the end of the scenario. Each scenario is run with 3 noise seeds (std 1e-3 on
voltage and current); MIONet with ../observer_config.json, ECM with the paper settings.

Output: data/table2_<scenario>.npz (noise-free plants), results/table2.{json,md}
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import observers as ob  # noqa: E402
from make_data import simulate  # noqa: E402

SEEDS = (0, 1, 2)
NOISE = 1e-3
# name: (profile, CR, soc_0, T_end [s], paper (MIONet RMSE, ECM RMSE, MIONet CWC, ECM CWC))
SCENARIOS = {
    "UDDS 1": ("UDDS", 1, 0.80, 2 * 3600, (0.058, 0.18, 0.63, 1.2)),
    "UDDS 2": ("UDDS", 3, 0.80, 1 * 3600, (0.044, 0.58, 0.60, 2.7)),
    "UDDS 3": ("UDDS", 5, 0.80, 1 * 3600, (0.25, 2.1, 2.1, 4.2)),
    "Ultra-slow charge": ("CC", -0.1, 0.25, 6 * 3600, (0.063, 0.49, 0.83, 3.4)),
    "Slow charge": ("CC", -0.2, 0.25, 4 * 3600, (0.079, 0.34, 1.0, 1.5)),
    "Moderate charge": ("CC", -0.5, 0.25, 1.5 * 3600, (0.13, 0.31, 1.6, 3.0)),
    "Fast charge": ("CC", -1.0, 0.25, 0.8 * 3600, (0.45, 0.37, 2.7, 2.8)),
}
fname = lambda s: os.path.join(HERE, "data", "table2_%s.npz" % s.replace(" ", "_").replace("-", "_"))


def make_plant(name):
    prof, CR, soc0, T_end, _ = SCENARIOS[name]
    if not os.path.exists(fname(name)):
        d = simulate(prof, CR, soc0, T_end, 25, thermal=False)
        d.pop("y_ref")
        np.savez_compressed(fname(name), **d)
    return name


def run_one(args):
    name, seed = args
    d = np.load(fname(name)); n = len(d["V_clean"])
    rng = np.random.default_rng(1000 * seed + 11)
    V = d["V_clean"] + rng.normal(0, NOISE, n); I = d["I_clean"] + rng.normal(0, NOISE, n)
    return name, seed, n * ob.DT / 3600, {"MIONet": ob.metrics(ob.MIONetObserver(25).run(V, I), d["soc_ref"]),
                                          "ECM": ob.metrics(ob.ECMObserver().run(V, I), d["soc_ref"])}


if __name__ == "__main__":
    with Pool(7) as pool:
        pool.map(make_plant, list(SCENARIOS))
    with Pool(min(21, os.cpu_count() or 1)) as pool:
        raw = pool.map(run_one, [(s, k) for s in SCENARIOS for k in SEEDS])
    res, dur = {}, {}
    for name, seed, h, out in raw:
        res.setdefault(name, {})[seed] = out; dur[name] = h
    json.dump(dict(results=res, duration_h=dur), open(os.path.join(HERE, "results", "table2.json"), "w"), indent=1)

    ms = lambda name, k, m: (np.mean([res[name][s][k][m] for s in SEEDS]), np.std([res[name][s][k][m] for s in SEEDS]))
    md = ["Table 2 recreated with the current repository: mean ± std over seeds %s. In brackets: Table 2 of the paper." % (SEEDS,),
          "RMSE [%] for t >= 600 s; CWC [%] with mu = 0.95, gamma = eta = 1 over the whole run.", "",
          "| Scenario | C-rate | Duration (h) | RMSE MIONet | RMSE ECM | CWC MIONet | CWC ECM |", "|---|---|---|---|---|---|---|"]
    for name, (prof, CR, soc0, T_end, paper) in SCENARIOS.items():
        cells = []
        for (k, m), p in zip([("MIONet", "rmse"), ("ECM", "rmse"), ("MIONet", "cwc"), ("ECM", "cwc")], paper):
            mu, sd = ms(name, k, m)
            cells.append("%.3f ± %.3f (%s)" % (mu, sd, p))
        md.append("| %s | %g | %.2f (%g) | %s |" % (name, abs(CR), dur[name], T_end / 3600, " | ".join(cells)))
    open(os.path.join(HERE, "results", "table2.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
