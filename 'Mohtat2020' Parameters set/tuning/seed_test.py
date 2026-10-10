"""
Step 5: 3-seed test on the 25 °C paper cases (re-simulated plants, data/plant25_*.npz).

For each case and seed, new voltage / current noise (std 1e-3) is drawn and the final MIONet
observer (../observer_config.json), the public-release MIONet settings and the ECM (paper
settings) are run. Reported: mean ± std over the seeds of SOC RMSE (t >= 600 s) and CWC,
next to the archived paper results (one realisation).

Output: results/seed_test.{json,md}   (python seed_test.py --report: table only, from the saved json)
"""

import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import observers as ob  # noqa: E402
from paper_reference import paper_metrics  # noqa: E402
from evaluate_test import VARIANTS  # noqa: E402

SEEDS = (0, 1, 2)
NOISE = 1e-3


def run_one(args):
    f, seed = args
    d = np.load(f); n = len(d["V_clean"])
    rng = np.random.default_rng(1000 * seed + 7)
    V = d["V_clean"] + rng.normal(0, NOISE, n); I = d["I_clean"] + rng.normal(0, NOISE, n)
    o = ob.MIONetObserver(25)
    out = {"MIONet (final)": ob.metrics(o.run(V, I), d["soc_ref"])}
    o.cfg = VARIANTS["public"]
    out["MIONet (public settings)"] = ob.metrics(o.run(V, I), d["soc_ref"])
    out["ECM"] = ob.metrics(ob.ECMObserver().run(V, I), d["soc_ref"])
    return os.path.basename(f)[8:-4], seed, out


if __name__ == "__main__":
    out_json = os.path.join(HERE, "results", "seed_test.json")
    if "--report" in sys.argv:  # only rebuild the table from the saved results
        res = {c: {int(s): v for s, v in r.items()} for c, r in json.load(open(out_json)).items()}
    else:
        jobs = [(f, s) for f in sorted(glob.glob(os.path.join(HERE, "data", "plant25_*.npz"))) for s in SEEDS]
        with Pool(min(len(jobs), 18)) as pool:
            raw = pool.map(run_one, jobs)
        res = {}
        for case, seed, out in raw:
            res.setdefault(case, {})[seed] = out
        json.dump(res, open(out_json, "w"), indent=1)

    obsv = ["MIONet (final)", "MIONet (public settings)", "ECM"]
    md = ["3-seed test on the 25 °C paper cases. RMSE [%%] (t >= 600 s) and CWC [%%] (whole run), mean ± std over "
          "seeds %s; paper = archived results (one realisation)." % (SEEDS,), "",
          "| case | metric | paper MIONet | paper ECM | " + " | ".join(obsv) + " |", "|---|---|---|---|" + "---|" * len(obsv)]
    for case in sorted(res):
        p = paper_metrics(case)
        for m in ("rmse", "cwc"):
            cells = []
            for k in obsv:
                v = np.array([res[case][s][k][m] for s in SEEDS])
                cells.append("%.3f ± %.3f" % (v.mean(), v.std()))
            md.append("| %s | %s | %.3f | %.3f | %s |" % (case, m.upper(), p["mionet"][m], p["ecm"][m], " | ".join(cells)))
    open(os.path.join(HERE, "results", "seed_test.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
