"""
Step 4: evaluates the observer variants on the TEST set (archived Fig. 9 / Table 2 cases,
one noise realisation each), next to the archived paper results.

MIONet variants (all with the j0-only clamp):
  public    : public-release weights, k_ohm = 1/600, surface Li balance
  ohmic     : public-release weights, fitted k_ohm, surface Li balance
  NES surf  : NES weights,            fitted k_ohm, surface Li balance
  NES no-Li : NES weights,            fitted k_ohm, voltage only
  NES avg   : NES weights,            fitted k_ohm, particle-averaged Li balance (= observer_config.json)
ECM: paper settings, on the same data.

Output: results/test_results.{json,md}
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import observers as ob  # noqa: E402
from paper_reference import paper_metrics  # noqa: E402
from tune_nes import LI_SURFACE_TARGET, PUBLIC  # noqa: E402

R = os.path.join(HERE, "results")
fit = json.load(open(os.path.join(R, "measurement_fit.json")))
nes = lambda v: json.load(open(os.path.join(R, "nes_%s.json" % v)))["config"]
surf = dict(li_mode="surface", li_target=LI_SURFACE_TARGET)
VARIANTS = {
    "public": dict(PUBLIC, k_ohm=1 / 600, **surf),
    "ohmic": dict(PUBLIC, k_ohm=fit["k_ohm"], **surf),
    "NES surf": nes("li_surface"),
    "NES no-Li": nes("no_li"),
    "NES avg": nes("li_avg"),
}

if __name__ == "__main__":
    obs, res = {}, {}
    ecm = ob.ECMObserver()
    for f in sorted(glob.glob(os.path.join(HERE, "data", "test_*.npz"))):
        d = np.load(f); case = os.path.basename(f)[5:-4]; T = int(d["T_amb"])
        o = obs.setdefault(T, ob.MIONetObserver(T, config=VARIANTS["public"]))
        res[case] = {"paper MIONet": (p := paper_metrics(case))["mionet"], "paper ECM": p["ecm"],
                     "ECM": ob.metrics(ecm.run(d["V"], d["I"]), d["soc_ref"])}
        for vn, cfg in VARIANTS.items():
            o.cfg = cfg
            res[case][vn] = ob.metrics(o.run(d["V"], d["I"]), d["soc_ref"])
        print(case, {k: round(v["rmse"], 3) for k, v in res[case].items()}, flush=True)

    json.dump(dict(variants=VARIANTS, results=res), open(os.path.join(R, "test_results.json"), "w"), indent=1)

    cols = list(next(iter(res.values())))
    md = ["SOC estimation on the archived paper cases (one noise realisation). RMSE [%] for t >= 600 s / CWC [%] "
          "(mu = 0.95, gamma = 1, eta = 1, whole run). 'paper' columns: archived results.", "",
          "| case | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for case, r in res.items():
        md.append("| %s | " % case + " | ".join("%.3f / %.2f" % (r[c]["rmse"], r[c]["cwc"]) for c in cols) + " |")
    md.append("| **mean** | " + " | ".join("%.3f / %.2f" % (np.mean([res[k][c]["rmse"] for k in res]),
                                                         np.mean([res[k][c]["cwc"] for k in res])) for c in cols) + " |")
    open(os.path.join(R, "test_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
