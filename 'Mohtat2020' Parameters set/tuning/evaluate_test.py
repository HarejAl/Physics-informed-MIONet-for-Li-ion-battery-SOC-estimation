"""
Step 3: evaluate the observer variants on the TEST set (archived Fig. 9 / Table 2 cases).

Variants
  A  public            : public weights, k_ohm = 1/600, surface Li balance
  B  ohmic             : public weights, fitted k_ohm, surface Li balance
  C  ohmic+NES         : NES weights,    fitted k_ohm, surface Li balance
  D  ohmic+NES, no Li  : NES weights,    fitted k_ohm, voltage only
  E  ohmic+NES, Li avg : NES weights,    fitted k_ohm, particle-averaged Li balance

All variants use the j0-only clamp (concentrations clamped to [0.05, 0.95] inside j0 only),
which is needed at the end of the 1C charge and inactive elsewhere.

Writes results/test_results.json and results/test_results.md
"""

import glob
import json
import os
import numpy as np
import fast_ukf as fu

R = os.path.join(fu.HERE, "results")
k_fit = json.load(open(os.path.join(fu.HERE, "k_ohm.json")))["k_ohm"]


def nes(v):
    f = os.path.join(R, "nes_%s.json" % v)
    if not os.path.exists(f):
        return None
    b = json.load(open(f))["best"]; b.setdefault("rLi", 1.0)
    return b


pub = {k: fu.PUBLIC[k] for k in ("P0", "q", "rV", "rLi")}
VARIANTS = {
    "A public": (pub, fu.PUBLIC["k_ohm"], True),
    "B ohmic": (pub, k_fit, True),
    "C ohmic+NES": (nes("li_surface"), k_fit, True),
    "D ohmic+NES no-Li": (nes("no_li"), k_fit, False),
    "E ohmic+NES Li-avg": (nes("li_avg"), k_fit, "avg"),
}
VARIANTS = {k: v for k, v in VARIANTS.items() if v[0] is not None}

obs = {}
res = {}
for f in sorted(glob.glob(os.path.join(fu.HERE, "data", "test_*.npz"))):
    d = np.load(f); name = os.path.basename(f)[5:-4]; T = int(d["T_amb"])
    o = obs.setdefault(T, fu.Observer(T))
    res[name] = dict(archive_mionet=float(d["archive_mionet_rmse"]), archive_ecm=float(d["archive_ecm_rmse"]))
    for vn, (w, k, li) in VARIANTS.items():
        Y, Pd = fu.run_ukf(o, d["V"], d["I"], w["P0"], w["q"], w["rV"], w["rLi"], k, use_li=li, sqrt="eigh", clamp=True)
        m = fu.metrics(o, Y, Pd, d["soc_ref"])
        res[name][vn] = dict(rmse=float(m["rmse"]), halfwidth=float(m["halfwidth"]), coverage=float(m["coverage"]))
    print(name, {vn: round(res[name][vn]["rmse"], 3) for vn in VARIANTS}, flush=True)

json.dump(dict(k_ohm=k_fit, weights={k: v[0] for k, v in VARIANTS.items()}, results=res),
          open(os.path.join(R, "test_results.json"), "w"), indent=1)

vn = list(VARIANTS)
lines = ["| case | paper MIONet | paper ECM | " + " | ".join(vn) + " |", "|" + "---|" * (3 + len(vn))]
for name, r in res.items():
    cells = ["%.3f (±%.2f, %.0f%%)" % (r[v]["rmse"], r[v]["halfwidth"], 100 * r[v]["coverage"]) for v in vn]
    lines.append("| %s | %.3f | %.3f | %s |" % (name, r["archive_mionet"], r["archive_ecm"], " | ".join(cells)))
mean = lambda key: np.mean([res[n][key]["rmse"] if isinstance(res[n][key], dict) else res[n][key] for n in res])
lines.append("| **mean** | %.3f | %.3f | %s |" % (mean("archive_mionet"), mean("archive_ecm"), " | ".join("%.3f" % mean(v) for v in vn)))
md = ("SOC RMSE [%] for t >= 600 s; in brackets: median 2-sigma half-width [%] and 2-sigma coverage.\n\n"
      + "k_ohm fitted = %.5f Ohm\n\n" % k_fit + "\n".join(lines) + "\n")
open(os.path.join(R, "test_results.md"), "w").write(md)
print(md)
