"""
Step 3: writes ../observer_config.json (used by observers.MIONetObserver) from the NES result
of the selected variant (default: li_avg).

Usage:  python make_config.py [variant]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
variant = sys.argv[1] if len(sys.argv) > 1 else "li_avg"
res = json.load(open(os.path.join(HERE, "results", "nes_%s.json" % variant)))
cfg = dict(res["config"], source="tuning/results/nes_%s.json (J = %.4f)" % (variant, res["J"]))
json.dump(cfg, open(os.path.join(os.path.dirname(HERE), "observer_config.json"), "w"), indent=1)
print(json.dumps(cfg, indent=1))
