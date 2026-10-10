"""Metrics of the archived paper results (../Saved_simulations), same definitions as observers.metrics."""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import observers as ob  # noqa: E402


def paper_metrics(case):
    D = pickle.load(open(os.path.join(os.path.dirname(HERE), "Saved_simulations", case), "rb"))
    soc_ref = np.asarray(D["plant"]["soc"], float)
    out = {}
    for k in ("mionet", "ecm"):
        a, b = np.asarray(D[k]["soc_upper"], float), np.asarray(D[k]["soc_bottom"], float)
        # in the archive the MIONet 'upper'/'bottom' labels are swapped: use min / max
        res = dict(soc=np.asarray(D[k]["soc"], float), soc_lower=np.minimum(a, b), soc_upper=np.maximum(a, b))
        out[k] = ob.metrics(res, soc_ref)
    return out
