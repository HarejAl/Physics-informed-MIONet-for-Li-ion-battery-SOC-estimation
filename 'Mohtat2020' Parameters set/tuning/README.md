# Tuning and testing the MIONet observer

Produces `../observer_config.json` and the comparison tables. Run from this folder, in order:

| step | script | what it does | output |
|---|---|---|---|
| 0 | `make_data.py` | training set: 5 new plant simulations, not used in the paper. Test set: the 12 archived cases in `../Saved_simulations`. Noise-free re-simulation of the 6 paper cases at 25 °C | `data/` |
| 1 | `fit_measurement.py` | least-squares fit of the ohmic term `k_ohm` on the true plant states, and the target of the particle-averaged lithium balance (training set) | `results/measurement_fit.json` |
| 2 | `tune_nes.py <variant>` | SNES tuning of the UKF weights (log10 of P0, q, rV, rLi) on the training set; variants `li_avg`, `li_surface`, `no_li` | `results/nes_<variant>.json` |
| 3 | `make_config.py [variant]` | writes `../observer_config.json` (default `li_avg`) | `../observer_config.json` |
| 4 | `evaluate_test.py` | all variants and the ECM on the 12 paper cases, next to the archived results | `results/test_results.{md,json}` |
| 5 | `seed_test.py` | 3 noise seeds on the 6 paper cases at 25 °C: final observer, public-release settings, ECM | `results/seed_test.{md,json}` |
| 6 | `table2.py` | recreates Table 2 of the paper (7 scenarios, 3 seeds, MIONet vs ECM) | `results/table2.{md,json}` |

`tune_nes.py` runs `python tune_nes.py li_avg 30 18 9` (30 generations, population 18,
9 worker processes) in about 15 min.

Notes:
- **NES objective:** mean SOC RMSE over the training scenarios (t ≥ 600 s, capped at 20 %
  per scenario), plus 5 × the shortfall of the 2σ coverage (whole run) below 95 %, matching the
  PICP of the CWC (μ = 0.95). The penalty stops NES
  from buying RMSE with an over-confident filter.
- **Lithium balance:** `li_avg` uses the particle-averaged concentrations, which the plant
  conserves to < 1e-4. `li_surface` (public release) uses the surface nodes, which are not
  conserved during current flow.
- **Clamp:** concentrations are clamped to [0.05, 0.95] only inside j0. Without it, the
  observer fails at the end of the 1C charge.
- **Temperature:** the MIONets were trained at 25 °C, so the observer is less accurate at
  other temperatures (UDDS 10–40 °C cases).
