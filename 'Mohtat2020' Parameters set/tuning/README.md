# Retuning the public MIONet observer

Retunes the observer in `../SOC_estimation_comparison.py` / `../utils.py`, after the public
release turned out not to reproduce the archived Fig. 9 / Table 2 results.

Pipeline (run from this folder, in order):

1. `make_data.py`: training set (5 new plant simulations, not used in the paper) and
   test set (the 12 archived Mohtat2020 cases in `../Saved_simulations`). Output in `data/`.
2. `fit_ohmic.py`: least-squares fit of the ohmic term `k_ohm` of the measurement model
   on the true plant states of the training set. Output in `k_ohm.json` (1/465 Ohm; public 1/600).
3. `tune_nes.py <li_surface|no_li|li_avg>`: SNES tuning of the UKF weights
   (log10 P0, q, rV, rLi) on the training set. Output in `results/nes_<variant>.json`.
4. `evaluate_test.py`: evaluates all variants on the test set. Output in
   `results/test_results.{md,json}`.

`fast_ukf.py` is a batched copy of the public UKF: all sigma points go through the
MIONets and the measurement model in a single call, about 150x faster. Checked against the
public FilterPy run (0.5C charge: RMSE 6.639e-1 vs 6.634e-1, states within 1.6e-5).

Notes:
- `li_avg` replaces the surface-node lithium balance of the public script (not conserved,
  target 0.9454 biased) with the particle-averaged one (conserved to < 1e-4, target 0.94116).
- The evaluation clamps concentrations to [0.05, 0.95] **only inside j0**. The OCV, the state
  and the sigma points stay unclamped. Without the clamp, every variant fails at the end
  of the 1C charge.
- One noise realisation per case; the ECM was not retuned (paper ECM numbers are the archived ones).
