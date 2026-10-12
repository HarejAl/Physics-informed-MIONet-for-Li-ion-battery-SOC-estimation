# Ai2020 (Enertech LCO-G cell): tuning and experimental validation

Produces `../observer_config.json` and Table 3 (SOC estimation on measured discharges).
Run from this folder, in order:

| step | script | what it does | output |
|---|---|---|---|
| 0 | `make_data.py` | 5 training DFN simulations (not the Table 3 discharges); DFN counterparts of the measured 0.5C / 1C / 2C discharges | `data/` |
| 1 | `fit_measurement.py` | ohmic term `k_ohm` (training simulations); exact lithium total from the parameter set; extra series resistance and voltage noise of the real cell from the **measured 2C** discharge only | `results/measurement_fit.json` |
| 2 | `tune_nes.py [--ecm]` | SNES tuning of the UKF weights on the training simulations plus the measured 2C discharge; writes `../observer_config.json` (MIONet) or, with `--ecm`, `../ecm_config.json` (ECM, same objective) | `results/nes.json`, `results/nes_ecm.json` |
| 3 | `table3.py` | measured 0.5C / 1C / 2C and simulated counterparts (3 seeds): MIONet, ECM (original settings), ECM (NES) | `results/table3.{md,json}` |

## Ai2020-specific points

- **34 parallel electrodes.** The MIONets were trained on the flux of one electrode pair, so they
  can't be fed the cell current directly. The observer uses a **flux-consistent step**: the MIONets
  give the shape of the radial profiles, and the particle averages follow the exact Coulomb-counted
  flux for the current density I / A_eff (A_eff = 34 × 0.0024 m²). Open loop against the DFN, SOC
  error drops from up to 43 % to ≤ 0.1 %.
- **Lithium balance.** Particle-averaged pseudo-measurement with the exact total lithium of the
  parameter set (2.04093). The original script's target (2.0581, simple mean of the nodes) wasn't
  consistent. The initial state (SOC 50 %) is built on the cell's ElectrodeSOH stoichiometry
  window, so it starts with the correct lithium total.
- **Covariances.** An SOC mode (all nodes shifted along the stoichiometry window, lithium
  conserved) plus independent node terms, for both P0 and Q.
- **Ohmic drop.** `k_ohm` fitted on the simulations. The real cell has about 17 mΩ of extra series
  resistance that the Ai2020 model lacks (measured vs DFN voltage gap grows with current). It's
  calibrated on the 2C discharge and applied to **both** observers on measured data, together with
  the voltage noise level (20.8 mV) from the same comparison.

## Notes and limitations

- **Calibration vs test.** 2C is the calibration and tuning case. 0.5C and 1C are never used for
  calibration or tuning.
- **SOC reference.** The measured data contain only the voltage. The reference is the linear SOC
  ramp from 100 % to the cut-off, as in the paper.
- **No 0.1C data.** The PyBaMM Ai2020 data set has 0.5C, 1C and 2C discharges only.
- **Fair ECM.** Besides its original settings, the ECM is NES-tuned on the same objective as the MIONet
  (`ecm_config.json`); both use the same extra resistance and voltage noise on measured data.
- **Intervals.** On measured data the MIONet's 2σ intervals don't cover the reference reliably
  (PICP 0.4–0.7), so the precision on the experimental case is not claimed.
- **Legacy files.** `../Comparison_with_exp.py` (measured vs simulated voltage) and
  `../Saved_simulations/` (archived runs of the original script) are kept for reference.
