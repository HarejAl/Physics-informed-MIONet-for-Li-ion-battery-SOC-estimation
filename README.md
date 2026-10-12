# SOC_PIMIONet

Code for "Physics-informed operator learning for real-time battery state estimation"
(Applied Energy, DOI: 10.1016/j.apenergy.2025.126987).

## Repository structure

- `Training/`: training of the two MIONets (positive and negative electrode).
- `TimeComplexityAnalysis/`: runtime and computational-effort comparison.
- `HCCP_test/`: simulation of the HCCP test used to identify the ECM parameters.
- `'Mohtat2020' Parameters set/`: MIONet- vs ECM-based observers, Mohtat2020 parameter set.
  - `SOC_estimation_comparison.py`: main script. Simulates the plant, runs both observers and
    prints the SOC RMSE and CWC.
  - `observers.py`: MIONet observer (batched UKF) and ECM observer, plus metrics (RMSE, CWC).
  - `utils.py`: plant (PyBaMM DFN), current profiles, SOC calculation.
  - `observer_config.json`: settings of the MIONet observer, produced by `tuning/`.
  - `tuning/`: fit of the measurement model and NES tuning of the UKF weights, test on the
    paper cases (see `tuning/README.md`).
  - `Saved_simulations/`: archived results of Fig. 9 / Table 2, with `plotting.py`.
- `'Ai2020' Parameters set/`: MIONet- vs ECM-based observers for the Enertech cell (Ai2020
  parameter set), validated on measured discharges.
  - `Soc_estimation_comparison.py`: main script (measured or simulated 0.5C / 1C / 2C discharge).
  - `observers_ai.py`, `utils_ai.py`, `observer_config.json`: observers, plant, settings.
  - `CC_05.npy`, `CC_1.npy`, `CC_2.npy`: measured discharges (time, voltage, SOC ramp).
  - `tuning/`: calibration, NES tuning and Table 3 (see `tuning/README.md`).
  - `Saved_simulations/`: archived results of the original script.

## Quick start

```bash
pip install -r requirements.txt
cd "'Mohtat2020' Parameters set"
python SOC_estimation_comparison.py
```

To reproduce the observer settings and the comparison tables, run the steps in
`'Mohtat2020' Parameters set/tuning/README.md`.

## Environment

Tested with Python 3.12.7, numpy 1.26.4, scipy 1.13.1, matplotlib 3.10.8, pybamm 25.8.0,
filterpy 1.4.5, torch 2.6.0 (see `requirements.txt`).
