Physics-Informed Operator Learning for Real-Time Battery State Estimation
This repository contains the code and trained models used in the work "Physics-Informed Operator Learning for Real-Time Battery State Estimation".

Repository structure
training.py – Code to train the MIONets to solve the SPM PDEs.

neg_saved_model.pt / pos_saved_model.pt – Trained MIONets for the Mohtat2020 parameter set (negative and positive electrodes, respectively).

neg_saved_model_tl.pt / pos_saved_model_tl.pt – Trained MIONets (via transfer learning) for the Prada2013 parameter set.

SOC_estimation.py – Code for online State of Charge (SOC) estimation using the MIONet-based observer.

utils.py – Utility library containing methods used by SOC_estimation.py.

testing_MIONet.py – Code for predicting lithium concentrations in both electrodes without the observer (direct MIONet predictions). Useful for testing the trained models independently.
