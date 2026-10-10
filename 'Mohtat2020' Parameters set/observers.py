"""
SOC observers (Mohtat2020 parameter set)
----------------------------------------

1. MIONet observer: electrochemical observer whose process model is the pair of MIONets
   (one per electrode), wrapped in an Unscented Kalman Filter.
   - State: normalised radial concentration profiles x = [c_pos (N); c_neg (N)].
   - Process model: one MIONet step per electrode (all sigma points in one batched call).
   - Measurements:
       * terminal voltage  U = Up(c_pos,surf) - Un(c_neg,surf) - eta_n + eta_p - k_ohm * i
       * lithium balance   sum_e  L_e * eps_e * c_e,max * <c_e>  = const   (pseudo-measurement)
         with <c_e> the particle-averaged concentration ('avg') or the surface node ('surface').
   - In the Butler-Volmer exchange current density j0 ~ sqrt(c (c_max - c)), the surface
     concentration is clamped to [0.05, 0.95] * c_max ONLY inside j0, which sits in the
     denominator of the overpotential. The OCPs, the state and the sigma points are not clamped,
     so the sigma points still see the measurement model across the whole state range.
   - UKF: Julier sigma points (kappa = 0); FilterPy update/predict semantics.

2. ECM observer: 3-RC equivalent circuit (parameters identified from an HCCP test,
   `ECM_parameters_3.mat`) with FilterPy's UKF. Settings as in the paper.

3. Metrics: SOC RMSE (t >= 600 s) and the coverage width-based criterion (CWC).

The observer settings are stored in `observer_config.json`, written by `tuning/` (ohmic fit +
NES tuning of the UKF weights).
"""

import json
import os
import pickle

import numpy as np
import scipy.io
import scipy.linalg
import torch
import torch.nn as nn

import utils

HERE = os.path.dirname(os.path.abspath(__file__))
N = 31        # radial nodes per electrode (= sensor points of the saved MIONets)
DT = 2.0      # observer time step [s]
IDX_SS = 300  # 600 s / DT: the initial transient (unknown initial SOC) is excluded from the RMSE

torch.set_num_threads(1)


def load_config(path=os.path.join(HERE, "observer_config.json")):
    return json.load(open(path))


######################################################################################
################################### MIONet observer ##################################

class MIONet(nn.Module):

    def forward(self, coordinate, cs_0, current):
        out1 = self.branch1(cs_0)
        out2 = self.branch2(current)
        out_trunk = self.trunk(coordinate)
        return torch.sum(out1 * out2 * out_trunk, 1)


class _CheckpointPickle:
    """pickle module for torch.load: the checkpoints were saved from a __main__.MIONet class."""
    load = staticmethod(pickle.load)

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and name == "MIONet":
                return MIONet
            return super().find_class(module, name)


def load_mionet(file):
    return torch.load(os.path.join(HERE, file), map_location="cpu", weights_only=False, pickle_module=_CheckpointPickle)


def _j0(m_ref, E_r, c_e, c_s, c_max, T):
    """Exchange current density [A.m-2] (Mohtat2020), with the j0-only clamp."""
    arrhenius = np.exp(E_r / 8.314462618 * (1 / 298.15 - 1 / T))
    c_s = np.clip(c_s, 0.05 * c_max, 0.95 * c_max)
    return m_ref * arrhenius * c_e ** 0.5 * c_s ** 0.5 * (c_max - c_s) ** 0.5


class MIONetObserver:

    def __init__(self, T_amb=25, config=None, dT=DT):
        cfg = config or load_config()
        self.cfg = cfg
        self.param = p = utils.make_param(T_amb)
        self.dT = dT
        self.net_p, self.net_n = load_mionet("pos_saved_model.pt"), load_mionet("neg_saved_model.pt")
        # The saved models were trained with time scaled by 0.1; for a fixed dT the trunk is constant
        t_ksi = torch.cat((torch.full((N, 1), 0.1 * dT), torch.linspace(0, 1, N)[:, None]), 1)
        with torch.no_grad():
            self.trunk_p = self.net_p.trunk(t_ksi).T.contiguous()
            self.trunk_n = self.net_n.trunk(t_ksi).T.contiguous()

        self.R_gas = p["Ideal gas constant [J.K-1.mol-1]"]
        self.F_c = p["Faraday constant [C.mol-1]"]
        self.T = p["Ambient temperature [K]"]
        self.ce = p["Initial concentration in electrolyte [mol.m-3]"]
        self.csp_max = p["Maximum concentration in positive electrode [mol.m-3]"]
        self.csn_max = p["Maximum concentration in negative electrode [mol.m-3]"]
        eps_n = p["Negative electrode active material volume fraction"]
        eps_p = p["Positive electrode active material volume fraction"]
        a_n = 3 * eps_n / p["Negative particle radius [m]"]
        a_p = 3 * eps_p / p["Positive particle radius [m]"]
        L_n, L_p = p["Negative electrode thickness [m]"], p["Positive electrode thickness [m]"]
        WH = p["Electrode width [m]"] * p["Electrode height [m]"]
        self.jn_den, self.jp_den = L_n * a_n * WH, L_p * a_p * WH
        self.li_p, self.li_n = L_p * eps_p * self.csp_max, L_n * eps_n * self.csn_max
        self.U_p, self.U_n = p["Positive electrode OCP [V]"], p["Negative electrode OCP [V]"]

        pl = utils.plant(self.param, N)
        self.x0, self.x100, self.y0, self.y100 = pl.x0, pl.x100, pl.y0, pl.y100
        r = np.linspace(0, 1, N)[:, None]
        self.w_avg = (3 * (r - r[1] / 2) ** 2 * r[1])[:, 0]  # same quadrature as utils.plant.SOC_calculator

    # ---------------- process model ----------------
    def F(self, X, I):
        """One MIONet step for a batch of states X (B x 2N)."""
        Xt = torch.as_tensor(X, dtype=torch.float32)
        i = torch.tensor([[I]], dtype=torch.float32)
        with torch.no_grad():
            cp, cn = Xt[:, :1], Xt[:, N:N + 1]  # the MIONets predict the profile relative to the centre
            bp = self.net_p.branch1(Xt[:, :N] - cp) * self.net_p.branch2(i)
            bn = self.net_n.branch1(Xt[:, N:] - cn) * self.net_n.branch2(i)
            out = torch.cat((bp @ self.trunk_p + cp, bn @ self.trunk_n + cn), 1)
        return out.numpy().astype(np.float64)

    # ---------------- measurement model ----------------
    def voltage(self, X, I):
        sp, sn = X[:, N - 1], X[:, -1]  # surface nodes
        j0p = _j0(4.824e-6, 39570, self.ce, sp * self.csp_max, self.csp_max, self.T)
        j0n = _j0(1.061e-6, 37480, self.ce, sn * self.csn_max, self.csn_max, self.T)
        eta_n = 2 * self.R_gas * self.T / self.F_c * np.arcsinh(I / self.jn_den / (2 * j0n))
        eta_p = 2 * self.R_gas * self.T / self.F_c * np.arcsinh(-I / self.jp_den / (2 * j0p))
        return self.U_p(sp) - self.U_n(sn) - eta_n + eta_p - self.cfg["k_ohm"] * I

    def lithium(self, X):
        if self.cfg["li_mode"] == "avg":
            return (X[:, :N] @ self.w_avg) * self.li_p + (X[:, N:] @ self.w_avg) * self.li_n
        return X[:, N - 1] * self.li_p + X[:, -1] * self.li_n  # surface nodes

    def H(self, X, I):
        U = self.voltage(X, I)
        if self.cfg["li_mode"] == "none":
            return U[:, None]
        return np.stack((U, self.lithium(X)), 1)

    def soc(self, Y):
        """Cell SOC [%] from Y (2N x T), as utils.plant.SOC_calculator."""
        pa, na = self.w_avg @ Y[:N], self.w_avg @ Y[N:]
        return 50 * ((pa - self.y0) / (self.y100 - self.y0) + (na - self.x0) / (self.x100 - self.x0))

    # ---------------- UKF ----------------
    def run(self, V, I, soc_0=0.5):
        """
        UKF over the measured voltage V and current I (same length, step dT).
        Initial state: uniform profiles at soc_0 (the true initial SOC is unknown).
        Returns dict: soc, soc_lower, soc_upper (+-2 sigma), voltage, concentration, P_diag.
        """
        cfg = self.cfg
        n = 2 * N
        Wc = np.full(2 * n + 1, 0.5 / n); Wc[0] = 0.0  # Julier sigma points, kappa = 0
        x = np.ones(n); x[:N] *= (1 - soc_0) * 0.89; x[N:] *= soc_0 * 0.83
        P = np.eye(n) * cfg["P0"]
        Q = np.eye(n) * cfg["q"]
        use_li = cfg["li_mode"] != "none"
        Rm = np.diag([cfg["rV"], cfg["rLi"]]) if use_li else np.array([[cfg["rV"]]])
        Y, Pd, Vm = np.full((n, len(V)), np.nan), np.full((n, len(V)), np.nan), np.full(len(V), np.nan)
        sig_f = None

        for j in range(len(V)):
            if j > 0:  # update (uses the sigma points of the last predict, as FilterPy)
                sig_h = self.H(sig_f, I[j])
                zp = Wc @ sig_h
                dz, dx = sig_h - zp, sig_f - x
                S = (dz * Wc[:, None]).T @ dz + Rm
                K = ((dx * Wc[:, None]).T @ dz) @ np.linalg.inv(S)
                z = np.array([V[j], cfg["li_target"]]) if use_li else np.array([V[j]])
                x = x + K @ (z - zp)
                P = P - K @ S @ K.T
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(P))):
                break  # diverged: the remaining samples stay NaN
            Y[:, j], Pd[:, j] = x, np.diag(P)
            Vm[j] = self.voltage(x[None], I[j])[0]
            # predict (symmetric square root of n*P, projected onto the PSD cone)
            w, U = np.linalg.eigh((P + P.T) / 2)
            Ssq = (U * np.sqrt(np.clip(n * w, 0, None))) @ U.T
            sig_f = self.F(np.vstack((x, x + Ssq, x - Ssq)), I[j])
            x = Wc @ sig_f
            d = sig_f - x
            P = (d * Wc[:, None]).T @ d + Q

        conf = 2 * np.sqrt(np.clip(Pd, 0, None)); conf[N:] *= -1  # a higher c_pos / lower c_neg is a lower SOC
        a, b = self.soc(Y + conf), self.soc(Y - conf)
        return dict(soc=self.soc(Y), soc_lower=np.minimum(a, b), soc_upper=np.maximum(a, b),
                    voltage=Vm, concentration=Y, P_diag=Pd)


######################################################################################
#################################### ECM observer ####################################

class ECMObserver:
    """3-RC ECM + UKF (FilterPy), settings of the paper."""

    def __init__(self, C=5.0, dT=DT, file="ECM_parameters_3.mat",
                 q_soc=(1e-3) ** 2, q_V=(1e-4) ** 2, P0_V=1e-6, P0_soc=5e-2, r=(1e-3) ** 2, k_sigma=1.96):
        p = scipy.io.loadmat(os.path.join(HERE, file))
        self.R0, self.Em, self.soc_grid = p["R0"].T[:, 0], p["Em"].T[:, 0], p["SOC"].T[:, 0]
        self.tau, self.Rx = p["taux"].T, p["Rx"].T
        self.Q_batt, self.dT = C * 3600, dT
        self.q_soc, self.q_V, self.P0_V, self.P0_soc, self.r, self.k_sigma = q_soc, q_V, P0_V, P0_soc, r, k_sigma

    def f(self, x, dt, I):
        s = x[0]
        tau = [np.interp(s, self.soc_grid, self.tau[:, m]) for m in range(3)]
        Rx = [np.interp(s, self.soc_grid, self.Rx[:, m]) for m in range(3)]
        A = np.diag([1] + [np.exp(-self.dT / t) for t in tau])
        B = np.array([-self.dT / self.Q_batt] + [(1 - np.exp(-self.dT / t)) * R for t, R in zip(tau, Rx)])
        return A @ x + B * I

    def h(self, x, I):
        s = x[0]
        return np.array([np.interp(s, self.soc_grid, self.Em) - np.interp(s, self.soc_grid, self.R0) * I - x[1] - x[2] - x[3]])

    def run(self, V, I, soc_0=0.5):
        from filterpy.kalman import JulierSigmaPoints, UnscentedKalmanFilter
        ukf = UnscentedKalmanFilter(dim_x=4, dim_z=1, dt=1, fx=self.f, hx=self.h, points=JulierSigmaPoints(4, kappa=0))
        ukf.x = np.array([soc_0, 0, 0, 0])
        ukf.P *= self.P0_V; ukf.P[0, 0] = self.P0_soc
        ukf.R *= self.r
        ukf.Q = np.diag([self.q_soc, self.q_V, self.q_V, self.q_V])
        soc, P, Vm = [], [], []
        for j in range(len(V)):
            if j > 0:
                ukf.update(np.array([V[j]]), I=I[j])
            soc.append(ukf.x[0] * 100); P.append(ukf.P[0, 0]); Vm.append(self.h(ukf.x, I[j])[0])
            ukf.predict(I=I[j])
        soc = np.array(soc); hw = self.k_sigma * np.sqrt(P) * 100
        return dict(soc=soc, soc_lower=soc - hw, soc_upper=soc + hw, voltage=np.array(Vm))


######################################################################################
###################################### Metrics #######################################

def cwc(y_true, y_lower, y_upper, mu=0.95, gamma=1, eta=1):
    """Coverage width-based criterion (as in Saved_simulations/plotting.py)."""
    picp = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    mpiw = np.mean(y_upper - y_lower)
    return mpiw * (1 + gamma * np.exp(-eta * (picp - mu))) if picp < mu else mpiw


def metrics(res, soc_ref, idx=IDX_SS):
    """RMSE [%] for t >= 600 s; CWC, PICP and MPIW over the whole run (paper convention)."""
    n = len(soc_ref)
    soc, lo, hi = res["soc"][:n], res["soc_lower"][:n], res["soc_upper"][:n]
    e = (soc - soc_ref)[idx:]
    ok = np.all(np.isfinite(soc))
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))) if ok else np.inf,
                cwc=float(cwc(soc_ref, lo, hi)) if ok else np.inf,
                picp=float(np.mean((soc_ref >= lo) & (soc_ref <= hi))),
                mpiw=float(np.mean(hi - lo)) if ok else np.inf)
