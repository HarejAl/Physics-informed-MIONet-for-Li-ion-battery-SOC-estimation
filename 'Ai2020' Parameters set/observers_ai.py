"""
SOC observers (Ai2020 parameter set, Enertech LCO-G cell)
---------------------------------------------------------

Same structure as `'Mohtat2020' Parameters set/observers.py`:

1. MIONet observer: MIONets retrained (transfer learning) on the Ai2020 SPM as process model,
   wrapped in a batched UKF (Julier sigma points, kappa = 0). Measurements:
     * terminal voltage  U = Up(c_pos,surf) - Un(c_neg,surf) - eta_n + eta_p - k_ohm * i
       (Enertech OCPs: lico2_ocp_Ai2020.csv / graphite_ocp_Enertech_Ai2020.csv)
     * lithium balance   sum_e L_e eps_e c_e,max <c_e> = const (particle-averaged pseudo-measurement)
   The surface concentration is clamped to [0.05, 0.95] c_max only inside j0.
   Process model: flux-consistent MIONet step (the MIONets give the profile shape, the particle
   averages follow the exact Coulomb-counted flux for the 34 parallel electrodes).
2. ECM observer: 3-RC ECM identified from a simulated HPPC test (`ecm_parameters_ai.mat`),
   OCV `soc_ocp_AI.npy`, FilterPy UKF, settings of the original Ai2020 script.
On measured data both observers use the same extra series resistance and voltage noise level,
calibrated on the 2C discharge (not part of Table 3).
3. Metrics: SOC RMSE (t >= 600 s) and CWC (mu = 0.95, gamma = eta = 1, whole run).

Settings of the MIONet observer: `observer_config.json` (written by tuning/).
"""

import json
import os
import pickle

import numpy as np
import pybamm
import scipy.io
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline

import utils_ai

HERE = os.path.dirname(os.path.abspath(__file__))
N = 31
DT = 2.0
IDX_SS = 300            # 600 s / DT
A_EFF = 0.08149800001137754  # effective electrode area [m2] (current density in the measurement model)

torch.set_num_threads(1)


def load_config(path=os.path.join(HERE, "observer_config.json")):
    return json.load(open(path))


class MIONet(nn.Module):

    def forward(self, coordinate, cs_0, current):
        return torch.sum(self.branch1(cs_0) * self.branch2(current) * self.trunk(coordinate), 1)


class _CheckpointPickle:
    load = staticmethod(pickle.load)

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and name == "MIONet":
                return MIONet
            return super().find_class(module, name)


def load_mionet(file):
    return torch.load(os.path.join(HERE, file), map_location="cpu", weights_only=False, pickle_module=_CheckpointPickle)


def _j0(E_r, c_e, c_s, c_max, T):
    """Exchange current density [A.m-2] (Ai2020), with the j0-only clamp."""
    m_ref = 1e-11 * 96485.33212
    arrhenius = np.exp(E_r / 8.314462618 * (1 / 298.15 - 1 / T))
    c_s = np.clip(c_s, 0.05 * c_max, 0.95 * c_max)
    return m_ref * arrhenius * c_e ** 0.5 * c_s ** 0.5 * (c_max - c_s) ** 0.5


def _spline(csv):
    _, (x, y) = pybamm.parameters.process_1D_data(os.path.join(HERE, csv))
    return CubicSpline(x[0], y)


class MIONetObserver:

    def __init__(self, T_amb=25, config=None, dT=DT):
        self.cfg = config or load_config()
        self.r_extra = 0.0  # extra series resistance of the real cell [Ohm] (measured data only)
        self.param = p = utils_ai.make_param(T_amb)
        self.dT = dT
        self.net_p, self.net_n = load_mionet("pos_saved_model_AI.pt"), load_mionet("neg_saved_model_AI.pt")
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
        L_n, L_p = p["Negative electrode thickness [m]"], p["Positive electrode thickness [m]"]
        self.jn_den = L_n * 3 * eps_n / p["Negative particle radius [m]"]
        self.jp_den = L_p * 3 * eps_p / p["Positive particle radius [m]"]
        self.li_p, self.li_n = L_p * eps_p * self.csp_max, L_n * eps_n * self.csn_max
        self.U_p, self.U_n = _spline("lico2_ocp_Ai2020.csv"), _spline("graphite_ocp_Enertech_Ai2020.csv")

        pl = utils_ai.plant(self.param, N)
        self.x0, self.x100, self.y0, self.y100 = pl.x0, pl.x100, pl.y0, pl.y100
        r = np.linspace(0, 1, N)[:, None]
        self.w_avg = (3 * (r - r[1] / 2) ** 2 * r[1])[:, 0]
        self.w_avg /= self.w_avg.sum()  # a uniform profile averages to itself

    def F(self, X, I):
        """
        One MIONet step for a batch of states X (B x 2N).
        Flux-consistent step: the MIONets give the shape of the radial profiles; each profile is
        then shifted so that its particle-averaged concentration changes exactly by the Coulomb-
        counted amount for the current density I / A_EFF (34 parallel electrodes). This is the
        boundary flux the MIONets were trained on, enforced exactly; total lithium is conserved.
        """
        Xt = torch.as_tensor(X, dtype=torch.float32)
        i = torch.tensor([[I * self.cfg.get("i_scale", 1.0)]], dtype=torch.float32)
        with torch.no_grad():
            cp, cn = Xt[:, :1], Xt[:, N:N + 1]
            bp = self.net_p.branch1(Xt[:, :N] - cp) * self.net_p.branch2(i)
            bn = self.net_n.branch1(Xt[:, N:] - cn) * self.net_n.branch2(i)
            out = torch.cat((bp @ self.trunk_p + cp, bn @ self.trunk_n + cn), 1)
        Xn = out.numpy().astype(np.float64)
        if self.cfg.get("flux_projection", True):
            dq = I * self.dT / (self.F_c * A_EFF)  # [mol.m-2] exchanged per unit electrode area in one step
            dp = (X[:, :N] @ self.w_avg + dq / self.li_p) - Xn[:, :N] @ self.w_avg  # discharge: positive lithiates
            dn = (X[:, N:] @ self.w_avg - dq / self.li_n) - Xn[:, N:] @ self.w_avg  # discharge: negative delithiates
            Xn[:, :N] += dp[:, None]; Xn[:, N:] += dn[:, None]
        return Xn

    def voltage(self, X, I):
        sp, sn = X[:, N - 1], X[:, -1]
        i_cell = I / A_EFF
        j0p = _j0(39570, self.ce, sp * self.csp_max, self.csp_max, self.T)
        j0n = _j0(5000, self.ce, sn * self.csn_max, self.csn_max, self.T)
        eta_n = 2 * self.R_gas * self.T / self.F_c * np.arcsinh(i_cell / self.jn_den / (2 * j0n))
        eta_p = 2 * self.R_gas * self.T / self.F_c * np.arcsinh(-i_cell / self.jp_den / (2 * j0p))
        return self.U_p(sp) - self.U_n(sn) - eta_n + eta_p - (self.cfg["k_ohm"] + self.r_extra) * I

    def lithium(self, X):
        return (X[:, :N] @ self.w_avg) * self.li_p + (X[:, N:] @ self.w_avg) * self.li_n

    def H(self, X, I):
        return np.stack((self.voltage(X, I), self.lithium(X)), 1)

    def soc(self, Y):
        pa, na = self.w_avg @ Y[:N], self.w_avg @ Y[N:]
        return 50 * ((pa - self.y0) / (self.y100 - self.y0) + (na - self.x0) / (self.x100 - self.x0))

    def run(self, V, I, soc_0=0.5, sigma_v=None, r_extra=0.0):
        """
        UKF over V, I (step dT). For measured data: sigma_v [V] replaces the voltage noise std of
        the configuration and r_extra [Ohm] is the extra series resistance of the real cell
        (both calibrated on the 2C discharge, see tuning/fit_measurement.py).
        """
        cfg = self.cfg
        self.r_extra = r_extra
        n = 2 * N
        Wc = np.full(2 * n + 1, 0.5 / n); Wc[0] = 0.0
        # uniform profiles at soc_0 on the cell's stoichiometry window (ElectrodeSOH): exact lithium total
        x = np.ones(n); x[:N] *= self.y0 + soc_0 * (self.y100 - self.y0); x[N:] *= self.x0 + soc_0 * (self.x100 - self.x0)
        # Covariances: SOC mode v (all nodes shifted together along the stoichiometry window, lithium
        # conserved) with variance P0_soc / q_soc [SOC fraction^2] + independent node terms P0 / q
        v = np.concatenate((np.full(N, self.y100 - self.y0), np.full(N, self.x100 - self.x0)))
        P = np.eye(n) * cfg["P0"] + cfg.get("P0_soc", 0.0) * np.outer(v, v)
        Q = np.eye(n) * cfg["q"] + cfg.get("q_soc", 0.0) * np.outer(v, v)
        Rm = np.diag([cfg["rV"] if sigma_v is None else sigma_v ** 2, cfg["rLi"]])
        I = np.broadcast_to(I, np.shape(V))
        Y, Pd, Vm = np.full((n, len(V)), np.nan), np.full((n, len(V)), np.nan), np.full(len(V), np.nan)
        sig_f = None
        for j in range(len(V)):
            if j > 0:
                sig_h = self.H(sig_f, I[j])
                zp = Wc @ sig_h
                dz, dx = sig_h - zp, sig_f - x
                S = (dz * Wc[:, None]).T @ dz + Rm
                K = ((dx * Wc[:, None]).T @ dz) @ np.linalg.inv(S)
                x = x + K @ (np.array([V[j], cfg["li_target"]]) - zp)
                P = P - K @ S @ K.T
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(P))):
                break
            Y[:, j], Pd[:, j] = x, np.diag(P)
            Vm[j] = self.voltage(x[None], I[j])[0]
            w, U = np.linalg.eigh((P + P.T) / 2)
            Ssq = (U * np.sqrt(np.clip(n * w, 0, None))) @ U.T
            sig_f = self.F(np.vstack((x, x + Ssq, x - Ssq)), I[j])
            x = Wc @ sig_f
            d = sig_f - x
            P = (d * Wc[:, None]).T @ d + Q
        conf = 2 * np.sqrt(np.clip(Pd, 0, None)); conf[N:] *= -1
        a, b = self.soc(Y + conf), self.soc(Y - conf)
        return dict(soc=self.soc(Y), soc_lower=np.minimum(a, b), soc_upper=np.maximum(a, b),
                    voltage=Vm, concentration=Y, P_diag=Pd)


class ECMObserver:
    """3-RC ECM + UKF (FilterPy), settings of the original Ai2020 script."""

    def __init__(self, C=utils_ai.C_RATE_CAPACITY, dT=DT, q_soc=0.1 * (0.5e-2) ** 2, q_V=0.1 * (1e-3) ** 2,
                 P0_V=1e-8, P0_soc=5e-2, r=(1e-3) ** 2, k_sigma=2.0):
        p = scipy.io.loadmat(os.path.join(HERE, "ecm_parameters_ai.mat"))
        self.R0, self.soc_grid = p["R0"].T[:, 0], p["SOC"].T[:, 0]
        self.tau, self.Rx = p["taux"].T, p["Rx"].T
        self.soc_ocv, self.Em = np.load(os.path.join(HERE, "soc_ocp_AI.npy"))
        self.Q_batt, self.dT = C * 3600, dT
        self.q_soc, self.q_V, self.P0_V, self.P0_soc, self.r, self.k_sigma = q_soc, q_V, P0_V, P0_soc, r, k_sigma
        self.r_extra = 0.0

    def f(self, x, dt, I):
        s = x[0]
        tau = [np.interp(s, self.soc_grid, self.tau[:, m]) for m in range(3)]
        Rx = [np.interp(s, self.soc_grid, self.Rx[:, m]) for m in range(3)]
        A = np.diag([1] + [np.exp(-self.dT / t) for t in tau])
        B = np.array([-self.dT / self.Q_batt] + [(1 - np.exp(-self.dT / t)) * R for t, R in zip(tau, Rx)])
        return A @ x + B * I

    def h(self, x, I):
        s = x[0]
        return np.array([np.interp(s, self.soc_ocv, self.Em) - (np.interp(s, self.soc_grid, self.R0) + self.r_extra) * I - x[1] - x[2] - x[3]])

    def run(self, V, I, soc_0=0.5, sigma_v=None, r_extra=0.0):
        """Same measured-data options as MIONetObserver.run (sigma_v, r_extra added to R0)."""
        self.r_extra = r_extra
        from filterpy.kalman import JulierSigmaPoints, UnscentedKalmanFilter
        ukf = UnscentedKalmanFilter(dim_x=4, dim_z=1, dt=self.dT, fx=self.f, hx=self.h, points=JulierSigmaPoints(4, kappa=0))
        ukf.x = np.array([soc_0, 0, 0, 0])
        ukf.P *= self.P0_V; ukf.P[0, 0] = self.P0_soc
        ukf.R *= self.r if sigma_v is None else sigma_v ** 2
        ukf.Q = np.diag([self.q_soc, self.q_V, self.q_V, self.q_V])
        I = np.broadcast_to(I, np.shape(V))
        soc, P, Vm = [], [], []
        for j in range(len(V)):
            ukf.update(np.array([V[j]]), I=I[j])
            soc.append(ukf.x[0] * 100); P.append(ukf.P[0, 0]); Vm.append(self.h(ukf.x, I[j])[0])
            ukf.predict(I=I[j])
        soc = np.array(soc); hw = self.k_sigma * np.sqrt(P) * 100
        return dict(soc=soc, soc_lower=soc - hw, soc_upper=soc + hw, voltage=np.array(Vm))


def cwc(y_true, y_lower, y_upper, mu=0.95, gamma=1, eta=1):
    picp = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    mpiw = np.mean(y_upper - y_lower)
    return mpiw * (1 + gamma * np.exp(-eta * (picp - mu))) if picp < mu else mpiw


def metrics(res, soc_ref, idx=IDX_SS):
    n = len(soc_ref)
    soc, lo, hi = res["soc"][:n], res["soc_lower"][:n], res["soc_upper"][:n]
    ok = np.all(np.isfinite(soc))
    return dict(rmse=float(np.sqrt(np.mean((soc - soc_ref)[idx:] ** 2))) if ok else np.inf,
                cwc=float(cwc(soc_ref, lo, hi)) if ok else np.inf,
                picp=float(np.mean((soc_ref >= lo) & (soc_ref <= hi))),
                mpiw=float(np.mean(hi - lo)) if ok else np.inf)
