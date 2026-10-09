"""
Batched re-implementation of the public MIONet + UKF observer
-------------------------------------------------------------

Same filter as `SOC_estimation_comparison.py` (FilterPy UnscentedKalmanFilter,
Julier sigma points with kappa = 0, scipy sqrtm), but all sigma points are pushed
through the MIONets and the measurement model in a single batched call. This
makes one 6 h run take minutes instead of hours, which is what makes the NES
tuning in `tune_nes.py` feasible.

Differences w.r.t. the public script are only exposed as options:
- `k_ohm`        : coefficient of the ohmic correction term  U -= k_ohm * i  (public: 1/600)
- `use_li`       : include the lithium-balance pseudo-measurement (public: True)
- `P0, q, rV, rLi`: UKF weights (public: 5e-3, (1e-4)^2, (20*1e-3)^2, 1e-3)
- `sqrt`         : 'sqrtm' (identical to the public run) or 'eigh' (PSD-projected,
                   avoids the NaN failures seen at 1C)
"""

import os
import pickle
import sys
import numpy as np
import scipy.linalg
import torch
import pybamm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # 'Mohtat2020' Parameters set
sys.path.insert(0, ROOT)
import utils  # noqa: E402

pybamm.set_logging_level("ERROR")
torch.set_num_threads(1)

N = 31
DT = 2.0
LI_TARGET = 0.9454372727901035  # lithium-balance target used in the public script (surface nodes)
LI_TARGET_AVG = 0.94116  # particle-averaged lithium balance, mean over the training trajectories (spread < 1e-4)
PUBLIC = dict(P0=5e-3, q=(1e-4) ** 2, rV=(20e-3) ** 2, rLi=1e-3, k_ohm=1 / 600)


class _CheckpointPickle:
    """pickle module for torch.load: the checkpoints reference __main__.MIONet, which does not
    exist in multiprocessing workers, so map it explicitly to utils.MIONet."""
    load = staticmethod(pickle.load)

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and name == "MIONet":
                return utils.MIONet
            return super().find_class(module, name)


def make_param(T_amb=25):
    param = pybamm.ParameterValues("Mohtat2020")
    param["Ambient temperature [K]"] = 273.12 + T_amb  # same convention as the public script
    param["Initial temperature [K]"] = param["Ambient temperature [K]"]
    return param


class Observer:
    """Batched MIONet process model + public measurement model."""

    def __init__(self, T_amb=25, dT=DT):
        self.param = make_param(T_amb)
        p = self.param
        self.dT = dT
        load = lambda f: torch.load(os.path.join(ROOT, f), map_location="cpu", weights_only=False,
                                    pickle_module=_CheckpointPickle)
        self.net_p, self.net_n = load("pos_saved_model.pt"), load("neg_saved_model.pt")
        # Trunk input is constant for a fixed dT: precompute it once
        t_ksi = torch.cat((torch.full((N, 1), 0.1 * dT), torch.linspace(0, 1, N)[:, None]), 1)
        with torch.no_grad():
            self.trunk_p = self.net_p.trunk(t_ksi).T.contiguous()
            self.trunk_n = self.net_n.trunk(t_ksi).T.contiguous()

        # Measurement model constants (as in utils.observer_mionet.Voltage)
        self.R = p["Ideal gas constant [J.K-1.mol-1]"]
        self.F = p["Faraday constant [C.mol-1]"]
        self.T = p["Ambient temperature [K]"]
        self.ce = p["Initial concentration in electrolyte [mol.m-3]"]
        self.csp_max = p["Maximum concentration in positive electrode [mol.m-3]"]
        self.csn_max = p["Maximum concentration in negative electrode [mol.m-3]"]
        eps_n, eps_p = p["Negative electrode active material volume fraction"], p["Positive electrode active material volume fraction"]
        a_n = 3 * eps_n / p["Negative particle radius [m]"]
        a_p = 3 * eps_p / p["Positive particle radius [m]"]
        self.L_n, self.L_p = p["Negative electrode thickness [m]"], p["Positive electrode thickness [m]"]
        WH = p["Electrode width [m]"] * p["Electrode height [m]"]
        self.jn_den = self.L_n * a_n * WH
        self.jp_den = self.L_p * a_p * WH
        self.li_p = self.L_p * eps_p * self.csp_max
        self.li_n = self.L_n * eps_n * self.csn_max
        self.U_p, self.U_n = p["Positive electrode OCP [V]"], p["Negative electrode OCP [V]"]

        pl = utils.plant(self.param, N)
        self.x0, self.x100, self.y0, self.y100 = pl.x0, pl.x100, pl.y0, pl.y100
        r = np.linspace(0, 1, N)[:, None]
        self.w_avg = 3 * (r - r[1] / 2) ** 2 * r[1]  # quadrature of utils.SOC_calculator

    # ---------------- process model ----------------
    def F_batch(self, X, I):
        """X: (B, 2N) -> (B, 2N)"""
        Xt = torch.as_tensor(X, dtype=torch.float32)
        i = torch.tensor([[I]], dtype=torch.float32)
        with torch.no_grad():
            cp = Xt[:, :1]; cn = Xt[:, N:N + 1]
            bp = self.net_p.branch1(Xt[:, :N] - cp) * self.net_p.branch2(i)
            bn = self.net_n.branch1(Xt[:, N:] - cn) * self.net_n.branch2(i)
            out = torch.cat((bp @ self.trunk_p + cp, bn @ self.trunk_n + cn), 1)
        return out.numpy().astype(np.float64)

    # ---------------- measurement model ----------------
    @staticmethod
    def _j0(m_ref, E_r, c_e, c_s, c_max, T, clamp):
        arr = np.exp(E_r / 8.314462618 * (1 / 298.15 - 1 / T))
        c_s = np.clip(c_s, 0.05 * c_max, 0.95 * c_max) if clamp else np.minimum(c_s, c_max)
        return m_ref * arr * c_e ** 0.5 * c_s ** 0.5 * (c_max - c_s) ** 0.5

    def voltage(self, X, I, k_ohm, clamp=False):
        sp, sn = X[:, N - 1], X[:, -1]
        j0p = self._j0(4.824e-6, 39570, self.ce, sp * self.csp_max, self.csp_max, self.T, clamp)
        j0n = self._j0(1.061e-6, 37480, self.ce, sn * self.csn_max, self.csn_max, self.T, clamp)
        with np.errstate(invalid="ignore", divide="ignore"):
            eta_n = 2 * self.R * self.T / self.F * np.arcsinh(I / self.jn_den / (2 * j0n))
            eta_p = 2 * self.R * self.T / self.F * np.arcsinh(-I / self.jp_den / (2 * j0p))
        return self.U_p(sp) - self.U_n(sn) - eta_n + eta_p - k_ohm * I

    def H_batch(self, X, I, k_ohm, use_li, clamp=False):
        """use_li: False | True / 'surface' (public, surface nodes) | 'avg' (particle-averaged)"""
        U = self.voltage(X, I, k_ohm, clamp)
        if not use_li:
            return U[:, None]
        if use_li == "avg":
            w = self.w_avg[:, 0]
            li = (X[:, :N] @ w) * self.li_p + (X[:, N:] @ w) * self.li_n
        else:
            li = X[:, N - 1] * self.li_p + X[:, -1] * self.li_n
        return np.stack((U, li), 1)

    def soc(self, Y):
        """Y: (2N, T) -> SOC [%] (same as utils.plant.SOC_calculator)"""
        pa = np.sum(self.w_avg * Y[:N], 0); na = np.sum(self.w_avg * Y[N:], 0)
        return 50 * ((pa - self.y0) / (self.y100 - self.y0) + (na - self.x0) / (self.x100 - self.x0))


def run_ukf(obs, V, I, P0, q, rV, rLi, k_ohm, use_li=True, sqrt="sqrtm", clamp=False, q_profile=None):
    """Runs the public update/log/predict loop. Returns (Y, Pdiag) with shape (2N, T)."""
    n = 2 * N
    Wc = np.full(2 * n + 1, 0.5 / n); Wc[0] = 0.0  # Julier, kappa = 0
    x = np.ones(n); x[:N] *= 0.5 * 0.89; x[N:] *= 0.5 * 0.83
    P = np.eye(n) * P0
    Q = np.eye(n) * q if q_profile is None else np.diag(np.concatenate([q_profile, q_profile])) * q
    Rm = np.diag([rV, rLi]) if use_li else np.array([[rV]])
    z_li = LI_TARGET_AVG if use_li == "avg" else LI_TARGET
    Y = np.empty((n, len(V))); Pd = np.empty((n, len(V)))
    sig_f = None

    def sqrt_mat(A):
        if sqrt == "sqrtm":
            S = scipy.linalg.sqrtm(A)
            return np.real(S)
        w, U = np.linalg.eigh((A + A.T) / 2)
        return (U * np.sqrt(np.clip(w, 0, None))) @ U.T

    for j in range(len(V)):
        if j > 0:  # update with sigmas from the last predict (FilterPy semantics)
            sig_h = obs.H_batch(sig_f.copy(), I[j], k_ohm, use_li, clamp)
            zp = Wc @ sig_h
            dz = sig_h - zp; dx = sig_f - x
            S = (dz * Wc[:, None]).T @ dz + Rm
            Pxz = (dx * Wc[:, None]).T @ dz
            K = Pxz @ np.linalg.inv(S)
            z = np.array([V[j], z_li]) if use_li else np.array([V[j]])
            x = x + K @ (z - zp)
            P = P - K @ S @ K.T
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(P)):
            Y[:, j:] = np.nan; Pd[:, j:] = np.nan
            return Y, Pd
        Y[:, j] = x; Pd[:, j] = np.diag(P)
        # predict
        U = sqrt_mat(n * P)
        sig = np.vstack((x, x + U, x - U))
        sig_f = obs.F_batch(sig, I[j])
        x = Wc @ sig_f
        d = sig_f - x
        P = (d * Wc[:, None]).T @ d + Q
    return Y, Pd


def metrics(obs, Y, Pd, soc_ref, idx=300):
    """SOC RMSE after 600 s, 2-sigma band (public convention) and its coverage."""
    soc = obs.soc(Y)
    conf = 2 * np.sqrt(np.clip(Pd, 0, None)); conf[N:] *= -1
    a, b = obs.soc(Y + conf), obs.soc(Y - conf)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    e = (soc - soc_ref)[idx:]
    rmse = np.sqrt(np.mean(e ** 2)) if np.all(np.isfinite(e)) else np.inf
    cover = np.mean((soc_ref[idx:] >= lo[idx:]) & (soc_ref[idx:] <= hi[idx:]))
    return dict(rmse=rmse, halfwidth=np.median((hi - lo)[idx:]) / 2, coverage=cover, soc=soc, lo=lo, hi=hi)
