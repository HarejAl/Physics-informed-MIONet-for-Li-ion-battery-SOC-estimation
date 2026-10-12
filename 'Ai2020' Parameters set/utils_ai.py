"""
Plant utilities (Ai2020 parameter set, Enertech LCO-G cell)
------------------------------------------------------------

- `make_param(T_amb)`  : Ai2020 parameter set at ambient temperature T_amb [°C]
- `C_RATE_CAPACITY`    : capacity used to convert C-rates into currents (nominal x 1.05, as in the
                         PyBaMM Ai2020 experimental comparison)
- `plant(param, N)`    : current profiles, DFN reference solution, SOC calculation
- `load_experiment(CR)`: measured constant-current discharge (0.5C, 1C, 2C)
"""

import os
import numpy as np
import pybamm
import scipy

pybamm.set_logging_level("ERROR")

HERE = os.path.dirname(os.path.abspath(__file__))
C_RATE_CAPACITY = 2.28 * 1.05  # [Ah]
EXPERIMENTS = {0.5: "CC_05.npy", 1: "CC_1.npy", 2: "CC_2.npy"}


def make_param(T_amb=25):
    param = pybamm.ParameterValues("Ai2020")
    param["Ambient temperature [K]"] = 273.15 + T_amb
    param["Initial temperature [K]"] = param["Ambient temperature [K]"]
    return param


def load_experiment(CR, dT=2):
    """Measured discharge at CR from fully charged: time [s], voltage [V], SOC reference [%] (linear ramp)."""
    d = np.load(os.path.join(HERE, EXPERIMENTS[CR]))[::int(dT)]
    return d[:, 0], d[:, 1], d[:, 2]


class plant():

    def __init__(self, param, N):
        self.param = param
        self.C = C_RATE_CAPACITY
        self.N = N
        self.x0, self.x100, self.y0, self.y100 = self.SOC_test()

    def UDDS(self, t_eval, CR):
        """UDDS profile (Mohtat2020 folder), repeated to cover t_eval and scaled by CR."""
        UDDS = scipy.io.loadmat(os.path.join(os.path.dirname(HERE), "'Mohtat2020' Parameters set", 'UDDS_profile.mat'))
        c = UDDS['C'][:, 0]
        for i in range(int(t_eval[-1] // 1370)):
            c = np.concatenate((c, UDDS['C'][:, 0]), axis=0)
        curr = np.interp(t_eval, np.arange(len(c)), c)
        return curr * CR * self.C, 'UDDS'

    def CC(self, t_eval, CR):
        """Constant current: CR > 0 discharge, CR < 0 charge."""
        return np.ones(t_eval.shape) * CR * self.C, 'Constant current'

    def SOC_test(self):
        """Stoichiometries of Li in the electrodes at SOC = 0 % and SOC = 100 %."""
        Vmin = self.param["Lower voltage cut-off [V]"]
        Vmax = self.param["Upper voltage cut-off [V]"]
        self.param.set_initial_stoichiometries(1)
        pv = pybamm.LithiumIonParameters()
        inputs = {"V_min": Vmin, "V_max": Vmax, "Q_n": self.param.evaluate(pv.n.Q_init),
                  "Q_p": self.param.evaluate(pv.p.Q_init), "Q_Li": self.param.evaluate(pv.Q_Li_particles_init)}
        sol = pybamm.lithium_ion.ElectrodeSOHSolver(self.param, pv).solve(inputs)
        return sol["x_0"], sol["x_100"], sol["y_0"], sol["y_100"]

    def ref_solution(self, soc_0, current, t_eval, thermal_model=False):
        """DFN plant; returns time [min], y_ref (2N x T), voltage [V], temperature [K], truncated at a cut-off."""
        batt_model = pybamm.lithium_ion.DFN(options={'thermal': 'lumped'}) if thermal_model else pybamm.lithium_ion.DFN()
        self.param.set_initial_stoichiometries(soc_0)
        var_pts = {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": self.N, "r_p": self.N}
        experiment = pybamm.Experiment([pybamm.step.current(np.column_stack([t_eval, current]))])
        solution = pybamm.Simulation(batt_model, parameter_values=self.param, var_pts=var_pts, experiment=experiment).solve()
        t_eval = t_eval[t_eval <= solution["Time [s]"].entries[-1]]
        y_ref = np.concatenate((np.mean(solution["Positive particle concentration"](t_eval), axis=1),
                                np.mean(solution["Negative particle concentration"](t_eval), axis=1)), axis=0)
        return t_eval / 60, y_ref, solution["Voltage [V]"](t_eval), solution["X-averaged cell temperature [K]"](t_eval)

    def SOC_calculator(self, y):
        """Cell SOC [%] as the average SOC of both electrodes; y = [c_pos; c_neg] (2N x T)."""
        POS, NEG = y[:self.N], y[self.N:]
        r = np.linspace(0, 1, len(POS[:, 0]))[:, None]; dr = r[1]
        w = 3 * (r - dr / 2) ** 2 * dr
        SOC_pos = (np.sum(w * POS, axis=0) - self.y0) / (self.y100 - self.y0)
        SOC_neg = (np.sum(w * NEG, axis=0) - self.x0) / (self.x100 - self.x0)
        return (SOC_pos + SOC_neg) / 2 * 100
