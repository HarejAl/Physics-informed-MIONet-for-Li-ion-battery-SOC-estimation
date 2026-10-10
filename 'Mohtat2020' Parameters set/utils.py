"""
Plant utilities (Mohtat2020 parameter set)
------------------------------------------

The "plant" is the reference battery model solved with PyBaMM (DFN, optional lumped
thermal model). Its solution is treated as ground truth by the observers in `observers.py`.

- `make_param(T_amb)`  : Mohtat2020 parameter set at ambient temperature T_amb [°C]
- `plant(param, N)`    :
    * `UDDS(t_eval, CR)` : UDDS current profile (UDDS_profile.mat), scaled by CR
    * `CC(t_eval, CR)`   : constant current (CR < 0 is charge)
    * `SOC_test()`       : electrode stoichiometries at SOC = 0 / 100 % (ElectrodeSOHSolver)
    * `ref_solution()`   : solves the DFN for a given initial SOC and current profile
    * `SOC_calculator()` : cell SOC from the radial concentration profiles of both electrodes
"""

import os
import numpy as np
import pybamm
import scipy

pybamm.set_logging_level("ERROR")

HERE = os.path.dirname(os.path.abspath(__file__))


def make_param(T_amb=25):
    """Mohtat2020 parameter set with ambient and initial temperature T_amb [°C]."""
    param = pybamm.ParameterValues("Mohtat2020")
    param["Ambient temperature [K]"] = 273.15 + T_amb
    param["Initial temperature [K]"] = param["Ambient temperature [K]"]
    return param


class plant():

    def __init__(self, param, N):
        self.param = param
        self.C = param['Nominal cell capacity [A.h]']
        self.N = N
        x0, x100, y0, y100 = self.SOC_test()
        self.x0 = x0
        self.x100 = x100
        self.y0 = y0
        self.y100 = y100

    def UDDS(self, t_eval, CR):
        """UDDS profile (one cycle lasts ~1370 s), repeated to cover t_eval and scaled by CR."""
        UDDS = scipy.io.loadmat(os.path.join(HERE, 'UDDS_profile.mat'))
        c = UDDS['C'][:, 0]

        n_rep = int(t_eval[-1] // 1370 + 1)
        for i in range(n_rep - 1):
            c = np.concatenate((c, UDDS['C'][:, 0]), axis=0)

        t_aux = np.linspace(0, len(c) - 1, len(c))
        curr = np.interp(t_eval, t_aux, c)
        return curr * CR * self.C, 'UDDS'

    def CC(self, t_eval, CR):
        """Constant current: CR > 0 discharge, CR < 0 charge."""
        curr = np.ones((t_eval.shape)) * CR * self.C
        return curr, 'Constant current'

    def SOC_test(self):
        """Stoichiometries of Li in the electrodes at SOC = 0 % and SOC = 100 %."""
        Vmin = self.param["Lower voltage cut-off [V]"]
        Vmax = self.param["Upper voltage cut-off [V]"]
        self.param.set_initial_stoichiometries(1)

        parameter_values = pybamm.LithiumIonParameters()
        Q_n = self.param.evaluate(parameter_values.n.Q_init)
        Q_p = self.param.evaluate(parameter_values.p.Q_init)
        Q_Li = self.param.evaluate(parameter_values.Q_Li_particles_init)

        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(self.param, parameter_values)
        inputs = {"V_min": Vmin, "V_max": Vmax, "Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        esoh_sol = esoh_solver.solve(inputs)

        return esoh_sol["x_0"], esoh_sol["x_100"], esoh_sol["y_0"], esoh_sol["y_100"]

    def ref_solution(self, soc_0, current, t_eval, thermal_model=False):
        """
        Solves the DFN plant for initial SOC soc_0 and the current profile on t_eval [s].
        Returns: time [min], y_ref = [c_pos; c_neg] (2N x T, normalised), voltage [V], temperature [K].
        If the simulation stops at a voltage cut-off, the outputs are truncated there.
        """
        if thermal_model:
            batt_model = pybamm.lithium_ion.DFN(options={'thermal': 'lumped'})
        else:
            batt_model = pybamm.lithium_ion.DFN()

        self.param.set_initial_stoichiometries(soc_0)

        var_pts = {
            "x_n": 10,  # negative electrode
            "x_s": 10,  # separator
            "x_p": 10,  # positive electrode
            "r_n": self.N,  # negative particle
            "r_p": self.N,  # positive particle
        }

        experiment = pybamm.Experiment([pybamm.step.current(np.column_stack([t_eval, current]))])
        sim = pybamm.Simulation(batt_model, parameter_values=self.param, var_pts=var_pts, experiment=experiment)
        solution = sim.solve()

        t_eval = t_eval[t_eval <= solution["Time [s]"].entries[-1]]
        cs_pos_ref = np.mean(solution["Positive particle concentration"](t_eval), axis=1)
        cs_neg_ref = np.mean(solution["Negative particle concentration"](t_eval), axis=1)
        Temp = solution["X-averaged cell temperature [K]"](t_eval)
        voltage_ref = solution["Voltage [V]"](t_eval)

        y_ref = np.concatenate((cs_pos_ref, cs_neg_ref), axis=0)
        return t_eval / 60, y_ref, voltage_ref, Temp

    def SOC_calculator(self, y):
        """
        Cell SOC [%] as the average SOC of both electrodes.
        Input: normalised concentration profiles y = [c_pos; c_neg] (2N x T).
        """
        POS = y[:self.N]
        NEG = y[self.N:]

        # Average concentration in a sphere: 3 * INT[c(r) r^2 dr]_0^1
        r = np.linspace(0, 1, len(POS[:, 0]))[:, None]; dr = r[1]; dr_2 = dr / 2
        argument = np.ones(POS.shape) * (r - dr_2) ** 2 * dr

        pos_avg = np.sum(3 * argument * POS, axis=0)
        neg_avg = np.sum(3 * argument * NEG, axis=0)

        SOC_pos = (pos_avg - self.y0) / (self.y100 - self.y0)
        SOC_neg = (neg_avg - self.x0) / (self.x100 - self.x0)

        return (SOC_pos + SOC_neg) / 2 * 100
