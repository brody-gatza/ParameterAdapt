"""Tests for time_integration: FDF, SSPRK2, SSPRK3, and BDF."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "time_integration" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FDF = _load_module("FDF", "FDF.py")
SSPRK2 = _load_module("SSPRK2", "SSPRK2.py")
SSPRK3 = _load_module("SSPRK3", "SSPRK3.py")
BDF = _load_module("BDF", "BDF.py")


def _fom_solver_state(cell_number: int, num_state_var: int, dt: float, d_flux_fill: float = 0.5):
    n = (cell_number + 4) * num_state_var
    solver_param = {"cell_number": cell_number, "num_state_var": num_state_var, "dt": dt}
    state = {
        "Q_cons": np.arange(n, dtype=float),
        "d_flux_dx": np.ones(n, dtype=float) * d_flux_fill,
    }
    return solver_param, state


# --- FDF ---


def test_fdf_explicit_euler():
    """FDF: Q_new = Q_old + dt * d_flux_dx."""
    dt = 0.05
    solver_param = {"dt": dt}
    Q_old = np.array([1.0, 2.0, -0.5])
    dQ_dt = np.array([0.1, -0.2, 0.0])
    state = {"Q_cons": Q_old.copy(), "d_flux_dx": dQ_dt.copy()}

    out = FDF.advance_time(solver_param, {}, state, physics=None)

    expected = Q_old + dt * dQ_dt
    np.testing.assert_allclose(out["Q_cons"], expected)


# --- SSPRK2 ---


def test_ssprk2_fom_heun_formula():
    """SSPRK2 FOM: q1 = Q + dt*k0, q2 = q1 + dt*k1, Q_new = 0.5*(q1+q2)."""
    dt = 0.1
    solver_param, state = _fom_solver_state(cell_number=4, num_state_var=2, dt=dt)
    Q_old = state["Q_cons"].copy()
    k0 = state["d_flux_dx"]
    k1_value = 2.0

    def residual_calculator(sp, rp, st):
        st["d_flux_dx"] = np.full_like(st["Q_cons"], k1_value)
        return st

    physics = MagicMock()
    physics.residual_calculator.side_effect = residual_calculator

    out = SSPRK2.advance_time(solver_param, {}, state, physics=physics)

    q1 = Q_old + dt * k0
    q2 = q1 + dt * k1_value
    expected = 0.5 * (q1 + q2)
    np.testing.assert_allclose(out["Q_cons"], expected)
    physics.residual_calculator.assert_called_once()


def test_ssprk2_rom_hyper_reduced():
    """SSPRK2 ROM: Heun on sampled DOFs via cons_results_save / S_indx_solver."""
    cell_number = 4
    num_state_var = 2
    dt = 0.1
    len_full = (cell_number + 4) * num_state_var

    S_indx_solver = np.array([0, 1, 2, 3], dtype=int)
    n_red = len(S_indx_solver)
    solver_param = {"cell_number": cell_number, "num_state_var": num_state_var, "dt": dt}
    cons_results_save = np.zeros((num_state_var, cell_number), dtype=float)
    rom_param = {"S_indx_solver": S_indx_solver}

    Q_old = np.linspace(0.0, 1.0, n_red)
    dQ_dt = np.ones(n_red) * 0.25
    state = {
        "Q_cons": Q_old.copy(),
        "d_flux_dx": dQ_dt.copy(),
        "cons_results_save": cons_results_save,
    }
    assert len(state["Q_cons"]) != len_full

    k1 = np.full(n_red, 1.5)

    def residual_calculator(sp, rp, st):
        assert st["Q_cons"].shape == (len_full,)
        st["d_flux_dx"] = k1.copy()
        return st

    physics = MagicMock()
    physics.residual_calculator.side_effect = residual_calculator

    out = SSPRK2.advance_time(solver_param, rom_param, state, physics=physics)

    q1 = Q_old + dt * dQ_dt
    q2 = q1 + dt * k1
    expected = 0.5 * (q1 + q2)
    np.testing.assert_allclose(out["Q_cons"], expected)
    physics.residual_calculator.assert_called_once()


# --- SSPRK3 ---


def test_ssprk3_fom_three_stage():
    """SSPRK3 FOM: three stages + final combination with mocked k1, k2, k3."""
    dt = 0.1
    solver_param, state = _fom_solver_state(cell_number=4, num_state_var=2, dt=dt)
    Q_old = state["Q_cons"].copy()
    k0 = state["d_flux_dx"]
    k1_val, k2_val, k3_val = 1.0, -0.5, 2.0

    stage = [0]

    def residual_calculator(sp, rp, st):
        stage[0] += 1
        vals = (k1_val, k2_val, k3_val)
        st["d_flux_dx"] = np.full_like(st["Q_cons"], vals[stage[0] - 1])
        return st

    physics = MagicMock()
    physics.residual_calculator.side_effect = residual_calculator

    out = SSPRK3.advance_time(solver_param, {}, state, physics=physics)

    q1 = Q_old + 0.5 * dt * k0
    q2 = q1 + 0.5 * dt * k1_val
    q3 = (2.0 * Q_old / 3.0) + (1.0 / 3.0) * (q2 + 0.5 * dt * k2_val)
    expected = q3 + 0.5 * dt * k3_val
    np.testing.assert_allclose(out["Q_cons"], expected)
    assert physics.residual_calculator.call_count == 3


def test_ssprk3_rom_hyper_reduced():
    """SSPRK3 ROM: three residual evaluations on reduced DOFs."""
    cell_number = 4
    num_state_var = 2
    dt = 0.1
    len_full = (cell_number + 4) * num_state_var

    S_indx_solver = np.array([0, 1, 2, 3], dtype=int)
    n_red = len(S_indx_solver)
    solver_param = {"cell_number": cell_number, "num_state_var": num_state_var, "dt": dt}
    cons_results_save = np.zeros((num_state_var, cell_number), dtype=float)
    rom_param = {"S_indx_solver": S_indx_solver}

    Q_old = np.linspace(0.0, 1.0, n_red)
    dQ_dt = np.ones(n_red) * 0.25
    state = {
        "Q_cons": Q_old.copy(),
        "d_flux_dx": dQ_dt.copy(),
        "cons_results_save": cons_results_save,
    }
    assert len(state["Q_cons"]) != len_full

    k1_arr = np.full(n_red, 1.0)
    k2_arr = np.full(n_red, -0.5)
    k3_arr = np.full(n_red, 2.0)
    stage = [0]
    arrs = (k1_arr, k2_arr, k3_arr)

    def residual_calculator(sp, rp, st):
        assert st["Q_cons"].shape == (len_full,)
        stage[0] += 1
        st["d_flux_dx"] = arrs[stage[0] - 1].copy()
        return st

    physics = MagicMock()
    physics.residual_calculator.side_effect = residual_calculator

    out = SSPRK3.advance_time(solver_param, rom_param, state, physics=physics)

    q1 = Q_old + dt * dQ_dt
    q2 = q1 + dt * k1_arr
    q3 = (2.0 * Q_old / 3.0) + (1.0 / 3.0) * (q2 + 0.5 * dt * k2_arr)
    expected = q3 + 0.5 * dt * k3_arr
    np.testing.assert_allclose(out["Q_cons"], expected)
    assert physics.residual_calculator.call_count == 3


# --- BDF ---


def test_bdf_newton_constant_flux():
    """BDF advance_time: constant flux gives Q_{n+1} = Q_n + dt*k."""
    dt = 0.1
    solver_param, state = _fom_solver_state(cell_number=4, num_state_var=2, dt=dt, d_flux_fill=0.25)
    Q_n = state["Q_cons"].copy()
    k = np.ones_like(Q_n) * 1.25

    def residual_calculator(sp, rp, st):
        st["d_flux_dx"] = k.copy()
        return st

    physics = MagicMock()
    physics.residual_calculator.side_effect = residual_calculator

    out = BDF.advance_time(solver_param, {}, state, physics)

    expected = Q_n + dt * k
    np.testing.assert_allclose(out["Q_cons"], expected, rtol=1e-5, atol=1e-7)
    physics.residual_calculator.assert_called()


def test_bdf_implicit_residual_linear():
    """implicit_bdf_residual_calculator vanishes at Q_n + dt * flux for constant flux."""
    dt = 0.1
    solver_param = {"cell_number": 4, "num_state_var": 2, "dt": dt}
    n = (4 + 4) * 2
    Q_n = np.linspace(0.0, 1.0, n)
    k = np.ones(n) * 0.5
    state = {"Q_cons": Q_n.copy(), "d_flux_dx": k.copy()}

    def residual_calculator(sp, rp, st):
        st["d_flux_dx"] = k.copy()
        return st

    physics = MagicMock()
    physics.residual_calculator.side_effect = residual_calculator

    q_sol = Q_n + dt * k
    rom_param = {}
    res = BDF.implicit_bdf_residual_calculator(
        q_sol, Q_n, solver_param, rom_param, state, physics
    )
    np.testing.assert_allclose(res, 0.0, atol=1e-10)
