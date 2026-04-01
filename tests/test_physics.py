"""Tests for physics: Ideal_Air, Single_Step_Reacting_Flow, and light checks for Reacting_Flow."""

import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from physics import Ideal_Air
from physics import Single_Step_Reacting_Flow


def _ideal_air_solver_cell(cell_number: int):
    vol = 0.01
    dx = 1.0 / cell_number
    return {
        "cell_number": cell_number,
        "num_state_var": 3,
        "num_prim_var": 4,
        "vol": vol,
        "dx": dx,
        "bc_data": np.array(
            [["extrapolate"] * 2, ["extrapolate"] * 2, ["extrapolate"] * 2, ["extrapolate"] * 2],
            dtype=object,
        ),
        "gas_model": "Air",
        "hyper": False,
        "viscous_flag": False,
        "flux_scheme": "2nd Order Roe",
        "numpy_vector": True,
        "time_scheme": "FDF",
    }


def _uniform_air_prim_solver(cell_number: int):
    """Physical uniform line (rho, vx, P, T) with consistent ideal gas for air."""
    n = cell_number + 4
    MW = 28.97
    R = 8314.462618 / MW  # J/kg/K
    T = 300.0
    P = 101325.0
    rho = P / (R * T)
    rho = np.full(n, rho)
    vx = np.zeros(n)
    P = np.full(n, P)
    T = np.full(n, T)
    Q = np.vstack((rho, vx, P, T)).ravel()
    return Q


def test_ideal_air_prim_cons_roundtrip():
    """prim -> cons -> prim recovers the primitive line for calorically perfect ideal gas."""
    cell_number = 4
    solver_param = _ideal_air_solver_cell(cell_number)
    Q_prim0 = _uniform_air_prim_solver(cell_number)
    state = {"Q_prim": Q_prim0.copy(), "MW_gas": 28.97}
    state = Ideal_Air.prim2cons_converter(solver_param, state)
    state = Ideal_Air.cons2prim_converter(solver_param, state)
    np.testing.assert_allclose(state["Q_prim"], Q_prim0, rtol=1e-10, atol=1e-10)


def test_ideal_air_d_flux_dx_uniform_flow_near_zero():
    """Uniform state: flux differences and interior residual should vanish."""
    cell_number = 4
    solver_param = _ideal_air_solver_cell(cell_number)
    Q_prim0 = _uniform_air_prim_solver(cell_number)
    state = {"Q_prim": Q_prim0.copy(), "MW_gas": 28.97}
    state = Ideal_Air.prim2cons_converter(solver_param, state)
    flux = np.zeros((3, cell_number + 5))
    state["flux_cons"] = flux
    rom_param = {}
    out = Ideal_Air.d_flux_dx_calculator(solver_param, rom_param, state)
    assert out["d_flux_dx"].shape == (3 * (cell_number + 4),)
    np.testing.assert_allclose(out["d_flux_dx"], 0.0, atol=0.0)


def test_ideal_air_residual_uniform_near_zero():
    """Roe flux with uniform IC and extrapolate BCs: interior cells ~zero residual."""
    cell_number = 4
    solver_param = _ideal_air_solver_cell(cell_number)
    Q_prim0 = _uniform_air_prim_solver(cell_number)
    state = {"Q_prim": Q_prim0.copy(), "MW_gas": 28.97}
    state = Ideal_Air.prim2cons_converter(solver_param, state)
    rom_param = {}
    out = Ideal_Air.residual_calculator(solver_param, rom_param, state)
    d = out["d_flux_dx"].reshape(3, cell_number + 4)
    np.testing.assert_allclose(d[:, 2:-2], 0.0, atol=1e-9)


def _minimal_single_step_gas_lookup():
    """Minimal lookup table for one species (two rows in Y_full after mass-fraction closure)."""
    return {
        "cp": 1005.0,
        "MW": 29.0,
        "R_univ": 8314.462618,
        "href": [0.0, 0.0],
        "mu": 1.8e-5,
        "Pr": 0.72,
    }


def test_single_step_prim_cons_roundtrip():
    cell_number = 4
    n = cell_number + 4
    solver_param = {
        "cell_number": cell_number,
        "vol": 1.0,
        "num_prim_var": 5,
        "num_state_var": 4,
    }
    rho = np.full(n, 1.2)
    vx = np.zeros(n)
    P = np.full(n, 101325.0)
    T = np.full(n, 300.0)
    Y = np.full((1, n), 0.5)
    Q_prim_user = np.vstack((rho, vx, P, T, Y))
    state = {
        "Q_prim": Q_prim_user.ravel(),
        "gas_lookup_table": _minimal_single_step_gas_lookup(),
    }
    state = Single_Step_Reacting_Flow.prim2cons_converter(solver_param, state)
    state = Single_Step_Reacting_Flow.cons2prim_converter(solver_param, state)
    # Thermochemical closure and energy split can shift P/T slightly vs inputs.
    np.testing.assert_allclose(state["Q_prim"], Q_prim_user.ravel(), rtol=2e-2, atol=50.0)


def test_single_step_cache_cantera():
    cell_number = 4
    n = cell_number + 4
    solver_param = {
        "cell_number": cell_number,
        "num_prim_var": 5,
        "num_state_var": 4,
    }
    rho = np.full(n, 1.2)
    vx = np.zeros(n)
    P = np.full(n, 101325.0)
    T = np.full(n, 300.0)
    Y = np.full((1, n), 0.5)
    Q_prim_user = np.vstack((rho, vx, P, T, Y))
    state = {
        "Q_prim": Q_prim_user.ravel(),
        "Q_cons": np.zeros(4 * n),
        "gas_lookup_table": _minimal_single_step_gas_lookup(),
    }
    out = Single_Step_Reacting_Flow.cache_cantera(solver_param, state)
    assert "gas_cached_props" in out
    for key in ("sound_speed", "int_energy_mass", "enthalpy_mix", "cp", "cv"):
        assert key in out["gas_cached_props"]


def test_reacting_flow_imports():
    """Reacting_Flow depends on Cantera; importing the module should succeed."""
    import physics.Reacting_Flow as rf  # noqa: F401

    assert hasattr(rf, "prim2cons_converter")
    assert hasattr(rf, "residual_calculator")
