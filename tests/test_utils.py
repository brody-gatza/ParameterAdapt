"""Tests for utils: reshape_func, input_read_func, and init_func (where practical)."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import reshape_func
from utils import input_read_func
from utils import init_func


# --- reshape_func ---


def test_results_solver2user_and_user2solver_roundtrip():
    num_state_var, num_cell = 2, 4
    n = num_state_var * (num_cell + 4)
    Q = np.arange(n, dtype=float)
    user = reshape_func.results_solver2user_converter(num_state_var, num_cell, Q)
    assert user.shape == (num_state_var, num_cell + 4)
    back = reshape_func.results_user2solver_converter(user)
    np.testing.assert_allclose(back, Q)


def test_user_solver_index_converter():
    """user2solver uses add.outer; solver2user is index % num_cell (cell index per DOF)."""
    S_user = np.array([0, 2, 3])
    num_consv_var, num_cell = 2, 10
    S_solver = reshape_func.user2solver_indx_converter(S_user, num_consv_var, num_cell)
    assert S_solver.shape == (S_user.size * num_consv_var,)
    # C-order ravel: all offsets for var 0, then var 1, ...
    expected_solver = np.add.outer(S_user, np.arange(num_consv_var) * num_cell).ravel()
    np.testing.assert_array_equal(S_solver, expected_solver)
    back = reshape_func.solver2user_indx_converter(S_solver, num_cell)
    np.testing.assert_array_equal(back, S_solver % num_cell)


def test_solver_add_eliminate_ghost_roundtrip():
    cell_number, num_var = 6, 2
    n_int = num_var * cell_number
    Q_int = np.linspace(0.0, 1.0, n_int)
    Q_full = reshape_func.solver_add_ghost(cell_number, num_var, Q_int)
    assert Q_full.shape == ((cell_number + 4) * num_var,)
    Q_back = reshape_func.solver_eliminate_ghost(cell_number, num_var, Q_full)
    np.testing.assert_allclose(Q_back, Q_int, rtol=1e-15)


def test_find_mass_fraction_full():
    MF = np.array([[0.2, 0.5], [0.3, 0.4]])
    out = reshape_func.find_mass_fraction_full(MF)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(np.sum(out, axis=0), 1.0)
    assert np.all(out[out == 0] == 1e-30) or np.all(out > 0)


def test_find_mass_fraction_full_cantera_shape():
    MF = np.array([[0.1, 0.2], [0.2, 0.3]])
    out = reshape_func.find_mass_fraction_full_cantera(MF)
    assert out.shape == (1, MF.shape[1], MF.shape[0] + 1)


def test_assemble_snapshots(tmp_path):
    solver_param = {
        "training_start_iter": 0,
        "training_end_iter": 4,
        "training_step_iter": 2,
        "num_state_var": 2,
        "cell_number": 3,
        "training_data_dir": str(tmp_path),
    }
    # iterations 0, 2
    for it in (0, 2):
        arr = np.ones((solver_param["num_state_var"], solver_param["cell_number"])) * it
        np.save(tmp_path / f"{it}iteration_cons.npy", arr)

    with patch("builtins.print"):
        data = reshape_func.assemble_snapshots(solver_param)

    assert data.shape == (2, 3, 2)
    np.testing.assert_allclose(data[:, :, 0], 0.0)
    np.testing.assert_allclose(data[:, :, 1], 2.0)


# --- input_read_func ---


def test_read_input_file_parsing(tmp_path):
    p = tmp_path / "input.txt"
    p.write_text(
        """
# comment
dt = 0.01
solver_mode = FOM

empty_line_below

flag = true
""",
        encoding="utf-8",
    )
    args = SimpleNamespace(input_file=str(p))
    d = input_read_func.read_input_file(args)
    assert d["dt"] == "0.01"
    assert d["solver_mode"] == "FOM"
    assert d["flag"] == "true"


def test_read_chem_file_eval(tmp_path):
    p = tmp_path / "chem.txt"
    p.write_text("foo = [1, 2]\nbar = 3.5\n", encoding="utf-8")
    d = input_read_func.read_chem_file(str(p))
    assert d["foo"] == [1, 2]
    assert d["bar"] == 3.5


def _minimal_input_param_air_ic_path(tmp_path, ic_path: str):
    """All keys required by init_solver_param (FOM, Air, ic_path, no injection)."""
    td = tmp_path / "training"
    td.mkdir()
    return {
        "solver_mode": "FOM",
        "dt": "0.01",
        "num_steps": "2",
        "time_scheme": "FDF",
        "x_initial": "0.0",
        "x_final": "1.0",
        "cell_number": "5",
        "rho_inlet": "1.0",
        "vel_inlet": "0.0",
        "press_inlet": "101325",
        "temp_inlet": "300",
        "mass_frac_inlet": "1.0",
        "rho_outlet": "1.0",
        "vel_outlet": "0.0",
        "press_outlet": "101325",
        "temp_outlet": "300",
        "mass_frac_outlet": "1.0",
        "ic_path": ic_path.replace("\\", "/"),
        "gas_model": "Air",
        "flux_scheme": "roe",
        "limiter": "False",
        "limiter_method": "none",
        "viscous": "False",
        "numpy_vector": "True",
        "injection": "False",
        "visual": "False",
        "save_visual": "False",
        "variable1": "rho",
        "variable2": "vx",
        "variable3": "P",
        "variable4": "T",
        "update_interval": "1",
        "save_interval": "1",
        "profiling": "False",
        "rom_method": "none",
        "nl_rom_model": "none",
        "arom_method": "none",
        "pod_energy": "0.99",
        "hyper": "False",
        "hyper_method": "none",
        "sampling_rate": "1.0",
        "unsampled_update_freq": "1",
        "init_training_win": "1.0",
        "training_data_dir": str(td).replace("\\", "/"),
        "training_start_iter": "0",
        "training_step_iter": "1",
        "training_end_iter": "1",
        "rom_basis_generate": "False",
        "rom_basis_dir": str(tmp_path).replace("\\", "/"),
        "sarom_training_step": "1",
        "sarom_srom_solver": "none",
        "sarom_cumsum_tol": "1e-6",
        "sarom_moving_avg_tol": "1e-6",
        "arom_restart": "False",
    }


def test_init_solver_param_air_ic_path(tmp_path):
    ic = np.zeros((4, 5))
    ic_path = tmp_path / "ic.npy"
    np.save(ic_path, ic)
    inp = _minimal_input_param_air_ic_path(tmp_path, str(ic_path))
    args = SimpleNamespace(working_directory=str(tmp_path))
    sp = input_read_func.init_solver_param(args, inp)
    assert sp["solver_mode"] == "FOM"
    assert sp["dt"] == 0.01
    assert sp["cell_number"] == 5
    assert sp["dx"] == (1.0 - 0.0) / 5
    assert np.allclose(sp["x"], np.linspace(0.0, 1.0, 5))
    assert sp["ic_data"].shape == (4, 5)
    assert sp["num_species"] == len(sp["ic_data"][:, 0]) - 4


# --- init_func ---


def test_init_injection():
    solver_param = {"non_inj_portion": 0.2, "non_inj_tail_portion": 0.1, "cell_number": 100}
    state = {}
    out = init_func.init_injection(solver_param, state)
    assert out["injection_add_final"] == 20
    assert out["injection_sub_init"] == 10


@pytest.mark.parametrize(
    "scheme, short",
    [("FDF", "FDF"), ("BDF", "BDF"), ("SSPRK2", "SSPRK2"), ("SSPRK3", "SSPRK3")],
)
def test_init_time_integration(scheme, short):
    solver_param = {"time_scheme": scheme}
    mod = init_func.init_time_integration(solver_param)
    assert mod.__name__.split(".")[-1] == short
    assert hasattr(mod, "advance_time")


def test_init_physics_air():
    solver_param = {"gas_model": "Air"}
    physics = init_func.init_physics(solver_param)
    assert physics.__name__.split(".")[-1] == "Ideal_Air"


def test_init_state_air_shapes():
    solver_param = {"gas_model": "Air", "cell_number": 4}
    state = init_func.init_state(solver_param)
    assert solver_param["num_state_var"] == 3
    assert solver_param["num_prim_var"] == 4
    n_cons = 3 * (4 + 4)
    n_prim = 4 * (4 + 4)
    assert state["Q_cons"].shape == (n_cons,)
    assert state["Q_prim"].shape == (n_prim,)
    assert state["cons_results_save"].shape == (3, 4)
    assert state["prim_results_save"].shape == (4, 4)


def test_init_dir_creates_results(tmp_path):
    solver_param = {
        "working_dir": str(tmp_path),
        "solver_mode": "FOM",
        "save_visual": False,
    }
    init_func.init_dir(solver_param)
    dr = tmp_path / "FOM_results"
    assert dr.is_dir()
    assert solver_param["dir_results"] == str(dr)


def test_init_dir_rom_subfolders(tmp_path):
    solver_param = {
        "working_dir": str(tmp_path),
        "solver_mode": "ROM",
        "save_visual": True,
    }
    init_func.init_dir(solver_param)
    dr = Path(solver_param["dir_results"])
    assert (dr / "cons_prim").is_dir()
    assert (dr / "plots").is_dir()


def test_ic_generator_air_from_ic_array(tmp_path):
    cell_number = 5
    ic = np.random.default_rng(0).random((4, cell_number))
    ic_path = tmp_path / "ic.npy"
    np.save(ic_path, ic)
    inp = _minimal_input_param_air_ic_path(tmp_path, str(ic_path))
    args = SimpleNamespace(working_directory=str(tmp_path))
    solver_param = input_read_func.init_solver_param(args, inp)
    state = init_func.init_state(solver_param)
    state = init_func.ic_generator(solver_param, state)
    assert state["Q_prim"].size == solver_param["num_prim_var"] * (cell_number + 4)
    assert np.all(np.isfinite(state["Q_prim"]))
