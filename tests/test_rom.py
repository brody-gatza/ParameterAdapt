"""Tests for rom: basis_func and sampling_func."""

import sys
from pathlib import Path
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from compflowlab.rom import basis_func
from compflowlab.rom import sampling_func


# --- basis_func.adapt_basis ---

N_STATE = 6
N_MODES = 2
WIN = 3
RNG = np.random.default_rng(42)


def _base_rom_param():
    q_ref = RNG.standard_normal(N_STATE)
    norm = np.ones(N_STATE)
    basis = RNG.standard_normal((N_STATE, N_MODES))
    basis, _, _ = np.linalg.svd(basis, full_matrices=False)
    basis = basis[:, :N_MODES]
    F = RNG.standard_normal((N_STATE, WIN))
    Q_R = RNG.standard_normal((N_MODES, WIN))
    return {
        "q_ref": q_ref,
        "norm": norm,
        "basis": basis.copy(),
        "old_basis": basis.copy(),
        "F": F.copy(),
        "Q_R": Q_R.copy(),
    }


def _new_snapshot(rom_param):
    return rom_param["q_ref"] + 0.1 * RNG.standard_normal(N_STATE)


@pytest.mark.parametrize(
    "method",
    ["direct", "svd", "ojas", "grouse", "isvd", "past"],
)
def test_adapt_basis_updates_basis_shape(method):
    solver_param = {"adaptive_rom_method": method}
    rom_param = _base_rom_param()
    if method == "isvd":
        rom_param["isvd_singular"] = np.ones(N_MODES)
    if method == "past":
        rom_param["past_rls"] = np.eye(N_MODES)

    Q_new = _new_snapshot(rom_param)
    out = basis_func.adapt_basis(solver_param, rom_param, Q_new)

    assert out["basis"].shape == (N_STATE, N_MODES)
    assert np.all(np.isfinite(out["basis"]))


def test_adapt_basis_direct_preserves_window_roll():
    solver_param = {"adaptive_rom_method": "direct"}
    rom_param = _base_rom_param()
    Q_new = _new_snapshot(rom_param)
    out = basis_func.adapt_basis(solver_param, rom_param, Q_new)
    assert out["F"].shape == (N_STATE, WIN)
    assert out["Q_R"].shape == (N_MODES, WIN)


# --- sampling_func: small synthetic basis ---


def _make_basis(num_cell=4, num_state_var=2, n_modes=3):
    n = num_cell * num_state_var
    b = RNG.standard_normal((n, n_modes))
    q, _ = np.linalg.qr(b)
    return q


def _solver_param_for_sampling(method, num_cell=4, num_state_var=2, sampling_rate=100.0):
    return {
        "cell_number": num_cell,
        "num_state_var": num_state_var,
        "sampling_method": method,
        "sampling_rate": sampling_rate,
    }


def test_hyper_precompute_gnat():
    num_cell, num_state_var = 4, 2
    basis = _make_basis(num_cell, num_state_var, n_modes=3)
    rom_param = {"basis": basis}
    solver_param = _solver_param_for_sampling("GNAT", num_cell, num_state_var)
    out = sampling_func.hyper_precompute(solver_param, rom_param, static_basis=True)
    assert "S_indx_user" in out and "S_indx_solver" in out
    assert "hyper_precompute" in out
    hp = out["hyper_precompute"]
    S = out["S_indx_solver"]
    # hyper_precompute = basis @ pinv(basis[S, :])  -> (N, |S|)
    assert hp.shape == (basis.shape[0], len(S))
    assert np.all(np.isfinite(hp))


def test_hyper_precompute_qdeim():
    num_cell, num_state_var = 4, 2
    basis = _make_basis(num_cell, num_state_var, n_modes=3)
    rom_param = {"basis": basis}
    solver_param = _solver_param_for_sampling("QDEIM", num_cell, num_state_var)
    out = sampling_func.hyper_precompute(solver_param, rom_param, static_basis=True)
    S = out["S_indx_solver"]
    assert out["hyper_precompute"].shape == (basis.shape[0], len(S))


def test_GNAT_sample_point_finder():
    num_cell, num_state_var = 4, 2
    basis = _make_basis(num_cell, num_state_var, n_modes=4)
    u = sampling_func.GNAT_sample_point_finder(basis, num_cell)
    assert u.ndim == 1
    assert np.all(u >= 0) and np.all(u < num_cell)
    assert len(np.unique(u)) == len(u)


def test_QDEIM_sample_point_finder():
    num_cell, num_state_var = 5, 2
    basis = _make_basis(num_cell, num_state_var, n_modes=3)
    u = sampling_func.QDEIM_sample_point_finder(basis, num_cell)
    assert u.dtype == int or np.issubdtype(u.dtype, np.integer)
    assert len(u) >= basis.shape[1]


def test_QDEIM_R_sample_point_finder():
    np.random.seed(0)
    num_cell, num_state_var = 6, 2
    basis = _make_basis(num_cell, num_state_var, n_modes=3)
    num_samples = 5
    u = sampling_func.QDEIM_R_sample_point_finder(basis, num_samples, num_cell)
    assert len(u) <= num_samples
    assert np.all(u >= 0) and np.all(u < num_cell)


def test_GappyPODE_sample_point_finder():
    num_cell, num_state_var = 8, 2
    n_modes = 4
    basis = _make_basis(num_cell, num_state_var, n_modes=n_modes)
    num_samples = 6
    u = sampling_func.GappyPODE_sample_point_finder(basis, num_samples, num_cell)
    assert u.ndim == 1
    assert len(u) == num_samples


def test_FGS_sample_point_finder():
    np.random.seed(1)
    num_cell, num_state_var = 10, 4
    n_modes = 3
    basis = _make_basis(num_cell, num_state_var, n_modes=n_modes)
    n_solver = num_state_var * (num_cell + 4)
    Q_bar = np.linspace(0.0, 1.0, n_solver)
    rom_param = {"Q_bar": Q_bar}
    num_samples = 8
    u = sampling_func.FGS_sample_point_finder(
        basis, num_samples, num_cell, Q_bar, num_state_var
    )
    assert len(u) <= num_samples
    assert np.all(u >= 0) and np.all(u < num_cell)


def test_ECSW_sample_point_finder_converges_rank_one():
    """ECSW loop exits when snapshot matrix is rank-1 (one column reproduces all)."""
    num_cell, num_state_var = 4, 2
    n_rows = num_cell * num_state_var
    col = np.random.default_rng(2).standard_normal(n_rows)
    tall_thin = col[:, np.newaxis] @ np.random.default_rng(3).standard_normal((1, 5))
    rom_param = {"cent_norm_train_data": tall_thin}
    solver_param = {"num_state_var": num_state_var, "cell_number": num_cell}
    u = sampling_func.ECSW_sample_point_finder(solver_param, rom_param)
    assert u.ndim == 1
    assert len(u) >= 1


def test_hyper_precompute_ecsw_integration():
    num_cell, num_state_var = 4, 2
    n_rows = num_cell * num_state_var
    col = RNG.standard_normal(n_rows)
    tall_thin = col[:, np.newaxis] * np.ones((1, 3))
    basis = _make_basis(num_cell, num_state_var, n_modes=2)
    rom_param = {"basis": basis, "cent_norm_train_data": tall_thin}
    solver_param = _solver_param_for_sampling("ECSW", num_cell, num_state_var)
    out = sampling_func.hyper_precompute(solver_param, rom_param, static_basis=True)
    assert "hyper_precompute" in out
