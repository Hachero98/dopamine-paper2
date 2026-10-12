"""The PDE terms and the parameter transforms."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import config as cfg
from src.model import dopamine_pde as pde
from src.model import parameters as prm
from src.model.initial_conditions import grid_coordinates, initial_field


def test_dose_zero_removes_eta(truth):
    for eta in (0.0, 0.3, 0.9):
        assert pde.effective_vmax(truth['Vmax0'], eta, 0.0) == truth['Vmax0']
    d_uptake_d_eta = jax.grad(lambda eta: pde.dat_uptake(0.3, {**truth, 'eta': eta}, 0.0))(0.6)
    assert float(d_uptake_d_eta) == 0.0


def test_full_dose_scales_capacity(truth):
    assert pde.effective_vmax(truth['Vmax0'], truth['eta'], 1.0) == pytest.approx(truth['Vmax0'] * (1 - truth['eta']))


def test_uptake_limits(truth):
    low, high = 1e-6 * truth['Km'], 1e6 * truth['Km']
    assert float(pde.dat_uptake(low, truth, 0.0)) == pytest.approx(truth['Vmax0'] / truth['Km'] * low, rel=1e-5)
    assert float(pde.dat_uptake(high, truth, 0.0)) == pytest.approx(truth['Vmax0'], rel=1e-5)
    linear_rate = float(pde.low_concentration_rate(truth, 0.5))
    assert float(pde.clearance(low, truth, 0.5)) == pytest.approx(linear_rate * low, rel=1e-5)


def test_residual_vanishes_for_uniform_linear_decay(truth):
    # With Vmax0 = 0 and no gradients, C = exp(-k t) solves the equation exactly.
    params = {**truth, 'Vmax0': 0.0}
    t = jnp.linspace(0.0, 50.0, 11)
    C = jnp.exp(-params['k'] * t)
    residual = pde.residual(C, -params['k'] * C, 0.0, params, 0.7)
    assert float(jnp.max(jnp.abs(residual))) < 1e-15


def test_stimulation_window():
    release = pde.gaussian_stimulation(amplitude=2.0, t_on=10.0, t_off=12.0)
    zero = jnp.asarray(0.0)
    assert float(release(zero, zero, 11.0)) == 2.0
    assert float(release(zero, zero, 9.9)) == 0.0 and float(release(zero, zero, 12.0)) == 0.0


def test_raw_transform_round_trip(truth):
    raw = prm.raw_dict(truth)
    back = prm.physical_dict(raw)
    for name in prm.PARAM_NAMES:
        assert float(back[name]) == pytest.approx(truth[name], rel=1e-12)
    assert float(raw['eta']) == pytest.approx(np.log(0.6 / 0.4))   # logit, not log


def test_params_from_raw_vector_needs_every_parameter(truth):
    theta = prm.raw_vector(truth, cfg.BASELINE_PARAM_NAMES)
    params = prm.params_from_raw_vector(theta, cfg.BASELINE_PARAM_NAMES, fixed={'eta': 0.6})
    assert {k: float(v) for k, v in params.items()} == pytest.approx(truth)
    with pytest.raises(ValueError):
        prm.params_from_raw_vector(theta, cfg.BASELINE_PARAM_NAMES)


def test_initial_field_matches_release_profile():
    field = initial_field(41)
    x = grid_coordinates(41)
    assert field[20, 20] == pytest.approx(cfg.C0)
    assert field[28, 12] == pytest.approx(cfg.C0 * np.exp(-(x[28]**2 + x[12]**2) / (2 * cfg.SIGMA**2)))
