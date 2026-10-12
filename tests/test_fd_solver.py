"""The reference solver: order of accuracy, conservation, limits, differentiability."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.model import dopamine_pde as pde
from src.solvers.convergence_test import limiting_cases, spatial_convergence, temporal_convergence
from src.solvers.fd_solver import interpolate, make_design_sampler, make_fd_solver, total_amount


def test_second_order_in_space(truth):
    result = spatial_convergence(truth, nx_list=(11, 21, 41, 81), dt=0.02, T_end=5.0)
    assert result['order'][-1] == pytest.approx(2.0, abs=0.15)


def test_second_order_in_time(truth):
    result = temporal_convergence(truth, dt_list=(0.1, 0.05, 0.025), dt_fine=0.005, nx=21, T_end=5.0)
    assert result['order'][-1] == pytest.approx(2.0, abs=0.2)


def test_limiting_cases(truth):
    limits = limiting_cases(truth, nx=21, dt=0.05)
    assert limits['dose_zero_vs_eta_zero'] < 1e-12
    assert limits['low_concentration'] < 1e-3
    assert limits['vanishing_uptake'] < 1e-5
    assert limits['negativity'] < 1e-10


def test_diffusion_alone_conserves_the_total_amount(solver, truth):
    amount = np.asarray(total_amount(solver({**truth, 'Vmax0': 0.0, 'k': 0.0}, 0.0), solver))
    assert np.max(np.abs(amount / amount[0] - 1.0)) < 1e-11


def test_treatment_leaves_more_dopamine(solver, truth):
    untreated, treated = np.asarray(solver(truth, 0.0)), np.asarray(solver(truth, 1.0))
    assert np.all(treated >= untreated - 1e-13)
    assert treated[-1].max() > untreated[-1].max()


def test_release_adds_the_prescribed_amount(solver, truth):
    # No sinks: the total amount rises by amplitude * duration * integral of the Gaussian.
    release = pde.gaussian_stimulation(amplitude=0.05, t_on=10.0, t_off=12.0, sigma=0.5)
    amount = np.asarray(total_amount(solver({**truth, 'Vmax0': 0.0, 'k': 0.0}, 0.0, release=release), solver))
    expected = 0.05 * 2.0 * 2.0 * np.pi * 0.5**2
    assert amount[-1] - amount[0] == pytest.approx(expected, rel=0.06)


def test_interpolation_is_exact_at_nodes(solver, truth):
    frames = solver(truth, 0.5)
    i, j, n = 4, 15, 120
    value = interpolate(frames, solver, jnp.array([solver.x[i]]), jnp.array([solver.x[j]]),
                        jnp.array([n * solver.dt_store]))
    assert float(value[0]) == pytest.approx(float(frames[n, i, j]), abs=1e-13)


def test_solver_is_differentiable_and_eta_is_absent_at_baseline(solver, truth):
    jacobian = jax.jacfwd(lambda p, A: solver(p, A)[-1])
    point = {k: jnp.asarray(v) for k, v in truth.items()}
    at_baseline, at_full_dose = jacobian(point, 0.0), jacobian(point, 1.0)
    assert float(jnp.max(jnp.abs(at_baseline['eta']))) == 0.0
    assert float(jnp.max(jnp.abs(at_full_dose['eta']))) > 0.0
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in at_full_dose.values())


def test_design_sampler_reads_each_point_from_its_own_dose(solver, truth):
    design = {'x_d': np.zeros(2), 'y_d': np.zeros(2), 't_d': np.full(2, 20.0), 'A_d': np.array([1.0, 0.0])}
    values = np.asarray(make_design_sampler(solver, design)(truth))
    centre = solver.nx // 2
    assert values[0] == pytest.approx(float(solver(truth, 1.0)[200, centre, centre]))
    assert values[1] == pytest.approx(float(solver(truth, 0.0)[200, centre, centre]))


def test_time_step_must_divide_the_frame_interval():
    with pytest.raises(ValueError):
        make_fd_solver(dt=0.03)
