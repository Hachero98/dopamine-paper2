"""The treatment design: what may enter a fit, and what the intervention makes identifiable."""
import numpy as np
import pytest

import config as cfg
from src.counterfactual.treatment_model import TreatmentDesign, potential_outcomes, validate_dose
from src.identifiability import analyse_design
from src.identifiability.sensitivities import sensitivity_matrix
from src.pinn.train_inverse import train_inverse
from src.solvers.reference_estimator import ReferenceEstimator
from src.utils.sampling import (assert_heldout_excluded, collocation_points, electrode_design, observe,
                                scattered_design, select_doses)


def test_design_rejects_heldout_as_training_dose():
    design = TreatmentDesign()
    assert design.all_doses == cfg.ALL_DOSES and design.treated_doses == (0.25, 0.75, 1.0)
    with pytest.raises(ValueError):
        TreatmentDesign(train_doses=(0.0, 0.5, 1.0))
    with pytest.raises(ValueError):
        TreatmentDesign(train_doses=(0.25, 1.0))        # no untreated condition
    with pytest.raises(ValueError):
        validate_dose(1.2)


def test_heldout_data_cannot_enter_training(solver, truth):
    obs = observe(scattered_design((0.0, cfg.HELDOUT_DOSE), 10, seed=1), solver, truth)
    with pytest.raises(ValueError):
        assert_heldout_excluded(obs)
    with pytest.raises(ValueError):
        train_inverse(obs)
    assert_heldout_excluded(select_doses(obs, 0.0))


def test_adding_a_dose_does_not_change_existing_data(solver, truth):
    baseline = observe(scattered_design((0.0,), 30, seed=7), solver, truth, seed=7)
    multidose = observe(scattered_design(cfg.TRAIN_DOSES, 30, seed=7), solver, truth, seed=7)
    for key, values in select_doses(multidose, 0.0).items():
        np.testing.assert_allclose(values, baseline[key], rtol=1e-12, atol=1e-15)


def test_noise_levels_share_locations_and_noise_pattern(solver, truth):
    design = scattered_design((0.0, 1.0), 20, seed=3)
    low, high = (observe(design, solver, truth, noise_sd=sd, seed=3) for sd in (0.01, 0.04))
    np.testing.assert_allclose((high['C_d'] - high['C_clean']), 4.0 * (low['C_d'] - low['C_clean']), atol=1e-15)


def test_electrode_design_layout():
    design = electrode_design((0.0, 1.0), sites=[(1.0, 0.0), (0.0, 2.0)], dt=5.0, t_window=(0.5, 45.0))
    n_times = len(np.arange(0.5, 45.0 + 1e-9, 5.0))
    assert len(design['t_d']) == 2 * 2 * n_times
    assert set(np.unique(design['A_d'])) == {0.0, 1.0}


def test_collocation_counts_and_dose_sampling():
    points = collocation_points(n_domain=200, n_initial=40, n_boundary=40, seed=0)
    assert points['x_f'].shape == (200,) and points['x_i'].shape == (40,) and points['x_b'].shape == (40,)
    assert 0.0 <= points['A_f'].min() and points['A_f'].max() <= 1.0
    assert np.allclose(np.hypot(points['nx_b'], points['ny_b']), 1.0)
    at_baseline = collocation_points(n_domain=50, n_initial=8, n_boundary=8, doses=(0.0,))
    assert not at_baseline['A_f'].any() and not at_baseline['A_i'].any() and not at_baseline['A_b'].any()


def test_eta_is_invisible_without_treatment(solver, truth):
    design = scattered_design((0.0, 1.0), 40, seed=5)
    S = sensitivity_matrix(design, solver, truth, scaling='relative')
    eta = cfg.PARAM_NAMES.index('eta')
    assert np.all(S[design['A_d'] == 0.0, eta] == 0.0)
    assert np.any(S[design['A_d'] == 1.0, eta] != 0.0)

    baseline = analyse_design(select_doses(design, 0.0), solver, cfg.PARAM_NAMES)
    treated = analyse_design(design, solver, cfg.PARAM_NAMES)
    assert baseline['sigma_min'] == 0.0 and np.isinf(baseline['condition_number'])
    assert treated['sigma_min'] > 0.0 and np.isfinite(treated['condition_number'])


def test_estimator_refuses_eta_from_untreated_data(solver, truth):
    obs = observe(scattered_design((0.0,), 20, seed=2), solver, truth)
    with pytest.raises(ValueError, match='eta cannot be estimated'):
        ReferenceEstimator(solver, cfg.PARAM_NAMES).fit(obs)


def test_estimator_recovers_identifiable_parameters_from_exact_data(solver, truth):
    # D, k and eta with the Michaelis-Menten pair known: a well-posed sub-problem.
    obs = observe(scattered_design(cfg.TRAIN_DOSES, 60, seed=11), solver, truth, noise_sd=0.0)
    names = ('D', 'k', 'eta')
    fit = ReferenceEstimator(solver, names).fit(obs, fixed={'Vmax0': truth['Vmax0'], 'Km': truth['Km']}, noise_sd=None)
    for name in names:
        assert fit['params'][name] == pytest.approx(truth[name], rel=1e-5)
    assert not fit['at_bound']


def test_potential_outcomes_are_keyed_by_dose(solver, truth):
    outcomes = potential_outcomes(solver, truth, (0.0, 0.5))
    assert set(outcomes) == {0.0, 0.5}
    assert outcomes[0.5].shape == (solver.n_frames + 1, solver.nx, solver.nx)
