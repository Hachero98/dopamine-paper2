"""Counterfactual effects, their summaries and the propagation of parameter uncertainty."""
import numpy as np
import pytest

import config as cfg
from src.counterfactual import counterfactual as cf
from src.counterfactual import uncertainty_propagation as up
from src.solvers.fd_solver import quadrature_weights
from src.solvers.reference_estimator import ReferenceEstimator
from src.utils.sampling import observe, scattered_design


@pytest.fixture(scope='module')
def effect(solver, truth):
    return cf.counterfactual_effect(solver, truth, cfg.HELDOUT_DOSE)


def test_no_dose_or_no_efficacy_means_no_effect(solver, truth):
    assert np.max(np.abs(cf.counterfactual_effect(solver, truth, 0.0)['tau'])) == 0.0
    assert np.max(np.abs(cf.counterfactual_effect(solver, {**truth, 'eta': 0.0}, 1.0)['tau'])) == 0.0


def test_effect_is_non_negative_and_grows_with_dose(solver, truth, effect):
    assert effect['tau'].min() >= -1e-13
    full = cf.counterfactual_effect(solver, truth, 1.0)
    assert full['GTE'] > effect['GTE'] > 0.0


def test_summaries_are_consistent(solver, effect):
    weights = quadrature_weights(solver)
    assert weights.sum() == pytest.approx(solver.L**2)
    # Integrating CE over space or ATE over time gives the same global effect.
    from_ce = np.sum(effect['CE'] * weights) / weights.sum() / solver.T_end
    assert effect['GTE'] == pytest.approx(from_ce, rel=1e-12)
    assert effect['ATE'].shape == (solver.n_frames + 1,) and effect['ATE'][0] == 0.0
    assert effect['peak'] == pytest.approx(np.max(effect['tau']))


def test_time_stride_does_not_change_the_global_effect(solver, effect):
    coarse = cf.global_treatment_effect(effect['tau'][::5], solver, time_stride=5)
    assert coarse == pytest.approx(effect['GTE'], rel=2e-3)


def test_dose_sensitivity_matches_a_finite_difference(solver, truth):
    h = 1e-4
    numerical = (np.asarray(solver(truth, 0.5 + h)) - np.asarray(solver(truth, 0.5 - h))) / (2 * h)
    np.testing.assert_allclose(cf.dose_sensitivity(solver, truth, 0.5), numerical, atol=1e-8)


def test_effect_errors_vanish_for_the_exact_effect(solver, effect):
    errors = cf.effect_errors(effect['tau'], effect['tau'], solver)
    assert errors['tau']['rel_l2'] == 0.0 and errors['GTE_rel'] == 0.0
    doubled = cf.effect_errors(2.0 * effect['tau'], effect['tau'], solver)
    assert doubled['GTE_rel'] == pytest.approx(1.0) and doubled['tau']['rel_l2'] == pytest.approx(1.0)


def test_propagating_identical_sets_gives_no_spread(solver, truth, effect):
    ensemble = up.propagate(solver, [truth] * 3, cfg.HELDOUT_DOSE, batch_size=2)
    np.testing.assert_allclose(ensemble['GTE'], effect['GTE'], rtol=1e-12)
    np.testing.assert_allclose(ensemble['tau_mean'], effect['tau'], atol=1e-14)
    assert ensemble['tau_sd'].max() < 1e-9
    summary = up.reliability_summary(ensemble, effect)
    assert summary['GTE']['width'] == pytest.approx(0.0, abs=1e-14)
    assert summary['GTE']['median'] == pytest.approx(effect['GTE'], rel=1e-12)


def test_efficacy_spread_becomes_effect_spread(solver, truth, effect):
    # Parameter sets that are identical at A = 0 but differ in eta: the baseline fit
    # cannot tell them apart, yet the predicted effects differ.
    sets = [{**truth, 'eta': eta} for eta in (0.2, 0.5, 0.8)]
    design = scattered_design((0.0,), 25, seed=1)
    ensemble = up.propagate(solver, sets, cfg.HELDOUT_DOSE, design=design, batch_size=3)
    assert up.fit_spread(ensemble).max() < 1e-10
    assert ensemble['GTE'][0] < ensemble['GTE'][1] < ensemble['GTE'][2]
    summary = up.reliability_summary(ensemble, effect)
    assert summary['relative']['GTE_width'] > 0.5 and summary['relative']['gte_covered']


def test_posterior_sampling_and_compatibility(solver, truth):
    obs = observe(scattered_design(cfg.TRAIN_DOSES, 40, seed=4), solver, truth, seed=4)
    names = ('D', 'k', 'eta')
    estimator = ReferenceEstimator(solver, names)
    fit = estimator.fit(obs, fixed={'Vmax0': truth['Vmax0'], 'Km': truth['Km']})

    samples = up.sample_laplace(fit, n_samples=6, seed=0)
    assert len(samples) == 6 and all(set(s) == set(cfg.PARAM_NAMES) for s in samples)
    kept, delta = up.select_compatible(estimator, obs, samples + [fit['params']], fit['chi2'])
    assert delta[-1] == pytest.approx(0.0, abs=1e-6) and fit['params'] in kept

    direction, steps = up.weak_direction_set(fit, n_points=5)
    assert steps[2] == 0.0 and direction[2] == pytest.approx(fit['params'])
    assert set(up.parameter_spread(samples)) >= set(cfg.PARAM_NAMES)


def test_baseline_fit_needs_a_prior_range_for_eta(solver, truth):
    obs = observe(scattered_design((0.0,), 40, seed=4), solver, truth, seed=4)
    fit = ReferenceEstimator(solver, ('D', 'k')).fit(obs, fixed={k: truth[k] for k in ('Vmax0', 'Km', 'eta')})
    with pytest.raises(ValueError, match='eta_range'):
        up.sample_laplace(fit, n_samples=3)
    samples = up.sample_laplace(fit, n_samples=20, eta_range=(0.2, 0.8))
    assert all(0.2 <= s['eta'] <= 0.8 for s in samples)


def test_posterior_sampler_returns_the_prior_for_an_unobserved_parameter(solver, truth):
    # Untreated data: eta must stay spread over its prior range, whatever the other parameters do.
    obs = observe(scattered_design((0.0,), 40, seed=4), solver, truth, seed=4)
    sets, diagnostics = up.sample_posterior(obs, solver, n_samples=40, n_chains=8, n_steps=30, burn_in=10,
                                            seed=1, batch_size=8)
    eta = np.array([s['eta'] for s in sets])
    low, high = cfg.PARAM_RANGES['eta']
    assert len(sets) == 40 and np.all((eta > low) & (eta < high))
    assert eta.max() - eta.min() > 0.3 * (high - low)
    for name, (a, b) in cfg.PARAM_RANGES.items():
        assert all(a < s[name] < b for s in sets)
    assert 0.0 <= diagnostics['acceptance_rate'] <= 1.0 and set(diagnostics['r_hat']) == set(cfg.PARAM_NAMES)
