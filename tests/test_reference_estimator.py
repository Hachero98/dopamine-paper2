"""Laplace covariance of the reference estimator on well-posed and nearly flat problems."""
import numpy as np
import pytest

from src.solvers.reference_estimator import laplace_covariance


def jacobian_with_columns(*columns):
    return np.stack(columns, axis=1)


def test_matches_the_direct_inverse_on_a_well_posed_problem():
    J = np.random.default_rng(0).normal(size=(50, 3))
    cov, unconstrained = laplace_covariance(J)
    assert not unconstrained.any()
    assert cov == pytest.approx(np.linalg.inv(J.T @ J), rel=1e-10)


def test_a_nearly_flat_direction_keeps_its_large_variance():
    # Two columns that differ by 1e-9: the shape of log Vmax0 and log Km deep in the
    # linear regime. The data constrain their difference, not their sum.
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=400), rng.normal(size=400)
    J = jacobian_with_columns(a, -a * (1 + 1e-9) + 1e-9 * b)
    cov, unconstrained = laplace_covariance(J)
    sum_variance = np.array([1.0, 1.0]) @ cov @ np.array([1.0, 1.0]) / 4

    assert not unconstrained.any()
    assert sum_variance > 1e12                     # weakly determined, not known
    # The pseudo-inverse of J^T J drops this direction and reports it as exactly known.
    assert np.diag(np.linalg.pinv(J.T @ J)).max() < 1e-1


def test_an_invisible_direction_is_flagged_and_the_rest_stays_finite():
    rng = np.random.default_rng(2)
    a, c = rng.normal(size=100), rng.normal(size=100)
    J = jacobian_with_columns(a, -a, c)            # only the difference of the first two is seen
    cov, unconstrained = laplace_covariance(J)
    assert unconstrained.tolist() == [True, True, False]
    assert np.all(np.isfinite(cov))
    assert cov[2, 2] == pytest.approx(1.0 / (c @ c - (c @ a)**2 / (a @ a)), rel=1e-8)
