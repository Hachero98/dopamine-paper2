"""Fisher information of an observation design under independent Gaussian noise.

    F = S^T S / noise_sd^2

with S the sensitivity matrix. F is a local summary of how much the design can tell
about the parameters, before any data are drawn:

    * its inverse is the Cramer-Rao bound, the smallest covariance an unbiased
      estimator can reach, and the covariance that the Laplace approximation of the
      reference estimator reports;
    * its eigenvectors with small eigenvalues are the weak directions of the SVD.

Use `sensitivity_matrix(..., scaling='raw')` so that F refers to the raw parameters
(log lambda, logit eta) that the estimators optimise. For the positive parameters a
standard deviation in raw space is a relative standard deviation.
"""
import numpy as np
import jax
import jax.numpy as jnp

from src.model.parameters import DERIVED, PARAM_NAMES, to_physical, to_raw

SINGULAR_TOLERANCE = 1e-12   # relative eigenvalue below which F is treated as singular


def fisher_information(S, noise_sd=1.0):
    """F = S^T S / noise_sd^2. Pass noise_sd = 1 if S is already divided by the noise."""
    return S.T @ S / noise_sd**2


def cramer_rao_bound(F):
    """Covariance bound F^-1, its standard deviations and correlation matrix.

    If F is singular, the non-identifiable directions have unbounded variance; the
    affected standard deviations are returned as inf and the correlations as NaN.
    """
    eigenvalues, vectors = np.linalg.eigh(F)
    informative = eigenvalues > SINGULAR_TOLERANCE * eigenvalues[-1]
    cov = (vectors[:, informative] / eigenvalues[informative]) @ vectors[:, informative].T
    sd = np.sqrt(np.diag(cov))
    if not informative.all():
        unbounded = np.any(np.abs(vectors[:, ~informative]) > 1e-8, axis=1)
        sd = np.where(unbounded, np.inf, sd)
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / np.outer(sd, sd)
    return {'cov': cov, 'sd': sd, 'corr': corr, 'singular': bool(not informative.all())}


def information_summary(F, names=PARAM_NAMES):
    """Scalar summaries of F used to compare designs.

        eigenvalues   ascending
        d_optimality  log det F: volume of the confidence ellipsoid (larger is better)
        e_optimality  smallest eigenvalue: the worst-determined direction
        a_optimality  trace of F^-1: mean variance (smaller is better)
        sd            {name: Cramer-Rao standard deviation of the raw parameter}
        weak_direction {name: component} of the eigenvector of the smallest eigenvalue
    """
    names = list(names)
    eigenvalues, vectors = np.linalg.eigh(F)
    bound = cramer_rao_bound(F)
    weak = vectors[:, 0] * np.sign(vectors[np.argmax(np.abs(vectors[:, 0])), 0])
    with np.errstate(divide='ignore'):
        log_det = float(np.sum(np.log(np.maximum(eigenvalues, 0.0))))
    return {
        'names': names,
        'eigenvalues': eigenvalues,
        'd_optimality': log_det,
        'e_optimality': float(eigenvalues[0]),
        'a_optimality': float('inf') if bound['singular'] else float(np.trace(bound['cov'])),
        'sd': dict(zip(names, bound['sd'].tolist())),
        'corr': bound['corr'],
        'weak_direction': dict(zip(names, weak.tolist())),
        'singular': bound['singular'],
    }


def information_gain(F_reference, F_new):
    """How much a new design improves on a reference design for the same parameters.

    Returns the ratio of the smallest eigenvalues, the change in log det F, and the
    factor by which each Cramer-Rao standard deviation shrinks (> 1 is an improvement).
    """
    reference, new = cramer_rao_bound(F_reference), cramer_rao_bound(F_new)
    smallest = lambda F: float(np.linalg.eigvalsh(F)[0])
    log_det = lambda F: float(np.linalg.slogdet(F)[1])
    with np.errstate(divide='ignore', invalid='ignore'):
        sd_ratio = reference['sd'] / new['sd']
    weakest = smallest(F_reference)
    return {'e_optimality_ratio': smallest(F_new) / weakest if weakest > 0 else float('inf'),
            'd_optimality_gain': log_det(F_new) - log_det(F_reference), 'sd_reduction': sd_ratio}


def derived_log_sd(cov_raw, params, names=PARAM_NAMES):
    """Standard deviation of the logarithm of each parameter combination (delta method).

    cov_raw: covariance of the raw parameters in `names` (a Cramer-Rao bound or a fitted
    Laplace covariance); params: the point of evaluation. A combination such as
    Vmax0 / Km + k can have a small SD while its factors have none: this is how the
    analysis shows what the data do determine. Combinations involving eta are skipped
    when eta is not among `names`.
    """
    theta = jnp.asarray([to_raw(name, jnp.asarray(params[name], jnp.float64)) for name in names])
    result = {}
    for label, function in DERIVED.items():
        if label == 'treatment_rate' and 'eta' not in names:
            continue
        log_value = lambda th: jnp.log(function({**params, **{n: to_physical(n, th[i]) for i, n in enumerate(names)}}))
        gradient = np.asarray(jax.grad(log_value)(theta))
        result[label] = float(np.sqrt(gradient @ cov_raw @ gradient))
    return result
