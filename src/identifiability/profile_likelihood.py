"""Profile likelihood of one parameter.

For a parameter psi, the profile is the best fit that remains possible when psi is held
at a prescribed value and every other free parameter is re-optimised:

    PL(psi) = min over the other parameters of chi^2(psi, others).

Unlike the Fisher information, the profile is not a local linearisation. It follows the
curved valley of a confounded pair such as (Vmax0, Km) and shows directly whether the
data bound the parameter:

    identifiable                 Delta chi^2 crosses the threshold on both sides
    practically non-identifiable the profile stays below the threshold on one or both
                                 sides within the scanned range (the interval is open)

The threshold is the chi^2 quantile with one degree of freedom (3.84 at 95 %).
"""
import numpy as np
from scipy.stats import chi2 as chi2_distribution

import config as cfg
from src.model.parameters import LOGIT_PARAMS
from src.solvers.reference_estimator import ReferenceEstimator


def profile_grid(name, centre, n_points=cfg.PROFILE_POINTS, span=cfg.PROFILE_SPAN):
    """Values at which a parameter is held fixed.

    Positive parameters: log-spaced from centre / span to centre * span.
    eta: linearly spaced over (0.02, 0.98).
    """
    if name in LOGIT_PARAMS:
        return np.linspace(0.02, 0.98, n_points)
    return centre * np.exp(np.linspace(-np.log(span), np.log(span), n_points))


def _crossing(values, delta, threshold, side):
    """Value where delta first exceeds the threshold, moving away from the minimum; None if never."""
    i_min = int(np.argmin(delta))
    order = range(i_min, -1, -1) if side == 'lower' else range(i_min, len(values))
    previous = i_min
    for i in order:
        if delta[i] > threshold:
            weight = (threshold - delta[previous]) / (delta[i] - delta[previous])
            return float(values[previous] + weight * (values[i] - values[previous]))
        previous = i
    return None


def profile_likelihood(obs, solver, name, names, fixed=None, best_fit=None, grid=None,
                       noise_sd=cfg.NOISE_SD, confidence=cfg.CONFIDENCE, max_evaluations=cfg.PROFILE_MAX_EVALUATIONS):
    """Profile of parameter `name` for one observation set.

    names     all free parameters of the fit, including `name`
    fixed     physical values of the parameters that are not free
    best_fit  result of `ReferenceEstimator(solver, names).fit(obs, ...)`; computed if omitted
    grid      values of `name` to scan; default `profile_grid` around the best fit
    max_evaluations  cap on solver calls of each constrained fit (they are warm-started)

    Returns a dict:
        values, chi2, delta_chi2   the profile
        nuisance                   {other name: its re-optimised values along the profile}
        interval                   (lower, upper); None marks a side that is not bounded
        identifiable               True if both sides are bounded
        log_width                  log(upper / lower) (difference for eta); inf if open
    """
    if name not in names:
        raise ValueError(f'{name} is not among the free parameters {names}')
    fixed = dict(fixed or {})
    if best_fit is None:
        best_fit = ReferenceEstimator(solver, names).fit(obs, fixed=fixed, noise_sd=noise_sd)
    others = tuple(n for n in names if n != name)
    estimator = ReferenceEstimator(solver, others)
    values = profile_grid(name, best_fit['params'][name]) if grid is None else np.asarray(grid, float)

    # Walk outwards from the optimum in both directions, warm-starting each constrained
    # fit from its neighbour so the nuisance parameters follow the valley floor.
    start = int(np.argmin(np.abs(np.log(values / best_fit['params'][name]))))
    chi2 = np.full(len(values), np.nan)
    nuisance = {n: np.full(len(values), np.nan) for n in others}
    for indices in (range(start, len(values)), range(start - 1, -1, -1)):
        guess = {n: best_fit['params'][n] for n in others}
        for i in indices:
            fit = estimator.fit(obs, fixed={**fixed, name: values[i]}, init=guess, noise_sd=noise_sd,
                                max_evaluations=max_evaluations)
            chi2[i] = fit['chi2']
            guess = {n: fit['params'][n] for n in others}
            for n in others:
                nuisance[n][i] = fit['params'][n]

    delta = chi2 - min(float(np.min(chi2)), best_fit['chi2'])
    threshold = float(chi2_distribution.ppf(confidence, 1))
    lower, upper = _crossing(values, delta, threshold, 'lower'), _crossing(values, delta, threshold, 'upper')
    bounded = lower is not None and upper is not None
    if not bounded:
        width = float('inf')
    else:
        width = upper - lower if name in LOGIT_PARAMS else float(np.log(upper / lower))
    return {'name': name, 'values': values, 'chi2': chi2, 'delta_chi2': delta, 'threshold': threshold,
            'nuisance': nuisance, 'interval': (lower, upper), 'identifiable': bounded, 'log_width': width}
