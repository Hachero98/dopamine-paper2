"""Reference estimator: nonlinear least squares through the trusted solver.

The estimator fits the parameters with the same finite-difference solver that generated
the data. Its error therefore reflects only the noise and the observation design, never
an optimisation or discretisation artefact. It is the yardstick for the PINN:

    * if the reference estimator cannot recover a parameter, the data do not contain
      it, and the PINN is not expected to do better;
    * if the reference estimator can and the PINN cannot, the PINN has an
      optimisation problem.

Parameters are estimated in raw space (log, logit for eta; `src.model.parameters`).
Jacobians are exact (forward-mode differentiation through the solver). The Laplace
approximation of the posterior under a flat prior on the raw parameters is

    cov_raw = noise_sd^2 (J^T J)^-1 .

This is the Paper 1 estimator (`dopamine_pinn.reference`) extended to the
five-parameter treatment model and to observations at several doses.
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

import config as cfg
from src.model.parameters import PARAM_NAMES, initial_params, raw_bounds, raw_vector, to_physical, to_raw
from src.solvers.fd_solver import interpolate, solve_doses


class ReferenceEstimator:
    """Least-squares estimator of a chosen subset of the parameters.

    names    the parameters to estimate, e.g. config.BASELINE_PARAM_NAMES for a
             baseline-only fit; every other parameter is held at a value given to `fit`.
    release  prescribed R(X, Y, t) assumed by the estimator (default: none).

    Build the estimator once and reuse it: the solver and its Jacobian are compiled on
    the first call and reused for every observation set with the same number of points
    and doses (for example all noise realizations of one design).
    """

    def __init__(self, solver, names=PARAM_NAMES, release=None, C_init=None):
        unknown = set(names) - set(PARAM_NAMES)
        if unknown:
            raise ValueError(f'unknown parameters: {sorted(unknown)}')
        self.solver = solver
        self.names = tuple(names)
        self.fixed_names = tuple(n for n in PARAM_NAMES if n not in self.names)
        free_index = jnp.asarray([PARAM_NAMES.index(n) for n in self.names], int)
        fixed_index = jnp.asarray([PARAM_NAMES.index(n) for n in self.fixed_names], int)

        def predict(theta, raw_fixed, doses, x, y, t, dose_index):
            raw = jnp.zeros(len(PARAM_NAMES)).at[free_index].set(theta).at[fixed_index].set(raw_fixed)
            params = {name: to_physical(name, raw[i]) for i, name in enumerate(PARAM_NAMES)}
            frames = solve_doses(solver, params, doses, C_init, release)
            return interpolate(frames, solver, x, y, t, dose_index)

        def predict_with_aux(*args):
            prediction = predict(*args)
            return prediction, prediction

        self._predict = jax.jit(predict)
        self._predict_many = jax.jit(jax.vmap(predict, in_axes=(0,) + (None,) * 6))
        # One forward-mode pass returns the Jacobian and the prediction together.
        self._jacobian_and_prediction = jax.jit(jax.jacfwd(predict_with_aux, has_aux=True))

    # ---- helpers -------------------------------------------------------------------------
    def _design_arguments(self, design):
        doses, dose_index = np.unique(np.asarray(design['A_d'], float), return_inverse=True)
        return (jnp.asarray(doses), *(jnp.asarray(design[k], jnp.float64) for k in ('x_d', 'y_d', 't_d')),
                jnp.asarray(dose_index))

    def _raw_fixed(self, fixed):
        fixed = fixed or {}
        missing = [n for n in self.fixed_names if n not in fixed]
        if missing:
            raise ValueError(f'values required for the parameters held fixed: {missing}')
        return jnp.asarray([to_raw(n, jnp.asarray(fixed[n], jnp.float64)) for n in self.fixed_names]).reshape(-1)

    def predict(self, params, design):
        """Model concentrations at the design points for a complete parameter dict."""
        theta = jnp.asarray(raw_vector(params, self.names))
        return np.asarray(self._predict(theta, self._raw_fixed(params), *self._design_arguments(design)))

    def chi2(self, thetas, obs, fixed=None, noise_sd=cfg.NOISE_SD, batch_size=8):
        """chi^2 of many raw parameter vectors at once: thetas (n, len(names)) -> (n,).

        The vectors are solved in parallel in groups of `batch_size` (memory grows with
        the group size: one full field per vector and dose).
        """
        thetas = np.atleast_2d(np.asarray(thetas, float))
        args, raw_fixed = self._design_arguments(obs), self._raw_fixed(fixed)
        C_obs = np.asarray(obs['C_d'], float)
        values = []
        for start in range(0, len(thetas), batch_size):
            group = thetas[start:start + batch_size]
            padded = np.concatenate([group, np.repeat(group[-1:], batch_size - len(group), axis=0)])
            predictions = np.asarray(self._predict_many(jnp.asarray(padded), raw_fixed, *args))[:len(group)]
            values.append(np.sum((predictions - C_obs)**2, axis=1) / noise_sd**2)
        return np.concatenate(values)

    # ---- estimation ----------------------------------------------------------------------
    def fit(self, obs, fixed=None, init=None, noise_sd=cfg.NOISE_SD, bounds=cfg.PARAM_BOUNDS,
            n_starts=1, seed=0, max_evaluations=None):
        """Estimate the free parameters from one observation set.

        fixed     physical values of the parameters not in `names`
        init      physical initial guess (default: config.INIT_PARAMS)
        noise_sd  noise SD used for chi^2 and the covariance; None estimates it from
                  the residuals (use this for noise-free data)
        n_starts  > 1 adds starts drawn from config.PARAM_RANGES and keeps the best fit
        max_evaluations  cap on solver calls per start (None: until converged); useful on
                  flat valleys, where the optimiser can creep for a long time

        Returns a dict:
            params      complete dict of physical values (estimated and fixed)
            theta       raw estimate, ordered as `names`
            cov_raw     Laplace covariance of theta;  sd_raw, corr  derived from it
            interval    {name: (lower, upper)} at config.CONFIDENCE, physical units
            chi2, dof, p_value, rms   goodness of fit
            at_bound    True if any estimate sits on a bound (a sign of non-identifiability)
            jacobian    d(prediction) / d(theta) at the optimum
        """
        if 'eta' in self.names and not np.any(np.asarray(obs['A_d']) > 0):
            raise ValueError('eta cannot be estimated from untreated data: at A = 0 the model does not '
                             'depend on eta. Remove it from `names` or add a treated dose.')
        args = self._design_arguments(obs)
        raw_fixed = self._raw_fixed(fixed)
        C_obs = np.asarray(obs['C_d'], float)
        lower, upper = raw_bounds(self.names, bounds)

        # SciPy asks for the residuals and then for the Jacobian at the same point; both
        # come from one solver pass, so the second request is served from this cache.
        cache = {}

        def evaluate(theta):
            key = theta.tobytes()
            if cache.get('key') != key:
                jac, prediction = self._jacobian_and_prediction(jnp.asarray(theta), raw_fixed, *args)
                cache.update(key=key, residuals=np.asarray(prediction) - C_obs, jacobian=np.asarray(jac))
            return cache

        residuals = lambda theta: evaluate(theta)['residuals']
        jacobian = lambda theta: evaluate(theta)['jacobian']

        starts = [raw_vector({**initial_params(), **(init or {})}, self.names)]
        rng = np.random.default_rng(seed)
        for _ in range(n_starts - 1):
            starts.append(raw_vector({n: rng.uniform(*cfg.PARAM_RANGES[n]) for n in self.names}, self.names))

        best = None
        for start in starts:
            start = np.clip(start, lower + 1e-9, upper - 1e-9)
            result = least_squares(residuals, start, jac=jacobian, bounds=(lower, upper), x_scale='jac',
                                   xtol=1e-8, ftol=1e-10, gtol=1e-8, max_nfev=max_evaluations)
            if best is None or result.cost < best.cost:
                best = result
        return self._summarise(best, fixed, noise_sd, lower, upper)

    def _summarise(self, result, fixed, noise_sd, lower, upper):
        n, p = result.fun.size, len(self.names)
        rss = float(np.sum(result.fun**2))
        sigma = float(np.sqrt(rss / max(n - p, 1))) if noise_sd is None else float(noise_sd)
        cov = sigma**2 * np.linalg.pinv(result.jac.T @ result.jac)
        sd = np.sqrt(np.diag(cov))
        z = float(np.sqrt(chi2_distribution.ppf(cfg.CONFIDENCE, 1)))
        estimate = {name: float(to_physical(name, result.x[i])) for i, name in enumerate(self.names)}
        interval = {name: (float(to_physical(name, result.x[i] - z * sd[i])),
                           float(to_physical(name, result.x[i] + z * sd[i]))) for i, name in enumerate(self.names)}
        return {
            'names': list(self.names),
            'params': {**{k: float(v) for k, v in (fixed or {}).items()}, **estimate},
            'theta': result.x, 'cov_raw': cov, 'sd_raw': sd,
            'corr': cov / np.where(np.outer(sd, sd) > 0, np.outer(sd, sd), 1.0),
            'interval': interval,
            'chi2': rss / sigma**2, 'dof': n - p, 'rms': float(np.sqrt(rss / n)),
            'p_value': float(chi2_distribution.sf(rss / sigma**2, n - p)),
            'at_bound': bool(np.any(np.isclose(result.x, lower) | np.isclose(result.x, upper))),
            'jacobian': result.jac, 'n_evaluations': int(result.nfev),
        }


def fit_reference(obs, solver, names=PARAM_NAMES, fixed=None, **fit_options):
    """One-off fit; builds a `ReferenceEstimator` and calls `fit` (recompiles every call)."""
    return ReferenceEstimator(solver, names).fit(obs, fixed=fixed, **fit_options)
