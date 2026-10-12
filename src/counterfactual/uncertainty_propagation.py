"""From parameter uncertainty to uncertainty of the counterfactual effect (Stage 7).

The central reliability question of the study:

    Two parameter vectors can fit the observed data almost equally well,
        H C(lambda_1, A) ~ H C(lambda_2, A)      at the observed doses,
    and still predict different effects at an unobserved dose,
        tau_a(lambda_1) != tau_a(lambda_2).

To answer it, sets of parameter vectors that the data cannot tell apart are generated
and pushed through the reference solver:

    sample_posterior      Metropolis sampling of the posterior through the solver, with a
                          flat prior on the robustness ranges. Valid whether or not the
                          parameters are identifiable; the method of experiment 07.
    sample_laplace        draws from the Gaussian (Laplace) approximation of a reference
                          fit. Fast, but only meaningful when the fit is well determined.
    weak_direction_set    points along the least-determined direction of a fit
    profile_set           the constrained optima of a profile likelihood
    select_compatible     chi^2 filter that removes candidates the data reject

`propagate` returns the resulting ensemble of effects, and `reliability_summary`
reduces it to bands and widths. If the bands are wide, a good fit to the observed doses
is not sufficient for a reliable treatment prediction. If they are narrow although the
parameters are uncertain, the effect is better identified than the parameters.
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import chi2 as chi2_distribution

import config as cfg
from src.model.parameters import BASELINE_PARAM_NAMES, PARAM_NAMES, raw_bounds, raw_vector, to_physical
from src.solvers.fd_solver import interpolate, quadrature_weights
from src.solvers.reference_estimator import ReferenceEstimator


TARGET_ACCEPTANCE = 0.25   # of the Metropolis sampler; near the optimum for a random walk in a few dimensions


# ---- sets of data-compatible parameter vectors ------------------------------------------
def sample_posterior(obs, solver, n_samples=cfg.N_POSTERIOR_SAMPLES, n_chains=cfg.MCMC_CHAINS,
                     n_steps=cfg.MCMC_STEPS, burn_in=cfg.MCMC_BURN_IN, ranges=cfg.PARAM_RANGES,
                     noise_sd=cfg.NOISE_SD, seed=0, batch_size=8, release=None):
    """Sample the posterior of all five parameters given one observation set.

    Likelihood  independent Gaussian noise of SD noise_sd around the reference solver.
    Prior       flat in the raw parameters (log, logit for eta) inside `ranges`, zero
                outside. The ranges are an assumption; report them with the result.
    Sampler     random-walk Metropolis with `n_chains` parallel chains. All chains are
                advanced by one batched solver call per step. The proposal combines the
                Gauss-Newton curvature at a bounded least-squares fit with the width of
                the ranges, so well-determined directions take small steps and
                undetermined ones take steps on the scale of the prior. Its overall
                size is tuned towards an acceptance rate of 0.25 during burn-in and is
                frozen afterwards, so the retained samples come from a fixed kernel.

    Nothing is linearised in the accept step, so the sample follows curved, flat valleys
    and fills the prior range of a parameter the data do not constrain. In particular,
    with untreated data only, eta simply returns its prior.

    Returns (parameter sets, diagnostics). The sets are `n_samples` draws thinned evenly
    from the chains after burn-in. diagnostics: acceptance rate after burn-in, and the Gelman-Rubin
    statistic R-hat per parameter (values near 1, say below 1.1, indicate convergence;
    increase n_steps otherwise).
    """
    rng = np.random.default_rng(seed)
    names, d = PARAM_NAMES, len(PARAM_NAMES)
    lower, upper = raw_bounds(names, ranges)
    estimator = ReferenceEstimator(solver, names, release)

    # Centre and curvature from a least-squares fit confined to the ranges. Untreated
    # data cannot move eta, so it is left out of that fit and starts at mid-range.
    treated = bool(np.any(np.asarray(obs['A_d']) > 0))
    fit_names = names if treated else BASELINE_PARAM_NAMES
    eta_mid = float(to_physical('eta', 0.5 * (lower[-1] + upper[-1])))
    fit = ReferenceEstimator(solver, fit_names, release).fit(
        obs, fixed=None if treated else {'eta': eta_mid}, bounds=ranges, noise_sd=noise_sd)
    centre = raw_vector(fit['params'], names)
    J = np.zeros((fit['jacobian'].shape[0], d))
    J[:, [names.index(n) for n in fit_names]] = fit['jacobian']
    precision = J.T @ J / noise_sd**2 + np.diag(1.0 / (0.5 * (upper - lower))**2)
    step = np.linalg.cholesky(np.linalg.inv(precision) * 2.38**2 / d)

    inside = lambda theta: np.all((theta > lower) & (theta < upper), axis=1)
    chains = np.clip(centre + 0.5 * rng.standard_normal((n_chains, d)) @ step.T, lower + 1e-9, upper - 1e-9)
    chi2 = estimator.chi2(chains, obs, noise_sd=noise_sd, batch_size=batch_size)
    history, accepted, log_scale = np.empty((n_steps, n_chains, d)), 0, 0.0
    for i in range(n_steps):
        proposal = chains + np.exp(log_scale) * rng.standard_normal((n_chains, d)) @ step.T
        valid = inside(proposal)
        chi2_new = np.full(n_chains, np.inf)
        if valid.any():
            chi2_new[valid] = estimator.chi2(proposal[valid], obs, noise_sd=noise_sd, batch_size=batch_size)
        accept = np.log(rng.uniform(size=n_chains)) < -0.5 * (chi2_new - chi2)
        chains[accept], chi2[accept] = proposal[accept], chi2_new[accept]
        history[i] = chains
        if i < burn_in:
            log_scale += (accept.mean() - TARGET_ACCEPTANCE) / np.sqrt(i + 1.0)
        else:
            accepted += int(accept.sum())

    kept = history[burn_in:]
    within = kept.var(axis=0, ddof=1).mean(axis=0)
    between = kept.mean(axis=0).var(axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_hat = np.sqrt((within * (1 - 1 / len(kept)) + between) / within)
    pooled = kept.reshape(-1, d)
    thinned = pooled[np.linspace(0, len(pooled) - 1, min(n_samples, len(pooled))).astype(int)]
    sets = [{name: float(to_physical(name, theta[j])) for j, name in enumerate(names)} for theta in thinned]
    return sets, {'acceptance_rate': accepted / ((n_steps - burn_in) * n_chains),
                  'proposal_scale': float(np.exp(log_scale)), 'r_hat': dict(zip(names, r_hat.tolist())),
                  'n_chains': n_chains, 'n_steps': n_steps, 'burn_in': burn_in, 'prior_ranges': dict(ranges),
                  'best_chi2': float(min(chi2.min(), fit['chi2'])), 'centre_fit': fit['params']}


def _to_param_dicts(thetas, fit, bounds=cfg.PARAM_BOUNDS):
    """Raw vectors of the fitted parameters -> complete dicts of physical values."""
    lower, upper = raw_bounds(fit['names'], bounds)
    sets = []
    for theta in np.clip(thetas, lower, upper):
        params = dict(fit['params'])
        params.update({name: float(to_physical(name, theta[i])) for i, name in enumerate(fit['names'])})
        sets.append(params)
    return sets


def sample_laplace(fit, n_samples=cfg.N_POSTERIOR_SAMPLES, seed=0, eta_range=None):
    """Draw parameter sets from the Laplace posterior N(theta_hat, cov_raw) of a reference fit.

    eta_range: needed for a baseline-only fit. Untreated data carry no information on
    eta, so eta is drawn uniformly from this range instead (e.g. config.PARAM_RANGES['eta']);
    the range is an assumption and must be reported with the result.
    """
    rng = np.random.default_rng(seed)
    thetas = rng.multivariate_normal(fit['theta'], fit['cov_raw'], size=n_samples, method='svd')
    sets = _to_param_dicts(thetas, fit)
    if 'eta' not in fit['names']:
        if eta_range is None:
            raise ValueError('the fit did not estimate eta; give eta_range to draw it from a prior range')
        for params, eta in zip(sets, rng.uniform(*eta_range, n_samples)):
            params['eta'] = float(eta)
    return sets


def weak_direction_set(fit, n_points=21, confidence=cfg.CONFIDENCE, bounds=cfg.PARAM_BOUNDS):
    """Parameter sets along the least-determined direction of a reference fit.

    The direction is the eigenvector of cov_raw with the largest variance. The points
    run from one side of the joint confidence region to the other, so under the local
    quadratic approximation every one of them fits the data acceptably. Points beyond
    `bounds` are moved onto them, which is where a non-identifiable direction ends.
    Returns (parameter sets, steps in units of the confidence radius from -1 to 1).
    """
    variances, vectors = np.linalg.eigh(fit['cov_raw'])
    radius = np.sqrt(chi2_distribution.ppf(confidence, len(fit['names'])) * variances[-1])
    steps = np.linspace(-1.0, 1.0, n_points)
    thetas = fit['theta'] + np.outer(steps * radius, vectors[:, -1])
    return _to_param_dicts(thetas, fit, bounds), steps


def profile_set(profile, fixed=None):
    """Parameter sets on a profile likelihood that lie below its confidence threshold."""
    sets = []
    for i in np.flatnonzero(profile['delta_chi2'] <= profile['threshold']):
        params = {**(fixed or {}), profile['name']: float(profile['values'][i])}
        params.update({name: float(values[i]) for name, values in profile['nuisance'].items()})
        sets.append(params)
    return sets


def select_compatible(estimator, obs, param_sets, best_chi2, noise_sd=cfg.NOISE_SD, confidence=cfg.CONFIDENCE):
    """Keep the parameter sets that the data cannot reject.

    A set is kept if its chi^2 exceeds the best-fit value by no more than the chi^2
    quantile with as many degrees of freedom as fitted parameters (the joint confidence
    region). The check uses the solver, not a linearisation, so it stays valid where the
    Laplace approximation is poor, e.g. along a curved, flat valley.

    estimator: any `ReferenceEstimator` built on the solver (used only to predict).
    Returns (kept parameter sets, delta chi^2 of every candidate).
    """
    C_obs = np.asarray(obs['C_d'], float)
    chi2 = np.array([np.sum((estimator.predict(params, obs) - C_obs)**2) for params in param_sets]) / noise_sd**2
    delta = chi2 - best_chi2
    threshold = chi2_distribution.ppf(confidence, len(estimator.names))
    return [params for params, d in zip(param_sets, delta) if d <= threshold], delta


def parameter_spread(param_sets, names=PARAM_NAMES):
    """SD of the logarithm of each parameter and combination over a set: its relative spread."""
    from src.model.parameters import derived_quantities
    spread = {name: float(np.std(np.log([params[name] for params in param_sets]))) for name in names}
    derived = [derived_quantities(params) for params in param_sets]
    spread.update({name: float(np.std(np.log([d[name] for d in derived]))) for name in derived[0]})
    return spread


# ---- propagation ---------------------------------------------------------------------------
def propagate(solver, param_sets, a=cfg.HELDOUT_DOSE, design=None, batch_size=8, release=None):
    """Solve for the effect tau_a under every parameter set.

    design: optional observation design. If given, the model prediction at its points is
    returned for every set, so that the spread of the fit can be compared with the
    spread of the effect.

    Returns a dict of arrays over the n parameter sets:
        ATE        (n, n_frames + 1)   spatial-average effect over time
        GTE        (n,)                global effect
        CE_centre  (n,)                cumulative effect at the release site
        tau_mean, tau_sd  (n_frames + 1, nx, nx)   pointwise mean and SD of the effect
        predictions (n, n_points)      only if `design` is given
    """
    weights = jnp.asarray(quadrature_weights(solver))
    centre = solver.nx // 2
    if design is not None:
        doses, dose_index = np.unique(np.asarray(design['A_d'], float), return_inverse=True)
        x, y, t = (jnp.asarray(design[k], jnp.float64) for k in ('x_d', 'y_d', 't_d'))
        doses, dose_index = jnp.asarray(doses), jnp.asarray(dose_index)

    def one(params):
        C_0, C_a = solver(params, 0.0, release=release), solver(params, float(a), release=release)
        tau = C_a - C_0
        ate = jnp.sum(tau * weights, axis=(-2, -1)) / jnp.sum(weights)
        out = {'ATE': ate, 'GTE': jnp.trapezoid(ate, dx=solver.dt_store) / solver.T_end,
               'CE_centre': jnp.trapezoid(tau[:, centre, centre], dx=solver.dt_store), 'tau': tau}
        if design is not None:
            fields = jax.vmap(lambda dose: solver(params, dose, release=release))(doses)
            out['predictions'] = interpolate(fields, solver, x, y, t, dose_index)
        return out

    run = jax.jit(jax.vmap(one))
    n = len(param_sets)
    collected = {key: [] for key in ('ATE', 'GTE', 'CE_centre') + (('predictions',) if design is not None else ())}
    tau_sum = tau_sum_sq = 0.0
    for start in range(0, n, batch_size):
        chunk = param_sets[start:start + batch_size]
        n_valid = len(chunk)
        chunk = chunk + [chunk[-1]] * (batch_size - n_valid)   # pad: one compiled shape for every batch
        stacked = {name: jnp.asarray([params[name] for params in chunk], jnp.float64) for name in PARAM_NAMES}
        out = run(stacked)
        for key in collected:
            collected[key].append(np.asarray(out[key])[:n_valid])
        tau = np.asarray(out['tau'])[:n_valid]
        tau_sum, tau_sum_sq = tau_sum + tau.sum(axis=0), tau_sum_sq + (tau**2).sum(axis=0)

    result = {key: np.concatenate(parts) for key, parts in collected.items()}
    result['tau_mean'] = tau_sum / n
    result['tau_sd'] = np.sqrt(np.maximum(tau_sum_sq / n - result['tau_mean']**2, 0.0))
    return result


def reliability_summary(ensemble, reference=None, confidence=cfg.CONFIDENCE):
    """Bands and widths of an effect ensemble from `propagate`.

    reference: the effect at the true parameters (a dict from `counterfactual_effect`),
    available in the synthetic study only. If given, widths are also reported relative
    to the true effect, together with whether the band contains it.

    Returns:
        GTE          {'median', 'lower', 'upper', 'width'}
        ATE_band     {'median', 'lower', 'upper'}, arrays over time
        tau_sd_max   largest pointwise SD of the effect, uM
        relative     (with reference) GTE width / true GTE, peak tau SD / peak true tau,
                     and gte_covered
    """
    tail = 100.0 * (1.0 - confidence) / 2.0
    quantiles = [tail, 50.0, 100.0 - tail]
    gte_lower, gte_median, gte_upper = np.percentile(ensemble['GTE'], quantiles)
    ate_lower, ate_median, ate_upper = np.percentile(ensemble['ATE'], quantiles, axis=0)
    summary = {'GTE': {'median': float(gte_median), 'lower': float(gte_lower), 'upper': float(gte_upper),
                       'width': float(gte_upper - gte_lower)},
               'ATE_band': {'median': ate_median, 'lower': ate_lower, 'upper': ate_upper},
               'tau_sd_max': float(ensemble['tau_sd'].max())}
    if reference is not None:
        summary['relative'] = {'GTE_width': float((gte_upper - gte_lower) / abs(reference['GTE'])),
                               'tau_sd_max': float(ensemble['tau_sd'].max() / reference['peak']),
                               'gte_covered': bool(gte_lower <= reference['GTE'] <= gte_upper)}
    return summary


def fit_spread(ensemble, noise_sd=cfg.NOISE_SD):
    """RMS distance of each set's predictions from the ensemble mean, in units of the noise SD.

    Values well below 1 mean the sets cannot be told apart by the data: this is the
    left-hand side H C(lambda_1) ~ H C(lambda_2) of the reliability test.
    """
    predictions = ensemble['predictions']
    return np.sqrt(np.mean((predictions - predictions.mean(axis=0))**2, axis=1)) / noise_sd
