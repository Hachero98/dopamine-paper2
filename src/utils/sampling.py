"""Where the model is observed and where the PINN enforces the physics.

Three kinds of point sets are produced here.

Design        Locations of the observations: a dict of equal-length arrays
              x_d, y_d (um), t_d (ms) and A_d (dose). A design holds no concentrations,
              which is all the identifiability analysis needs.

Observations  A design plus C_clean (noise-free solver values, uM) and C_d (with
              Gaussian noise). This is what the PINN and the reference estimator fit.

Collocation   Points where the PINN evaluates the PDE residual, the initial condition
              and the wall condition.

Reproducibility rule: the points and the noise of dose A under seed s depend only on
(s, A). A baseline-only data set is therefore the A = 0 part of the multi-dose data set
with the same seed, and adding a dose never changes the data already there.
"""
import numpy as np
from scipy.stats import qmc

import config as cfg
from src.model.initial_conditions import gaussian_release
from src.solvers.fd_solver import make_design_sampler

DESIGN_KEYS = ('x_d', 'y_d', 't_d', 'A_d')
OBSERVATION_KEYS = DESIGN_KEYS + ('C_d', 'C_clean')


def _dose_rng(seed, dose, stream):
    """Generator that depends only on (seed, dose, stream); stream 0 = locations, 1 = noise."""
    return np.random.default_rng([int(seed), int(round(1000 * dose)), stream])


# ---- observation designs --------------------------------------------------------------
def scattered_design(doses=cfg.TRAIN_DOSES, n_obs=cfg.N_OBS, seed=cfg.OBS_SEEDS[0],
                     t_window=cfg.T_WINDOW, L=cfg.L):
    """Design I: n_obs points per dose, uniform over the square and the time window.

    n_obs is one number for every dose or a sequence with one entry per dose.
    """
    counts = np.broadcast_to(np.asarray(n_obs, int), (len(doses),))
    parts = []
    for dose, n in zip(doses, counts):
        rng = _dose_rng(seed, dose, 0)
        parts.append({'x_d': rng.uniform(-L / 2, L / 2, n), 'y_d': rng.uniform(-L / 2, L / 2, n),
                      't_d': rng.uniform(*t_window, n), 'A_d': np.full(n, float(dose))})
    return merge(*parts)


def electrode_design(doses=cfg.TRAIN_DOSES, sites=cfg.ELECTRODE_DESIGNS['S1_r1.0'], dt=cfg.ELECTRODE_DT,
                     t_window=cfg.T_WINDOW):
    """Design II (FSCV-like): a time series every dt ms at each fixed site, for each dose.

    sites: [(x, y), ...] in um. The same sites record every dose, as one electrode would.
    """
    times = np.arange(t_window[0], t_window[1] + 1e-9, dt)
    parts = [{'x_d': np.full_like(times, sx), 'y_d': np.full_like(times, sy), 't_d': times,
              'A_d': np.full_like(times, float(dose))} for dose in doses for sx, sy in sites]
    return merge(*parts)


def observe(design, solver, params, noise_sd=cfg.NOISE_SD, seed=cfg.OBS_SEEDS[0], release=None, C_init=None):
    """Evaluate the reference solver at a design and add Gaussian noise of SD noise_sd.

    The standardised noise depends only on (seed, dose), so data sets that differ only
    in noise_sd share their noise pattern (as in the Paper 1 noise sweep).
    """
    clean = np.asarray(make_design_sampler(solver, design, C_init, release)(params))
    noise = np.empty_like(clean)
    for dose in np.unique(design['A_d']):
        at_dose = design['A_d'] == dose
        noise[at_dose] = _dose_rng(seed, dose, 1).standard_normal(int(at_dose.sum()))
    return {**{key: np.asarray(design[key], float) for key in DESIGN_KEYS},
            'C_clean': clean, 'C_d': clean + noise_sd * noise}


def select_doses(points, doses):
    """The part of a design or observation set recorded at the given doses."""
    keep = np.isin(np.round(points['A_d'], 6), np.round(np.atleast_1d(doses), 6))
    return {key: np.asarray(value)[keep] for key, value in points.items()}


def merge(*point_sets):
    """Concatenate designs or observation sets that share the same keys."""
    return {key: np.concatenate([np.asarray(p[key]) for p in point_sets]) for key in point_sets[0]}


def assert_heldout_excluded(points, heldout=cfg.HELDOUT_DOSE):
    """Raise if a training set contains the held-out dose. Call before every fit."""
    if np.any(np.isclose(points['A_d'], heldout)):
        raise ValueError(f'the held-out dose A = {heldout} must not enter training or tuning')


# ---- PINN collocation -----------------------------------------------------------------
def _sample_doses(unit, doses):
    """Map uniform numbers in [0, 1) to doses: A ~ U[0, 1], or an equal-weight discrete set."""
    if isinstance(doses, str):
        if doses != 'continuous':
            raise ValueError("doses must be 'continuous' or a sequence of dose levels")
        return unit
    levels = np.asarray(doses, float)
    return levels[np.minimum((unit * len(levels)).astype(int), len(levels) - 1)]


def collocation_points(n_domain=cfg.N_DOMAIN, n_initial=cfg.N_INITIAL, n_boundary=cfg.N_BOUNDARY,
                       seed=cfg.NET_SEED, doses=cfg.COLLOCATION_DOSES, T_end=cfg.T, L=cfg.L,
                       C0=cfg.C0, sigma=cfg.SIGMA):
    """Latin-hypercube interior points and uniform initial and wall points in (x, y, t, A).

    doses = 'continuous' draws A uniformly from [0, 1]. The PDE is known for every dose,
    so enforcing it at untrained doses uses no held-out data; it is what lets the network
    interpolate in A. Pass a tuple (e.g. (0.0,)) to restrict the physics to given doses.

    Returns a dict with
        x_f, y_f, t_f, A_f                 interior (residual) points
        x_i, y_i, A_i, C_i                 initial points and their target concentration
        x_b, y_b, t_b, A_b, nx_b, ny_b     wall points and the outward unit normal
    """
    rng = np.random.default_rng(seed)
    u = qmc.LatinHypercube(d=4, seed=seed).random(n_domain)
    interior = {'x_f': -L / 2 + L * u[:, 0], 'y_f': -L / 2 + L * u[:, 1], 't_f': T_end * u[:, 2],
                'A_f': _sample_doses(u[:, 3], doses)}

    x_i, y_i = rng.uniform(-L / 2, L / 2, n_initial), rng.uniform(-L / 2, L / 2, n_initial)
    initial = {'x_i': x_i, 'y_i': y_i, 'A_i': _sample_doses(rng.uniform(size=n_initial), doses),
               'C_i': gaussian_release(x_i, y_i, C0, sigma)}

    per_wall = n_boundary // 4
    along = lambda: rng.uniform(-L / 2, L / 2, per_wall)           # coordinate along a wall
    fixed = lambda value: np.full(per_wall, value)                 # coordinate across a wall
    zeros, ones = np.zeros(per_wall), np.ones(per_wall)
    walls = [(fixed(-L / 2), along(), -ones, zeros), (fixed(L / 2), along(), ones, zeros),     # x = -+L/2
             (along(), fixed(-L / 2), zeros, -ones), (along(), fixed(L / 2), zeros, ones)]     # y = -+L/2
    stack = lambda i: np.concatenate([wall[i] for wall in walls])
    boundary = {'x_b': stack(0), 'y_b': stack(1), 'nx_b': stack(2), 'ny_b': stack(3),
                't_b': rng.uniform(0.0, T_end, 4 * per_wall),
                'A_b': _sample_doses(rng.uniform(size=4 * per_wall), doses)}
    return {**interior, **initial, **boundary}
