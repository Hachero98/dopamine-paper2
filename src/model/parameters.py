"""The physical parameter vector and its unconstrained ("raw") representation.

    lambda0 = (D, Vmax0, Km, k)         baseline parameters, shared by every dose
    lambda  = (D, Vmax0, Km, k, eta)    with the treatment efficacy

Optimisers never see the physical values. D, Vmax0, Km and k are positive and are
optimised as logarithms; eta lies in (0, 1) and is optimised as a logit. Both the PINN
and the reference estimator use these transforms, so their estimates and their
uncertainties are directly comparable.

Two containers are used:

    dict    {'D': ..., 'Vmax0': ..., ...}   readable; what the physics functions take
    vector  values in the order of `names`  what optimisers and linear algebra take
"""
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import expit, logit

import config as cfg

PARAM_NAMES = cfg.PARAM_NAMES
BASELINE_PARAM_NAMES = cfg.BASELINE_PARAM_NAMES
LOGIT_PARAMS = ('eta',)   # parameters confined to (0, 1); all others are positive

LABELS = {'D': r'$D$', 'Vmax0': r'$V_{\max,0}$', 'Km': r'$K_m$', 'k': r'$k$', 'eta': r'$\eta$'}
UNITS = {'D': 'um^2/ms', 'Vmax0': 'uM/ms', 'Km': 'uM', 'k': '1/ms', 'eta': '-'}


# ---- transforms ------------------------------------------------------------------------
def to_raw(name, value):
    """Physical value -> unconstrained value (log, or logit for eta)."""
    return logit(value) if name in LOGIT_PARAMS else jnp.log(value)


def to_physical(name, raw):
    """Unconstrained value -> physical value (exp, or sigmoid for eta)."""
    return expit(raw) if name in LOGIT_PARAMS else jnp.exp(raw)


def raw_scale(name, value):
    """d(physical) / d(raw) at `value`: lambda for a log, eta (1 - eta) for a logit.

    Multiplying a sensitivity dC/d(lambda) by this factor gives the sensitivity to the
    raw parameter, which is dimensionless and comparable across parameters.
    """
    return value * (1.0 - value) if name in LOGIT_PARAMS else value


def raw_dict(params):
    """Dict of physical values -> dict of raw values (same keys)."""
    return {name: to_raw(name, jnp.asarray(value, jnp.float64)) for name, value in params.items()}


def physical_dict(raw):
    """Dict of raw values -> dict of physical values (same keys)."""
    return {name: to_physical(name, value) for name, value in raw.items()}


# ---- dict <-> vector ---------------------------------------------------------------------
def pack(params, names=PARAM_NAMES):
    """Dict -> NumPy vector ordered as `names`."""
    return np.array([float(params[name]) for name in names])


def unpack(vector, names=PARAM_NAMES):
    """Vector ordered as `names` -> dict (elements stay JAX-traceable)."""
    return {name: vector[i] for i, name in enumerate(names)}


def raw_vector(params, names=PARAM_NAMES):
    """Dict of physical values -> NumPy vector of raw values ordered as `names`."""
    return np.array([float(to_raw(name, params[name])) for name in names])


def params_from_raw_vector(theta, names, fixed=None):
    """Raw vector of the free parameters -> complete dict of physical values.

    names: the free parameters, in the order of `theta`.
    fixed: physical values of the parameters that are not estimated (e.g. eta in a
           baseline-only fit). Together `names` and `fixed` must cover PARAM_NAMES.
    """
    params = {name: jnp.asarray(value, jnp.float64) for name, value in (fixed or {}).items()}
    params.update({name: to_physical(name, theta[i]) for i, name in enumerate(names)})
    missing = set(PARAM_NAMES) - set(params)
    if missing:
        raise ValueError(f'parameters neither free nor fixed: {sorted(missing)}')
    return params


def raw_bounds(names, bounds=cfg.PARAM_BOUNDS):
    """Lower and upper bounds of the raw vector for the parameters in `names`."""
    lower = np.array([float(to_raw(name, bounds[name][0])) for name in names])
    upper = np.array([float(to_raw(name, bounds[name][1])) for name in names])
    return lower, upper


# ---- convenience ---------------------------------------------------------------------------
def true_params():
    """A fresh copy of the synthetic truth."""
    return dict(cfg.TRUE_PARAMS)


def initial_params():
    """A fresh copy of the offset initial guesses of the inverse problems."""
    return dict(cfg.INIT_PARAMS)


def sample_params(rng, ranges=cfg.PARAM_RANGES):
    """One parameter set drawn uniformly from the robustness ranges."""
    return {name: float(rng.uniform(*ranges[name])) for name in PARAM_NAMES}


# ---- parameter combinations ----------------------------------------------------------------
# When C << Km the sink is first order and the data see the parameters only through these
# combinations. They can be well determined even when Vmax0, Km, k and eta are not.
DERIVED = {
    'uptake_ratio': lambda p: p['Vmax0'] / p['Km'],                 # linearised DAT rate, 1/ms
    'baseline_rate': lambda p: p['Vmax0'] / p['Km'] + p['k'],       # total clearance rate at A = 0, 1/ms
    'treatment_rate': lambda p: p['eta'] * p['Vmax0'] / p['Km'],    # rate removed by a full dose, 1/ms
}


def derived_quantities(params):
    """{name: value} of the combinations in `DERIVED` for one parameter dict."""
    return {name: float(function(params)) for name, function in DERIVED.items()}
