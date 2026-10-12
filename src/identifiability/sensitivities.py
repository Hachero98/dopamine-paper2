"""Local sensitivities of the observations to the physical parameters.

The sensitivity matrix of an observation design is

    S_ij = dC(z_i; lambda, A_i) / d(lambda_j),      z_i = (x_i, y_i, t_i),

one row per observation, one column per parameter. It is computed by forward-mode
differentiation through the reference solver, so it is exact for the discrete model
(no finite-difference step to choose).

Three column scalings are offered:

    'absolute'   dC / d(lambda_j)                 units differ between columns
    'relative'   lambda_j dC / d(lambda_j)        uM per unit relative change; columns comparable
    'raw'        dC / d(raw_j)                    with respect to the optimiser variables
                                                  (log lambda, logit eta); gives the Fisher
                                                  information of the estimators

'relative' and 'raw' coincide for D, Vmax0, Km and k. Use 'relative' for spectra and
correlations, and 'raw' wherever the result is compared with a fitted covariance.

Exact structural fact: every row with A_i = 0 has a zero in the eta column.
"""
import numpy as np
import jax
import jax.numpy as jnp

from src.model.parameters import PARAM_NAMES, raw_scale, true_params
from src.solvers.fd_solver import make_design_sampler

SCALINGS = ('absolute', 'relative', 'raw')


def sensitivity_matrix(design, solver, params=None, names=PARAM_NAMES, scaling='relative',
                       noise_sd=None, release=None, C_init=None):
    """Sensitivity matrix of a design, shape (n_observations, len(names)).

    params    point of evaluation (default: the synthetic truth)
    names     parameters that form the columns, in this order
    noise_sd  if given, rows are divided by it, so S^T S is the Fisher information
    """
    if scaling not in SCALINGS:
        raise ValueError(f'scaling must be one of {SCALINGS}')
    params = {**true_params(), **(params or {})}
    sample = make_design_sampler(solver, design, C_init, release)

    def model(free):
        return sample({**params, **dict(zip(names, free))})

    point = jnp.asarray([params[name] for name in names], jnp.float64)
    S = np.asarray(jax.jit(jax.jacfwd(model))(point))
    if scaling == 'relative':
        S = S * np.array([params[name] for name in names])
    elif scaling == 'raw':
        S = S * np.array([float(raw_scale(name, params[name])) for name in names])
    return S if noise_sd is None else S / noise_sd


def field_sensitivities(solver, params=None, A=0.0, names=PARAM_NAMES, scaling='relative'):
    """Sensitivity of the whole field: {name: array (n_frames + 1, nx, nx)}.

    Used for maps of where and when each parameter acts (Figure 3 insets).
    """
    params = {**true_params(), **(params or {})}
    point = {name: jnp.asarray(params[name], jnp.float64) for name in names}
    jacobian = jax.jit(jax.jacfwd(lambda free: solver({**params, **free}, A)))(point)
    scale = {'absolute': lambda n: 1.0, 'relative': lambda n: params[n],
             'raw': lambda n: float(raw_scale(n, params[n]))}[scaling]
    return {name: np.asarray(jacobian[name]) * scale(name) for name in names}
