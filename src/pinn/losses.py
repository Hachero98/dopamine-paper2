"""Loss terms of the physics-informed network.

    L = w_f L_PDE + w_i L_IC + w_b L_BC + w_d L_data

    L_PDE    mean square of the PDE residual at the interior collocation points
             r = C_t - D (C_xx + C_yy) - R + Vmax0 (1 - eta A) C / (Km + C) + k C
    L_IC     mean square mismatch with the Gaussian release at t = 0
    L_BC     mean square of the normal derivative on the walls (zero flux)
    L_data   mean square mismatch with the observations

The residual itself comes from `src.model.dopamine_pde`, the same definition the
reference solver integrates.

Trainable state is one pytree:

    p = {'net': <network weights>, 'raw': {name: raw value of each free physical parameter}}

Physical parameters that are not trained (known in a forward problem, frozen in a
training stage) are passed as `pinned` raw values and merged in before the physics is
evaluated. Raw means log, or logit for eta (`src.model.parameters`).
"""
import numpy as np
import jax
import jax.numpy as jnp

from src.model import dopamine_pde as pde
from src.model.parameters import physical_dict

TERM_NAMES = ('pde', 'ic', 'bc', 'data')

_UNIT_X = jnp.array([1.0, 0.0, 0.0, 0.0])
_UNIT_Y = jnp.array([0.0, 1.0, 0.0, 0.0])


def point_derivatives(C, z):
    """C, its gradient (C_x, C_y, C_t, C_A) and the Laplacian C_xx + C_yy at one point z."""
    gradient = jax.grad(C)
    grad_z, hess_x = jax.jvp(gradient, (z,), (_UNIT_X,))
    _, hess_y = jax.jvp(gradient, (z,), (_UNIT_Y,))
    return C(z), grad_z, hess_x[0] + hess_y[1]


def make_loss_terms(net, pinned=None, release=None):
    """Build terms(p, batch) -> array [L_PDE, L_IC, L_BC, L_data] of unweighted mean squares.

    pinned   {name: raw value} of the physical parameters held fixed
    release  prescribed R(x, y, t) assumed by the PINN (default: none)

    Attached helpers:
        terms.physical(p)                the complete dict of physical parameters
        terms.residual(p, x, y, t, A)    PDE residual at arbitrary points
    """
    pinned = {name: jnp.asarray(value, jnp.float64) for name, value in (pinned or {}).items()}
    source = pde.no_release if release is None else release

    def physical(p):
        return physical_dict({**pinned, **p.get('raw', {})})

    def residual(p, x, y, t, A):
        C, params = net.apply(p['net']), physical(p)

        def at_point(x, y, t, A):
            value, grad, laplacian = point_derivatives(C, jnp.stack([x, y, t, A]))
            return pde.residual(value, grad[2], laplacian, params, A, source(x, y, t))
        return jax.vmap(at_point)(x, y, t, A)

    def terms(p, batch):
        C = net.apply(p['net'])
        value = jax.vmap(lambda *z: C(jnp.stack(z)))
        normal_derivative = jax.vmap(
            lambda x, y, t, A, nx, ny: jnp.dot(jax.grad(C)(jnp.stack([x, y, t, A]))[:2], jnp.stack([nx, ny])))

        r = residual(p, batch['x_f'], batch['y_f'], batch['t_f'], batch['A_f'])
        initial = value(batch['x_i'], batch['y_i'], jnp.zeros_like(batch['x_i']), batch['A_i']) - batch['C_i']
        flux = normal_derivative(batch['x_b'], batch['y_b'], batch['t_b'], batch['A_b'], batch['nx_b'], batch['ny_b'])
        data = value(batch['x_d'], batch['y_d'], batch['t_d'], batch['A_d']) - batch['C_d']
        return jnp.stack([jnp.mean(r**2), jnp.mean(initial**2), jnp.mean(flux**2), jnp.mean(data**2)])

    terms.physical, terms.residual, terms.net = physical, residual, net
    return terms


def make_batch(points, obs=None):
    """Collocation points and observations as one dict of float64 JAX arrays.

    obs = None (forward problem) inserts a single dummy observation; give the data term
    zero weight in that case.
    """
    batch = dict(points)
    if obs is None:
        obs = {'x_d': [0.0], 'y_d': [0.0], 't_d': [0.0], 'A_d': [0.0], 'C_d': [0.0]}
    batch.update({key: obs[key] for key in ('x_d', 'y_d', 't_d', 'A_d', 'C_d')})
    return {key: jnp.asarray(np.asarray(value, np.float64)) for key, value in batch.items()}
