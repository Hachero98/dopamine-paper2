"""Training loop shared by the forward and inverse problems: Adam, then L-BFGS.

This is the optimiser schedule of Paper 1 (`dopamine_pinn.pinn.train`) with fixed loss
weights. Adam runs in jit-compiled chunks so that the Python loop only logs; L-BFGS
then polishes the same weighted loss on the full batch.
"""
import time
from dataclasses import dataclass, replace

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax
from jaxopt import LBFGS

import config as cfg
from src.pinn.losses import TERM_NAMES


@dataclass(frozen=True)
class TrainConfig:
    """Optimiser schedule. Defaults are the Paper 1 inverse setting.

    weights are (w_f, w_i, w_b, w_d) for the PDE, initial, wall and data terms.
    """
    adam_iters: int = cfg.ADAM_ITERS
    lbfgs_iters: int = cfg.LBFGS_ITERS
    lr: float = cfg.LR
    n_chunk: int = cfg.N_CHUNK       # Adam steps per compiled chunk
    weights: tuple = cfg.W_INVERSE
    log_every: int = 2_000           # print every log_every Adam steps
    track_every: int = 500           # record the physical parameters every track_every Adam steps
    verbose: bool = True

    @classmethod
    def quick(cls, **overrides):
        """A schedule of seconds, for smoke tests only; never for results."""
        return cls(**{**dict(adam_iters=200, lbfgs_iters=20, log_every=10_000, track_every=100, verbose=False),
                      **overrides})

    def replace(self, **changes):
        return replace(self, **changes)


def train(p, terms, batch, tc=TrainConfig(), label=''):
    """Minimise sum_j w_j L_j over the pytree p. Returns (p, info).

    terms: the function built by `losses.make_loss_terms`.
    info:  loss terms and total at the end, the trajectory of the physical parameters
           [(step, {name: value}), ...] and the wall-clock minutes.
    """
    if tc.adam_iters % tc.n_chunk:
        raise ValueError('adam_iters must be a multiple of n_chunk')
    weights = jnp.asarray(tc.weights, jnp.float64)
    loss = lambda q, b: jnp.dot(weights, terms(q, b))
    optimiser = optax.adam(tc.lr)
    state = optimiser.init(p)

    @jax.jit
    def adam_chunk(q, state, b):
        def body(_, carry):
            q, state, _ = carry
            value, grads = jax.value_and_grad(loss)(q, b)
            updates, state = optimiser.update(grads, state, q)
            return optax.apply_updates(q, updates), state, value
        return lax.fori_loop(0, tc.n_chunk, body, (q, state, jnp.zeros((), jnp.float64)))

    free = lambda q: {name: float(value) for name, value in terms.physical(q).items() if name in q.get('raw', {})}
    describe = lambda q: '  '.join(f'{name} {value:.5f}' for name, value in free(q).items())
    trajectory = [(0, free(p))]
    start = time.time()

    for step in range(tc.n_chunk, tc.adam_iters + 1, tc.n_chunk):
        p, state, value = adam_chunk(p, state, batch)
        if step % tc.track_every == 0:
            trajectory.append((step, free(p)))
        if tc.verbose and (step == tc.n_chunk or step % tc.log_every == 0 or step == tc.adam_iters):
            print(f'    [{label}] Adam {step:>6d}  loss {float(value):.4e}  {describe(p)}  {time.time() - start:.0f}s')

    if tc.lbfgs_iters:
        p = LBFGS(fun=loss, maxiter=tc.lbfgs_iters, tol=1e-9).run(p, b=batch).params
        trajectory.append((tc.adam_iters + tc.lbfgs_iters, free(p)))
        if tc.verbose:
            print(f'    [{label}] L-BFGS done            {describe(p)}  {time.time() - start:.0f}s')

    final = np.asarray(terms(p, batch))
    return p, {'terms': dict(zip(TERM_NAMES, map(float, final))),
               'loss': float(np.dot(np.asarray(weights), final)),
               'trajectory': trajectory, 'minutes': (time.time() - start) / 60.0}
