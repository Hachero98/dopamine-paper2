"""The concentration network C_theta(x, y, t, A).

The network of Paper 1 (a tanh multilayer perceptron with four hidden layers of 64
units) with one more input, the dose A. A single network therefore represents the
solution at every dose, and its derivative with respect to A is the dose sensitivity.

The trainable weights are a Flax NNX state, a plain pytree that Optax and jaxopt
optimise directly. All parameters are float64.
"""
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

import config as cfg

INPUT_NAMES = ('x', 'y', 't', 'A')


class MLP(nnx.Module):
    """Fully connected tanh network; Glorot-normal weights, zero biases, linear output."""

    def __init__(self, layers, *, rngs):
        self.n_layers = len(layers) - 1
        for i in range(self.n_layers):
            setattr(self, f'lin_{i}', nnx.Linear(
                layers[i], layers[i + 1], kernel_init=nnx.initializers.glorot_normal(),
                bias_init=nnx.initializers.zeros_init(), param_dtype=jnp.float64, rngs=rngs))

    def __call__(self, z):
        h = z
        for i in range(self.n_layers - 1):
            h = jnp.tanh(getattr(self, f'lin_{i}')(h))
        return getattr(self, f'lin_{self.n_layers - 1}')(h)


class ConcentrationNet:
    """Wrapper that separates the network structure from its trainable weights.

        net = ConcentrationNet()
        weights = net.init(seed)            # pytree of trainable weights
        C = net.apply(weights)              # C(z) -> scalar, z = (x, y, t, A)

    normalize = False feeds the raw coordinates, as in Paper 1. normalize = True maps
    every input to [-1, 1] first; it is one of the options of the architecture pilot.
    """

    def __init__(self, layers=cfg.LAYERS, normalize=cfg.NORMALIZE_INPUTS, L=cfg.L, T_end=cfg.T):
        if layers[0] != len(INPUT_NAMES) or layers[-1] != 1:
            raise ValueError('layers must start with 4 inputs (x, y, t, A) and end with 1 output')
        self.layers = tuple(layers)
        self.normalize = normalize
        self._structure = nnx.split(MLP(self.layers, rngs=nnx.Rngs(0)))[0]
        self._centre = jnp.array([0.0, 0.0, T_end / 2, 0.5])
        self._half_range = jnp.array([L / 2, L / 2, T_end / 2, 0.5])

    def init(self, seed):
        """Fresh trainable weights for a given seed."""
        return nnx.split(MLP(self.layers, rngs=nnx.Rngs(seed)))[1]

    def apply(self, weights):
        """Return the scalar function C(z) for the given weights."""
        mlp = nnx.merge(self._structure, weights)
        if self.normalize:
            return lambda z: mlp((z - self._centre) / self._half_range)[0]
        return lambda z: mlp(z)[0]

    @property
    def n_weights(self):
        return sum(a * b + b for a, b in zip(self.layers[:-1], self.layers[1:]))


def predict(net, weights, x, y, t, A, chunk=50_000):
    """Network concentrations at arrays of points (broadcast together, evaluated in chunks)."""
    x, y, t, A = np.broadcast_arrays(*(np.asarray(v, float) for v in (x, y, t, A)))
    points = np.stack([x.ravel(), y.ravel(), t.ravel(), A.ravel()], axis=1)
    evaluate = jax.jit(lambda w, z: jax.vmap(net.apply(w))(z))
    values = [np.asarray(evaluate(weights, jnp.asarray(points[i:i + chunk]))) for i in range(0, len(points), chunk)]
    return np.concatenate(values).reshape(x.shape)


def predict_dose_sensitivity(net, weights, x, y, t, A, chunk=50_000):
    """dC/dA of the network at arrays of points: the dose sensitivity S_A by autodiff."""
    x, y, t, A = np.broadcast_arrays(*(np.asarray(v, float) for v in (x, y, t, A)))
    points = np.stack([x.ravel(), y.ravel(), t.ravel(), A.ravel()], axis=1)
    evaluate = jax.jit(lambda w, z: jax.vmap(lambda zi: jax.grad(net.apply(w))(zi)[3])(z))
    values = [np.asarray(evaluate(weights, jnp.asarray(points[i:i + chunk]))) for i in range(0, len(points), chunk)]
    return np.concatenate(values).reshape(x.shape)


def predict_field(net, weights, solver, A, time_stride=1):
    """Network concentrations on the solver grid at dose A: shape (n_times, nx, nx).

    The layout matches `solver(params, A)[::time_stride]`, so network and reference
    fields can be subtracted and integrated with the same quadrature.
    """
    times = np.arange(0, solver.n_frames + 1, time_stride) * solver.dt_store
    T, X, Y = np.meshgrid(times, solver.x, solver.x, indexing='ij')
    return predict(net, weights, X, Y, T, A)
