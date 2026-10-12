"""Finite-difference reference solver of the nonlinear treatment model.

    C_t = D lap(C) + R(x, y, t) - Vmax0 (1 - eta A) C / (Km + C) - k C

Space   Second-order central differences on the node-centred nx x nx grid of Paper 1.
        The zero-flux walls use mirror nodes, i.e. half control volumes on the boundary.

Time    Second-order IMEX-BDF2 (SBDF2). Diffusion is implicit, so the step is not limited
        by the grid; the reaction and release terms are explicit, so no nonlinear solve
        is needed:

            (3 C^{n+1} - 4 C^n + C^{n-1}) / (2 dt) = D lap(C^{n+1}) + 2 N^n - N^{n-1}

        with N = R - uptake. The first step is IMEX Euler, which keeps the global
        accuracy at second order.

Solve   The discrete Neumann Laplacian is lap(C) = A1 C + C A1^T with a one-dimensional
        operator A1 = P diag(mu) P^-1. In that eigenbasis the implicit step is a division
        by (a0 - dt D (mu_i + mu_j)), so each step costs four small matrix products.

The solver is a pure JAX function of the parameters and the dose. It can therefore be
differentiated (`jax.jacfwd`) for exact sensitivities and Jacobians, and vectorised
(`jax.vmap`) over doses or over parameter samples. D enters only through the divisor,
so the time step does not depend on the parameters.

This replaces the explicit Euler scheme of Paper 1 (`dopamine_pinn.solver`); the grid,
the boundary treatment and the stored-frame layout are unchanged.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

import config as cfg
from src.model import dopamine_pde as pde
from src.model.initial_conditions import grid_coordinates, initial_field


def _neumann_eigenbasis(nx, dx):
    """Eigen-decomposition A1 = P diag(mu) P^-1 of the 1D mirror-Neumann second difference.

    A1 = W^-1 K with K the symmetric second-difference matrix and W = diag(1/2, 1, ..., 1, 1/2)
    the control-volume widths, so A1 is similar to the symmetric matrix W^-1/2 K W^-1/2.
    Returns (P, P^-1, mu); mu <= 0 and mu[-1] = 0 is the constant (mass) mode.
    """
    w = np.ones(nx)
    w[0] = w[-1] = 0.5
    K = (np.diag(np.full(nx - 1, 1.0), -1) + np.diag(np.full(nx - 1, 1.0), 1)
         - np.diag(np.r_[1.0, np.full(nx - 2, 2.0), 1.0])) / dx**2
    mu, V = np.linalg.eigh(K / np.sqrt(np.outer(w, w)))
    return V / np.sqrt(w)[:, None], V.T * np.sqrt(w)[None, :], mu


def make_fd_solver(nx=cfg.NX_FD, dt=cfg.DT_FD, dt_store=cfg.DT_STORE, T_end=cfg.T, L=cfg.L):
    """Build the solver for one grid and time step.

    Returns solve(params, A, C_init=None, release=None) -> frames of shape
    (n_frames + 1, nx, nx), where frames[j, ix, iy] is C(x_ix, y_iy, j * dt_store).

        params   dict of physical parameters (keys of config.PARAM_NAMES)
        A        dose in [0, 1]
        C_init   initial field on the grid; default is the Gaussian release
        release  R(X, Y, t) on the grid in uM/ms; default is no release

    The grid and step are attached as attributes: nx, dx, x, dt, stride, dt_store,
    n_frames, T_end, L.
    """
    stride = int(round(dt_store / dt))
    n_frames = int(round(T_end / dt_store))
    if not np.isclose(stride * dt, dt_store) or not np.isclose(n_frames * dt_store, T_end):
        raise ValueError('dt must divide dt_store, and dt_store must divide T_end')

    dx = L / (nx - 1)
    x = grid_coordinates(nx, L)
    P, P_inv, mu = (jnp.asarray(a) for a in _neumann_eigenbasis(nx, dx))
    mu_sum = mu[:, None] + mu[None, :]              # eigenvalues of the 2D Laplacian
    X, Y = (jnp.asarray(a) for a in np.meshgrid(x, x, indexing='ij'))
    default_init = jnp.asarray(initial_field(nx, L=L))

    def solve(params, A, C_init=None, release=None):
        C_start = default_init if C_init is None else jnp.asarray(C_init, jnp.float64)
        source = pde.no_release if release is None else release
        D = params['D']

        def explicit_term(C, t):
            return pde.reaction(C, params, A, source(X, Y, t))

        def step(_, carry):
            C, C_prev, N_prev, n = carry
            N = explicit_term(C, n * dt)
            first = n == 0
            # (a0 C^{n+1} + a1 C^n + a2 C^{n-1}) / dt = D lap(C^{n+1}) + b0 N^n + b1 N^{n-1}
            a0, a1, a2 = (jnp.where(first, e, b) for e, b in ((1.0, 1.5), (-1.0, -2.0), (0.0, 0.5)))
            b0, b1 = (jnp.where(first, e, b) for e, b in ((1.0, 2.0), (0.0, -1.0)))
            rhs = -a1 * C - a2 * C_prev + dt * (b0 * N + b1 * N_prev)
            C_next = P @ ((P_inv @ rhs @ P_inv.T) / (a0 - dt * D * mu_sum)) @ P.T
            return C_next, C, N, n + 1

        def advance_one_frame(carry, _):
            carry = lax.fori_loop(0, stride, step, carry)
            return carry, carry[0]

        # C_prev and N_prev are placeholders: the first (Euler) step multiplies them by zero.
        carry = (C_start, C_start, explicit_term(C_start, 0.0), 0)
        _, frames = lax.scan(advance_one_frame, carry, None, length=n_frames)
        return jnp.concatenate([C_start[None], frames])

    solve.nx, solve.dx, solve.x, solve.L = nx, dx, x, L
    solve.dt, solve.stride, solve.dt_store = dt, stride, dt_store
    solve.n_frames, solve.T_end = n_frames, T_end
    return solve


def solve_doses(solver, params, doses, C_init=None, release=None):
    """Fields for several doses at once: shape (n_doses, n_frames + 1, nx, nx)."""
    return jax.vmap(lambda A: solver(params, A, C_init, release))(jnp.asarray(doses, jnp.float64))


def interpolate(frames, solver, x, y, t, dose_index=None):
    """Trilinear interpolation of stored frames at points (x, y, t); differentiable.

    frames: (n_frames + 1, nx, nx), or (n_doses, n_frames + 1, nx, nx) together with
    `dose_index`, the integer position of each point's dose along the first axis.
    """
    n_t, nx = frames.shape[-3], frames.shape[-1]
    ft = t / solver.dt_store
    it = jnp.clip(jnp.floor(ft).astype(int), 0, n_t - 2)
    a = ft - it
    fx = (x + solver.L / 2) / solver.dx
    ix = jnp.clip(jnp.floor(fx).astype(int), 0, nx - 2)
    b = fx - ix
    fy = (y + solver.L / 2) / solver.dx
    iy = jnp.clip(jnp.floor(fy).astype(int), 0, nx - 2)
    c = fy - iy
    lead = () if dose_index is None else (dose_index,)

    def bilinear(ti):
        return ((1 - b) * (1 - c) * frames[lead + (ti, ix, iy)] + b * (1 - c) * frames[lead + (ti, ix + 1, iy)]
                + (1 - b) * c * frames[lead + (ti, ix, iy + 1)] + b * c * frames[lead + (ti, ix + 1, iy + 1)])
    return (1 - a) * bilinear(it) + a * bilinear(it + 1)


def quadrature_weights(solver):
    """Control-volume areas of the grid (half volumes on the walls); they sum to L^2."""
    w = np.ones(solver.nx)
    w[0] = w[-1] = 0.5
    return np.outer(w, w) * solver.dx**2


def total_amount(frames, solver):
    """Total dopamine in the domain for each frame: sum of C over the control volumes."""
    return jnp.sum(frames * jnp.asarray(quadrature_weights(solver)), axis=(-2, -1))


def frame_times(solver):
    """Times of the stored frames, ms."""
    return np.arange(solver.n_frames + 1) * solver.dt_store


def make_design_sampler(solver, design, C_init=None, release=None):
    """Return sample(params) -> model concentrations at the points of an observation design.

    design: dict with arrays x_d, y_d, t_d, A_d (see `src.utils.sampling`). The solver
    runs once per distinct dose and each point is read from the field of its own dose.
    The returned function is differentiable in `params`.
    """
    doses, dose_index = np.unique(np.asarray(design['A_d'], float), return_inverse=True)
    x, y, t = (jnp.asarray(design[key], jnp.float64) for key in ('x_d', 'y_d', 't_d'))
    dose_index = jnp.asarray(dose_index)

    def sample(params):
        frames = solve_doses(solver, params, doses, C_init, release)
        return interpolate(frames, solver, x, y, t, dose_index)
    return sample
