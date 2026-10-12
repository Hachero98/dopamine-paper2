"""Terms of the nonlinear dopamine transport equation.

    C_t = D lap(C) + R(x, y, t) - Vmax(A) C / (Km + C) - k C,
    Vmax(A) = Vmax0 (1 - eta A),      A in [0, 1].

This module is the single definition of the physics. The finite-difference solver calls
`reaction` for its explicit term and the PINN calls `residual`, so the two cannot drift
apart. All functions are pointwise, written with `jax.numpy`, and differentiable in the
state, the parameters and the dose.

`params` is always a dict with the keys of `config.PARAM_NAMES` in physical units.
"""
import jax.numpy as jnp

import config as cfg


# ---- structural intervention mechanism ----------------------------------------------
def effective_vmax(Vmax0, eta, A):
    """DAT uptake capacity under dose A: Vmax(A) = Vmax0 (1 - eta A).

    A = 0 returns Vmax0 for every eta, which is why eta is structurally absent from
    untreated observations.
    """
    return Vmax0 * (1.0 - eta * A)


# ---- sinks and sources ----------------------------------------------------------------
def dat_uptake(C, params, A):
    """Michaelis-Menten DAT uptake rate Vmax(A) C / (Km + C), uM/ms.

    The denominator uses |C|. It equals Km + C wherever C >= 0 (always true for the
    reference solver) and keeps the term finite when an untrained network outputs a
    negative concentration.
    """
    return effective_vmax(params['Vmax0'], params['eta'], A) * C / (params['Km'] + jnp.abs(C))


def clearance(C, params, A):
    """Total removal rate: DAT uptake plus the first-order clearance k C, uM/ms."""
    return dat_uptake(C, params, A) + params['k'] * C


def reaction(C, params, A, R=0.0):
    """Every non-diffusive term of the right-hand side: R - clearance, uM/ms."""
    return R - clearance(C, params, A)


def residual(C, C_t, lap_C, params, A, R=0.0):
    """PDE residual C_t - D lap(C) - R + clearance; zero for an exact solution."""
    return C_t - params['D'] * lap_C - reaction(C, params, A, R)


def low_concentration_rate(params, A):
    """First-order rate that the sink approaches when C << Km: Vmax(A) / Km + k, 1/ms.

    In this regime Vmax0 and Km enter only through their ratio, and that ratio is itself
    confounded with k at A = 0.
    """
    return effective_vmax(params['Vmax0'], params['eta'], A) / params['Km'] + params['k']


# ---- prescribed release R(x, y, t) -------------------------------------------------------
def no_release(x, y, t):
    """R = 0 after the initial pulse: the primary experiment."""
    return jnp.zeros_like(x * t)


def gaussian_stimulation(amplitude, t_on, t_off, sigma=cfg.SIGMA):
    """Prescribed stimulation used by the robustness extension (experiment 08).

    Returns release(x, y, t) = amplitude exp(-(x^2 + y^2) / (2 sigma^2)) for
    t_on <= t < t_off and zero otherwise; amplitude in uM/ms.
    """
    def release(x, y, t):
        active = (t >= t_on) & (t < t_off)
        return jnp.where(active, amplitude * jnp.exp(-(x**2 + y**2) / (2.0 * sigma**2)), 0.0)
    return release
