"""Mechanistic counterfactual intervention effects and their summaries.

For a dose a, with C^(a) the potential outcome under do(A = a):

    tau_a(x, y, t) = C^(a) - C^(0)                             pointwise effect
    ATE_a(t)       = |Omega|^-1 integral_Omega tau_a dOmega    spatial average over time
    CE_a(x, y)     = integral_0^T tau_a dt                     cumulative effect map
    GTE_a          = (|Omega| T)^-1 double integral of tau_a   one number per dose
    S_A(x, y, t; a) = dC^(a) / da                              dose sensitivity

Fields are arrays (n_times, nx, nx) on the solver grid. Space integrals use the
control-volume weights of the solver and time integrals the trapezoidal rule, so the
summaries of a network field and of a reference field are computed identically.

Inhibiting uptake leaves more dopamine in the extracellular space, so tau_a >= 0 for
a > 0 and eta > 0.
"""
import numpy as np
import jax
import jax.numpy as jnp

from src.solvers.fd_solver import quadrature_weights
from src.utils.metrics import field_errors, relative_l2


def treatment_effect(C_a, C_0):
    """Pointwise effect tau_a = C^(a) - C^(0)."""
    return np.asarray(C_a) - np.asarray(C_0)


def average_treatment_effect(tau, solver):
    """ATE_a(t): spatial mean of tau at each time, shape (n_times,)."""
    weights = quadrature_weights(solver)
    return np.sum(tau * weights, axis=(-2, -1)) / weights.sum()


def cumulative_effect(tau, solver, time_stride=1):
    """CE_a(x, y): time integral of tau, shape (nx, nx), in uM ms.

    time_stride: spacing of the frames in `tau` in units of solver.dt_store.
    """
    return np.trapezoid(tau, dx=solver.dt_store * time_stride, axis=-3)


def global_treatment_effect(tau, solver, time_stride=1):
    """GTE_a: mean of tau over space and time, uM."""
    ate = average_treatment_effect(tau, solver)
    duration = solver.dt_store * time_stride * (tau.shape[-3] - 1)
    return float(np.trapezoid(ate, dx=solver.dt_store * time_stride) / duration)


def effect_summaries(tau, solver, time_stride=1):
    """ATE(t), CE(x, y), GTE and the peak effect of one effect field."""
    return {'ATE': average_treatment_effect(tau, solver),
            'CE': cumulative_effect(tau, solver, time_stride),
            'GTE': global_treatment_effect(tau, solver, time_stride),
            'peak': float(np.max(np.abs(tau)))}


def dose_sensitivity(solver, params, a, release=None):
    """S_A = dC^(a)/da on the solver grid, by forward-mode differentiation through the solver."""
    field = lambda dose: solver(params, dose, release=release)
    return np.asarray(jax.jit(jax.jacfwd(field))(jnp.asarray(float(a))))


def counterfactual_effect(solver, params, a, release=None):
    """Solve for C^(0) and C^(a) with one parameter set and return fields, effect and summaries."""
    fields = jax.vmap(lambda dose: solver(params, dose, release=release))(jnp.asarray([0.0, float(a)]))
    C_0, C_a = np.asarray(fields[0]), np.asarray(fields[1])
    tau = treatment_effect(C_a, C_0)
    return {'dose': float(a), 'C_0': C_0, 'C_a': C_a, 'tau': tau, **effect_summaries(tau, solver)}


def effect_errors(tau_estimate, tau_reference, solver, time_stride=1):
    """Errors of an estimated effect field and of its summaries against the reference.

    Returns the relative L2 and L-infinity errors of tau, the relative L2 errors of
    ATE(t) and CE(x, y), and the relative error of GTE.
    """
    estimate = effect_summaries(tau_estimate, solver, time_stride)
    reference = effect_summaries(tau_reference, solver, time_stride)
    return {'tau': field_errors(tau_estimate, tau_reference),
            'ATE_rel_l2': relative_l2(estimate['ATE'], reference['ATE']),
            'CE_rel_l2': relative_l2(estimate['CE'], reference['CE']),
            'GTE_rel': float((estimate['GTE'] - reference['GTE']) / reference['GTE'])}
