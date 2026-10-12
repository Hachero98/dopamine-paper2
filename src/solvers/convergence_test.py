"""Verification of the nonlinear reference solver (Stage 1 of the workflow).

Decision rule of the study: if the forward solver is not converged, no inverse
experiment is started. This module provides the evidence:

    spatial_convergence    observed order under grid refinement (expected: 2)
    temporal_convergence   observed order under time-step refinement (expected: 2)
    limiting_cases         A = 0 against eta = 0, the low-concentration linear limit,
                           vanishing nonlinear uptake, positivity

Each function returns a plain dict, so the results can be asserted in the tests and
written to JSON by experiment 01. Run the whole verification with

    python -m src.solvers.convergence_test
"""
import numpy as np
import jax.numpy as jnp

import config as cfg
from src.model import dopamine_pde as pde
from src.model.initial_conditions import initial_field
from src.model.parameters import true_params
from src.solvers.fd_solver import make_fd_solver, total_amount


def _observed_orders(errors):
    """Orders log2(e_i / e_{i+1}) of a sequence of errors under successive halving."""
    errors = np.asarray(errors)
    return np.log2(errors[:-1] / errors[1:])


def spatial_convergence(params=None, A=1.0, nx_list=(21, 41, 81, 161), dt=cfg.DT_FD, T_end=10.0):
    """Self-convergence under grid refinement at a fixed, small time step.

    Each grid must contain the nodes of the previous one (nx -> 2 nx - 1). The error of
    grid h is the difference to grid h/2 at the shared nodes, in the discrete relative
    L2 norm over all stored frames.
    """
    params = params or true_params()
    fields = [np.asarray(make_fd_solver(nx=nx, dt=dt, T_end=T_end)(params, A)) for nx in nx_list]
    errors = []
    for coarse, fine in zip(fields[:-1], fields[1:]):
        shared = fine[:, ::2, ::2]
        errors.append(float(np.linalg.norm(coarse - shared) / np.linalg.norm(shared)))
    return {'nx': list(nx_list), 'dx': [cfg.L / (nx - 1) for nx in nx_list], 'error': errors,
            'order': _observed_orders(errors).tolist()}


def temporal_convergence(params=None, A=1.0, dt_list=(0.1, 0.05, 0.025, 0.0125), dt_fine=0.0025,
                         nx=cfg.NX_FD, T_end=10.0):
    """Convergence under time-step refinement on a fixed grid, against a much finer step."""
    params = params or true_params()
    reference = np.asarray(make_fd_solver(nx=nx, dt=dt_fine, T_end=T_end)(params, A))
    errors = [float(np.linalg.norm(np.asarray(make_fd_solver(nx=nx, dt=dt, T_end=T_end)(params, A)) - reference)
                    / np.linalg.norm(reference)) for dt in dt_list]
    return {'dt': list(dt_list), 'dt_fine': dt_fine, 'error': errors, 'order': _observed_orders(errors).tolist()}


def limiting_cases(params=None, nx=cfg.NX_FD, dt=cfg.DT_FD):
    """Checks that have a known answer. Every entry is a number that should be ~0.

    dose_zero_vs_eta_zero   A = 0 and eta = 0 switch the treatment off in different ways
                            and must give the same field (difference, uM).
    low_concentration       with C0 << Km the model must approach the linear model with
                            rate Vmax0 (1 - eta A) / Km + k (relative difference).
    vanishing_uptake        with Vmax0 = 0 the total amount must decay as exp(-k t)
                            (relative difference at t = T).
    negativity              most negative concentration over all doses (uM, as a
                            positive number; 0 means the field stayed non-negative).
    """
    params = params or true_params()
    solver = make_fd_solver(nx=nx, dt=dt)
    untreated = np.asarray(solver(params, 0.0))

    # A = 0 against eta = 0 at full dose.
    no_efficacy = np.asarray(solver({**params, 'eta': 0.0}, 1.0))
    dose_zero_vs_eta_zero = float(np.max(np.abs(untreated - no_efficacy)))

    # Low concentration: a release far below Km behaves linearly.
    tiny = 1e-4 * params['Km']
    C_init = initial_field(nx, C0=tiny)
    nonlinear = np.asarray(solver(params, 1.0, C_init=C_init))
    rate = float(pde.low_concentration_rate(params, 1.0))
    linear = np.asarray(solver({**params, 'Vmax0': 0.0, 'k': rate}, 1.0, C_init=C_init))
    low_concentration = float(np.linalg.norm(nonlinear - linear) / np.linalg.norm(linear))

    # Vmax0 = 0: pure diffusion with first-order clearance, total amount is exponential.
    amount = np.asarray(total_amount(solver({**params, 'Vmax0': 0.0}, 0.5), solver))
    vanishing_uptake = float(abs(amount[-1] / amount[0] / np.exp(-params['k'] * solver.T_end) - 1.0))

    minimum = min(float(jnp.min(solver(params, A))) for A in cfg.ALL_DOSES)
    return {'dose_zero_vs_eta_zero': dose_zero_vs_eta_zero, 'low_concentration': low_concentration,
            'vanishing_uptake': vanishing_uptake, 'negativity': max(0.0, -minimum)}


def verify(order_tolerance=0.2, nx_list=(21, 41, 81, 161), dt_list=(0.1, 0.05, 0.025, 0.0125), nx=cfg.NX_FD):
    """Run every check; returns the results and a single pass/fail flag.

    The observed orders on the finest pair must be within `order_tolerance` of 2.
    """
    spatial = spatial_convergence(nx_list=nx_list)
    temporal = temporal_convergence(dt_list=dt_list, nx=nx)
    limits = limiting_cases(nx=nx)
    passed = bool(abs(spatial['order'][-1] - 2.0) < order_tolerance
                  and abs(temporal['order'][-1] - 2.0) < order_tolerance
                  and limits['dose_zero_vs_eta_zero'] < 1e-12
                  and limits['low_concentration'] < 1e-3
                  and limits['vanishing_uptake'] < 1e-6
                  and limits['negativity'] < 1e-10)
    return {'spatial': spatial, 'temporal': temporal, 'limiting_cases': limits, 'passed': passed}


if __name__ == '__main__':
    report = verify()
    print('spatial orders :', np.round(report['spatial']['order'], 3))
    print('temporal orders:', np.round(report['temporal']['order'], 3))
    for name, value in report['limiting_cases'].items():
        print(f'{name:<24s}{value:.3e}')
    print('PASSED' if report['passed'] else 'FAILED')
