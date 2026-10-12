"""Structural and practical identifiability: sensitivities, SVD, Fisher information, profiles.

`analyse_design` runs the local diagnostics of one observation design in a single call;
the modules hold the individual steps.
"""
import numpy as np

import config as cfg
from src.identifiability import fisher_information as fisher
from src.identifiability import svd_analysis as svd
from src.identifiability.sensitivities import sensitivity_matrix
from src.model.parameters import PARAM_NAMES, true_params


def analyse_design(design, solver, names=PARAM_NAMES, params=None, noise_sd=cfg.NOISE_SD, **solver_options):
    """Local identifiability diagnostics of a design, evaluated at `params` (default: the truth).

    Sensitivities are taken with respect to the raw parameters and divided by the noise
    SD, so the singular values are signal-to-noise ratios and S^T S is the Fisher
    information.

    Returns a JSON-ready dict:
        n_obs, names
        singular_values, sigma_min, condition_number, collinearity_index, weak_direction
        correlations         cosine similarity of the sensitivity columns
        sd                   {name: Cramer-Rao SD of the raw parameter}
        derived_log_sd       {combination: SD of its logarithm}  (see parameters.DERIVED)
        d_optimality, e_optimality
        fisher               the information matrix itself
    """
    params = {**true_params(), **(params or {})}
    S = sensitivity_matrix(design, solver, params, names, scaling='raw', noise_sd=noise_sd, **solver_options)
    spectrum = svd.svd_summary(S, names)
    F = fisher.fisher_information(S)
    information = fisher.information_summary(F, names)
    bound = fisher.cramer_rao_bound(F)
    derived = ({} if bound['singular'] else fisher.derived_log_sd(bound['cov'], params, names))
    return {'n_obs': int(S.shape[0]), 'names': list(names),
            'singular_values': spectrum['singular_values'], 'sigma_min': spectrum['sigma_min'],
            'condition_number': spectrum['condition_number'], 'collinearity_index': svd.collinearity_index(S),
            'weak_direction': spectrum['weak_direction'], 'correlations': svd.sensitivity_correlations(S),
            'sd': information['sd'], 'derived_log_sd': derived,
            'd_optimality': information['d_optimality'], 'e_optimality': information['e_optimality'],
            'fisher': np.asarray(F)}
