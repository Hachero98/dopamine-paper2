"""Singular-value analysis of a sensitivity matrix.

With S = U diag(s) V^T, a parameter perturbation along the right singular vector v_j
changes the observations by s_j per unit step. A small singular value therefore marks a
weak direction: a combination of parameters that the design barely constrains. The
smallest one, s_min, and its vector are the primary identifiability diagnostics of the
study; a reproducible increase of s_min under treated data is the evidence of
intervention-induced practical identifiability.

Pass relatively scaled sensitivities (`sensitivity_matrix(..., scaling='relative')`) so
that the directions are in units of relative parameter change.
"""
import numpy as np

from src.model.parameters import PARAM_NAMES

RANK_TOLERANCE = 1e-10   # singular values below RANK_TOLERANCE * s_max count as zero


def svd_summary(S, names=PARAM_NAMES):
    """Spectrum, conditioning and weak direction of a sensitivity matrix.

    Returns a dict:
        singular_values     descending
        sigma_min           smallest singular value
        condition_number    s_max / s_min (inf when S is rank deficient)
        rank                numerical rank
        directions          rows are right singular vectors, ordered like singular_values
        weak_direction      {name: component} of the vector belonging to sigma_min
        column_norms        {name: ||S[:, j]||}, the overall strength of each parameter
    """
    names = list(names)
    _, s, Vt = np.linalg.svd(S, full_matrices=False)
    rank = int(np.sum(s > RANK_TOLERANCE * s[0])) if s[0] > 0 else 0
    weak = Vt[-1] * np.sign(Vt[-1][np.argmax(np.abs(Vt[-1]))])   # fix the arbitrary sign
    return {
        'names': names,
        'singular_values': s,
        'sigma_min': float(s[-1]),
        'condition_number': float(s[0] / s[-1]) if rank == len(names) else float('inf'),
        'rank': rank,
        'directions': Vt,
        'weak_direction': dict(zip(names, weak.tolist())),
        'column_norms': dict(zip(names, np.linalg.norm(S, axis=0).tolist())),
    }


def sensitivity_correlations(S):
    """Pairwise cosine similarity of the sensitivity columns, shape (p, p).

    +-1 means two parameters change the observations in the same pattern and cannot be
    separated by this design; 0 means their effects are orthogonal. A zero column (eta
    at baseline) gives NaN, because a direction that is not observed has no angle.
    """
    norms = np.linalg.norm(S, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        unit = np.where(norms > 0, S / norms, np.nan)
    return unit.T @ unit


def collinearity_index(S):
    """Collinearity index of Brun, Reichert & Kunsch (2001): 1 / s_min of the unit-norm columns.

    It removes the overall magnitude of each sensitivity and measures only how nearly
    dependent the columns are. Values above roughly 10-15 indicate a parameter set that
    is practically non-identifiable. Returns inf if any column is zero.
    """
    norms = np.linalg.norm(S, axis=0)
    if np.any(norms == 0):
        return float('inf')
    return float(1.0 / np.linalg.svd(S / norms, compute_uv=False)[-1])


def compare_designs(matrices, names=PARAM_NAMES):
    """Summaries of several designs side by side.

    matrices: {label: S}. Returns {label: {'sigma_min', 'condition_number',
    'collinearity_index', 'singular_values'}}, ready for a table or a JSON file.
    """
    table = {}
    for label, S in matrices.items():
        summary = svd_summary(S, names)
        table[label] = {'sigma_min': summary['sigma_min'], 'condition_number': summary['condition_number'],
                        'collinearity_index': collinearity_index(S),
                        'singular_values': summary['singular_values'].tolist(),
                        'weak_direction': summary['weak_direction']}
    return table
