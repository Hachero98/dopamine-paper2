"""The structural intervention and the potential outcomes it defines.

The intervention is a normalised DAT-targeting inhibitor with dose A in [0, 1]. It acts
on one term of the PDE only:

    Vmax(A) = Vmax0 (1 - eta A).

Setting the dose by hand, do(A = a), and solving the PDE gives the potential outcome

    C^(a)(x, y, t) = C(x, y, t | do(A = a)).

These are mechanistic counterfactuals of a synthetic model: the causal pathway is
imposed by the structure of the PDE. They are not experimentally identified causal
effects in a biological population, and results must not be worded as if they were.
"""
from dataclasses import dataclass

import numpy as np

import config as cfg
from src.solvers.fd_solver import solve_doses


@dataclass(frozen=True)
class TreatmentDesign:
    """Which doses are observed for fitting and which dose is predicted without data."""
    train_doses: tuple = cfg.TRAIN_DOSES
    heldout_dose: float = cfg.HELDOUT_DOSE

    def __post_init__(self):
        for dose in (*self.train_doses, self.heldout_dose):
            validate_dose(dose)
        if any(np.isclose(dose, self.heldout_dose) for dose in self.train_doses):
            raise ValueError(f'the held-out dose A = {self.heldout_dose} must not be a training dose')
        if not any(np.isclose(dose, 0.0) for dose in self.train_doses):
            raise ValueError('the training doses must include the untreated condition A = 0')

    @property
    def treated_doses(self):
        """Training doses with A > 0."""
        return tuple(dose for dose in self.train_doses if dose > 0)

    @property
    def all_doses(self):
        """Every dose for which a reference field is generated, in increasing order."""
        return tuple(sorted({*self.train_doses, self.heldout_dose}))


def validate_dose(A):
    """Raise unless 0 <= A <= 1."""
    if not 0.0 <= float(A) <= 1.0:
        raise ValueError(f'dose A = {A} is outside [0, 1]')


def uptake_capacity_fraction(eta, A):
    """Fraction of the untreated DAT capacity that remains under dose A: 1 - eta A."""
    return 1.0 - eta * A


def potential_outcomes(solver, params, doses, release=None, C_init=None):
    """Potential outcomes {dose: field (n_frames + 1, nx, nx)} from the reference solver."""
    for dose in doses:
        validate_dose(dose)
    fields = np.asarray(solve_doses(solver, params, doses, C_init, release))
    return {float(dose): fields[i] for i, dose in enumerate(doses)}
