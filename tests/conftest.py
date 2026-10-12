"""Shared fixtures: a coarse solver that keeps the whole suite to about a minute on a CPU."""
import pytest

from src.model.parameters import true_params
from src.solvers.fd_solver import make_fd_solver


@pytest.fixture(scope='session')
def solver():
    """Coarse grid and time step: accurate enough for identities, far too coarse for results."""
    return make_fd_solver(nx=21, dt=0.1)


@pytest.fixture(scope='session')
def truth():
    return true_params()
