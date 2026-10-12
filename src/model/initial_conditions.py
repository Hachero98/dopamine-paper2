"""Spatial grid and the Gaussian release profile of Paper 1.

    C(x, y, 0) = C0 exp(-(x^2 + y^2) / (2 sigma^2))

The same profile serves as the initial field of the finite-difference solver and as the
target of the PINN initial-condition loss, so the two solve the same problem.
"""
import numpy as np

import config as cfg


def grid_coordinates(nx, L=cfg.L):
    """Node coordinates along one side of the square: nx points from -L/2 to L/2 (um)."""
    return np.linspace(-L / 2, L / 2, nx)


def gaussian_release(x, y, C0=cfg.C0, sigma=cfg.SIGMA):
    """Release profile at points (x, y); works on NumPy and JAX arrays (uM)."""
    return C0 * np.e ** (-(x**2 + y**2) / (2.0 * sigma**2))


def initial_field(nx, C0=cfg.C0, sigma=cfg.SIGMA, L=cfg.L):
    """Release profile on the nx x nx solver grid, indexed [i, j] = (x_i, y_j)."""
    x = grid_coordinates(nx, L)
    X, Y = np.meshgrid(x, x, indexing='ij')
    return gaussian_release(X, Y, C0, sigma)
