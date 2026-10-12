"""Paper 2: intervention-induced identifiability and mechanistic counterfactual inference
in nonlinear spatial dopamine dynamics.

The package extends the Paper 1 code base (https://github.com/Hachero98/dopamine-PINN-)
from linear uptake to Michaelis-Menten DAT uptake under a pharmacological intervention:

    C_t = D lap(C) + R - Vmax0 (1 - eta A) C / (Km + C) - k C

Sub-packages
------------
model            the PDE terms, the parameter vector and the initial condition
solvers          finite-difference reference solver, its verification, reference estimator
pinn             network C(x, y, t, A), loss terms, forward and staged inverse training
identifiability  sensitivities, SVD, Fisher information, profile likelihood
counterfactual   treatment design, counterfactual effects, uncertainty propagation
utils            observation designs and collocation, metrics, figures, file I/O
"""
import jax as _jax

# float64 is required: the uptake parameters change the field by less than float32 resolves.
_jax.config.update('jax_enable_x64', True)

__version__ = '0.1.0'
