# Code guide

How the code is organised, what each piece takes and returns, and how to extend it.
Read this once before changing anything in `src/`.

For the meaning of every number see [PARAMETER_GUIDE.md](PARAMETER_GUIDE.md); for how
to run the study see [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md).

---

## 1. The idea in one page

The study has one model and two ways of solving it.

```
                       config.py   (every number of the study)
                           |
                 src/model/dopamine_pde.py   (the physics, defined once)
                    /                    \
   src/solvers/fd_solver.py          src/pinn/losses.py
   reference solver (trusted)        PDE residual of the network
          |                                   |
   reference_estimator.py            train_forward.py / train_inverse.py
   least squares through             network + parameters fitted
   the solver                        to the same observations
          |                                   |
          +-------- same observations --------+      src/utils/sampling.py
          |                                   |
   src/identifiability/              src/counterfactual/
   what can the data determine?      what does the model predict at an unseen dose,
                                     and how certain is that prediction?
```

Three rules follow from this picture and hold everywhere in the code.

1. **The physics is written once.** `dopamine_pde.reaction` feeds the solver and
   `dopamine_pde.residual` feeds the PINN. Changing the model means changing that file.
2. **The reference solver is the yardstick.** Identifiability, the reference estimator,
   the counterfactual truth and every error of the PINN are computed with it.
3. **The held-out dose never enters a fit.** `assert_heldout_excluded` guards every
   training entry point.

---

## 2. Data structures

There are no custom classes for data. Everything is a NumPy/JAX array or a dict of them.

### Parameters

| Form | Example | Used by |
|---|---|---|
| dict of physical values | `{'D': 0.32, 'Vmax0': 0.004, 'Km': 0.20, 'k': 0.020, 'eta': 0.60}` | physics, solver, results |
| raw vector | `[log D, log Vmax0, log Km, log k, logit eta]` | optimisers, covariances |

The order of every vector is `config.PARAM_NAMES`. "Raw" always means the unconstrained
variable an optimiser works on: the logarithm of a positive parameter and the logit of
`eta`. `src/model/parameters.py` converts between the forms.

For the four positive parameters, a standard deviation in raw space is a relative
standard deviation (0.05 means about 5 %).

### Fields

A field is an array `frames[j, ix, iy] = C(x_ix, y_iy, j * dt_store)` of shape
`(n_frames + 1, nx, nx)`. With the defaults this is `(501, 81, 81)`: 0 to 50 ms every
0.1 ms on an 81 x 81 grid. Several doses add a leading axis.

When plotting, `ix` is the horizontal axis; `plotting._map` transposes for you.

### Designs and observations

A dict of equal-length arrays. See [`data/README.md`](../data/README.md).

```
design        x_d, y_d, t_d, A_d
observations  x_d, y_d, t_d, A_d, C_clean, C_d
```

### PINN state

```python
p = {'net': <network weights>, 'raw': {'D': ..., 'Km': ...}}   # only the free parameters
```

Parameters that are not trained are passed separately as `pinned` raw values.

### Results

Every function that produces a result returns a plain dict. Keys that begin with an
underscore (`_net`, `_weights`) are in-memory handles and are not written to disk.

---

## 3. Module by module

### `config.py`

All settings, grouped: model, treatment design, observation designs, solver, PINN,
identifiability, paths. Library functions read them as keyword defaults.

### `src/model/`

| File | Content |
|---|---|
| `dopamine_pde.py` | `effective_vmax`, `dat_uptake`, `clearance`, `reaction`, `residual`, `low_concentration_rate`, and the release functions `no_release`, `gaussian_stimulation` |
| `parameters.py` | names, labels, the raw transforms, dict/vector conversion, bounds, and `DERIVED`: the combinations `Vmax0/Km`, `Vmax0/Km + k`, `eta Vmax0/Km` |
| `initial_conditions.py` | grid coordinates and the Gaussian release |

### `src/solvers/`

| File | Content |
|---|---|
| `fd_solver.py` | `make_fd_solver` returns `solve(params, A, C_init=None, release=None)`. Also `solve_doses` (several doses at once), `interpolate` (values at arbitrary points), `make_design_sampler` (model values at a design), `quadrature_weights`, `total_amount`, `frame_times` |
| `convergence_test.py` | `spatial_convergence`, `temporal_convergence`, `limiting_cases`, `verify` |
| `reference_estimator.py` | `ReferenceEstimator(solver, names).fit(obs, fixed=...)`: least squares with exact Jacobians and a Laplace covariance; `.chi2(thetas, obs)` evaluates many parameter vectors in one batch |

The solver is second order in space and time (IMEX-BDF2, diffusion implicit). It is a
pure JAX function, so:

```python
import jax
from src.solvers.fd_solver import make_fd_solver
from src.model.parameters import true_params

solver = make_fd_solver()
frames = solver(true_params(), 0.5)                                   # one dose
sensitivity = jax.jacfwd(lambda p: solver(p, 0.5))(true_params())     # dC/d(each parameter)
```

`reference_estimator.py` is not in the original folder plan. It was added because the
research guide requires a conventional estimator next to the PINN.

### `src/pinn/`

| File | Content |
|---|---|
| `network.py` | `ConcentrationNet`: the tanh MLP with inputs `(x, y, t, A)`; `predict`, `predict_field`, `predict_dose_sensitivity` |
| `losses.py` | `make_loss_terms(net, pinned, release)` returns `terms(p, batch) -> [L_PDE, L_IC, L_BC, L_data]`; `make_batch` |
| `optimizer.py` | `TrainConfig` and `train`: Adam in compiled chunks, then L-BFGS |
| `train_forward.py` | `train_forward`, `forward_errors` |
| `train_inverse.py` | `train_inverse` (one optimisation), `train_staged` (baseline, treatment response, joint) |

`optimizer.py` is also an addition to the folder plan: it holds the loop that the
forward and the inverse trainers share.

### `src/identifiability/`

| File | Question it answers |
|---|---|
| `sensitivities.py` | How does each observation change with each parameter? `sensitivity_matrix` |
| `svd_analysis.py` | Which parameter combinations are weak? `svd_summary`, `sensitivity_correlations`, `collinearity_index` |
| `fisher_information.py` | What is the best achievable precision? `fisher_information`, `cramer_rao_bound`, `information_gain`, `derived_log_sd` |
| `profile_likelihood.py` | Do the data bound the parameter at all? `profile_likelihood` |
| `__init__.py` | `analyse_design`: all local diagnostics of one design in one call |

### `src/counterfactual/`

| File | Content |
|---|---|
| `treatment_model.py` | `TreatmentDesign` (training doses, held-out dose, validated), `potential_outcomes` |
| `counterfactual.py` | `treatment_effect`, `average_treatment_effect`, `cumulative_effect`, `global_treatment_effect`, `dose_sensitivity`, `counterfactual_effect`, `effect_errors` |
| `uncertainty_propagation.py` | `sample_posterior` (Metropolis through the solver), `sample_laplace`, `weak_direction_set`, `profile_set`, `select_compatible`, `propagate`, `reliability_summary`, `fit_spread`, `parameter_spread` |

### `src/utils/`

| File | Content |
|---|---|
| `sampling.py` | `scattered_design`, `electrode_design`, `observe`, `select_doses`, `merge`, `assert_heldout_excluded`, `collocation_points` |
| `metrics.py` | field errors, percent errors, summaries over realizations, interval coverage |
| `plotting.py` | one function per figure |
| `io.py` | JSON and `.npz` I/O, network weights, `run_metadata` |

---

## 4. A complete example

```python
import config as cfg
from src.model.parameters import true_params
from src.solvers.fd_solver import make_fd_solver
from src.solvers.reference_estimator import ReferenceEstimator
from src.utils.sampling import scattered_design, observe
from src.identifiability import analyse_design
from src.pinn.train_inverse import train_staged
from src.counterfactual.counterfactual import counterfactual_effect, effect_errors

truth, solver = true_params(), make_fd_solver()

# 1. what can this design determine?
design = scattered_design(cfg.TRAIN_DOSES, n_obs=400, seed=7001)
report = analyse_design(design, solver)
print(report['sigma_min'], report['sd'], report['derived_log_sd'])

# 2. one noise realization, fitted two ways
obs = observe(design, solver, truth, seed=7001)
reference = ReferenceEstimator(solver).fit(obs)
pinn = train_staged(obs)                                   # minutes on a GPU

# 3. predict the unseen dose with the calibrated parameters and score it
true_effect = counterfactual_effect(solver, truth, cfg.HELDOUT_DOSE)
predicted = counterfactual_effect(solver, pinn['params'], cfg.HELDOUT_DOSE)
print(effect_errors(predicted['tau'], true_effect['tau'], solver))
```

---

## 5. Conventions

* **Units**: um, ms, uM everywhere. No conversions inside the code.
* **Precision**: float64 is switched on when `src` is imported. Do not switch it off;
  the uptake parameters change the field by less than float32 resolves.
* **Randomness**: every random draw takes a seed. Observations depend only on
  `(seed, dose)`; the network and collocation points on `config.NET_SEED`.
* **Names**: functions say what they compute (`cumulative_effect`), arguments say what
  they are (`noise_sd`, `time_stride`). Short symbols are used only where they are the
  symbols of the paper (`C`, `D`, `A`, `tau`).
* **Errors**: a function raises a `ValueError` with the reason when it is asked for
  something that cannot be right (eta from untreated data, the held-out dose in a fit).
* **No hidden state**: nothing is cached in module globals. Reuse comes from passing
  the solver, the estimator or the network explicitly.

---

## 6. How to extend

**Another uptake law or a new term.** Edit `dopamine_pde.py` only. If the term has a
new parameter, add its name to `config.PARAM_NAMES`, a value to `TRUE_PARAMS`,
`INIT_PARAMS`, `PARAM_RANGES` and `PARAM_BOUNDS`, and, if it is confined to (0, 1), to
`parameters.LOGIT_PARAMS`. Then run `pytest`.

**A prescribed stimulation.** Pass `release=gaussian_stimulation(...)` (or any function
`R(x, y, t)`) to the solver, `observe`, `ReferenceEstimator` and `make_loss_terms`.

**A new observation design.** Write a function that returns the four design arrays;
`observe` and `analyse_design` then work unchanged. See `electrode_design`.

**A different network or schedule.** `ConcentrationNet(layers=..., normalize=...)` and
`TrainConfig(...)`; both are arguments of the trainers.

**Three dimensions.** The eigen-decomposition in `fd_solver.py` extends directly (add a
third matrix product per transform); the network needs a fifth input. The research
guide postpones this until the 2D study is stable.

---

## 7. Tests

```bash
python -m pytest -q        # 45 tests, about 30 s on a laptop CPU
```

| File | What it protects |
|---|---|
| `test_pde.py` | limits of the uptake term, the residual, the raw transforms |
| `test_fd_solver.py` | second order in space and time, conservation, limiting cases, exact sensitivities |
| `test_treatment.py` | the held-out rule, reproducible data, eta invisible at `A = 0`, the reference estimator |
| `test_counterfactual.py` | effect identities, dose sensitivity, uncertainty propagation, the posterior sampler |
| `test_pinn.py` | network derivatives, loss terms, frozen parameters, the three training stages |

The tests use a coarse grid so that they are fast. They check identities and
implementation correctness, not the scientific results.
