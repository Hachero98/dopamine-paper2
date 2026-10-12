# Data

Everything in this folder is **synthetic** and is regenerated from `config.py` by the
code. Nothing here is measured data, and the generated files are not tracked by git.

| Folder | Content | Written by |
|---|---|---|
| `reference/` | `fields_truth.npz`: reference fields `C_true(x, y, t; A)` at the true parameters for the five doses `A = 0, 0.25, 0.5, 0.75, 1` | `experiments/01_forward_verification.py` |
| `observations/` | Optional exports of observation sets (see below) | you, with `src.utils.io.save_arrays` |

## `reference/fields_truth.npz`

| Array | Shape | Meaning |
|---|---|---|
| `doses` | (5,) | treatment levels `A` |
| `x` | (81,) | node coordinates along one side, um (the grid is `x` by `x`) |
| `t` | (101,) | frame times, ms (every 0.5 ms) |
| `C` | (5, 101, 81, 81) | concentration, uM, indexed `[dose, time, ix, iy]`, float32 |
| `param_*` | scalars | the parameters the fields were computed with |

The file is a record for inspection and plotting. The experiments do not read it: they
call the solver, which takes about a second, so results never depend on a stale file.

## Observation sets

Observations are not stored because they are a pure function of
`(design, parameters, noise SD, seed)`:

```python
from src.model.parameters import true_params
from src.solvers.fd_solver import make_fd_solver
from src.utils.sampling import scattered_design, observe

solver = make_fd_solver()
design = scattered_design(doses=(0.0, 0.25, 0.75, 1.0), n_obs=400, seed=7001)
obs = observe(design, solver, true_params(), noise_sd=0.02, seed=7001)
```

`obs` is a dict of equal-length arrays:

| Key | Unit | Meaning |
|---|---|---|
| `x_d`, `y_d` | um | location |
| `t_d` | ms | time |
| `A_d` | - | dose of the experiment the point belongs to |
| `C_clean` | uM | solver value without noise |
| `C_d` | uM | observed value, `C_clean` plus Gaussian noise |

## The held-out dose

The field at `A = 0.5` exists in `reference/` for scoring only. No observation at
`A = 0.5` may enter a fit, a tuning run or a model-selection step. The code enforces
this: `src.utils.sampling.assert_heldout_excluded` is called by every training entry
point and raises an error.
