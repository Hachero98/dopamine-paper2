# Parameter guide

Every number that defines the study, what it means, and where it comes from. All of
them live in [`config.py`](../config.py).

Units: length in um, time in ms, concentration in uM.

---

## 1. The model

```
C_t = D lap(C) + R(x, y, t) - Vmax0 (1 - eta A) C / (Km + C) - k C      in the square
grad(C) . n = 0                                                         on the walls
C(x, y, 0) = C0 exp(-(x^2 + y^2) / (2 sigma^2))
```

## 2. Physical parameters (the unknowns)

| Name in code | Symbol | True value | Unit | Meaning |
|---|---|---|---|---|
| `D` | D | 0.32 | um^2/ms | effective diffusion coefficient in the extracellular space |
| `Vmax0` | Vmax,0 | 0.004 | uM/ms | maximum DAT uptake rate without treatment |
| `Km` | Km | 0.20 | uM | concentration at which DAT uptake is half of its maximum |
| `k` | k | 0.020 | 1/ms | additional first-order clearance |
| `eta` | eta | 0.60 | - | treatment efficacy: fraction of `Vmax0` removed at full dose |

`lambda0 = (D, Vmax0, Km, k)` are the baseline parameters; `lambda` adds `eta`.

**Where the values come from.** `D` and `k` are the Paper 1 operating point. `Km` and
`eta` are the values chosen in the research guide. `Vmax0 = 0.004` makes
`Vmax0 / Km = 0.020 1/ms`, the linearised DAT rate of Paper 1, so the two papers agree
in the low-concentration limit. These are design values of a synthetic study, not
patient-specific values.

**A consequence worth knowing.** With these values the sink at low concentration has
the rate `Vmax0/Km + k = 0.040 1/ms`: twice the Paper 1 rate, because the model keeps
both the Michaelis-Menten term and `k C` as the research guide specifies.

### Combinations (`parameters.DERIVED`)

| Name | Formula | True value | Why it matters |
|---|---|---|---|
| `uptake_ratio` | `Vmax0 / Km` | 0.020 1/ms | what the data see of DAT uptake when `C << Km` |
| `baseline_rate` | `Vmax0 / Km + k` | 0.040 1/ms | total low-concentration clearance at `A = 0` |
| `treatment_rate` | `eta Vmax0 / Km` | 0.012 1/ms | clearance removed by a full dose |

When concentrations are mostly below `Km`, the data constrain these combinations much
better than the individual parameters. The experiments report both.

### How the optimisers see the parameters

| Parameter | Constraint | Raw variable |
|---|---|---|
| `D`, `Vmax0`, `Km`, `k` | positive | logarithm |
| `eta` | between 0 and 1 | logit, `log(eta / (1 - eta))` |

### Three sets of numbers per parameter

| Dict | Purpose |
|---|---|
| `INIT_PARAMS` | starting point of every inverse problem; offset from the truth by 6 to 20 % |
| `PARAM_RANGES` | robustness sweeps, and the flat prior of the posterior sampler |
| `PARAM_BOUNDS` | hard limits of the reference estimator; deliberately much wider |

| Parameter | Initial guess | Range | Bounds |
|---|---|---|---|
| `D` | 0.30 | 0.16 to 0.50 | 0.02 to 2.0 |
| `Vmax0` | 0.0032 | 0.002 to 0.008 | 1e-5 to 0.5 |
| `Km` | 0.24 | 0.10 to 0.40 | 1e-3 to 50 |
| `k` | 0.016 | 0.005 to 0.040 | 1e-4 to 0.5 |
| `eta` | 0.50 | 0.20 to 0.80 | 0.001 to 0.999 |

An estimate that ends on a *bound* is reported (`at_bound`). It means the data did not
stop the optimiser, which is a sign of non-identifiability and not a numerical failure.

## 3. Domain and release (unchanged from Paper 1)

| Name | Value | Meaning |
|---|---|---|
| `L` | 5.0 um | side of the square domain, centred on the release site |
| `T` | 50 ms | record length, about one linear reuptake time constant |
| `SIGMA` | 0.5 um | width of the Gaussian release |
| `C0` | 1.0 uM | peak concentration of the release |

`R = 0` after the initial pulse in the primary experiment. A prescribed stimulation is
used only in the mismatch part of experiment 08.

## 4. Treatment design

| Name | Value | Meaning |
|---|---|---|
| `TRAIN_DOSES` | 0, 0.25, 0.75, 1 | doses whose observations may enter a fit |
| `HELDOUT_DOSE` | 0.5 | predicted without data; never used for fitting or tuning |
| `ALL_DOSES` | 0, 0.25, 0.5, 0.75, 1 | doses with a reference field |
| `SINGLE_TREATED_DOSES` | 0.25, 0.75, 1 | "baseline plus one treated dose" designs |

`A` is a normalised dose: `A = 0` is no intervention and `A = 1` reduces the DAT
capacity to `Vmax0 (1 - eta)`.

## 5. Observations

| Name | Value | Meaning |
|---|---|---|
| `NOISE_SD` | 0.02 uM | SD of the Gaussian noise: 2 % of `C0` |
| `T_WINDOW` | 0.5 to 45 ms | observation times (Paper 1) |
| `N_OBS` | 400 | scattered observations **per dose** |
| `ELECTRODE_DT` | 0.5 ms | sampling interval of a virtual electrode |
| `ELECTRODE_DESIGNS` | `S1_r1.0`, `S1_r2.0`, `S3` | one site at 1 um, one at 2 um, three sites |
| `OBS_SEEDS` | 7001 to 7020 | the 20 independent noise realizations |
| `N_REALIZATIONS_PINN` | 10 | how many of them the PINN is trained on |
| `NOISE_SWEEP`, `N_OBS_SWEEP` | see `config.py` | sweeps of experiment 08 |

`N_OBS` is per dose, so a four-dose design has 1600 points. More doses therefore also
means more data. Experiment 04 includes a control with the same number of points and
no treatment, to keep the two effects apart.

## 6. Reference solver

| Name | Value | Meaning |
|---|---|---|
| `NX_FD` | 81 | grid nodes per side (`dx = 0.0625 um`) |
| `DT_FD` | 0.02 ms | time step of the IMEX-BDF2 integrator |
| `DT_STORE` | 0.1 ms | interval between stored frames |

Measured on this grid at the true parameters, against a run with `dt = 0.0025 ms`:

| `dt` (ms) | max error, whole record (uM) | max error for `t >= 0.5 ms` (uM) |
|---|---|---|
| 0.05 | 9.4e-3 | 5.4e-4 |
| 0.02 | 1.6e-3 | 8.1e-5 |
| 0.01 | 3.5e-4 | 1.9e-5 |

The largest error sits in the first half millisecond, while the narrow pulse spreads.
Inside the observation window `dt = 0.02` gives an error of 0.4 % of the noise SD.
Observed convergence orders are 2.0 in space and 2.0 in time
(`python -m src.solvers.convergence_test`).

## 7. PINN

| Name | Value | Meaning |
|---|---|---|
| `LAYERS` | (4, 64, 64, 64, 64, 1) | inputs `(x, y, t, A)`, four hidden layers of 64 tanh units |
| `NORMALIZE_INPUTS` | False | raw inputs as in Paper 1; `True` maps each input to [-1, 1] |
| `N_DOMAIN` | 30 000 | interior collocation points (Latin hypercube) |
| `N_INITIAL` | 800 | initial-condition points |
| `N_BOUNDARY` | 2 000 | wall points |
| `COLLOCATION_DOSES` | `'continuous'` | the PDE is enforced for `A` drawn uniformly from [0, 1] |
| `ADAM_ITERS`, `LBFGS_ITERS` | 20 000, 2 000 | Paper 1 schedule |
| `LR` | 1e-3 | Adam learning rate |
| `W_INVERSE` | (3000, 1, 100, 5) | weights of the PDE, initial, wall and data terms |
| `W_FORWARD` | (10, 1, 1, 0) | the same for the forward problem |
| `STAGE_SCHEDULE` | see `config.py` | Adam and L-BFGS iterations of the three stages |
| `NET_SEED` | 1234 | network initialisation and collocation seed |

**These are starting values.** The architecture and the loss weights are the Paper 1
settings for a three-input network and a linear sink. The research guide asks for a
small pilot on the nonlinear problem before they are frozen:
`python experiments/01_forward_verification.py --pilot`.

**Why the PDE is enforced at every dose.** The PDE is known for all `A`, so residual
points at `A = 0.5` use no held-out *data*. They are what allows one network to
interpolate between the training doses.

## 8. Identifiability and uncertainty

| Name | Value | Meaning |
|---|---|---|
| `PROFILE_POINTS` | 15 | grid points of a profile likelihood |
| `PROFILE_SPAN` | 4 | profiles cover a factor of 4 either side of the centre |
| `PROFILE_MAX_EVALUATIONS` | 40 | cap on solver calls of each constrained fit |
| `CONFIDENCE` | 0.95 | level of all intervals and bands |
| `N_POSTERIOR_SAMPLES` | 200 | parameter vectors propagated into the effect |
| `MCMC_CHAINS`, `MCMC_STEPS`, `MCMC_BURN_IN` | 16, 2000, 500 | Metropolis sampler of experiment 07; raise the steps if R-hat exceeds about 1.1 |
