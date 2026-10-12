# Dopamine Paper 2

**Intervention-induced identifiability and mechanistic counterfactual inference in
nonlinear spatial dopamine dynamics using inverse physics-informed neural networks.**

Code of the second paper of the series. It extends the Paper 1 code base
([Hachero98/dopamine-PINN-](https://github.com/Hachero98/dopamine-PINN-)) from a linear
sink to Michaelis-Menten DAT uptake under a pharmacological intervention, and asks
three questions in order:

1. **Identifiability.** From sparse, noisy observations, which of the biological
   parameters can be determined at all?
2. **Intervention.** Does observing the system under a DAT-targeting treatment add the
   missing information?
3. **Counterfactual reliability.** Is that information enough to predict the effect of
   a treatment level that was never observed?

Implemented in JAX (Flax NNX, Optax, jaxopt), float64 throughout, written to run on a
Google Colab A100.

---

## The model

Dopamine concentration `C(x, y, t)` on the square `(-2.5, 2.5)^2 um^2` for `0 < t <= 50 ms`:

$$
\frac{\partial C}{\partial t} = D\,\nabla^2 C + R(x,y,t)
 - \frac{V_{\max,0}\,(1-\eta A)\,C}{K_m + C} - k\,C ,
\qquad \nabla C\cdot n = 0 \ \text{on the walls},
$$

with the Gaussian release of Paper 1 as initial condition and `R = 0` afterwards.

| Symbol | Meaning | Synthetic truth |
|---|---|---|
| `D` | effective diffusion coefficient | 0.32 um^2/ms |
| `Vmax0` | DAT uptake capacity without treatment | 0.004 uM/ms |
| `Km` | half-saturation concentration | 0.20 uM |
| `k` | additional first-order clearance | 0.020 1/ms |
| `eta` | treatment efficacy | 0.60 |
| `A` | normalised dose, 0 (none) to 1 (full) | 0, 0.25, 0.75, 1 observed; **0.5 held out** |

The treatment acts on one term only: `Vmax(A) = Vmax0 (1 - eta A)`. The counterfactual
effect of a dose `a` is `tau_a = C^(a) - C^(0)`. These are mechanistic contrasts of a
synthetic model, not experimentally identified causal effects.

---

## Repository layout

```
dopamine-paper2/
├── config.py                 every number of the study, in one place
├── requirements.txt
├── src/
│   ├── model/                the PDE terms, the parameter vector, the initial condition
│   ├── solvers/              finite-difference reference solver, its verification,
│   │                         and the reference estimator (least squares through the solver)
│   ├── pinn/                 network C(x, y, t, A), losses, optimiser, forward and staged inverse training
│   ├── identifiability/      sensitivities, SVD, Fisher information, profile likelihood
│   ├── counterfactual/       treatment design, effects and summaries, uncertainty propagation
│   └── utils/                observation designs and collocation, metrics, figures, file I/O
├── experiments/              01 ... 08: one script per stage of the study
├── notebooks/run_on_colab.ipynb
├── tests/                    45 tests, about 30 s on a CPU
├── data/                     generated reference fields (not tracked)
├── results/                  JSON, tables and figures written by the experiments
└── docs/
    ├── CODE_GUIDE.md         how the code is organised and how to extend it
    ├── EXPERIMENT_GUIDE.md   how to run each experiment and read its output
    └── PARAMETER_GUIDE.md    every parameter: meaning, value, source
```

---

## Quick start

### Google Colab (A100)

Open `notebooks/run_on_colab.ipynb`, choose an A100 runtime and run the cells from the
top. The notebook installs the pinned stack, checks the GPU, links `results/` to your
Google Drive and runs the study in order.

### Local

```bash
pip install -r requirements.txt       # CPU; on a GPU install jax[cuda12]==0.10.2 instead of jax
python -m pytest -q                   # about 30 s
python experiments/01_forward_verification.py --quick   # smoke test, about 1 min
```

No installation step is needed: run everything from the repository root.

### In Python

```python
import config as cfg
from src.model.parameters import true_params
from src.solvers.fd_solver import make_fd_solver
from src.utils.sampling import scattered_design
from src.identifiability import analyse_design

solver = make_fd_solver()
field = solver(true_params(), 0.5)                       # C(x, y, t) at dose A = 0.5

design = scattered_design(cfg.TRAIN_DOSES, n_obs=400, seed=7001)
report = analyse_design(design, solver)                  # what can this design determine?
print(report['sigma_min'], report['sd'])
```

---

## The study

| # | Script | Stage of the research guide | Needs the PINN |
|---|---|---|---|
| 01 | `01_forward_verification.py` | 1. Solver convergence, limiting cases, forward PINN at all doses | optional |
| 02 | `02_baseline_identifiability.py` | 2. What untreated data can separate | no |
| 03 | `03_baseline_inverse.py` | 3. Recovery of `(D, Vmax0, Km, k)` at `A = 0` | yes |
| 04 | `04_single_treatment.py` | 4. Information added by one treated dose; `eta` | optional |
| 05 | `05_multidose_inverse.py` | 5. Joint five-parameter recovery from four doses | yes |
| 06 | `06_heldout_counterfactual.py` | 6. Prediction of the unseen dose `A = 0.5` | uses 05 |
| 07 | `07_counterfactual_reliability.py` | 7. Uncertainty of the effect under parameter ambiguity | no |
| 08 | `08_robustness.py` | Noise, sparsity, electrode geometry, other truths, model mismatch | optional |

Every script accepts `--quick` (smoke test), `--n-realizations N`, `--skip-pinn` and
`--force`, and resumes from the results already on disk. Details, outputs and how to
read them: [docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md).

### Rules built into the code

* **The held-out dose never enters a fit.** Every training entry point rejects data at
  `A = 0.5`.
* **`eta` cannot be estimated from untreated data.** At `A = 0` the model does not
  depend on it; the estimators refuse instead of returning a meaningless number.
* **One set of baseline parameters for all doses.** There is no per-dose copy of `D`,
  `Vmax0`, `Km` or `k`.
* **No inverse experiment on an unconverged solver.** Experiment 01 exits if the
  verification fails.
* **Every headline number comes from repeated noise realizations**, and every PINN
  estimate stands next to the reference estimator on the same data.

---

## Relation to Paper 1

Paper 2 is a controlled extension of the Paper 1 package `dopamine_pinn`, not a rewrite
of its methods.

| Kept from Paper 1 | Changed for Paper 2 |
|---|---|
| Domain, zero-flux walls, Gaussian release, 81 x 81 node-centred grid | Sink: Michaelis-Menten uptake plus `k C`, with `Vmax(A) = Vmax0 (1 - eta A)` |
| tanh network with four hidden layers of 64 units, float64 | Fourth network input: the dose `A` |
| Latin-hypercube collocation, Adam then L-BFGS, loss weights as the starting point | Collocation in `(x, y, t, A)`; staged training: baseline, treatment response, joint |
| Positive parameters learned as logarithms | `eta` learned as a logit; five physical parameters |
| Scattered and electrode-like observation designs, 2 % noise, independent realizations | Multi-dose designs; data of dose `A` depend only on `(seed, A)` |
| Reference estimator: least squares through the solver, Laplace covariance | Extended to five parameters and several doses |
| Differentiable JAX finite-difference solver | Time integration changed from explicit Euler to second-order IMEX-BDF2, as the research guide specifies |
| | New: sensitivities, SVD, Fisher information, profile likelihood |
| | New: counterfactual effects, posterior sampling, uncertainty propagation |

The code was reorganised from one package into the folder plan of this repository.
Two files were added to that plan because the research guide requires them:
`src/solvers/reference_estimator.py` and `src/pinn/optimizer.py`.

---

## Status

**Verified here** (laptop CPU):

* the test suite passes (45 tests);
* the reference solver converges with observed order 2.0 in space and 2.0 in time and
  passes the limiting-case checks;
* all eight experiments run end to end in `--quick` mode.

**Not yet run:** the full-size experiments. They need the GPU; no result of the paper
exists yet, and `results/` is empty apart from its folders. The network architecture
and loss weights are the Paper 1 settings and still have to go through the pilot of
experiment 01.

**A first diagnostic to be aware of.** A solver-only check at full resolution (one
design, 400 scattered points per dose, 2 % noise, evaluated at the true parameters)
gives these Cramer-Rao standard deviations of the log-parameters, where 0.05 means 5 %
and values above 1 mean the parameter is not determined:

| Design | `D` | `Vmax0` | `Km` | `k` | `eta` | `Vmax0/Km + k` |
|---|---|---|---|---|---|---|
| `A = 0` | 0.11 | 40 | 27 | 15 | - | 0.67 |
| `A = 0`, twice the points | 0.07 | 25 | 16 | 9.1 | - | 0.44 |
| `A = 0` and `A = 1` | 0.07 | 10 | 7.6 | 3.2 | 10 | 0.37 |
| all four training doses | 0.04 | 8.4 | 6.5 | 2.6 | 8.1 | 0.31 |
| all four doses, `eta` known | 0.04 | 2.1 | 2.5 | 0.12 | - | 0.22 |

At the default design only `D` is determined. At an equal number of points, replacing
half of the untreated observations by a treated dose improves the weak directions by a
factor of about 2.5 (rows 2 and 3), and treatment separates `k` from DAT uptake if
`eta` is known (last row). It does not make `Vmax0`, `Km`, `k` and `eta` individually
identifiable.
The research guide anticipates this case ("equally publishable negative result") and
experiments 02, 04, 07 and 08 are built to characterise it: which combinations of
parameters, and which summaries of the treatment effect, remain determined. This is
one local diagnostic from one design draw, not a result of the paper.

---

## Citation

Paper 1, on which this code builds:

```bibtex
@article{Hackman2026Dopamine,
  author  = {Hackman, Emmanuel and Zhu, Huiqing},
  title   = {Inverse Physics-Informed Neural Networks for Synaptic Dopamine Transport:
             Parameter Recovery at the Information Limit of Sparse, Noisy Data},
  journal = {Medical \& Biological Engineering \& Computing},
  year    = {2026},
  note    = {Submitted}
}
```

## Author

Emmanuel Hackman, School of Mathematics and Natural Sciences, University of Southern
Mississippi.

## License

[MIT](LICENSE).
