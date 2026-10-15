# Experiment guide

How to run the study, what each experiment answers, what it writes, and how to read
the output. Scripts 01 to 08 follow the seven-stage workflow of the research guide and
its experiment matrix E1 to E8; script 09 checks whether the PINN estimates come from
the data or from the starting point.

---

## 1. Before you start

### On Google Colab (A100)

Open [`notebooks/run_on_colab.ipynb`](../notebooks/run_on_colab.ipynb), select
**Runtime > Change runtime type > A100 GPU**, and run the cells from the top. The
notebook

1. installs the pinned JAX stack with CUDA support,
2. checks that JAX sees the GPU and that float64 is on,
3. links `results/` to a folder on your Google Drive, so that nothing is lost when the
   session disconnects,
4. runs the tests, then the experiments in order.

### On your own machine

```bash
pip install -r requirements.txt
python -m pytest -q                                   # about 30 s
python experiments/01_forward_verification.py --quick # smoke test, about 1 min
```

### Options shared by every script

| Option | Effect |
|---|---|
| `--quick` | tiny sizes everywhere. Checks that the script runs; **the numbers are not results**. Output goes to `results/_quick/`. |
| `--trial` | the paper grid, designs and network, but 10 000 collocation points, half-length training and three realizations (`TRIAL_*` in `config.py`). Checks the whole study in a fraction of the time before the final run; **the numbers are not results**. Output goes to `results/_trial/`. |
| `--n-realizations N` | use the first `N` noise realizations instead of the number in `config.py` |
| `--skip-pinn` | solver-based analysis only (no network training) |
| `--force` | recompute results that are already on disk |

### Runs are resumable

The inverse experiments write one JSON file per noise realization
(`results/inverse/<experiment>/seed_<seed>.json`). When a script is started again it
loads the files that exist and computes only the missing ones. After a Colab
disconnect, rerun the same cell.

### Runtime

Measured on a Colab A100 (October 2026) with the paper settings: one forward PINN run
3.3 minutes, one baseline inverse PINN run (experiment 03) about 3.2 minutes, one staged
multi-dose run (experiments 04, 05, 09) about 7.3 minutes. The PINN cost grows roughly
in proportion to the number of collocation points, each of which needs second
derivatives in float64. On a CPU (Apple M4 Pro) one solve takes 0.5 s, one Jacobian of a
four-dose design 7 s, one five-parameter reference fit 1 to 2 minutes.

Run the study once with `--trial` before the final run. It writes to `results/_trial/`,
so it never mixes with the results that are reported.

---

## 2. The order to run them in

```
01 forward verification      must pass before anything else
02 baseline identifiability  solver only; decides what to expect from 03
03 baseline inverse          PINN and reference estimator, A = 0
04 single treatment          does one treated dose add information?
05 multi-dose inverse        joint five-parameter recovery; stores calibrated models
06 held-out counterfactual   needs 05; predicts A = 0.5
07 counterfactual reliability   solver only
08 robustness                solver only by default
09 start sensitivity         PINN only; does the PINN estimate follow its starting point?
```

02, 04 (steps 1 and 2), 07 and 08 need no network training. Running them first, with
`--skip-pinn` where it applies, gives the identifiability picture of the whole study
before any PINN is trained. The research guide asks for exactly that order of thinking.

---

## 3. The experiments

### 01 Forward verification (Stage 1)

```bash
python experiments/01_forward_verification.py            # add --pilot for the architecture pilot
```

| | |
|---|---|
| **Question** | Are the reference solver and the PINN residual correct at every dose? |
| **Writes** | `results/forward/solver_verification.json`, `forward_pinn.json`, `pilot.json`; `results/figures/solver_convergence`, `fig2_forward_agreement`; `data/reference/fields_truth.npz` |
| **Read it as** | Orders close to 2 and limiting-case numbers close to 0 mean the solver is trustworthy. The forward PINN error per dose shows whether the nonlinear residual and the dose input work. |
| **Stop if** | the script prints `Solver verification FAILED`. It exits there on purpose. |

`--pilot` trains a handful of variants (normalised inputs, wider, deeper, larger
residual weight) and scores them on the training doses only. Choose the configuration
for the inverse runs from this table, set it in `config.py`, and then leave it alone.

### 02 Baseline identifiability (Stage 2, E1)

```bash
python experiments/02_baseline_identifiability.py
```

| | |
|---|---|
| **Question** | With untreated data only, which of `D`, `Vmax0`, `Km`, `k` can be separated? |
| **Writes** | `results/identifiability/baseline.json`; figures `baseline_identifiability`, `baseline_profiles` |
| **Read it as** | `sigma_min` and the condition number measure the weakest direction. Sensitivity correlations near +-1 mark parameters that act alike. The Cramer-Rao SD of a log-parameter is its best possible relative precision: 0.05 is 5 %, above 1 means not determined. The line "combinations" gives the same for `Vmax0/Km` and `Vmax0/Km + k`. A profile reported as `open` is not bounded by the data. |
| **Saturation sweep** | repeats the analysis for release amplitudes from `C0 << Km` to `C0 >> Km`, answering when `Vmax0` and `Km` are visible only through their ratio. |

### 03 Baseline inverse recovery (Stage 3, E1)

```bash
python experiments/03_baseline_inverse.py
```

| | |
|---|---|
| **Question** | Can the four baseline parameters be recovered, and does the PINN match the reference estimator? |
| **Writes** | `results/inverse/baseline/seed_*.json`, `baseline_summary.json`; `results/tables/baseline_recovery.csv`; figure `baseline_recovery` |
| **Read it as** | Compare the two methods on the same realizations. Where the reference estimator fails, the data do not contain the parameter. Where only the PINN fails, the optimisation is at fault. "on a bound" counts fits that ran to a limit of `PARAM_BOUNDS`. |

### 04 Treatment-response identification (Stage 4, E2)

```bash
python experiments/04_single_treatment.py
```

| | |
|---|---|
| **Question** | Does one treated condition add an independent sensitivity direction? |
| **Writes** | `results/identifiability/single_treatment.json`; `results/inverse/single_treatment/seed_*.json`; figures `fig3_identifiability`, `fig5_profiles` |
| **Read it as** | Each row is a design; the columns are Cramer-Rao SDs of the log-parameters with `eta` unknown, as a real fit would face it. Compare a treated design with the row `A = 0 (2x points)`: that control has the same number of points and no treatment, so only a gain beyond it is due to the intervention. The last columns repeat the comparison with `eta` assumed known, which is the most the intervention could add. |

### 05 Joint multi-dose refinement (Stage 5, E3)

```bash
python experiments/05_multidose_inverse.py
```

| | |
|---|---|
| **Question** | Do four doses, fitted jointly with shared baseline parameters, give practical five-parameter recovery? |
| **Writes** | `results/inverse/multidose/seed_*.json`, `weights_*.npz`, `multidose_summary.json`; `results/tables/multidose_recovery.csv`; figure `fig4_parameter_recovery` |
| **Read it as** | As for 03, now with `eta`. The coverage line reports how often the 95 % Laplace interval of the reference estimator contained the truth; values far below 0.95 mean the Gaussian approximation is not to be trusted for that parameter. |

### 06 Held-out counterfactual validation (Stage 6, E4)

```bash
python experiments/06_heldout_counterfactual.py
```

| | |
|---|---|
| **Question** | How well does the frozen, calibrated model predict the unseen dose `A = 0.5`? |
| **Needs** | experiment 05 |
| **Writes** | `results/counterfactual/heldout.json`; `results/tables/heldout_counterfactual.csv`; figure `fig6_heldout_counterfactual` |
| **Read it as** | Three predictors are scored: the network itself (`pinn_network`), the solver with the PINN parameters (`pinn_parameters`), and the solver with the reference-estimator parameters (`reference`). `field` is the error of `C^(0.5)`; the `tau` and `GTE` errors concern the effect, which is a small difference of two fields and therefore much harder. |

### 07 Counterfactual reliability (Stage 7, E5)

```bash
python experiments/07_counterfactual_reliability.py
```

| | |
|---|---|
| **Question** | Do parameter vectors that fit the data equally well agree on the effect at the unseen dose? |
| **Writes** | `results/counterfactual/reliability.json`; figure `fig7_counterfactual_uncertainty` |
| **Method** | For each design the posterior of all five parameters is sampled by Metropolis through the solver, with a flat prior on `PARAM_RANGES`. Every sample is turned into an effect `tau_0.5`. |
| **Read it as** | "spread of log parameter" is how uncertain the parameters and their combinations are. "GTE interval width / true GTE" is how uncertain the effect is. Compare the two across designs: this is the central result of the paper. Check `R-hat` first; above 1.1 the chains have not converged and `MCMC_STEPS` must be raised. |
| **Note** | With `A = 0` only, `eta` returns its prior range. The band of that design shows what untreated data alone say about a treatment. The prior ranges are an assumption and must be stated with the result. |

### 08 Robustness (E6, E7, E8)

```bash
python experiments/08_robustness.py --part noise        # or sparsity, electrode, truths, mismatch, all
```

| Part | Question |
|---|---|
| `noise` | Does the information gain survive more noise? |
| `sparsity` | ... and fewer observations? |
| `electrode` | Does it persist with FSCV-like sampling at one to three sites? (`--with-pinn` also trains the PINN) |
| `truths` | Is the conclusion specific to the chosen truth, or general over the parameter ranges? |
| `mismatch` | What happens when the fitted model ignores a release, or holds `k` at a wrong value? The control `k_true` holds `k` at its true value. |

Writes `results/identifiability/robustness_<part>.json` and
`results/tables/robustness_<part>.csv`. Each row gives the local diagnostics and the
recovery of the two rate combinations and of the global effect over noise realizations.

### 09 Start sensitivity of the PINN

```bash
python experiments/09_start_sensitivity.py --trial    # three realizations x three starts
```

| | |
|---|---|
| **Question** | Vmax0 and Km are not separately identifiable (experiments 02 and 03), yet the PINN returns them close to the truth. Is that the data, or the starting point, which lies 20 % from the truth? |
| **Does** | The staged PINN of experiment 05 from three starts that share the uptake ratio Vmax0 / Km but differ by a factor 16 in Vmax0 and Km (`START_FACTORS` in `config.py`). |
| **Writes** | `results/inverse/start_sensitivity/<start>_seed_<seed>.json`, `results/tables/start_sensitivity.csv` |
| **Read it as** | The printed "spread across starts" is the ratio of the largest to the smallest final value for one realization. Near 1: the start does not matter. Near 16 for Vmax0 and Km, with the same data misfit: the PINN keeps its start along the flat direction and does not identify Vmax0 and Km (decision rule 3). Only the combinations can then be reported. |

---

## 4. Figures of the manuscript

| Figure | Produced by | File in `results/figures/` |
|---|---|---|
| 1 workflow schematic | drawn by hand | - |
| 2 forward agreement across doses | 01 | `fig2_forward_agreement` |
| 3 sensitivity spectra and correlation maps | 04 | `fig3_identifiability` |
| 4 parameter recovery over realizations | 05 | `fig4_parameter_recovery` |
| 5 profile geometry and its change under intervention | 04 | `fig5_profiles` |
| 6 held-out field and effect maps | 06 | `fig6_heldout_counterfactual` |
| 7 counterfactual uncertainty | 07 | `fig7_counterfactual_uncertainty` |

Each figure is written as PNG and PDF.

---

## 5. Decision rules (from the research guide)

1. If the forward solver is not converged, do not start inverse experiments.
   *Enforced: experiment 01 exits.*
2. If baseline sensitivities show near-exact confounding, document it before changing
   the model. Do not hide it by tuning the PINN.
3. If the reference estimator cannot recover a parameter under a design, do not expect
   the PINN to.
4. If treated data improve the PINN estimates but not the reference-estimator geometry,
   it is an optimisation effect, not restored identifiability.
5. If the `A = 0.5` error stays large although the training doses fit well, look at
   parameter uncertainty and model mismatch before enlarging the network.
6. Real FSCV data and 3D come only after the synthetic study is stable.

## 6. Claims to avoid when writing up

* Synthetic mechanistic contrasts are not experimentally identified causal effects.
* Intervention does not automatically guarantee identifiability.
* Michaelis-Menten kinetics alone is not the novelty.
* A low PINN field error is not proof of parameter identifiability.
* The held-out `A = 0.5` field must not be used for tuning, model selection or
  loss-weight selection. *Enforced in code for fitting; for tuning it is your discipline:
  the pilot of experiment 01 scores on the training doses only.*
