"""Central configuration of the Paper 2 study.

Every number that defines the study lives in this file: the synthetic truth, the
treatment levels, the observation designs, the network, the training schedule and the
reference-solver resolution. Functions in ``src/`` take these values as keyword
defaults, so an experiment changes a setting by passing an argument and never by
editing library code.

Units are fixed throughout the repository:

    length         micrometre (um)
    time           millisecond (ms)
    concentration  micromolar (uM)

See ``docs/PARAMETER_GUIDE.md`` for the meaning and the source of each value.
"""
from pathlib import Path

# =====================================================================================
# 1. Mechanistic model
#
#    C_t = D lap(C) + R(x, y, t) - Vmax0 (1 - eta A) C / (Km + C) - k C     in Omega
#    grad(C) . n = 0                                                        on the walls
#    C(x, y, 0) = C0 exp(-(x^2 + y^2) / (2 sigma^2))
# =====================================================================================

# ---- Domain and release (identical to Paper 1) --------------------------------------
L = 5.0        # side of the square domain Omega = (-L/2, L/2)^2, um
T = 50.0       # record length, ms (about one linear reuptake time constant)
SIGMA = 0.5    # width of the Gaussian release, um
C0 = 1.0       # peak release concentration, uM

# ---- Synthetic truth -----------------------------------------------------------------
# Computational design values for the synthetic study, not patient-specific values.
# D and k are the Paper 1 operating point; Vmax0 / Km = 0.020 1/ms equals the Paper 1
# linearised DAT rate, so the low-concentration limit of the two papers coincides.
PARAM_NAMES = ('D', 'Vmax0', 'Km', 'k', 'eta')   # order of every parameter vector
BASELINE_PARAM_NAMES = PARAM_NAMES[:4]           # parameters that act at A = 0

TRUE_PARAMS = {
    'D': 0.32,       # effective diffusion coefficient, um^2/ms
    'Vmax0': 0.004,  # untreated DAT uptake capacity, uM/ms
    'Km': 0.20,      # Michaelis-Menten half-saturation concentration, uM
    'k': 0.020,      # additional first-order clearance, 1/ms
    'eta': 0.60,     # treatment efficacy: fraction of Vmax0 removed at full dose (A = 1)
}

# ---- Ranges used by the robustness sweeps (and as estimator bounds) -----------------
PARAM_RANGES = {
    'D': (0.16, 0.50),
    'Vmax0': (0.002, 0.008),
    'Km': (0.10, 0.40),
    'k': (0.005, 0.040),
    'eta': (0.20, 0.80),
}

# Hard bounds of the reference estimator, deliberately wider than PARAM_RANGES so that
# an estimate sitting on a bound signals a non-identifiable direction, not a tight box.
PARAM_BOUNDS = {
    'D': (0.02, 2.0),
    'Vmax0': (1e-5, 0.5),
    'Km': (1e-3, 50.0),
    'k': (1e-4, 0.5),
    'eta': (1e-3, 1.0 - 1e-3),
}

# Offset initial guesses of the inverse problems (never the truth).
INIT_PARAMS = {
    'D': 0.30,       # -6.25 %
    'Vmax0': 0.0032, # -20 %
    'Km': 0.24,      # +20 %
    'k': 0.016,      # -20 %
    'eta': 0.50,     # centre of the admissible interval
}

# =====================================================================================
# 2. Treatment design
# =====================================================================================
TRAIN_DOSES = (0.0, 0.25, 0.75, 1.0)   # doses whose observations may enter a fit
HELDOUT_DOSE = 0.5                     # never used for fitting, tuning or model selection
ALL_DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
SINGLE_TREATED_DOSES = (0.25, 0.75, 1.0)   # the "baseline + one treated dose" designs (E2)

# =====================================================================================
# 3. Observation designs
# =====================================================================================
NOISE_SD = 0.02             # Gaussian noise SD, uM (2 % of C0)
T_WINDOW = (0.5, 0.9 * T)   # observation window, ms (Paper 1)

# Design I: scattered space-time points (continuity with Paper 1).
N_OBS = 400                 # observations per treatment level

# Design II: FSCV-like virtual electrodes, dense in time at fixed sites.
ELECTRODE_DT = 0.5          # sampling interval, ms
ELECTRODE_DESIGNS = {
    'S1_r1.0': [(1.0, 0.0)],
    'S1_r2.0': [(2.0, 0.0)],
    'S3': [(1.0, 0.0), (0.0, 2.0), (-2.5 / 2**0.5, -2.5 / 2**0.5)],
}

# Sweeps of experiment 08.
NOISE_SWEEP = (0.0, 0.01, 0.02, 0.05, 0.10)   # noise SD, uM
N_OBS_SWEEP = (100, 200, 400, 800, 1600)      # observations per treatment level

# Independent noise realizations behind every headline number. A single favourable
# realization is not evidence (Paper 1, Section 3.3).
OBS_SEEDS = tuple(range(7001, 7021))   # 20 realizations
N_REALIZATIONS_PINN = 10               # PINN runs use the first N of OBS_SEEDS

# =====================================================================================
# 4. Finite-difference reference solver
# =====================================================================================
NX_FD = 81        # nodes per side (dx = 0.0625 um)
DT_FD = 0.02      # time step of the IMEX-BDF2 integrator, ms. Time-stepping error is below
                  # 1e-4 uM inside T_WINDOW (0.5 % of NOISE_SD); see docs/PARAMETER_GUIDE.md
DT_STORE = 0.1    # interval between stored frames, ms

# =====================================================================================
# 5. PINN (Paper 1 network with the dose A as a fourth input)
# =====================================================================================
LAYERS = (4, 64, 64, 64, 64, 1)   # inputs (x, y, t, A); four hidden layers of 64 tanh units
NORMALIZE_INPUTS = False          # False = raw inputs as in Paper 1; True maps each input to [-1, 1]

N_DOMAIN = 30_000      # interior collocation points, Latin hypercube in (x, y, t, A)
N_INITIAL = 800        # initial-condition points
N_BOUNDARY = 2_000     # wall points
COLLOCATION_DOSES = 'continuous'   # 'continuous': A ~ U[0, 1]; or a tuple of discrete doses

ADAM_ITERS = 20_000
LBFGS_ITERS = 2_000
LR = 1e-3
N_CHUNK = 50           # Adam steps per jitted chunk

# Loss weights (w_f, w_i, w_b, w_d) of L = w_f L_PDE + w_i L_IC + w_b L_BC + w_d L_data.
# Paper 1 values, used as the starting point of the nonlinear pilot (experiment 01).
W_INVERSE = (3000.0, 1.0, 100.0, 5.0)
W_FORWARD = (10.0, 1.0, 1.0, 0.0)

NET_SEED = 1234        # network initialisation and collocation seed

# Staged inverse training: (Adam iterations, L-BFGS iterations) of each stage.
#   baseline             A = 0 data; learns D, Vmax0, Km, k
#   treatment_response   all training doses; learns eta with the baseline parameters frozen
#   joint                all training doses; refines the five parameters together
STAGE_SCHEDULE = {
    'baseline': (20_000, 2_000),
    'treatment_response': (10_000, 1_000),
    'joint': (20_000, 2_000),
}

# ---- Trial run (option --trial) -------------------------------------------------------
# A reduced run that checks the whole pipeline end to end in a fraction of the time
# before the final run. Only the PINN cost and the number of realizations shrink; the
# solver grid, the observation designs, the network and the noise are the paper values,
# and the MCMC keeps its full length because its convergence is one of the checks.
# Trial numbers show that the study works and which way the results go. They are not
# results: fewer collocation points and shorter training change the PINN errors.
TRIAL_COLLOCATION = dict(n_domain=10_000, n_initial=400, n_boundary=1_000)
TRIAL_ADAM_ITERS = 10_000      # single-stage PINN runs (experiments 01 and 03)
TRIAL_LBFGS_ITERS = 1_000
TRIAL_STAGE_SCHEDULE = {       # half of STAGE_SCHEDULE
    'baseline': (10_000, 1_000),
    'treatment_response': (5_000, 500),
    'joint': (10_000, 1_000),
}
TRIAL_N_REALIZATIONS = 3       # for the reference estimator and the PINN

# ---- Sensitivity of the PINN estimate to its starting point (experiment 09) ------------
# Each start keeps the uptake ratio Vmax0 / Km of INIT_PARAMS and moves Vmax0 and Km
# together by a factor. The low-concentration data barely distinguish these starts, so
# if the final Vmax0 and Km follow the start, the PINN is not identifying them.
START_FACTORS = {'offset': 1.0, 'Km_x4': 4.0, 'Km_div4': 0.25}

# =====================================================================================
# 6. Identifiability and uncertainty
# =====================================================================================
PROFILE_POINTS = 15            # grid points of each profile likelihood
PROFILE_MAX_EVALUATIONS = 40   # cap on solver calls of each constrained fit along a profile
PROFILE_SPAN = 4.0             # profile grid covers a factor of PROFILE_SPAN either side of the optimum
CONFIDENCE = 0.95              # level of profile-likelihood and posterior intervals
N_POSTERIOR_SAMPLES = 200      # parameter samples propagated into the counterfactual effect

# Metropolis sampling of the posterior through the reference solver (experiment 07).
# The posterior is flat along the weak directions, so a random walk mixes slowly: check the
# R-hat values that experiment 07 prints and raise MCMC_STEPS if any exceeds about 1.1.
# 2 000 steps per chain gave R-hat 1.2 to 1.6 for k, so the chains are four times longer.
MCMC_CHAINS = 16               # parallel chains, advanced by one batched solver call per step
MCMC_STEPS = 8_000             # steps per chain
MCMC_BURN_IN = 2_000           # steps discarded at the start of every chain (step size is tuned here)

# =====================================================================================
# 7. Paths
# =====================================================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
REFERENCE_DIR = DATA_DIR / 'reference'
OBSERVATION_DIR = DATA_DIR / 'observations'
RESULTS_DIR = ROOT / 'results'
