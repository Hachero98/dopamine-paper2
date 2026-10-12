"""Shared set-up of the experiment scripts.

Importing this module puts the repository root on the import path, so every experiment
runs as a plain script from any working directory:

    python experiments/01_forward_verification.py
    python experiments/01_forward_verification.py --quick     # seconds, smoke test only

It also defines the two run profiles. `--quick` shrinks every size (grid, network,
iterations, realizations) so the whole script finishes in about a minute on a CPU; its
output goes to `results/_quick/` and must never be reported.
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use('Agg')   # write figures to files; no display is needed on Colab or a cluster

import config as cfg                                          # noqa: E402
from src.pinn.network import ConcentrationNet                 # noqa: E402
from src.pinn.optimizer import TrainConfig                    # noqa: E402
from src.pinn.train_inverse import default_schedule           # noqa: E402
from src.solvers.fd_solver import make_fd_solver              # noqa: E402
from src.utils import io                                      # noqa: E402


def parse_args(description, **extra):
    """Command line of an experiment. `extra` adds options: name -> dict of argparse keywords."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--quick', action='store_true', help='tiny sizes for a smoke test (results are not valid)')
    parser.add_argument('--n-realizations', type=int, default=None,
                        help='number of noise realizations (default: the value in config.py)')
    parser.add_argument('--skip-pinn', action='store_true', help='run only the solver-based analysis')
    parser.add_argument('--force', action='store_true', help='recompute results that are already on disk')
    for name, options in extra.items():
        parser.add_argument(f'--{name}', **options)
    return parser.parse_args()


def profile(args):
    """Sizes of the run: the paper settings, or the smoke-test settings under --quick."""
    if args.quick:
        return SimpleNamespace(
            name='quick', results=cfg.RESULTS_DIR / '_quick',
            solver=dict(nx=41, dt=0.05), n_obs=80,
            seeds=cfg.OBS_SEEDS[:args.n_realizations or 2], pinn_seeds=cfg.OBS_SEEDS[:args.n_realizations or 1],
            layers=(4, 16, 16, 1), collocation=dict(n_domain=400, n_initial=100, n_boundary=80),
            forward=TrainConfig.quick(weights=cfg.W_FORWARD), inverse=TrainConfig.quick(),
            schedule={stage: TrainConfig.quick() for stage in cfg.STAGE_SCHEDULE},
            profile_points=5, n_samples=8, batch_size=4, mcmc=dict(n_chains=4, n_steps=40, burn_in=20),
            verification=dict(nx_list=(11, 21, 41), dt_list=(0.1, 0.05, 0.025), nx=21, order_tolerance=0.5))
    return SimpleNamespace(
        name='paper', results=cfg.RESULTS_DIR,
        solver=dict(nx=cfg.NX_FD, dt=cfg.DT_FD), n_obs=cfg.N_OBS,
        seeds=cfg.OBS_SEEDS[:args.n_realizations or len(cfg.OBS_SEEDS)],
        pinn_seeds=cfg.OBS_SEEDS[:args.n_realizations or cfg.N_REALIZATIONS_PINN],
        layers=cfg.LAYERS, collocation={},
        forward=TrainConfig(weights=cfg.W_FORWARD), inverse=TrainConfig(), schedule=default_schedule(),
        profile_points=cfg.PROFILE_POINTS, n_samples=cfg.N_POSTERIOR_SAMPLES, batch_size=8, mcmc={},
        verification={})


def start(title, args):
    """Print a header, build the run profile and the reference solver. Returns (profile, solver)."""
    run = profile(args)
    print('=' * 88)
    print(title)
    print(f'profile: {run.name}   device: {io.device_name()}   results: {run.results}')
    print('=' * 88)
    if args.quick:
        print('QUICK MODE: smoke test only. These numbers are not results.')
    return run, make_fd_solver(**run.solver)


def new_net(run):
    return ConcentrationNet(layers=run.layers)


def cached(path, compute, force=False):
    """Return the JSON at `path` if it exists, otherwise compute, save and return it.

    Long runs are split into one file per noise realization, so a Colab session that
    disconnects resumes where it stopped instead of starting again.
    """
    path = Path(path)
    if path.exists() and not force:
        print(f'  (cached) {path.relative_to(ROOT)}')
        return io.load_json(path)
    result = compute()
    io.save_json(path, result)
    return io.load_json(path)   # the same plain types whether cached or fresh


def write_table(path, header, rows):
    """Write a small CSV table for the manuscript. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [','.join(header)] + [','.join(f'{v:.6g}' if isinstance(v, float) else str(v) for v in row) for row in rows]
    path.write_text('\n'.join(lines) + '\n')
    return path


class Timer:
    """`with Timer('label'):` prints the elapsed time of a block."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *_):
        print(f'  {self.label}: {time.time() - self.start:.1f} s')


# ---- pieces shared by the inverse experiments ---------------------------------------------
def training_observations(run, solver, truth, seed, doses=cfg.TRAIN_DOSES, noise_sd=cfg.NOISE_SD):
    """Scattered observations (Design I) at the given training doses for one noise realization."""
    from src.utils.sampling import observe, scattered_design
    return observe(scattered_design(doses, run.n_obs, seed), solver, truth, noise_sd=noise_sd, seed=seed)


def recovery_record(estimate, truth, names):
    """Estimate, signed percent errors of the parameters and of their combinations."""
    from src.model.parameters import derived_quantities
    from src.utils.metrics import parameter_errors
    derived, derived_true = derived_quantities(estimate), derived_quantities(truth)
    return {'params': {name: estimate[name] for name in names},
            'error_percent': parameter_errors(estimate, truth, names),
            'derived': derived, 'derived_error_percent': parameter_errors(derived, derived_true)}


def print_recovery(title, records, key='error_percent'):
    """Mean and SD of the percent errors over realizations, one line per method."""
    from src.utils.metrics import summarise_errors
    print(f'  {title}')
    for method, per_seed in records.items():
        if not per_seed:
            continue
        summary = summarise_errors([record[key] for record in per_seed])
        cells = (f"{name} {s['mean']:+.1f} +- {s['sd']:.1f} %" for name, s in summary.items())
        print(f'    {method:<10s}' + '  '.join(cells))
