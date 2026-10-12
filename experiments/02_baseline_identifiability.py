"""Experiment 02 - baseline identifiability (Stage 2, experiment E1).

Question   Using untreated observations only (A = 0), which of D, Vmax0, Km and k can be
           separated, and what are the weak directions? This is settled with the
           reference solver before any inverse PINN is trained.

Steps      1. Local diagnostics of the scattered design and of the electrode designs:
              singular values, sensitivity correlations, Cramer-Rao bounds, and the
              bounds of the parameter combinations (Vmax0 / Km, Vmax0 / Km + k).
           2. Saturation sweep: repeat for release amplitudes from C0 << Km to C0 >> Km
              to show when Vmax0 and Km are seen only through their ratio.
           3. Profile likelihoods of Vmax0, Km and k on one noise realization.

Outputs    results/identifiability/baseline.json
           results/figures/baseline_identifiability.*, baseline_profiles.*

Decision rule: near-exact confounding found here is documented as a result. It is not
to be hidden by tuning the PINN.
"""
from _common import cfg, io, parse_args, start, Timer

from src.identifiability import analyse_design
from src.identifiability.profile_likelihood import profile_grid, profile_likelihood
from src.model.initial_conditions import initial_field
from src.model.parameters import BASELINE_PARAM_NAMES, true_params
from src.solvers.reference_estimator import ReferenceEstimator
from src.utils import plotting
from src.utils.sampling import electrode_design, observe, scattered_design

SATURATION_C0 = (0.05, 0.2, 1.0, 5.0, 20.0)   # release amplitudes, uM (Km = 0.2 uM)
PROFILED = ('Vmax0', 'Km', 'k')
NAMES = BASELINE_PARAM_NAMES


def print_report(label, report):
    sd = '  '.join(f'{name} {value:.3g}' for name, value in report['sd'].items())
    print(f"  {label:<18s} sigma_min {report['sigma_min']:.3g}  cond {report['condition_number']:.3g}  "
          f"collinearity {report['collinearity_index']:.3g}")
    print(f"  {'':<18s} CRB SD of log: {sd}")
    print(f"  {'':<18s} combinations : " + '  '.join(f'{k} {v:.3g}' for k, v in report['derived_log_sd'].items()))


def main():
    args = parse_args(__doc__.split('\n')[0])
    run, solver = start('Experiment 02: baseline identifiability (A = 0)', args)
    truth, seed = true_params(), run.seeds[0]

    # ---- 1. local diagnostics per design ------------------------------------------------
    designs = {'scattered': scattered_design((0.0,), run.n_obs, seed)}
    for name, sites in cfg.ELECTRODE_DESIGNS.items():
        designs[f'electrode {name}'] = electrode_design((0.0,), sites)
    reports = {}
    with Timer('sensitivity analysis'):
        for label, design in designs.items():
            reports[label] = analyse_design(design, solver, NAMES)
            print_report(label, reports[label])

    # ---- 2. saturation sweep --------------------------------------------------------------
    # The noise stays at 2 % of the release amplitude, so only the degree of saturation changes.
    saturation = {}
    for C0 in SATURATION_C0:
        report = analyse_design(designs['scattered'], solver, NAMES, noise_sd=cfg.NOISE_SD * C0 / cfg.C0,
                                C_init=initial_field(solver.nx, C0=C0))
        saturation[C0] = report
        index = {name: i for i, name in enumerate(NAMES)}
        print(f"  C0 = {C0:<5g} uM  corr(Vmax0, Km) {report['correlations'][index['Vmax0'], index['Km']]:+.4f}  "
              f"SD log Vmax0 {report['sd']['Vmax0']:.3g}  SD log Km {report['sd']['Km']:.3g}")

    # ---- 3. profile likelihoods -----------------------------------------------------------
    obs = observe(designs['scattered'], solver, truth, seed=seed)
    fixed = {'eta': truth['eta']}   # inactive at A = 0; any value gives the same fit
    best = ReferenceEstimator(solver, NAMES).fit(obs, fixed=fixed)
    profiles = {}
    with Timer('profile likelihoods'):
        for name in PROFILED:
            grid = profile_grid(name, truth[name], run.profile_points)
            profiles[name] = profile_likelihood(obs, solver, name, NAMES, fixed, best, grid)
            print(f"  profile {name:<6s} interval {profiles[name]['interval']}  "
                  f"identifiable {profiles[name]['identifiable']}")

    io.save_json(run.results / 'identifiability' / 'baseline.json',
                 {'designs': reports, 'saturation_sweep': saturation, 'reference_fit': best, 'profiles': profiles,
                  'seed': seed, 'meta': io.run_metadata()})
    plotting.plot_identifiability({k: r['singular_values'] for k, r in reports.items()},
                                  {k: r['correlations'] for k, r in reports.items()}, NAMES,
                                  run.results / 'figures' / 'baseline_identifiability')
    plotting.plot_profiles({'A = 0': profiles}, truth, run.results / 'figures' / 'baseline_profiles')


if __name__ == '__main__':
    main()
