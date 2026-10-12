"""Experiment 04 - treatment-response identification (Stage 4, experiment E2).

Question   Does one treated condition add an independent sensitivity direction? How do
           the spectrum, the correlations, the profiles and the uncertainty change
           relative to the untreated baseline?

Steps      1. Local diagnostics of four kinds of design, all at the true parameters:
                A = 0                      the baseline
                A = 0, doubled budget      control: same number of points as a two-dose
                                           design, but no treatment
                A = 0 + one treated dose   for each dose in config.SINGLE_TREATED_DOSES
                all training doses         for reference (used jointly in experiment 05)
              The control separates the information added by the intervention from the
              information added by simply having more observations. Each design is
              analysed twice: with eta unknown (what a fit faces) and with eta known
              (the most the intervention could add).
           2. Profile likelihoods of Vmax0, Km, k and eta with and without treated data
              (Figure 5).
           3. Estimate eta from A = 0 plus the full dose over noise realizations, with the
              reference estimator and with the staged PINN.

Outputs    results/identifiability/single_treatment.json
           results/inverse/single_treatment/seed_<seed>.json
           results/figures/fig3_identifiability.*, fig5_profiles.*

Decision rule 4: an improvement that appears in the PINN estimates but not in the
reference-estimator geometry is an optimisation effect, not restored identifiability.
"""
from _common import (cached, cfg, io, new_net, parse_args, print_recovery, recovery_record, start,
                     training_observations, Timer)

from src.identifiability import analyse_design
from src.identifiability.fisher_information import information_gain
from src.identifiability.profile_likelihood import profile_grid, profile_likelihood
from src.model.parameters import BASELINE_PARAM_NAMES, PARAM_NAMES, true_params
from src.pinn.train_inverse import train_staged
from src.solvers.reference_estimator import ReferenceEstimator
from src.utils import plotting
from src.utils.sampling import scattered_design, select_doses

ESTIMATION_DOSE = 1.0   # treated dose used in step 3


def main():
    args = parse_args(__doc__.split('\n')[0])
    run, solver = start('Experiment 04: treatment-response identification', args)
    truth, seed, n = true_params(), run.seeds[0], run.n_obs

    # ---- 1. local diagnostics -------------------------------------------------------------
    designs = {'A = 0': scattered_design((0.0,), n, seed),
               'A = 0 (2x points)': scattered_design((0.0,), 2 * n, seed)}
    designs.update({f'A = 0 + {a:g}': scattered_design((0.0, a), n, seed) for a in cfg.SINGLE_TREATED_DOSES})
    designs['all training doses'] = scattered_design(cfg.TRAIN_DOSES, n, seed)

    # Two views of every design:
    #   reports     the parameters one would actually fit (eta included once there is treated
    #               data). Its Cramer-Rao SDs are marginal: they account for not knowing eta.
    #   eta_known   the four baseline parameters with eta given. This is an upper limit on
    #               what the intervention can add, reached only if eta is known independently.
    reports, eta_known = {}, {}
    with Timer('sensitivity analysis'):
        for label, design in designs.items():
            treated = bool((design['A_d'] > 0).any())
            reports[label] = analyse_design(design, solver, PARAM_NAMES if treated else BASELINE_PARAM_NAMES)
            eta_known[label] = analyse_design(design, solver, BASELINE_PARAM_NAMES)
    reference = eta_known['A = 0']
    print('  Cramer-Rao SD of the log-parameters (0.05 = 5 %; above 1 = not determined)')
    for label, report in reports.items():
        known = eta_known[label]
        known['gain_over_baseline'] = information_gain(reference['fisher'], known['fisher'])
        sd = {**{'eta': float('nan')}, **report['sd']}
        print(f"  {label:<20s} n {report['n_obs']:<5d} Vmax0 {sd['Vmax0']:<6.3g} Km {sd['Km']:<6.3g} k {sd['k']:<6.3g} "
              f"eta {sd['eta']:<6.3g} baseline rate {report['derived_log_sd'].get('baseline_rate', float('nan')):<6.3g}"
              f"| eta known: sigma_min x{known['sigma_min'] / reference['sigma_min']:.2f}, k {known['sd']['k']:.3g}")

    # ---- 2. profile likelihoods -----------------------------------------------------------
    obs_all = training_observations(run, solver, truth, seed)
    cases = {'A = 0': (select_doses(obs_all, 0.0), BASELINE_PARAM_NAMES, {'eta': truth['eta']}),
             f'A = 0 + {ESTIMATION_DOSE:g}': (select_doses(obs_all, (0.0, ESTIMATION_DOSE)), PARAM_NAMES, {}),
             'all training doses': (obs_all, PARAM_NAMES, {})}
    profiles = {}
    with Timer('profile likelihoods'):
        for label, (obs, names, fixed) in cases.items():
            best = ReferenceEstimator(solver, names).fit(obs, fixed=fixed)
            profiles[label] = {name: profile_likelihood(obs, solver, name, names, fixed, best,
                                                        profile_grid(name, truth[name], run.profile_points))
                               for name in names if name != 'D'}
            print(f'  {label:<22s}' + '  '.join(f"{name}: {'bounded' if p['identifiable'] else 'open'}"
                                                 for name, p in profiles[label].items()))

    io.save_json(run.results / 'identifiability' / 'single_treatment.json',
                 {'designs': reports, 'designs_with_eta_known': eta_known, 'profiles': profiles, 'seed': seed,
                  'meta': io.run_metadata()})
    shown = ['A = 0', f'A = 0 + {ESTIMATION_DOSE:g}', 'all training doses']
    plotting.plot_identifiability({k: reports[k]['singular_values'] for k in shown},
                                  {k: reports[k]['correlations'] for k in shown}, PARAM_NAMES,
                                  run.results / 'figures' / 'fig3_identifiability')
    plotting.plot_profiles(profiles, truth, run.results / 'figures' / 'fig5_profiles')

    # ---- 3. estimate eta from baseline + one treated dose --------------------------------
    estimator = ReferenceEstimator(solver, PARAM_NAMES)
    out = run.results / 'inverse' / 'single_treatment'

    def one_realization(s):
        obs = training_observations(run, solver, truth, s, doses=(0.0, ESTIMATION_DOSE))
        fit = estimator.fit(obs)
        record = {'seed': s, 'reference': {**recovery_record(fit['params'], truth, PARAM_NAMES),
                                           'sd_raw': fit['sd_raw'], 'at_bound': fit['at_bound']}}
        if not args.skip_pinn and s in run.pinn_seeds:
            result = train_staged(obs, schedule=run.schedule, net=new_net(run), label=f'single/{s}', **run.collocation)
            record['pinn'] = {**recovery_record(result['params'], truth, PARAM_NAMES), 'stages': result['stages'],
                              'minutes': result['minutes']}
        return record

    records = [cached(out / f'seed_{s}.json', lambda: one_realization(s), args.force) for s in run.seeds]
    by_method = {m: [r[m] for r in records if m in r] for m in ('reference', 'pinn')}
    print_recovery(f'A = 0 + {ESTIMATION_DOSE:g}: parameter error (mean +- SD)', by_method)
    print_recovery(f'A = 0 + {ESTIMATION_DOSE:g}: combination error (mean +- SD)', by_method, 'derived_error_percent')


if __name__ == '__main__':
    main()
