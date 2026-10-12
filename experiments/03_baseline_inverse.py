"""Experiment 03 - baseline inverse recovery (Stage 3, experiment E1).

Question   Can (D, Vmax0, Km, k) be recovered from untreated sparse, noisy data, and
           does the PINN reach the accuracy of the reference estimator?

Steps      For each independent noise realization:
             1. reference estimator: least squares through the finite-difference solver;
             2. inverse PINN with the four baseline parameters free (eta is inactive).
           Then summarise the signed errors over the realizations (Figure 4, baseline).

Outputs    results/inverse/baseline/seed_<seed>.json   one file per realization (resumable)
           results/inverse/baseline_summary.json
           results/tables/baseline_recovery.csv
           results/figures/baseline_recovery.*

Reading the result: if the reference estimator cannot recover a parameter, the data do
not contain it and the PINN is not expected to (decision rule 3). Errors of the
combinations Vmax0 / Km + k show what the baseline data do determine.
"""
from _common import (cached, cfg, io, new_net, parse_args, print_recovery, recovery_record, start,
                     training_observations, write_table)

from src.model.parameters import BASELINE_PARAM_NAMES, true_params
from src.pinn.train_inverse import train_inverse
from src.solvers.reference_estimator import ReferenceEstimator
from src.utils import plotting
from src.utils.metrics import summarise_errors
from src.utils.sampling import collocation_points

NAMES = BASELINE_PARAM_NAMES


def main():
    args = parse_args(__doc__.split('\n')[0])
    run, solver = start('Experiment 03: baseline inverse recovery (A = 0)', args)
    truth = true_params()
    fixed = {'eta': truth['eta']}   # inactive at A = 0
    estimator = ReferenceEstimator(solver, NAMES)
    points = collocation_points(**{**run.collocation, 'doses': (0.0,)})
    out = run.results / 'inverse' / 'baseline'

    def one_realization(seed):
        obs = training_observations(run, solver, truth, seed, doses=(0.0,))
        fit = estimator.fit(obs, fixed=fixed)
        record = {'seed': seed, 'reference': {**recovery_record(fit['params'], truth, NAMES),
                                              'sd_raw': fit['sd_raw'], 'interval': fit['interval'],
                                              'chi2': fit['chi2'], 'dof': fit['dof'], 'at_bound': fit['at_bound']}}
        if not args.skip_pinn and seed in run.pinn_seeds:
            result = train_inverse(obs, free=NAMES, tc=run.inverse, net=new_net(run), points=points,
                                   label=f'baseline/{seed}')
            record['pinn'] = {**recovery_record(result['params'], truth, NAMES), 'terms': result['terms'],
                              'rms_data_over_noise': result['rms_data'] / cfg.NOISE_SD, 'minutes': result['minutes']}
        return record

    records = [cached(out / f'seed_{seed}.json', lambda: one_realization(seed), args.force) for seed in run.seeds]
    by_method = {method: [r[method] for r in records if method in r] for method in ('reference', 'pinn')}
    print_recovery('parameter error over realizations (mean +- SD)', by_method)
    print_recovery('combination error over realizations (mean +- SD)', by_method, 'derived_error_percent')
    at_bound = sum(r['reference']['at_bound'] for r in records)
    print(f'  reference estimate on a bound in {at_bound} of {len(records)} realizations')

    summary = {method: {'parameters': summarise_errors([r['error_percent'] for r in rs]),
                        'combinations': summarise_errors([r['derived_error_percent'] for r in rs])}
               for method, rs in by_method.items() if rs}
    io.save_json(run.results / 'inverse' / 'baseline_summary.json',
                 {'summary': summary, 'n_at_bound_reference': at_bound, 'seeds': list(run.seeds),
                  'meta': io.run_metadata()})
    write_table(run.results / 'tables' / 'baseline_recovery.csv',
                ['method', 'parameter', 'mean_error_percent', 'sd_error_percent', 'median_error_percent', 'n'],
                [[method, name, s['mean'], s['sd'], s['median'], s['n']]
                 for method, block in summary.items() for name, s in block['parameters'].items()])
    plotting.plot_parameter_recovery(
        {method: {name: [r['error_percent'][name] for r in rs] for name in NAMES}
         for method, rs in by_method.items() if rs},
        NAMES, run.results / 'figures' / 'baseline_recovery')


if __name__ == '__main__':
    main()
