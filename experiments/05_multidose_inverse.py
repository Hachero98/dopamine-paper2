"""Experiment 05 - joint multi-dose refinement (Stage 5, experiment E3).

Question   Do observations at A = 0, 0.25, 0.75 and 1, fitted jointly with one shared set
           of baseline parameters, give practical recovery of (D, Vmax0, Km, k, eta)?

Steps      For each independent noise realization:
             1. reference estimator on all training doses (five parameters);
             2. staged inverse PINN: baseline -> treatment response -> joint refinement.
           The calibrated parameters and the network weights of every realization are
           stored; experiment 06 uses them, frozen, to predict the held-out dose.

Outputs    results/inverse/multidose/seed_<seed>.json, weights_<seed>.npz
           results/inverse/multidose_summary.json
           results/tables/multidose_recovery.csv
           results/figures/fig4_parameter_recovery.*

The held-out dose A = 0.5 is not used anywhere in this script.
"""
from _common import (cached, cfg, io, new_net, parse_args, print_recovery, recovery_record, start,
                     training_observations, write_table)

from src.model.parameters import PARAM_NAMES, true_params
from src.pinn.train_inverse import train_staged
from src.solvers.reference_estimator import ReferenceEstimator
from src.utils import plotting
from src.utils.metrics import interval_coverage, summarise_errors

NAMES = PARAM_NAMES


def main():
    args = parse_args(__doc__.split('\n')[0])
    run, solver = start('Experiment 05: joint multi-dose inverse recovery', args)
    truth = true_params()
    estimator = ReferenceEstimator(solver, NAMES)
    out = run.results / 'inverse' / 'multidose'

    def one_realization(seed):
        obs = training_observations(run, solver, truth, seed)
        fit = estimator.fit(obs)
        record = {'seed': seed, 'reference': {**recovery_record(fit['params'], truth, NAMES),
                                              'theta': fit['theta'], 'cov_raw': fit['cov_raw'],
                                              'sd_raw': fit['sd_raw'], 'interval': fit['interval'],
                                              'chi2': fit['chi2'], 'dof': fit['dof'], 'at_bound': fit['at_bound']}}
        if not args.skip_pinn and seed in run.pinn_seeds:
            result = train_staged(obs, schedule=run.schedule, net=new_net(run), label=f'multidose/{seed}',
                                  **run.collocation)
            io.save_weights(out / f'weights_{seed}.npz', result['_weights'])
            record['pinn'] = {**recovery_record(result['params'], truth, NAMES), 'stages': result['stages'],
                              'terms': result['terms'], 'rms_data_over_noise': result['rms_data'] / cfg.NOISE_SD,
                              'minutes': result['minutes']}
        return record

    records = [cached(out / f'seed_{seed}.json', lambda: one_realization(seed), args.force) for seed in run.seeds]
    by_method = {method: [r[method] for r in records if method in r] for method in ('reference', 'pinn')}
    print_recovery('parameter error over realizations (mean +- SD)', by_method)
    print_recovery('combination error over realizations (mean +- SD)', by_method, 'derived_error_percent')

    coverage = {name: interval_coverage([r['reference']['interval'][name] for r in records], truth[name])
                for name in NAMES}
    at_bound = sum(r['reference']['at_bound'] for r in records)
    print('  coverage of the reference 95 % intervals: ' + '  '.join(f'{k} {v:.2f}' for k, v in coverage.items()))
    print(f'  reference estimate on a bound in {at_bound} of {len(records)} realizations')

    summary = {method: {'parameters': summarise_errors([r['error_percent'] for r in rs]),
                        'combinations': summarise_errors([r['derived_error_percent'] for r in rs])}
               for method, rs in by_method.items() if rs}
    io.save_json(run.results / 'inverse' / 'multidose_summary.json',
                 {'summary': summary, 'reference_interval_coverage': coverage, 'n_at_bound_reference': at_bound,
                  'seeds': list(run.seeds), 'meta': io.run_metadata()})
    write_table(run.results / 'tables' / 'multidose_recovery.csv',
                ['method', 'parameter', 'mean_error_percent', 'sd_error_percent', 'median_error_percent', 'n'],
                [[method, name, s['mean'], s['sd'], s['median'], s['n']]
                 for method, block in summary.items() for name, s in block['parameters'].items()])
    plotting.plot_parameter_recovery(
        {method: {name: [r['error_percent'][name] for r in rs] for name in NAMES}
         for method, rs in by_method.items() if rs},
        NAMES, run.results / 'figures' / 'fig4_parameter_recovery')


if __name__ == '__main__':
    main()
