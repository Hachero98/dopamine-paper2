"""Experiment 09 - does the PINN estimate depend on where training starts?

Question   The reference estimator and the profile likelihoods show that Vmax0 and Km
           are not separately determined by the data: only combinations such as
           Vmax0 / Km + k are. Yet the PINN of experiment 05 returns Vmax0 and Km within
           about 10 % of the truth. Is that information from the data, or memory of the
           starting point, which lies only 20 % from the truth?

Steps      For each noise realization and each start in config.START_FACTORS:
             staged inverse PINN on all training doses (as in experiment 05), started
             from INIT_PARAMS with Vmax0 and Km both multiplied by the factor.
           Every start has the same uptake ratio Vmax0 / Km, so the low-concentration
           data hardly tell the starts apart.

Outputs    results/inverse/start_sensitivity/<start>_seed_<seed>.json   (resumable)
           results/tables/start_sensitivity.csv

Reading the result: compare the final Vmax0 and Km across starts for the same
realization. If they follow the start (larger start, larger estimate) while the data fit
(rms / noise) and the combinations stay the same, the PINN does not identify Vmax0 and
Km, and their PINN errors in experiments 03 to 05 must not be reported as recovery. If
all starts end at the same values, the PINN finds information that the reference
estimator does not, and that needs an explanation before it is claimed.

Run it as a trial first (about 1 to 2 minutes per PINN run on an A100):

    python experiments/09_start_sensitivity.py --trial
"""
from _common import cached, cfg, new_net, parse_args, recovery_record, start, training_observations, write_table

from src.model.parameters import PARAM_NAMES, initial_params, true_params
from src.pinn.train_inverse import train_staged

NAMES = PARAM_NAMES
COMBINATIONS = ('uptake_ratio', 'baseline_rate', 'treatment_rate')


def start_params(factor):
    """INIT_PARAMS with Vmax0 and Km scaled together, so Vmax0 / Km is unchanged."""
    params = initial_params()
    params['Vmax0'] *= factor
    params['Km'] *= factor
    return params


def main():
    args = parse_args(__doc__.split('\n')[0])
    run, solver = start('Experiment 09: sensitivity of the PINN estimate to its starting point', args)
    truth = true_params()
    out = run.results / 'inverse' / 'start_sensitivity'

    def one_run(seed, label, factor):
        obs = training_observations(run, solver, truth, seed)
        init = start_params(factor)
        result = train_staged(obs, init=init, schedule=run.schedule, net=new_net(run),
                              label=f'start/{label}/{seed}', **run.collocation)
        return {'seed': seed, 'start': label, 'factor': factor, 'init': init,
                **recovery_record(result['params'], truth, NAMES),
                'rms_data_over_noise': result['rms_data'] / cfg.NOISE_SD, 'minutes': result['minutes']}

    records = [cached(out / f'{label}_seed_{seed}.json', lambda: one_run(seed, label, factor), args.force)
               for seed in run.pinn_seeds for label, factor in cfg.START_FACTORS.items()]

    print('  final estimate per realization (percent error from the truth)')
    columns = list(NAMES) + list(COMBINATIONS)
    print(f"    {'seed':<6s}{'start':<10s}" + ''.join(f'{name:>15s}' for name in columns) + f"{'rms/noise':>11s}")
    rows = []
    for r in records:
        errors = {**r['error_percent'], **r['derived_error_percent']}
        print(f"    {r['seed']:<6d}{r['start']:<10s}" + ''.join(f'{errors[name]:>+14.1f}%' for name in columns)
              + f"{r['rms_data_over_noise']:>11.3f}")
        rows.append([r['seed'], r['start'], r['factor']] + [errors[name] for name in columns]
                    + [r['rms_data_over_noise'], r['minutes']])

    # How far apart the starts end, per realization: the ratio largest / smallest estimate.
    print('  spread across starts, largest / smallest final value (1.00 = the start does not matter)')
    for seed in run.pinn_seeds:
        own = [r for r in records if r['seed'] == seed]
        spread = {name: max(r['params'][name] for r in own) / min(r['params'][name] for r in own)
                  for name in ('Vmax0', 'Km', 'k', 'eta')}
        print(f'    seed {seed}: ' + '  '.join(f'{name} {value:.2f}' for name, value in spread.items()))
    start_spread = max(cfg.START_FACTORS.values()) / min(cfg.START_FACTORS.values())
    print(f'  (the starts themselves differ by a factor {start_spread:g} in Vmax0 and in Km)')

    write_table(run.results / 'tables' / 'start_sensitivity.csv',
                ['seed', 'start', 'factor'] + [f'{name}_error_percent' for name in columns]
                + ['rms_data_over_noise', 'minutes'], rows)


if __name__ == '__main__':
    main()
