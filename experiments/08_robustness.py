"""Experiment 08 - robustness (experiments E6, E7 and E8).

Question   Do the identifiability conclusions survive worse data, a realistic sampling
           geometry, other parameter values and an imperfect model?

Parts      noise      E6  noise SD sweep at fixed locations
           sparsity   E6  number of observations per dose
           electrode  E7  FSCV-like virtual electrodes (dense in time at fixed sites)
           truths     --  parameter sets drawn from the robustness ranges
           mismatch   E8  the fitted model is mildly wrong:
                            'stimulation'  the data contain a release R the estimator ignores
                            'k_low/k_high' the estimator holds k at 0.75 / 1.25 of its value
                          and one control that is not a mismatch:
                            'k_true'       the estimator holds k at its true value, which
                                           separates the effect of fixing k from that of
                                           fixing it at a wrong value

           Each case reports the local diagnostics for the baseline and the multi-dose
           design, and the reference-estimator recovery over noise realizations: errors of
           the parameter combinations and of the global effect GTE at the held-out dose.
           Add --with-pinn to also train the staged PINN on the electrode designs.

Usage      python experiments/08_robustness.py --part noise
           python experiments/08_robustness.py --part all

Outputs    results/identifiability/robustness_<part>.json
           results/tables/robustness_<part>.csv
"""
from _common import cached, cfg, io, new_net, parse_args, recovery_record, start, write_table

import numpy as np

from src.counterfactual.counterfactual import counterfactual_effect
from src.identifiability import analyse_design
from src.model.dopamine_pde import gaussian_stimulation
from src.model.parameters import BASELINE_PARAM_NAMES, PARAM_NAMES, sample_params, true_params
from src.pinn.train_inverse import train_staged
from src.solvers.reference_estimator import ReferenceEstimator
from src.utils.metrics import summarise, summarise_errors
from src.utils.sampling import electrode_design, observe, scattered_design, select_doses

A = cfg.HELDOUT_DOSE
PARTS = ('noise', 'sparsity', 'electrode', 'truths', 'mismatch')
STIMULATION = dict(amplitude=0.02, t_on=10.0, t_off=12.0)   # uM/ms for 2 ms: an unmodelled second release
N_TRUTHS = 10


def diagnostics(design, solver, truth, noise_sd):
    """Local diagnostics of the baseline part and of the whole multi-dose design."""
    keep = ('n_obs', 'sigma_min', 'condition_number', 'collinearity_index', 'sd', 'derived_log_sd')
    baseline = analyse_design(select_doses(design, 0.0), solver, BASELINE_PARAM_NAMES, truth, noise_sd)
    multidose = analyse_design(design, solver, PARAM_NAMES, truth, noise_sd)
    return {'baseline': {k: baseline[k] for k in keep}, 'multidose': {k: multidose[k] for k in keep}}


def recovery(solver, design, truth, seeds, noise_sd, data_release=None, names=PARAM_NAMES, fixed=None):
    """Reference-estimator fits over noise realizations; errors of parameters, combinations and GTE."""
    estimator = ReferenceEstimator(solver, names)
    true_gte = counterfactual_effect(solver, truth, A)['GTE']
    records = []
    for seed in seeds:
        obs = observe(design, solver, truth, noise_sd=noise_sd, seed=seed, release=data_release)
        fit = estimator.fit(obs, fixed=fixed, noise_sd=noise_sd if noise_sd > 0 else None)
        gte = counterfactual_effect(solver, fit['params'], A)['GTE']
        record = recovery_record(fit['params'], truth, PARAM_NAMES)
        records.append({**record, 'GTE_error_percent': 100.0 * (gte - true_gte) / true_gte,
                        'at_bound': fit['at_bound'], 'chi2_per_dof': fit['chi2'] / fit['dof']})
    return {'parameters': summarise_errors([r['error_percent'] for r in records]),
            'combinations': summarise_errors([r['derived_error_percent'] for r in records]),
            'GTE_error_percent': summarise([r['GTE_error_percent'] for r in records]),
            'chi2_per_dof': summarise([r['chi2_per_dof'] for r in records]),
            'n_at_bound': int(sum(r['at_bound'] for r in records))}


def case(solver, design, truth, seeds, noise_sd=cfg.NOISE_SD, **recovery_options):
    result = {'recovery': recovery(solver, design, truth, seeds, noise_sd, **recovery_options)}
    if noise_sd > 0:
        result['diagnostics'] = diagnostics(design, solver, truth, noise_sd)
    return result


def run_part(part, run, solver, args):
    truth, seed = true_params(), run.seeds[0]
    scattered = scattered_design(cfg.TRAIN_DOSES, run.n_obs, seed)

    if part == 'noise':
        return {f'noise_sd={sd:g}': case(solver, scattered, truth, run.seeds, sd) for sd in cfg.NOISE_SWEEP}
    if part == 'sparsity':
        counts = cfg.N_OBS_SWEEP if not args.quick else (40, 80)
        return {f'n_obs={n}': case(solver, scattered_design(cfg.TRAIN_DOSES, n, seed), truth, run.seeds)
                for n in counts}
    if part == 'electrode':
        cases = {}
        for name, sites in cfg.ELECTRODE_DESIGNS.items():
            design = electrode_design(cfg.TRAIN_DOSES, sites)
            cases[name] = case(solver, design, truth, run.seeds)
            if args.with_pinn:
                obs = observe(design, solver, truth, seed=seed)
                result = train_staged(obs, schedule=run.schedule, net=new_net(run), label=f'electrode/{name}',
                                      **run.collocation)
                cases[name]['pinn'] = {**recovery_record(result['params'], truth, PARAM_NAMES),
                                       'minutes': result['minutes'], 'seed': seed}
        return cases
    if part == 'truths':
        rng = np.random.default_rng(seed)
        cases = {}
        for i in range(N_TRUTHS if not args.quick else 2):
            params = sample_params(rng)
            cases[f'truth_{i}'] = {'params': params,
                                   'diagnostics': diagnostics(scattered, solver, params, cfg.NOISE_SD)}
        return cases
    if part == 'mismatch':
        free_without_k = tuple(n for n in PARAM_NAMES if n != 'k')
        return {
            'none': case(solver, scattered, truth, run.seeds),
            'stimulation': case(solver, scattered, truth, run.seeds, data_release=gaussian_stimulation(**STIMULATION)),
            'k_true': case(solver, scattered, truth, run.seeds, names=free_without_k, fixed={'k': truth['k']}),
            'k_low': case(solver, scattered, truth, run.seeds, names=free_without_k, fixed={'k': 0.75 * truth['k']}),
            'k_high': case(solver, scattered, truth, run.seeds, names=free_without_k, fixed={'k': 1.25 * truth['k']}),
        }
    raise ValueError(f'unknown part {part!r}')


def report(part, cases, run):
    rows = []
    for label, result in cases.items():
        row = [label]
        if 'diagnostics' in result:
            d = result['diagnostics']
            row += [d['baseline']['sigma_min'], d['multidose']['sigma_min'], d['multidose']['condition_number'],
                    d['multidose']['derived_log_sd'].get('baseline_rate', float('nan')),
                    d['multidose']['derived_log_sd'].get('treatment_rate', float('nan'))]
        else:
            row += [float('nan')] * 5
        if 'recovery' in result:
            r = result['recovery']
            row += [r['combinations']['baseline_rate']['mean'], r['combinations']['baseline_rate']['sd'],
                    r['combinations']['treatment_rate']['mean'], r['combinations']['treatment_rate']['sd'],
                    r['GTE_error_percent']['mean'], r['GTE_error_percent']['sd'], r['n_at_bound']]
            print(f"  {label:<16s} baseline rate {row[6]:+.1f} +- {row[7]:.1f} %   treatment rate {row[8]:+.1f} +- "
                  f"{row[9]:.1f} %   GTE {row[10]:+.1f} +- {row[11]:.1f} %   on bound {row[12]}")
        else:
            row += [float('nan')] * 6 + [0]
            print(f"  {label:<16s} sigma_min baseline {row[1]:.3g} -> multi-dose {row[2]:.3g}")
        rows.append(row)
    write_table(run.results / 'tables' / f'robustness_{part}.csv',
                ['case', 'sigma_min_baseline', 'sigma_min_multidose', 'condition_multidose',
                 'crb_log_sd_baseline_rate', 'crb_log_sd_treatment_rate', 'baseline_rate_error_mean',
                 'baseline_rate_error_sd', 'treatment_rate_error_mean', 'treatment_rate_error_sd',
                 'gte_error_mean', 'gte_error_sd', 'n_at_bound'], rows)


def main():
    args = parse_args(__doc__.split('\n')[0],
                      **{'part': dict(choices=PARTS + ('all',), default='all', help='which robustness study to run'),
                         'with-pinn': dict(action='store_true', help='also train the staged PINN (electrode part)')})
    run, solver = start('Experiment 08: robustness', args)
    for part in (PARTS if args.part == 'all' else (args.part,)):
        print(f'--- {part} ---')
        path = run.results / 'identifiability' / f'robustness_{part}.json'
        compute = lambda: {'cases': run_part(part, run, solver, args), 'meta': io.run_metadata()}
        result = cached(path, compute, args.force)
        report(part, result['cases'], run)


if __name__ == '__main__':
    main()
