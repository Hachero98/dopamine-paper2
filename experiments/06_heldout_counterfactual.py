"""Experiment 06 - held-out counterfactual validation (Stage 6, experiment E4).

Question   With the model calibrated on A = 0, 0.25, 0.75 and 1 and then frozen, how well
           is the unseen dose A = 0.5 predicted: the field C^(0.5) and the effect
           tau_0.5 = C^(0.5) - C^(0)?

Steps      For each realization calibrated by experiment 05, three predictions are scored
           against the synthetic reference:
             pinn_network     the trained network evaluated at A = 0.5 and A = 0
             pinn_parameters  the reference solver run with the PINN parameter estimates
             reference        the reference solver run with the reference-estimator estimates
           Scores: relative L2 error of the field, L2 and L-infinity errors of tau_0.5, and
           errors of ATE(t), CE(x, y), GTE; also the dose sensitivity dC/dA at A = 0.5.

Requires   experiment 05 (same profile).

Outputs    results/counterfactual/heldout.json
           results/tables/heldout_counterfactual.csv
           results/figures/fig6_heldout_counterfactual.*

The A = 0.5 reference field is created here for scoring only. Nothing in this script
feeds back into training, tuning or model selection.
"""
from _common import cfg, io, new_net, parse_args, start, write_table

import numpy as np

from src.counterfactual.counterfactual import (counterfactual_effect, dose_sensitivity, effect_errors,
                                               treatment_effect)
from src.model.parameters import true_params
from src.pinn.network import predict_dose_sensitivity, predict_field
from src.utils import plotting
from src.utils.metrics import relative_l2, summarise

A = cfg.HELDOUT_DOSE


def score(C_a, C_0, truth_effect, solver):
    """Errors of a predicted held-out field and of the effect derived from it."""
    return {'field_rel_l2': relative_l2(C_a, truth_effect['C_a']),
            **effect_errors(treatment_effect(C_a, C_0), truth_effect['tau'], solver)}


def main():
    args = parse_args(__doc__.split('\n')[0])
    run, solver = start(f'Experiment 06: held-out counterfactual validation (A = {A})', args)
    truth = true_params()
    truth_effect = counterfactual_effect(solver, truth, A)
    truth_sensitivity = dose_sensitivity(solver, truth, A)
    calibrated = run.results / 'inverse' / 'multidose'

    records, first_network_tau = [], None
    for seed in run.seeds:
        path = calibrated / f'seed_{seed}.json'
        if not path.exists():
            continue
        fit = io.load_json(path)
        effect = counterfactual_effect(solver, {**truth, **fit['reference']['params']}, A)
        record = {'seed': seed, 'reference': score(effect['C_a'], effect['C_0'], truth_effect, solver)}
        if 'pinn' in fit and (calibrated / f'weights_{seed}.npz').exists():
            effect = counterfactual_effect(solver, fit['pinn']['params'], A)
            record['pinn_parameters'] = score(effect['C_a'], effect['C_0'], truth_effect, solver)
            net = new_net(run)
            weights = io.load_weights(calibrated / f'weights_{seed}.npz', net)
            C_a, C_0 = predict_field(net, weights, solver, A), predict_field(net, weights, solver, 0.0)
            record['pinn_network'] = score(C_a, C_0, truth_effect, solver)
            T, X, Y = np.meshgrid(np.arange(solver.n_frames + 1) * solver.dt_store, solver.x, solver.x, indexing='ij')
            record['pinn_network']['dose_sensitivity_rel_l2'] = relative_l2(
                predict_dose_sensitivity(net, weights, X, Y, T, A), truth_sensitivity)
            if first_network_tau is None:
                first_network_tau = treatment_effect(C_a, C_0)
        records.append(record)
    if not records:
        raise SystemExit('No calibrated realizations found. Run experiments/05_multidose_inverse.py first '
                         '(with the same --quick or --trial setting).')

    methods = [m for m in ('reference', 'pinn_parameters', 'pinn_network') if any(m in r for r in records)]
    metrics = ('field_rel_l2', 'ATE_rel_l2', 'CE_rel_l2', 'GTE_rel')
    summary = {}
    for method in methods:
        rows = [r[method] for r in records if method in r]
        summary[method] = {metric: summarise([row[metric] for row in rows]) for metric in metrics}
        summary[method]['tau_rel_l2'] = summarise([row['tau']['rel_l2'] for row in rows])
        summary[method]['tau_rel_linf'] = summarise([row['tau']['rel_linf'] for row in rows])
        print(f"  {method:<16s} field {100 * summary[method]['field_rel_l2']['mean']:.2f} %   "
              f"tau L2 {100 * summary[method]['tau_rel_l2']['mean']:.1f} %   "
              f"tau Linf {100 * summary[method]['tau_rel_linf']['mean']:.1f} %   "
              f"GTE {100 * summary[method]['GTE_rel']['mean']:+.1f} %   (n = {len(rows)})")

    io.save_json(run.results / 'counterfactual' / 'heldout.json',
                 {'dose': A, 'truth': {'GTE': truth_effect['GTE'], 'peak_tau': truth_effect['peak']},
                  'summary': summary, 'realizations': records, 'meta': io.run_metadata()})
    write_table(run.results / 'tables' / 'heldout_counterfactual.csv',
                ['method', 'metric', 'mean', 'sd', 'median', 'n'],
                [[method, metric, s['mean'], s['sd'], s['median'], s['n']]
                 for method, block in summary.items() for metric, s in block.items()])
    if first_network_tau is not None:
        plotting.plot_counterfactual_maps(truth_effect['tau'], first_network_tau, solver, dose=A,
                                          path=run.results / 'figures' / 'fig6_heldout_counterfactual')


if __name__ == '__main__':
    main()
