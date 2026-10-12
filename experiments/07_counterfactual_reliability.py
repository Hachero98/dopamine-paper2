"""Experiment 07 - counterfactual reliability under non-identifiability (Stage 7, E5).

Question   If several parameter vectors fit the observed data equally well, do they agree
           on the effect at the unseen dose, tau_0.5? In other words: is the treatment
           effect better or worse identified than the parameters themselves?

Steps      For three designs (A = 0 only; A = 0 plus the full dose; all training doses):
             1. sample the posterior of all five parameters given one noise realization
                (Metropolis through the reference solver, flat prior on the robustness
                ranges config.PARAM_RANGES);
             2. solve for tau_0.5 under every sampled parameter vector;
             3. report the spread of the parameters next to the spread of the effect:
                bands of ATE(t), interval of GTE, largest pointwise SD of tau, and how
                far apart the sampled vectors are in their fit to the data.
           A second, deterministic set follows the weakest direction of a least-squares fit.

           With A = 0 only, eta is structurally absent from the data and the sampler
           returns its prior range. The resulting band is the honest statement of what
           untreated data say about a treatment: nothing beyond the prior.

Outputs    results/counterfactual/reliability.json
           results/figures/fig7_counterfactual_uncertainty.*
"""
from _common import cfg, io, parse_args, start, training_observations, Timer

import numpy as np

from src.counterfactual.counterfactual import counterfactual_effect
from src.counterfactual.uncertainty_propagation import (fit_spread, parameter_spread, propagate,
                                                        reliability_summary, sample_posterior, weak_direction_set)
from src.model.parameters import BASELINE_PARAM_NAMES, PARAM_NAMES, true_params
from src.solvers.fd_solver import frame_times
from src.solvers.reference_estimator import ReferenceEstimator
from src.utils import plotting
from src.utils.sampling import select_doses

A = cfg.HELDOUT_DOSE
DESIGNS = {'A = 0': (0.0,), 'A = 0 + 1': (0.0, 1.0), 'all training doses': cfg.TRAIN_DOSES}


def main():
    args = parse_args(__doc__.split('\n')[0])
    run, solver = start(f'Experiment 07: reliability of the counterfactual effect at A = {A}', args)
    truth, seed = true_params(), run.seeds[0]
    truth_effect = counterfactual_effect(solver, truth, A)
    obs_all = training_observations(run, solver, truth, seed)
    ranges = cfg.PARAM_RANGES
    mcmc = dict(n_samples=run.n_samples, batch_size=run.batch_size, **run.mcmc)

    results, summaries = {}, {}
    for label, doses in DESIGNS.items():
        with Timer(label):
            obs = select_doses(obs_all, doses)
            sets, diagnostics = sample_posterior(obs, solver, seed=seed, **mcmc)
            ensemble = propagate(solver, sets, A, design=obs, batch_size=run.batch_size)
            summary = reliability_summary(ensemble, truth_effect)

            # Deterministic companion: walk along the weakest direction of a bounded fit.
            treated = len(doses) > 1
            names = PARAM_NAMES if treated else BASELINE_PARAM_NAMES
            fit = ReferenceEstimator(solver, names).fit(obs, fixed=None if treated else {'eta': truth['eta']},
                                                        bounds=ranges)
            direction, steps = weak_direction_set(fit, bounds=ranges)
            along = propagate(solver, direction, A, batch_size=run.batch_size)

        summaries[label] = summary
        results[label] = {
            'sampler': diagnostics, 'n_samples': len(sets),
            'fit_spread_over_noise_median': float(np.median(fit_spread(ensemble))),
            'parameter_log_sd': parameter_spread(sets),
            'effect': summary,
            'weak_direction': {'fit': fit['params'], 'steps': steps, 'GTE': along['GTE'],
                               'CE_centre': along['CE_centre']},
        }
        relative = summary['relative']
        worst = max(diagnostics['r_hat'].values())
        print(f"    acceptance {diagnostics['acceptance_rate']:.2f}   largest R-hat {worst:.2f}"
              + ('   (not converged: increase MCMC_STEPS)' if worst > 1.1 else '')
              + f"   fit spread {results[label]['fit_spread_over_noise_median']:.2f} noise SD")
        spread = results[label]['parameter_log_sd']
        print('    spread of log parameter: ' + '  '.join(f'{k} {v:.2f}' for k, v in spread.items()))
        print(f"    GTE interval width / true GTE {relative['GTE_width']:.2f}   "
              f"max SD of tau / peak tau {relative['tau_sd_max']:.2f}   truth inside: {relative['gte_covered']}")

    io.save_json(run.results / 'counterfactual' / 'reliability.json',
                 {'dose': A, 'seed': seed, 'prior_ranges': ranges,
                  'truth': {'GTE': truth_effect['GTE'], 'peak_tau': truth_effect['peak'], 'ATE': truth_effect['ATE']},
                  'designs': results, 'meta': io.run_metadata()})
    plotting.plot_counterfactual_uncertainty(summaries, frame_times(solver), truth_effect['ATE'], dose=A,
                                             path=run.results / 'figures' / 'fig7_counterfactual_uncertainty')


if __name__ == '__main__':
    main()
