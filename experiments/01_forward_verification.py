"""Experiment 01 - nonlinear forward verification (Stage 1).

Question   Is the forward problem solved correctly, by the reference solver and by the
           PINN residual, at every treatment level?

Steps      1. Verify the finite-difference solver: observed order in space and time,
              limiting cases, positivity. The script stops here if a check fails,
              because no inverse experiment may start from an unconverged solver.
           2. Generate the reference fields C_true(x, y, t; A) for all five doses.
           3. Train one forward PINN at the true parameters and compare it with the
              reference field at every dose (Figure 2).
           4. Optional (--pilot): small architecture / weight pilot on the forward problem.

Outputs    results/forward/solver_verification.json, forward_pinn.json, pilot.json
           results/figures/solver_convergence.*, fig2_forward_agreement.*
           data/reference/fields_truth.npz
"""
from _common import cfg, io, new_net, parse_args, start, Timer

import numpy as np

from src.model.parameters import true_params
from src.pinn.network import ConcentrationNet, predict_field
from src.pinn.train_forward import forward_errors, train_forward
from src.solvers.convergence_test import verify
from src.solvers.fd_solver import frame_times, solve_doses
from src.utils import plotting
from src.utils.sampling import collocation_points

# Pilot grid: each entry overrides the baseline configuration. The pilot is scored on the
# training doses only; the held-out dose takes no part in choosing a configuration.
PILOT = {
    'paper1': dict(),
    'normalised_inputs': dict(normalize=True),
    'wide_128': dict(layers=(4, 128, 128, 128, 128, 1)),
    'deep_6x64': dict(layers=(4, 64, 64, 64, 64, 64, 64, 1)),
    'residual_weight_x10': dict(weights=(100.0, 1.0, 1.0, 0.0)),
}
FIELD_STRIDE = 5   # reference fields are stored every FIELD_STRIDE frames (0.5 ms)


def main():
    args = parse_args(__doc__.split('\n')[0], pilot=dict(action='store_true', help='run the architecture pilot'))
    run, solver = start('Experiment 01: nonlinear forward verification', args)
    truth = true_params()
    out = run.results / 'forward'

    # ---- 1. solver verification ---------------------------------------------------------
    with Timer('solver verification'):
        report = verify(**run.verification)
    io.save_json(out / 'solver_verification.json', {**report, 'meta': io.run_metadata()})
    plotting.plot_convergence(report['spatial'], report['temporal'], run.results / 'figures' / 'solver_convergence')
    orders = report['spatial']['order'][-1], report['temporal']['order'][-1]
    print(f'  spatial order {orders[0]:.3f}   temporal order {orders[1]:.3f}')
    for name, value in report['limiting_cases'].items():
        print(f'  {name:<24s}{value:.2e}')
    if not report['passed']:
        raise SystemExit('Solver verification FAILED. Do not start inverse experiments (decision rule 1).')
    print('  solver verification passed')

    # ---- 2. reference fields ------------------------------------------------------------
    fields = np.asarray(solve_doses(solver, truth, cfg.ALL_DOSES))
    if not args.quick:
        io.save_arrays(cfg.REFERENCE_DIR / 'fields_truth.npz', doses=np.array(cfg.ALL_DOSES), x=solver.x,
                       t=frame_times(solver)[::FIELD_STRIDE], C=fields[:, ::FIELD_STRIDE].astype(np.float32),
                       **{f'param_{name}': value for name, value in truth.items()})
    print('  peak concentration at t = T per dose (uM):',
          {A: round(float(fields[i, -1].max()), 5) for i, A in enumerate(cfg.ALL_DOSES)})
    if args.skip_pinn:
        return

    # ---- 3. forward PINN ----------------------------------------------------------------
    points = collocation_points(**run.collocation)
    result = train_forward(truth, run.forward, net=new_net(run), points=points)
    errors = forward_errors(result, solver)
    io.save_json(out / 'forward_pinn.json', {**result, 'errors_percent': errors, 'meta': io.run_metadata()})
    for A, entry in errors.items():
        print(f"  A = {A:<5g} relative L2 error {entry['field']:.3f} %")
    stride = 10
    reference = {A: fields[i, ::stride] for i, A in enumerate(cfg.ALL_DOSES)}
    network = {A: predict_field(result['_net'], result['_weights'], solver, A, stride) for A in cfg.ALL_DOSES}
    plotting.plot_forward_agreement(reference, network, solver, t=5.0, time_stride=stride,
                                    path=run.results / 'figures' / 'fig2_forward_agreement')

    # ---- 4. architecture / weight pilot -------------------------------------------------
    if args.pilot:
        pilot = {}
        for label, change in PILOT.items():
            net = ConcentrationNet(layers=change.get('layers', run.layers), normalize=change.get('normalize', False))
            tc = run.forward.replace(weights=change.get('weights', run.forward.weights))
            trial = train_forward(truth, tc, net=net, points=points, label=f'pilot/{label}')
            scores = forward_errors(trial, solver, doses=cfg.TRAIN_DOSES)
            pilot[label] = {'change': change, 'errors_percent': scores, 'minutes': trial['minutes'],
                            'mean_error_percent': float(np.mean([e['field'] for e in scores.values()]))}
            print(f"  pilot {label:<22s} mean error {pilot[label]['mean_error_percent']:.3f} %")
        io.save_json(out / 'pilot.json',
                     {'pilot': pilot, 'scored_on_doses': cfg.TRAIN_DOSES, 'meta': io.run_metadata()})


if __name__ == '__main__':
    main()
