"""Forward PINN: solve the treatment model at known parameters (Stage 1).

All five physical parameters are pinned at given values and only the network is
trained, on the PDE residual, the initial condition and the wall condition. One network
covers every dose. Agreement with the finite-difference fields at all treatment levels
(Figure 2) shows that the nonlinear residual is implemented correctly before any
inverse problem is attempted.
"""
import numpy as np

import config as cfg
from src.model.parameters import raw_dict, true_params
from src.pinn.losses import make_batch, make_loss_terms
from src.pinn.network import ConcentrationNet, predict_field
from src.pinn.optimizer import TrainConfig, train
from src.utils.metrics import relative_l2
from src.utils.sampling import collocation_points


def train_forward(params=None, tc=None, seed=cfg.NET_SEED, net=None, points=None, release=None,
                  label='forward'):
    """Train the network at known parameters.

    params  physical parameters (default: the synthetic truth)
    tc      schedule; default uses the forward weights (w_d = 0)
    points  collocation set; default `collocation_points(seed=seed)` over continuous doses

    Returns a dict with the loss terms, the minutes, and the handles '_net' and
    '_weights' for `network.predict`.
    """
    params = params or true_params()
    tc = tc or TrainConfig(weights=cfg.W_FORWARD)
    net = net or ConcentrationNet()
    points = collocation_points(seed=seed) if points is None else points
    terms = make_loss_terms(net, pinned=raw_dict(params), release=release)
    p, info = train({'net': net.init(seed)}, terms, make_batch(points), tc, label=label)
    return {'params': dict(params), 'layers': list(net.layers), 'n_weights': net.n_weights,
            'terms': info['terms'], 'loss': info['loss'], 'minutes': info['minutes'],
            '_net': net, '_weights': p['net']}


def forward_errors(result, solver, doses=cfg.ALL_DOSES, snapshots=(1.0, 5.0, 10.0, 20.0, 50.0),
                   time_stride=10, release=None):
    """Relative L2 error (%) of the forward PINN against the finite-difference field.

    Returns {dose: {'field': error over the whole record, 't=...': error of each snapshot}}.
    The comparison uses every `time_stride`-th stored frame on the full spatial grid.
    """
    errors = {}
    for A in doses:
        reference = np.asarray(solver(result['params'], A, release=release))[::time_stride]
        network = predict_field(result['_net'], result['_weights'], solver, A, time_stride)
        entry = {'field': 100.0 * relative_l2(network, reference)}
        for t in snapshots:
            i = int(round(t / (solver.dt_store * time_stride)))
            entry[f't={t:g}'] = 100.0 * relative_l2(network[i], reference[i])
        errors[float(A)] = entry
    return errors
