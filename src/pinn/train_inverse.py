"""Inverse PINN: estimate the physical parameters from sparse, noisy observations.

`train_inverse` runs one optimisation with a chosen set of free parameters.
`train_staged` chains the three stages of the study and carries the network and the
parameter estimates from one stage to the next:

    1. baseline             A = 0 data only; learns D, Vmax0, Km, k            (Stage 3)
    2. treatment_response   all training doses; learns eta, baseline frozen    (Stage 4)
    3. joint                all training doses; refines all five together      (Stage 5)

One set of baseline parameters is shared by every dose. There is never a separate copy
of D, Vmax0, Km or k per dose: the doses differ only through Vmax(A) = Vmax0 (1 - eta A).

The held-out dose is rejected by every entry point, so it cannot enter a fit by mistake.
"""
import numpy as np

import config as cfg
from src.model.parameters import PARAM_NAMES, BASELINE_PARAM_NAMES, initial_params, raw_dict
from src.pinn.losses import make_batch, make_loss_terms
from src.pinn.network import ConcentrationNet
from src.pinn.optimizer import TrainConfig, train
from src.utils.sampling import assert_heldout_excluded, collocation_points, select_doses

STAGES = ('baseline', 'treatment_response', 'joint')


def train_inverse(obs, free=PARAM_NAMES, init=None, tc=TrainConfig(), seed=cfg.NET_SEED, net=None,
                  points=None, weights=None, release=None, label='inverse'):
    """Fit the network and the parameters in `free` to one observation set.

    free     names of the parameters to estimate; the others stay at their `init` value
    init     physical starting values (default: config.INIT_PARAMS, never the truth)
    points   collocation set (default: `collocation_points(seed=seed)`)
    weights  network weights to continue from (default: a fresh initialisation)

    Returns a dict:
        params       all five physical parameters after training
        free         the parameters that were estimated
        terms, loss  final loss terms; rms_data is sqrt(L_data) in uM
        trajectory   [(step, {name: value}), ...] of the free parameters
        _net, _weights   handles for `network.predict`
    """
    assert_heldout_excluded(obs)
    if 'eta' in free and not np.any(np.asarray(obs['A_d']) > 0):
        raise ValueError('eta cannot be estimated from untreated data: at A = 0 the model does not depend on eta')
    net = net or ConcentrationNet()
    points = collocation_points(seed=seed) if points is None else points
    raw = raw_dict({**initial_params(), **(init or {})})
    pinned = {name: value for name, value in raw.items() if name not in free}
    terms = make_loss_terms(net, pinned=pinned, release=release)

    p = {'net': net.init(seed) if weights is None else weights, 'raw': {name: raw[name] for name in free}}
    p, info = train(p, terms, make_batch(points, obs), tc, label=label)
    return {'params': {name: float(value) for name, value in terms.physical(p).items()},
            'free': list(free), 'terms': info['terms'], 'loss': info['loss'],
            'rms_data': float(np.sqrt(info['terms']['data'])), 'trajectory': info['trajectory'],
            'minutes': info['minutes'], '_net': net, '_weights': p['net']}


def default_schedule(weights=cfg.W_INVERSE, **overrides):
    """{stage: TrainConfig} from config.STAGE_SCHEDULE; overrides apply to every stage."""
    return {stage: TrainConfig(adam_iters=adam, lbfgs_iters=lbfgs, weights=weights, **overrides)
            for stage, (adam, lbfgs) in cfg.STAGE_SCHEDULE.items()}


def train_staged(obs, init=None, schedule=None, seed=cfg.NET_SEED, net=None, release=None, label='staged',
                 **collocation):
    """Baseline -> treatment-response -> joint refinement on a multi-dose observation set.

    obs          observations at A = 0 and at one or more treated training doses
    schedule     {stage: TrainConfig}; default `default_schedule()`
    collocation  keyword arguments of `collocation_points` (e.g. n_domain) for every stage

    The baseline stage enforces the physics at A = 0 only, where its data are; the later
    stages enforce it over the whole dose range. Returns the result of the joint stage
    with an extra entry 'stages' = {stage: {'params', 'terms', 'rms_data', 'minutes'}}.
    """
    assert_heldout_excluded(obs)
    schedule = schedule or default_schedule()
    net = net or ConcentrationNet()
    points_baseline = collocation_points(seed=seed, **{**collocation, 'doses': (0.0,)})
    points_all = collocation_points(seed=seed, **collocation)

    plan = [('baseline', select_doses(obs, 0.0), BASELINE_PARAM_NAMES, points_baseline),
            ('treatment_response', obs, ('eta',), points_all),
            ('joint', obs, PARAM_NAMES, points_all)]
    estimate, weights, stages = {**initial_params(), **(init or {})}, None, {}
    for stage, data, free, points in plan:
        result = train_inverse(data, free=free, init=estimate, tc=schedule[stage], seed=seed, net=net,
                               points=points, weights=weights, release=release, label=f'{label}/{stage}')
        estimate, weights = result['params'], result['_weights']
        stages[stage] = {key: result[key] for key in ('params', 'terms', 'rms_data', 'minutes')}
    return {**result, 'stages': stages, 'minutes': sum(s['minutes'] for s in stages.values())}
