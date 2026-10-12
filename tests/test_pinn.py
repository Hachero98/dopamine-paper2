"""PINN smoke tests: derivatives, loss terms, a short training run, frozen parameters."""
import numpy as np
import jax.numpy as jnp
import pytest

import config as cfg
from src.model.parameters import raw_dict
from src.pinn.losses import make_batch, make_loss_terms, point_derivatives
from src.pinn.network import ConcentrationNet, predict, predict_dose_sensitivity, predict_field
from src.pinn.optimizer import TrainConfig, train
from src.pinn.train_inverse import train_inverse, train_staged
from src.utils.sampling import collocation_points, observe, scattered_design

SMALL = dict(n_domain=200, n_initial=40, n_boundary=40)


@pytest.fixture(scope='module')
def net():
    return ConcentrationNet(layers=(4, 12, 12, 1))


def test_point_derivatives_of_a_known_function():
    f = lambda z: jnp.exp(-0.3 * z[2]) * jnp.cos(z[0]) * jnp.sin(2.0 * z[1]) * (1.0 + z[3])
    z = jnp.array([0.4, -0.7, 2.0, 0.5])
    value, gradient, laplacian = point_derivatives(f, z)
    assert float(gradient[2]) == pytest.approx(-0.3 * float(value))          # C_t
    assert float(laplacian) == pytest.approx(-5.0 * float(value))            # C_xx + C_yy
    assert float(gradient[3]) == pytest.approx(float(value) / 1.5)           # C_A


def test_network_shapes_and_input_normalisation(net, solver):
    weights = net.init(0)
    values = predict(net, weights, np.zeros((3, 2)), 0.0, 1.0, 0.5)
    assert values.shape == (3, 2) and np.all(np.isfinite(values))
    assert predict_dose_sensitivity(net, weights, np.zeros(4), 0.0, 1.0, 0.5).shape == (4,)
    assert predict_field(net, weights, solver, 0.5, time_stride=100).shape == (6, solver.nx, solver.nx)
    scaled = ConcentrationNet(layers=(4, 12, 12, 1), normalize=True)
    assert np.isfinite(float(scaled.apply(scaled.init(0))(jnp.array([2.5, -2.5, 50.0, 1.0]))))
    with pytest.raises(ValueError):
        ConcentrationNet(layers=(3, 8, 1))


def test_loss_terms_are_finite_and_training_lowers_the_loss(net, solver, truth):
    obs = observe(scattered_design((0.0, 1.0), 30, seed=1), solver, truth)
    batch = make_batch(collocation_points(**SMALL), obs)
    terms = make_loss_terms(net)
    p = {'net': net.init(1), 'raw': raw_dict(cfg.INIT_PARAMS)}
    before = np.asarray(terms(p, batch))
    assert before.shape == (4,) and np.all(np.isfinite(before))
    _, info = train(p, terms, batch, TrainConfig.quick())
    assert np.isfinite(info['loss']) and info['loss'] < float(np.dot(cfg.W_INVERSE, before))


def test_frozen_parameters_do_not_move(net, solver, truth):
    obs = observe(scattered_design((0.0,), 30, seed=1), solver, truth)
    result = train_inverse(obs, free=('D', 'k'), tc=TrainConfig.quick(), net=net,
                           points=collocation_points(**SMALL, doses=(0.0,)))
    assert result['free'] == ['D', 'k']
    for name in ('Vmax0', 'Km', 'eta'):
        assert result['params'][name] == pytest.approx(cfg.INIT_PARAMS[name], rel=1e-12)
    assert result['params']['D'] != pytest.approx(cfg.INIT_PARAMS['D'], rel=1e-9)


def test_staged_training_runs_every_stage(net, solver, truth):
    obs = observe(scattered_design((0.0, 1.0), 30, seed=1), solver, truth)
    schedule = {stage: TrainConfig.quick(adam_iters=50, lbfgs_iters=5) for stage in cfg.STAGE_SCHEDULE}
    result = train_staged(obs, schedule=schedule, net=net, **SMALL)
    assert list(result['stages']) == ['baseline', 'treatment_response', 'joint']
    # eta is untouched by the baseline stage and learned afterwards.
    assert result['stages']['baseline']['params']['eta'] == pytest.approx(cfg.INIT_PARAMS['eta'], rel=1e-12)
    assert result['stages']['treatment_response']['params']['D'] == pytest.approx(
        result['stages']['baseline']['params']['D'], rel=1e-12)
    assert all(np.isfinite(v) for v in result['params'].values())


def test_inverse_refuses_eta_from_untreated_data(net, solver, truth):
    obs = observe(scattered_design((0.0,), 10, seed=1), solver, truth)
    with pytest.raises(ValueError, match='eta cannot be estimated'):
        train_inverse(obs, tc=TrainConfig.quick(), net=net)
