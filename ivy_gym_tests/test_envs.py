"""
Collection of tests for differentiable gym environments written with Ivy.
"""

# global
import ivy.functional.backends.jax
import ivy.functional.backends.mxnet
import ivy.functional.backends.numpy
import ivy.functional.backends.tensorflow
import ivy.functional.backends.torch

# local
from ivy_gym.cartpole import CartPole
from ivy_gym.mountain_car import MountainCar
from ivy_gym.pendulum import Pendulum
from ivy_gym.reacher import Reacher
from ivy_gym.swimmer import Swimmer

FWS = [ivy.functional.backends.jax, ivy.functional.backends.mxnet, ivy.functional.backends.numpy, ivy.functional.backends.tensorflow, ivy.functional.backends.torch]


def _test_env(env):
    ac_dim = env.action_space.shape[0]
    env.reset()
    for _ in range(10):
        ac = ivy.random_uniform(-1, 1, (ac_dim,))
        env.step(ac)


def test_cartpole(dev_str, call):
    _test_env(CartPole())


def test_mountain_car(dev_str, call):
    _test_env(MountainCar())


def test_pendulum(dev_str, call):
    _test_env(Pendulum())


def test_reacher(dev_str, call):
    _test_env(Reacher())


def test_swimmer(dev_str, call):
    _test_env(Swimmer())
