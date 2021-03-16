"""
Collection of tests for differentiable gym environments written with Ivy.
"""

# global
import ivy.jax
import ivy.mxnd
import ivy.numpy
import ivy.tensorflow
import ivy.torch

# local
from ivy_gym.cartpole import CartPole
from ivy_gym.mountain_car import MountainCar
from ivy_gym.pendulum import Pendulum
from ivy_gym.reacher import Reacher
from ivy_gym.swimmer import Swimmer

FWS = [ivy.jax, ivy.mxnd, ivy.numpy, ivy.tensorflow, ivy.torch]


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
