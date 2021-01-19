"""
Collection of tests for differentiable gym environments written with Ivy.
"""

# global
import ivy.jax
import ivy.mxnd
import ivy.mxsym
import ivy.numpy
import ivy.tensorflow
import ivy.torch

# local
from ivy_gym.cartpole import CartPole
from ivy_gym.mountain_car import MountainCar
from ivy_gym.pendulum import Pendulum
from ivy_gym.reacher import Reacher
from ivy_gym.swimmer import Swimmer

FWS = [ivy.jax, ivy.mxnd, ivy.mxsym, ivy.numpy, ivy.tensorflow, ivy.torch]


def _test_env(env, f):
    ac_dim = env.action_space.shape[0]
    env.reset()
    for _ in range(10):
        ac = ivy.random_uniform(-1, 1, (ac_dim,), f=f)
        env.step(ac)


def test_cartpole():
    for f in FWS:
        _test_env(CartPole(f=f), f)


def test_mountain_car():
    for f in FWS:
        _test_env(MountainCar(f=f), f)


def test_pendulum():
    for f in FWS:
        _test_env(Pendulum(f=f), f)


def test_reacher():
    for f in FWS:
        _test_env(Reacher(f=f), f)


def test_swimmer():
    for f in FWS:
        _test_env(Swimmer(f=f), f)
