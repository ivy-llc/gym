"""
Collection of tests for ivy gym demos
"""

# global
import ivy.jax
import ivy.mxnd
import ivy.tensorflow
import ivy.torch

FWS = [ivy.jax, ivy.mxnd, ivy.tensorflow, ivy.torch]


def test_demo_run_through():
    from demos.run_through import main
    for f in FWS:
        for env in ['CartPole', 'Pendulum', 'MountainCar', 'Reacher', 'Swimmer']:
            main(env, visualize=False, f=f)


def test_demo_optimize_trajectory():
    from demos.optimization.optimize_trajectory import main
    for f in FWS:
        for env in ['CartPole', 'Pendulum', 'MountainCar', 'Reacher', 'Swimmer']:
            main(env, steps=1, iters=1, lr=0.1, seed=0, visualize=False, f=f)


def test_demo_optimize_policy():
    from demos.optimization.optimize_policy import main
    for f in FWS:
        for env in ['CartPole', 'Pendulum', 'MountainCar', 'Reacher', 'Swimmer']:
            main(env, steps=1, iters=1, lr=0.1, seed=0, visualize=False, f=f)
