"""
Collection of tests for ivy gym demos
"""

# global
import pytest
import ivy.functional.backends.jax
import ivy.functional.backends.mxnet
import ivy.functional.backends.tensorflow
import ivy.functional.backends.torch
import ivy_tests.test_ivy.helpers as helpers

FWS = [ivy.functional.backends.jax, ivy.functional.backends.mxnet, ivy.functional.backends.tensorflow, ivy.functional.backends.torch]


@pytest.mark.parametrize(
    "env", ['CartPole', 'Pendulum', 'MountainCar', 'Reacher', 'Swimmer'])
def test_demo_run_through(env, device, f, call, fw):
    if call in [helpers.np_call, helpers.tf_graph_call]:
        # numpy does not support gradients, and demo compiles already, so no need to use tf_graph_call
        pytest.skip()
    from ivy_gym_demos.run_through import main
    main(env, visualize=False, f=f, fw=fw)


@pytest.mark.parametrize(
    "env", ['CartPole', 'Pendulum', 'MountainCar', 'Reacher', 'Swimmer'])
def test_demo_optimize_trajectory(env, device, f, call, fw):
    if call in [helpers.np_call, helpers.tf_graph_call]:
        # numpy does not support gradients, and demo compiles already, so no need to use tf_graph_call
        pytest.skip()
    from ivy_gym_demos.optimization.optimize_trajectory import main
    main(env, steps=1, iters=1, lr=0.1, seed=0, visualize=False, f=f, fw=fw)


@pytest.mark.parametrize(
    "env", ['CartPole', 'Pendulum', 'MountainCar', 'Reacher', 'Swimmer'])
def test_demo_optimize_policy(env, device, f, call, fw):
    if call in [helpers.np_call, helpers.tf_graph_call]:
        # numpy does not support gradients, and demo compiles already, so no need to use tf_graph_call
        pytest.skip()
    from ivy_gym_demos.optimization.optimize_policy import main
    main(env, steps=1, iters=1, lr=0.1, seed=0, visualize=False, f=f, fw=fw)
