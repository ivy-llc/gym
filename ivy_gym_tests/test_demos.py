"""
Collection of tests for ivy gym demos
"""

# global
import pytest
import ivy.jax
import ivy.mxnet
import ivy.tensorflow
import ivy.torch
import ivy_tests.helpers as helpers

FWS = [ivy.jax, ivy.mxnet, ivy.tensorflow, ivy.torch]


@pytest.mark.parametrize(
    "env", ['CartPole', 'Pendulum', 'MountainCar', 'Reacher', 'Swimmer'])
def test_demo_run_through(env, dev_str, f, call):
    if call in [helpers.np_call, helpers.tf_graph_call]:
        # numpy does not support gradients, and demo compiles already, so no need to use tf_graph_call
        pytest.skip()
    from demos.run_through import main
    main(env, visualize=False, f=f)


@pytest.mark.parametrize(
    "env", ['CartPole', 'Pendulum', 'MountainCar', 'Reacher', 'Swimmer'])
def test_demo_optimize_trajectory(env, dev_str, f, call):
    if call in [helpers.np_call, helpers.tf_graph_call]:
        # numpy does not support gradients, and demo compiles already, so no need to use tf_graph_call
        pytest.skip()
    from demos.optimization.optimize_trajectory import main
    main(env, steps=1, iters=1, lr=0.1, seed=0, visualize=False, f=f)


@pytest.mark.parametrize(
    "env", ['CartPole', 'Pendulum', 'MountainCar', 'Reacher', 'Swimmer'])
def test_demo_optimize_policy(env, dev_str, f, call):
    if call in [helpers.np_call, helpers.tf_graph_call]:
        # numpy does not support gradients, and demo compiles already, so no need to use tf_graph_call
        pytest.skip()
    from demos.optimization.optimize_policy import main
    main(env, steps=1, iters=1, lr=0.1, seed=0, visualize=False, f=f)
