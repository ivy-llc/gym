# global
import pytest
from typing import Dict

# local
# from ivy_tests.test_ivy import helpers
import ivy
import ivy.ivy_tests as ivy_tests

FW_STRS = ['numpy', 'jax', 'tensorflow', 'torch', 'mxnet']


TEST_FRAMEWORKS: Dict[str, callable] = {'numpy': lambda: ivy_tests.test_ivy.helpers.get_ivy_numpy(),
                                        'jax': lambda: ivy_tests.test_ivy.helpers.get_ivy_jax(),
                                        'tensorflow': lambda: ivy_tests.test_ivy.helpers.get_ivy_tensorflow(),
                                        'torch': lambda: ivy_tests.test_ivy.helpers.get_ivy_torch(),
                                        'mxnet': lambda: ivy_tests.test_ivy.helpers.get_ivy_mxnet()}
TEST_CALL_METHODS: Dict[str, callable] = {'numpy': ivy_tests.test_ivy.helpers.np_call,
                                          'jax': ivy_tests.test_ivy.helpers.jnp_call,
                                          'tensorflow': ivy_tests.test_ivy.helpers.tf_call,
                                          'torch': ivy_tests.test_ivy.helpers.torch_call,
                                          'mxnet': ivy_tests.test_ivy.helpers.mx_call}


@pytest.fixture(autouse=True)
def run_around_tests(dev_str, f, wrapped_mode, compile_graph, call):
    if wrapped_mode and call is ivy_tests.test_ivy.helpers.tf_graph_call:
        # ToDo: add support for wrapped_mode and tensorflow compilation
        pytest.skip()
    if wrapped_mode and call is ivy_tests.test_ivy.helpers.jnp_call:
        # ToDo: add support for wrapped_mode with jax, presumably some errenously wrapped jax methods
        pytest.skip()
    if 'gpu' in dev_str and call is ivy_tests.test_ivy.helpers.np_call:
        # Numpy does not support GPU
        pytest.skip()
    ivy.clear_framework_stack()
    with f.use:
        f.set_wrapped_mode(wrapped_mode)
        ivy.set_default_device(dev_str)
        yield


def pytest_generate_tests(metafunc):

    # dev_str
    raw_value = metafunc.config.getoption('--dev_str')
    if raw_value == 'all':
        dev_strs = ['cpu', 'gpu:0', 'tpu:0']
    else:
        dev_strs = raw_value.split(',')

    # framework
    raw_value = metafunc.config.getoption('--framework')
    if raw_value == 'all':
        f_strs = TEST_FRAMEWORKS.keys()
    else:
        f_strs = raw_value.split(',')

    # wrapped_mode
    raw_value = metafunc.config.getoption('--wrapped_mode')
    if raw_value == 'both':
        wrapped_modes = [True, False]
    elif raw_value == 'true':
        wrapped_modes = [True]
    else:
        wrapped_modes = [False]

    # compile_graph
    raw_value = metafunc.config.getoption('--compile_graph')
    if raw_value == 'both':
        compile_modes = [True, False]
    elif raw_value == 'true':
        compile_modes = [True]
    else:
        compile_modes = [False]

    # create test configs
    configs = list()
    for f_str in f_strs:
        for dev_str in dev_strs:
            for wrapped_mode in wrapped_modes:
                for compile_graph in compile_modes:
                    configs.append(
                        (dev_str, TEST_FRAMEWORKS[f_str](), wrapped_mode, compile_graph, TEST_CALL_METHODS[f_str]))
    metafunc.parametrize('dev_str,f,wrapped_mode,compile_graph,call', configs)


def pytest_addoption(parser):
    parser.addoption('--dev_str', action="store", default="cpu")
    parser.addoption('--framework', action="store", default="numpy,jax,tensorflow,torch,mxnet")
    parser.addoption('--wrapped_mode', action="store", default="false")
    parser.addoption('--compile_graph', action="store", default="true")
