.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='docs/partial_source/logos/logo.png'>
    </p>

.. raw:: html

    <br/>
    <a href="https://pypi.org/project/ivy-gym">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/ivy-gym.svg">
    </a>
    <a href="https://github.com/ivy-dl/gym/actions?query=workflow%3Adocs">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/workflow/status/ivy-dl/gym/docs?label=docs">
    </a>
    <a href="https://github.com/ivy-dl/gym/actions?query=workflow%3Anightly-tests">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/workflow/status/ivy-dl/gym/nightly-tests?label=tests">
    </a>
    <a href="https://discord.gg/EN9YS3QW8w">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/799879767196958751?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
    <br clear="all" />

**Fully differentiable reinforcement learning environments, written in Ivy.**

.. raw:: html

    <div style="display: block;">
        <img width="4%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://jax.readthedocs.io">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/jax_logo.png">
        </a>
        <img width="6.5%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://www.tensorflow.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/tensorflow_logo.png">
        </a>
        <img width="6.5%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://pytorch.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/pytorch_logo.png">
        </a>
        <img width="6.5%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://mxnet.apache.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/mxnet_logo.png">
        </a>
        <img width="6.5%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://numpy.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/numpy_logo.png">
        </a>
    </div>

Contents
--------

* `Overview`_
* `Run Through`_
* `Optimization Demos`_
* `Get Involed`_

Overview
--------

.. _docs: https://ivy-dl.org/gym

**What is Ivy Gym?**

Ivy Gym opens the door for intersectional research between supervised learning (SL), reinforcement learning (RL),
and trajectory optimization (TO),
by implementing RL environments in a fully differentiable manner.

Specifically, Ivy gym provides differentiable implementations of the classic control tasks from OpenAI Gym,
as well as a new Swimmer task, which illustrates the simplicity of creating new tasks using Ivy.
The differentiable nature of the environments means that the cumulative reward can be directly optimized for in a supervised manner,
without need for reinforcement learning, which is the de facto approach for optimizing cumulative rewards. Check out the docs_ for more info!

The library is built on top of the Ivy deep learning framework.
This means all environments simultaneously support:
Jax, Tensorflow, PyTorch, MXNet, and Numpy.

**Ivy Libraries**

There are a host of derived libraries written in Ivy, in the areas of mechanics, 3D vision, robotics,
differentiable memory, and differentiable gym environments. Click on the icons below for their respective github pages.

.. raw:: html

    <div style="display: block;">
        <img width="9%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/mech">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_mech.png">
        </a>
        <img width="7%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/vision">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_vision.png">
        </a>
        <img width="7%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/robot">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_robot.png">
        </a>
        <img width="7%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/gym">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_gym.png">
        </a>

        <br clear="all" />

        <img width="10%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-mech">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-mech.svg">
        </a>
        <img width="9%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-vision">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-vision.svg">
        </a>
        <img width="9%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-robot">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-robot.svg">
        </a>
        <img width="9%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-gym">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-gym.svg">
        </a>

        <br clear="all" />

        <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/mech/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/mech/nightly-tests?label=tests">
        </a>
        <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/vision/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/vision/nightly-tests?label=tests">
        </a>
        <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/robot/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/robot/nightly-tests?label=tests">
        </a>
        <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/gym/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/gym/nightly-tests?label=tests">
        </a>

        <br clear="all" />

        <img width="20%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/memory">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_memory.png">
        </a>
        <img width="7%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/builder">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_builder.png">
        </a>
        <img width="7%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/models">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_models.png">
        </a>

        <br clear="all" />

        <img width="21%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-memory">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-memory.svg">
        </a>
        <img width="9%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-builder">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-builder.svg">
        </a>
        <img width="9%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-models">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-models.svg">
        </a>

        <br clear="all" />

        <img width="23%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/memory/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/memory/nightly-tests?label=tests">
        </a>
        <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/builder/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/builder/nightly-tests?label=tests">
        </a>
        <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/models/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/models/nightly-tests?label=tests">
        </a>

        <br clear="all" />

    </div>
    <br clear="all" />

**Quick Start**

Ivy gym can be installed like so: ``pip install ivy-gym``

.. _demos: https://github.com/ivy-dl/gym/tree/master/ivy_gym_demos
.. _optimization: https://github.com/ivy-dl/gym/tree/master/ivy_gym_demos/optimization

To quickly see the different environments provided, we suggest you check out the demos_!
We suggest you start by running the script ``run_through.py``,
and read the "Run Through" section below which explains this script.

For demos which optimize performance on the different tasks, we suggest you run either
``optimize_trajectory.py`` or ``optimize_policy.py`` in the optimization_ demos folder.

Run Through
-----------

The different environemnts can be visualized via a simple script,
which executes random motion for 250 steps in one of the environments.
The script is available in the demos_ folder, as file ``run_through.py``.
First, we select a random backend framework to use for the examples, from the options
``ivy.jax``, ``ivy.tensorflow``, ``ivy.torch``, ``ivy.mxnet`` or ``ivy.numpy``,
and use this to set the ivy backend framework.

.. code-block:: python

    import ivy
    from ivy_demo_utils.framework_utils import choose_random_framework
    ivy.set_framework(choose_random_framework())

We then select an environment to use and execute 250 random actions,
while rendering the environment after each step.

By default, the demos all use the ``CartPole`` environment, but this can be changed using the ``--env`` argument,
choosing from the options ``CartPole``, ``Pendulum``, ``MountainCar``, ``Reacher`` or ``Swimmer``.

.. code-block:: python

    env = getattr(ivy_gym, env_str)()

    env.reset()
    ac_dim = env.action_space.shape[0]
    for _ in range(250):
        ac = ivy.random_uniform(-1, 1, (ac_dim,))
        env.step(ac)
        env.render()

Here, we briefly discuss each of the five environments,
before showing example episodes from a learnt policy network.
We use a learnt policy in these visualizations rather than random actions as used in the script,
because we find this to be more descriptive for visually explaining each task.
We also plot the instantaneous reward corresponding to each frame.

**CartPole**

For this task, a pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
The system is controlled by applying a force to the cart.
A reward is given based on the angle of the pendulum from being upright.
Example trajectories are given below.

.. raw:: html

    <p align="center">
        <img width="40%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_gym/cartpole.gif?raw=true'>
    </p>

**MountainCar**

For this task, a car is on a one-dimensional track, positioned between two "mountains".
The goal is to drive up the mountain on the right.
However, the car's engine is not strong enough to scale the mountain in a single pass.
Therefore, the only way to succeed is to drive back and forth to build up momentum.
Here, the reward is greater if you spend less energy to reach the goal.
Example trajectories are given below.

.. raw:: html

    <p align="center">
        <img width="40%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_gym/mountain_car.gif?raw=true'>
    </p>

**Pendulum**

For this task, an inverted pendulum starts in a random position, and the goal is to swing it up so it stays upright.
Again, a reward is given based on the angle of the pendulum from being upright.
Example trajectories are given below.

.. raw:: html

    <p align="center">
        <img width="40%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_gym/pendulum.gif?raw=true'>
    </p>

**Reacher**

For this task, a 2-link robot arm must reach a target position.
Reward is given based on the distance of the end effector to the target.
Example trajectories are given below.

.. raw:: html

    <p align="center">
        <img width="40%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_gym/reacher.gif?raw=true'>
    </p>

**Swimmer**

We implemented this task ourselves, in order to highlight the simplicity of creating new custom environments.
For this task, a fish must swim to reach a target 2D positions whilst avoiding sharp obstacles.
Reward is given for being close to the target, and negative reward is given for colliding with the sharp objects.
Example trajectories are given below.

.. raw:: html

    <p align="center">
        <img width="40%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_gym/swimmer.gif?raw=true'>
    </p>

Optimization Demos
------------------

We provide two demo scripts which optimize performance on these tasks in a supervised manner,
either via trajectory optimization or policy optimization.

In the case of trajectory optimization, we optimize for a specific starting state of the environment,
whereas for policy optimization we train a policy network which is conditioned on the environment state,
and the starting state is then randomized between training steps.

Rather than presenting the code here, we show visualizations of the demos.
The scripts for these demos can be found in the optimization_ demos folder.

**Trajectory Optimization**

In this demo, we show trajectories on each of the five ivy gym environments during the course of trajectory optimization.
The optimization iteration is shown in the bottom right, along with the step in the environment.

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_gym/demo_a.gif?raw=true'>
    </p>

**Policy Optimization**

In this demo, we show trajectories on each of the five ivy gym environments during the course of policy optimization.
The optimization iteration is shown in the bottom right, along with the step in the environment.

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_gym/demo_b.gif?raw=true'>
    </p>

Get Involed
-----------

We hope the differentiable environments in this library are useful to a wide range of deep learning developers.
However, there are many more tasks which could be implemented.

If there are any particular tasks you feel are missing,
or you would like to implement your own custom task,
then we are very happy to accept pull requests!

We look forward to working with the community on expanding and improving the Ivy gym library.

Citation
--------

::

    @article{lenton2021ivy,
      title={Ivy: Templated Deep Learning for Inter-Framework Portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }