# Ivy Gym Demos

We provide a simple set of interactive demos for the Ivy Gym library.
Running these demos is quick and simple.

## Install

First, clone this repo, and install the requirements provided in this demos folder like so:

```bash
git clone https://github.com/ivy-dl/gym.git ~/ivy_gym
cd ~/ivy_gym/demos
python3 -m pip install -r requirements.txt
```

## Demos

All demos can be run by executing the python scripts directly.
If a demo script is run without command line arguments, then a random backend framework will be selected from those installed.
Alternatively, the `--framework` argument can be used to manually specify a framework from the options
`jax`, `tensorflow`, `torch`, `mxnd` or `numpy`.

By default, the demos all use the `CartPole` environment, but this can be changed using the `--env` argument,
choosing from the options `CartPole`, `Pendulum`, `MountainCar`, `Reacher` or `Swimmer`.

To further explore the demos, breakpoints can be added to the scripts at any stage.
Adding `import pdb; pdb.set_trace()` works for python < 3.7,
and the built-in `breakpoint()` can be used for python > 3.7.

### Run Through

For a basic run through each of the gym environments:

```bash
cd ~/ivy_gym/demos
python3 run_through.py
```

This script, and the different gym environments, are further discussed in the [Run Through](https://github.com/ivy-dl/gym#run-through) section of the main README.
We advise following along with this section for maximum effect. The demo script should also be opened locally,
and breakpoints added to step in at intermediate points to further explore.

To run the script using a specific backend, tensorflow for example, then run like so:

```bash
python3 run_through.py --framework tensorflow
```

### Trajectory Optimization

In this demo, we show trajectories on each of the five ivy gym environments during the course of trajectory optimization.
The optimization iteration is shown in the bottom right, along with the step in the environment.

```bash
cd ~/ivy_gym/demos/optimization
python3 optimize_trajectory.py
```

Example output is given below:

<p align="center">
    <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_gym/demo_a.gif?raw=true'>
</p>

### Policy Optimization

In this demo, we show trajectories on each of the five ivy gym environments during the course of policy optimization.
The optimization iteration is shown in the bottom right, along with the step in the environment.

```bash
cd ~/ivy_gym/demos/optimization
python3 optimize_policy.py
```
Example output is given below:

<p align="center">
    <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_gym/demo_b.gif?raw=true'>
</p>

## Get Involved

If you have any issues running any of the demos, would like to request further demos, or would like to implement your own, then get it touch.
Feature requests, pull requests, and [tweets](https://twitter.com/ivythread) all welcome!