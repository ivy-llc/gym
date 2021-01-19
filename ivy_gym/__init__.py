from .cartpole import CartPole
from .mountain_car import MountainCar
from .pendulum import Pendulum
from .reacher import Reacher
from .swimmer import Swimmer


envs = [CartPole, MountainCar, Pendulum, Reacher, Swimmer]
__all__ = envs
