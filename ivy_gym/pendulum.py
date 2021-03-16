"""Pendulum task adapted from:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
"""

# global
import ivy
import gym
import numpy as np


# noinspection PyAttributeOutsideInit
class Pendulum(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):  # noqa
        """
        Initialize Pendulum environment
        """
        self.torque_scale = 1.
        self.g = 9.8
        self.dt = 0.05
        self.m = 1.
        self.l = 1.
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=[1], dtype=np.float32)
        high = np.array([1., 1., np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-high, high=high, dtype=np.float32)
        self.viewer = None
        self._logged_headless_message = False

    def get_observation(self):
        """
        Get observation from environment.

        :return: observation array
        """
        return ivy.concatenate(
            (ivy.cos(self.angle), ivy.sin(self.angle),
             self.angle_vel),
            axis=-1)

    def get_reward(self):
        """
        Get reward based on current state

        :return: Reward array
        """
        # Pole verticality.
        rew = (ivy.cos(self.angle) + 1) / 2
        return ivy.reshape(rew, (1,))

    def get_state(self):
        """
        Get current state in environment.

        :return: angle and angular velocity arrays
        """
        return self.angle, self.angle_vel

    def set_state(self, state):
        """
        Set current state in environment.

        :param state: tuple of angle and angular_velocity
        :type state: tuple of arrays
        :return: observation array
        """
        self.angle, self.angle_vel = state
        return self.get_observation()

    def reset(self):
        self.angle = ivy.random_uniform(-np.pi, np.pi, [1])
        self.angle_vel = ivy.random_uniform(-1., 1., [1])
        return self.get_observation()

    def step(self, action):
        action = action * self.torque_scale

        angle_acc = (
            -3 * self.g / (2 * self.l) * ivy.sin(self.angle + np.pi) +
            3. / (self.m * self.l ** 2) * action)

        self.angle_vel = self.angle_vel + self.dt * angle_acc
        self.angle = self.angle + self.dt * self.angle_vel

        return self.get_observation(), self.get_reward(), False, {}

    def render(self, mode='human'):
        """
        Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        :param mode: Render mode, one of [human|rgb_array], default human
        :type mode: str, optional
        :return: Rendered image.
        """
        if self.viewer is None:
            # noinspection PyBroadException
            try:
                from gym.envs.classic_control import rendering
            except:
                if not self._logged_headless_message:
                    print('Unable to connect to display. Running the Ivy environment in headless mode...')
                    self._logged_headless_message = True
                return

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            # Pole.
            self.pole_geom = rendering.make_capsule(1, .2)
            self.pole_geom.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            self.pole_geom.add_attr(self.pole_transform)
            self.viewer.add_geom(self.pole_geom)

            # Axle.
            axle = rendering.make_circle(0.05)
            axle.set_color(0., 0., 0.)
            self.viewer.add_geom(axle)

        self.pole_transform.set_rotation(ivy.to_numpy(self.angle)[0] + np.pi / 2)
        rew = ivy.to_numpy(self.get_reward())[0]
        self.pole_geom.set_color(1 - rew, rew, 0.)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        """
        Close environment.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
