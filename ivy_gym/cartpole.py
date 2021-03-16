"""Cart-pole task adapted from:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""

# global
import ivy
import gym
import numpy as np


# noinspection PyAttributeOutsideInit
class CartPole(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):  # noqa
        """
        Initialize CartPole environment
        """
        self.torque_scale = 10.
        self.g = 9.8
        self.dt = 0.02
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = (self.pole_mass + self.cart_mass)
        self.pole_length = 0.5  # actually half the pole's length
        self.pole_mass_length = (self.pole_mass * self.pole_length)
        self.action_space = gym.spaces.Box(-1., 1., [1], np.float32)
        high = np.array([np.inf, np.inf, 1., 1., np.inf])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.viewer = None
        self._logged_headless_message = False

    def get_observation(self):
        """
        Get observation from environment.

        :return: observation array
        """
        return ivy.concatenate(
            [self.x, self.x_vel, ivy.cos(self.angle),
             ivy.sin(self.angle), self.angle_vel], axis=-1)

    def get_reward(self):
        """
        Get reward based on current state

        :return: Reward array
        """
        # Center proximity.
        rew = ivy.exp(-1 * (self.x ** 2))
        # Pole verticality.
        rew = rew * (ivy.cos(self.angle) + 1) / 2
        return ivy.reshape(rew[0], (1,))

    def get_state(self):
        """
        Get current state in environment.

        :return: x, x velocity, angle, and angular velocity arrays
        """
        return self.x, self.x_vel, self.angle, self.angle_vel

    def set_state(self, state):
        """
        Set current state in environment.

        :param state: tuple of x, x_velocity, angle, and angular_velocity
        :type state: tuple of arrays
        :return: observation array
        """
        self.x, self.x_vel, self.angle, self.angle_vel = state
        return self.get_observation()

    def reset(self):
        self.x = ivy.random_uniform(-1., 1., [1])
        self.x_vel = ivy.random_uniform(-0.3, 0.3, [1])
        self.angle = ivy.random_uniform(-np.pi, np.pi, [1])
        self.angle_vel = ivy.random_uniform(-0.3, 0.3, [1])
        return self.get_observation()

    def step(self, action):
        force = self.torque_scale * action
        angle_cos = ivy.cos(self.angle)
        angle_sin = ivy.sin(self.angle)
        temp = (
            (force + self.pole_mass_length * self.angle_vel ** 2 * angle_sin) /
            self.total_mass)
        angle_acc = (
            (self.g * angle_sin - angle_cos * temp) /
            (self.pole_length * (4.0 / 3.0 - self.pole_mass * angle_cos ** 2 /
             self.total_mass)))
        x_acc = (
            temp - self.pole_mass_length * angle_acc * angle_cos /
            self.total_mass)
        self.x_vel = self.x_vel + self.dt * x_acc
        self.x = self.x + self.dt * self.x_vel
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
        screen_width = 500
        screen_height = 500
        world_width = 4
        scale = screen_width / world_width
        pole_width = 10.0
        pole_len = scale * (2 * self.pole_length)
        cart_width = 50.0
        cart_height = 30.0
        cart_y = screen_height / 2

        if self.viewer is None:
            # noinspection PyBroadException
            try:
                from gym.envs.classic_control import rendering
            except:
                if not self._logged_headless_message:
                    print('Unable to connect to display. Running the Ivy environment in headless mode...')
                    self._logged_headless_message = True
                return

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Track.
            track = rendering.Line((0., cart_y), (screen_width, cart_y))
            track.set_color(0., 0., 0.)
            self.viewer.add_geom(track)

            # Cart.
            l = -cart_width / 2
            r = cart_width / 2
            t = cart_height / 2
            b = -cart_height / 2
            cart_geom = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])
            self.cart_tr = rendering.Transform()
            cart_geom.add_attr(self.cart_tr)
            cart_geom.set_color(0., 0., 0.)
            self.viewer.add_geom(cart_geom)

            # Pole.
            l = -pole_width / 2
            r = pole_width / 2
            t = pole_len - pole_width / 2
            b = -pole_width / 2
            self.pole_geom = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])
            self.pole_tr = rendering.Transform(translation=(0, 0))
            self.pole_geom.add_attr(self.pole_tr)
            self.pole_geom.add_attr(self.cart_tr)
            self.viewer.add_geom(self.pole_geom)

            # Axle.
            axle_geom = rendering.make_circle(pole_width / 2)
            axle_geom.add_attr(self.pole_tr)
            axle_geom.add_attr(self.cart_tr)
            axle_geom.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(axle_geom)

        cart_x = ivy.to_numpy(self.x * scale + screen_width / 2.0)[0]
        self.cart_tr.set_translation(cart_x, cart_y)
        self.pole_tr.set_rotation(-ivy.to_numpy(self.angle)[0])
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
