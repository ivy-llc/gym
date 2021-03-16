"""Mountain-car task adapted from:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/
mountain_car.py
"""

# global
import ivy
import gym
import numpy as np


# noinspection PyAttributeOutsideInit
class MountainCar(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):  # noqa
        """
        Initialize MountainCar environment
        """
        self.torque_scale = 3.
        self.g = 9.8
        self.dt = 0.02
        self.goal_x = ivy.array([0.45])
        self.action_space = gym.spaces.Box(-1., 1., [1], np.float32)
        high = np.array([np.inf, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.viewer = None
        self._logged_headless_message = False

    def get_observation(self):
        """
        Get observation from environment.

        :return: observation array
        """
        return ivy.concatenate([self.x, self.x_vel], axis=-1)

    def get_reward(self):
        """
        Get reward based on current state

        :return: Reward array
        """
        # Goal proximity.
        return ivy.reshape(ivy.exp(-5 * ((self.x - self.goal_x) ** 2)), (1,))

    def get_state(self):
        """
        Get current state in environment.

        :return: x and x velocity arrays
        """
        return self.x, self.x_vel

    def set_state(self, state):
        """
        Set current state in environment.

        :param state: tuple of x and x_velocity
        :type state: tuple of arrays
        :return: observation array
        """
        self.x, self.x_vel = state
        return self.get_observation()

    def reset(self):
        self.x = ivy.random_uniform(-0.9, -0.2, [1])
        self.x_vel = ivy.zeros([1])
        return self.get_observation()

    def step(self, action):
        x_acc = action * self.torque_scale - self.g * ivy.cos(3 * self.x)
        self.x_vel = self.x_vel + self.dt * x_acc
        self.x = self.x + self.dt * self.x_vel
        return self.get_observation(), self.get_reward(), False, {}

    @staticmethod
    def _height(xs):
        return ivy.sin(3 * xs) * 0.45 + 0.55

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
        x_min = -1.2
        x_max = 0.6
        world_width = x_max - x_min
        scale = screen_width / world_width
        car_width = 40
        car_height = 20

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
            xs = ivy.linspace(x_min, x_max, 100)
            ys = self._height(xs)
            xys = list((ivy.to_numpy(xt).item(), ivy.to_numpy(yt).item())
                       for xt, yt in zip((xs - x_min) * scale, ys * scale))
            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(2)
            self.viewer.add_geom(self.track)

            # Car.
            clearance = 10
            l, r, t, b = -car_width / 2, car_width / 2, car_height, 0
            self.car_geom = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])
            self.car_geom.add_attr(
                rendering.Transform(translation=(0, clearance)))
            self.car_tr = rendering.Transform()
            self.car_geom.add_attr(self.car_tr)
            self.viewer.add_geom(self.car_geom)

            # Wheels.
            front_wheel = rendering.make_circle(car_height / 2.5)
            front_wheel.set_color(0.5, 0.5, 0.5)
            front_wheel.add_attr(
                rendering.Transform(translation=(car_width / 4, clearance)))
            front_wheel.add_attr(self.car_tr)
            self.viewer.add_geom(front_wheel)
            back_wheel = rendering.make_circle(car_height / 2.5)
            back_wheel.add_attr(
                rendering.Transform(translation=(-car_width / 4, clearance)))
            back_wheel.add_attr(self.car_tr)
            back_wheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(back_wheel)

            # Flag.
            flag_x = (ivy.to_numpy(self.goal_x)[0] - x_min) * scale
            flagy_y1 = ivy.to_numpy(self._height(self.goal_x))[0] * scale
            flagy_y2 = flagy_y1 + 50
            flagpole = rendering.Line((flag_x, flagy_y1), (flag_x, flagy_y2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flag_x, flagy_y2), (flag_x, flagy_y2 - 10),
                 (flag_x + 25, flagy_y2 - 5)])
            flag.set_color(0.4, 0.6, 1.)
            self.viewer.add_geom(flag)

        self.car_tr.set_translation(
            (ivy.to_numpy(self.x)[0] - x_min) * scale, ivy.to_numpy(self._height(self.x))[0] * scale)
        self.car_tr.set_rotation(ivy.to_numpy(ivy.cos(3 * self.x))[0])
        rew = ivy.to_numpy(self.get_reward()).item()
        self.car_geom.set_color(1 - rew, rew, 0.)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        """
        Close environment.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
