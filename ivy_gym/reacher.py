"""Reacher task."""

# global
import gym
import numpy as np
from ivy.framework_handler import get_framework as _get_framework


# noinspection PyAttributeOutsideInit
class Reacher(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, num_joints=2, f=None):  # noqa
        """
        Initialize Reacher environment

        :param num_joints: Number of joints in reacher.
        :type num_joints: int, optional
        :param f: Machine learning framework.
        :type f: ml_framework, optional
        """
        self._f = _get_framework(f=f)
        self.num_joints = num_joints
        self.torque_scale = 1.
        self.dt = 0.05
        self.action_space = gym.spaces.Box(
            low=-1., high=1., shape=[num_joints], dtype=np.float32)
        high = np.array([np.inf] * (num_joints * 3 + 2), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-high, high=high, dtype=np.float32)
        self.viewer = None
        self._logged_headless_message = False

    def get_observation(self):
        """
        Get observation from environment.

        :return: observation array
        """
        ob = (self._f.reshape(self._f.cos(self.angles), (1, 2)), self._f.reshape(self._f.sin(self.angles), (1, 2)),
              self._f.reshape(self.angle_vels, (1, 2)), self._f.reshape(self.goal_xy, (1, 2)))
        ob = self._f.concatenate(ob, axis=0)
        return self._f.reshape(ob, (-1,))

    def get_reward(self):
        """
        Get reward based on current state

        :return: Reward array
        """
        # Goal proximity.
        x = self._f.reduce_sum(self._f.cos(self.angles), -1)
        y = self._f.reduce_sum(self._f.sin(self.angles), -1)
        xy = self._f.concatenate([self._f.expand_dims(x, 0), self._f.expand_dims(y, 0)], axis=0)
        rew = self._f.reshape(self._f.exp(-1 * self._f.reduce_sum((xy - self.goal_xy) ** 2, -1)), (-1,))
        return self._f.reduce_mean(rew, axis=0, keepdims=True)

    def get_state(self):
        """
        Get current state in environment.

        :return: angles, angular velocities, and goal xy arrays
        """
        return self.angles, self.angle_vels, self.goal_xy

    def set_state(self, state):
        """
        Set current state in environment.

        :param state: tuple of angles, angular_velocities, and goal xy arrays
        :type state: tuple of arrays
        :return: observation array
        """
        self.angles, self.angle_vels, self.goal_xy = state
        return self.get_observation()

    def reset(self):
        self.angles = self._f.random_uniform(-np.pi, np.pi, [self.num_joints])
        self.angle_vels = self._f.random_uniform(
            -1, 1, [self.num_joints])
        self.goal_xy = self._f.random_uniform(
            -self.num_joints, self.num_joints, [2])
        return self.get_observation()

    def step(self, action):
        angle_accs = self.torque_scale * action
        self.angle_vels = self.angle_vels + self.dt * angle_accs
        self.angles = self.angles + self.dt * self.angle_vels
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
            bound = self.num_joints + 0.2
            self.viewer.set_bounds(-bound, bound, -bound, bound)

            # Goal.
            goal_geom = rendering.make_circle(0.2)
            goal_geom.set_color(0.4, 0.6, 1.)
            self.goal_tr = rendering.Transform()
            goal_geom.add_attr(self.goal_tr)
            self.viewer.add_geom(goal_geom)

            # Arm segments and joints.
            l, r, t, b = 0, 1., 0.1, -0.1
            self.segment_trs = []
            for _ in range(self.num_joints):
                # Segment.
                segment_geom = rendering.FilledPolygon(
                    [(l, b), (l, t), (r, t), (r, b)])
                segment_tr = rendering.Transform()
                self.segment_trs.append(segment_tr)
                segment_geom.add_attr(segment_tr)
                segment_geom.set_color(0., 0., 0.)
                self.viewer.add_geom(segment_geom)

                # Joint.
                joint_geom = rendering.make_circle(0.1)
                joint_geom.set_color(0.5, 0.5, 0.5)
                joint_geom.add_attr(segment_tr)
                self.viewer.add_geom(joint_geom)

            # End effector.
            self.end_geom = rendering.make_circle(0.1)
            self.end_tr = rendering.Transform()
            self.end_geom.add_attr(self.end_tr)
            self.viewer.add_geom(self.end_geom)

        self.goal_tr.set_translation(*self._f.to_numpy(self.goal_xy).tolist())

        x, y = 0., 0.
        for segment_tr, angle in zip(self.segment_trs, self._f.reshape(self.angles, (-1, 1))):
            segment_tr.set_rotation(self._f.to_numpy(angle)[0])
            segment_tr.set_translation(x, y)
            x = self._f.to_numpy(x + self._f.cos(self._f.expand_dims(angle, 0))[0])[0]
            y = self._f.to_numpy(y + self._f.sin(self._f.expand_dims(angle, 0))[0])[0]
        self.end_tr.set_translation(x, y)
        rew = self._f.to_numpy(self.get_reward())[0]
        self.end_geom.set_color(1 - rew, rew, 0.)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        """
        Close environment.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
