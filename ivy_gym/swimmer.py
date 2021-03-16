"""Path finding task.
A fish needs to reach a goal location while avoiding urchins.
"""

import ivy
import gym
import numpy as np


# Environment Class #
# ------------------#

# noinspection PyAttributeOutsideInit
class Swimmer(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, num_urchins=5):  # noqa
        """
        Initialize Swimmer environment.

        :param num_urchins: Number of urchins.
        :type num_urchins: int, optional
        """
        self.num_urchins = num_urchins
        self.dt = 0.05
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=[2], dtype=np.float32)
        high = np.array([np.inf] * (num_urchins * 2 + 6), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-high, high=high, dtype=np.float32)
        self.viewer = None
        self._logged_headless_message = False

    def get_observation(self):
        """
        Get observation from environment.

        :return: observation array
        """
        ob = (ivy.reshape(self.urchin_xys, (-1, 2)), ivy.reshape(self.xy, (-1, 2)),
              ivy.reshape(self.xy_vel, (-1, 2)), ivy.reshape(self.goal_xy, (-1, 2)))
        ob = ivy.concatenate(ob, axis=0)
        return ivy.reshape(ob, (-1,))

    def get_reward(self):
        """
        Get reward based on current state

        :return: Reward array
        """
        # Goal proximity.
        rew = ivy.exp(
            -0.5 * ivy.reduce_sum((self.xy - self.goal_xy) ** 2, -1))
        # Urchins proximity.
        rew = rew * ivy.reduce_prod(
            1 - ivy.exp(-30 * ivy.reduce_sum(
                (self.xy - self.urchin_xys) ** 2, -1)), -1)
        return ivy.reshape(rew, (1,))

    def get_state(self):
        """
        Get current state in environment.

        :return: Urchin xys, xy, xy velocity, and goal xy arrays
        """
        return self.urchin_xys, self.xy, self.xy_vel, self.goal_xy

    def set_state(self, state):
        """
        Set current state in environment.

        :param state: tuple of urchin xys, xy, xy velocity, and goal xy
        :type state: tuple of arrays
        :return: observation array
        """
        self.urchin_xys, self.xy, self.xy_vel, self.goal_xy = state
        return self.get_observation()

    def reset(self):
        self.urchin_xys = ivy.random_uniform(
            -1, 1, (self.num_urchins, 2))
        self.xy = ivy.random_uniform(-1, 1, (2,))
        self.xy_vel = ivy.zeros((2,))
        self.goal_xy = ivy.random_uniform(-1, 1, (2,))
        return self.get_observation()

    def step(self, action):
        self.xy_vel = self.xy_vel + self.dt * action
        self.xy = self.xy + self.dt * self.xy_vel
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
            from pyglet import gl

            class _StarGeom(rendering.Geom):
                def __init__(self, r1, r2, n):
                    super().__init__()
                    self.r1 = r1
                    self.r2 = r2
                    self.n = n

                def render1(self):
                    n = self.n * 2
                    for i in range(0, n, 2):
                        gl.glBegin(gl.GL_TRIANGLES)
                        a0 = 2 * np.pi * i / n
                        a1 = 2 * np.pi * (i + 1) / n
                        a2 = 2 * np.pi * (i - 1) / n
                        gl.glVertex3f(np.cos(a0) * self.r1, np.sin(a0) * self.r1, 0)
                        gl.glVertex3f(np.cos(a1) * self.r2, np.sin(a1) * self.r2, 0)
                        gl.glVertex3f(np.cos(a2) * self.r2, np.sin(a2) * self.r2, 0)
                        gl.glEnd()
                    gl.glBegin(gl.GL_POLYGON)
                    for i in range(0, n, 2):
                        a = 2 * np.pi * (i + 1) / n
                        gl.glVertex3f(np.cos(a) * self.r2, np.sin(a) * self.r2, 0)
                    gl.glEnd()

            class _FishGeom(rendering.Geom):
                def __init__(self):
                    super().__init__()
                    self.color = 0., 0., 0.

                def render1(self):
                    points = [
                        [0.08910714285714288, -0.009017857142857133],
                        [0.13910714285714287, -0.04026785714285712],
                        [0.12285714285714289, 0.07098214285714288],
                        [0.08535714285714285, 0.03348214285714288],
                        [0.10535714285714287, 0.07848214285714286],
                        [0.04910714285714285, 0.13348214285714285],
                        [-0.03589285714285714, 0.11723214285714287],
                        [-0.14964285714285713, 0.08598214285714287],
                        [-0.21714285714285714, 0.023482142857142868],
                        [-0.18589285714285714, -0.004017857142857129],
                        [-0.12714285714285714, -0.11151785714285713],
                        [-0.039642857142857146, -0.15651785714285713],
                        [0.044107142857142845, -0.15651785714285713],
                        [0.12035714285714288, -0.06526785714285713]]
                    gl.glColor3f(*self.color)
                    gl.glBegin(gl.GL_POLYGON)
                    for p0, p1 in points:
                        gl.glVertex3f(p0, -p1, 0)
                    gl.glEnd()
                    points = [
                        [-0.14964285714285713, -0.016517857142857112],
                        [-0.11214285714285714, 0.020982142857142866],
                        [-0.15839285714285714, 0.06973214285714288],
                        [-0.17089285714285712, 0.013482142857142887]]
                    gl.glColor3f(0.5, 0.4, 0.3)
                    gl.glBegin(gl.GL_POLYGON)
                    for p0, p1 in points:
                        gl.glVertex3f(p0, -p1, 0)
                    gl.glEnd()
                    points = []
                    for i in range(20):
                        ang = 2 * np.pi * i / 20
                        points.append(
                            (np.cos(ang) * 0.018 - 0.16, np.sin(ang) * 0.018 - 0.01))
                    gl.glColor3f(0, 0, 0)
                    gl.glBegin(gl.GL_POLYGON)
                    for p0, p1 in points:
                        gl.glVertex3f(p0, p1, 0)
                    gl.glEnd()

                def set_color(self, r, g, b):
                    self.color = r, g, b

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-1.5, 1.5, -1.5, 1.5)

            # Goal.
            goal_geom = rendering.make_circle(.2)
            self.goal_tr = rendering.Transform()
            goal_geom.add_attr(self.goal_tr)
            goal_geom.set_color(0.4, 0.6, 1.)
            self.viewer.add_geom(goal_geom)

            # Urchins.
            self.urchin_trs = []
            for _ in range(self.num_urchins):
                urchin_geom = _StarGeom(0.2, 0.05, 15)
                urchin_tr = rendering.Transform()
                self.urchin_trs.append(urchin_tr)
                urchin_geom.add_attr(urchin_tr)
                urchin_geom.set_color(0., 0., 0.)
                self.viewer.add_geom(urchin_geom)

            # Fish.
            self.fish_geom = _FishGeom()
            self.fish_tr = rendering.Transform()
            self.fish_geom.add_attr(self.fish_tr)
            self.viewer.add_geom(self.fish_geom)

        self.goal_tr.set_translation(*ivy.to_numpy(self.goal_xy).tolist())
        for urchin_tr, (x, y) in zip(self.urchin_trs, ivy.reshape(self.urchin_xys, (5, 2, 1))):
            urchin_tr.set_translation(ivy.to_numpy(x)[0], ivy.to_numpy(y)[0])
        self.fish_tr.set_translation(*ivy.to_numpy(ivy.reshape(self.xy, (2,))).tolist())
        rew = ivy.to_numpy(self.get_reward())[0]
        self.fish_geom.set_color(1 - rew, rew, 0.)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        """
        Close environment.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
