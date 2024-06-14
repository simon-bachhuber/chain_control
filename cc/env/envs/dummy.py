from collections import OrderedDict

from dm_control.rl import control
from dm_env import specs
import mujoco
import numpy as np

from ...utils.sample_from_spec import _spec_from_observation


class Physics(control.Physics):
    def __init__(self):
        self.reset()

    def step(self, n_sub_steps=1):
        for _ in range(n_sub_steps):
            self._time += self.timestep()

    def reset(self):
        self._time = 0.0

    def time(self):
        return self._time

    def timestep(self):
        return 0.01

    def after_reset(self):
        pass


class Task(control.Task):
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def initialize_episode(self, physics):
        pass

    def before_step(self, action, physics: Physics):
        pass

    def after_step(self, physics):
        pass

    def action_spec(self, physics):
        return specs.BoundedArray(
            (self.input_dim,),
            dtype=float,
            minimum=-mujoco.mjMAXVAL,
            maximum=mujoco.mjMAXVAL,
            name="input",
        )

    def get_observation(self, physics: Physics) -> OrderedDict:
        obs = OrderedDict()
        obs["output"] = np.zeros((self.output_dim,))
        return obs

    def get_reward(self, physics):
        return np.array(0.0)

    def observation_spec(self, physics):
        return _spec_from_observation(self.get_observation(physics))
