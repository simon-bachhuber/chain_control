from collections import OrderedDict

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_env import specs as dm_env_specs

from ...utils.sample_from_spec import _spec_from_observation
from .common import ASSETS, read_model


def cocontraction(x, center=0.2, delta=0.2):
    assert (center - delta) >= 0
    assert (center + delta) <= 1
    u1 = np.tanh(x) * delta + center
    u2 = np.tanh(-x) * delta + center
    return np.concatenate((u1, u2), axis=0)


class Physics(mujoco.Physics):
    @staticmethod
    def cocontraction(x):
        return cocontraction(x)

    def set_tendon_forces(self, u):
        self.set_control(self.cocontraction(u))

    def get_endeffector_angle(self):
        return np.rad2deg(self.named.data.qpos["shoulder"].copy())


def load_physics(**physics_kwargs):
    return Physics.from_xml_string(read_model("muscle_siso.xml"), assets=ASSETS)


class Task(control.Task):
    def __init__(self, random: int = 1):
        # seed is unused
        del random
        super().__init__()

    def initialize_episode(self, physics):
        pass

    def before_step(self, action, physics: Physics):
        action = np.atleast_1d(action)
        physics.set_tendon_forces(action)

    def after_step(self, physics):
        pass

    def action_spec(self, physics):
        return dm_env_specs.BoundedArray(
            (1,),
            dtype=float,
            minimum=-mujoco.mjMAXVAL,
            maximum=mujoco.mjMAXVAL,
            name="cocontraction_input",
        )

    def get_observation(self, physics: Physics) -> OrderedDict:
        obs = OrderedDict()
        obs["endeffector_phi_deg"] = np.atleast_1d(physics.get_endeffector_angle())
        return obs

    def get_reward(self, physics):
        return np.array(0.0)

    def observation_spec(self, physics):
        return _spec_from_observation(self.get_observation(physics))
