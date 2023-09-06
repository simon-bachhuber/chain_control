from collections import OrderedDict

from dm_control import mujoco
from dm_control.rl import control
from dm_env import specs as dm_env_specs
import numpy as np

from ...utils.sample_from_spec import _spec_from_observation
from .common import ASSETS
from .common import read_model


class MusclePhysics(mujoco.Physics):
    def get_endeffector_angle(self):
        return self.named.data.qpos["shoulder"].copy()


def cocontraction(x, center, delta):
    assert (center - delta) >= 0
    assert (center + delta) <= 1
    u1 = np.tanh(x) * delta + center
    u2 = np.tanh(-x) * delta + center
    return np.concatenate((u1, u2), axis=0)


def load_physics_cocontraction(**physics_kwargs):
    center = physics_kwargs.pop("center", 0.1)
    delta = physics_kwargs.pop("delta", 0.1)

    class Physics(MusclePhysics):
        def set_tendon_forces(self, u):
            self.set_control(cocontraction(u, center, delta))

    return Physics.from_xml_string(
        read_model("muscle_siso_cocontraction.xml"), assets=ASSETS
    )


def load_physics_asymmetric(**physics_kwargs):
    corner = direct = None
    if "corner" not in physics_kwargs:
        if "direct" not in physics_kwargs:
            direct = np.array([0.03])
        else:
            direct = np.atleast_1d(physics_kwargs["direct"])
    else:
        if "direct" not in physics_kwargs:
            corner = np.atleast_1d(physics_kwargs["corner"])
        else:
            raise Exception("Both `corner` and `direct` can not be provided.")

    # 1st is SF, then SE
    # SF is direct, SE is corner
    class Physics(MusclePhysics):
        def set_tendon_forces(self, u):
            if corner is not None:
                control = (np.tanh(u), corner)
            else:
                control = (direct, np.tanh(-u))

            self.set_control(np.concatenate(control))

    return Physics.from_xml_string(
        read_model("muscle_siso_asymmetric.xml"), assets=ASSETS
    )


class Task(control.Task):
    def __init__(self, random: int = 1):
        # seed is unused
        del random
        super().__init__()

    def initialize_episode(self, physics):
        pass

    def before_step(self, action, physics):
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
            name="muscle_stimuli",
        )

    def get_observation(self, physics) -> OrderedDict:
        obs = OrderedDict()
        obs["endeffector_phi_rad"] = np.atleast_1d(physics.get_endeffector_angle())
        return obs

    def get_reward(self, physics):
        return np.array(0.0)

    def observation_spec(self, physics):
        return _spec_from_observation(self.get_observation(physics))
