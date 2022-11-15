from collections import OrderedDict

import numpy as np
from dm_control import mujoco
from dm_control.rl import control

from ...utils.sample_from_spec import _spec_from_observation
from .common import ASSETS, read_model


class SegmentPhysics(mujoco.Physics):
    def xpos_of_segment_end(self):
        return self.named.data.xpos["segment_end", "x"]

    def set_torque_of_cart(self, u):
        u = np.arctan(u)
        self.set_control(u)


def load_physics():
    xml_path = "two_segments_v2.xml"
    return SegmentPhysics.from_xml_string(read_model(xml_path), assets=ASSETS)


class SegmentTask(control.Task):
    def __init__(self, random: int = 1):
        # seed is unused
        del random
        super().__init__()

    def initialize_episode(self, physics):
        pass

    def before_step(self, action, physics: SegmentPhysics):
        physics.set_torque_of_cart(action)

    def after_step(self, physics):
        pass

    def action_spec(self, physics):
        return mujoco.action_spec(physics)

    def get_observation(self, physics) -> OrderedDict:
        obs = OrderedDict()
        obs["xpos_of_segment_end"] = np.atleast_1d(physics.xpos_of_segment_end())
        return obs

    def get_reward(self, physics):
        return np.array(0.0)

    def observation_spec(self, physics):
        return _spec_from_observation(self.get_observation(physics))
