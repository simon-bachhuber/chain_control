from collections import OrderedDict

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_env import specs as dm_env_specs

from ...utils.sample_from_spec import _spec_from_observation
from .common import ASSETS, read_model


class Physics(mujoco.Physics):
    def set_steering_angle(self, u):
        """In radians."""
        drive = self.control()[1:]

        current_rover_angle = self.get_rover_angle_from_straight_dir_deg()
        if drive >= 0.0:  # Front Wheel Steering Rover
            if np.abs(current_rover_angle) > 60.0:
                u = np.array([0.0])
        else:  # Rear Wheel Steering Rover
            if current_rover_angle >= 30.0:
                u = np.array([0.5])
            elif current_rover_angle <= -30.0:
                u = np.array([-0.5])

        u = np.tanh(u) * 0.29

        self.set_control(np.concatenate((u, drive)))

    def set_drive(self, u):
        steering = self.control()[:1]
        self.set_control(np.concatenate((steering, np.atleast_1d(u))))

    def get_xy_position(self, at):
        return self.named.data.xpos[at].copy()[:2]

    def get_xy_position_at_center(self):
        return self.get_xy_position("ghost-steer-wheel")

    def get_xy_position_at_front(self):
        return self.get_xy_position("ghost-steer-wheel-at-front")

    def get_rover_angle_from_straight_dir_deg(self):
        forward_dir = self.get_xy_position_at_front() - self.get_xy_position_at_center()
        return np.rad2deg(np.arctan2(forward_dir[1], forward_dir[0]))


def load_physics(**physics_kwargs):
    return Physics.from_xml_string(read_model("rover.xml"), assets=ASSETS)


class Task_Steering(control.Task):
    def __init__(self, drive: float = 0.66, random: int = 1):
        # seed is unused
        del random
        self.drive = drive
        super().__init__()

    def initialize_episode(self, physics):
        pass

    def before_step(self, action, physics: Physics):
        action = np.atleast_1d(action)
        physics.set_drive(self.drive)
        physics.set_steering_angle(action)

    def after_step(self, physics):
        pass

    def action_spec(self, physics):
        return dm_env_specs.BoundedArray(
            (1,),
            dtype=float,
            minimum=-mujoco.mjMAXVAL,
            maximum=mujoco.mjMAXVAL,
            name="steering_angle",
        )

    def get_observation(self, physics: Physics) -> OrderedDict:
        obs = OrderedDict()
        xy = physics.get_xy_position_at_center()
        obs["y_position_of_rover"] = np.atleast_1d(xy[1])
        return obs

    def get_reward(self, physics):
        return np.array(0.0)

    def observation_spec(self, physics):
        return _spec_from_observation(self.get_observation(physics))
