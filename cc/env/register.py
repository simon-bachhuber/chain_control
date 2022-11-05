from dataclasses import dataclass
from typing import Callable

from dm_control import mujoco
from dm_control.rl import control

from .envs import two_segments_v1, two_segments_v2, two_segments_v3


@dataclass
class EnvRegisterValue:
    load_physics: Callable[[], mujoco.Physics]
    Task: control.Task
    time_limit: float
    control_timestep: float = 0.01


_register = {
    "two_segments_v1": EnvRegisterValue(
        two_segments_v1.load_physics, two_segments_v1.SegmentTask, 10.0
    ),
    "two_segments_v2": EnvRegisterValue(
        two_segments_v2.load_physics, two_segments_v2.SegmentTask, 10.0
    ),
    "two_segments_v3": EnvRegisterValue(
        two_segments_v3.load_physics, two_segments_v3.SegmentTask, 10.0
    ),
}
