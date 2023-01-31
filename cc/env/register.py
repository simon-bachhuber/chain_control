from dataclasses import dataclass
from typing import Callable, List

from dm_control import mujoco
from dm_control.rl import control

from .envs import two_segments
from .envs.two_segments import CartParams, JointParams, SegmentTask


def _load_physics_wrapper(cart_params: List[CartParams]) -> Callable[[], mujoco.Physics]:
    def load_physics():
        return two_segments.load_physics(cart_params)

    return load_physics


@dataclass
class EnvRegisterValue:
    load_physics: Callable[[], mujoco.Physics]
    Task: control.Task
    time_limit: float
    control_timestep: float = 0.01


_register = {
    "two_segments_v1": EnvRegisterValue(
        _load_physics_wrapper(
            [
                CartParams(
                    name="cart",
                    slider_joint_params=JointParams(damping=1e-3),
                    hinge_joint_params=JointParams(
                        damping=1e-1, springref=0, stiffness=10
                    ),
                )
            ]
        ),
        SegmentTask,
        10.0,
    ),
    "two_segments_v2": EnvRegisterValue(
        _load_physics_wrapper(
            [
                CartParams(
                    name="cart",
                    slider_joint_params=JointParams(damping=1e-3),
                    hinge_joint_params=JointParams(
                        damping=3e-2, springref=0, stiffness=3
                    ),
                )
            ]
        ),
        SegmentTask,
        10.0,
    ),
    "two_segments_v3": EnvRegisterValue(
        _load_physics_wrapper(
            [
                CartParams(
                    name="cart",
                    slider_joint_params=JointParams(damping=1, stiffness=1),
                    hinge_joint_params=JointParams(
                        damping=3e-2, springref=0, stiffness=2
                    ),
                )
            ]
        ),
        SegmentTask,
        10.0,
    ),
}


def register_new_env(id: str, value: EnvRegisterValue):
    _register[id] = value


def register_new_two_segment_env(
    id: str, cart_params: CartParams, time_limit: float = 10.0
):
    _register[id] = EnvRegisterValue(
        id, _load_physics_wrapper(cart_params), SegmentTask, time_limit
    )
