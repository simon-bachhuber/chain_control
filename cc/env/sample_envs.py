from typing import Final

from .envs import ackermann, muscle, rover, two_segments
from .envs.two_segments import CartParams, JointParams, generate_env_config
from .make_env import EnvConfig

TWO_SEGMENT_V1: Final = generate_env_config(
    CartParams(
        name="cart",
        slider_joint_params=JointParams(damping=1e-3),
        hinge_joint_params=JointParams(damping=1e-1, springref=0, stiffness=10),
    )
)


TWO_SEGMENT_V2: Final = generate_env_config(
    CartParams(
        name="cart",
        slider_joint_params=JointParams(damping=1e-3),
        hinge_joint_params=JointParams(damping=3e-2, springref=0, stiffness=3),
    )
)

TWO_SEGMENT_V3: Final = generate_env_config(
    CartParams(
        name="cart",
        slider_joint_params=JointParams(damping=1, stiffness=1),
        hinge_joint_params=JointParams(damping=3e-2, springref=0, stiffness=2),
    )
)


# TODO
# clean it up at some point
def two_segments_load_physics(damping=3e-3, stiffness=3):
    env_config = generate_env_config(
        CartParams(
            "cart",
            slider_joint_params=JointParams(damping=1e-3),
            hinge_joint_params=JointParams(damping=damping, stiffness=stiffness),
        )
    )
    return env_config.load_physics()


# TODO
# delete `v1`, `v2`, `v3`
_id_accessible_envs: Final = {
    "two_segments_v1": TWO_SEGMENT_V1,
    "two_segments_v2": TWO_SEGMENT_V2,
    "two_segments_v3": TWO_SEGMENT_V3,
    "two_segments": EnvConfig(two_segments_load_physics, two_segments.SegmentTask),
    "muscle_asymmetric": EnvConfig(muscle.load_physics_asymmetric, muscle.Task),
    "muscle_cocontraction": EnvConfig(muscle.load_physics_cocontraction, muscle.Task),
    "rover": EnvConfig(rover.load_physics, rover.Task_Steering),
    "ackermann": EnvConfig(ackermann.Physics, ackermann.Task),
}
