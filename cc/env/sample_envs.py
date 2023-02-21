from typing import Final

from .envs import muscle, rover
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

_id_accessible_envs: Final = {
    "two_segments_v1": TWO_SEGMENT_V1,
    "two_segments_v2": TWO_SEGMENT_V2,
    "two_segments_v3": TWO_SEGMENT_V3,
    "muscle_asymmetric": EnvConfig(muscle.load_physics_asymmetric, muscle.Task),
    "muscle_cocontraction": EnvConfig(muscle.load_physics_cocontraction, muscle.Task),
    "rover": EnvConfig(rover.load_physics, rover.Task_Steering),
}
