from typing import Final

from cc.env.make_env import EnvConfig

from .envs import two_segments
from .envs.two_segments import CartParams, JointParams, SegmentTask

TWO_SEGMENT_V1: Final = EnvConfig(
    load_physics=two_segments.load_physics(
        CartParams(
            name="cart",
            slider_joint_params=JointParams(damping=1e-3),
            hinge_joint_params=JointParams(
                damping=1e-1, springref=0, stiffness=10
            ),
        )
    ),
    Task=SegmentTask,
)

TWO_SEGMENT_V2: Final = EnvConfig(
    load_physics=two_segments.load_physics(
        CartParams(
            name="cart",
            slider_joint_params=JointParams(damping=1e-3),
            hinge_joint_params=JointParams(
                damping=3e-2, springref=0, stiffness=3
            ),
        )
    ),
    Task=SegmentTask,
)

TWO_SEGMENT_V3: Final = EnvConfig(
    load_physics=two_segments.load_physics(
        CartParams(
            name="cart",
            slider_joint_params=JointParams(damping=1, stiffness=1),
            hinge_joint_params=JointParams(
                damping=3e-2, springref=0, stiffness=2
            ),
        )
    ),
    Task=SegmentTask,
)
