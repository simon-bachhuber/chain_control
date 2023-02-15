# TODO

# test that time_limit_control_timestep and track_time
# wrapping does not alter the actual environment


from cc.env.sample_envs import TWO_SEGMENT_V1

from ..make_env import make_env, make_unwrapped_env
from .test_auto_reset import tree_equal, unroll_env1


def test_tracking_time_does_not_hurt():
    env1 = make_unwrapped_env(
        TWO_SEGMENT_V1, time_limit=10, control_timestep=0.01
    )
    env2 = make_env("two_segments_v1", random=1, single_precision=False)
    traj1, traj2 = unroll_env1(env1), unroll_env1(env2)
    assert tree_equal(traj1, traj2)
