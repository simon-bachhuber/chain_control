# TODO

# test that time_limit_control_timestep and track_time
# wrapping does not alter the actual environment


from ..make_env import make_env, make_unwrapped_env
from .test_auto_reset import tree_equal, unroll_env1


def test_tracking_time_does_not_hurt():
    env1 = make_unwrapped_env("two_segments_v1", random=1)
    env2 = make_env("two_segments_v1", random=1, single_precision=False)
    traj1, traj2 = unroll_env1(env1), unroll_env1(env2)
    assert tree_equal(traj1, traj2)
