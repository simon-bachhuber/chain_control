# this file is supposed to test the auto reset mechanism
# that can be used

import numpy as np
import pytest
from equinox import tree_equal

from cc.env.sample_envs import TWO_SEGMENT_V1

from ..make_env import make_env, make_unwrapped_env
from .__init__ import AddRefSignalRewardFnWrapper, ReplacePhysicsByModelWrapper
from .test_add_reference_and_reward import dummy_source
from .test_replace_physics_by_model import dummy_model

N_STEPS_FOR_EPISODE = 1001
action = np.array([1.0])
env_unwrapped = make_unwrapped_env(
    TWO_SEGMENT_V1, time_limit=10, control_timestep=0.01
)
env = make_env("two_segments_v1", random=1)


def unroll_env1(env):
    trajectory = []
    ts = env.reset()
    trajectory.append(ts)
    while not ts.last():
        ts = env.step(action)
        trajectory.append(ts)
    return trajectory


def unroll_env2(env):
    trajetory = []
    for _ in range(2 * N_STEPS_FOR_EPISODE):
        ts = env.step(action)
        trajetory.append(ts)
    return trajetory[:N_STEPS_FOR_EPISODE], trajetory[N_STEPS_FOR_EPISODE:]


@pytest.mark.parametrize(
    "env",
    [
        env,
        env_unwrapped,
        ReplacePhysicsByModelWrapper(env, dummy_model()),
        AddRefSignalRewardFnWrapper(env, dummy_source(env)),
    ],
)
def test_auto_reset(env):
    traj1 = unroll_env1(env)
    traj2, traj3 = unroll_env2(env)
    assert tree_equal(traj1, traj2, traj3)
    assert tree_equal(
        traj1[333].observation, traj2[333].observation, traj3[333].observation
    )
