import dm_env
import numpy as np
import pytest
from absl.testing import absltest
from dm_env import test_utils

from .make_env import make_env

LENGTH_ACTION_SEQUENCE = 2001


class TestTwoSegmentsV1(test_utils.EnvironmentTestMixin, absltest.TestCase):
    def make_object_under_test(self):
        return make_env("two_segments_v1", random=1)

    def make_action_sequence(self):
        for _ in range(LENGTH_ACTION_SEQUENCE):
            yield self.make_action()


class TestTwoSegmentsV2(test_utils.EnvironmentTestMixin, absltest.TestCase):
    def make_object_under_test(self):
        return make_env("two_segments_v2", random=1)

    def make_action_sequence(self):
        for _ in range(LENGTH_ACTION_SEQUENCE):
            yield self.make_action()


class TestTwoSegmentsV3(test_utils.EnvironmentTestMixin, absltest.TestCase):
    def make_object_under_test(self):
        return make_env("two_segments_v3", random=1)

    def make_action_sequence(self):
        for _ in range(LENGTH_ACTION_SEQUENCE):
            yield self.make_action()


def unroll_env(env: dm_env.Environment, n_steps: int = None, action=None):
    trajectory = []
    if action is None:
        action = env.action_spec().generate_value()
    ts = env.reset()
    trajectory.append(ts)
    while not ts.last():
        ts = env.step(action)
        trajectory.append(ts)
        if n_steps:
            if len(trajectory) > n_steps:
                break
    return trajectory


# no randomness in environments
@pytest.mark.parametrize(
    "env_id", ["two_segments_v1", "two_segments_v2", "two_segments_v3"]
)
def test_no_randomness(env_id):
    env1, env2 = make_env(env_id, random=1), make_env(env_id, random=2)
    action = np.array([1.0])
    ts1, ts2 = unroll_env(env1, 10, action)[-1], unroll_env(env2, 10, action)[-1]
    assert (
        ts1.observation["xpos_of_segment_end"] == ts2.observation["xpos_of_segment_end"]
    ).all()


# test time_limit and control_timestep
@pytest.mark.parametrize(
    "env_id,time_limit,control_timestep,n_steps",
    [
        ("two_segments_v1", 10.0, 0.01, 1001),
        ("two_segments_v1", 5.0, 0.01, 501),
        ("two_segments_v1", 10.0, 0.1, 101),
        ("two_segments_v1", 5.0, 0.1, 51),
        ("two_segments_v2", 10.0, 0.01, 1001),
        ("two_segments_v2", 5.0, 0.01, 501),
        ("two_segments_v2", 10.0, 0.1, 101),
        ("two_segments_v2", 5.0, 0.1, 51),
        ("two_segments_v3", 10.0, 0.01, 1001),
        ("two_segments_v3", 5.0, 0.01, 501),
        ("two_segments_v3", 10.0, 0.1, 101),
        ("two_segments_v3", 5.0, 0.1, 51),
    ],
)
def test_time_limit_control_timestep(env_id, time_limit, control_timestep, n_steps):
    env = make_env(
        env_id, time_limit=time_limit, control_timestep=control_timestep, random=1
    )
    assert len(unroll_env(env)) == n_steps
    assert env.time_limit == time_limit
    assert env.control_timestep == control_timestep
    assert (env.ts == np.arange(time_limit, step=control_timestep)).all()
