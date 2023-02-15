from absl.testing import absltest
from dm_env import test_utils

from ..collect import sample_feedforward_collect_and_make_source
from ..make_env import make_env
from .add_reference_and_reward import AddRefSignalRewardFnWrapper
from ...utils.utils import time_limit_from_env, timestep_array_from_env

LENGTH_ACTION_SEQUENCE = 2001


def dummy_env():
    return make_env("two_segments_v1", random=1)


def dummy_source(env):
    return sample_feedforward_collect_and_make_source(env, seeds=[0])[0]


def test_attributes():
    env = dummy_env()
    env_w_rew = AddRefSignalRewardFnWrapper(env, dummy_source(env))

    assert time_limit_from_env(env) == time_limit_from_env(env_w_rew)
    assert env.control_timestep() == env_w_rew.control_timestep()
    assert (timestep_array_from_env(env) == timestep_array_from_env(env_w_rew)).all()


class TestTwoSegmentsV1(test_utils.EnvironmentTestMixin, absltest.TestCase):
    def make_object_under_test(self):
        env = dummy_env()
        return AddRefSignalRewardFnWrapper(env, dummy_source(env))

    def make_action_sequence(self):
        for _ in range(LENGTH_ACTION_SEQUENCE):
            yield self.make_action()


# TODO
# add test for different reward functions

# TODO
# add test that checks that changing `source._actor_id`
# changes both the reference and experienced rewards
