import jax.random as jrand
from absl.testing import absltest
from dm_env import test_utils

from ...model import LinearModel, LinearModelOptions
from ..make_env import make_env
from .replace_physics_by_model import ReplacePhysicsByModelWrapper

LENGTH_ACTION_SEQUENCE = 2001


def dummy_model():
    options = LinearModelOptions(
        3,
        1,
        1,
        "EE",
        jrand.PRNGKey(
            1,
        ),
    )
    return LinearModel(options)


def test_attributes():
    env = make_env("two_segments_v1", random=1)
    env_model = ReplacePhysicsByModelWrapper(env, dummy_model())

    assert env.time_limit == env_model.time_limit
    assert env.control_timestep == env_model.control_timestep
    assert (env.ts == env_model.ts).all()


class TestTwoSegmentsV1(test_utils.EnvironmentTestMixin, absltest.TestCase):
    def make_object_under_test(self):
        env = make_env("two_segments_v1", random=1)
        return ReplacePhysicsByModelWrapper(env, dummy_model())

    def make_action_sequence(self):
        for _ in range(LENGTH_ACTION_SEQUENCE):
            yield self.make_action()
