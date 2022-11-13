import jax.random as jrand
from absl.testing import absltest
from dm_env import test_utils

from ...module_examples.neural_ode_v1 import make_neural_ode
from ..make_env import make_env
from .replace_physics_by_model import ReplacePhysicsByModelWrapper

LENGTH_ACTION_SEQUENCE = 2001


def dummy_env():
    return make_env("two_segments_v1", random=1)


def dummy_model():
    env = dummy_env()
    return make_neural_ode(
        env.action_spec(),
        env.observation_spec(),
        env.control_timestep,
        3,
        jrand.PRNGKey(
            1,
        ),
        "dummy-model",
    )


def test_attributes():
    env = dummy_env()
    model = dummy_model()
    env_model = ReplacePhysicsByModelWrapper(env, model)

    assert env.time_limit == env_model.time_limit
    assert env.control_timestep == env_model.control_timestep
    assert (env.ts == env_model.ts).all()


class TestTwoSegmentsV1(test_utils.EnvironmentTestMixin, absltest.TestCase):
    def make_object_under_test(self):
        env = dummy_env()
        model = dummy_model()
        return ReplacePhysicsByModelWrapper(env, model)

    def make_action_sequence(self):
        for _ in range(LENGTH_ACTION_SEQUENCE):
            yield self.make_action()
