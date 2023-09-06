from absl.testing import absltest
from dm_env import test_utils
import tree_utils

from cc.env.make_env import make_env
from cc.env.wrappers import NoisyActionsWrapper
from cc.env.wrappers import NoisyObservationsWrapper

LENGTH_ACTION_SEQUENCE = 2001


class TestNoisyActions(test_utils.EnvironmentTestMixin, absltest.TestCase):
    def make_object_under_test(self):
        return NoisyActionsWrapper(make_env("two_segments_v1"))

    def make_action_sequence(self):
        for _ in range(LENGTH_ACTION_SEQUENCE):
            yield self.make_action()


class TestNoisyObservations(test_utils.EnvironmentTestMixin, absltest.TestCase):
    def make_object_under_test(self):
        return NoisyObservationsWrapper(make_env("two_segments_v1"))

    def make_action_sequence(self):
        for _ in range(LENGTH_ACTION_SEQUENCE):
            yield self.make_action()


def test_noisy_output():
    make = lambda seed: NoisyObservationsWrapper(make_env("two_segments_v1"), seed=seed)
    env = make(1)

    first_ts_first_seed = env.reset()
    assert tree_utils.tree_close(first_ts_first_seed, env.reset(True))
    assert not tree_utils.tree_close(first_ts_first_seed, env.reset())
    assert not tree_utils.tree_close(first_ts_first_seed, make(2).reset())


def test_noisy_action():
    make = lambda seed: NoisyActionsWrapper(make_env("two_segments_v1"), seed=seed)

    assert tree_utils.tree_close(make(1).reset(), make(2).reset())

    env = make(1)
    env.reset()
    ts = env.step([0.5])
    env.reset()
    ts_wo = env.step([0.5])
    env.reset(True)
    ts_w = env.step([0.5])
    assert not tree_utils.tree_close(ts, ts_wo)
    assert tree_utils.tree_close(ts, ts_w)

    env1, env2 = make(1), make(2)
    env1.reset()
    env2.reset()
    action = [0.5]
    ts1, ts2 = env1.step(action), env2.step(action)
    assert not tree_utils.tree_close(ts1, ts2)
