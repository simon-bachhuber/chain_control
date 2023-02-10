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


DYNAMIC_INPUTS = [
    0.04,
    0.51,
    0.64,
    0.29,
    0.99,
    -0.15,
    0.41,
    0.39,
    0.84,
    0.64,
    0.81,
    0.70,
    0.47,
    0.15,
    0.54,
    0.69,
    0.53,
    -0.09,
    0.74,
    0.32,
    -6.48,
    8.78,
    8.98,
    2.00,
    5.50,
    6.98,
    -9.78,
    1.42,
    4.4,
    6.13,
    7.91,
    3.71,
    6.29,
    9.94,
    5.12,
    8.86,
    9.61,
    6.2,
    9.37,
    9.3,
]


@pytest.mark.parametrize(
    "env_id,expected_positions",
    [
        (
            "two_segments_v1",
            [
                2.5717583e-16,
                1.17395275e-05,
                2.8966924e-05,
                5.200203e-06,
                -9.083662e-05,
                -0.00031538485,
                -0.0006903377,
                -0.0011962753,
                -0.0018162847,
                -0.002541724,
                -0.0033696352,
                -0.004295176,
                -0.0053182584,
                -0.0064323083,
                -0.007595568,
                -0.008756365,
                -0.009888207,
                -0.010986462,
                -0.012003554,
                -0.012900721,
                -0.013712076,
                -0.014316577,
                -0.014615711,
                -0.01468627,
                -0.014618133,
                -0.014504818,
                -0.014522608,
                -0.014644113,
                -0.014727252,
                -0.014733659,
                -0.01464793,
                -0.014467297,
                -0.014184259,
                -0.013784379,
                -0.013258573,
                -0.012589387,
                -0.011750143,
                -0.010712488,
                -0.009438066,
                -0.007881451,
            ],
        ),
        (
            "two_segments_v2",
            [
                2.5717583e-16,
                1.45026e-05,
                5.4280306e-05,
                9.83285e-05,
                0.00013637848,
                0.00013150358,
                5.3611562e-05,
                -9.701852e-05,
                -0.00032819706,
                -0.00065433944,
                -0.0010973752,
                -0.0016772475,
                -0.0024211162,
                -0.003357453,
                -0.0044886335,
                -0.005799737,
                -0.007288479,
                -0.008969179,
                -0.010817105,
                -0.012797643,
                -0.014949502,
                -0.01718296,
                -0.019367592,
                -0.021498963,
                -0.023572784,
                -0.025584381,
                -0.027641464,
                -0.02974657,
                -0.031798713,
                -0.033792112,
                -0.035743672,
                -0.03768542,
                -0.039651778,
                -0.04167442,
                -0.043794014,
                -0.04604906,
                -0.048471745,
                -0.051094096,
                -0.053938996,
                -0.05701878,
            ],
        ),
        (
            "two_segments_v3",
            [
                2.5717583e-16,
                1.4536366e-05,
                5.520506e-05,
                0.00010365636,
                0.00015368096,
                0.00017352053,
                0.000138419,
                5.265217e-05,
                -8.830468e-05,
                -0.00029599993,
                -0.0005897668,
                -0.0009876428,
                -0.0015158466,
                -0.002203603,
                -0.003056767,
                -0.0040666927,
                -0.0052390685,
                -0.0065974975,
                -0.008128704,
                -0.009811281,
                -0.01169719,
                -0.013712974,
                -0.015745524,
                -0.017801387,
                -0.019880839,
                -0.021977874,
                -0.02419255,
                -0.026519243,
                -0.028851504,
                -0.031174827,
                -0.033493396,
                -0.03582325,
                -0.038180877,
                -0.040579565,
                -0.043042097,
                -0.045590986,
                -0.04824565,
                -0.05102948,
                -0.053961862,
                -0.057057504,
            ],
        ),
    ],
)
def test_dynamic_environments(env_id, expected_positions):
    env = make_env(env_id, random=1)

    for index, expected_position in enumerate(expected_positions):
        res = env.step(DYNAMIC_INPUTS[index])
        assert res.observation["xpos_of_segment_end"] == expected_position
