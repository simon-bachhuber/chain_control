import time

import jax.random as jrand
import pytest
import ray
from acme import EnvironmentLoop

from ..collect.actor import RandomActor
from ..env import make_env
from .adder import Adder
from .rate_limiting import NoRateLimitingLimiter
from .replay_buffer import RayReplayBuffer, buffer_to_iterator
from .sampler import Sampler


def env_fn(time_limit, control_timestep):
    return make_env(
        "two_segments_v1",
        random=1,
        time_limit=time_limit,
        control_timestep=control_timestep,
    )


@ray.remote
class RayEnvironmentLoop:
    def __init__(self, buffer, i, env_fn):
        adder = Adder(buffer)
        adder.set_actor_id(i)
        env = env_fn()
        actor = RandomActor(
            env.action_spec(),
            adder=adder,
            key=jrand.PRNGKey(
                i,
            ),
            reset_key=True,
        )
        self.loop = EnvironmentLoop(env, actor)

    def run(self):
        self.loop.run()


@ray.remote
class Learner:
    def __init__(self, buffer):
        self.iterator = buffer_to_iterator(buffer, 4)

    def run(self):
        while True:
            next(self.iterator)


@pytest.mark.parametrize(
    "target_ratio",
    [
        1.0,
    ],
)  # 0.5, 1.0, 2.0])
def test_rate_limiting(target_ratio):
    time_limit = 10.0
    control_timestep = 0.01
    # rate_limiter = RateLimiter(target_ratio=target_ratio, error_margin=0.1,
    # update_ratio_freq=10)
    rate_limiter = NoRateLimitingLimiter()
    env_fn_const = lambda: env_fn(time_limit, control_timestep)
    env = env_fn_const()
    buffer = RayReplayBuffer.remote(
        maxlen=50_000,
        sampler=Sampler(env, length_of_trajectories=3, episodic=False),
        rate_limiter=rate_limiter,
    )

    loop = RayEnvironmentLoop.remote(buffer, 0, env_fn_const)
    learner = Learner.remote(buffer)

    loop.run.remote()
    learner.run.remote()

    time.sleep(10)

    # TODO
    # This is super weird.
    # Baseline for `NoRateLimiter` ~5000 inserts
    # Only for a large enough bs in `buffer_to_iterator(buffer, bs)`
    # Can something close ~4000 inserts be achieved
    # Small bs will reduce the number of inserts
    # even though there might be now really rate limiting
    # because there is a huge error_margin
    # ... TBC

    # current_ratio = ray.get(buffer.current_ratio.remote())
    # print("Current ratio: ", current_ratio)
    # print("Number of inserts: ", ray.get(buffer.len.remote()))
    # assert False
    # assert target_ratio - 0.01 < current_ratio < target_ratio + 0.01
