import time

import jax.random as jrand
import pytest
import ray
from acme import EnvironmentLoop

from ..collect.actor import RandomActor
from ...env import make_env
from .adder import Adder
from .rate_limiting import RateLimiter
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
        adder = Adder(buffer, 25)
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
        self.iterator = buffer_to_iterator(buffer, 64)

    def run(self):
        while True:
            next(self.iterator)


@pytest.mark.parametrize(
    "target_ratio",
    [0.1, 0.2, 0.4],
)
def test_rate_limiting(target_ratio):
    time_limit = 10.0
    control_timestep = 0.01
    error_margin = 0.1

    rate_limiter = RateLimiter(target_ratio=target_ratio, error_margin=error_margin)
    env_fn_const = lambda: env_fn(time_limit, control_timestep)
    env = env_fn_const()
    buffer = RayReplayBuffer.remote(
        maxlen=50_000,
        sampler=Sampler(env, length_of_trajectories=5, episodic=False),
        rate_limiter=rate_limiter,
    )

    loop = RayEnvironmentLoop.remote(buffer, 0, env_fn_const)
    learner = Learner.remote(buffer)

    loop.run.remote()
    learner.run.remote()

    time.sleep(10)

    # comment 07.11.22
    # the more expensive the `iterator` / sampling becomes
    # the more optimal a lower `target_ratio` becomes
    # ~15.000 insertions is optimal
    #
    # example:
    # `buffer_to_iterator(buffer, 64)`
    # `Adder(buffer, 100)`
    # `target_ratio` = 0.1
    # -> ~15.000 insertions
    #
    # `buffer_to_iterator(buffer, 64)`
    # `Adder(buffer, 25)`
    # `target_ratio` = 0.1
    # -> ~12.500 insertions

    current_ratio = ray.get(buffer.current_ratio.remote())
    print("Current ratio: ", current_ratio)
    print("Number of inserts: ", ray.get(buffer.len.remote()))
    assert target_ratio - error_margin < current_ratio < target_ratio + error_margin
