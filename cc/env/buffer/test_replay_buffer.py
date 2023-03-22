import jax.random as jrand
import pytest
import ray
from equinox import tree_equal
from tree_utils import tree_slice

from cc.acme import EnvironmentLoop

from ...env import make_env
from ...utils.utils import timestep_array_from_env
from ..collect.actor import RandomActor
from .adder import Adder
from .make_buffer_adder_iterator import make_episodic_buffer_adder_iterator
from .replay_buffer import RayReplayBuffer, buffer_to_iterator
from .sampler import Sampler


def env_fn(time_limit, control_timestep):
    return make_env(
        "two_segments_v1",
        random=1,
        time_limit=time_limit,
        control_timestep=control_timestep,
    )


@pytest.mark.parametrize(
    "time_limit,control_timestep,bs_of_iterator,n_timesteps",
    [
        (5.0, 0.1, 4, 50),
        (5.0, 0.1, 8, 50),
        (5.0, 0.01, 8, 500),
        (10.0, 0.01, 8, 1000),
    ],
)
def test_episodic_buffer_adder_iterator(
    time_limit, control_timestep, bs_of_iterator, n_timesteps
):
    env = env_fn(time_limit, control_timestep)

    buffer, adder, iterator = make_episodic_buffer_adder_iterator(
        bs_of_iterator, env, force_batch_size=False
    )

    # Same RNG

    actor = RandomActor(env.action_spec(), adder=adder, reset_key=True)
    EnvironmentLoop(env, actor).run(num_episodes=2)

    sample = next(iterator)
    assert sample.bs == 2
    assert sample.n_timesteps == n_timesteps
    assert tree_equal(tree_slice(sample, start=0), tree_slice(sample, start=1))

    buffer.reset()
    assert next(iterator) == []

    # Different RNG

    actor = RandomActor(env.action_spec(), adder=adder, reset_key=False)
    EnvironmentLoop(env, actor).run(num_episodes=2)

    sample = next(iterator)
    assert sample.bs == 2
    assert sample.n_timesteps == n_timesteps

    assert not (tree_equal(tree_slice(sample, start=0), tree_slice(sample, start=1)))

    buffer, adder, iterator = make_episodic_buffer_adder_iterator(
        bs_of_iterator, env, True
    )

    # Same RNG

    actor = RandomActor(env.action_spec(), adder=adder, reset_key=True)
    EnvironmentLoop(env, actor).run(num_episodes=2)

    sample = next(iterator)
    assert sample.bs == bs_of_iterator
    assert sample.n_timesteps == n_timesteps
    assert tree_equal(tree_slice(sample, start=0), tree_slice(sample, start=1))


# TODO
def test_nonepisodic_buffer():
    pass


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

    def run_episode(self):
        self.loop.run_episode()


@pytest.mark.parametrize(
    "time_limit,control_timestep,n_timesteps",
    [
        (5.0, 0.1, 50),
        (5.0, 0.01, 500),
        (10.0, 0.01, 1000),
    ],
)
def test_episodic_ray_buffer(time_limit, control_timestep, n_timesteps):
    env_fn_const = lambda: env_fn(time_limit, control_timestep)
    env = env_fn_const()

    buffer, adder, iterator = make_episodic_buffer_adder_iterator(
        3, env, force_batch_size=False, buffer_size_n_trajectories=3
    )

    for i in range(3):
        actor = RandomActor(
            env.action_spec(),
            adder=adder,
            key=jrand.PRNGKey(
                i,
            ),
        )
        EnvironmentLoop(env, actor).run_episode()

    sample_sync = next(iterator)

    buffer = RayReplayBuffer.remote(
        sampler=Sampler(
            env,
            len(timestep_array_from_env(env)),
            episodic=True,
            sample_with_replacement=False,
        )
    )

    loops = [RayEnvironmentLoop.remote(buffer, i, env_fn_const) for i in range(3)]
    for loop in loops:
        # otherwise the ordering of the 3 trajectories is ambiguous
        ray.get(loop.run_episode.remote())

    assert ray.get(buffer.len.remote()) == 3 * len(timestep_array_from_env(env))

    iterator = buffer_to_iterator(buffer, 3, force_batch_size=False)
    sample_async1 = next(iterator)

    assert sample_async1.bs == 3
    assert sample_async1.n_timesteps == len(timestep_array_from_env(env))
    assert tree_equal(sample_sync.action, sample_async1.action)
    assert tree_equal(sample_sync, sample_async1)

    # tests reset mechanism
    ray.get(buffer.reset.remote())

    for loop in loops:
        ray.get(loop.run_episode.remote())

    assert ray.get(buffer.len.remote()) == 3 * len(timestep_array_from_env(env))
    sample_async2 = next(iterator)
    assert tree_equal(sample_async1, sample_async2)

    ray.shutdown()


# TODO
def test_nonepisodic_ray_buffer():
    pass
