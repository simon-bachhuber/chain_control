from typing import Tuple, Union

import dm_env
from acme import EnvironmentLoop
from acme.utils import loggers
from tqdm.auto import tqdm

from ..abstract import AbstractController, AbstractModel
from ..buffer import ReplaySample, make_episodic_buffer_adder_iterator
from ..config import use_tqdm
from ..controller import FeedforwardController
from ..env.wrappers import AddRefSignalRewardFnWrapper, ReplacePhysicsByModelWrapper
from ..types import TimeSeriesOfAct
from ..utils import to_jax, to_numpy, tree_concat, tree_shape
from .actor import PolicyActor
from .source import (
    ObservationReferenceSource,
    constant_after_transform_source,
    draw_u_from_cosines,
    draw_u_from_gaussian_process,
)


def concat_iterators(*iterators) -> ReplaySample:
    return tree_concat([next(iterator) for iterator in iterators], True)


def collect_sample(
    env: dm_env.Environment, seeds_gp: list[int], seeds_cos: list[Union[int, float]]
) -> ReplaySample:

    _, sample_gp = collect_feedforward_and_make_source(env, seeds=seeds_gp)
    _, sample_cos = collect_feedforward_and_make_source(
        env, draw_u_from_cosines, seeds=seeds_cos
    )

    return tree_concat([sample_gp, sample_cos], True)


def collect_reference_source(
    env: dm_env.Environment,
    seeds: list[int],
    constant_after: bool = False,
    constant_after_T: float = 3.0,
):

    source, _ = collect_feedforward_and_make_source(env, seeds=seeds)

    if constant_after:
        source = constant_after_transform_source(source, constant_after_T)

    return source


def collect_exhaust_source(
    env: dm_env.Environment,
    source: ObservationReferenceSource,
    controller: AbstractController,
    model: AbstractModel = None,
) -> ReplaySample:

    if model:
        env = ReplacePhysicsByModelWrapper(env, model)

    # wrap env with source
    env = AddRefSignalRewardFnWrapper(env, source)

    N = tree_shape(source._yss)
    # collect performance of controller in environment
    pbar = tqdm(range(N), desc="Reference Iterator", disable=not use_tqdm())
    iterators = []
    for i_actor in pbar:
        source.change_reference_of_actor(i_actor)
        iterator = collect(env, controller)
        iterators.append(iterator)

    # concat samples
    sample = concat_iterators(*iterators)

    return sample


def collect(env: dm_env.Environment, controller: AbstractController):

    env.reset()

    _, adder, iterator = make_episodic_buffer_adder_iterator(
        bs=1,
        env=env,
        buffer_size_n_trajectories=1,
    )

    actor = PolicyActor(policy=controller, action_spec=env.action_spec(), adder=adder)
    loop = EnvironmentLoop(env, actor, logger=loggers.NoOpLogger())
    loop.run_episode()
    return iterator


def collect_feedforward_and_make_source(
    env: dm_env.Environment,
    draw_fn=draw_u_from_gaussian_process,
    seeds: list[int] = [
        0,
    ],
) -> Tuple[ObservationReferenceSource, ReplaySample]:

    assert len(seeds) > 0

    ts = env.ts

    iterators = []
    for seed in seeds:
        us: TimeSeriesOfAct = to_jax(draw_fn(to_numpy(ts), seed=seed))
        policy = FeedforwardController(us)
        iterator = collect(env, policy)
        iterators.append(iterator)

    sample = concat_iterators(*iterators)
    source = ObservationReferenceSource(ts, sample.obs, sample.action)
    return source, sample
