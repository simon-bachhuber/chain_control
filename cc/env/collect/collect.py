from collections import OrderedDict
from typing import Sequence, Tuple, Union

import dm_env
import numpy as np
from acme import EnvironmentLoop
from acme.utils import loggers
from tqdm.auto import tqdm

from ...config import use_tqdm
from ...core import AbstractController
from ...core.types import BatchedTimeSeriesOfRef, TimeSeriesOfAct
from ...env.wrappers import AddRefSignalRewardFnWrapper
from ...examples.feedforward_controller import make_feedforward_controller
from ...utils import to_jax, to_numpy, tree_concat, tree_shape
from ..buffer import ReplaySample, make_episodic_buffer_adder_iterator
from ..loop_observer import EnvLoopObserver
from .actor import ModuleActor
from cc.env.wrappers import RecordVideoWrapper
from .source import (
    ObservationReferenceSource,
    draw_u_from_cosines,
    draw_u_from_gaussian_process,
)


def concat_samples(*samples) -> ReplaySample:
    return tree_concat(samples, True)


def sample_feedforward_and_collect(
    env: dm_env.Environment, seeds_gp: list[int], seeds_cos: list[Union[int, float]]
) -> ReplaySample:

    _, sample_gp, _ = sample_feedforward_collect_and_make_source(env, seeds=seeds_gp)
    _, sample_cos, _ = sample_feedforward_collect_and_make_source(
        env, draw_u_from_cosines, seeds=seeds_cos
    )

    return concat_samples(sample_gp, sample_cos)


def collect_exhaust_source(
    env: dm_env.Environment,
    controller: AbstractController,
    observers: Sequence[EnvLoopObserver] = (),
) -> Tuple[ReplaySample, dict]:

    assert isinstance(env, AddRefSignalRewardFnWrapper) or isinstance(env, RecordVideoWrapper)
    source = env._source

    N = tree_shape(source._yss)
    # collect performance of controller in environment
    pbar = tqdm(range(N), desc="Reference Iterator", disable=not use_tqdm())
    samples, loop_results = [], []
    for i_actor in pbar:
        source.change_reference_of_actor(i_actor)
        sample, loop_result = collect(env, controller, observers)
        samples.append(sample)
        loop_results.append(loop_result)

    # concat samples
    sample = concat_samples(*samples)

    return sample, tree_concat(loop_results)


def collect(
    env: dm_env.Environment,
    controller: AbstractController,
    observers: Sequence[EnvLoopObserver] = (),
) -> Tuple[ReplaySample, dict]:

    env.reset()

    buffer, adder, iterator = make_episodic_buffer_adder_iterator(
        bs=1,
        env=env,
        buffer_size_n_trajectories=1,
    )

    actor = ModuleActor(
        controller=controller, action_spec=env.action_spec(), adder=adder
    )
    loop = EnvironmentLoop(env, actor, logger=loggers.NoOpLogger(), observers=observers)
    loop_result = loop.run_episode()
    sample = next(iterator)
    buffer.close()
    return sample, loop_result


def sample_feedforward_collect_and_make_source(
    env: dm_env.Environment,
    draw_fn=draw_u_from_gaussian_process,
    seeds: list[int] = [
        0,
    ],
    observers: Sequence[EnvLoopObserver] = (),
) -> Tuple[ObservationReferenceSource, ReplaySample, dict]:

    assert len(seeds) > 0

    ts = env.ts

    samples, loop_results = [], []
    for seed in seeds:
        us: TimeSeriesOfAct = to_jax(draw_fn(to_numpy(ts), seed=seed))
        controller = make_feedforward_controller(us)
        sample, loop_result = collect(env, controller, observers)
        samples.append(sample)
        loop_results.append(loop_result)

    sample = concat_samples(*samples)
    source = ObservationReferenceSource(sample.obs, ts, sample.action)
    return source, sample, tree_concat(loop_results)


def collect_random_step_source(env: dm_env.Environment, seeds: list[int]):
    ts = env.ts
    yss = np.zeros((len(seeds), len(ts) + 1, 1))

    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        yss[i] = np.random.uniform(-3.0, 3.0)

    _yss = OrderedDict()
    _yss["xpos_of_segment_end"] = yss
    _yss = BatchedTimeSeriesOfRef(_yss)
    return ObservationReferenceSource(_yss, ts=ts)


def append_source(first: ObservationReferenceSource, second: ObservationReferenceSource):
    yss = OrderedDict()
    for key in first._yss.keys():
        yss[key] = np.concatenate([first._yss[key], second._yss[key]], axis=0)
    
    return ObservationReferenceSource(yss, ts=first._ts)
