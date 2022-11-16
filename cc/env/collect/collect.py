from collections import OrderedDict
from typing import Tuple, Union

import dm_env
import numpy as np
from acme import EnvironmentLoop
from acme.utils import loggers
from tqdm.auto import tqdm

from ...config import use_tqdm
from ...core import AbstractController
from ...core.types import TimeSeriesOfAct
from ...env.wrappers import AddRefSignalRewardFnWrapper
from ...examples.feedforward_controller import make_feedforward_controller
from ...utils import to_jax, to_numpy, tree_concat, tree_shape
from ...utils.legacy_wrapper import WrapController
from ..buffer import ReplaySample, make_episodic_buffer_adder_iterator
from .actor import ModuleActor
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

    _, sample_gp = sample_feedforward_collect_and_make_source(env, seeds=seeds_gp)
    _, sample_cos = sample_feedforward_collect_and_make_source(
        env, draw_u_from_cosines, seeds=seeds_cos
    )

    return concat_samples(sample_gp, sample_cos)


def collect_exhaust_source(
    env: dm_env.Environment,
    controller: AbstractController,
) -> ReplaySample:

    assert isinstance(env, AddRefSignalRewardFnWrapper)
    source = env._source

    N = tree_shape(source._yss)
    # collect performance of controller in environment
    pbar = tqdm(range(N), desc="Reference Iterator", disable=not use_tqdm())
    samples = []
    for i_actor in pbar:
        source.change_reference_of_actor(i_actor)
        sample = collect(env, controller)
        samples.append(sample)

    # concat samples
    sample = concat_samples(*samples)

    return sample


def collect(env: dm_env.Environment, controller: AbstractController) -> ReplaySample:

    env.reset()

    buffer, adder, iterator = make_episodic_buffer_adder_iterator(
        bs=1,
        env=env,
        buffer_size_n_trajectories=1,
    )

    actor = ModuleActor(
        controller=controller, action_spec=env.action_spec(), adder=adder
    )
    loop = EnvironmentLoop(env, actor, logger=loggers.NoOpLogger())
    loop.run_episode()
    sample = next(iterator)
    buffer.close()
    return sample


def sample_feedforward_collect_and_make_source(
    env: dm_env.Environment,
    draw_fn=draw_u_from_gaussian_process,
    seeds: list[int] = [
        0,
    ],
) -> Tuple[ObservationReferenceSource, ReplaySample]:

    assert len(seeds) > 0

    ts = env.ts

    samples = []
    for seed in seeds:
        us: TimeSeriesOfAct = to_jax(draw_fn(to_numpy(ts), seed=seed))
        policy = WrapController(make_feedforward_controller(us))
        sample = collect(env, policy)
        samples.append(sample)

    sample = concat_samples(*samples)
    source = ObservationReferenceSource(ts, sample.obs, sample.action)
    return source, sample


def collect_random_step_source(env: dm_env.Environment, seeds: list[int]):
    ts = env.ts
    yss = np.zeros((len(seeds), len(ts) + 1, 1))

    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        yss[i] = np.random.uniform(-3.0, 3.0)

    _yss = OrderedDict()
    _yss["xpos_of_segment_end"] = yss
    return ObservationReferenceSource(ts, _yss)
