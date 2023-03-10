"""Many prebuilt `ReferenceSource` object that contain useful maneuvers."""

import dm_env
import jax
import numpy as np
from tree_utils import tree_batch

from ...core.types import BatchedTimeSeriesOfRef
from ...utils import timestep_array_from_env
from .collect import append_source
from .source import ObservationReferenceSource


def random_steps_source(
    env: dm_env.Environment,
    seeds: list[int],
    min_abs_amplitude: float = 0.0,
    max_abs_amplitude: float = 3.0,
):
    ts = timestep_array_from_env(env)

    def tree_constant(tree):
        sign = np.random.choice([-1.0, 1.0])
        value = lambda arr: sign * np.random.uniform(
            min_abs_amplitude, max_abs_amplitude, size=(1,) + arr.shape
        )
        duplicate = lambda arr: np.repeat(value(arr), len(ts) + 1, axis=0)
        return jax.tree_map(duplicate, tree)

    yss = []
    for seed in seeds:
        np.random.seed(seed)
        yss.append(tree_constant(env.observation_spec()))

    _yss = BatchedTimeSeriesOfRef(tree_batch(yss))
    return ObservationReferenceSource(_yss, ts=ts)


def high_steps_source(env, amplitude: float = 6.0):
    source = random_steps_source(env, seeds=[0, 1])
    yss = jax.tree_util.tree_leaves(source._yss)[0]
    yss[0, :] = amplitude
    yss[1, :] = -amplitude
    return source


def double_step_source(env, amplitude: float = 2.0):
    source = random_steps_source(env, seeds=[0, 1])
    halfpoint = int((len(source._ts) - 1) / 2)
    yss = jax.tree_util.tree_leaves(source._yss)[0]
    yss[:, :halfpoint] = amplitude
    yss[0, halfpoint:] = 2 * amplitude
    yss[1, halfpoint:] = 0.0
    return source


def random_double_steps_source(
    env,
    seeds: list[int],
    min_abs_amplitude: float = 0.0,
    max_abs_amplitude: float = 3.0,
):
    sources = []
    for seed in seeds:
        # TODO
        # shift seed such that it is different
        source = random_steps_source(
            env, [seed + 7000, 1], min_abs_amplitude, max_abs_amplitude
        )
        halfpoint = int((len(source._ts) - 1) / 2)
        yss = jax.tree_util.tree_leaves(source._yss)[0]
        amplitude = yss[0, 0]
        yss[:, :halfpoint] = amplitude
        yss[0, halfpoint:] = 2 * amplitude
        yss[1, halfpoint:] = 0.0
        sources.append(source)

    source = sources[0]
    for next_source in sources[1:]:
        source = append_source(source, next_source)

    return source
