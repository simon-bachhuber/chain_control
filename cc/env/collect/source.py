import copy
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from ...core import AbstractObservationReferenceSource
from ...core.types import (
    BatchedTimeSeriesOfAct,
    BatchedTimeSeriesOfRef,
    TimeSeriesOfRef,
)
from ...utils import to_numpy, tree_slice


class ObservationReferenceSource(AbstractObservationReferenceSource):
    def __init__(
        self,
        ts: jnp.ndarray,
        yss: BatchedTimeSeriesOfRef,
        uss: Optional[BatchedTimeSeriesOfAct] = None,
        i_actor=0,
        reference_transform=lambda arr: arr,
    ):
        self._i_actor = i_actor
        self._ts = ts
        self._yss = yss
        self._uss = uss
        self._reference_transform = reference_transform

    def set_reference_transform(self, transform):
        self._reference_transform = transform

    def get_references_for_optimisation_untransformed(self) -> BatchedTimeSeriesOfRef:
        transform = lambda arr: jnp.atleast_3d(arr)
        return jax.tree_util.tree_map(transform, self._yss)

    def get_references_for_optimisation(self) -> BatchedTimeSeriesOfRef:
        transform = lambda arr: jax.vmap(self._reference_transform)(jnp.atleast_3d(arr))
        return jax.tree_util.tree_map(transform, self._yss)

    def get_reference_actor(self) -> TimeSeriesOfRef:
        return tree_slice(self.get_references_for_optimisation(), start=self._i_actor)

    def change_reference_of_actor(self, i_actor: int) -> None:
        self._i_actor = i_actor


default_kernel = 0.2 * RBF(1.5)


def draw_u_from_gaussian_process(ts, kernel=default_kernel, seed=1):
    ts = ts[:, np.newaxis]
    us = GaussianProcessRegressor(kernel=kernel).sample_y(ts, random_state=seed)
    us = (us - np.mean(us)) / np.std(us)
    return us.astype(np.float32)


def draw_u_from_cosines(ts, seed):
    freq = seed
    ts = ts / ts[-1]
    omega = 2 * np.pi * freq
    return np.cos(omega * ts)[:, None] * np.sqrt(freq)


def _constant_after_transform(refs: jnp.ndarray, idx, new_ts):
    N = len(new_ts) + 1
    refs_out = jnp.zeros((N, refs.shape[1]))
    refs_out = refs_out.at[:idx].set(refs[:idx])
    refs_out = refs_out.at[idx:].set(refs[idx])
    return refs_out


def constant_after_transform_source(
    source: ObservationReferenceSource, after_T: float, new_ts: jnp.ndarray = None
) -> ObservationReferenceSource:
    if new_ts is None:
        new_ts = source._ts

    source = copy.deepcopy(source)
    idx = jnp.where(new_ts == after_T)[0][0]
    transform = lambda arr: _constant_after_transform(arr, idx, new_ts)
    source.set_reference_transform(transform)
    return ObservationReferenceSource(
        new_ts, to_numpy(source.get_references_for_optimisation())
    )
