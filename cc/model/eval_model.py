from typing import Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ..abstract import AbstractModel
from ..buffer import ReplaySample
from ..train.unroll import unroll_model
from ..types import BatchedTimeSeriesOfObs
from ..utils import to_jax


def eval_model(
    model: AbstractModel, sample: ReplaySample
) -> Tuple[BatchedTimeSeriesOfObs, float]:
    train_sample = to_jax(sample)
    yhatss = jax.vmap(lambda us: unroll_model(model, us))(train_sample.action)
    rmse = jtu.tree_map(
        lambda a1, a2: jnp.sqrt(jnp.mean((a1 - a2) ** 2)), train_sample.obs, yhatss
    )
    return yhatss, rmse
