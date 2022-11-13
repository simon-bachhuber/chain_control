import functools as ft
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jrand

from ..core import PRNGKey, PyTree, make_module_from_function
from ..utils import tree_indices, tree_shape


@ft.partial(jax.jit, static_argnums=(1, 2))
def gen_minibatch_indices(key, batch_size: int, minibatch_size: int) -> jnp.ndarray:
    permutation = jax.random.permutation(key, jnp.arange(batch_size))

    def scan_fn(carry, _):
        start_idx = carry
        y = jnp.take(
            jnp.arange(batch_size),
            jax.lax.dynamic_slice_in_dim(permutation, start_idx, minibatch_size),
        )
        carry = start_idx + minibatch_size
        return carry, y

    return jax.lax.scan(scan_fn, 0, length=batch_size // minibatch_size, xs=None)[1]


@ft.partial(jax.jit, static_argnums=(1, 2))
def gen_minibatch_masks(key, batch_size: int, minibatch_size: int) -> jnp.ndarray:

    idxss = gen_minibatch_indices(key, batch_size, minibatch_size)

    # generate masks from idxs
    def to_mask(idxs):
        return jnp.in1d(jnp.arange(batch_size), idxs)

    return jax.vmap(to_mask)(idxss)


class MiniBatchState(NamedTuple):
    indices: jnp.ndarray
    i: int
    bs: int
    n_minibatches: int
    minibatch_size: int
    key: PRNGKey


def make_dataloader(
    data: PyTree,
    key,
    n_minibatches: int = 1,
    batch_axis: int = 0,
    reshuffle: bool = True,
    data_transform: Optional[Callable] = None,
    data_split_fn: Callable = lambda data: (data[0], data[1]),
):

    bs = tree_shape(data, batch_axis)
    assert bs % n_minibatches == 0
    minibatch_size = bs // n_minibatches
    key, consume = jrand.split(key)

    init_state = MiniBatchState(
        gen_minibatch_indices(consume, bs, minibatch_size),
        0,
        bs,
        n_minibatches,
        minibatch_size,
        key,
    )

    # closures `data`
    def forward(params, state: MiniBatchState, x):
        indices = state.indices
        key = state.key
        if state.i >= state.n_minibatches:
            # iteration over one epoch is done
            if reshuffle:
                key, consume = jrand.split(key)
                indices = gen_minibatch_indices(consume, state.bs, state.minibatch_size)

        # reset counter if required
        i = state.i % state.n_minibatches

        batch_of_data = tree_indices(data, indices[i], batch_axis)

        if data_transform:
            key, consume = jrand.split(key)
            batch_of_data = data_transform(consume, batch_of_data, state.minibatch_size)

        return (
            MiniBatchState(
                indices,
                i + 1,
                state.bs,
                state.n_minibatches,
                state.minibatch_size,
                key,
            ),
            data_split_fn(batch_of_data),
        )

    return make_module_from_function(forward, {}, init_state, name="dataloader")
