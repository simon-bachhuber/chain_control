import functools as ft
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrand

from ..core import PRNGKey, PyTree
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


MiniBatchUpdateFn = Callable[[MiniBatchState], Tuple[MiniBatchState, PyTree]]


class Dataset(NamedTuple):
    inputs: PyTree[jnp.ndarray]
    targets: PyTree[jnp.ndarray]


class Dataloader(NamedTuple):
    minibatch_state: MiniBatchState
    update_fn: MiniBatchUpdateFn


# TODO `data` is incorrectly typed; should be `PyTree[np.ndarray]`;
# use `to_jax` explicitly and not some implicit promotion
def make_dataloader(
    data: PyTree[jnp.ndarray],
    key: PRNGKey,
    n_minibatches: int = 1,
    axis: int = 0,
    reshuffle: bool = True,
    tree_transform: Optional[Callable] = None,
) -> Dataloader:
    def init_minibatch_state():
        bs = tree_shape(data, axis)
        assert bs % n_minibatches == 0
        minibatch_size = bs // n_minibatches
        inner_key, consume = jrand.split(key)

        return MiniBatchState(
            gen_minibatch_indices(consume, bs, minibatch_size),
            0,
            bs,
            n_minibatches,
            minibatch_size,
            inner_key,
        )

    # closures `data`
    def update_fn(state: MiniBatchState) -> Tuple[MiniBatchState, PyTree]:

        indices = state.indices
        key = state.key
        if state.i >= state.n_minibatches:
            # iteration over one epoch is done
            if reshuffle:
                key, consume = jrand.split(key)
                indices = gen_minibatch_indices(consume, state.bs, state.minibatch_size)

        # reset counter if required
        i = state.i % state.n_minibatches

        batch_of_tree = tree_indices(data, indices[i], axis)

        if tree_transform:
            key, consume = jrand.split(key)
            batch_of_tree = tree_transform(consume, batch_of_tree, state.minibatch_size)

        return (
            MiniBatchState(
                indices, i + 1, state.bs, state.n_minibatches, state.minibatch_size, key
            ),
            batch_of_tree,
        )

    return Dataloader(init_minibatch_state(), update_fn)
