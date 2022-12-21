import functools as ft
from typing import Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np

from ..core import PRNGKey, PyTree
from ..utils import to_jax, tree_indices, tree_shape


@ft.partial(jax.jit, static_argnums=(1, 2, 3))
def gen_minibatch_indices(
    key, batch_size: int, n_minibatches: int, minibatch_size: int
) -> jnp.ndarray:
    consume1, consume2 = jrand.split(key)
    permutation = jax.random.permutation(consume1, jnp.arange(batch_size))
    permutation_bootstrap = jax.random.permutation(consume2, jnp.arange(batch_size))
    permutation = jnp.hstack((permutation, permutation_bootstrap))

    def scan_fn(carry, _):
        start_idx = carry
        y = jnp.take(
            jnp.arange(batch_size),
            jax.lax.dynamic_slice_in_dim(permutation, start_idx, minibatch_size),
        )
        carry = start_idx + minibatch_size
        return carry, y

    return jax.lax.scan(scan_fn, 0, length=n_minibatches, xs=None)[1]


@ft.partial(jax.jit, static_argnums=(1, 2, 3))
def gen_minibatch_masks(
    key, batch_size: int, n_minibatches: int, minibatch_size: int
) -> jnp.ndarray:

    idxss = gen_minibatch_indices(key, batch_size, n_minibatches, minibatch_size)

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


class SupervisedDataset(NamedTuple):
    inputs: PyTree
    targets: PyTree


class SupervisedDatasetWithWeights(NamedTuple):
    inputs: PyTree
    targets: PyTree
    weights: np.ndarray


class UnsupervisedDataset(NamedTuple):
    refss: PyTree


Dataset = Union[SupervisedDataset, SupervisedDatasetWithWeights, UnsupervisedDataset]
MiniBatchUpdateFn = Callable[
    [MiniBatchState],
    Tuple[MiniBatchState, Union[SupervisedDatasetWithWeights, UnsupervisedDataset]],
]


class Dataloader(NamedTuple):
    minibatch_state: MiniBatchState
    update_fn: MiniBatchUpdateFn


def bootstrap_minibatch_size(
    n_minibatches: int, batchsize: int, do_bootstrapping: bool
) -> int:
    if not do_bootstrapping:
        assert batchsize % n_minibatches == 0
    else:
        for i in range(1000):  # TODO
            batchsize += i
            if batchsize % n_minibatches == 0:
                break

    return batchsize // n_minibatches


def make_dataloader(
    dataset: Dataset,
    key: PRNGKey,
    n_minibatches: int = 1,
    axis: int = 0,
    reshuffle: bool = True,
    tree_transform: Optional[Callable] = None,
    do_bootstrapping: bool = False,
) -> Dataloader:

    if not isinstance(
        dataset, (SupervisedDataset, SupervisedDatasetWithWeights, UnsupervisedDataset)
    ):
        raise ValueError(
            "`dataset` should be either a `SupervisedDataset`, \
        `SupervisedDatasetWithWeights` or `UnsupervisedDataset`."
        )

    bs = tree_shape(dataset, axis)

    if isinstance(dataset, SupervisedDataset):
        dataset = SupervisedDatasetWithWeights(
            dataset.inputs, dataset.targets, np.ones((bs,))
        )

    dataset = to_jax(dataset)

    def init_minibatch_state():
        minibatch_size = bootstrap_minibatch_size(n_minibatches, bs, do_bootstrapping)
        inner_key, consume = jrand.split(key)

        return MiniBatchState(
            gen_minibatch_indices(consume, bs, n_minibatches, minibatch_size),
            0,
            bs,
            n_minibatches,
            minibatch_size,
            inner_key,
        )

    # closures `data`
    def update_fn(
        state: MiniBatchState,
    ) -> Tuple[
        MiniBatchState, Union[SupervisedDatasetWithWeights, UnsupervisedDataset]
    ]:

        indices = state.indices
        key = state.key
        if state.i >= state.n_minibatches:
            # iteration over one epoch is done
            if reshuffle:
                key, consume = jrand.split(key)
                indices = gen_minibatch_indices(
                    consume, state.bs, state.n_minibatches, state.minibatch_size
                )

        # reset counter if required
        i = state.i % state.n_minibatches

        batch_of_tree = tree_indices(dataset, indices[i], axis)

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
