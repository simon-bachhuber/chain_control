from typing import Callable, NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from tree_utils import PyTree, tree_dataloader, tree_shape

from ..core import PRNGKey
from ..utils import to_jax


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

    bs = tree_shape(dataset)

    if isinstance(dataset, SupervisedDataset):
        dataset = SupervisedDatasetWithWeights(
            dataset.inputs, dataset.targets, np.ones((bs,))
        )

    dataset = to_jax(dataset)

    treeDataloader = tree_dataloader(
        key, n_minibatches, axis, reshuffle, tree_transform, do_bootstrapping
    )

    # closures `data`
    def update_fn(
        state: MiniBatchState,
    ) -> Tuple[
        MiniBatchState, Union[SupervisedDatasetWithWeights, UnsupervisedDataset]
    ]:
        return treeDataloader.next(state, dataset)

    return Dataloader(treeDataloader.init(dataset), update_fn)
