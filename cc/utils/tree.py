from functools import partial, reduce
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from acme.jax.utils import add_batch_dim, batch_concat, ones_like, zeros_like
from equinox import tree_equal

from ..core.types import PyTree

tree_zeros_like = zeros_like
tree_ones_like = ones_like


def tree_bools_like(tree, where=None, invert=False) -> "PyTree[bool]":
    t, f = (True, False) if not invert else (False, True)
    default_tree = jax.tree_util.tree_map(lambda _: t, tree)
    if where:
        return eqx.tree_at(where, default_tree, f)
    else:
        return default_tree


def tree_insert_IMPURE(tree, subtree, batch_idxs: tuple[int, ...]):
    def insert(a1, a2):
        a1[batch_idxs] = a2
        return a1

    jax.tree_util.tree_map(insert, tree, subtree)


def tree_concat(trees: list, along_existing_first_axis=False, backend="numpy"):
    if backend == "jax":
        concat = jnp.concatenate
    else:
        concat = np.concatenate

    if len(trees) == 0:
        return trees
    if len(trees) == 1:
        if along_existing_first_axis:
            return trees[0]
        else:
            return jax.tree_util.tree_map(lambda arr: arr[None], trees[0])

    if along_existing_first_axis:
        sl = (slice(None),)
    else:
        sl = (
            None,
            slice(None),
        )

    initial = jax.tree_map(
        lambda a1, a2: concat((a1[sl], a2[sl]), axis=0), trees[0], trees[1]
    )
    stack = reduce(
        lambda tree1, tree2: jax.tree_map(
            lambda a1, a2: concat((a1, a2[sl]), axis=0), tree1, tree2
        ),
        trees[2:],
        initial,
    )
    return stack


def tree_shape(tree, axis: int = 0):
    return jtu.tree_flatten(tree)[0][0].shape[axis]


@partial(jax.jit, static_argnums=(2, 3, 4))
def tree_slice(tree, start, slice_size=1, axis=0, keepdim=False):
    def slicing_fun(arr):

        if slice_size > 1:
            return jax.lax.dynamic_slice_in_dim(
                arr, start_index=start, slice_size=slice_size, axis=axis
            )
        else:
            return jax.lax.dynamic_index_in_dim(
                arr, index=start, axis=axis, keepdims=keepdim
            )

    return jax.tree_util.tree_map(slicing_fun, tree)


@partial(jax.jit, static_argnums=(2,))
def tree_indices(tree, indices: jnp.ndarray, axis=0):
    """Extract an array of indices in an axis for every tree-element

    Args:
        tree (_type_): Tree of Arrays
        indices (jnp.ndarray): Array of Integers
        axis (int, optional): _description_. Defaults to 0.
    """

    def extract_indices_of_axis(arr):
        return jax.vmap(
            lambda index: jax.lax.dynamic_index_in_dim(
                arr, index, axis, keepdims=False
            ),
            out_axes=axis,
        )(indices)

    return jtu.tree_map(extract_indices_of_axis, tree)
