import copy
from functools import partial, reduce

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from acme.jax.utils import add_batch_dim, batch_concat, zeros_like, ones_like


def to_jax(tree):
    return jax.tree_util.tree_map(jnp.asarray, tree)


def to_numpy(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


def to_cpu(tree):
    device = jax.devices("cpu")[0]
    return jax.tree_util.tree_map(
        lambda arr: jax.device_put(arr, device=device), tree
    )


def swap_function_output(fun):
    def _fun(*args, **kwargs):
        out = fun(*args, **kwargs)
        assert len(out)==2
        return out[1], out[0]
    return _fun 


def tree_insert_IMPURE(tree, subtree, batch_idxs: tuple[int]):

    def insert(a1, a2):
        a1[batch_idxs] = a2
        return a1 

    jax.tree_util.tree_map(insert, tree, subtree)
    

def tree_concat(trees: list, along_existing_first_axis=False, backend="numpy"):
    if backend == "jax":
        concat = jnp.concatenate
    else:
        concat = np.concatenate

    if len(trees) ==0:
        return trees 
    if len(trees) ==1:
        if along_existing_first_axis:
            return trees[0]
        else:
            return jax.tree_util.tree_map(lambda arr: arr[None], trees[0])

    if along_existing_first_axis:
        sl = (slice(None),)
    else:
        sl = (None, slice(None),)
    
    initial = jax.tree_map(lambda a1, a2: concat((a1[sl], a2[sl]), axis=0), trees[0], trees[1])
    stack = reduce(
        lambda tree1, tree2: jax.tree_map(lambda a1, a2: concat((a1, a2[sl]), axis=0), tree1, tree2),
        trees[2:], initial
    )
    return stack


def tree_shape(tree, axis: int = 0):
    return jtu.tree_flatten(tree)[0][0].shape[axis]


@partial(jax.jit, static_argnums=(2,3,4))
def idx_in_pytree(tree, start, slice_size=1, axis=0, keepdim=False):

    def slicing_fun(arr):

        if slice_size>1:
            return jax.lax.dynamic_slice_in_dim(arr, \
                start_index=start, slice_size=slice_size, axis=axis)
        else:
            return jax.lax.dynamic_index_in_dim(arr, \
                index=start, axis=axis, keepdims=keepdim)

    return jax.tree_util.tree_map(slicing_fun, tree)


@partial(jax.jit, static_argnums=(2,))
def tree_index(tree, indices: jnp.ndarray, axis=0):
    """Extract an array of indices in an axis for every tree-element

    Args:
        tree (_type_): Tree of Arrays
        indices (jnp.ndarray): Array of Integers
        axis (int, optional): _description_. Defaults to 0.
    """

    def extract_indices_of_axis(arr):
        return jax.vmap(
            lambda index: jax.lax.dynamic_index_in_dim(arr, index, axis, keepdims=False), 
            out_axes=axis)(indices)

    return jtu.tree_map(extract_indices_of_axis, tree)


def generate_ts(time_limit, control_timestep):
    """Generate action sampling times

    Args:
        time_limit (float): Upper bound of time. Not included
        control_timestep (float): Sample rate in seconds

    Returns:
        _type_: Array of times
    """
    return jax.numpy.arange(0,time_limit,step=control_timestep)


def extract_timelimit_timestep_from_env(env):
    time_limit = env.time_limit
    control_timestep = env.control_timestep
    ts = generate_ts(time_limit, control_timestep)
    return time_limit, control_timestep, ts 

