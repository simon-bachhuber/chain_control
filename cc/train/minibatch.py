import functools as ft

import jax
import jax.random as jrand

from ..types import *
from ..utils import tree_index, tree_shape


@ft.partial(jax.jit, static_argnums=(1,2))
def gen_minibatch_indices(key, batch_size: int, minibatch_size: int) -> jnp.ndarray:
    permutation = jax.random.permutation(key, jnp.arange(batch_size))

    def scan_fn(carry, _):
        start_idx = carry
        y = jnp.take(
                jnp.arange(batch_size), 
                jax.lax.dynamic_slice_in_dim(permutation, start_idx, minibatch_size)
            )
        carry = start_idx + minibatch_size  
        return carry, y

    return jax.lax.scan(scan_fn, 0, length=batch_size//minibatch_size, xs=None)[1]


@ft.partial(jax.jit, static_argnums=(1,2))
def gen_minibatch_masks(key, batch_size: int, minibatch_size: int) -> jnp.ndarray:
    
    idxss = gen_minibatch_indices(key, batch_size, minibatch_size)
    
    # generate masks from idxs
    def to_mask(idxs):
        return jnp.in1d(jnp.arange(batch_size), idxs)

    return jax.vmap(to_mask)(idxss) 


Tree = TypeVar("Tree")


class MiniBatchState(NamedTuple):
    indices: jnp.ndarray
    i: int 
    bs: int 
    n_minibatches: int 
    minibatch_size: int 
    key: PRNGKey


class MiniBatch(NamedTuple):
    init: Callable[[Tree], MiniBatchState]
    update: Callable[[MiniBatchState, Tree], Tuple[MiniBatchState, Tree]]


def minibatch(
    key: PRNGKey,
    n_minibatches: int = 1, 
    axis: int = 0,
    reshuffle: bool = True,
    tree_transform: Optional[Callable] = None 
    ):

    def init(tree: Tree):
        bs = tree_shape(tree, axis)
        assert bs % n_minibatches == 0
        minibatch_size = bs // n_minibatches
        inner_key, consume = jrand.split(key)

        return MiniBatchState(
            gen_minibatch_indices(consume, bs, minibatch_size),
            0,
            bs,
            n_minibatches,
            minibatch_size,
            inner_key 
        )

    def update(state: MiniBatchState, tree: Tree) -> Tuple[MiniBatchState, Tree]:

        indices = state.indices
        key = state.key 
        if state.i >= state.n_minibatches:
            # iteration over one epoch is done
            if reshuffle:
                key, consume = jrand.split(key)
                indices = gen_minibatch_indices(consume, state.bs, state.minibatch_size)

        # reset counter if required
        i = state.i % state.n_minibatches
        
        batch_of_tree = tree_index(tree, indices[i], axis)

        if tree_transform:
            key, consume = jrand.split(key)
            batch_of_tree = tree_transform(consume, batch_of_tree, state.minibatch_size)

        return MiniBatchState(
            indices, i+1, state.bs, state.n_minibatches, state.minibatch_size, key
        ), batch_of_tree

    return MiniBatch(init, update)


class NoOpMiniBatchState(NamedTuple):
    n_minibatches: int 


def no_op_minibatch(n_minibatches):
    def init(tree):
        return NoOpMiniBatchState(n_minibatches)
    def update(state, tree):
        return state, tree
    return MiniBatch(init, update)


#minibatch = no_op_minibatch