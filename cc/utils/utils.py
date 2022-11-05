import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


def to_jax(tree):
    return jtu.tree_map(jnp.asarray, tree)


def to_numpy(tree):
    return jtu.tree_map(np.asarray, tree)
