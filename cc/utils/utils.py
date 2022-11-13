import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.flatten_util import ravel_pytree

from ..env.sample_from_spec import sample_from_tree_of_specs


def to_jax(tree):
    return jtu.tree_map(jnp.asarray, tree)


def to_numpy(tree):
    return jtu.tree_map(np.asarray, tree)


def make_postprocess_fn(output_specs=None, toy_output=None):
    if output_specs is not None and toy_output is not None:
        raise Exception("Please specifiy either one or the other function argument.")
    if output_specs:
        return ravel_pytree(sample_from_tree_of_specs(output_specs))[1]
    else:
        return ravel_pytree(toy_output)[1]


def l2_norm(vector):
    assert vector.ndim == 1
    return jnp.sqrt(jnp.sum(vector**2))


def mae(y, yhat):
    return jnp.mean(jnp.abs(y - yhat))


def mse(y, yhat):
    return jnp.mean((y - yhat) ** 2)


def rmse(y, yhat):
    return jnp.sqrt(mse(y, yhat))
