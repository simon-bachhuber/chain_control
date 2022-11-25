import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.flatten_util import ravel_pytree

from .sample_from_spec import sample_from_tree_of_specs
from .tree import batch_concat


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


def l1_norm(vector):
    assert vector.ndim == 1
    return jnp.sum(jnp.abs(vector))


def mae(y, yhat):
    y, yhat = batch_concat(y, 0), batch_concat(yhat, 0)
    return jnp.mean(jnp.abs(y - yhat))


def mse(y, yhat):
    y, yhat = batch_concat(y, 0), batch_concat(yhat, 0)
    return jnp.mean((y - yhat) ** 2)


def weighted_mse(y, yhat, weights):
    y, yhat = batch_concat(y, 0), batch_concat(yhat, 0)
    se = (y - yhat) ** 2
    # moves batchaxis to the right; multiply and sum over it
    sse = jnp.sum(weights * jnp.moveaxis(se, 0, -1), axis=-1)
    return jnp.mean(sse)


def rmse(y, yhat):
    return jnp.sqrt(mse(y, yhat))


def primes(n: int) -> list[int]:
    """Find factorization of integer. Slow implementation."""
    primfac = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        primfac.append(n)
    return primfac
