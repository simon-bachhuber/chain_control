import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from acme.jax.utils import batch_concat

from .rhs import NotAParameter, PossibleParameter, Parameter


def guarantee_not_parameter(state: PossibleParameter):
    return NotAParameter(state())


def is_param(is_param, data) -> PossibleParameter:
    if is_param:
        return Parameter(data)
    else:
        return NotAParameter(data)


def filter_module(module):
    # set every leaf node to False, set every array to True
    filter = jtu.tree_map(
        eqx.is_array, module, is_leaf=lambda node: isinstance(node, NotAParameter)
    )
    return filter


def flatten_module(module) -> jnp.ndarray:
    params, _ = jtu.tree_flatten(eqx.filter(module, filter_module(module)))
    return batch_concat(params, 0)
