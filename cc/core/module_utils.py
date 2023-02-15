import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from tree_utils import batch_concat


def filter_scan_module(scan_fn, init, xs, length, filter_spec=eqx.is_array):
    init_arrays, init_funcs = eqx.partition(init, filter_spec=filter_spec)

    def _scan_fn(arrays, x):
        carry = eqx.combine(arrays, init_funcs)
        carry, y = scan_fn(carry, x)
        arrays, _ = eqx.partition(carry, filter_spec=filter_spec)
        return arrays, y

    return jax.lax.scan(_scan_fn, init=init_arrays, xs=xs, length=length)


def flatten_module(model_or_controller) -> jnp.ndarray:
    params, _ = jtu.tree_flatten(
        eqx.filter(model_or_controller, model_or_controller.grad_filter_spec())
    )
    return batch_concat(params, 0)


def number_of_params(model_or_controller) -> int:
    return len(flatten_module(model_or_controller))
