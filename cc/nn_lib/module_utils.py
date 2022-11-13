import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ..core import Module, PyTree, replace_module
from ..utils import batch_concat


class NoGrad(eqx.Module):
    module: Module

    def __call__(self, x):
        new_module, y = self.module(x)
        return NoGrad(new_module), y


def filter_module(tree, include_init_state: bool = False) -> PyTree[bool]:
    """Make a mask of tree where arrays are `True` if they are part of
    `parametric_function` field but not part of `state` field.
    Useful for `eqx.filter_grad(arg=...)`
    """

    def handle_leaf(leaf):

        if isinstance(leaf, Module):
            if include_init_state:
                # modules carry their own initial state
                # this will be captured via `filter_module(leaf.state)`
                # hence replace this state here by `None`
                init_state = replace_module(leaf.init_state)
                init_state = eqx.is_array(init_state)
            else:
                init_state = False
            return Module(
                jtu.tree_map(eqx.is_array, leaf.parametric_function),
                filter_module(leaf.state, include_init_state),
                init_state,
                False,
            )
        else:
            return False

    def is_leaf(leaf):
        return isinstance(leaf, (Module, NoGrad))

    return jtu.tree_map(handle_leaf, tree, is_leaf=is_leaf)


def filter_scan_module(scan_fn, init, xs, length, filter_spec=eqx.is_array):
    init_arrays, init_funcs = eqx.partition(init, filter_spec=filter_spec)

    def _scan_fn(arrays, x):
        carry = eqx.combine(arrays, init_funcs)
        carry, y = scan_fn(carry, x)
        arrays, _ = eqx.partition(carry, filter_spec=filter_spec)
        return arrays, y

    return jax.lax.scan(_scan_fn, init=init_arrays, xs=xs, length=length)


def find_module(tree, name):
    def leaf_fn(leaf):
        if isinstance(leaf, Module):
            match = find_module(leaf.state, name)
            if match is not None:
                if leaf.name is not None:
                    assert leaf.name != match, "Ambiguous naming of modules."
                return match
            elif leaf.name == name:
                return leaf
            else:
                return None
        return None

    tree_of_flat_list_of_modules = jtu.tree_map(
        leaf_fn, tree, is_leaf=lambda leaf: isinstance(leaf, Module)
    )

    list_of_modules = jtu.tree_flatten(
        tree_of_flat_list_of_modules, is_leaf=lambda leaf: isinstance(leaf, Module)
    )[0]

    matched_module = None
    for module in list_of_modules:
        if module is None:
            continue

        if module.name == name:
            if matched_module is not None:
                raise Exception("Ambiguous naming of modules.")
            matched_module = module
    return matched_module


def _replace_module_name_with_none(module):
    return Module(
        module.parametric_function,
        replace_module(module.state, _replace_module_name_with_none),
        replace_module(module.init_state, _replace_module_name_with_none),
        None,
    )


def flatten_module(module, include_init_state=False) -> jnp.ndarray:
    params, _ = jtu.tree_flatten(
        eqx.filter(module, filter_module(module, include_init_state))
    )
    return batch_concat(params, 0)
