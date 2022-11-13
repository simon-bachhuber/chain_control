from typing import Callable, Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from ..config import print_compile_warn
from .abstract import AbstractModule
from .types import PyTree


class NoReset(eqx.Module):
    module: "Module"

    def __call__(self, x):
        new_module, y = self.module(x)
        return NoReset(new_module), y


def make_module_from_eqx_module(
    parametric_function: eqx.Module, init_state, name=None
) -> "Module":
    assert isinstance(parametric_function, eqx.Module)
    return _init_module(parametric_function, init_state, name=name)


class Module(AbstractModule):
    parametric_function: eqx.Module
    state: PyTree[Union[jnp.ndarray, eqx.Module, "Module"]]
    init_state: PyTree[Union[jnp.ndarray, eqx.Module, "Module"]]
    name: Union[str, None]

    def __call__(
        self, x: Optional[PyTree[jnp.ndarray]] = None
    ) -> Tuple["Module", PyTree[jnp.ndarray]]:
        if print_compile_warn():
            print(f"COMPILING: Module with name={self.name}")
        state, y = self.parametric_function(self.state, x)
        return self._replace(state), y

    def reset(self):
        def _reset(state_leaf, init_state_leaf):
            if isinstance(state_leaf, Module):
                return state_leaf.reset()
            elif isinstance(state_leaf, NoReset):
                return state_leaf
            else:
                return init_state_leaf

        state = jtu.tree_map(
            _reset,
            self.state,
            self.init_state,
            is_leaf=lambda leaf: isinstance(leaf, (Module, NoReset)),
        )
        return self._replace(state)

    def _replace(self, state):
        return Module(self.parametric_function, state, self.init_state, self.name)


def _init_module(parametric_function, init_state, state=None, name=None) -> Module:
    if state is None:
        state = init_state
    return Module(parametric_function, state, replace_module(init_state), name)


def replace_module(tree, replace_fn=lambda module: None):
    """Replace every leaf that is a `Module` in tree `tree` using `replace_fn`.
    Defaults to `None`."""

    def is_module(leaf):
        if isinstance(leaf, Module):
            return replace_fn(leaf)
        else:
            return leaf

    return jtu.tree_map(is_module, tree, is_leaf=lambda leaf: isinstance(leaf, Module))


def make_module_from_function(apply_fn, init_params, init_state, name=None) -> Module:
    class parametric_function(eqx.Module):
        params: dict
        apply_fn: Callable

        def __call__(self, state, x):
            state, y = self.apply_fn(self.params, state, x)
            return state, y

    return _init_module(
        parametric_function(init_params, apply_fn), init_state, name=name
    )
