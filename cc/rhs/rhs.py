from typing import Callable, Generic, Tuple, TypeVar, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrand

from ..core import PRNGKey, PyTree
from ..utils import batch_concat
from .integrate import integrate

X = TypeVar("X")
Y = TypeVar("Y")
T = TypeVar("T")
S = TypeVar("S")


class NotAParameter(eqx.Module, Generic[T]):
    _: T

    def __call__(self) -> T:
        return self._


class Parameter(eqx.Module, Generic[T]):
    _: T

    def __call__(self) -> T:
        return self._


S_w_key = Tuple[S, NotAParameter[PRNGKey]]


PossibleParameter = Union[Parameter[T], NotAParameter[T]]


class LinearRHS_Params(eqx.Module):
    A: PossibleParameter  # pytype: disable=invalid-annotation
    B: PossibleParameter  # pytype: disable=invalid-annotation
    C: PossibleParameter  # pytype: disable=invalid-annotation
    D: PossibleParameter  # pytype: disable=invalid-annotation


class LinearRHS(eqx.Module):
    params: LinearRHS_Params
    _init_state: PossibleParameter  # pytype: disable=invalid-annotation
    method: str = eqx.static_field()

    def __call__(self, x_tm1: S, u_tm1: PyTree) -> Tuple[S, jnp.ndarray]:
        u_tm1 = batch_concat(u_tm1, 0)
        # unpack
        A, B, C, D = self.params.A(), self.params.B(), self.params.C(), self.params.D()
        rhs = lambda t, x: A @ x + B @ u_tm1
        x_t = integrate(rhs, x_tm1, self.method)
        y_t = C @ x_tm1 + D @ u_tm1
        return x_t, y_t

    def init_state(self) -> Tuple["LinearRHS", PossibleParameter[S]]:
        return self, self._init_state


class NonlinearRHS(eqx.Module):
    f: eqx.Module
    g: eqx.Module
    input_act_fn: Callable
    _init_state: PossibleParameter[S_w_key]
    method: str = eqx.static_field()
    reset_key: bool = eqx.static_field()

    def __call__(
        self, x_tm1_w_key: S_w_key, u_tm1: PyTree
    ) -> Tuple[S_w_key, jnp.ndarray]:
        x_tm1, key = x_tm1_w_key
        key = key()

        key, c1, c2 = jrand.split(key, 3)

        u_tm1 = self.input_act_fn(u_tm1)
        rhs = lambda t, x: self.f(batch_concat((x, u_tm1), 0), key=c1)
        x_t = integrate(rhs, x_tm1, self.method)
        y_t = self.g(x_t, key=c2)

        return (x_t, NotAParameter(key)), y_t

    def init_state(self) -> Tuple["NonlinearRHS", PossibleParameter[S_w_key]]:

        s, key = self._init_state()
        key, consume = jrand.split(key())

        type_init_state = type(self._init_state)
        init_state = lambda key: type_init_state((s, NotAParameter(key)))

        if self.reset_key:
            new = eqx.tree_at(lambda obj: obj._init_state, self, init_state(key))
        else:
            new = self

        return new, init_state(consume)
