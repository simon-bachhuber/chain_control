import jax.random as jrand

from ..abstract import AbstractRHS, S, S_w_key
from ..types import *
from ..utils import batch_concat
from .integrate import integrate
from .parameter import PossibleParameter


class LinearRHS_Params(eqx.Module):
    A: PossibleParameter
    B: PossibleParameter
    C: PossibleParameter
    D: PossibleParameter


class LinearRHS(AbstractRHS):
    params: LinearRHS_Params
    _init_state: PossibleParameter
    method: str = eqx.static_field()

    def __call__(self, x_tm1: S, u_tm1: PyTree) -> Tuple[S, jnp.ndarray]:
        u_tm1 = batch_concat(u_tm1,0)
        # unpack
        A,B,C,D = self.params.A(), self.params.B(), self.params.C(), self.params.D()
        rhs = lambda t,x: A@x + B@u_tm1
        x_t = integrate(rhs, x_tm1, self.method)
        y_t = C@x_tm1 + D@u_tm1
        return x_t, y_t 

    def init_state(self) -> Tuple["LinearRHS", PossibleParameter[S]]:
        return self, self._init_state

    
class NonlinearRHS(AbstractRHS):
    f: eqx.Module
    g: eqx.Module
    _init_state: PossibleParameter[S_w_key]
    method: str = eqx.static_field()
    reset_key: bool = eqx.static_field()

    def __call__(self, x_tm1_w_key: S_w_key, u_tm1: PyTree) -> Tuple[S_w_key, jnp.ndarray]:
        x_tm1, key = x_tm1_w_key
        key = key()

        key, c1, c2 = jrand.split(key, 3)

        rhs = lambda t,x: self.f(batch_concat((x, u_tm1), 0), key=c1)
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

