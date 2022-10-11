from ..utils import batch_concat
from .integrate import integrate
from ..types import *
from ..abstract import S, AbstractRHS
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

    def init_state(self) -> PossibleParameter[S]:
        return self._init_state

    
class NonlinearRHS(AbstractRHS):
    f: eqx.Module
    g: eqx.Module
    _init_state: PossibleParameter
    method: str = eqx.static_field()

    def __call__(self, x_tm1: S, u_tm1: PyTree) -> Tuple[S, jnp.ndarray]:
        rhs = lambda t,x: self.f(batch_concat((x, u_tm1), 0))
        x_t = integrate(rhs, x_tm1, self.method)
        y_t = self.g(x_t)
        return x_t, y_t 

    def init_state(self) -> PossibleParameter[S]:
        return self._init_state

