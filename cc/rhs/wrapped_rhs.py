from typing import Callable, Tuple, TypeVar, Union

import equinox as eqx

from ..abstract import AbstractRHS, AbstractWrappedRHS, S, S_w_key, X, Y
from ..config import print_compile_warn
from ..types import NotAParameter, PyTree

WrappedRHS = TypeVar("WrappedRHS")


class WrappedRHS(AbstractWrappedRHS):
    rhs: AbstractRHS
    state: NotAParameter[Union[S, S_w_key]]
    input_size: int = eqx.static_field()
    output_size: int = eqx.static_field()
    preprocess_x: Callable[[PyTree], X] = eqx.static_field()
    postprocess_y: Callable[[Y], PyTree] = eqx.static_field()

    def __call__(self, x: PyTree) -> Tuple[WrappedRHS, PyTree]:
        if print_compile_warn():
            print(
                f"""WARNING: Object {type(self)} is being traced.
                If this message is display continuously then you probably forgot \
                    to compile the model or controller.
                This can be fixed by calling `*model/controller* = equniox.filter_jit(\
                    *model/controller*).
                """
            )

        x = self.preprocess_x(x)

        # unpack `S` from NotAParameter
        state = self.state()

        new_state, y = self.rhs(state, x)
        new = self._update_state(new_state)
        return new, self.postprocess_y(y)

    def reset(self) -> WrappedRHS:
        # unpack
        new_rhs, init_state = self.rhs.init_state()
        self = self._update_state(init_state())
        return eqx.tree_at(lambda obj: obj.rhs, self, new_rhs)

    def _update_state(self, new_state: Union[S, S_w_key]) -> WrappedRHS:
        new_state = NotAParameter(new_state)
        return eqx.tree_at(lambda obj: obj.state, self, new_state)
