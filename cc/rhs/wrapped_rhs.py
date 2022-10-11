from ..types import *
from ..abstract import AbstractWrappedRHS, AbstractRHS, S, X, Y 


WrappedRHS = TypeVar("WrappedRHS")
class WrappedRHS(AbstractWrappedRHS):
    rhs: AbstractRHS
    state: NotAParameter[S]
    input_size: int = eqx.static_field()
    output_size: int = eqx.static_field()

    def __call__(self, x: PyTree) -> Tuple[WrappedRHS, PyTree]:
        print(f"""WARNING: Object {type(self)} is being traced. 
            If this message is display continuously then you probably forgot to compile the model or controller. 
            This can be fixed by calling `*model/controller* = equniox.filter_jit(*model/controller*).
            """)
        
        x = self.preprocess_x(x)

        # unpack `S` from NotAParameter
        state = self.state()
        
        new_state, y = self.rhs(state, x)
        new = self._update_state(new_state)
        return new, self.postprocess_y(y)

    def reset(self) -> WrappedRHS:
        # unpack 
        init_state = self.rhs.init_state()()
        return self._update_state(init_state)

    def _update_state(self, new_state: S) -> WrappedRHS:
        new_state = NotAParameter(new_state)
        return eqx.tree_at(lambda obj: obj.state, self, new_state)

    @staticmethod
    def preprocess_x(x: PyTree) -> X:
        return x 

    @staticmethod
    def postprocess_y(y: Y) -> PyTree:
        return y 

        