import jax.random as jrand

from ..abstract import AbstractController, AbstractRHS, X
from ..rhs.common_controller_model import (LinearControllerModelOptions,
                                           NonlinearControllerModelOptions,
                                           rhs_state_LinearControllerModel,
                                           rhs_state_NonlinearControllerModel,
                                           set_pre_postprocess)
from ..rhs.wrapped_rhs import WrappedRHS
from ..types import *
from ..utils import batch_concat

LinearControllerOptions = LinearControllerModelOptions
NonlinearControllerOptions = NonlinearControllerModelOptions


class Controller(WrappedRHS, AbstractController):
    pass 


def preprocess_error_as_controller_input(x: PyTree) -> jnp.ndarray:
    # capture x, split into ref / obs 
    ref, obs = batch_concat(x["ref"], 0), batch_concat(x["obs"], 0)
    # calculate error based on
    err = ref - obs 
    return err


class LinearController(Controller):
    def __init__(self, options: LinearControllerOptions):
        rhs, state = rhs_state_LinearControllerModel(options)
        self.rhs = rhs 
        self.state = state 
        self.input_size = options.input_size
        self.output_size = options.output_size
        self.preprocess_x = preprocess_error_as_controller_input
        self.postprocess_y = lambda y: y 


def create_pi_controller(p_gain: float, i_gain: float, delta_t: float) -> LinearControllerOptions:

    integrate_method = "no-integrate"
    # delta_t is there to rescale the error input
    # we integrate it so I_state = I_state_tm1 + delta_t * current_error

    def ABCD_init(*args):
        A = jnp.array([[0.0, 0.0], [0.0, 1.0]])
        B = jnp.array([[1.0], [1.0*delta_t]])
        C = jnp.array([[p_gain, i_gain]])
        D = jnp.array([[0.0]])
        return A,B,C,D 

    state_size = 2 
    input_size = 1 
    output_size = 1 

    c = LinearControllerOptions(state_size, input_size, 
        output_size, integrate_method, jrand.PRNGKey(1,), ABCD_init=ABCD_init,
        A_is_param=False, B_is_param=False, D_is_param=False
        )

    return c


class NonlinearController(Controller):
    def __init__(self, options: NonlinearControllerOptions):
        rhs, state = rhs_state_NonlinearControllerModel(options)
        self.rhs = rhs 
        self.state = state 
        self.input_size = options.input_size
        self.output_size = options.output_size   
        set_pre_postprocess(self, options)


class FeedforwardCounterState(eqx.Module):
    counter: jnp.ndarray


class FeedforwardRHS(AbstractRHS):
    us: jnp.ndarray

    def __call__(self, state: FeedforwardCounterState, x: PyTree
        ) -> Tuple[FeedforwardCounterState, jnp.ndarray]:
        
        # state is only a counter 
        return FeedforwardCounterState(state.counter+1), self.us[state.counter[0]]

    def init_state(self) -> Tuple["FeedforwardRHS", PossibleParameter[FeedforwardCounterState]]:
        # initialise counter
        return self, NotAParameter(FeedforwardCounterState(jnp.array([0])))


class FeedforwardController(Controller):

    def __init__(self, us: TimeSeriesOfAct):    
        self.input_size = 0 
        self.output_size = us.shape[-1] 
        self.rhs = FeedforwardRHS(us)
        init_state = self.rhs.init_state()[1]
        self.state = init_state 
        self.preprocess_x = lambda x: x 
        self.postprocess_y = lambda y: y 

