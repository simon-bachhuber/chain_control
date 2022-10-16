from dataclasses import dataclass

import jax
import jax.random as jrand
from scipy.io import loadmat

from ..types import *
from ..utils import to_jax
from .initializers import ABCD_init, f_g_init, state_init
from .parameter import guarantee_not_parameter, is_param
from .rhs import LinearRHS, LinearRHS_Params, NonlinearRHS


@dataclass
class _ControllerModelOptions:
    state_size: int 
    input_size: int 
    output_size: int 
    integrate_method: str 
    key: PRNGKey
    state_init: FunctionType = state_init
    init_state_is_param: bool = False 


@dataclass
class LinearControllerModelOptions(_ControllerModelOptions):
    ABCD_init: FunctionType = ABCD_init
    A_is_param: bool = True
    B_is_param: bool = True
    C_is_param: bool = True
    D_is_param: bool = False 


del state_init, ABCD_init


unity = lambda x:x
@dataclass
class NonlinearControllerModelOptions(_ControllerModelOptions):
    depth_f: int = 1
    width_f: int = 10
    act_fn_f: FunctionType = jax.nn.relu
    act_final_f: FunctionType = unity
    use_bias_f: bool = True
    use_dropout_f: bool = False 
    dropout_rate_f: float = 0.5
    depth_g: int = 0
    width_g: int = 0
    act_fn_g: FunctionType = jax.nn.relu
    act_final_g: FunctionType = unity
    use_bias_g: bool = True
    use_dropout_g: bool = False 
    dropout_rate_g: float = 0.5
    reset_key: bool = False 


def rhs_state_LinearControllerModel(c: LinearControllerModelOptions):
    key, c1, c2 = jrand.split(c.key, 3)
    A,B,C,D = c.ABCD_init(c1, c.state_size, c.input_size, c.output_size)

    init_params = LinearRHS_Params(
        is_param(c.A_is_param, A), 
        is_param(c.B_is_param, B),
        is_param(c.C_is_param, C),
        is_param(c.D_is_param, D)
    )

    init_state = is_param(c.init_state_is_param, c.state_init(c2, c.state_size))

    rhs = LinearRHS(init_params, init_state, c.integrate_method)

    return rhs, guarantee_not_parameter(init_state)


def rhs_state_NonlinearControllerModel(c: NonlinearControllerModelOptions):
    key, c1, c2 = jrand.split(c.key, 3)
    f,g = f_g_init(c1, c)

    s = c.state_init(c2, c.state_size)
    init_state = is_param(c.init_state_is_param, (s, NotAParameter(key)))

    rhs = NonlinearRHS(f, g, init_state, c.integrate_method, c.reset_key)
    return rhs, guarantee_not_parameter(init_state)


def LinearControllerModelOptions_FromMatlab(path_to_mat: str):
    mat = loadmat(path_to_mat)

    def ABCD_init(*args):
        return to_jax((mat[key] for key in ["A", "B", "C", "D"]))

    A,B,C,D = ABCD_init()

    input_size = B.shape[-1]
    output_size = C.shape[0]
    state_size = A.shape[0]

    c = LinearControllerModelOptions(state_size, input_size, 
        output_size, "no-integrate", jrand.PRNGKey(1,), ABCD_init=ABCD_init)

    return c

