import jax.numpy as jnp

from ..nn_lib import make_module_from_function
from ..core.types import Observation
from ..utils import batch_concat


# Observation comes from a env wrapped with a reference signal
def preprocess_error_as_controller_input(x: Observation) -> jnp.ndarray:
    # capture x, split into ref / obs
    ref, obs = batch_concat(x["ref"], 0), batch_concat(x["obs"], 0)
    # calculate error based on
    err = ref - obs
    return err


def make_pid_controller(
    p_gain: float, i_gain: float, d_gain: float, control_timestep: float
):

    init_state = {"last_error": jnp.array(0.0), "sum_of_errors": jnp.array(0.0)}
    init_params = {
        "p_gain": jnp.array(p_gain),
        "i_gain": jnp.array(i_gain),
        "d_gain": jnp.array(d_gain),
    }

    def forward(params, state, x):
        last_error = state["last_error"]
        current_error = preprocess_error_as_controller_input(x)

        new_state = {
            "last_error": current_error,
            "sum_of_errors": state["sum_of_errors"] + control_timestep * current_error,
        }

        p_term = current_error
        i_term = new_state["sum_of_errors"]
        d_term = (current_error - last_error) / control_timestep

        control = (
            params["p_gain"] * p_term
            + params["i_gain"] * i_term
            + params["d_gain"] * d_term
        )

        return new_state, control

    return make_module_from_function(
        forward, init_params, init_state, name="pid-controller"
    )
