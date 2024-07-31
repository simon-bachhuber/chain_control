import equinox as eqx
import jax.numpy as jnp

from ..core import AbstractController
from ..utils import batch_concat


# Observation comes from a env wrapped with a reference signal
def preprocess_error_as_controller_input(x) -> jnp.ndarray:
    # capture x, split into ref / obs
    ref, obs = batch_concat(x["ref"], 0), batch_concat(x["obs"], 0)
    # calculate error based on
    err = ref - obs
    return err


def make_pid_controller(
    p_gain: float,
    i_gain: float,
    d_gain: float,
    control_timestep: float,
    p_gain_trainable: bool = True,
    i_gain_trainable: bool = True,
    d_gain_trainable: bool = True,
):
    init_state = {"last_error": jnp.array([0.0]), "sum_of_errors": jnp.array([0.0])}
    init_params = {
        "p_gain": jnp.array([p_gain]),
        "i_gain": jnp.array([i_gain]),
        "d_gain": jnp.array([d_gain]),
    }

    class PIDController(AbstractController):
        state: dict
        params: dict

        def step(self, x):
            last_error = self.state["last_error"]
            current_error = preprocess_error_as_controller_input(x)

            new_state = {
                "last_error": current_error,
                "sum_of_errors": self.state["sum_of_errors"]
                + control_timestep * current_error,
            }

            p_term = current_error
            i_term = new_state["sum_of_errors"]
            d_term = (current_error - last_error) / control_timestep

            control = (
                self.params["p_gain"] * p_term
                + self.params["i_gain"] * i_term
                + self.params["d_gain"] * d_term
            )

            return PIDController(new_state, self.params), control

        def reset(self):
            return PIDController(init_state, self.params)

        def grad_filter_spec(self):
            filter_spec = super().grad_filter_spec()
            return eqx.tree_at(
                lambda ctrb: (ctrb.state, ctrb.params),
                filter_spec,
                (
                    False,
                    {
                        "p_gain": p_gain_trainable,
                        "i_gain": i_gain_trainable,
                        "d_gain": d_gain_trainable,
                    },
                ),
            )

    return PIDController(init_state, init_params)
