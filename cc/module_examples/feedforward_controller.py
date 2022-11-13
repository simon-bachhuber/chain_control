import jax.numpy as jnp

from ..core import make_module_from_function


def make_feedforward_controller(us: jnp.ndarray):
    init_state = jnp.array([0])
    init_params = us

    def forward(params, state, x):
        return state + 1, params[state[0]]

    return make_module_from_function(
        forward, init_params, init_state, name="feedforward-controller"
    )
