import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand

from ..core import AbstractController
from ..core import PyTree
from ..utils import ArraySpecs
from ..utils import batch_concat
from ..utils import make_postprocess_fn
from ..utils import sample_from_tree_of_specs
from .nn_lib.integrate import integrate
from .pid_controller import make_pid_controller


def mlp_network(
    in_size, out_size, width, depth, act_fn, act_fn_final, key
) -> eqx.Module:
    layers = []
    sizes = [in_size] + depth * [width] + [out_size]

    for i, (s_in, s_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, consume = jrand.split(key)
        layers.append(eqx.nn.Linear(s_in, s_out, key=consume))

        if i < len(sizes) - 2:
            layers.append(eqx.nn.Lambda(act_fn))

    layers.append(eqx.nn.Lambda(act_fn_final))

    return eqx.nn.Sequential(layers)


def make_neural_ode_controller(
    input_specs: ArraySpecs,
    output_specs: ArraySpecs,
    control_timestep: float,
    state_dim: int,
    key=jrand.PRNGKey(
        1,
    ),
    f_integrate_method: str = "RK4",
    f_width_size: int = 10,
    f_depth: int = 2,
    f_activation=jax.nn.relu,
    f_final_activation=lambda x: x,
    g_width_size: int = 10,
    g_depth: int = 0,
    g_activation=jax.nn.relu,
    g_final_activation=lambda x: x,
    superposition: bool = False,
    superposition_trainable: bool = False,
):
    toy_input = sample_from_tree_of_specs(input_specs)
    toy_output = sample_from_tree_of_specs(output_specs)
    input_dim = batch_concat(toy_input, 0).size
    output_dim = batch_concat(toy_output, 0).size

    f_input_dim = state_dim + input_dim
    f_output_dim = state_dim
    g_input_dim = state_dim
    g_output_dim = output_dim

    key, f_key, g_key = jrand.split(key, 3)
    init_state = jnp.zeros((state_dim,))
    alpha0 = jnp.array(0.0)
    postprocess_fn = make_postprocess_fn(toy_output=toy_output)

    f_init = mlp_network(
        f_input_dim,
        f_output_dim,
        f_width_size,
        f_depth,
        f_activation,
        f_final_activation,
        f_key,
    )
    g_init = mlp_network(
        g_input_dim,
        g_output_dim,
        g_width_size,
        g_depth,
        g_activation,
        g_final_activation,
        g_key,
    )

    class NeuralOdeController(AbstractController):
        f: eqx.Module
        g: eqx.Module
        alpha: jnp.ndarray
        secondary: AbstractController
        state: PyTree[jnp.ndarray]
        init_state: PyTree[jnp.ndarray]

        def reset(self):
            return NeuralOdeController(
                self.f,
                self.g,
                self.alpha,
                self.secondary.reset(),
                self.init_state,
                self.init_state,
            )

        def step(self, u):  # u has shape identical to `toy_input`
            # f,g is time-invariant, so integration absolute time is just
            # fixed to a constant value
            t = jnp.array(0.0)
            rhs = lambda t, x: self.f(batch_concat((x, u), 0))
            x = self.state
            x_next = integrate(rhs, x, t, control_timestep, f_integrate_method)
            y_next1 = self.g(batch_concat((x,), 0))
            next_secondary, y_next2 = self.secondary.step(u)
            alpha = jax.nn.sigmoid(self.alpha)
            y_next = postprocess_fn(alpha * y_next1 + (1 - alpha) * y_next2)
            state_next = x_next

            return (
                NeuralOdeController(
                    self.f,
                    self.g,
                    self.alpha,
                    next_secondary,
                    state_next,
                    self.init_state,
                ),
                y_next,
            )  # y_next has shape identical to `toy_output`

        def grad_filter_spec(self) -> PyTree[bool]:
            filter_spec = super().grad_filter_spec()
            filter_spec = eqx.tree_at(
                lambda model: (
                    model.alpha,
                    model.secondary,
                    model.state,
                    model.init_state,
                ),
                filter_spec,
                (
                    superposition_trainable,
                    self.secondary.grad_filter_spec(),
                    False,
                    False,
                ),  # both `state` and `init_state` are not optimized
            )
            return filter_spec

    if superposition:
        gains = jax.random.uniform(key, (3,), maxval=1.0)
        secondary = make_pid_controller(
            float(gains[0]),
            float(gains[1]),
            float(gains[2]),
            control_timestep,
            transform_params=lambda p: {
                "p_gain": p["p_gain"],
                "i_gain": p["i_gain"],
                "d_gain": p["p_gain"] * 0.1,
            },
        )
    else:
        secondary = DummyController(g_output_dim)

    return NeuralOdeController(
        f_init, g_init, alpha0, secondary, init_state, init_state
    )


class DummyController(AbstractController):
    out_dim: int

    def reset(self) -> AbstractController:
        return self

    def step(self, u):
        return self, jnp.zeros((self.out_dim))

    def grad_filter_spec(self) -> PyTree[bool]:
        return False
