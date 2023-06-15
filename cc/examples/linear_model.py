import equinox as eqx
import jax
import jax.numpy as jnp

from ..core import AbstractModel, PyTree
from ..utils import (
    ArraySpecs,
    batch_concat,
    make_postprocess_fn,
    sample_from_tree_of_specs,
)


def make_linear_model(
    input_specs: ArraySpecs,
    output_specs: ArraySpecs,
    control_timestep: float,
    state_dim: int,
    continuous_time: bool = False,
    include_D: bool = True,
    seed: int = 1,
    randomize_x0: bool = False,
):
    toy_input = sample_from_tree_of_specs(input_specs)
    toy_output = sample_from_tree_of_specs(output_specs)
    input_dim = batch_concat(toy_input, 0).size
    output_dim = batch_concat(toy_output, 0).size

    postprocess_fn = make_postprocess_fn(toy_output=toy_output)

    # A0 = jnp.zeros((state_dim, state_dim))
    key, consume = jax.random.split(jax.random.PRNGKey(seed))
    A0 = jax.random.normal(consume, (state_dim, state_dim))
    # B0 = jnp.zeros((state_dim, input_dim))
    key, consume = jax.random.split(key)
    B0 = jax.random.normal(consume, (state_dim, input_dim))
    # C0 = jnp.zeros((output_dim, state_dim))
    key, consume = jax.random.split(key)
    C0 = jax.random.normal(consume, (output_dim, state_dim))
    # D0 = jnp.zeros((output_dim, input_dim))
    key, consume = jax.random.split(key)
    D0 = jax.random.normal(consume, (output_dim, input_dim))
    x0 = jnp.zeros((state_dim,))
    if randomize_x0:
        key, consume = jax.random.split(key)
        x0 = jax.random.normal(consume, (state_dim,))

    class LinearModel(AbstractModel):
        A: jax.Array
        B: jax.Array
        C: jax.Array
        D: jax.Array
        x: jax.Array
        x0: jax.Array

        def reset(self):
            return LinearModel(self.A, self.B, self.C, self.D, self.x0, self.x0)

        def step(self, u):
            if continuous_time:
                x_next = self.x + (self.A @ self.x + self.B @ u) * control_timestep
            else:
                x_next = self.A @ self.x + self.B @ u

            y = self.C @ self.x + self.D @ u
            return LinearModel(
                self.A, self.B, self.C, self.D, x_next, self.x0
            ), postprocess_fn(y)

        def grad_filter_spec(self) -> PyTree[bool]:
            filter_spec = super().grad_filter_spec()
            filter_spec = eqx.tree_at(
                lambda model: (model.x, model.x0, model.D),
                filter_spec,
                (False, False, include_D),  # both `x` and `x0` are not optimized
            )
            return filter_spec

        def y0(self) -> PyTree[jax.Array]:
            return postprocess_fn(self.C @ self.x0)

    return LinearModel(A0, B0, C0, D0, x0, x0)
