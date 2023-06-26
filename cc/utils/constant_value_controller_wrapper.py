from typing import Sequence

import jax
import jax.numpy as jnp

from cc.core import AbstractController


def constant_value_controller_wrapper(
    controller: AbstractController,
    control_timestep: float,
    T: float,
    action_shape: Sequence[int],
    method: str = "last_action",
) -> AbstractController:
    assert method in ["last_action", "zeros"]

    max_count = int(T / control_timestep)
    _zero_action = jnp.zeros(action_shape)

    class ConstantValueControllerWrapper(AbstractController):
        last_action: jax.Array
        count: int
        wrapped_controller: AbstractController

        def step(self, x):
            ctrb, action = self.wrapped_controller.step(x)
            action = jax.lax.cond(
                self.count > max_count,
                lambda action: _zero_action if method == "zeros" else self.last_action,
                lambda action: action,
                action,
            )
            return ConstantValueControllerWrapper(action, self.count + 1, ctrb), action

        def reset(self):
            return ConstantValueControllerWrapper(_zero_action, 0, controller.reset())

        def grad_filter_spec(self):
            return (False, False, controller.grad_filter_spec())

    return ConstantValueControllerWrapper(_zero_action, 0, controller.reset())
