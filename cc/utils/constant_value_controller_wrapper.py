from typing import Optional, Sequence

import jax
import jax.numpy as jnp

from cc.core import AbstractController


def constant_value_controller_wrapper(
    controller: AbstractController,
    control_timestep: float,
    T: float,
    action_shape: Sequence[int],
    method: str = "last_action",
    decay: Optional[float] = None,
) -> AbstractController:
    """Wraps a controller such that after time `T` it returns a controlled value.

    Args:
        controller (AbstractController): Controller to wrap.
        control_timestep (float): Time delta between controller steps.
        T (float): Time after which the wrapped controller action is overwritten.
        action_shape (Sequence[int]): Shape of the returned action of the controller.
        method (str, optional): Whether to return zeros or hold the last action value.
            Defaults to "last_action".
        decay (Optional[float], optional): Decay factor with which the last value
            gets multiplied. Should be in the interval (0, 1). Defaults to None.

    Returns:
        AbstractController: Wrapped controller.
    """
    assert method in ["last_action", "zeros"]
    if decay is not None:
        if method == "zeros":
            raise Exception("You can not decay a value of zero.")
        assert 0 < decay < 1

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
                lambda action: _zero_action
                if method == "zeros"
                else (self.last_action * decay if decay else self.last_action),
                lambda action: action,
                action,
            )
            return ConstantValueControllerWrapper(action, self.count + 1, ctrb), action

        def reset(self):
            return ConstantValueControllerWrapper(_zero_action, 0, controller.reset())

        def grad_filter_spec(self):
            return (False, False, controller.grad_filter_spec())

    return ConstantValueControllerWrapper(_zero_action, 0, controller.reset())
