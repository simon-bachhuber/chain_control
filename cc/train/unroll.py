from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from ..core import PRNGKey, PyTree
from ..utils import add_batch_dim, tree_concat


def initialize_closed_loop(model, controller, delay, u_transform):
    delayed_us = jnp.zeros((delay + 1, controller.output_size))
    ring_array_idx = jnp.array(0)
    return ClosedLoop(
        model,
        controller,
        u_transform,
        jnp.array([0]),
        delayed_us,
        ring_array_idx,
        delay,
    )


class ClosedLoop(eqx.Module):
    model: Any
    controller: Any
    u_transform: eqx.Module
    timestep: jnp.ndarray
    delayed_us: jnp.ndarray
    ring_array_idx: jnp.ndarray
    delay: int = eqx.static_field()

    def __call__(self, x: PyTree):
        assert self.delay == 0

        # x here is (ref, obs)
        controller, u = self.controller(x)

        # store u in ring-array
        # delayed_us = jax.lax.dynamic_update_index_in_dim(self.delayed_us, u, \
        # self.ring_array_idx, axis=0)
        delayed_us = self.delayed_us

        # increase ring array idx
        # ring_array_idx = (self.ring_array_idx + 1) % self.delay
        ring_array_idx = self.ring_array_idx

        # get u of index one to the right
        # u_apply = jax.lax.dynamic_index_in_dim(delayed_us, ring_array_idx)
        u_apply = self.u_transform(u, self.timestep)
        model, y = self.model(u_apply)

        return (
            ClosedLoop(
                model,
                controller,
                self.u_transform,
                self.timestep + 1,
                delayed_us,
                ring_array_idx,
                self.delay,
            ),
            y,
        )

    def reset(self):
        controller, model = self.controller.reset(), self.model.reset()
        return initialize_closed_loop(model, controller, self.delay, self.u_transform)


def _unroll_static(f, xs, merge_x_and_y, y0):
    f = f.reset()

    ys = [y0]
    y = y0
    for x in xs:
        f, y = f(merge_x_and_y(x, y))
        ys.append(y)
    return jnp.stack(ys)


def unroll(f, xs, merge_x_and_y, y0, tbptt: int = 100):
    f = f.reset()
    f_arr, f_funcs = eqx.partition(f, eqx.is_array)

    def body_fn(carry, x):
        f_arr, y, counter = carry
        f = eqx.combine(f_arr, f_funcs)
        f, y = f(merge_x_and_y(x, y))
        f_arr, _ = eqx.partition(f, eqx.is_array)

        # truncated backprop through time
        # def stop_grad(f_arr):
        #    return eqx.tree_at(lambda tree: tree.state, f_arr, \
        # replace_fn=jax.lax.stop_gradient)
        # f_arr = jax.lax.cond((counter % tbptt) == 0, stop_grad, lambda x: x, f_arr)

        return (f_arr, y, counter + 1), y

    carry0 = (f_arr, y0, 0)
    return tree_concat(
        [add_batch_dim(y0), jax.lax.scan(body_fn, carry0, xs)[1]], True, backend="jax"
    )


def unroll_model(model, us):
    return unroll(model, us, lambda x, y: x, model.y0())


def unroll_closed_loop(
    model,
    controller,
    refs,
    y0,
    merge_x_y,
    delay: int,
    u_transform_factory: Callable,
    key: PRNGKey,
):

    u_transform = u_transform_factory(key)

    closed_loop = initialize_closed_loop(model, controller, delay, u_transform)
    return unroll(closed_loop, refs, merge_x_y, y0)
