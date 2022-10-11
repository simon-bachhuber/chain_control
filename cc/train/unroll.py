from ..types import *
from ..abstract import AbstractModel, AbstractController, AbstractWrappedRHS
from ..utils import add_batch_dim, tree_concat


from beartype import beartype
import jax


def initialize_closed_loop(model, controller, delay):
    delayed_us = jnp.zeros((delay+1, controller.output_size))
    ring_array_idx = jnp.array(0)
    return ClosedLoop(model, controller, delayed_us, ring_array_idx, delay)


class ClosedLoop(AbstractWrappedRHS):
    model: AbstractModel 
    controller: AbstractController
    delayed_us: jnp.ndarray
    ring_array_idx: jnp.ndarray
    delay: int = eqx.static_field()

    def __call__(self, x: PyTree):
        assert self.delay == 0

        # x here is (ref, obs)
        controller, u = self.controller(x)

        # store u in ring-array
        #delayed_us = jax.lax.dynamic_update_index_in_dim(self.delayed_us, u, self.ring_array_idx, axis=0)
        delayed_us = self.delayed_us

        # increase ring array idx 
        #ring_array_idx = (self.ring_array_idx + 1) % self.delay
        ring_array_idx = self.ring_array_idx

        # get u of index one to the right
        #u_apply = jax.lax.dynamic_index_in_dim(delayed_us, ring_array_idx)
        u_apply = u
        model, y = self.model(u_apply)

        return ClosedLoop(model, controller, delayed_us, ring_array_idx, self.delay), y 

    def reset(self):
        controller, model = self.controller.reset(), self.model.reset()
        return initialize_closed_loop(model, controller, self.delay)


def _unroll_static(f, xs, merge_x_and_y, y0):
    f = f.reset()
    
    ys = [y0]
    y = y0
    for x in xs:
        f, y = f(merge_x_and_y(x, y))
        ys.append(y)
    return jnp.stack(ys)


@beartype
def unroll(f: AbstractWrappedRHS, xs, merge_x_and_y, y0, tbptt: int=100):
    f = f.reset()
    f_arr, f_funcs = eqx.partition(f, eqx.is_array)
    
    def body_fn(carry, x):
        f_arr, y, counter = carry 
        f = eqx.combine(f_arr, f_funcs)
        f, y = f(merge_x_and_y(x,y))
        f_arr, _ = eqx.partition(f, eqx.is_array)

        # truncated backprop through time 
        #def stop_grad(f_arr):
        #    return eqx.tree_at(lambda tree: tree.state, f_arr, replace_fn=jax.lax.stop_gradient)
        #f_arr = jax.lax.cond((counter % tbptt) == 0, stop_grad, lambda x: x, f_arr)

        return (f_arr, y, counter+1), y

    carry0 = (f_arr, y0, 0)
    return tree_concat([
        add_batch_dim(y0),
        jax.lax.scan(body_fn, carry0, xs)[1]
    ], True, backend="jax")

    
@beartype
def unroll_model(model: AbstractModel, us: TimeSeriesOfAct) -> TimeSeriesOfObs:
    return TimeSeriesOfObs(unroll(model, us, lambda x,y: x, model.y0()))


@beartype
def unroll_closed_loop(model: AbstractModel, controller: AbstractController, 
    refs: TimeSeriesOfRef, y0: Observation, merge_x_y, delay: int) -> TimeSeriesOfObs:

    closed_loop = initialize_closed_loop(model, controller, delay)
    return TimeSeriesOfObs(unroll(closed_loop, refs, merge_x_y, y0))

