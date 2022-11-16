import equinox as eqx
import jax.numpy as jnp

from ..core import AbstractController


def make_feedforward_controller(us: jnp.ndarray):
    init_count = jnp.array([0])

    class FeedforwardController(AbstractController):
        us: jnp.ndarray  # parameters
        count: jnp.ndarray  # not a parameter

        def step(self, x):
            return (
                FeedforwardController(self.us, self.count + 1),
                self.us[self.count[0]],
            )

        def reset(self):
            return FeedforwardController(self.us, init_count)

        def grad_filter_spec(self):
            # by default everything is set to `True`
            filter_spec = super().grad_filter_spec()
            return eqx.tree_at(lambda ctrb: ctrb.count, filter_spec, False)

    return FeedforwardController(us, init_count)
