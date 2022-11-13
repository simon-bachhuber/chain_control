import equinox as eqx
import jax.numpy as jnp
import jax.random as jrand

from .module import make_module_from_eqx_module, make_module_from_function
from ..nn_lib import filter_scan_module


def make_counter(step_size):
    init_state = jnp.array(0)
    init_params = step_size
    init_key = jrand.PRNGKey(
        1,
    )

    class CountUp(eqx.Module):
        step_size: jnp.ndarray

        def __call__(self, state, key, x):
            del x
            return state + self.step_size, key, state

    return make_module_from_eqx_module(CountUp(init_params), init_state, init_key)


def make_many_counters(number_of_counters: int = 3):
    init_state = [
        make_counter(jnp.array(step_size)) for step_size in range(number_of_counters)
    ]
    init_params = {}
    init_key = jrand.PRNGKey(
        1,
    )

    def apply_fn(params, state, key, x):
        counters = state
        ys = [counter()[1] for counter in counters]
        new_counters = [counter()[0] for counter in counters]
        return new_counters, key, ys

    return make_module_from_function(apply_fn, init_params, init_state, init_key)


def test_counters():
    counters = make_many_counters()

    counters, y = counters()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(0), jnp.array(0)])

    counters, y = counters()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(1), jnp.array(2)])

    counters, y = counters()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(2), jnp.array(4)])

    counters = counters.reset()
    counters, y = counters()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(0), jnp.array(0)])

    # repeat with jit

    counters = counters.reset()
    counters = eqx.filter_jit(counters)

    counters, y = counters()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(0), jnp.array(0)])

    counters, y = counters()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(1), jnp.array(2)])

    counters, y = counters()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(2), jnp.array(4)])

    counters = counters.reset()
    counters, y = counters()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(0), jnp.array(0)])


def test_filter_scan():
    def scan_fn(counters, x):
        new_counters, ys = counters()
        return new_counters, ys

    counters = make_many_counters()

    _, ys = filter_scan_module(scan_fn, init=counters, xs=None, length=3)

    assert eqx.tree_equal(ys[0], jnp.array([0, 0, 0]))
    assert eqx.tree_equal(ys[1], jnp.array([0, 1, 2]))
    assert eqx.tree_equal(ys[2], jnp.array([0, 2, 4]))

    # repeat with jit

    counters = counters.reset()

    @eqx.filter_jit
    def unroll(counters):
        _, ys = filter_scan_module(scan_fn, init=counters, xs=None, length=3)
        return ys

    ys = unroll(counters)

    assert eqx.tree_equal(ys[0], jnp.array([0, 0, 0]))
    assert eqx.tree_equal(ys[1], jnp.array([0, 1, 2]))
    assert eqx.tree_equal(ys[2], jnp.array([0, 2, 4]))
