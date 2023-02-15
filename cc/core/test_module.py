import equinox as eqx
import jax.numpy as jnp

from .abstract import AbstractModel


class Counter(AbstractModel):
    step_size: jnp.ndarray
    count: jnp.ndarray = jnp.array(0)

    def step(self, x={}):
        return Counter(self.step_size, self.count + self.step_size), self.count

    def grad_filter_spec(self):
        return super().grad_filter_spec()

    def reset(self):
        return Counter(self.step_size, jnp.array(0))

    def y0(self):
        return self.count


init_counters = [Counter(jnp.array(i)) for i in range(3)]


class ManyCounters(AbstractModel):
    counters: list[Counter]

    def step(self, x={}):
        y = [counter.step()[1] for counter in self.counters]
        new_state = [counter.step()[0] for counter in self.counters]
        print(y)
        return ManyCounters(new_state), y

    def reset(self):
        return ManyCounters(init_counters)

    def grad_filter_spec(self):
        return super().grad_filter_spec()

    def y0(self):
        return [counter.y0() for counter in self.counters]


def make_counters():
    return ManyCounters(init_counters)


def test_counters():
    counters = make_counters()

    counters, y = counters.step()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(0), jnp.array(0)])

    counters, y = counters.step()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(1), jnp.array(2)])

    counters, y = counters.step()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(2), jnp.array(4)])

    counters = counters.reset()
    counters, y = counters.step()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(0), jnp.array(0)])

    # repeat with jit

    counters = counters.reset()

    counters, y = eqx.filter_jit(counters.step)()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(0), jnp.array(0)])

    counters, y = eqx.filter_jit(counters.step)()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(1), jnp.array(2)])

    counters, y = eqx.filter_jit(counters.step)()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(2), jnp.array(4)])

    counters = counters.reset()
    counters, y = eqx.filter_jit(counters.step)()
    assert eqx.tree_equal(y, [jnp.array(0), jnp.array(0), jnp.array(0)])


def test_unroll():
    counters = make_counters()
    ys = counters.unroll(jnp.zeros((3, 1)), include_y0=False)

    # this the trajectory of the first counter
    assert eqx.tree_equal(ys[0], jnp.array([0, 0, 0]))
    # .. the second counter ..
    assert eqx.tree_equal(ys[1], jnp.array([0, 1, 2]))
    # and the third.
    assert eqx.tree_equal(ys[2], jnp.array([0, 2, 4]))

    # repeat with jit

    counters = counters.reset()

    ys = eqx.filter_jit(counters.unroll)(jnp.zeros((3, 1)), include_y0=False)

    assert eqx.tree_equal(ys[0], jnp.array([0, 0, 0]))
    assert eqx.tree_equal(ys[1], jnp.array([0, 1, 2]))
    assert eqx.tree_equal(ys[2], jnp.array([0, 2, 4]))
