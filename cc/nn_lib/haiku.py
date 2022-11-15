import haiku as hk
import jax.random as jrand

from ..utils import sample_from_tree_of_specs
from ..core import Module, make_module_from_function


def make_module_from_haiku(forward, input_specs, init_key, name=None) -> Module:
    toy_input = sample_from_tree_of_specs(input_specs)
    forward = hk.transform_with_state(forward)
    init_key, consume = jrand.split(init_key)
    init_params, init_state = forward.init(consume, toy_input)
    return make_module_from_function(forward.apply, init_params, init_state, name)
