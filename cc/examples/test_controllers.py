import jax

from cc.utils.high_level.defaults import Env

from ..env import make_env
from ..env.collect import collect
from ..env.collect import sample_feedforward_collect_and_make_source
from ..env.wrappers import AddRefSignalRewardFnWrapper
from ..utils.utils import timestep_array_from_env
from .feedforward_controller import make_feedforward_controller
from .neural_ode_controller import make_neural_ode_controller
from .neural_ode_controller_compact_example import (
    make_neural_ode_controller as make_neural_ode_controller_compact,
)
from .pid_controller import make_pid_controller
from .pole_placement_controller import make_pole_placed_controller


def dummy_env():
    return make_env("two_segments_v1", random=1)


_env_data = {
    "train_gp": list(range(1)),
    "train_cos": list(range(1)),
    "val_gp": list(range(2)),
    "val_cos": [2],
}


def test_controllers():
    env = dummy_env()
    source, _, _ = sample_feedforward_collect_and_make_source(env, seeds=[0])
    env_w_source = AddRefSignalRewardFnWrapper(env, source)

    controllers = [
        make_pid_controller(10.0, 2.0, 1.0, env.control_timestep()),
        make_neural_ode_controller(
            env_w_source.observation_spec(),
            env.action_spec(),
            env.control_timestep(),
            10,
            jax.random.PRNGKey(1),
            f_time_invariant=False,
        ),
        make_neural_ode_controller_compact(
            env_w_source.observation_spec(),
            env.action_spec(),
            env.control_timestep(),
            10,
        ),
        make_feedforward_controller(timestep_array_from_env(env)),
        make_pole_placed_controller(
            Env("two_segments_v1", {}, {}, data=_env_data),
            [-0.1] * 2,
            verbose=False,
            state_dim=1,
        )[0](),
    ]

    for controller in controllers:
        collect(env_w_source, controller)
