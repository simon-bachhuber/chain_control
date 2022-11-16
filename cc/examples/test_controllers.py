from ..env import make_env
from ..env.collect import collect, sample_feedforward_collect_and_make_source
from ..env.wrappers import AddRefSignalRewardFnWrapper
from ..utils.legacy_wrapper import WrapController
from .feedforward_controller import make_feedforward_controller

# from .neural_ode_controller import make_neural_ode_controller
from .neural_ode_controller_compact_example import make_neural_ode_controller
from .pid_controller import make_pid_controller


def dummy_env():
    return make_env("two_segments_v1", random=1)


def test_controllers():
    env = dummy_env()
    source, _ = sample_feedforward_collect_and_make_source(env, seeds=[0])
    env_w_source = AddRefSignalRewardFnWrapper(env, source)

    controllers = [
        make_pid_controller(10.0, 2.0, 1.0, env.control_timestep),
        make_neural_ode_controller(
            env_w_source.observation_spec(), env.action_spec(), env.control_timestep, 10
        ),
        make_feedforward_controller(env.ts),
    ]

    for controller in controllers:
        collect(env_w_source, WrapController(controller))
