import jax
import numpy as np

from cc.env import make_env
from cc.env.collect import collect, sample_feedforward_collect_and_make_source
from cc.env.wrappers import AddRefSignalRewardFnWrapper
from cc.examples.linear_model import make_linear_model
from cc.examples.pid_controller import make_pid_controller
from cc.train.step_fn import merge_x_y
from cc.utils.constant_value_controller_wrapper import constant_value_controller_wrapper


def test_constant_value_controller_wrapper():
    T1 = 1.0
    T2 = 0.5

    env = make_env("two_segments_v2", time_limit=T1)
    controller = make_pid_controller(2.0, 0, 0, env.control_timestep())
    source, *_ = sample_feedforward_collect_and_make_source(env)
    env_w_ref = AddRefSignalRewardFnWrapper(env, source)

    def method(method):
        safe_controller = constant_value_controller_wrapper(
            controller, env.control_timestep(), T2, env.action_spec().shape, method
        )
        sample, _ = collect(env_w_ref, safe_controller)
        return sample.action[0]

    max_count = int(T2 / env.control_timestep()) + 1
    action = method("zeros")
    N_after = action.shape[0] - max_count
    np.testing.assert_allclose(
        action[max_count:],
        np.zeros((N_after, *env.action_spec().shape)),
    )

    action = method("last_action")
    np.testing.assert_allclose(
        action[max_count:],
        np.repeat(action[max_count - 1][None], N_after, 0),
    )

    # test unrolling
    model = make_linear_model(
        env.action_spec(), env.observation_spec(), env.control_timestep(), 3
    )
    safe_controller = constant_value_controller_wrapper(
        controller, env.control_timestep(), T2, env.action_spec().shape
    )

    safe_controller.unroll(model, merge_x_y)(
        source.get_reference_actor(), jax.random.PRNGKey(1)
    )
