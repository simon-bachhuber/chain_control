import equinox as eqx
from cc.env import make_env_from_config
from cc.env.collect import sample_feedforward_collect_and_make_source
from cc.env.collect.circus import double_step_source
from cc.env.envs.two_segments import (
    CartParams,
    Color,
    JointParams,
    Marker,
    generate_duplicate_env_config,
)
from cc.env.sample_envs import TWO_SEGMENT_V1
from cc.env.wrappers import VideoWrapper
from cc.env.wrappers.add_reference_and_reward import AddRefSignalRewardFnWrapper
from cc.examples.neural_ode_controller_compact_example import make_neural_ode_controller
from cc.env.collect.collect import collect_exhaust_source
from cc.utils.multiple_controller_wrapper import MultipleControllerWrapper
from cc.env.collect import (
    collect_exhaust_source,
    sample_feedforward_collect_and_make_source,
)
from cc.env.collect.collect import (
    collect_exhaust_source,
    sample_feedforward_collect_and_make_source,
)
from cc.env.collect.source import *

env = make_env_from_config(TWO_SEGMENT_V1, time_limit=10.0, control_timestep=0.01)
source, _, _ = sample_feedforward_collect_and_make_source(env, seeds=[100])
env_w_source = AddRefSignalRewardFnWrapper(env, source)

controller1 = make_neural_ode_controller(
    env_w_source.observation_spec(),
    env.action_spec(),
    env.control_timestep(),
    state_dim=80,
    f_width_size=0,
    f_depth=0,
    g_width_size=0,
    g_depth=0,
)

controller2 = make_neural_ode_controller(
    env_w_source.observation_spec(),
    env.action_spec(),
    env.control_timestep(),
    state_dim=80,
    f_width_size=0,
    f_depth=0,
    g_width_size=0,
    g_depth=0,
)

# Replace this with your controllers
controller1 = eqx.tree_deserialise_leaves(f"controller1.eqx", controller1)
controller2 = eqx.tree_deserialise_leaves(f"controller2.eqx", controller2)


video_env_config = generate_duplicate_env_config(
    CartParams(
        name="cart",
        slider_joint_params=JointParams(damping=1e-3),
        hinge_joint_params=JointParams(damping=1e-1, springref=0, stiffness=10),
    ),
    2,
    marker_params=[
        Marker(pos=3, material=Color.RED),
        Marker(pos=6, material=Color.PINK),
    ],
)

video_env = make_env_from_config(
    video_env_config, time_limit=10.0, control_timestep=0.01
)
video_source = double_step_source(video_env, 3)
video_env_w_source = AddRefSignalRewardFnWrapper(video_env, video_source)


wrapper = MultipleControllerWrapper(controller1, controller2)

video_wrapped_env = VideoWrapper(
    video_env_w_source, camera_id="skyview", width=1920, height=1080, path="./video"
)

controller_performance_sample = collect_exhaust_source(video_wrapped_env, wrapper)
