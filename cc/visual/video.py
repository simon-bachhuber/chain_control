from lib.utils import *
from lib.experimental.discrete_time_17_09.train import ControllerOrModelConfig
from lib.rl.envs.wrappers import AddReferenceObservationWrapper, RecordVideoWrapper
from lib.experimental.discrete_time_17_09.collect import collect

def make_video_(path_controller_config: str, camera_id: str) -> None:
    cc = path_controller_config
    controller_config = ControllerOrModelConfig(f"configs/controllers/{cc}/controller.json")
    env = controller_config.env_config.env 
    source = controller_config.env_config.source 
    ts = generate_ts(env.time_limit, env.control_timestep)
    env = AddReferenceObservationWrapper(env, source)
    video_env = RecordVideoWrapper(env, height=1080, width=1920, video_name=f"{cc}_refid_0_camera_{camera_id}", camera_id=camera_id)
    _ = collect(video_env, controller_config.controller.obj, ts)

