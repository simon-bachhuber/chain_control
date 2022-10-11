from typing import Callable
from dataclasses import dataclass
from dm_control.rl import control 
from dm_control import mujoco
from .wrappers import (
    DelayActionWrapper,
    MetaDataWrapper
)
from acme.wrappers import SinglePrecisionWrapper


@dataclass
class EnvRegisterValue:
    load_physics: Callable[[], mujoco.Physics]
    Task: control.Task  
    time_limit: float 
    control_timestep: float = 0.01 


from .envs import (
    two_segments_v1,
    two_segments_v2
)


_register = {
    "two_segments_v1": 
        EnvRegisterValue(
            two_segments_v1.load_physics,
            two_segments_v1.SegmentTask,
            10.0 
        ),
    "two_segments_v2": 
        EnvRegisterValue(
            two_segments_v2.load_physics,
            two_segments_v2.SegmentTask,
            10.0 
        ),
}


def make_env(
    id: str, 
    time_limit: float = None,
    control_timestep: float = None,
    single_precision: bool = True,
    delay: int = 0,
    **task_kwargs
    ):

    if "random" not in task_kwargs:
        raise Exception("Please fix the seed to ensure deterministic behaviour, use `random=*int*`")

    physics = _register[id].load_physics()
    task = _register[id].Task(**task_kwargs)
    time_limit = time_limit if time_limit else _register[id].time_limit
    control_timestep = control_timestep if control_timestep else _register[id].control_timestep

    env = control.Environment(
        physics, 
        task, 
        time_limit=time_limit, 
        control_timestep=control_timestep
    )

    if single_precision:
        env = SinglePrecisionWrapper(env) 

    if delay>0:
        env = DelayActionWrapper(env, delay)
    
    # add some metadata
    env = MetaDataWrapper(env, 
        time_limit=time_limit,
        control_timestep=control_timestep,
        delay=delay 
    )

    # to avoid that the first .step can actually 
    # be the .reset - timestep
    env.reset()
    
    return env 

