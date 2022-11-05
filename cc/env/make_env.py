from typing import Optional

from acme.wrappers import SinglePrecisionWrapper
from dm_control.rl import control

from .register import _register
from .wrappers import DelayActionWrapper, TimelimitControltimestepWrapper


def make_env(
    id: str,
    time_limit: Optional[float] = None,
    control_timestep: Optional[float] = None,
    single_precision: Optional[bool] = True,
    delay: int = 0,
    **task_kwargs
) -> TimelimitControltimestepWrapper:

    if "random" not in task_kwargs:
        raise Exception(
            "Please fix the seed to ensure deterministic behaviour, use `random=*int*`"
        )

    physics = _register[id].load_physics()
    task = _register[id].Task(**task_kwargs)
    time_limit = time_limit if time_limit else _register[id].time_limit
    control_timestep = (
        control_timestep if control_timestep else _register[id].control_timestep
    )

    env = control.Environment(
        physics, task, time_limit=time_limit, control_timestep=control_timestep
    )

    if single_precision:
        env = SinglePrecisionWrapper(env)

    if delay > 0:
        env = DelayActionWrapper(env, delay)

    # add some metadata
    env = TimelimitControltimestepWrapper(
        env,
        time_limit=time_limit,
        control_timestep=control_timestep,
        delay=delay,
    )

    # to avoid that the first .step can actually
    # be the .reset - timestep
    # TODO This is disabled because it violates tests
    # env.reset()

    return env
