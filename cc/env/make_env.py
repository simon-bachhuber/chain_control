from typing import Optional

import dm_env
from acme.wrappers import SinglePrecisionWrapper
from dm_control.rl import control

from .register import _register
from .wrappers import (
    DelayActionWrapper,
    TimelimitControltimestepWrapper,
    TrackTimeWrapper,
)


def _make_unwrapped_env(
    id: str,
    time_limit: Optional[float] = None,
    control_timestep: Optional[float] = None,
    **task_kwargs
):

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

    return env, time_limit, control_timestep


def make_unwrapped_env(
    id: str,
    time_limit: Optional[float] = None,
    control_timestep: Optional[float] = None,
    **task_kwargs
) -> dm_env.Environment:

    return _make_unwrapped_env(id, time_limit, control_timestep, **task_kwargs)[0]


def _make_almost_unwrapped_env(
    id: str,
    time_limit: Optional[float] = None,
    control_timestep: Optional[float] = None,
    single_precision: Optional[bool] = True,
    **task_kwargs
):

    env, time_limit, control_timestep = _make_unwrapped_env(
        id, time_limit, control_timestep, **task_kwargs
    )

    if single_precision:
        env = SinglePrecisionWrapper(env)

    return env, time_limit, control_timestep


def make_almost_unwrapped_env(
    id: str,
    time_limit: Optional[float] = None,
    control_timestep: Optional[float] = None,
    single_precision: Optional[bool] = True,
    **task_kwargs
) -> dm_env.Environment:

    env = make_unwrapped_env(id, time_limit, control_timestep, **task_kwargs)

    if single_precision:
        env = SinglePrecisionWrapper(env)

    return env


def make_env(
    id: str,
    time_limit: Optional[float] = None,
    control_timestep: Optional[float] = None,
    single_precision: Optional[bool] = True,
    delay: int = 0,
    **task_kwargs
) -> TimelimitControltimestepWrapper:  # TODO pytype will fight you on this

    env, time_limit, control_timestep = _make_almost_unwrapped_env(
        id, time_limit, control_timestep, single_precision, **task_kwargs
    )

    if delay > 0:
        env = DelayActionWrapper(env, delay)

    # add some metadata
    env = TimelimitControltimestepWrapper(
        env,
        time_limit=time_limit,
        control_timestep=control_timestep,
        delay=delay,
    )

    env = TrackTimeWrapper(env)

    return env
