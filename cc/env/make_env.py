from typing import Optional, Type
from dataclasses import dataclass
from typing import Callable, List
from dm_control import mujoco

import dm_env
from acme.wrappers import SinglePrecisionWrapper
from dm_control.rl import control

from .register import _register
from .wrappers import (
    DelayActionWrapper,
    TimelimitControltimestepWrapper,
    TrackTimeWrapper,
)


@dataclass
class EnvConfig:
    load_physics: Callable[[], mujoco.Physics]
    Task: Type[control.Task]


def _make_unwrapped_env(
    env_config: EnvConfig,
    time_limit: float,
    control_timestep: float,
    **task_kwargs
):

    if "random" not in task_kwargs:
        raise Exception(
            "Please fix the seed to ensure deterministic behaviour, use `random=*int*`"
        )

    physics = env_config.load_physics()
    task = env_config.Task(**task_kwargs)

    env = control.Environment(
        physics, task, time_limit, control_timestep
    )

    return env, time_limit, control_timestep


def make_unwrapped_env(
    env_config: EnvConfig,
    time_limit: float,
    control_timestep: float,
    **task_kwargs
) -> dm_env.Environment:

    return _make_unwrapped_env(env_config, time_limit, control_timestep, **task_kwargs)[0]


def _make_almost_unwrapped_env(
    env_config: EnvConfig,
    time_limit: float,
    control_timestep: float,
    single_precision: Optional[bool] = True,
    **task_kwargs
):

    env, time_limit, control_timestep = _make_unwrapped_env(
        env_config, time_limit, control_timestep, **task_kwargs
    )

    if single_precision:
        env = SinglePrecisionWrapper(env)

    return env, time_limit, control_timestep


def make_almost_unwrapped_env(
    env_config: EnvConfig,
    time_limit: float,
    control_timestep: float,
    single_precision: Optional[bool] = True,
    **task_kwargs
) -> dm_env.Environment:

    env = make_unwrapped_env(env_config, time_limit, control_timestep, **task_kwargs)

    if single_precision:
        env = SinglePrecisionWrapper(env)

    return env


def make_env_from_config(
    env_config: EnvConfig,
    time_limit: float = 10.0,
    control_timestep: float = 0.01,
    single_precision: Optional[bool] = True,
    delay: int = 0,
    **task_kwargs
) -> TimelimitControltimestepWrapper:  # TODO pytype will fight you on this

    env, time_limit, control_timestep = _make_almost_unwrapped_env(
        env_config, time_limit, control_timestep, single_precision, **task_kwargs
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
