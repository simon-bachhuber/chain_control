from dataclasses import dataclass
from typing import Callable, Optional, Type

from acme.wrappers import SinglePrecisionWrapper
from dm_control import mujoco
from dm_control.rl import control


@dataclass
class EnvConfig:
    load_physics: Callable[[], mujoco.Physics]
    task: Type[control.Task]


def make_unwrapped_env(
    env_config: EnvConfig,
    time_limit: float,
    control_timestep: float,
    task_kwargs={},
    physics_kwargs={},
):
    physics = env_config.load_physics(**physics_kwargs)
    task = env_config.task(**task_kwargs)

    env = control.Environment(physics, task, time_limit, control_timestep)

    return env


def make_env_from_config(
    env_config: EnvConfig,
    time_limit: float = 10.0,
    control_timestep: float = 0.01,
    single_precision: Optional[bool] = True,
    task_kwargs={},
    physics_kwargs={},
):
    env = make_unwrapped_env(
        env_config,
        time_limit,
        control_timestep,
        task_kwargs,
        physics_kwargs,
    )

    if single_precision:
        env = SinglePrecisionWrapper(env)

    return env


def make_env(
    id: str,
    time_limit: float = 10.0,
    control_timestep: float = 0.01,
    single_precision: Optional[bool] = True,
    task_kwargs={},
    physics_kwargs={},
    **kwargs,
):
    # prevent circular import
    from cc.env.sample_envs import _id_accessible_envs

    if id not in _id_accessible_envs:
        raise Exception(
            f"Unknown environment id {id}, available ids are: "
            f"{_id_accessible_envs.keys()}"
        )

    env_config = _id_accessible_envs[id]

    return make_env_from_config(
        env_config,
        time_limit,
        control_timestep,
        single_precision,
        task_kwargs,
        physics_kwargs,
    )
