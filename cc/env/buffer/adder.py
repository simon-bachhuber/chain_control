from abc import ABC, abstractmethod

import dm_env
import numpy as np

from .ray_utils import if_ray_actor
from .replay_buffer import AbstractReplayBuffer
from .replay_element_sample import ReplayElement


class AbstractAdder(ABC):
    @abstractmethod
    def add_first(self, timestep: dm_env.TimeStep):
        pass

    @abstractmethod
    def add(self, action: np.ndarray, next_timestep: dm_env.TimeStep, extras: dict):
        pass

    @abstractmethod
    def reset(self):
        pass


class Adder(AbstractAdder):
    def __init__(self, replay_buffer: AbstractReplayBuffer, flush_cache_every: int = 1):
        self._replay_buffer = replay_buffer
        self._actor_id = None
        self._flush_cache_every = flush_cache_every
        self._episode_id = -1
        self._cache = []
        self.reset()

    def reset(self):
        self._prev_ts = None
        self._timestep_id = 0

        if len(self._cache) != 0:
            print(
                f"""WARNING: You have just flushed the cache of an `adder` that was not
                 empty. It contained {len(self._cache)} `ReplayElement`(s)."""
            )
        self._cache = []

    def set_actor_id(self, actor_id: int):
        self._actor_id = actor_id

    def add_first(self, timestep: dm_env.TimeStep):
        self._prev_ts = timestep
        self._episode_id += 1
        self._timestep_id = 0

    def add(
        self,
        action: np.ndarray,
        next_timestep: dm_env.TimeStep,
        extras: dict = {},
    ):

        if self._prev_ts is None:
            raise Exception()

        ele = ReplayElement(
            self._prev_ts,
            action,
            next_timestep,
            None,
            self._actor_id,
            self._episode_id,
            self._timestep_id,
            extras,
        )

        self._cache.append(ele)

        if len(self._cache) >= self._flush_cache_every:
            if_ray_actor(
                self._replay_buffer,
                "insert",
                self._cache,
                self._actor_id,
                blocking=True,
            )
            self._cache = []

        self._prev_ts = next_timestep

        self._timestep_id += 1
