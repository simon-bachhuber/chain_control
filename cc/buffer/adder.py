from .replay_element import ReplayElement
from .deque_buffer import AbstractReplayBuffer
from ..utils.ray_utils import if_is_actor
from .sampler import default_extra_specs
import dm_env
import numpy as np 
from abc import ABC, abstractmethod
import ray 


class AbstractAdder(ABC):
    @abstractmethod
    def add_first(self, timestep: dm_env.TimeStep):
        pass 
    @abstractmethod
    def add(self, action: np.ndarray, next_timestep: dm_env.TimeStep, extras: dict):
        pass 


class Adder(AbstractAdder):
    def __init__(self, replay_buffer: AbstractReplayBuffer): 
        self._replay_buffer = replay_buffer
        self._prev_ts = None 
        self._actor_id = None 
        self._episode_id = -1
        self._timestep_id = 0

    def set_actor_id(self, actor_id: int):
        self._actor_id = actor_id

    def add_first(self, timestep: dm_env.TimeStep):
        self._prev_ts = timestep
        self._episode_id += 1
        self._timestep_id = 0

    def add(self, action: np.ndarray, next_timestep: dm_env.TimeStep, extras: dict = default_extra_specs):

        if self._prev_ts is None:
            raise Exception()
        
        ele = ReplayElement(
            self._prev_ts, action, next_timestep, None, \
                self._actor_id, self._episode_id, self._timestep_id, extras)

        # TODO not sure if this should be blocking
        if_is_actor(self._replay_buffer, "insert", 
            False, ele, self._actor_id
        )

        self._prev_ts = next_timestep

        self._timestep_id += 1

