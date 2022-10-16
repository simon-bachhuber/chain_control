from collections import deque
import asyncio
from abc import ABC, abstractmethod
import numpy as np 
import jax 
from beartype import beartype

from .replay_element import ReplayElement
from .rate_limiting import AbstractRateLimiter, NoRateLimitingLimiter
from ._ring_array import RingArray
from beartype.typing import Optional, Iterator
from .sampler import Sampler, ReplaySample
from ..utils.ray_utils import if_is_actor, SyncOrAsyncClass


class AbstractReplayBuffer(ABC):

    @abstractmethod
    def __len__(self) -> int:
        pass 

    def len(self) -> int:
        return len(self)

    @abstractmethod
    def sample(self, bs: int) -> list[ReplayElement]:
        """Accumulate the transitions then to N-Transition Experiences. 
        Pre-allocate `bs` arrays, as zeros and leave empty if a transition
        doesn't reach back far enough

        Args:
            bs (int): _description_

        Returns:
            list[ReplayElement]: _description_
        """
        pass 

    @abstractmethod
    def insert(self, 
        replay_element: ReplayElement, 
        actor_id: Optional[int] = None
        ) -> None:
        pass 


@beartype
def buffer_to_iterator(buffer: AbstractReplayBuffer, bs: int, force_batch_size: bool = True) -> Iterator[ReplaySample]:
    while True:
        yield if_is_actor(buffer, "sample", True, bs, force_batch_size)


class _SharedDequeReplayBuffer:

    def __init__(self, maxlen: int = 10_000, 
        rate_limiter: AbstractRateLimiter = NoRateLimitingLimiter(),
        sampler: Optional[Sampler] = None
        ):
        self._deque = deque(maxlen=maxlen)
        self._dones = RingArray(maxlen=maxlen)
        self._weights = RingArray(maxlen=maxlen)

        self.maxlen = maxlen
        self._rate_limiter = rate_limiter
        self._last_replay_elements_by_actor_id = {}
        self._n_inserts_guaranteed_ready = 5 
        self._sampler = sampler
        self._sample_only_dones = sampler.episodic

    def __len__(self) -> int:
        return len(self._deque)

    def _convert_deque_to_list(self) -> list[ReplayElement]:
        return list(self._deque)

    def _sample(self, bs: int, force_batch_size) -> ReplaySample:

        self._rate_limiter.count_sample_up()

        # this one calls `_convert_deque_to_list` internally
        deque_as_list = self._async_convert_deque_to_list()

        # update weights
        self._weights._arr = self._sampler.update_weights_when_sampling(self._weights._arr)

        weights = self._weights[:]

        if self._sample_only_dones:
            # sample only ReplayElements that are at the last timestep
            weights *= self._dones[:]

        # convert to probabilities
        weights_that_sum_to_1 = weights/np.sum(weights)

        # sample indices using weights
        idxs = self._sampler.draw_idxs_from_weights(weights_that_sum_to_1, bs)

        pre_alloc_bs = None 
        if force_batch_size:
            pre_alloc_bs = bs 

        # TODO list comprehensions are kinda meh
        return self._sampler.sample([deque_as_list[idx] for idx in idxs], pre_alloc_bs)

    def _insert(self, replay_element: ReplayElement, actor_id):

        self._rate_limiter.count_insert_up()

        # TODO What is up with that part?
        # It is due to vecEnv ..
        if isinstance(replay_element.timestep, list):
            one_timestep = replay_element.timestep[0]
            next_timestep = replay_element.next_timestep[0]
        else:
            one_timestep = replay_element.timestep
            next_timestep = replay_element.next_timestep

        # fill up the dones
        if next_timestep.last():
            self._dones.append(1.0)
        else:
            self._dones.append(0.0)

        # fill up the weights
        self._weights.append(1.0)

        # update weights
        self._weights._arr = self._sampler.update_weights_when_inserting(self._weights._arr)

        if one_timestep.first():
            prev_replay_element = None 
        else:
            prev_replay_element = self._last_replay_elements_by_actor_id[actor_id]

        # save replay element for when this actor_id next inserts
        self._last_replay_elements_by_actor_id[actor_id] = replay_element

        if len(self._deque) >= self.maxlen:
            # de-reference the element that is about the get deleted
            self._deque[1].prev = None 

        replay_element.prev = prev_replay_element

        self._async_append(replay_element)

    def _append(self, replay_element):
        self._deque.append(replay_element)

    def _sample_ready(self, bs: int) -> bool:
        if len(self) < bs:
            return False 
        else:
            return not self._rate_limiter.sample_block()
    
    def _insert_ready(self) -> bool:
        if len(self) < self._n_inserts_guaranteed_ready:
            return True 
        else:
            return not self._rate_limiter.insert_block()


class _SyncDequeReplayBuffer(_SharedDequeReplayBuffer, AbstractReplayBuffer):

    def _async_convert_deque_to_list(self):
        # not async
        return self._convert_deque_to_list()

    def sample(self, bs: int, force_batch_size: bool) -> list[ReplayElement]:
        if not self._sample_ready(bs):
            return []
        return self._sample(bs, force_batch_size)

    def _async_append(self, replay_element):
        # not async
        self._append(replay_element)

    def insert(self, replay_element: ReplayElement, actor_id):
        del actor_id
        
        if not self._insert_ready():
            return 
        self._insert(replay_element, actor_id = 0)


class _AsyncDequeReplayBuffer(_SharedDequeReplayBuffer, AbstractReplayBuffer):

    def _async_convert_deque_to_list(self):
        # not async
        return self._convert_deque_to_list()

    async def sample(self, bs: int) -> list[ReplayElement]:
        while not self._sample_ready(bs):
            await asyncio.sleep(0)
        return self._sample(bs)

    def _async_append(self, replay_element):
        # not async
        self._append(replay_element)

    async def insert(self, replay_element: ReplayElement, actor_id):
        while not self._insert_ready():
            await asyncio.sleep(0)
        self._insert(replay_element, actor_id)

#DequeReplayBuffer = SyncOrAsyncClass(_AsyncDequeReplayBuffer, _SyncDequeReplayBuffer)
DequeReplayBuffer = _SyncDequeReplayBuffer
