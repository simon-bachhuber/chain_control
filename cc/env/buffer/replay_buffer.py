import asyncio
from abc import ABC, abstractmethod
from collections import deque
from typing import Iterator, Optional, Union

import ray

from .ray_utils import if_ray_actor
from .rate_limiting import AbstractRateLimiter, NoRateLimitingLimiter
from .replay_element_sample import ReplayElement
from .ring_array import RingArray
from .sampler import ReplaySample, Sampler


class AbstractReplayBuffer(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    def len(self) -> int:
        return len(self)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def sample(self, bs: int) -> list[ReplayElement]:
        pass

    @abstractmethod
    def insert(
        self, replay_element: ReplayElement, actor_id: Optional[int] = None
    ) -> None:
        pass


def buffer_to_iterator(
    buffer: AbstractReplayBuffer, bs: int, force_batch_size: bool = True
) -> Iterator[ReplaySample]:
    while True:
        yield if_ray_actor(buffer, "sample", bs, force_batch_size, blocking=True)


class _ReplayBuffer:
    def __init__(
        self,
        maxlen: int = 10_000,
        rate_limiter: AbstractRateLimiter = NoRateLimitingLimiter(),
        sampler: Optional[Sampler] = None,
    ):
        self._rate_limiter = rate_limiter
        self._sampler = sampler
        self.maxlen = maxlen
        self.reset()

    def reset(self):

        self._deque = deque(maxlen=self.maxlen)
        self._dones = RingArray(maxlen=self.maxlen)
        self._weights = RingArray(maxlen=self.maxlen)

        self._rate_limiter.reset()
        self._last_replay_elements_by_actor_id = {}
        self._episode_ready = False

    def close(self):
        del self._deque, self._dones, self._weights, self._rate_limiter

    def __len__(self) -> int:
        return len(self._deque)

    def _sample(self, bs: int, force_batch_size) -> ReplaySample:

        self._rate_limiter.count_sample_up()

        # update weights
        self._weights._arr = self._sampler.update_weights_when_sampling(
            self._weights._arr
        )

        # sample indices using weights
        idxs = self._sampler.draw_idxs_from_weights(
            self._weights[:], self._dones[:], bs
        )

        pre_alloc_bs = None
        if force_batch_size:
            pre_alloc_bs = bs

        # indexing a list is faster than a deque
        deque_as_list = list(self._deque)
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
            self._episode_ready = True
            self._dones.append(1.0)
        else:
            self._dones.append(0.0)

        # fill up the weights
        self._weights.append(1.0)

        # update weights
        self._weights._arr = self._sampler.update_weights_when_inserting(
            self._weights._arr
        )

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

        self._deque.append(replay_element)

    def _sample_ready(self) -> bool:
        if len(self) == 0:
            return False
        if self._sampler._episodic and not self._episode_ready:
            return False
        return not self._rate_limiter.sample_block()

    def _insert_ready(self) -> bool:
        return not self._rate_limiter.insert_block()


class ReplayBuffer(_ReplayBuffer, AbstractReplayBuffer):
    def sample(self, bs: int, force_batch_size: bool) -> list[ReplayElement]:
        if not self._sample_ready():
            return []
        return self._sample(bs, force_batch_size)

    def insert(
        self, replay_element: Union[ReplayElement, list[ReplayElement]], actor_id
    ):
        if not self._insert_ready():
            return

        if isinstance(replay_element, ReplayElement):
            replay_element = [replay_element]

        for ele in replay_element:
            self._insert(ele, actor_id)


@ray.remote
class RayReplayBuffer(_ReplayBuffer, AbstractReplayBuffer):
    def current_ratio(self):
        return self._rate_limiter._current_ratio()

    async def sample(self, bs: int, force_batch_size: bool) -> list[ReplayElement]:
        while not self._sample_ready():
            await asyncio.sleep(0)
        return self._sample(bs, force_batch_size)

    async def insert(
        self, replay_element: Union[ReplayElement, list[ReplayElement]], actor_id
    ):
        while not self._insert_ready():
            await asyncio.sleep(0)

        if isinstance(replay_element, ReplayElement):
            replay_element = [replay_element]

        for ele in replay_element:
            self._insert(ele, actor_id)
