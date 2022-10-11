from .adder import Adder, AbstractAdder
from ._ring_array import RingArray
from .deque_buffer import DequeReplayBuffer, AbstractReplayBuffer, buffer_to_iterator
from .rate_limiting import AbstractRateLimiter, NoRateLimitingLimiter, RateLimiter
from .replay_element import ReplayElement
from .sampler import ReplaySample, Sampler, AbstractSampler, default_extra_specs
import numpy as np 


def make_episodic_buffer_adder_iterator(bs, ts, env_specs, actor_id):

    class _Sampler(Sampler):
        def draw_idxs_from_weights(self, probs, bs: int):
            probs = probs[:,0]
            idxs = np.where(probs)[0]
            return idxs 

    buffer = DequeReplayBuffer(
        maxlen=len(ts)*bs, sampler=_Sampler(env_specs=env_specs, ts=ts, episodic=True)
    )

    iterator = buffer_to_iterator(buffer, bs)
    adder = Adder(buffer)
    adder.set_actor_id(actor_id)
    return buffer, adder, iterator 
