from abc import ABC
from .replay_element import ReplayElement
import numpy as np 
import jax 
from flax import struct
from acme.specs import EnvironmentSpec
from acme.jax import utils
from beartype import beartype
from ..utils import tree_insert_IMPURE


class AbstractSampler(ABC):
    pass 


class TransitionAccumulation:

    def __init__(self, n_transitions = None, episodic = None):
        self.n_transitions = n_transitions
        self.episodic = episodic
        self.reset()

    def first_transition(self, ele: ReplayElement) -> bool:

        if self.episodic:
            if ele.timestep.first():
                return True 
            else:
                return False 

        self._i += 1
        if self.n_transitions:
            if self._i >= self.n_transitions:
                return True 
            else: 
                return False 

    def reset(self):
        self._i = 0


@struct.dataclass
class ReplaySample:
    obs: np.ndarray
    action: np.ndarray
    rew: np.ndarray
    done: np.ndarray
    extras: dict 

    @property
    def bs(self):
        assert self.action.ndim == 3 
        return self.action.shape[0]

    @property
    def n_timesteps(self):
        assert self.action.ndim == 3
        return self.action.shape[1]

from beartype.typing import NewType
ExtrasSpecs = NewType("ExtrasSpecs", dict)
default_extra_specs = {"NONE": np.array([0.0])}

class Sampler:

    def __init__(self, 
        env_specs: EnvironmentSpec,
        extras_specs: ExtrasSpecs = default_extra_specs,
        n_transitions = None, 
        episodic = None,
        ts: np.ndarray = None, 
        sample_with_replacement: bool = True,
        dtype_of_dones = np.float32,
        ):
        self.episodic = episodic
        self.n_transitions = n_transitions
        self.extras_specs = extras_specs

        self.transition_accumulation = TransitionAccumulation(n_transitions, episodic)

        self.obs_specs = env_specs.observations
        self.action_specs = env_specs.actions
        self.reward_specs = env_specs.rewards
        self.dtype_dones = dtype_of_dones

        self.sample_with_replacement = sample_with_replacement

        if episodic:
            self.n = len(ts)
        else:
            self.n = self.n_transitions

    def _preallocate(self, bs, n) -> ReplaySample:
        map = jax.tree_util.tree_map

        obs = utils.zeros_like(self.obs_specs)
        # n+1 because of initial state after env.reset()
        obs = map(lambda arr: np.zeros((bs, n+1, *arr.shape), arr.dtype), obs)

        # pre-allocate extras
        extras = map(lambda arr: np.zeros((bs, n, *arr.shape), arr.dtype), self.extras_specs)

        return (
            obs, # obs 
            np.zeros((bs, n, *self.action_specs.shape), self.action_specs.dtype), # action
            np.zeros((bs, n, 1), self.reward_specs.dtype), # rewards
            np.zeros((bs, n, 1), self.dtype_dones), # done
            extras # extras
        )

    def update_weights_when_sampling(self, weights):
        return weights 

    def update_weights_when_inserting(self, weigths):
        return weigths 

    def draw_idxs_from_weights(self, probs, bs: int):
        idxs = np.random.choice(np.arange(len(probs)), size=bs, replace=self.sample_with_replacement, p = probs[:,0])
        return idxs 
    
    @beartype
    def sample(self, samples: list[ReplayElement], bs = None) -> ReplaySample:

        if bs is None:
            bs = len(samples)

        alloc_obs, alloc_action, alloc_rew, alloc_done, \
            alloc_extras = self._preallocate(bs, self.n)

        for i, sample in enumerate(samples):

            trajectory = []
            while True:
                trajectory.append(sample)
                
                if self.transition_accumulation.first_transition(sample):
                    trajectory_long_enough = True 
                    break 

                if sample.prev is None:
                    trajectory_long_enough = False 
                    break 

                sample = sample.prev 

            self.transition_accumulation.reset()
     
            if not trajectory_long_enough:
                continue 

            # we started at the last timestep, so time is reversed
            trajectory.reverse()

            for t, sample in enumerate(trajectory):

                tree_insert_IMPURE(alloc_extras, sample.extras, (i,t))

                tree_insert_IMPURE(alloc_obs, sample.timestep.observation, (i,t))
                # save final `next_obs`
                if t==(self.n-1):
                    tree_insert_IMPURE(alloc_obs, sample.next_timestep.observation, (i,self.n))
                
                alloc_action[i,t] = sample.action
                alloc_rew[i,t] = sample.next_timestep.reward
                alloc_done[i,t] = float(sample.next_timestep.last())
            
        return ReplaySample(alloc_obs, alloc_action, alloc_rew, alloc_done, alloc_extras)

