from acme.wrappers import EnvironmentWrapper
import dm_env
import numpy as np 

from ...types import *
from ..sample_from_spec import _spec_from_observation
from ...utils import batch_concat, idx_in_pytree
from ...abstract import AbstractObservationReferenceSource


class _RunningSource:
    """Wraps `ObservationReferenceSource` and captures a counter. Used only inside Environments to add a reference-observation.
    """
    def __init__(self, source: AbstractObservationReferenceSource):
        self._source = source 
        self.reset()

    def get_reference_actor_at_timestep(self) -> Reference:
        return idx_in_pytree(self._source.get_reference_actor(), self._timestep)

    def increase_timestep(self):
        self._timestep += 1 

    def reset(self):
        self._timestep = 0


def default_reward_fn(obs, obs_ref):
    obs = batch_concat(obs, 0)
    obs_ref = batch_concat(obs_ref, 0)

    return (-np.mean((obs_ref - obs)**2)).item()


class AddReferenceObservationWrapper(EnvironmentWrapper):
    def __init__(self, 
        environment: dm_env.Environment,
        source: AbstractObservationReferenceSource,
        reward_fn: FunctionType = default_reward_fn
    ):
        self._source = _RunningSource(source)
        self._reward_fn = reward_fn
        super().__init__(environment)

    def _modify_timestep(self, timestep: dm_env.TimeStep):
        padded_obs = OrderedDict()
        padded_obs["obs"] = timestep.observation 
        padded_obs["ref"] = self._source.get_reference_actor_at_timestep()
        
        # calculate reward
        # dm_env has convention that first timestep has no reward
        # it is None then
        if timestep.first():
            reward = None 
        else:
            reward = self._reward_fn(padded_obs["obs"], padded_obs["ref"])
        timestep = timestep._replace(observation=padded_obs)
        timestep = timestep._replace(reward=reward)
        return timestep

    def step(self, action) -> dm_env.TimeStep:
        timestep = super().step(action)
        self._source.increase_timestep()
        return self._modify_timestep(timestep)

    def reset(self) -> dm_env.TimeStep:
        timestep = super().reset()
        # reset source
        self._source.reset()
        return self._modify_timestep(timestep)

    def observation_spec(self):
        timestep = self._modify_timestep(super().reset())
        return _spec_from_observation(timestep.observation)

