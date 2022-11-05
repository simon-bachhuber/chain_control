import dm_env
import numpy as np
from acme.wrappers import EnvironmentWrapper

from ...abstract import AbstractObservationReferenceSource
from ...types import *
from ...utils import batch_concat, tree_slice
from ..sample_from_spec import _spec_from_observation


def default_reward_fn(obs, obs_ref) -> np.ndarray:
    obs = batch_concat(obs, 0)
    obs_ref = batch_concat(obs_ref, 0)
    return -np.mean((obs_ref - obs) ** 2)


class AddRefSignalRewardFnWrapper(EnvironmentWrapper):
    def __init__(
        self,
        environment: dm_env.Environment,
        source: AbstractObservationReferenceSource,
        reward_fn: FunctionType = default_reward_fn,
    ):
        self._source = source

        reward_spec = environment.reward_spec()
        assert reward_spec.shape == ()
        # if the reference is float64, numpy will promote
        # even if the environment observations are float32
        # this would conflict with the dtype of the reward
        self._reward_fn = lambda *args: reward_fn(*args).astype(reward_spec.dtype)

        super().__init__(environment)

    def _modify_timestep(self, timestep: dm_env.TimeStep):
        padded_obs = OrderedDict()
        padded_obs["obs"] = timestep.observation
        padded_obs["ref"] = tree_slice(
            self._source.get_reference_actor(), self.i_timestep
        )

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
        if self.requires_reset:
            return self.reset()

        timestep = super().step(action)
        return self._modify_timestep(timestep)

    def reset(self) -> dm_env.TimeStep:
        timestep = super().reset()
        return self._modify_timestep(timestep)

    def observation_spec(self):
        timestep = self._modify_timestep(super().reset())
        return _spec_from_observation(timestep.observation)
