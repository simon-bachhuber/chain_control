import dm_env
from acme.wrappers import EnvironmentWrapper
from tree_utils import batch_concat

from ...utils.utils import to_numpy


class FlatObservationWrapper(EnvironmentWrapper):
    def _flat_obs(self, ts: dm_env.TimeStep):
        return ts._replace(observation=to_numpy(batch_concat(ts.observation, 0)))

    def step(self, action) -> dm_env.TimeStep:
        ts = super().step(action)
        return self._flat_obs(ts)

    def reset(self) -> dm_env.TimeStep:
        ts = super().reset()
        return self._flat_obs(ts)
