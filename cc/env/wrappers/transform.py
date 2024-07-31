import dm_env
from dm_env import Environment

from cc.acme.wrappers import EnvironmentWrapper


class TransformWrapper(EnvironmentWrapper):
    def __init__(self, environment: Environment, f_action=None, f_obs=None, f_rew=None):
        super().__init__(environment)
        noop = lambda x: x
        self.f_action = f_action if f_action else noop
        self.f_obs = f_obs if f_obs else noop
        self.f_rew = f_rew if f_rew else noop

    def _transform(self, ts: dm_env.TimeStep):
        return ts._replace(
            observation=self.f_obs(ts.observation), reward=self.f_rew(ts.reward)
        )

    def step(self, action) -> dm_env.TimeStep:
        ts = super().step(self.f_action(action))
        return self._transform(ts)

    def reset(self) -> dm_env.TimeStep:
        ts = super().reset()
        return self._transform(ts)
