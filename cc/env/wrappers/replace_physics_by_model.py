import dm_env
import numpy as np
from acme.wrappers import EnvironmentWrapper

from ...abstract import AbstractModel
from ...types import *
from ...utils import to_jax, to_numpy


class ReplacePhysicsByModelWrapper(EnvironmentWrapper):
    def __init__(
        self,
        env: dm_env.Environment,
        model: AbstractModel,
        process_observation: Callable = lambda obs: obs,
    ):

        super().__init__(env)
        self._model = model
        self._process_obs = process_observation
        self._dummy_reward = self._environment.reward_spec().generate_value()
        self._dummy_discount = self._environment.discount_spec().generate_value()
        self._requires_reset = True

    def reset(self):
        self._requires_reset = False

        # reset time
        self._t = 0.0

        # reset model state
        self._model = self._model.reset()

        obs0 = to_numpy(self._model.y0())

        return self._build_timestep(dm_env.StepType.FIRST, obs0)

    def _build_timestep(
        self, step_type: dm_env.StepType, obs: Observation
    ) -> dm_env.TimeStep:

        # envs return OrderedDict of numpy-values
        obs = self._process_obs(to_numpy(obs))

        # dm_env convention is that the first timestep
        # has no reward, i.e. it is None
        # has no discount, i.e. is is None
        if step_type == dm_env.StepType.FIRST:
            rew = None
            discount = None
        else:
            rew = self._dummy_reward
            discount = self._dummy_discount

        return dm_env.TimeStep(step_type, rew, discount, obs)

    def step(self, action: Union[list, np.ndarray]) -> dm_env.TimeStep:

        if self._requires_reset:
            return self.reset()

        if self._t < (self.time_limit - self.control_timestep - 1e-8):
            step_type = dm_env.StepType.MID
        else:
            step_type = dm_env.StepType.LAST
            self._requires_reset = True

        # model is a jax function, so convert to DeviceArray
        action = to_jax(action)
        assert action.ndim == 1

        self._model, obs = eqx.filter_jit(self._model)(action)

        # count up pseudo-time
        self._t += self.control_timestep

        return self._build_timestep(step_type, obs)
