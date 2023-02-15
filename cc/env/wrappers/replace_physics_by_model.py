from typing import Callable, Union

import dm_env
import equinox as eqx
import numpy as np
from acme.wrappers import EnvironmentWrapper
from dm_control.rl.control import Environment

from ...core import AbstractModel
from ...core.types import Observation
from ...utils import to_jax, to_numpy


class ReplacePhysicsByModelWrapper(EnvironmentWrapper):
    def __init__(
        self,
        env: Environment,
        model: AbstractModel,
        process_observation: Callable = lambda obs: obs,
        y0_from_env: bool = False,
    ):
        super().__init__(env)
        self._model = model
        self._y0_from_env = y0_from_env
        self._process_obs = process_observation
        self._dummy_reward = self._environment.reward_spec().generate_value()
        self._dummy_discount = self._environment.discount_spec().generate_value()

    def reset(self):
        super().reset()

        # reset model state
        self._model = self._model.reset()

        if not self._y0_from_env:
            obs0 = to_numpy(self._model.y0())
        else:
            obs0 = self._environment.reset().observation

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
        if self._reset_next_step:
            return self.reset()

        ts = super().step(action)

        # model is a jax function, so convert to DeviceArray
        action = to_jax(action)
        assert action.ndim == 1

        self._model, obs = eqx.filter_jit(self._model.step)(action)

        return self._build_timestep(ts.step_type, obs)
