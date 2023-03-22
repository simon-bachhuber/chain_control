from typing import Optional

import dm_env
import equinox as eqx
import jax.random as jrand

from cc.acme import core

from ...core import AbstractController
from ...core.types import Action, Observation, PRNGKey
from ...utils import sample_action_from_action_spec, to_jax, to_numpy
from ..buffer import AbstractAdder


class ModuleActor(core.Actor):
    def __init__(
        self,
        action_spec,
        controller: Optional[AbstractController] = None,
        key: PRNGKey = jrand.PRNGKey(
            1,
        ),
        adder: Optional[AbstractAdder] = None,
        reset_key=False,
        jit=True,
    ):
        self.action_spec = action_spec
        self._adder = adder
        self._controller = controller
        self.reset_key = reset_key
        self._initial_key = self._key = key
        self._jit = jit
        self.reset()

    def observe_first(self, timestep: dm_env.TimeStep):
        self.reset()
        if self._adder:
            self._adder.add_first(timestep)

    def reset(self):
        if self.reset_key:
            self._key = self._initial_key
        else:
            pass

        if self._controller:
            self._controller = self._controller.reset()

        self._last_extras = None
        if self._adder:
            self._adder.reset()

    def observe(self, action, next_timestep):
        if self._adder:
            if self._last_extras:
                self._adder.add(
                    action, next_timestep=next_timestep, extras=self._last_extras
                )
            else:
                self._adder.add(action, next_timestep=next_timestep)

    def update(self, wait: bool = False):
        pass

    def select_action(self, obs: Observation) -> Action:
        self._key, consume = jrand.split(self._key)
        action = self.query_policy(to_jax(obs), consume)
        return to_numpy(action)

    def query_policy(self, obs: Observation, key: PRNGKey) -> Action:
        if self._jit:
            self._controller, action = eqx.filter_jit(self._controller.step)(obs)
        else:
            self._controller, action = self._controller.step(obs)
        return action


class RandomActor(ModuleActor):
    def query_policy(self, obs: Observation, key: PRNGKey) -> Action:
        return sample_action_from_action_spec(key, self.action_spec)
