from typing import Any, Dict

from dm_control.rl.control import Environment
import dm_env
import numpy as np
import tree_utils

from ...acme.utils.observers import EnvLoopObserver


class DenoisifyObserver(EnvLoopObserver):
    """Tracks the observation directly from the physics object. They are not noisy."""

    def __init__(self) -> None:
        self._obs = []

    def _obs_from_task(self, env: Environment):
        self._obs.append(env.task.get_observation(env.physics))

    def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep) -> None:
        self._obs = []
        self._obs_from_task(env)

    def observe(
        self, env: dm_env.Environment, timestep: dm_env.TimeStep, action: np.ndarray
    ) -> None:
        self._obs_from_task(env)

    def get_metrics(self) -> Dict[str, Any]:
        return {"no_noise_obs": tree_utils.tree_batch(self._obs)}
