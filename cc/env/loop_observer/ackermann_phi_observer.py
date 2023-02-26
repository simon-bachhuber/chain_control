from typing import Dict

import dm_env
import numpy as np

from .observer import EnvLoopObserver


class AckermannPhiObserver(EnvLoopObserver):
    def __init__(self) -> None:
        self._phi = []  # for pytype ..

    def _append_phi(self, env):
        self._phi.append(env.physics._theta.copy())

    def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep) -> None:
        # reset
        self._phi = []
        self._append_phi(env)

    def observe(
        self, env: dm_env.Environment, timestep: dm_env.TimeStep, action: np.ndarray
    ) -> None:
        self._append_phi(env)

    def get_metrics(self) -> Dict[str, np.ndarray]:
        return {
            "phi [deg]": np.rad2deg(np.array(self._phi)),
        }
