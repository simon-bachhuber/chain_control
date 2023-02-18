from typing import Dict

import dm_env
import numpy as np

from .observer import EnvLoopObserver


class AnglesEnvLoopObserver(EnvLoopObserver):
    def __init__(self) -> None:
        self._hinge_1, self._hinge_2 = [], []  # for pytype ..

    def _get_hinges_and_append(self, env):
        hinge1, hinge2 = env.physics.named.data.qpos[["cart_hinge_1", "cart_hinge_2"]]
        self._hinge_1.append(hinge1)
        self._hinge_2.append(hinge2)

    def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep) -> None:
        # reset
        self._hinge_1, self._hinge_2 = [], []

        self._get_hinges_and_append(env)

    def observe(
        self, env: dm_env.Environment, timestep: dm_env.TimeStep, action: np.ndarray
    ) -> None:
        self._get_hinges_and_append(env)

    def get_metrics(self) -> Dict[str, np.ndarray]:
        return {
            "hinge_1 [deg]": np.rad2deg(np.array(self._hinge_1)),
            "hinge_2 [deg]": np.rad2deg(np.array(self._hinge_2)),
        }
