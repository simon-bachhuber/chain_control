import dm_env
import jax
import numpy as np
from acme.wrappers import EnvironmentWrapper
from dm_control.rl.control import Environment


class NoisyObservationsWrapper(EnvironmentWrapper):
    def __init__(
        self,
        environment: Environment,
        noise_scale: float = 1.0,
        seed: int = 1,
        reset_seed: bool = False,
    ):
        super().__init__(environment)
        self.seed = seed
        self.noise_scale = noise_scale
        self.reset_seed = reset_seed
        np.random.seed(self.seed)

    def _add_noise_to_ts(self, ts: dm_env.TimeStep):
        def add_noise(arr):
            noise = np.random.normal(scale=self.noise_scale, size=arr.shape)
            return arr + noise.astype(arr.dtype)

        return ts._replace(observation=jax.tree_map(add_noise, ts.observation))

    def reset(self, reset_seed: bool = False) -> dm_env.TimeStep:
        if reset_seed or self.reset_seed:
            np.random.seed(self.seed)

        ts = super().reset()
        return self._add_noise_to_ts(ts)

    def step(self, action) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()
        ts = super().step(action)
        return self._add_noise_to_ts(ts)
