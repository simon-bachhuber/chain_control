from dm_control.rl.control import Environment
import dm_env
import numpy as np

from cc.acme.wrappers import EnvironmentWrapper


class NoisyActionsWrapper(EnvironmentWrapper):
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

    def reset(self, reset_seed: bool = False) -> dm_env.TimeStep:
        if reset_seed or self.reset_seed:
            np.random.seed(self.seed)

        return super().reset()

    def step(self, action) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()
        action = np.atleast_1d(action)
        noise = np.random.normal(scale=self.noise_scale, size=action.shape)
        action += noise.astype(action.dtype)
        return super().step(action)
