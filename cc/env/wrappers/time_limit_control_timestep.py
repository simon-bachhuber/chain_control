import dm_env
import numpy as np
from acme.wrappers import EnvironmentWrapper


class TimelimitControltimestepWrapper(EnvironmentWrapper):
    def __init__(
        self,
        env: dm_env.Environment,
        time_limit: float,
        control_timestep: float,
        delay: int = 0,
    ):
        super().__init__(env)
        self.time_limit = time_limit
        self.control_timestep = control_timestep
        self.ts = np.arange(time_limit, step=control_timestep)
        self.delay = delay
