import dm_env
from acme.wrappers import EnvironmentWrapper


class TrackTimeWrapper(EnvironmentWrapper):
    def __init__(self, environment: dm_env.Environment):
        super().__init__(environment)
        self.requires_reset = True

    def reset(self):
        self.requires_reset = False
        self.i_timestep = 0
        self.t = 0
        return super().reset()

    def step(self, action) -> dm_env.TimeStep:

        if self.requires_reset:
            return self.reset()

        self.i_timestep += 1
        self.t += self.control_timestep

        ts = super().step(action)

        if ts.last():
            self.requires_reset = True

        return ts
