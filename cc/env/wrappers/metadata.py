import dm_env
from acme.wrappers import EnvironmentWrapper


class MetaDataWrapper(EnvironmentWrapper):
    def __init__(self, env: dm_env.Environment, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().__init__(env)

        