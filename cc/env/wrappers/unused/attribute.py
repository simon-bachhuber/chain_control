import dm_env

from cc.acme.wrappers import EnvironmentWrapper


class AttributeWrapper(EnvironmentWrapper):
    def __init__(self, env: dm_env.Environment, safe=True, **attrs):
        for key, value in attrs.items():
            if hasattr(env, key):
                if safe:
                    raise Exception(
                        f"The environment {env} to be wrapped \
                        already has an attribute `{key}`. This will make this value \
                            unreachable."
                    )
            setattr(self, key, value)
        super().__init__(env)
