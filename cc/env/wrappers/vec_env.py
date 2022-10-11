import dm_env
import numpy as np 


class VectorizeEnv(dm_env.Environment):
    def __init__(self, envs: list[dm_env.Environment]):
        self.envs = envs 

    def reset(self):
        return list(env.reset() for env in self.envs)

    def step(self, actions: np.ndarray):
        assert actions.ndim == 2 
        return list(env.step(action) for env, action in zip(self.envs, actions))

    def close(self):
        for env in self.envs:
            env.close()
    
    def action_spec(self):
        return list(env.action_spec() for env in self.envs)

    def observation_spec(self):
        return list(env.observation_spec() for env in self.envs)

        