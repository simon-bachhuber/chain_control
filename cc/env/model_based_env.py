from ..types import *
import numpy as np 
import dm_env
from ..abstract import AbstractModel
from ..utils import to_numpy


from beartype import beartype 


_default_discount = np.array(1.0, dtype=np.float32)
_default_reward = np.array(0.0)

class ModelBasedEnv(dm_env.Environment):
    def __init__(self, 
        env: dm_env.Environment, 
        model: AbstractModel, 
        process_obs: Callable = lambda obs: obs,
        time_limit: Optional[float] = None,
        control_timestep: float = 0.01, 
        use_env_initial_obs: bool = False,
        ):

        self._env = env 
        self._model = model 
        self._process_obs = process_obs

        self._time_limit = time_limit
        self._control_timestep = control_timestep

        self._use_env_initial_obs = use_env_initial_obs

        self.reset()

    def reset(self):

        # reset time
        self.t = 0.0

        # reset model state
        self._model = self._model.reset()

        if self._use_env_initial_obs:
            obs0 = self._env.reset().observation
        else:
            obs0 = to_numpy(self._model.y0())
        
        return self._build_timestep(dm_env.StepType.FIRST, obs0)

    @beartype
    def _build_timestep(self, step_type: dm_env.StepType, obs: Observation) -> dm_env.TimeStep:

        # envs return OrderedDict of numpy-values
        obs = self._process_obs(to_numpy(obs))

        rew = _default_reward
        # dm_env convention is that the first timestep
        # has no reward, i.e. it is None
        if step_type == dm_env.StepType.FIRST:
            rew = None 

        return dm_env.TimeStep(step_type, rew, _default_discount, obs)

    def step(self, action: Union[list, np.ndarray]) -> dm_env.TimeStep:

        if self._time_limit is None or \
            self.t < (self._time_limit-self._control_timestep-1e-8):

            step_type = dm_env.StepType.MID 
        else:
            step_type = dm_env.StepType.LAST

        # model is a jax function, so convert to DeviceArray
        action = jnp.array(action)
        assert action.ndim == 1

        self._model, obs = eqx.filter_jit(self._model)(action)

        # count up pseudo-time
        self.t += self._control_timestep

        return self._build_timestep(step_type, obs)

    def close(self):
        self._env.close()

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        return self._env.observation_spec()
