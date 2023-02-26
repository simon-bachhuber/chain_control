"""Ackermann Steering Dynamics. Taken from
https://www.xarg.org/book/kinematics/ackerman-steering/
"""

from collections import OrderedDict

import mujoco
import numpy as np
from dm_control.rl import control
from dm_env import specs

from ...utils.sample_from_spec import _spec_from_observation

V = 3  # Velocity of the vehicle.
L = 4.0  # Length of the car
W = 1.5  # Width of the car
MAX_STEER = np.deg2rad(30)


class Physics(control.Physics):
    def __init__(
        self,
        V: float = V,
        L: float = L,
        W: float = W,
        MAX_STEER: float = MAX_STEER,
        use_tanh: bool = True,
    ):
        self._V = V
        self._L = L
        self._W = W
        self._MAX_STEER = MAX_STEER
        self._use_tanh = use_tanh
        self.reset()

    def _step(self):
        phi_outer = self._steering_rad

        # solve for `r`
        r = self._L / np.tan(phi_outer + 1e-8) - self._W / 2

        # calculate `phi`
        phi = np.arctan(self._L / r)

        # advance dynamics
        dt = self.timestep()
        self._theta += dt * (self._V / self._L) * np.tan(phi)
        self._xpos += dt * self._V * np.cos(self._theta)
        self._ypos += dt * self._V * np.sin(self._theta)

    def step(self, n_sub_steps=1):
        # TODO
        # ensure that this is really correct
        # i am a little unsure about the `n_sub_steps`
        # why whould dm_control then not expect me to implement
        # `self._step` directly and provide the `.step` logic
        for _ in range(n_sub_steps):
            self._step()
            self._time += self.timestep()

    def reset(self):
        self._time = 0.0
        self._xpos = 0.0
        self._ypos = 0.0
        self._steering_rad = 0.0
        self._theta = np.array([0.0])

    def time(self):
        return self._time

    def timestep(self):
        return 0.01

    def set_control(self, control):
        if self._use_tanh:
            control = np.tanh(control) * self._MAX_STEER
        else:
            control = np.clip(control, -self._MAX_STEER, self._MAX_STEER)
        self._steering_rad = control

    def after_reset(self):
        pass


class Task(control.Task):
    def initialize_episode(self, physics):
        pass

    def before_step(self, action, physics: Physics):
        action = np.atleast_1d(action)
        physics.set_control(action)

    def after_step(self, physics):
        pass

    def action_spec(self, physics):
        return specs.BoundedArray(
            (1,),
            dtype=float,
            minimum=-mujoco.mjMAXVAL,
            maximum=mujoco.mjMAXVAL,
            name="steering_wheel_unitless",
        )

    def get_observation(self, physics: Physics) -> OrderedDict:
        obs = OrderedDict()
        obs["y_position_of_car"] = np.atleast_1d(physics._ypos).copy()
        return obs

    def get_reward(self, physics):
        return np.array(0.0)

    def observation_spec(self, physics):
        return _spec_from_observation(self.get_observation(physics))
