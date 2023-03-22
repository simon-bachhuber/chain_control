from collections import deque

import dm_env
import numpy as np

from cc.acme.wrappers import EnvironmentWrapper


class DelayActionWrapper(EnvironmentWrapper):
    def __init__(self, env: dm_env.Environment, delay: int):
        self.delay = delay
        super().__init__(env)
        self.reset()

    def _init_deque(self, action: np.ndarray):
        zeros_like = np.zeros_like(action)
        for _ in range(self.deque.maxlen):
            self.deque.append(zeros_like)
        self.init_deque = False

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        if self.init_deque:
            self._init_deque(action)

        self.deque.append(action)
        action = self.deque.popleft()

        return super().step(action)

    def reset(self):
        self.init_deque = True
        self.deque = deque(maxlen=self.delay + 1)
        return super().reset()
