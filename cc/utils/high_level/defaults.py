from dataclasses import dataclass
from typing import Optional

import optax

from cc.env import make_env
from cc.env.collect import random_steps_source, sample_feedforward_and_collect
from cc.env.wrappers import AddRefSignalRewardFnWrapper, NoisyObservationsWrapper

two_segments_4min = dict(
    train_gp=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    train_cos=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14],
    val_gp=[15, 16, 17, 18],
    val_cos=[2.5, 5.0, 7.5, 10.0],
)

train_cos = [4, 8, 12, 2, 6, 10, 3, 7, 11, 1, 5, 9, 3.5, 6.5]
# This is from the results of `results/masterplots/4_8.pdf` this one works really well
two_segments_2min = dict(
    train_gp=[0, 1, 2, 3],
    train_cos=[4, 8, 12, 2, 6, 10, 3, 7],
    val_gp=[15, 16, 17, 18],
    val_cos=[2.5, 5.0, 7.5, 10.0],
)

rover = {
    "train_gp": list(range(12)),
    "train_cos": list(range(12)),
    "val_gp": list(range(12, 15)),
    "val_cos": [2.5, 4.5, 6.5],
}

data = {
    "rover": rover,
    "ackermann": rover,
    "two_segments": two_segments_2min,
    "two_segments_v2": two_segments_4min,
}


def optimizer(lr, clip, global_clip):
    opt = optax.chain(
        optax.clip_by_global_norm(global_clip),
        optax.clip(clip),
        optax.adam(lr),
    )
    return opt


@dataclass
class Env:
    env_id: str
    task_kwargs: dict
    physics_kwargs: dict
    noise_level: float = 0.0
    data: Optional[dict] = None

    @property
    def env(self):
        env = make_env(
            self.env_id,
            task_kwargs=self.task_kwargs,
            physics_kwargs=self.physics_kwargs,
        )

        if self.noise_level > 0.0:
            env = NoisyObservationsWrapper(env, self.noise_level)

        return env

    @property
    def train_sample(self):
        data_config = self.data if self.data else data[self.env_id]
        sample = sample_feedforward_and_collect(
            self.env, data_config["train_gp"], data_config["train_cos"]
        )
        return sample

    @property
    def val_sample(self):
        data_config = self.data if self.data else data[self.env_id]
        sample = sample_feedforward_and_collect(
            self.env, data_config["val_gp"], data_config["val_cos"]
        )
        return sample

    @property
    def test_source(self):
        return random_steps_source(self.env, list(range(6)))

    @property
    def env_w_source(self):
        return AddRefSignalRewardFnWrapper(self.env, self.test_source)

    @staticmethod
    def optimizer(lr, clip, global_clip):
        return optimizer(lr, clip, global_clip)
