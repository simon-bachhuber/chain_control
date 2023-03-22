from typing import Iterable

import dm_env
import jax
import numpy as np
from tree_utils import tree_insert_IMPURE, tree_zeros_like

from ...utils.utils import timestep_array_from_env
from .replay_element_sample import ReplayElement, ReplaySample


class _TransitionAccumulation:
    def __init__(self, n_transitions=None, episodic=None):
        self.n_transitions = n_transitions
        self.episodic = episodic
        self.reset()

    def first_transition(self, ele: ReplayElement) -> bool:
        if self.episodic:
            if ele.timestep.first():
                return True
            else:
                return False

        self._i += 1
        if self.n_transitions:
            if self._i >= self.n_transitions:
                return True
            else:
                return False

    def reset(self):
        self._i = 0


class Sampler:
    def __init__(
        self,
        env: dm_env.Environment,
        length_of_trajectories: int,
        toy_extras: dict = {},
        episodic: bool = False,
        sample_with_replacement: bool = True,
    ):
        self._episodic = episodic
        self._length_of_trajectories = length_of_trajectories
        self._dtype_dones = np.float32

        self._extras_specs = tree_zeros_like(toy_extras)
        self._obs_specs = env.observation_spec()
        self._action_specs = env.action_spec()
        self._reward_specs = env.reward_spec()

        self._transition_accumulation = _TransitionAccumulation(
            length_of_trajectories, episodic
        )

        self._sample_with_replacement = sample_with_replacement

        if episodic:
            self.n = len(timestep_array_from_env(env))
        else:
            self.n = self._length_of_trajectories

    def _preallocate(self, bs, n) -> ReplaySample:
        map = jax.tree_util.tree_map

        obs = tree_zeros_like(self._obs_specs)
        # n+1 because of initial state after env.reset()
        obs = map(lambda arr: np.zeros((bs, n + 1, *arr.shape), arr.dtype), obs)

        # pre-allocate extras
        extras = map(
            lambda arr: np.zeros((bs, n, *arr.shape), arr.dtype), self._extras_specs
        )

        return (
            obs,  # obs
            np.zeros(
                (bs, n, *self._action_specs.shape), self._action_specs.dtype
            ),  # action
            np.zeros((bs, n, 1), self._reward_specs.dtype),  # rewards
            np.zeros((bs, n, 1), self._dtype_dones),  # done
            extras,  # extras
        )

    def update_weights_when_sampling(self, weights: np.ndarray) -> np.ndarray:
        return weights

    def update_weights_when_inserting(self, weigths: np.ndarray) -> np.ndarray:
        return weigths

    def draw_idxs_from_weights(
        self, weights: np.ndarray, dones: np.ndarray, bs: int
    ) -> Iterable[int]:
        # sample only ReplayElements that are at the last timestep
        if self._episodic:
            weights *= dones

            if not self._sample_with_replacement:
                idxs = np.where(weights[:, 0])[0]
                if len(idxs) <= bs:
                    return idxs

        # convert to probabilities
        probs = weights / np.sum(weights)

        idxs = np.random.choice(
            np.arange(len(probs)),
            size=bs,
            replace=self._sample_with_replacement,
            p=probs[:, 0],
        )
        return idxs

    def sample(self, samples: list[ReplayElement], bs=None) -> ReplaySample:
        if bs is None:
            bs = len(samples)

        (
            alloc_obs,
            alloc_action,
            alloc_rew,
            alloc_done,
            alloc_extras,
        ) = self._preallocate(bs, self.n)

        for i, sample in enumerate(samples):
            trajectory = []
            while True:
                trajectory.append(sample)

                if self._transition_accumulation.first_transition(sample):
                    trajectory_long_enough = True
                    break

                if sample.prev is None:
                    trajectory_long_enough = False
                    break

                sample = sample.prev

            self._transition_accumulation.reset()

            if not trajectory_long_enough:
                continue

            # we started at the last timestep, so time is reversed
            trajectory.reverse()

            for t, sample in enumerate(trajectory):
                tree_insert_IMPURE(alloc_extras, sample.extras, (i, t))

                tree_insert_IMPURE(alloc_obs, sample.timestep.observation, (i, t))
                # save final `next_obs`
                if t == (self.n - 1):
                    tree_insert_IMPURE(
                        alloc_obs, sample.next_timestep.observation, (i, self.n)
                    )

                alloc_action[i, t] = sample.action
                alloc_rew[i, t] = sample.next_timestep.reward
                alloc_done[i, t] = float(sample.next_timestep.last())

        return ReplaySample(
            alloc_obs, alloc_action, alloc_rew, alloc_done, alloc_extras
        )
