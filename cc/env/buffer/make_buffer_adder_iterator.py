from ...env.wrappers import TimelimitControltimestepWrapper
from .adder import Adder
from .replay_buffer import ReplayBuffer, buffer_to_iterator
from .sampler import Sampler


def make_episodic_buffer_adder_iterator(
    bs: int,
    env: TimelimitControltimestepWrapper,
    actor_id: int = 0,
    buffer_size_n_trajectories: int = 30,
    force_batch_size: bool = True,
):

    buffer = ReplayBuffer(
        maxlen=len(env.ts) * buffer_size_n_trajectories,
        sampler=Sampler(
            env=env,
            length_of_trajectories=len(env.ts),
            episodic=True,
            sample_with_replacement=False,
        ),
    )

    iterator = buffer_to_iterator(buffer, bs, force_batch_size)

    adder = Adder(buffer, flush_cache_every=len(env.ts))
    adder.set_actor_id(actor_id)
    return buffer, adder, iterator
