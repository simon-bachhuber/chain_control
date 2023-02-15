from .adder import Adder
from .replay_buffer import ReplayBuffer, buffer_to_iterator
from .sampler import Sampler
from ...utils.utils import timestep_array_from_env


def make_episodic_buffer_adder_iterator(
    bs: int,
    env,
    actor_id: int = 0,
    buffer_size_n_trajectories: int = 30,
    force_batch_size: bool = True,
):
    
    ts = timestep_array_from_env(env)

    buffer = ReplayBuffer(
        maxlen=len(ts) * buffer_size_n_trajectories,
        sampler=Sampler(
            env=env,
            length_of_trajectories=len(ts),
            episodic=True,
            sample_with_replacement=False,
        ),
    )

    iterator = buffer_to_iterator(buffer, bs, force_batch_size)

    adder = Adder(buffer, flush_cache_every=len(ts))
    adder.set_actor_id(actor_id)
    return buffer, adder, iterator
