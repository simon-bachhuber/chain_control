from .source import (
    draw_u_from_cosines,
    constant_after_transform_source,
    draw_u_from_gaussian_process,
    ObservationReferenceSource
)
from ..buffer import ReplaySample, make_episodic_buffer_adder_iterator
from ..types import *
from ..utils import tree_concat, extract_timelimit_timestep_from_env, tree_shape, to_jax, to_numpy
from ..env.wrappers.delay import DelayActionWrapper
from ..env.wrappers.add_source import AddReferenceObservationWrapper
from .actor import PolicyActor
from ..env import ModelBasedEnv
from ..abstract import AbstractModel, AbstractController
from ..controller import FeedforwardController


from beartype import beartype
import dm_env
from acme import make_environment_spec, EnvironmentLoop
from acme.utils import loggers
from tqdm.auto import tqdm 
from ..config import use_tqdm



def concat_iterators(*iterators) -> ReplaySample:
    return tree_concat([next(iterator) for iterator in iterators], True)


def _checks_and_infos(env):
    assert not isinstance(env, ModelBasedEnv)
    time_limit, control_timestep, ts = extract_timelimit_timestep_from_env(env)
    return time_limit, control_timestep, ts


@beartype
def collect_sample(env: dm_env.Environment, 
    seeds_gp: list[int], seeds_cos: list[Union[int, float]]) -> ReplaySample:

    _, _, ts = _checks_and_infos(env)

    _, sample_gp = collect_feedforward_and_make_source(env, ts, seeds=seeds_gp)
    _, sample_cos = collect_feedforward_and_make_source(env, ts, draw_u_from_cosines, seeds=seeds_cos)

    return tree_concat([sample_gp, sample_cos], True)


@beartype
def collect_reference_source(env: dm_env.Environment, seeds: list[int], 
    constant_after: bool = False, constant_after_T: float = 3.0,
):

    _, _, ts = _checks_and_infos(env)
    
    source, _ = collect_feedforward_and_make_source(env, ts, seeds=seeds)

    if constant_after:
        source = constant_after_transform_source(source, constant_after_T)

    return source 


@beartype
def collect_exhaust_source(
    env: dm_env.Environment, 
    source: ObservationReferenceSource,
    controller: AbstractController,
    model: AbstractModel = None, 
    ) -> ReplaySample:

    time_limit, control_timestep, ts = _checks_and_infos(env)

    if model:

        # grab delay of Mujoco Env before overwriting
        delay = env.delay 

        env = ModelBasedEnv(env, model, time_limit=time_limit, control_timestep=control_timestep) 
        
        if delay>0:
            env = DelayActionWrapper(env, delay)

    # wrap env with source
    env = AddReferenceObservationWrapper(env, source)

    N = tree_shape(source._yss)
    # collect performance of controller in environment
    pbar = tqdm(range(N), desc="Reference Iterator", disable= not use_tqdm())
    iterators = []
    for i_actor in pbar:
        source.change_reference_of_actor(i_actor)
        iterator = collect(env, controller, ts)
        iterators.append(iterator)

    # concat samples 
    sample = concat_iterators(*iterators)

    return sample 


def collect(env: dm_env.Environment, controller: AbstractController, ts: jnp.ndarray):

    env.reset()

    _, adder, iterator = make_episodic_buffer_adder_iterator(
        bs=1,
        ts=ts,
        env_specs=make_environment_spec(env),
        actor_id=1,
    )

    actor = PolicyActor(policy=controller, action_spec=env.action_spec(), adder=adder)
    loop = EnvironmentLoop(env, actor, logger=loggers.NoOpLogger())
    loop.run_episode()
    return iterator 


def collect_feedforward_and_make_source(
    env: dm_env.Environment,
    ts: jnp.ndarray,
    draw_fn = draw_u_from_gaussian_process,
    seeds: list[int] = [0,],
) -> Tuple[ObservationReferenceSource, ReplaySample]:

    iterators = []
    for seed in seeds:
        us: TimeSeriesOfAct = to_jax(draw_fn(to_numpy(ts), seed=seed))
        policy = FeedforwardController(us)
        iterator = collect(env, policy, ts)
        iterators.append(iterator)
    
    sample = concat_iterators(*iterators)
    source = ObservationReferenceSource(ts, sample.obs, sample.action)
    return source, sample 

