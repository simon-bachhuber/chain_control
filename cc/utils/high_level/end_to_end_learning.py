import os

import jax
import jax.numpy as jnp
import jax.random as jrand
import optax

from cc.acme.utils.paths import process_path
from cc.core import save_eqx
from cc.env.collect import random_steps_source
from cc.examples.neural_ode_controller_compact_example import make_neural_ode_controller
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model
from cc.train import DictLogger
from cc.train import EvaluationMetrices
from cc.train import l1_l2_regularisers
from cc.train import make_dataloader
from cc.train import ModelControllerTrainer
from cc.train import SupervisedDataset
from cc.train import Tracker
from cc.train import TrainingOptionsController
from cc.train import TrainingOptionsModel
from cc.train import UnsupervisedDataset
from cc.utils import rmse
from cc.utils import split_filename
from cc.utils.high_level import build_extra_sources
from cc.utils.high_level import loop_observer_configs
from cc.utils.high_level import masterplot_siso

from .defaults import Env


def make_model(
    env,
    train_sample,
    val_sample,
    model_kwargs: dict,
    n_steps: int,
    lambda_l1: float,
    lambda_l2: float,
    optimizer,
    seed,
):
    model = make_neural_ode_model(
        env.action_spec(),
        env.observation_spec(),
        env.control_timestep(),
        key=jrand.PRNGKey(seed),
        **model_kwargs,
    )

    model_train_dataloader = make_dataloader(
        SupervisedDataset(train_sample.action, train_sample.obs),  # <- (X, y)
        n_minibatches=4,
        do_bootstrapping=True,
    )

    metrices = (
        EvaluationMetrices(
            data=(val_sample.action, val_sample.obs),  # <- (X, y)
            metrices=(lambda y, yhat: {"val_rmse": rmse(y, yhat)},),
        ),
    )

    model_train_options = TrainingOptionsModel(
        model_train_dataloader,
        optimizer,
        regularisers=l1_l2_regularisers(lambda_l1, lambda_l2),
        metrices=metrices,
    )

    model_trainer = ModelControllerTrainer(
        model,
        model_train_options=model_train_options,
        loggers=[DictLogger()],
        trackers=[Tracker("val_rmse")],
    )

    model_trainer.run(n_steps)

    return model_trainer


def tree_transform(bound: float = 3.0):
    upper_bound = bound
    lower_bound = -bound

    @jax.vmap
    def _random_step(ref, key):
        return jnp.ones_like(ref) * jrand.uniform(
            key, (), minval=lower_bound, maxval=upper_bound
        )

    def _tree_transform(key, ref, bs):
        keys = jrand.split(key, bs)
        return jax.tree_map(lambda ref: _random_step(ref, keys), ref)

    return _tree_transform


def make_controller(
    env,
    env_w_source,
    model,
    training_step_source_amplitude: float,
    controller_kwargs: dict,
    optimizer,
    n_steps: int,
    lambda_l1,
    lambda_l2,
    noise_scale,
    seed,
):
    obs_spec = env_w_source.observation_spec()
    act_spec = env_w_source.action_spec()
    control_timestep = env_w_source.control_timestep()
    init_fn = lambda: make_neural_ode_controller(
        obs_spec,
        act_spec,
        control_timestep,
        key=jrand.PRNGKey(seed),
        **controller_kwargs,
    )
    controller = init_fn()

    controller_dataloader = make_dataloader(
        UnsupervisedDataset(
            random_steps_source(env, list(range(30))).get_references_for_optimisation()
        ),
        n_minibatches=5,
        tree_transform=tree_transform(training_step_source_amplitude),
    )

    controller_train_options = TrainingOptionsController(
        controller_dataloader,
        optimizer,
        regularisers=l1_l2_regularisers(lambda_l1, lambda_l2),
        noise_scale=noise_scale,
    )
    controller_trainer = ModelControllerTrainer(
        model,
        controller,
        controller_train_options=controller_train_options,
        trackers=[Tracker("loss")],
        loggers=[DictLogger()],
    )
    controller_trainer.run(n_steps)

    return controller_trainer, init_fn


default_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))


def make_masterplot(
    env: Env,
    filename=None,
    record_video=False,
    path: str = "~/chain_control",
    experiment_id="",
    model_kwargs={},
    model_n_steps: int = 700,
    model_optimizer=default_optimizer,
    model_l1: float = 0.0,
    model_l2: float = 0.0,
    model_seed: int = 1,
    controller_kwargs={},
    controller_n_steps: int = 700,
    controller_optimizer=default_optimizer,
    controller_l1: float = 0.0,
    controller_l2: float = 0.0,
    controller_seed: int = 1,
    controller_noise_scale=None,
    controller_training_step_ampl: float = 3.0,
    dump_controller: bool = False,
    controller_use_tracker: bool = True,
) -> float:
    train_sample = env.train_sample
    val_sample = env.val_sample

    model_trainer = make_model(
        env.env,
        train_sample,
        val_sample,
        model_kwargs,
        model_n_steps,
        model_l1,
        model_l2,
        model_optimizer,
        model_seed,
    )
    model = model_trainer.trackers[0].best_model_or_controller()

    controller_trainer, init_fn = make_controller(
        env.env,
        env.env_w_source,
        model,
        controller_training_step_ampl,
        controller_kwargs,
        controller_optimizer,
        controller_n_steps,
        controller_l1,
        controller_l2,
        controller_noise_scale,
        controller_seed,
    )
    if controller_use_tracker:
        controller = controller_trainer.trackers[0].best_model_or_controller()
    else:
        controller = controller_trainer._controller

    results = masterplot_siso(
        env.env,
        env.test_source,
        controller,
        filename,
        build_extra_sources(env.env_id, record_video),
        controller_trainer.get_logs()[0],
        controller_trainer.get_tracker_logs()[0],
        model,
        model_trainer.get_logs()[0],
        model_trainer.get_tracker_logs()[0],
        train_sample,
        [0, 1, 2, 3],
        val_sample,
        [0, 2, 4, 5],
        path=path,
        experiment_id=experiment_id,
        loop_observer_config=loop_observer_configs[env.env_id],
    )

    if dump_controller:
        path_folder = process_path(
            path, experiment_id, "fitted_controllers", add_uid=False
        )
        path_controller = os.path.join(path_folder, split_filename(filename)[0])
        save_eqx(path_controller, controller, init_fn)

    return results
