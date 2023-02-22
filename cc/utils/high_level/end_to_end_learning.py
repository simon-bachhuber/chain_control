import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import optax

from cc.env.collect import random_steps_source, sample_feedforward_and_collect
from cc.env.wrappers import AddRefSignalRewardFnWrapper
from cc.examples.neural_ode_controller_compact_example import make_neural_ode_controller
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model
from cc.train import (
    DictLogger,
    EvaluationMetrices,
    ModelControllerTrainer,
    Regularisation,
    SupervisedDataset,
    Tracker,
    TrainingOptionsController,
    TrainingOptionsModel,
    UnsupervisedDataset,
    make_dataloader,
)
from cc.utils import l1_norm, l2_norm, rmse
from cc.utils.high_level import (
    build_extra_sources,
    loop_observer_configs,
    masterplot_siso,
)

from .baselines.data import data


def make_model(
    env,
    sample_train,
    sample_val,
    model_kwargs: dict,
    seed_model: int,
    n_steps: int,
    lambda_l1_norm: float = 0.0,
    lambda_l2_norm: float = 0.0,
    optimizer=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)),
):
    model = make_neural_ode_model(
        env.action_spec(),
        env.observation_spec(),
        env.control_timestep(),
        key=jrand.PRNGKey(seed_model),
        **model_kwargs,
    )

    model_train_dataloader = make_dataloader(
        SupervisedDataset(sample_train.action, sample_train.obs),  # <- (X, y)
        n_minibatches=4,
        do_bootstrapping=True,
    )

    regularisers = (
        Regularisation(
            prefactor=lambda_l1_norm,
            reduce_weights=lambda vector_of_params: {
                "l1_norm": l1_norm(vector_of_params)
            },
        ),
        Regularisation(
            prefactor=lambda_l2_norm,
            reduce_weights=lambda vector_of_params: {
                "l2_norm": l2_norm(vector_of_params)
            },
        ),
    )

    metrices = (
        EvaluationMetrices(
            data=(sample_val.action, sample_val.obs),  # <- (X, y)
            metrices=(lambda y, yhat: {"val_rmse": rmse(y, yhat)},),
        ),
    )

    model_train_options = TrainingOptionsModel(
        model_train_dataloader, optimizer, regularisers=regularisers, metrices=metrices
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
    seed_controller: int,
    training_step_source_amplitude: float,
    controller_kwargs: dict,
    optimizer,
    n_steps: int,
    lambda_l1,
    lambda_l2,
    noise_scale=None,
):

    controller = make_neural_ode_controller(
        env_w_source.observation_spec(),
        env_w_source.action_spec(),
        env_w_source.control_timestep(),
        key=jrand.PRNGKey(seed_controller),
        **controller_kwargs,
    )

    controller_dataloader = make_dataloader(
        UnsupervisedDataset(
            random_steps_source(env, list(range(30))).get_references_for_optimisation()
        ),
        n_minibatches=5,
        tree_transform=tree_transform(training_step_source_amplitude),
    )

    regularisers = (
        Regularisation(
            prefactor=lambda_l1,
            reduce_weights=lambda vector_of_params: {
                "l1_norm": l1_norm(vector_of_params)
            },
        ),
        Regularisation(
            prefactor=lambda_l2,
            reduce_weights=lambda vector_of_params: {
                "l2_norm": l2_norm(vector_of_params)
            },
        ),
    )

    controller_train_options = TrainingOptionsController(
        controller_dataloader,
        optimizer,
        regularisers=regularisers,
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

    return controller_trainer


default_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))


def make_masterplot(
    env_id,
    env,
    filename,
    record_video,
    experiment_id,
    model_kwargs,
    seed_model,
    n_steps_model,
    controller_kwargs,
    seed_controller,
    n_steps_controller,
    noise_scale_controller=None,
    model_optimizer=default_optimizer,
    controller_optimizer=default_optimizer,
) -> float:
    training_step_source_amplitude = 3.0
    if env_id == "muscle_asymmetric":
        training_step_source_amplitude = np.deg2rad(60.0)

    dc = data[env_id]
    train_sample = sample_feedforward_and_collect(env, dc["train_gp"], dc["train_cos"])
    val_sample = sample_feedforward_and_collect(env, dc["val_gp"], dc["val_cos"])

    test_source = random_steps_source(
        env, list(range(6)), training_step_source_amplitude
    )
    env_w_source = AddRefSignalRewardFnWrapper(env, test_source)

    lambda_l1_norm = model_kwargs.pop("lambda_l1_norm", 0.0)
    lambda_l2_norm = model_kwargs.pop("lambda_l2_norm", 0.0)
    model_trainer = make_model(
        env,
        train_sample,
        val_sample,
        model_kwargs,
        seed_model,
        n_steps_model,
        lambda_l1_norm,
        lambda_l2_norm,
        model_optimizer,
    )
    model = model_trainer.trackers[0].best_model_or_controller()

    lambda_l1_norm = controller_kwargs.pop("lambda_l1_norm", 0.0)
    lambda_l2_norm = controller_kwargs.pop("lambda_l2_norm", 0.0)
    controller_trainer = make_controller(
        env,
        env_w_source,
        model,
        seed_controller,
        training_step_source_amplitude,
        controller_kwargs,
        controller_optimizer,
        n_steps_controller,
        lambda_l1_norm,
        lambda_l2_norm,
        noise_scale_controller,
    )
    controller = controller_trainer.trackers[0].best_model_or_controller()

    results = masterplot_siso(
        env,
        test_source,
        controller,
        filename,
        build_extra_sources(env_id, record_video),
        controller_trainer.get_logs()[0],
        controller_trainer.get_tracker_logs()[0],
        model,
        model_trainer.get_logs()[0],
        model_trainer.get_tracker_logs()[0],
        train_sample,
        [0, 1, 4, 5],
        val_sample,
        [0, 2, 3],
        experiment_id=experiment_id,
        loop_observer_config=loop_observer_configs[env_id],
    )

    return results
