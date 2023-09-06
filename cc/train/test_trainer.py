import jax.numpy as jnp
import jax.random as jrand
import optax

from cc.env import make_env
from cc.env.collect import random_steps_source
from cc.env.collect import sample_feedforward_and_collect
from cc.env.wrappers import AddRefSignalRewardFnWrapper
from cc.examples.neural_ode_controller_compact_example import \
    make_neural_ode_controller
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model
from cc.train import DictLogger
from cc.train import EvaluationMetrices
from cc.train import make_dataloader
from cc.train import ModelControllerTrainer
from cc.train import Regularisation
from cc.train import SupervisedDataset
from cc.train import Tracker
from cc.train import TrainingOptionsController
from cc.train import TrainingOptionsModel
from cc.train import UnsupervisedDataset
from cc.utils import l2_norm
from cc.utils import rmse


def test_trainer():
    time_limit = 10.0
    control_timestep = 0.01

    env = make_env(
        "two_segments_v1",
        time_limit=time_limit,
        control_timestep=control_timestep,
        random=1,
    )

    sample_train = sample_feedforward_and_collect(env, seeds_gp=[0], seeds_cos=[1])

    sample_val = sample_feedforward_and_collect(env, seeds_gp=[15], seeds_cos=[2.5])

    model = make_neural_ode_model(
        env.action_spec(),
        env.observation_spec(),
        env.control_timestep(),
        state_dim=1,
        f_depth=0,
        u_transform=jnp.arctan,
    )

    model_train_dataloader = make_dataloader(
        SupervisedDataset(sample_train.action, sample_train.obs),  # <- (X, y)
        jrand.PRNGKey(
            2,
        ),
        n_minibatches=1,
    )

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-3))

    regularisers = (
        Regularisation(
            prefactor=0.5,
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
    model_trainer.run(1)
    fitted_model = model_trainer.trackers[0].best_model_or_controller()

    source = random_steps_source(env, seeds=list(range(1)))
    env_w_source = AddRefSignalRewardFnWrapper(env, source)
    controller = make_neural_ode_controller(
        env_w_source.observation_spec(),
        env.action_spec(),
        env.control_timestep(),
        1,
        f_depth=0,
    )

    controller_dataloader = make_dataloader(
        UnsupervisedDataset(source.get_references_for_optimisation()),
        jrand.PRNGKey(
            1,
        ),
        n_minibatches=1,
    )

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))

    controller_train_options = TrainingOptionsController(
        controller_dataloader,
        optimizer,
    )

    controller_trainer = ModelControllerTrainer(
        fitted_model,
        controller,
        controller_train_options=controller_train_options,
        trackers=[Tracker("model0", "train_mse")],
    )
    controller_trainer.run(1)

    model2 = make_neural_ode_model(
        env.action_spec(),
        env.observation_spec(),
        env.control_timestep(),
        state_dim=2,
        f_depth=0,
        u_transform=jnp.arctan,
    )

    # test that multiple models can be used
    controller_trainer = ModelControllerTrainer(
        dict(m1=fitted_model, m2=model2),
        controller,
        controller_train_options=controller_train_options,
        trackers=[Tracker("m2", "train_mse")],
    )
    controller_trainer.run(1)

    # test that multiple models can be used
    controller_trainer = ModelControllerTrainer(
        dict(m1=fitted_model, m2=model2),
        controller,
        controller_train_options=controller_train_options,
        trackers=[Tracker("loss_without_regu")],
    )
    controller_trainer.run(1)
