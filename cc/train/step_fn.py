import functools as ft
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from tree_utils import PyTree, batch_concat, tree_concat

from ..core import AbstractController, AbstractModel
from ..core.module_utils import flatten_module
from ..utils import mse
from .minibatch import (
    Dataloader,
    MiniBatchState,
    SupervisedDatasetWithWeights,
    UnsupervisedDataset,
)

NAME_AND_VALUE = dict[str, float]

# FOR EXAMPLE
# vector of parameters -> {"l2_norm": l2_norm(vector of parameters)}
REGU_FN = Callable[[jnp.ndarray], NAME_AND_VALUE]

# FOR EXAMPLE
# y, yhat -> {"rmse": rmse(y, yhat)}
METRIC_FN = Callable[[PyTree, PyTree], NAME_AND_VALUE]

# FOR EXAMPLE
# y, yhat, weights -> {"train_mse": mse(y, yhat)}
# weights will be a vector of floats with shape (mini-batchsize,)
LOSS_FN_MODEL = Callable[[PyTree, PyTree, jnp.ndarray], NAME_AND_VALUE]

# FOR EXAMPLE
# y, yhat -> {"train_mse": mse(y, yhat)}
LOSS_FN_CONTROLLER = Callable[[PyTree, PyTree], NAME_AND_VALUE]

# FOR EXAMPLE
# see below
LOSS_FN_REDUCE_ALONG_MODELS = Callable[[dict[str, NAME_AND_VALUE]], NAME_AND_VALUE]


def default_loss_fn_reduce_along_models(
    log_of_loss_values: dict[str, NAME_AND_VALUE]
) -> NAME_AND_VALUE:
    flat_logs = batch_concat(log_of_loss_values, 0)
    return {"loss_without_regu": jnp.mean(flat_logs)}


@dataclass
class EvaluationMetrices:
    data: PyTree[jnp.ndarray]
    metrices: tuple[METRIC_FN]


@dataclass
class Regularisation:
    prefactor: float
    reduce_weights: REGU_FN


@dataclass
class TrainingOptionsModel:
    training_data: Dataloader
    optimizer: optax.GradientTransformation
    metrices: tuple[
        EvaluationMetrices
    ] = tuple()  # pytype: disable=annotation-type-mismatch
    regularisers: tuple[
        Regularisation
    ] = tuple()  # pytype: disable=annotation-type-mismatch
    loss_fn: LOSS_FN_MODEL = lambda y, yhat, weights: dict(train_mse=mse(y, yhat))


def compute_regularisers(model_or_controller, regularisers: tuple[Regularisation]):
    if len(regularisers) == 0:
        return 0.0, {}

    flat_module = flatten_module(model_or_controller)

    regu_value, log_of_regus = 0.0, {}
    for regu in regularisers:
        this_value = regu.reduce_weights(flat_module)
        log_of_regus.update(this_value)

        this_value_flat = batch_concat(this_value, 0)
        regu_value += regu.prefactor * this_value_flat
    return regu_value, log_of_regus


def eval_metrices(model, metrices: tuple[EvaluationMetrices]):
    if len(metrices) == 0:
        return {}

    log_of_metrices = {}
    for metric in metrices:
        (inputs, targets) = metric.data  # pytype: disable=attribute-error
        preds = eqx.filter_vmap(model.unroll)(inputs)
        for metric_fn in metric.metrices:
            this_value = metric_fn(preds, targets)
            log_of_metrices.update(this_value)

    return log_of_metrices


def make_step_fn_model(model: AbstractModel, options: TrainingOptionsModel):
    @ft.partial(eqx.filter_value_and_grad, arg=model.grad_filter_spec(), has_aux=True)
    def loss_fn_model(model: AbstractModel, inputs, targets, weights):
        logs = {}
        preds = eqx.filter_vmap(model.unroll)(inputs)
        loss_value = options.loss_fn(targets, preds, weights)
        logs.update(loss_value)

        regu_value, log_of_regus = compute_regularisers(model, options.regularisers)
        logs.update(log_of_regus)

        loss_value = batch_concat(loss_value, 0) + regu_value
        assert loss_value.ndim == 1 or loss_value.ndim == 0

        return jnp.squeeze(loss_value), logs

    optimizer = options.optimizer

    def step_fn_model(model: AbstractModel, opt_state, minibatch_state: MiniBatchState):

        minibatched_logs = []
        for _ in range(minibatch_state.n_minibatches):
            minibatch_state, minibatched_dataset = options.training_data.update_fn(
                minibatch_state
            )
            assert isinstance(minibatched_dataset, SupervisedDatasetWithWeights)
            (loss, logs), grads = loss_fn_model(
                model,
                minibatched_dataset.inputs,
                minibatched_dataset.targets,
                minibatched_dataset.weights,
            )
            logs.update(dict(train_loss=loss))
            minibatched_logs.append(logs)

            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)

        # concat logs
        stacked_logs = tree_concat(minibatched_logs, False, "jax")
        logs = jtu.tree_map(lambda arr: jnp.mean(arr, axis=0), stacked_logs)

        # eval metrices
        log_of_metrices = eval_metrices(model, options.metrices)
        logs.update(log_of_metrices)

        return model, opt_state, minibatch_state, logs

    opt_state = optimizer.init(eqx.filter(model, model.grad_filter_spec()))
    minibatch_state = options.training_data.minibatch_state

    return step_fn_model, opt_state, minibatch_state


def merge_x_y(x, y):
    d = OrderedDict()
    d["ref"] = x
    d["obs"] = y
    return d


@dataclass
class TrainingOptionsController:
    refss: Dataloader  # Batched TimeSeries of References
    optimizer: optax.GradientTransformation
    regularisers: tuple[
        Regularisation
    ] = tuple()  # pytype: disable=annotation-type-mismatch
    merge_x_y: Callable[[PyTree, PyTree], OrderedDict] = merge_x_y
    loss_fn: LOSS_FN_CONTROLLER = lambda y, yhat: dict(train_mse=mse(y, yhat))
    loss_fn_reduce_along_models: LOSS_FN_REDUCE_ALONG_MODELS = (
        default_loss_fn_reduce_along_models
    )


def make_step_fn_controller(
    controller: AbstractController,
    models: dict[str, AbstractModel],
    options: TrainingOptionsController,
):
    @ft.partial(
        eqx.filter_value_and_grad, arg=controller.grad_filter_spec(), has_aux=True
    )
    def loss_fn_controller(controller: AbstractController, refss):
        logs = {}

        log_of_loss_values = {}
        for model_name, model in models.items():
            refsshat = eqx.filter_vmap(controller.unroll(model, merge_x_y))(refss)
            loss_name_and_value = options.loss_fn(refss, refsshat)
            log_of_loss_values.update({model_name: loss_name_and_value})

        regu_value, log_of_regus = compute_regularisers(
            controller, options.regularisers
        )
        logs.update(log_of_regus)

        loss_value = options.loss_fn_reduce_along_models(log_of_loss_values)
        logs.update(loss_value)

        log_of_loss_values_flat = {}
        for model_name, loss_name_and_value in log_of_loss_values.items():
            for loss_name, model_loss_value in loss_name_and_value.items():
                log_of_loss_values_flat.update(
                    {model_name + "_" + loss_name: model_loss_value}
                )
        logs.update(log_of_loss_values_flat)

        loss_value = batch_concat(loss_value, 0) + regu_value
        logs.update({"loss": loss_value})
        assert loss_value.ndim == 0 or loss_value.ndim == 1

        return jnp.squeeze(loss_value), logs

    optimizer = options.optimizer

    def step_fn_controller(
        controller: AbstractController, opt_state, minibatch_state: MiniBatchState
    ):

        minibatched_logs = []
        for i in range(minibatch_state.n_minibatches):
            minibatch_state, batch_of_refss = options.refss.update_fn(minibatch_state)
            assert isinstance(batch_of_refss, UnsupervisedDataset)
            (loss, logs), grads = loss_fn_controller(controller, batch_of_refss.refss)
            logs.update(dict(train_loss=loss))
            minibatched_logs.append(logs)

            updates, opt_state = optimizer.update(grads, opt_state)
            controller = eqx.apply_updates(controller, updates)

        # concat logs
        stacked_logs = tree_concat(minibatched_logs, False, "jax")
        logs = jtu.tree_map(lambda arr: jnp.mean(arr, axis=0), stacked_logs)

        return controller, opt_state, minibatch_state, logs

    opt_state = optimizer.init(eqx.filter(controller, controller.grad_filter_spec()))
    minibatch_state = options.refss.minibatch_state

    return step_fn_controller, opt_state, minibatch_state
