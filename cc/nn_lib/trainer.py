import functools as ft
from collections import deque

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from tqdm.auto import tqdm

from ..config import use_tqdm
from ..core import Module, make_module_from_function
from ..utils import batch_concat, l2_norm, mse, to_numpy, tree_concat
from .module_utils import filter_module, flatten_module


def vmap_module_eval_and_reduce(module, inputs, targets=None, reduce_fn=None):
    _, preds = eqx.filter_vmap(module)(inputs)
    if targets is None:
        return preds

    # TODO
    # what if the batch-shape is not consistent in the tree
    shape = jtu.tree_flatten(preds)[0][0].shape
    targets, preds = batch_concat(targets, num_batch_dims=len(shape) - 1), batch_concat(
        preds, num_batch_dims=len(shape) - 1
    )

    if reduce_fn:
        return reduce_fn(targets, preds)
    else:
        return targets, preds


def eval_module_on_datalaoder(module, dataloader):
    print("COMPILING `eval_module_on_dataloader`")
    dataloader, (inputs, targets) = dataloader()
    targets, preds = vmap_module_eval_and_reduce(module, inputs, targets)
    return dataloader, (targets, preds)


def make_step_fn(
    module,
    dataloader,
    optimizer,
    _lambda_l2: float,
    _lambda_l2_legacy: float,
    reduce_fn,
    include_init_state: bool,
    name,
):
    # delete .name attribute in optimizer state, otherwise `find_module`
    # will not work
    # TODO
    # `find_module` currently broken
    parameters = eqx.filter(module, filter_module(module, include_init_state))
    init_opt_state = optimizer.init(parameters)

    @ft.partial(
        eqx.filter_value_and_grad,
        arg=filter_module(module, include_init_state),
        has_aux=True,
    )
    def loss_fn(module, inputs, targets):
        error_signal = vmap_module_eval_and_reduce(
            module, inputs, targets, reduce_fn=reduce_fn
        )
        flat_parameters = flatten_module(module, include_init_state)
        l2_parameter_norm = l2_norm(flat_parameters)
        return (
            error_signal + _lambda_l2 * l2_parameter_norm + _lambda_l2_legacy * jnp.mean(flat_parameters**2),
            (
                error_signal,
                l2_parameter_norm,
            ),
        )

    def forward(params, state, x):
        data = x
        module, opt_state, dataloader = state

        loss, error, l2_parameter_norm = [], [], []
        for _ in range(dataloader.state.n_minibatches):
            dataloader, (inputs, targets) = dataloader(data)
            (loss_value, (error_value, l2_parameter_norm_value)), grad = loss_fn(
                module, inputs, targets
            )
            updates, opt_state = optimizer.update(grad, opt_state)
            module = eqx.apply_updates(module, updates)
            loss.append(loss_value)
            error.append(error_value)
            l2_parameter_norm.append(l2_parameter_norm_value)
        state = (module, opt_state, dataloader)
        return state, {
            "train_loss": jnp.array(loss),
            "train_error": jnp.array(error),
            "l2_parameter_norm": jnp.array(l2_parameter_norm),
        }

    init_state = module, init_opt_state, dataloader
    return make_module_from_function(forward, {}, init_state, name=name)


class ModuleTracker:
    def __init__(self, metric_key: str, moving_average_samples: int = 1, mode="min"):
        self.metric = deque(maxlen=moving_average_samples)
        self.metric_key = metric_key
        self._best_model = None
        self._associated_metric = None
        assert mode == "min"

    def report(self, model, metrics):
        metric = metrics[self.metric_key]

        self.metric.append(metric)

        if self._best_model is None:
            self._best_model = model
            self._associated_metric = metric
            return

        current_metric = np.mean(list(self.metric))
        if current_metric < self._associated_metric:

            self._associated_metric = current_metric
            self._best_model = model

    def best_model(self):
        return self._best_model

    def best_metric(self):
        return self._associated_metric


class DictLogger:
    def __init__(self):
        self._logs = None

    def log(self, metrics: dict[str, np.array]):
        metrics = tree_concat([metrics])
        if self._logs is None:
            self._logs = metrics
        else:
            self._logs = tree_concat([self._logs, metrics], True)

    def get_logs(self):
        return self._logs


class ModuleTrainer:
    """Trains a `Module`, evaluates and logs it."""

    def __init__(
        self,
        module: Module,
        optimizer,
        train_dataloader: Module,
        module_tracker=None,
        error_fn=mse,
        val_dataloader: Module = None,
        val_log_fns=[lambda y, yhat: {"val_mse": mse(y, yhat)}],
        val_log_every: int = 1,
        test_dataloader: Module = None,
        test_log_fns=[lambda y, yhat: {"test_mse": mse(y, yhat)}],
        test_log_every: int = 1,
        opt_state_log_fns=[],
        _lambda_l2: float = 0.0,
        _lambda_l2_legacy: float = 0.0,
        init_state_is_param: bool = False,
        use_wandb: bool = False,
        wandb_config: dict = {},
        loggers=None,
        jit: bool = True,
    ):

        if module_tracker is None:
            self.module_tracker = ModuleTracker("train_loss")
        else:
            self.module_tracker = module_tracker

        self.loggers = loggers if loggers else [DictLogger()]

        self._jit = jit

        name = None
        if module.name is not None:
            name = f"step-fn-{module.name}"

        self._step_fn = make_step_fn(
            module,
            train_dataloader,
            optimizer,
            _lambda_l2,
            _lambda_l2_legacy,
            error_fn,
            init_state_is_param,
            name,
        )

        self._val_dataloader = val_dataloader
        self._val_log_fns = val_log_fns
        self._val_log_every = val_log_every
        self._test_dataloader = test_dataloader
        self._test_log_fns = test_log_fns
        self._test_log_every = test_log_every

        self._opt_state_log_fns = opt_state_log_fns

        self._last_metrices = {}

    def get_current_module(self):
        return self._step_fn.state[0]

    def get_current_opt_state(self):
        return self._step_fn.state[1]

    def get_best_module(self):
        return self.module_tracker.best_model()

    def get_best_module_metric(self):
        return self.module_tracker.best_metric()

    def update_pbar(self, metrics: dict[str, np.array]):
        s = ""
        for key, value in metrics.items():
            s += "{}: {:10.4f} | ".format(key, float(np.mean(value)))
        self.pbar.set_description(s)

    def step(self, step: int = 0):

        # TRAINING STEP
        if self._jit:
            self._step_fn, metrics = eqx.filter_jit(self._step_fn)()
        else:
            self._step_fn, metrics = self._step_fn()

        _eval_module_on_dataloader = eval_module_on_datalaoder
        if self._jit:
            _eval_module_on_dataloader = eqx.filter_jit(eval_module_on_datalaoder)

        # EVALUATION STEP
        if self._val_dataloader:
            if (step % self._val_log_every) == 0:
                self._val_dataloader, (targets, preds) = _eval_module_on_dataloader(
                    self.get_current_module(), self._val_dataloader
                )

                for reduce_fn in self._val_log_fns:
                    metrics.update(reduce_fn(targets, preds))

        if self._test_dataloader:
            self._test_dataloader, (targets, preds) = _eval_module_on_dataloader(
                self.get_current_module(), self._test_dataloader, self._test_reduce_fn
            )
            for reduce_fn in self._test_log_fns:
                metrics.update(reduce_fn(targets, preds))

        for log_fn in self._opt_state_log_fns:
            metrics.update(log_fn(self.get_current_opt_state()))

        self._last_metrices.update(to_numpy(metrics))
        return self._last_metrices

    def run(self, steps: int = 1):
        self.pbar = tqdm(range(steps), disable=not use_tqdm())
        for step in self.pbar:
            metrics = self.step(step)
            self.update_pbar(metrics)
            for logger in self.loggers:
                logger.log(metrics)
            self.module_tracker.report(self.get_current_module(), metrics)

    def get_logs(self):
        logs = []
        for logger in self.loggers:
            logs.append(logger.get_logs())
        return logs
