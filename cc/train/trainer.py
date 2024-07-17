from collections import deque
from typing import Optional, Union

import equinox as eqx
import numpy as np
from tqdm.auto import tqdm
from tree_utils import tree_batch

from ..core import AbstractController
from ..core import AbstractModel
from ..core.config import use_tqdm
from ..utils import to_numpy
from .step_fn import make_step_fn_controller
from .step_fn import make_step_fn_model
from .step_fn import TrainingOptionsController
from .step_fn import TrainingOptionsModel


class Tracker:
    def __init__(
        self, *tracking_metric: str, moving_average_samples: int = 1, mode="min"
    ):
        self.metric = deque(maxlen=moving_average_samples)
        self.metric_key = tracking_metric[0]
        for s in tracking_metric[1:]:
            self.metric_key += "_" + s

        self._best_model = None
        self._associated_metric = None
        assert mode == "min"

        self._i = 0
        self._i_best = 0

    def report(self, model, logs):
        self._i += 1

        metric = logs[self.metric_key]

        self.metric.append(metric)

        if self._best_model is None:
            self._best_model = model
            self._associated_metric = metric
            self._i_best = self._i
            return

        current_metric = np.mean(list(self.metric))
        if current_metric < self._associated_metric:
            self._associated_metric = current_metric
            self._best_model = model
            self._i_best = self._i

    def best_model_or_controller(self):
        return self._best_model

    def best_metric(self):
        return self._associated_metric

    def log_entry(self):
        return {self.metric_key: self.best_metric()}

    def best_metric_i(self):
        return self._i_best


class Logger:
    def log(self, metrics: dict[str, np.ndarray]):
        pass

    def get_logs(self):
        pass


class DictLogger(Logger):
    def __init__(self):
        self._logs = None

    def log(self, metrics: dict[str, np.ndarray]):
        metrics = tree_batch([metrics])
        if self._logs is None:
            self._logs = metrics
        else:
            self._logs = tree_batch([self._logs, metrics], True)

    def get_logs(self):
        return self._logs


class Callback:
    def __call__(
        self,
        model: None | AbstractModel,
        controller: None | AbstractController,
        opt_state,
        minibatch_state,
        logs: dict,
        loggers: list[Logger],
        trackers: list[Tracker],
    ) -> None:
        pass


class ModelControllerTrainer:
    """Trains a `Module`, evaluates and logs it."""

    def __init__(
        self,
        model: Union[AbstractModel, dict[str, AbstractModel]],
        controller: Optional[AbstractController] = None,
        model_train_options: Optional[TrainingOptionsModel] = None,
        controller_train_options: Optional[TrainingOptionsController] = None,
        loggers: list[Logger] = [],
        trackers: list[Tracker] = [],
        callbacks: list[Callback] = [],
        jit: bool = True,
    ):
        assert not (
            model_train_options is not None and controller_train_options is not None
        ), """Specify only `TrainingOptionsModel` or `TrainingOptionsController`
        depending on what you want to optimise for.."""

        self._model = self._controller = None

        if model_train_options is not None:
            assert not isinstance(
                model, dict
            ), "Multiple models are only suppored for controller training."
            self._model = model
            self._step_fn, self._opt_state, self._minibatch_state = make_step_fn_model(
                model, model_train_options
            )
        elif controller_train_options is not None:
            assert controller is not None, "Controller is missing."

            if not isinstance(model, dict):
                model = dict(model0=model)
                print(
                    """This model has been registered with model name `model0`.
                    When using multiple models individual model names have to
                    be provided by passing a dictionary in the `model` argument"""
                )

            self._controller = controller
            (
                self._step_fn,
                self._opt_state,
                self._minibatch_state,
            ) = make_step_fn_controller(controller, model, controller_train_options)

        if jit:
            self._step_fn = eqx.filter_jit(self._step_fn)

        self.loggers = loggers
        self.trackers = trackers
        self.callbacks = callbacks

    def update_pbar(self, metrics: dict[str, np.ndarray]):
        s = ""
        for key, value in metrics.items():
            s += "{}: {:10.8f} | ".format(key, float(np.mean(value)))
        self.pbar.set_description(s)

    def step(self, i_step: int = 0):
        if self._model:
            self._model, self._opt_state, self._minibatch_state, logs = self._step_fn(
                self._model, self._opt_state, self._minibatch_state
            )
        else:
            (
                self._controller,
                self._opt_state,
                self._minibatch_state,
                logs,
            ) = self._step_fn(self._controller, self._opt_state, self._minibatch_state)

        logs = to_numpy(logs)

        for callback in self.callbacks:
            callback(
                self._model,
                self._controller,
                self._opt_state,
                self._minibatch_state,
                logs,
                self.loggers,
                self.trackers,
            )

        for logger in self.loggers:
            logger.log(logs)

        for tracker in self.trackers:
            m = self._model if self._model is not None else self._controller
            tracker.report(m, logs)

        return logs

    def run(self, steps: int = 1):
        self.pbar = tqdm(range(steps), disable=not use_tqdm())
        for i in self.pbar:
            logs = self.step(i)
            self.update_pbar({"train_loss": logs["train_loss"]})

    def get_tracker_logs(self):
        logs = []
        for tracker in self.trackers:
            logs.append(tracker.log_entry())
        return logs

    def get_logs(self):
        logs = []
        for logger in self.loggers:
            logs.append(logger.get_logs())
        return logs
