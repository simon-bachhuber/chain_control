from collections import deque
from typing import Optional

import equinox as eqx
import numpy as np
from tqdm.auto import tqdm

from ..config import use_tqdm
from ..core import AbstractController, AbstractModel
from ..utils import to_numpy, tree_concat
from .step_fn import (
    TrainingOptionsController,
    TrainingOptionsModel,
    make_step_fn_controller,
    make_step_fn_model,
)


class Tracker:
    def __init__(
        self, tracking_metric: str, moving_average_samples: int = 1, mode="min"
    ):
        self.metric = deque(maxlen=moving_average_samples)
        self.metric_key = tracking_metric
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


class Logger:
    def log(self, metrics: dict[str, np.ndarray]):
        pass

    def get_logs(self):
        pass


class DictLogger(Logger):
    def __init__(self):
        self._logs = None

    def log(self, metrics: dict[str, np.ndarray]):
        metrics = tree_concat([metrics])
        if self._logs is None:
            self._logs = metrics
        else:
            self._logs = tree_concat([self._logs, metrics], True)

    def get_logs(self):
        return self._logs


class ModelControllerTrainer:
    """Trains a `Module`, evaluates and logs it."""

    def __init__(
        self,
        model: Optional[AbstractModel] = None,
        controller: Optional[AbstractController] = None,
        model_train_options: Optional[TrainingOptionsModel] = None,
        controller_train_options: Optional[TrainingOptionsController] = None,
        loggers: list[Logger] = [],
        trackers: list[Tracker] = [],
        jit: bool = True,
    ):

        assert not (
            model_train_options is not None and controller_train_options is not None
        ), """Specify only `TrainingOptionsModel` or `TrainingOptionsController`
        depending on what you want to optimise for.."""

        assert model is not None, "Model is missing."

        self._model = self._controller = None

        if model_train_options is not None:
            self._model = model
            self._step_fn, self._opt_state, self._minibatch_state = make_step_fn_model(
                model, model_train_options
            )
        elif controller_train_options is not None:
            assert controller is not None, "Controller is missing."
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

    def update_pbar(self, metrics: dict[str, np.ndarray]):
        s = ""
        for key, value in metrics.items():
            s += "{}: {:10.4f} | ".format(key, float(np.mean(value)))
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
            self.update_pbar(logs)

    def get_logs(self):
        logs = []
        for logger in self.loggers:
            logs.append(logger.get_logs())
        return logs
