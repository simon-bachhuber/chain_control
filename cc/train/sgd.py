from ..abstract import AbstractWrappedRHS
from tqdm.auto import tqdm 
from ..config import use_tqdm
import numpy as np 
from ..rhs.parameter import flatten_module
from .minibatch import MiniBatchState
from .step_fn import ModelTrainLoss, ModelTrainTestLoss
import jax.numpy as jnp 
from collections import deque


def pbar_desc_(pbar, train_loss, test_loss, module):
    if test_loss:
        pbar.set_description("Trainings-Loss: {:10.4f} | Test-Loss: {:10.4f} | ParamsRegu: {:10.4f}".format(train_loss, test_loss, jnp.mean(flatten_module(module)**2)))
    else:
        pbar.set_description("Trainings-Loss: {:10.4f} | ParamsRegu: {:10.4f}".format(train_loss, jnp.mean(flatten_module(module)**2)))


class ModelReport:
    def __init__(self, moving_average_samples: int = 1, mode = "min"):
        self.metric = deque(maxlen=moving_average_samples)
        self._best_model = None 
        self._associated_metric = None
        assert mode == "min"

    def report(self, model, metric):
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


class SGD_Loop:
    def __init__(self, step_fn):
        self._step_fn = step_fn
        self._module = None 
        self._opt_state = None 
        self._model_report = ModelReport()

    def gogogo(self, steps: int, module: AbstractWrappedRHS = None, opt_state = None, minibatch_state: MiniBatchState = None):


        if module:
            self._module = module   
        if self._module is None:
            raise Exception("No initial module / parameters")
        if opt_state:
            self._opt_state = opt_state
        if self._opt_state is None:
            raise Exception("No initial optimizer state.")
        if minibatch_state:
            self._minibatch_state = minibatch_state
        if self._minibatch_state is None:
            raise Exception("No initial minibatch state")


        pbar = tqdm(range(steps), disable=not use_tqdm())
        train_loss_values = []
        test_loss_values = []
        test_loss = None 

        for i in pbar:
            self._module, self._opt_state, self._minibatch_state, value = self._step_fn(self._module, self._opt_state, self._minibatch_state)
            
            if isinstance(value, ModelTrainTestLoss):
                train_loss = float(value.train_loss)
                test_loss = float(value.test_loss)
                test_loss_values.append(test_loss)
            elif isinstance(value, ModelTrainLoss):
                train_loss = float(value.train_loss)
            else:
                train_loss = float(value)

            # Track best module based on test loss if available
            self._model_report.report(self._module, train_loss if test_loss is None else test_loss)

            pbar_desc_(pbar, train_loss, test_loss, self._module)
            train_loss_values.append(train_loss)

        if test_loss:
            test_loss_values = np.array(test_loss_values)
        else:
            test_loss_values = None 
        
        return ModelTrainTestLoss(np.array(train_loss_values), test_loss_values)

