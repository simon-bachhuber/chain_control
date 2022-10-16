from dataclasses import dataclass

import optax
from beartype import beartype

from ..abstract import AbstractController, AbstractModel
from ..buffer import ReplaySample
from ..collect.source import ObservationReferenceSource
from ..rhs.parameter import filter_module
from ..types import *
from .sgd import SGD_Loop
from .step_fn import step_fn_controller, step_fn_model


@dataclass
class _TrainingOptions:
    key: PRNGKey
    optimizer: optax.GradientTransformation
    l2_regu: float 
    n_gradient_steps: int 
    number_of_minibatches: int 


@dataclass
class TrainingOptionsModel(_TrainingOptions):
    eval_test_loss: bool 


def default_u_transform_factory(key):
    return lambda u,t: u 


@dataclass
class TrainingOptionsController(_TrainingOptions):
    models: list[AbstractModel]
    delay: int = 0
    u_transform_factory: Callable = default_u_transform_factory
    tree_transform: Optional[Callable] = None 


def _train(module, step_fn_missing_training_options, 
        training_options: Union[TrainingOptionsModel, TrainingOptionsController]):
    
    optimizer = training_options.optimizer
    opt_state = optimizer.init(eqx.filter(module, filter_module(module)))

    step_fn, minibatch_state = step_fn_missing_training_options(training_options)

    loop = SGD_Loop(step_fn)
    losses = loop.gogogo(training_options.n_gradient_steps, module, opt_state, minibatch_state)
    return loop._module, losses 


@beartype
def train_controller(
    controller: AbstractController, 
    source: ObservationReferenceSource,
    training_options: TrainingOptionsController,
    ):

    def step_fn(training_options: TrainingOptionsController):
        return step_fn_controller(
            controller, 
            training_options.models,
            source,
            key=training_options.key,
            u_transform_factory=training_options.u_transform_factory,
            _lambda=training_options.l2_regu,
            optimizer=training_options.optimizer,
            delay=training_options.delay,
            n_minibatches=training_options.number_of_minibatches,
            tree_transform=training_options.tree_transform
        )
    
    return _train(controller, step_fn, training_options)


@beartype 
def train_model(
    model: AbstractModel,
    train_sample: ReplaySample,
    training_options: TrainingOptionsModel,
    test_sample: Optional[ReplaySample] = None 
    ):

    if training_options.eval_test_loss:
        if test_sample is None:
            raise Exception("You have to provide the argument `test_sample` if `training_options.eval_test_loss` is `True`")

    def step_fn(training_options: TrainingOptionsModel):
        return step_fn_model(
            model, 
            train_sample,
            training_options.key,
            test_sample,
            training_options.l2_regu,
            training_options.optimizer,
            training_options.number_of_minibatches,
            training_options.eval_test_loss
        )
    
    return _train(model, step_fn, training_options)

