import functools as ft

import jax
import jax.random as jrand
import jax.tree_util as jtu
import optax

from ..abstract import (AbstractController, AbstractModel,
                        AbstractObservationReferenceSource)
from ..buffer import ReplaySample
from ..rhs.parameter import filter_module, flatten_module
from ..types import *
from ..utils import to_jax
from .minibatch import MiniBatchState, minibatch
from .unroll import unroll_closed_loop, unroll_model


class ModelTrainLoss(NamedTuple):
    train_loss: jnp.ndarray


class ModelTrainTestLoss(NamedTuple):
    train_loss: jnp.ndarray
    test_loss: jnp.ndarray


def step_fn_model(
    model: AbstractModel, 
    train_sample: ReplaySample, 
    key: PRNGKey,
    test_sample: ReplaySample = None,
    _lambda: float = 0.1, 
    optimizer = optax.adam(3e-3), 
    number_of_minibatches: int = 1, 
    eval_test_loss: bool = True,
) -> Tuple[FunctionType, MiniBatchState]:

    # TODO 
    # otherwise some weird static hash error
    #@eqx.filter_jit
    def loss_fn_model(model: AbstractModel, sample: ReplaySample):
        yhatss = jax.vmap(lambda us: unroll_model(model, us))(sample.action)["xpos_of_segment_end"]
        yss = sample.obs["xpos_of_segment_end"]
        params = flatten_module(model)
        return jnp.mean((yss-yhatss)**2) + _lambda*jnp.mean(params**2)

    grad_loss_fn_model = eqx.filter_value_and_grad(loss_fn_model, arg=filter_module(model))

    key, consume = jrand.split(key)
    # TODO key is hard coded; only to get the "0.4675"-model
    minibatcher = minibatch(n_minibatches=number_of_minibatches, key=jrand.PRNGKey(1,))
    minibatch_state = minibatcher.init(train_sample)

    @eqx.filter_jit
    def _step_fn(model: AbstractModel, opt_state, minibatch_state: MiniBatchState) -> Tuple[AbstractModel, Any, MiniBatchState, ModelTrainTestLoss]:

        if eval_test_loss:
            # TODO 
            # Write proper error message
            if test_sample is None:
                raise Exception()

            test_loss = loss_fn_model(model, test_sample)

        train_loss = []
        for _ in range(minibatch_state.n_minibatches):
            minibatch_state, batch_of_sample = minibatcher.update(minibatch_state, train_sample)
            value, grad = grad_loss_fn_model(model, batch_of_sample)
            updates, opt_state = optimizer.update(grad, opt_state)
            model = eqx.apply_updates(model, updates)
            train_loss.append(value)

        if eval_test_loss:
            train_test_loss = ModelTrainTestLoss(jnp.mean(jnp.array(train_loss)), test_loss)
        else:
            train_test_loss = ModelTrainLoss(jnp.mean(jnp.array(train_loss)))

        return model, opt_state, minibatch_state, train_test_loss

    return _step_fn, minibatch_state


def default_merge_x_y(x: Reference, y: Observation):
    d = OrderedDict()
    d["ref"] = x
    d["obs"] = y 
    return d 


def step_fn_controller(
    controller: AbstractController, 
    models: list[AbstractModel], 
    source: AbstractObservationReferenceSource,
    key: PRNGKey,
    u_transform_factory,
    merge_x_y: Callable = default_merge_x_y,
    _lambda: float = 0.1, 
    optimizer = optax.adam(3e-3), 
    delay: int = 0,
    n_minibatches: int = 1,
    tree_transform = None 
    ):

    refss: BatchedTimeSeriesOfRef = to_jax(source.get_references_for_optimisation())
    key, consume = jrand.split(key)
    minibatcher = minibatch(consume, n_minibatches, tree_transform=tree_transform)
    minibatch_state = minibatcher.init(refss)
    

    @eqx.filter_jit 
    @ft.partial(eqx.filter_value_and_grad, arg=filter_module(controller))
    def grad_loss_fn_controller(controller: AbstractController, refss: BatchedTimeSeriesOfRef, key: PRNGKey):

        # for regularisation on parameter norm
        params = flatten_module(controller)

        # u1 = controller(y0, ref0) -> y1 = model(u1)
        # so to create y_0_T we only need ref_0_Tm1
        refss_0_Tm1 = jtu.tree_map(lambda arr: arr[:,:-1], refss)

        # split keys for vmap
        keys = jrand.split(key, minibatch_state.minibatch_size)

        if len(models)>1:

            yhatsss = []
            for model in models:
                yhatss = jax.vmap(lambda refs, key: 
                    unroll_closed_loop(model, controller, refs, model.y0(), merge_x_y, delay, u_transform_factory, key))(refss_0_Tm1, keys)["xpos_of_segment_end"]
                yhatsss.append(yhatss)

            yhatsss = jnp.stack(yhatsss)
            refsss = jnp.repeat(refss["xpos_of_segment_end"][None], len(models), axis=0)

            return jnp.mean((refsss-yhatsss)**2) + _lambda*jnp.mean(params**2)
        
        else:
            model = models[0]
            yhatss = jax.vmap(
                        lambda refs, key: unroll_closed_loop(model, controller, refs, model.y0(), merge_x_y, delay, u_transform_factory, key)
                    ) (refss_0_Tm1, keys)["xpos_of_segment_end"]
            
            return jnp.mean((refss["xpos_of_segment_end"] - yhatss)**2) + _lambda*jnp.mean(params**2)


    @eqx.filter_jit
    def _step_fn(controller: AbstractController, opt_state, minibatch_state: MiniBatchState):

        loss = []
        for _ in range(minibatch_state.n_minibatches):
            minibatch_state, minibatch_refss = minibatcher.update(minibatch_state, refss)
            # minibatch state already carries a key 
            key, consume = jrand.split(minibatch_state.key)
            minibatch_state = minibatch_state._replace(key=key)

            value, grad = grad_loss_fn_controller(controller, minibatch_refss, consume)
            updates, opt_state = optimizer.update(grad, opt_state)
            controller = eqx.apply_updates(controller, updates)
            loss.append(value)

        return controller, opt_state, minibatch_state, jnp.mean(jnp.array(loss))


    return _step_fn, minibatch_state

