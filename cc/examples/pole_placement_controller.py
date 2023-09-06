import os
from typing import Protocol

import control as ct
from dm_control.rl.control import PhysicsError
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.linalg import expm
from tree_utils import PyTree

from cc.acme.utils.paths import process_path
from cc.core.save_load import save
from cc.env.collect import collect_exhaust_source
from cc.env.collect import random_steps_source
from cc.env.wrappers import AddRefSignalRewardFnWrapper
from cc.env.wrappers import NoisyObservationsWrapper
from cc.examples.linear_model import make_linear_model
from cc.train import DictLogger
from cc.train import EvaluationMetrices
from cc.train import make_dataloader
from cc.train import ModelControllerTrainer
from cc.train import SupervisedDataset
from cc.train import Tracker
from cc.train import TrainingOptionsModel
from cc.utils import rmse
from cc.utils import split_filename
from cc.utils.high_level.defaults import Env

from ..core import AbstractController
from ..utils import batch_concat


# Observation comes from a env wrapped with a reference signal
def _preprocess_error_as_controller_input(x) -> jnp.ndarray:
    # capture x, split into ref / obs
    ref, obs = batch_concat(x["ref"], 0), batch_concat(x["obs"], 0)
    # calculate error based on
    err = ref - obs
    return err


def _make_linear_controller(A, B, C, D, T):
    class LinearController(AbstractController):
        A: jax.Array
        B: jax.Array
        C: jax.Array
        D: jax.Array
        x: jax.Array
        x0: jax.Array

        def reset(self):
            return LinearController(self.A, self.B, self.C, self.D, self.x0, self.x0)

        def step(self, inp):
            error = _preprocess_error_as_controller_input(inp)
            x_next = self.A @ self.x + self.B @ error

            out = self.C @ self.x + self.D @ error
            return (
                LinearController(self.A, self.B, self.C, self.D, x_next, self.x0),
                out,
            )

        def grad_filter_spec(self) -> PyTree[bool]:
            filter_spec = super().grad_filter_spec()
            return filter_spec

    Ad = expm(A * T)
    Bd = np.linalg.inv(A) @ (Ad - np.eye(Ad.shape[0])) @ B
    x0 = jnp.zeros((A.shape[0],))
    return LinearController(Ad, Bd, C, D, x0, x0)


def _train_linear_model(
    env: Env,
    state_dim: int = 5,
    n_steps: int = 30_000,
    optimizer=optax.adam(1e-2),
    continuous_time: bool = True,
    include_D: bool = True,
    seed: int = 1,
):
    model = make_linear_model(
        env.env.action_spec(),
        env.env.observation_spec(),
        env.env.control_timestep(),
        state_dim=state_dim,
        continuous_time=continuous_time,
        include_D=include_D,
        seed=seed,
    )

    train_sample = env.train_sample

    model_train_dataloader = make_dataloader(
        SupervisedDataset(train_sample.action, train_sample.obs)
    )

    metrices = (
        EvaluationMetrices(
            data=(env.val_sample.action, env.val_sample.obs),  # <- (X, y)
            metrices=(lambda y, yhat: {"val_rmse": rmse(y, yhat)},),
        ),
    )

    model_train_options = TrainingOptionsModel(
        model_train_dataloader,
        optimizer,
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


def make_pole_placed_controller(
    env: Env,
    poles: list[float],
    verbose: bool = True,
    pretrained_model=None,
    **kwargs,
):
    """
    NOTE: len(poles) == kwargs["state_dim"] * 2
    """
    if pretrained_model is None:
        model_trainer = _train_linear_model(env, **kwargs)
        model = model_trainer._model
        report = jax.tree_map(lambda arr: arr[-1], model_trainer.get_logs()[0])

        if verbose:
            print("System Identification Report: ", report)
    else:
        report = None
        model = pretrained_model

    # this assumes that model is trained *in continuous time*
    # TODO
    # specify dt dynamically based on whether or not the linear
    # model is trained in continuous time or not
    assert kwargs.get("continuous_time", True)
    ss = ct.ss(model.A, model.B, model.C, model.D)
    if verbose:
        print("Poles of linearized plant: ", ss.poles())

    tf = ct.tf(ss)
    b_n, a_n = tf.num[0][0], tf.den[0][0]

    b_n = [b for b in b_n]
    while len(b_n) < len(a_n):
        b_n.insert(0, 0)
    b_n = np.array(b_n)

    n = model.A.shape[0]
    m = n

    s = ct.tf("s")
    q = 1.0

    for pole in poles:
        q *= s - pole

    qsoll = [v for v in q.num[0][0]]  # pytype: disable=attribute-error
    qsoll.insert(0, 1.0)
    qsoll = np.array(qsoll)

    A = np.zeros((2 * (n + 1), 2 * (n + 1)))
    A[0, 0] = 1.0
    for i in range(n + 1):
        for j in range(m + 1):
            A[j + i + 1, j] = a_n[i]
            A[j + i + 1, j + n + 1] = b_n[i]

    alpha_beta = np.linalg.inv(A) @ qsoll
    alpha_n = alpha_beta[: (m + 1)]
    beta_n = alpha_beta[(m + 1) :]
    K = ct.tf(beta_n, alpha_n)

    if verbose:
        print("Poles of closed loop: ", ct.feedback(ss, K).poles())

    K = ct.ss(K)

    init_fn = lambda: _make_linear_controller(
        K.A, K.B, K.C, K.D, env.env.control_timestep()
    )
    return init_fn, model, report


class Search(Protocol):
    def query(self) -> float:
        ...

    def update(self, x, fx):
        pass

    def finished(self) -> bool:
        ...


class ConvexSearch(Search):
    def __init__(
        self, a, b, n_trials: int, minimize: bool = True, scale: float = 0.1
    ) -> None:
        self.a, self.b = a, b
        self.fs = {}
        self.minimize = minimize
        self.scale = scale
        self._count = 0
        self.n_trials = n_trials

    def query(self):
        self._count += 1

        if self.a not in self.fs:
            return self.a
        if self.b not in self.fs:
            return self.b
        x = (self.b + self.a) / 2

        # because the collect functions fix seed internally
        np.random.seed(self._count)
        disturbance = np.random.normal() * self.scale
        return x + disturbance

    def update(self, x, fx):
        if np.isnan(fx):
            return

        self.fs[x] = fx

        if len(self.fs) < 2:
            return

        # find two lowest f values
        xs_fxs = [(x, fx) for x, fx in self.fs.items()]
        xs_fxs.sort(key=lambda ele: ele[1], reverse=not self.minimize)

        winners = [x for x, fx in xs_fxs[:2]]
        self.a, self.b = winners

    def finished(self) -> bool:
        return len(self.fs) >= self.n_trials


class GridSearch(Search):
    def __init__(self, grid: list[float]):
        self.grid = grid
        self._count = 0

    def query(self):
        self._count += 1
        return self.grid[self._count - 1]

    def finished(self) -> bool:
        return len(self.grid) <= self._count


def search_pole_placed_controller(
    env: Env,
    filename: str,
    search: Search,
    ident_model_every_search: bool = False,
    verbose: bool = True,
    path: str = "~/chain_control",
    experiment_id: str = "",
    **kwargs,
):
    "len(poles) == 2*state_dim"

    order = kwargs.get("state_dim", 5)

    res = []
    model = None
    i = -1
    while not search.finished():
        i += 1
        p = search.query()

        if ident_model_every_search:
            model = None

        init_fn, model, report = make_pole_placed_controller(
            env, [p] * 2 * order, verbose, model, **kwargs
        )

        if report is not None:
            if report["train_mse"] > 1e-5:
                # re-draw model
                kwargs["seed"] = kwargs.get("seed", 1) + i
                model = None
                continue

        source = random_steps_source(env.env, list(range(7000, 7010)))
        env_w_source = AddRefSignalRewardFnWrapper(
            NoisyObservationsWrapper(env.env, 0.02), source
        )
        try:
            sample, _ = collect_exhaust_source(env_w_source, init_fn())
        except PhysicsError:
            continue

        rmse = np.sqrt(np.mean(-sample.rew))
        res.append((init_fn, p, rmse))

        search.update(p, rmse)

    print("Results: ", [(p, rmse) for _, p, rmse in res])
    res.sort(key=lambda ele: ele[2])
    print("Best is: ", res[0][1])

    init_fn = res[0][0]
    path_folder = process_path(
        path, "lcss_baseline_tf_ctrb", experiment_id, add_uid=False
    )
    path_controller = os.path.join(path_folder, split_filename(filename)[0])
    save(init_fn, path_controller + ".pkl")

    return init_fn()
