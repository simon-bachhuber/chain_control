from collections import OrderedDict
import os
from pathlib import Path
import pickle
from typing import Optional

import equinox as eqx
import fire
import jax
import numpy as np
import optax

from cc.core import save_eqx
from cc.core.abstract import AbstractController
from cc.core.abstract import AbstractModel
from cc.env import make_env
from cc.env.collect.source import ObservationReferenceSource
from cc.env.wrappers import AddRefSignalRewardFnWrapper
from cc.examples.linear_model import make_linear_model
from cc.examples.neural_ode_controller_compact_example import make_neural_ode_controller
from cc.train import Callback
from cc.train import EvaluationMetricesController
from cc.train import make_dataloader
from cc.train import ModelControllerTrainer
from cc.train import Tracker
from cc.train import TrainingOptionsController
from cc.train import UnsupervisedDataset
from cc.train.trainer import DictLogger
from cc.train.trainer import Logger
from cc.utils import mse


def _make_env(time_limit=1.0):
    env = make_env(
        "dummy", time_limit=time_limit, task_kwargs=dict(input_dim=1, output_dim=1)
    )
    return env


def _load_lti_from_path(path: str):
    "Loads (A, B.T, C) dynamics from .npy file"

    # time_limit arg is here unused; it is just some dummy value
    env = _make_env()

    m = make_linear_model(
        env.action_spec(), env.observation_spec(), env.control_timestep(), 3
    )
    ABC = np.load(path)
    state_dim = ABC.shape[1]
    A, B, C, D = (
        ABC[:state_dim],
        ABC[state_dim],
        ABC[state_dim + 1],
        np.zeros((1, 1)),
    )
    B, C = B[:, None], C[None, :]
    return eqx.tree_at(lambda m: (m.A, m.B, m.C, m.D), m, (A, B, C, D))


def _load_refs_from_path_to_folder(path: str):
    refss = np.stack(tuple(np.load(file) for file in listdir(path)))
    d = OrderedDict()
    d["output"] = refss
    return ObservationReferenceSource(yss=d)


def listdir(path, extension=None):
    files = [Path(path).joinpath(file) for file in os.listdir(path)]
    if extension is not None:
        files = [file for file in files if file.suffix == extension]
    return files


class MeanValidationCallback(Callback):
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
        logs.update(
            {
                "validation": np.mean(
                    [value for key, value in logs.items() if key[:3] == "val"]
                )
            }
        )


def main(
    output_path: str,
    path_folder_train_dynamics: str,
    path_folder_train_refs: str,
    path_folder_val_dynamics: Optional[str] = None,
    path_folder_val_refs: Optional[str] = None,
    n_minibatches: int = 15,
    n_episodes: int = 750,
    controller_hidden_state_dim: int = 25,
    controller_f_depth: int = 2,
    controller_f_width: int = 10,
    controller_g_depth: int = 0,
    controller_g_width: int = 10,
    lr: float = 1e-3,
    grad_clip_norm: float = 1.0,
):
    assert (path_folder_val_dynamics is None and path_folder_val_refs is None) or (
        path_folder_val_dynamics is not None and path_folder_val_refs is not None
    )
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    assert (
        len(listdir(output_path)) == 0
    ), "Output path not empty. This might overwrite files."

    ms = [
        _load_lti_from_path(file)
        for file in listdir(path_folder_train_dynamics, ".npy")
    ]

    env = _make_env()
    source = _load_refs_from_path_to_folder(path_folder_train_refs)
    env_w_source = AddRefSignalRewardFnWrapper(env, source)

    controller_init_fn = lambda: make_neural_ode_controller(
        env_w_source.observation_spec(),
        env.action_spec(),
        env.control_timestep(),
        state_dim=controller_hidden_state_dim,
        f_depth=controller_f_depth,
        f_width_size=controller_f_width,
        g_depth=controller_g_depth,
        g_width_size=controller_g_width,
    )
    controller = controller_init_fn()

    controller_dataloader = make_dataloader(
        UnsupervisedDataset(source.get_references_for_optimisation()),
        jax.random.PRNGKey(
            1,
        ),
        n_minibatches=n_minibatches,
    )

    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(lr))

    if path_folder_val_dynamics is None:
        tracker = Tracker("loss")
        metrices = tuple()
        callbacks = []
    else:
        tracker = Tracker("validation")
        val_source = _load_refs_from_path_to_folder(path_folder_val_refs)
        val_ms = [
            _load_lti_from_path(file)
            for file in listdir(path_folder_val_dynamics, ".npy")
        ]
        metrices = []
        for i, val_m in enumerate(val_ms):

            def metric_fn(obs, ref):
                return {f"val{i}": mse(obs, ref)}

            metrices.append(
                EvaluationMetricesController(
                    val_source.get_references_for_optimisation(),
                    val_m,
                    tuple([metric_fn]),
                )
            )
        metrices = tuple(metrices)

        callbacks = [MeanValidationCallback()]

    controller_train_options = TrainingOptionsController(
        controller_dataloader, optimizer, metrices=metrices
    )

    controller_trainer = ModelControllerTrainer(
        {f"system{i}": ms[i] for i in range(len(ms))},
        controller,
        controller_train_options=controller_train_options,
        trackers=[tracker],
        callbacks=callbacks,
        loggers=[DictLogger()],
    )

    controller_trainer.run(n_episodes)
    logs = controller_trainer.get_logs()[0]
    fitted_controller = tracker.best_model_or_controller()

    # dump controller and logs
    with open(output_path.joinpath("logs.pickle"), "wb") as f:
        pickle.dump(logs, f)

    save_eqx(output_path.joinpath("controller"), fitted_controller, controller_init_fn)

    print(
        f"Best controller has achieved a `{tracker.metric_key}` value of "
        f"{tracker.best_metric()} at episode {tracker.best_metric_i()}"
    )


if __name__ == "__main__":
    fire.Fire(main)
