import os
from dataclasses import dataclass
from typing import Callable, Optional

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import myplotlib
import numpy as np

from cc.acme.utils.observers import EnvLoopObserver
from cc.acme.utils.paths import process_path
from cc.env.collect import collect_exhaust_source
from cc.env.collect.source import ObservationReferenceSource
from cc.env.loop_observer import DenoisifyObserver
from cc.env.wrappers import (
    AddRefSignalRewardFnWrapper,
    ReplacePhysicsByModelWrapper,
    VideoWrapper,
)
from cc.utils import rmse, timestep_array_from_env

bbox = lambda color: dict(boxstyle="round", facecolor=color, alpha=0.35)
colors = ["green", "blue", "orange", "brown", "red", "black"]


def obs_flatten(obs):
    return jax.tree_util.tree_flatten(obs)[0][0][..., 0]


def obs_take(obs, idx):
    return obs_flatten(obs)[idx, :-1]


def key_of_obs(obs):
    assert len(obs.keys()) == 1
    return str([key for key in obs.keys()][0])


@dataclass
class LoopObserverConfig:
    loop_observer: EnvLoopObserver
    y_label: str
    arr_from_loop_results: Callable


def _eval_model_1(ax, sample, ts, model, indices):
    m = eqx.filter_vmap(model.unroll)

    start_idx = indices[0]
    for idx, color in zip(indices, colors):
        obs_hat = m(sample.action)
        m_rmse = rmse(sample.obs, obs_hat)

        # dummy plot for boxed legend
        if idx == start_idx:
            (line1,) = ax.plot(
                ts,
                obs_take(sample.obs, idx),
                label="Ground Truth / Output in Env",
                color="black",
            )
            (line2,) = ax.plot(
                ts,
                obs_take(obs_hat, idx),
                label="Prediction / Output in Model",
                linestyle="--",
                color="black",
            )

        ax.plot(ts, obs_take(sample.obs, idx), color=color)
        ax.plot(
            ts,
            obs_take(obs_hat, idx),
            label=f"Prediction / Output in Model | DataID={idx}",
            linestyle="--",
            color=color,
        )

    ax.text(
        0.05,
        0.75,
        "RMSE: {:10.4f}".format(m_rmse),
        transform=ax.transAxes,
        weight="bold",
    )

    ax.legend(handles=[line1, line2])

    return m_rmse


def _eval_model_2(ax, title, ylabel=None):
    ax.grid()
    ax.set_xlabel("time [s]")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(f"{title} Performance of Model")


def _eval_controller_1(
    ax,
    env,
    source,
    controller,
    loop_observer_config_xy=None,
    loop_observer_config: Optional[LoopObserverConfig] = None,
    rmse_center_right: bool = False,
    plot_noise: bool = True,
    mae: bool = False,
    fill_between: bool = False,
    fill_between_kwargs={},
):
    ts = timestep_array_from_env(env)

    observers = [DenoisifyObserver()]
    if loop_observer_config:
        observers.append(loop_observer_config.loop_observer)

    env_w_source = AddRefSignalRewardFnWrapper(env, source)
    sample, loop_results = collect_exhaust_source(
        env_w_source,
        controller,
        observers,
    )

    if plot_noise:
        obs = sample.obs["obs"]
    else:
        obs = loop_results["no_noise_obs"]

    model_or_env = "Environment"
    if isinstance(env, ReplacePhysicsByModelWrapper):
        model_or_env = "Model"

    if loop_observer_config is not None:
        loc = loop_observer_config
        loc_xy = loop_observer_config_xy
        inline_axes = plt.axes([loc_xy[0], loc_xy[1], 0.12, 0.05])
        inline_axes.set_ylabel(loc.y_label)
        inline_axes.grid()

    for idx in range(sample.bs):
        color = colors[idx]

        # dummy plots only for boxed legend
        if idx == 0:
            legend_color = "black"
            (line_ref,) = ax.plot(
                ts,
                obs_take(sample.obs["ref"], idx),
                color=legend_color,
                label="Reference",
            )
            (line_obs,) = ax.plot(
                ts,
                obs_take(obs, idx),
                linestyle="--",
                color=legend_color,
                label=f"Output in {model_or_env}",
            )

        ax.plot(ts, obs_take(sample.obs["ref"], idx), color=color)
        ax.plot(
            ts,
            obs_take(obs, idx),
            linestyle="--",
            color=color,
        )

        if fill_between:
            ax.fill_between(
                ts,
                obs_take(sample.obs["ref"], idx),
                obs_take(obs, idx),
                **fill_between_kwargs,
            )

        if loop_observer_config is not None:
            # left-right, top-bottom, width, heigt
            inline_axes.plot(
                ts,
                loc.arr_from_loop_results(loop_results, idx)[:-1],
                linestyle="--",
                color=color,
                dashes=(5, 1),
            )

    if mae:
        error = np.mean(np.sqrt(-sample.rew))
        label = "MAE"
    else:
        error = np.sqrt(np.mean(-sample.rew))
        label = "RMSE"

    mse_x, mse_y = 0.05, 0.95
    if rmse_center_right:
        mse_x, mse_y = 0.7, 0.45

    kwargs = {}
    if "color" in fill_between_kwargs:
        kwargs["color"] = fill_between_kwargs["color"]

    ax.text(
        mse_x,
        mse_y,
        "{}: {:10.4f}".format(label, error),
        transform=ax.transAxes,
        weight="bold",
        **kwargs,
    )

    # Create boxed legend
    ax.legend(handles=[line_ref, line_obs], loc="upper right")

    return {label: error}


def _eval_controller_2(ax, title: str, ylim: tuple[float] = None, ylabel=None):
    ax.grid(True)
    ax.set_xlabel("time [s]")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ymin=ylim[0], ymax=ylim[1])


@dataclass
class ExtraSource:
    source: ObservationReferenceSource
    name: str
    camera_id: str
    record_video: bool


def masterplot_siso(
    env,
    test_source,
    controller,
    filename: Optional[str] = None,
    extra_sources: list[ExtraSource] = [],
    controller_logs=None,
    controller_tracker_logs=None,
    model=None,
    model_logs=None,
    model_tracker_logs=None,
    train_sample=None,
    train_sample_plot_indices=[0, 1, 2],
    val_sample=None,
    val_sample_plot_indices=[0, 1, 2],
    path: str = "~/chain_control",
    experiment_id: str = "",
    loop_observer_config=None,
    plot_noise_extras: bool = False,
    use_mae: bool = False,
):
    results = {}
    ylabel = key_of_obs(env.observation_spec())
    ts = timestep_array_from_env(env)

    fig = plt.figure(figsize=myplotlib.figsize(1.5, 0.5))
    ax11 = plt.subplot(521)
    ax12 = plt.subplot(522)
    ax21 = plt.subplot(523)
    ax22 = plt.subplot(524)
    ax31 = plt.subplot(525)
    ax32 = plt.subplot(526, sharey=ax31)
    ax41 = plt.subplot(527)
    ax42 = plt.subplot(528)
    ax51 = plt.subplot(529)
    ax52 = fig.add_subplot(5, 2, 10)
    axes = np.array(
        [[ax11, ax12], [ax21, ax22], [ax31, ax32], [ax41, ax42], [ax51, ax52]]
    )

    for i, sample, indices, title in zip(
        [1, 0],
        [train_sample, val_sample],
        [train_sample_plot_indices, val_sample_plot_indices],
        ["Training", "Test"],
    ):
        ax = axes[i, 0]

        if model is not None and sample is not None:
            m_rmse = _eval_model_1(ax, sample, ts, model, indices)
            results[f"{title.lower()}_rmse_model"] = m_rmse
        _eval_model_2(ax, title, ylabel=ylabel)

    # PLOT [0,1] ####
    ax = axes[0, 1]

    if model:
        train_loss = model_logs["train_loss"]
        test_loss = model_logs["val_rmse"] ** 2
        ax.plot(train_loss, label="train_loss")
        ax.plot(test_loss, label="val_mse")

        assert len(model_tracker_logs) == 1
        for key, value in model_tracker_logs.items():
            ax.text(
                0.2,
                0.15,
                "Model used has {}={:10.4f}".format(key, float(value)),
                transform=ax.transAxes,
                weight="bold",
            )

        ax.text(
            0.2,
            0.25,
            "Last Trainings-Loss-value: {:10.4f}".format(float(train_loss[-1])),
            transform=ax.transAxes,
            weight="bold",
        )

        ax.text(
            0.2,
            0.35,
            "Last Validation-MSE-Value: {:10.4f}".format(float(test_loss[-1])),
            transform=ax.transAxes,
            weight="bold",
        )

        # subplot inside plot
        # zoomed in
        zoomed_in_after_steps = 0.25
        start = int(len(train_loss) * zoomed_in_after_steps)
        steps = np.arange(start=start, stop=len(train_loss))

        # left-right, top-bottom, width, heigt
        inline_axes = plt.axes([0.65, 0.9, 0.1, 0.07])
        inline_axes.plot(steps, train_loss[start:])
        if test_loss is not None:
            inline_axes.plot(steps, test_loss[start:])
        inline_axes.grid()

        zoomed_in_after_steps = 0.5
        start = int(len(train_loss) * zoomed_in_after_steps)
        steps = np.arange(start=start, stop=len(train_loss))

        inline_axes = plt.axes([0.81, 0.9, 0.1, 0.07])
        inline_axes.plot(steps, train_loss[start:])
        if test_loss is not None:
            inline_axes.plot(steps, test_loss[start:])
        inline_axes.grid()

    ax.grid()
    ax.set_title("Optimization History of Model")
    ax.set_xlabel("number of gradient steps")

    # PLOT [1,1] ####
    ax = axes[1, 1]

    if controller_logs:
        losses = np.squeeze(controller_logs["train_loss"])
        losses = np.atleast_1d(losses)

        N = len(losses)
        steps = np.arange(N)
        start = int(N * 0.1)

        ax.plot(steps, losses)

        ax.set_ylim((0, np.max(losses[start:])))

        ax.text(
            0.05,
            0.05,
            "Initial Loss-value: {:10.4f}".format(np.mean(losses[:5])),
            transform=ax.transAxes,
            weight="bold",
        )

        ax.text(
            0.05,
            0.15,
            "Last Loss-value: {:10.4f}".format(np.mean(losses[-1])),
            transform=ax.transAxes,
            weight="bold",
        )

        if controller_tracker_logs:
            assert (
                len(controller_tracker_logs) == 1
            ), f"""The Tracker logs should only contain one key, value pair that
            corresponds to the controller that has been used but got
            {len(controller_tracker_logs)} key-value pairs."""

            for key, value in controller_tracker_logs.items():
                ax.text(
                    0.2,
                    0.95,
                    "Controller used has {} = {:10.4f}".format(key, float(value)),
                    transform=ax.transAxes,
                    weight="bold",
                )

    ax.set_ylabel("Training Loss of Controller")
    ax.grid()
    ax.set_title("Optimization History of Controller using Model")
    ax.set_xlabel("number of gradient steps")

    # PLOT [2,0] ####
    if model:
        env_model = ReplacePhysicsByModelWrapper(env, model)
        label_and_error = _eval_controller_1(
            axes[2, 0], env_model, test_source, controller, mae=use_mae
        )
        label = [key for key in label_and_error][0]
        results[f"train_{label.lower()}_controller"] = label_and_error[label]
    _eval_controller_2(axes[2, 0], "Training References (Model)", ylabel=ylabel)

    # PLOT [2,1] ####
    label_and_error = _eval_controller_1(
        axes[2, 1], env, test_source, controller, plot_noise=True, mae=use_mae
    )
    label = [key for key in label_and_error][0]
    results[f"test_{label.lower()}_controller"] = label_and_error[label]
    _eval_controller_2(axes[2, 1], "Training References (Simulation)", ylabel=ylabel)

    ANLGE_SUBPLOT_X_ANCHOR = 0.12
    ANLGE_SUBPLOT_X_DELTA = 0.49
    ANLGE_SUBPLOT_Y_ANCHOR = 0.325
    ANGLE_SUBPLOT_Y_DELTA = -0.2

    for i, j, extra in zip([0, 0, 1, 1], [0, 1, 0, 1], extra_sources):
        env_video = None
        if extra.record_video:
            path_video = os.path.join(path, experiment_id)
            env_video = VideoWrapper(
                env,
                path_video,
                extra.name,
                camera_id=extra.camera_id,
                add_uid_to_path=False,
            )

        xy = (
            ANLGE_SUBPLOT_X_ANCHOR + j * ANLGE_SUBPLOT_X_DELTA,
            ANLGE_SUBPLOT_Y_ANCHOR + i * ANGLE_SUBPLOT_Y_DELTA,
        )
        ax = axes[i + 3, j]
        label_and_error = _eval_controller_1(
            ax,
            env_video if env_video else env,
            extra.source,
            controller,
            xy,
            loop_observer_config,
            True,
            plot_noise=plot_noise_extras,
            mae=use_mae,
        )
        label = [key for key in label_and_error][0]
        results[extra.name + "_" + label.lower()] = label_and_error[label]
        _eval_controller_2(ax, extra.name, ylabel=ylabel)

        # Background color for test plots
        ax.set_facecolor("green")
        ax.patch.set_alpha(0.05)

    # END OF PLOTS

    fig.tight_layout(pad=0.33)
    fig.align_labels()

    if filename is not None:
        path_to_folder = process_path(path, experiment_id, "masterplots", add_uid=False)
        path_figure = os.path.join(path_to_folder, filename)
        myplotlib.savefig(path_figure, tight=False, transparent=False)

    return results
