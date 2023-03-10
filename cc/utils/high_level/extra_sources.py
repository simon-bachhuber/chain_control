import numpy as np

from cc.env import make_env
from cc.env.collect import (
    constant_after_transform_source,
    double_step_source,
    high_steps_source,
    sample_feedforward_collect_and_make_source,
)
from cc.env.loop_observer import AnglesEnvLoopObserver
from cc.env.loop_observer.ackermann_phi_observer import AckermannPhiObserver

from .masterplot_siso import ExtraSource, LoopObserverConfig

two_segments = LoopObserverConfig(
    AnglesEnvLoopObserver(),
    "Pendulum Angle [deg]",
    lambda lr, idx: lr["hinge_1 [deg]"][idx] + lr["hinge_2 [deg]"][idx],
)
ackermann = LoopObserverConfig(
    AckermannPhiObserver(), "Car Angle [deg]", lambda lr, idx: lr["phi [deg]"][idx]
)

loop_observer_configs = {
    "two_segments_v2": two_segments,
    "two_segments": two_segments,
    "rover": None,
    "muscle_asymmetric": None,
    "ackermann": ackermann,
}


def build_extra_sources(env_id: str, record_video):
    if env_id == "two_segments_v2" or "two_segments":
        camera_id = "skyview"
        high_amp = 6.0
        step_amp = 2.0
    elif env_id == "ackermann":
        camera_id = "skyview"
        high_amp = 6.0
        step_amp = 2.0
    elif env_id == "rover":
        camera_id = "target"
        high_amp = 5.0
        step_amp = 2.0
    elif env_id == "muscle_asymmetric":
        camera_id = "upfront"
        high_amp = np.deg2rad(110)
        step_amp = np.deg2rad(45)
    else:
        raise NotImplementedError()

    env = make_env(env_id)

    smooth_source, _, _ = sample_feedforward_collect_and_make_source(env, seeds=[1, 2])
    smooth_source_constant = constant_after_transform_source(smooth_source, 5.0)

    extras = [
        ExtraSource(
            high_steps_source(env, high_amp), "high_amplitude", camera_id, record_video
        ),
        ExtraSource(
            double_step_source(env, step_amp), "double_steps", camera_id, record_video
        ),
        ExtraSource(smooth_source, "smooth_refs", camera_id, record_video),
        ExtraSource(
            smooth_source_constant, "smooth_to_constant_refs", camera_id, record_video
        ),
    ]

    return extras
