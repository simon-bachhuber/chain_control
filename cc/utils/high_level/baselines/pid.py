"""Best PID-Gains for various environments."""

from cc.env import make_env

best_pd = {
    make_env("rover"): {"PD": {"P": 1.0, "D": 1.836274}},
    # these were from before i switched from degrees to rad
    # thuse they should now be scaled with np.rad2deg
    make_env("muscle_asymmetric"): {"PD": {"D": 0.000273, "P": 0.004213}},
    make_env("muscle_asymmetric", physics_kwargs={"corner": 0.03}): {
        "PD": {"P": 0.026084, "D": 0.005709}
    },
    make_env("two_segments_v2"): {"PD": {"P": 0.08, "D": 0.08}},
}
