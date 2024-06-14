from pathlib import Path

import fire
import numpy as np
from train_controller import _load_lti_from_path
from train_controller import _load_refs_from_path_to_folder
from train_controller import _make_env
from train_controller import listdir

from cc.core import load_eqx
from cc.env.collect import collect_exhaust_source
from cc.env.wrappers import AddRefSignalRewardFnWrapper
from cc.env.wrappers import ReplacePhysicsByModelWrapper


def main(
    path_folder_controller: str,
    path_dynamics_npy: str,
    path_folder_refs: str,
    path_output_folder: str,
):
    Path(path_output_folder).mkdir(parents=True, exist_ok=True)
    assert len(listdir(path_output_folder)) == 0, "output folder is not empty"

    m = _load_lti_from_path(path_dynamics_npy)
    source = _load_refs_from_path_to_folder(path_folder_refs)
    time_limit = (source.get_references_for_optimisation()["output"].shape[1] - 1) / 100
    env = _make_env(time_limit)
    env_m = ReplacePhysicsByModelWrapper(env, m)
    env_m_source = AddRefSignalRewardFnWrapper(env_m, source)

    c = load_eqx(Path(path_folder_controller).joinpath("controller.pkl"))
    sample, loop_results = collect_exhaust_source(env_m_source, c)

    def save(filename, arr):
        np.save(Path(path_output_folder).joinpath(filename), arr)

    save("input", sample.action)
    save("output", sample.obs["obs"]["output"])
    save("reference", sample.obs["ref"]["output"])


if __name__ == "__main__":
    fire.Fire(main)
