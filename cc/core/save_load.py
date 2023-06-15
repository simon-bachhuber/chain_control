import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import cloudpickle
import equinox as eqx


@dataclass
class SavedEqxModule:
    init_fn: Callable
    params_path: str

    def load(self):
        module = self.init_fn()
        return eqx.tree_deserialise_leaves(self.params_path, module)


def save_eqx(path, trained_obj: eqx.Module, init_fn: Callable):
    "path should be without file extension"
    params_path = path + "_params.eqx"
    eqx.tree_serialise_leaves(params_path, trained_obj)
    saved_module = SavedEqxModule(init_fn, params_path)
    save(saved_module, path + ".pkl")


def load_eqx(path):
    path = os.path.expanduser(path)
    trained_module = load(path)
    return trained_module.load()


def load(path):
    path = os.path.expanduser(path)
    with open(path, "rb") as file:
        obj = cloudpickle.load(file)
    return obj


def save(obj, path, metadata={}, verbose=True):
    if isinstance(obj, eqx.Module):
        raise Exception(
            """Not possible. Use `eqx.tree_serialise_leaves(path, obj)` instead.
            To de-serialise use `eqx.tree_deserialise_leaves`."""
        )

    if metadata == {}:
        with open(path, "wb") as file:
            cloudpickle.dump(obj, file)
        return

    if isinstance(metadata, dict):
        metadata = SimpleNamespace(**metadata)

    obj_w_metadata = SimpleNamespace()
    obj_w_metadata.obj = obj
    obj_w_metadata.meta = metadata
    with open(path, "wb") as file:
        cloudpickle.dump(obj_w_metadata, file)

    if verbose:
        print(f"Saving object {type(obj)} as `{path}`.")
