from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as random
import jax.tree_util as jtu
from tree_utils import PyTree, add_batch_dim, tree_batch, tree_map_flat

from .module_utils import filter_scan_module
from .types import BatchedTimeSeriesOfRef, TimeSeriesOfRef


class AbstractObservationReferenceSource(ABC):
    def __init__(self) -> None:
        self._ts = None

    @abstractmethod
    def get_reference_actor(self) -> TimeSeriesOfRef:
        pass

    @abstractmethod
    def get_references_for_optimisation(self) -> BatchedTimeSeriesOfRef:
        pass

    @abstractmethod
    def change_reference_of_actor(self, i: int) -> None:
        pass

    def change_references_for_optimisation(self) -> None:
        raise NotImplementedError


class AbstractModel(eqx.Module, ABC):
    @abstractmethod
    def step(
        self, x: PyTree[jnp.ndarray]
    ) -> Tuple["AbstractModel", PyTree[jnp.ndarray]]:
        pass

    @abstractmethod
    def grad_filter_spec(self) -> PyTree[bool]:
        return jtu.tree_map(lambda leaf: eqx.is_array(leaf), self)

    @abstractmethod
    def reset(self) -> "AbstractModel":
        pass

    def unroll(
        self, time_series_of_x: PyTree[jnp.ndarray], include_y0: bool = True
    ) -> PyTree[jnp.ndarray]:
        model = self.reset()

        def scan_fn(model, x):
            model, y = model.step(x)
            return model, y

        time_series_of_y = filter_scan_module(scan_fn, model, time_series_of_x, None)[1]

        if include_y0:
            return tree_batch(
                [add_batch_dim(model.y0()), time_series_of_y], True, "jax"
            )
        else:
            return time_series_of_y

    @abstractmethod
    def y0(self) -> PyTree[jnp.ndarray]:
        pass


def _add_noise_to_output(arr, key, noise_scale):
    noise = random.normal(key, shape=arr.shape, dtype=arr.dtype) * noise_scale
    return arr + noise


class AbstractController(eqx.Module, ABC):
    @abstractmethod
    def step(
        self, x: PyTree[jnp.ndarray]
    ) -> Tuple["AbstractController", PyTree[jnp.ndarray]]:
        pass

    @abstractmethod
    def grad_filter_spec(self) -> PyTree[bool]:
        return jtu.tree_map(lambda leaf: eqx.is_array(leaf), self)

    @abstractmethod
    def reset(self) -> "AbstractController":
        pass

    def unroll(
        self,
        model: AbstractModel,
        merge_x_y: Callable,
        noise_scale: Optional[float] = None,
    ) -> Callable:
        def unroll_closed_loop(
            time_series_of_x: PyTree[jnp.ndarray], key: random.PRNGKey
        ) -> PyTree[jnp.ndarray]:
            def scan_fn(carry, x):
                (controller, model, y, key) = carry
                controller, u = controller.step(merge_x_y(x, y))
                model, y = model.step(u)

                # add noise to output
                if noise_scale is not None:
                    key, consume = random.split(key)
                    y = tree_map_flat(y, _add_noise_to_output, consume, noise_scale)

                return (controller, model, y, key), (y, u)

            (time_series_of_y, time_series_of_u) = filter_scan_module(
                scan_fn,
                (self.reset(), model.reset(), model.y0(), key),
                time_series_of_x,
                None,
            )[1]

            return (time_series_of_y, time_series_of_u)[0]

        return unroll_closed_loop
