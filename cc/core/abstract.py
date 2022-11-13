from abc import ABC, abstractmethod
from typing import Optional, Tuple

import equinox as eqx
import jax.numpy as jnp

from .types import BatchedTimeSeriesOfRef, PyTree, TimeSeriesOfRef


class AbstractObservationReferenceSource(ABC):
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


class AbstractModule(eqx.Module, ABC):
    @abstractmethod
    def __call__(
        self, x: Optional[PyTree[jnp.ndarray]] = None
    ) -> Tuple["AbstractModule", PyTree[jnp.ndarray]]:
        pass

    @abstractmethod
    def reset(self) -> "AbstractModule":
        pass
