from abc import ABC, abstractmethod

from .types import *


class AbstractObservationReferenceSource(ABC):
    @abstractmethod
    def get_reference_actor(self) -> TimeSeriesOfRef:
        pass 

    @abstractmethod
    def get_references_for_optimisation(self) -> BatchedTimeSeriesOfRef:
        pass 

    def change_reference_of_actor(self, i: int) -> None:
        raise NotImplementedError 

    def change_references_for_optimisation(self) -> None:
        raise NotImplementedError


S = TypeVar("S")
S_w_key = Tuple[S, NotAParameter[PRNGKey]]
X = TypeVar("X")
Y = TypeVar("Y")


AbstractRHS = TypeVar("AbstractRHS")
class AbstractRHS(eqx.Module, ABC):

    @abstractmethod
    def __call__(self, state: S, x: X) -> Tuple[S, Y]:
        pass 

    @abstractmethod
    def init_state(self) -> PossibleParameter[S]:
        pass 


AbstractWrappedRHS = TypeVar("AbstractWrappedRHS")
class AbstractWrappedRHS(eqx.Module, ABC):

    @abstractmethod
    def __call__(self, x: PyTree) -> Tuple[AbstractWrappedRHS, PyTree]:
        pass 

    @abstractmethod
    def reset(self) -> AbstractWrappedRHS:
        pass 

    #@abstractproperty
    def input_size(self):
        raise NotImplementedError() 

    #@abstractproperty
    def output_size(self):
        raise NotImplementedError()


class AbstractController(AbstractWrappedRHS):
    pass 


class AbstractModel(AbstractWrappedRHS):
    @abstractmethod
    def y0(self) -> Observation:
        """Initial Observation of Model
        """
        pass  

