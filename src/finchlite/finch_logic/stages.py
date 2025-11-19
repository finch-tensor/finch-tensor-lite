from abc import ABC, abstractmethod

from ..algebra import TensorFType
from ..finch_assembly import AssemblyLibrary

from .. import finch_notation as ntn
from ..symbolic import Stage
from . import nodes as lgc


class LogicEvaluator(Stage):
    @abstractmethod
    def __call__(self, term: lgc.LogicNode, bindings: dict[lgc.Alias, lgc.TableValue]|None=None) -> lgc.TableValue | tuple[lgc.TableValue]:
        """
        Evaluate the given logic.
        """


class LogicLoader(ABC):
    @abstractmethod
    def __call__(
        self, term: lgc.LogicNode, bindings: dict[lgc.Alias, TensorFType]
    ) -> AssemblyLibrary:
        """
        Generate Finch Library from the given logic and input types,
        with a single method called main which implements the logic.
        """


class LogicLowerer(ABC):
    @abstractmethod
    def __call__(
        self, term: lgc.LogicNode, bindings: dict[lgc.Alias, TensorFType]
    ) -> tuple[ntn.Module, dict[lgc.Alias, TensorFType]]:
        """
        Generate Finch Notation from the given logic and input types,
        types for all aliases.
        """


class LogicTransform(ABC):
    @abstractmethod
    def __call__(self, term: lgc.LogicNode, bindings: dict[lgc.Alias, TensorFType]) -> tuple[lgc.LogicNode, dict[lgc.Alias, TensorFType]]:
        """
        Transform the given logic term into another logic term.
        """
