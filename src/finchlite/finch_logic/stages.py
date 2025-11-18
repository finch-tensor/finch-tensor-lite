from abc import ABC, abstractmethod

from finchlite.algebra.tensor import TensorFType

from .. import finch_notation as ntn
from ..symbolic import Stage
from . import nodes as lgc


class LogicEvaluator(Stage):
    @abstractmethod
    def __call__(self, term: lgc.LogicNode, bindings: dict[lgc.Alias, lgc.TableValue]|None=None) -> lgc.TableValue | tuple[lgc.TableValue]:
        """
        Evaluate the given logic.
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
