from abc import ABC, abstractmethod

from .. import finch_einsum as ein
from .. import finch_notation as ntn
from ..finch_assembly import AssemblyLibrary
from ..symbolic import Stage
from . import nodes as lgc
from .nodes import TableValueFType


class LogicEvaluator(Stage):
    @abstractmethod
    def __call__(
        self,
        term: lgc.LogicNode,
        bindings: dict[lgc.Alias, lgc.TableValue] | None = None,
    ) -> lgc.TableValue | tuple[lgc.TableValue, ...]:
        """
        Evaluate the given logic.
        """


class LogicLoader(ABC):
    @abstractmethod
    def __call__(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, TableValueFType]
    ) -> tuple[AssemblyLibrary, dict[lgc.Alias, TableValueFType]]:
        """
        Generate Finch Library from the given logic and input types, with a
        single method called main which implements the logic. Also return a
        dictionary including additional tables needed to run the kernel.
        """


class LogicNotationLowerer(ABC):
    @abstractmethod
    def __call__(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, TableValueFType]
    ) -> tuple[ntn.Module, dict[lgc.Alias, TableValueFType]]:
        """
        Generate Finch Notation from the given logic and input types.  Also
        return a dictionary including additional tables needed to run the kernel.
        """


class LogicEinsumLowerer(ABC):
    @abstractmethod
    def __call__(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, TableValueFType]
    ) -> tuple[ein.EinsumNode, dict[lgc.Alias, TableValueFType]]:
        """
        Generate Finch Einsum from the given logic and input types,
        types for all aliases.
        """


class LogicTransform(ABC):
    @abstractmethod
    def __call__(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, TableValueFType]
    ) -> tuple[lgc.LogicStatement, dict[lgc.Alias, TableValueFType]]:
        """
        Transform the given logic term into another logic term.
        """


class OptLogicLoader(LogicLoader):
    def __init__(self, *opts: LogicTransform, ctx: LogicLoader):
        self.ctx = ctx
        self.opts = opts

    def __call__(
        self,
        term: lgc.LogicStatement,
        bindings: dict[lgc.Alias, lgc.TableValueFType],
    ) -> tuple[AssemblyLibrary, dict[lgc.Alias, lgc.TableValueFType]]:
        for opt in self.opts:
            term, bindings = opt(term, bindings or {})
        return self.ctx(term, bindings)
