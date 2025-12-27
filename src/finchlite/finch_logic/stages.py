from __future__ import annotations

from abc import ABC, abstractmethod

from finchlite.algebra.tensor import Tensor, TensorFType

from .. import finch_einsum as ein
from ..finch_assembly import AssemblyLibrary
from ..symbolic import Stage
from . import nodes as lgc


class LogicEvaluator(Stage):
    @abstractmethod
    def __call__(
        self,
        term: lgc.LogicNode,
        bindings: dict[lgc.Alias, Tensor] | None = None,
    ) -> lgc.TableValue | tuple[Tensor, ...]:
        """
        Evaluate the given logic.
        """


class LogicLoader(ABC):
    @abstractmethod
    def __call__(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, TensorFType]
    ) -> tuple[AssemblyLibrary, lgc.LogicStatement, dict[lgc.Alias, TensorFType]]:
        """
        Generate Finch Library from the given logic and input types, with a
        single method called main which implements the logic. Also return a
        dictionary including additional tables needed to run the kernel.
        """


class LogicEinsumLowerer(ABC):
    @abstractmethod
    def __call__(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, TensorFType]
    ) -> tuple[ein.EinsumNode, dict[lgc.Alias, TensorFType]]:
        """
        Generate Finch Einsum from the given logic and input types,
        types for all aliases.
        """


class LogicTransform(ABC):
    @abstractmethod
    def __call__(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, TensorFType]
    ) -> tuple[lgc.LogicStatement, dict[lgc.Alias, TensorFType]]:
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
        bindings: dict[lgc.Alias, TensorFType],
    ) -> tuple[AssemblyLibrary, lgc.LogicStatement, dict[lgc.Alias, TensorFType]]:
        for opt in self.opts:
            term, bindings = opt(term, bindings or {})
        return self.ctx(term, bindings)
