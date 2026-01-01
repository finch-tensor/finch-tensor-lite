from abc import ABC, abstractmethod
from typing import Any

from ..algebra import TensorFType
from ..finch_assembly import AssemblyLibrary
from ..symbolic import Stage
from . import nodes as ein


class EinsumEvaluator(Stage):
    @abstractmethod
    def __call__(
        self,
        term: ein.EinsumNode,
        bindings: dict[ein.Alias, Any] | None = None,
    ) -> Any | tuple[Any, ...]:  # TODO eventually Any->Tensor
        """
        Evaluate the given logic.
        """


class EinsumLoader(ABC):
    @abstractmethod
    def __call__(
        self, term: ein.EinsumStatement, bindings: dict[ein.Alias, TensorFType]
    ) -> tuple[AssemblyLibrary, ein.EinsumStatement, dict[ein.Alias, TensorFType]]:
        """
        Generate Finch Library from the given logic and input types, with a
        single method called main which implements the logic. Also return a
        dictionary including additional tables needed to run the kernel.
        """


class EinsumTransform(ABC):
    @abstractmethod
    def __call__(
        self, term: ein.EinsumStatement, bindings: dict[ein.Alias, TensorFType]
    ) -> tuple[ein.EinsumStatement, dict[ein.Alias, TensorFType]]:
        """
        Transform the given logic term into another logic term.
        """


class OptEinsumLoader(EinsumLoader):
    def __init__(self, *opts: EinsumTransform, ctx: EinsumLoader):
        self.ctx = ctx
        self.opts = opts

    def __call__(
        self,
        term: ein.EinsumStatement,
        bindings: dict[ein.Alias, TensorFType],
    ) -> tuple[AssemblyLibrary, ein.EinsumStatement, dict[ein.Alias, TensorFType]]:
        for opt in self.opts:
            term, bindings = opt(term, bindings or {})
        return self.ctx(term, bindings)
