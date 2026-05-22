from abc import abstractmethod

from finchlite.algebra.tensor import TensorFType

from .. import finch_einsum as ein
from .. import finch_logic as lgc
from .. import finch_notation as ntn
from ..symbolic import Stage


class LogicNotationLowerer(Stage):
    @abstractmethod
    def transform(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, TensorFType]
    ) -> tuple[ntn.Module]:
        """
        Generate Finch Notation from the given logic and input types.  Also
        return a dictionary including additional tables needed to run the kernel.
        """


class LogicEinsumLowerer(Stage):
    @abstractmethod
    def transform(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, TensorFType]
    ) -> tuple[ein.EinsumStatement, dict[ein.Alias, TensorFType]]:
        """
        Generate Einsum Notation from the given logic and input types.  Also
        return a dictionary including additional tables needed to run the kernel.
        """
