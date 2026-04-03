# AI modified: 2026-04-03T00:24:22Z 7e517b16f3803378be07f55bd66f95bd09981f0c
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Self

from finchlite.finch_logic import Field, TensorStats

from ...algebra import FinchOperator
from .tensor_def import TensorDef


class BaseTensorStats(TensorStats):
    tensordef: TensorDef

    def __init__(self, tensor: Any, fields: tuple[Field, ...]):
        self.tensordef = TensorDef.from_tensor(tensor, fields)

    @classmethod
    @abstractmethod
    def copy_stats(cls, stat: Self) -> Self:
        """
        Return a copy of a TensorStats object.
        """
        ...

    @classmethod
    @abstractmethod
    def mapjoin(cls, op: FinchOperator, *args: Self) -> Self:
        """
        Return a new statistic representing the tensor resulting
        from calling op on args... in an elementwise fashion
        """
        ...

    @classmethod
    @abstractmethod
    def aggregate(
        cls,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: Self,
    ) -> Self:
        """
        Return a new statistic representing the tensor resulting
        from aggregating arg over fields with the op aggregation function
        """
        ...

    @classmethod
    @abstractmethod
    def issimilar(cls, a: Self, b: Self) -> bool:
        """
        Returns whether two statistics objects represent similarly distributed tensors,
        and only returns true if the tensors have the same dimensions and fill value
        """
        ...

    @classmethod
    @abstractmethod
    def relabel(cls, stats: Self, relabel_indices: tuple[Field, ...]) -> Self:
        """ """
        ...

    @classmethod
    @abstractmethod
    def reorder(cls, stats: Self, reorder_indices: tuple[Field, ...]) -> Self:
        """ """
        ...

    @property
    def dim_sizes(self) -> Mapping[Field, float]:
        return self.tensordef.dim_sizes

    @dim_sizes.setter
    def dim_sizes(self, value: Mapping[Field, float]):
        self.tensordef.dim_sizes = value

    def get_dim_size(self, idx: Field) -> float:
        return self.tensordef.get_dim_size(idx)

    @property
    def index_order(self) -> tuple[Field, ...]:
        return self.tensordef.index_order

    @index_order.setter
    def index_order(self, value: tuple[Field, ...]):
        self.tensordef.index_order = value

    @property
    def fill_value(self) -> Any:
        return self.tensordef.fill_value

    @fill_value.setter
    def fill_value(self, value: Any):
        self.tensordef.fill_value = value

    def get_dim_space_size(self, idx: Iterable[Field]):
        return self.tensordef.get_dim_space_size(idx)
