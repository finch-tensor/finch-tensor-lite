from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .nodes import Field


class AbstractStats(ABC):
    @staticmethod
    @abstractmethod
    def copy_stats(stat: "AbstractStats") -> "AbstractStats":
        """
        Return a copy of a AbstractStats object.
        """
        ...

    @staticmethod
    @abstractmethod
    def mapjoin(op: Callable, *args: "AbstractStats") -> "AbstractStats":
        """
        Return a new statistic representing the tensor resulting
        from calling op on args... in an elementwise fashion
        """
        ...

    @staticmethod
    @abstractmethod
    def aggregate(
        op: Callable[..., Any],
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: "AbstractStats",
    ) -> "AbstractStats":
        """
        Return a new statistic representing the tensor resulting
        from aggregating arg over fields with the op aggregation function
        """
        ...

    @staticmethod
    @abstractmethod
    def issimilar(a: "AbstractStats", b: "AbstractStats") -> bool:
        """
        Returns whether two statistics objects represent similarly distributed tensors,
        and only returns true if the tensors have the same dimensions and fill value
        """
        ...

    @staticmethod
    @abstractmethod
    def relabel(
        stats: "AbstractStats", relabel_indices: tuple[Field, ...]
    ) -> "AbstractStats":
        """ """
        ...

    @staticmethod
    @abstractmethod
    def reorder(
        stats: "AbstractStats", reorder_indices: tuple[Field, ...]
    ) -> "AbstractStats":
        """ """
        ...

    @property
    @abstractmethod
    def idxs(self) -> tuple[Field, ...]: ...

    @property
    @abstractmethod
    def fill_value(self) -> Any: ...
