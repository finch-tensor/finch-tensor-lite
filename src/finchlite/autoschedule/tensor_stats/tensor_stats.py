from __future__ import annotations

import copy
import math
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any, Generic, Self, TypeVar

import numpy as np

from finchlite.algebra import (
    FinchOperator,
    is_idempotent,
    is_identity,
    repeat_operator,
)
from finchlite.finch_logic import (
    Alias,
    Field,
    Literal,
    MapJoin,
    StatsFactory,
    Table,
    TensorStats,
)

TS = TypeVar("TS", bound="BaseTensorStats")


class BaseTensorStats(TensorStats):
    """
    Base class for tensor statistics.
    """

    def __init__(self, tensor: Any, fields: tuple[Field, ...]):
        """Build the definition state by reading the shape and fill value off
        of ``tensor``.
        """
        shape = tensor.shape
        self._index_order = tuple(fields)
        self._dim_sizes: OrderedDict[Field, float] = OrderedDict(
            (axis, float(shape[i])) for i, axis in enumerate(fields)
        )
        self._fill_value = tensor.fill_value

    @classmethod
    def from_fields(
        cls,
        index_order: Iterable[Field],
        dim_sizes: Mapping[Field, float],
        fill_value: Any,
    ) -> Self:
        """Build an instance directly from raw definition fields, bypassing
        ``__init__`` (which expects a tensor).
        """
        obj = object.__new__(cls)
        obj._index_order = tuple(index_order)
        obj._dim_sizes = OrderedDict(dim_sizes)
        obj._fill_value = fill_value
        return obj

    @classmethod
    def from_def(cls, d: BaseTensorStats, **fields: Any) -> Self:
        """Build a ``cls`` instance that reuses the definition state (index
        order, dimension sizes, fill value) of ``d``, setting any
        subclass-specific attributes from ``fields``.
        """
        obj = cls.from_fields(d.index_order, d.dim_sizes, d.fill_value)
        for name, value in fields.items():
            setattr(obj, name, value)
        return obj

    def copy(self) -> Self:
        new = object.__new__(type(self))
        new.__dict__ = {name: copy.copy(value) for name, value in self.__dict__.items()}
        return new

    def set_fill_value(self, fill_value: Any) -> BaseTensorStats:
        """Return a new definition with ``fill_value`` substituted in."""
        return BaseTensorStats.from_fields(
            self._index_order, self._dim_sizes, fill_value
        )

    def add_dummy_idx(self, idx: Field) -> BaseTensorStats:
        """Return a new definition with a size-1 axis ``idx`` appended.

        If ``idx`` already exists, the object is returned unchanged.
        """
        if idx in self.index_order:
            return self

        new_index_order = self.index_order + (idx,)
        new_dim_sizes = dict(self.dim_sizes)
        new_dim_sizes[idx] = 1.0

        return BaseTensorStats.from_fields(
            new_index_order, new_dim_sizes, self.fill_value
        )

    @property
    def dim_sizes(self) -> Mapping[Field, float]:
        return self._dim_sizes

    @dim_sizes.setter
    def dim_sizes(self, value: Mapping[Field, float]):
        self._dim_sizes = OrderedDict(value)

    def get_dim_size(self, idx: Field) -> float:
        return self.dim_sizes[idx]

    @property
    def index_order(self) -> tuple[Field, ...]:
        return self._index_order

    @index_order.setter
    def index_order(self, value: Iterable[Field]):
        self._index_order = tuple(value)

    @property
    def idxs(self) -> tuple[Field, ...]:
        return self.index_order

    @property
    def fill_value(self) -> Any:
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value: Any):
        self._fill_value = value

    def get_dim_space_size(self, idx: Iterable[Field]) -> float:
        prod = 1
        for i in idx:
            prod *= int(self.dim_sizes[i])
            if prod == 0 or prod > np.iinfo(np.int64).max:
                return float("inf")
        return float(prod)


class BaseTensorStatsFactory(StatsFactory[TS], Generic[TS]):
    def __init__(self, stats_cls: type[TS]):
        self.stats_cls = stats_cls

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> TS:
        return self.stats_cls(tensor, fields)

    def copy(self, stat: TS) -> TS:
        if not isinstance(stat, self.stats_cls):
            raise TypeError(f"copy expected a {self.stats_cls.__name__} instance")
        return stat.copy()

    def mapjoin(self, op: FinchOperator, *args: TS) -> TS:
        new_def = self.merge_defs(op, *args)

        join_args: list[TS] = []
        union_args: list[TS] = []
        for s in args:
            if op.is_annihilator(s.fill_value):
                join_args.append(s)
            else:
                union_args.append(s)

        if union_args:
            join_args.append(self._mapjoin_union(new_def, op, union_args))
            # Add test cases - To test both join and union

        return self._mapjoin_join(new_def, op, join_args)

    @abstractmethod
    def _mapjoin_union(
        self, new_def: BaseTensorStats, op: FinchOperator, union_args: list[TS]
    ) -> TS: ...

    @abstractmethod
    def _mapjoin_join(
        self, new_def: BaseTensorStats, op: FinchOperator, join_args: list[TS]
    ) -> TS: ...

    @abstractmethod
    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: TS,
    ) -> TS: ...

    @abstractmethod
    def relabel(self, stats: TS, relabel_indices: tuple[Field, ...]) -> TS: ...

    @abstractmethod
    def reorder(self, stats: TS, reorder_indices: tuple[Field, ...]) -> TS: ...

    @staticmethod
    def merge_defs(op: FinchOperator, *args: BaseTensorStats) -> BaseTensorStats:
        new_fill_value = op(*(s.fill_value for s in args))
        new_index_order = MapJoin(
            Literal(op),
            tuple(
                Table(Alias(f"_{i}"), tuple(a.index_order)) for i, a in enumerate(args)
            ),
        ).fields()
        new_dim_sizes: dict = {}
        for index in new_index_order:
            for s in args:
                if index in s.index_order:
                    new_dim_sizes[index] = s.dim_sizes[index]
                    break
        assert set(new_dim_sizes.keys()) == set(new_index_order)
        return BaseTensorStats.from_fields(
            new_index_order, new_dim_sizes, new_fill_value
        )

    @staticmethod
    def aggregate_def(
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        d: BaseTensorStats,
    ) -> BaseTensorStats:
        red_set = set(reduce_indices) & set(d.index_order)
        n = math.prod(int(d.dim_sizes[x]) for x in red_set)

        if init is None:
            if is_identity(op, d.fill_value) or is_idempotent(op):
                init = op(d.fill_value, d.fill_value)
            else:
                try:
                    init = repeat_operator(op)(d.fill_value, n)
                except AttributeError:
                    # This is going to be VERY SLOW. Should raise a warning about
                    #  reductions over non-identity fill values. Depending on the
                    # semantics of reductions, we might be able to do this faster.
                    print(
                        "Warning: A reduction can take place over a tensor whose fill"
                        "value is not the reduction operator's identity. This can"
                        "result in a large slowdown as the new fill is calculated."
                    )
                    acc = d.fill_value
                    for _ in range(max(n - 1, 0)):
                        acc = op(acc, d.fill_value)
                    init = acc

        new_dim_sizes = OrderedDict(
            (ax, d.dim_sizes[ax]) for ax in d.dim_sizes if ax not in red_set
        )
        new_index_order = tuple(new_dim_sizes)
        return BaseTensorStats.from_fields(new_index_order, new_dim_sizes, init)

    @staticmethod
    def relabel_def(
        d: BaseTensorStats, relabel_indices: tuple[Field, ...]
    ) -> BaseTensorStats:
        if len(relabel_indices) != len(d.index_order):
            raise ValueError(
                f"Tensor has {len(d.index_order)} dims, "
                f"but {len(relabel_indices)} names provided."
            )

        new_dim_sizes = OrderedDict(
            zip(relabel_indices, d.dim_sizes.values(), strict=True)
        )

        return BaseTensorStats.from_fields(relabel_indices, new_dim_sizes, d.fill_value)

    @staticmethod
    def reorder_def(
        d: BaseTensorStats, reorder_indices: tuple[Field, ...]
    ) -> BaseTensorStats:
        for old_idx in d.index_order:
            if old_idx not in set(reorder_indices) and d.get_dim_size(old_idx) != 1:
                raise ValueError(
                    f"Trying to drop dimension '{old_idx}' of size"
                    f" {d.get_dim_size(old_idx)}."
                    " Only size 1 dimensions can be dropped."
                )

        new_dims: OrderedDict[Field, float] = OrderedDict()
        for idx in reorder_indices:
            if idx in d.index_order:
                new_dims[idx] = d.get_dim_size(idx)
            else:
                new_dims[idx] = 1

        return BaseTensorStats.from_fields(reorder_indices, new_dims, d.fill_value)
