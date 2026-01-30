import math
import operator
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from ...algebra import is_idempotent, is_identity


class TensorDef:
    def __init__(
        self,
        index_set: Iterable[str],
        dim_sizes: Mapping[str, float],
        fill_value: Any,
    ):
        self._index_set = set(index_set)
        self._dim_sizes = OrderedDict(dim_sizes)
        self._fill_value = fill_value

    def copy(self) -> "TensorDef":
        """
        Return:
            Deep copy of TensorDef fields
        """
        return TensorDef(
            index_set=self._index_set.copy(),
            dim_sizes=self._dim_sizes.copy(),
            fill_value=self._fill_value,
        )

    @classmethod
    def from_tensor(cls, tensor: Any, indices: Iterable[str]) -> "TensorDef":
        """
        Storing axis, sizes, and fill_value of the tensor

        """
        shape = tensor.shape
        dim_sizes = OrderedDict(
            (axis, float(shape[i])) for i, axis in enumerate(indices)
        )
        fv = tensor.fill_value

        return cls(
            index_set=indices,
            dim_sizes=dim_sizes,
            fill_value=fv,
        )

    def reindex_def(self, new_axis: Iterable[str]) -> "TensorDef":
        """
        Return
            :TensorDef with a new reindexed index_set and dim sizes
        """
        new_axis = list(new_axis)
        new_dim_sizes = OrderedDict((axis, self.dim_sizes[axis]) for axis in new_axis)
        return TensorDef(
            index_set=new_axis,
            dim_sizes=new_dim_sizes,
            fill_value=self.fill_value,
        )

    def set_fill_value(self, fill_value: Any) -> "TensorDef":
        """
        Return
            :TensorDef with  new fill_value
        """
        return TensorDef(
            index_set=self.index_set,
            dim_sizes=self.dim_sizes,
            fill_value=fill_value,
        )

    def relabel_index(self, i: str, j: str) -> "TensorDef":
        """
        If axis `i == j` or axis ` j ` not present, returns self unchanged.
        """
        if i == j or i not in self.index_set:
            return self

        new_index_set = (self.index_set - {i}) | {j}
        new_dim_sizes = dict(self.dim_sizes)
        new_dim_sizes[j] = new_dim_sizes.pop(i)

        return TensorDef(
            index_set=new_index_set,
            dim_sizes=new_dim_sizes,
            fill_value=self.fill_value,
        )

    def add_dummy_idx(self, idx: str) -> "TensorDef":
        """
        Add a new axis `idx` of size 1

        Return:
        TensorDef with new axis `idx` of size 1

        """
        if idx in self.index_set:
            return self

        new_index_set = set(self.index_set)
        new_index_set.add(idx)
        new_dim_sizes = dict(self.dim_sizes)
        new_dim_sizes[idx] = 1.0

        return TensorDef(new_index_set, new_dim_sizes, self.fill_value)

    @property
    def dim_sizes(self) -> Mapping[str, float]:
        return self._dim_sizes

    @dim_sizes.setter
    def dim_sizes(self, value: Mapping[str, float]):
        self._dim_sizes = OrderedDict(value)

    def get_dim_size(self, idx: str) -> float:
        return self.dim_sizes[idx]

    @property
    def index_set(self) -> set[str]:
        return self._index_set

    @index_set.setter
    def index_set(self, value: Iterable[str]):
        self._index_set = set(value)

    @property
    def fill_value(self) -> Any:
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value: Any):
        self._fill_value = value

    def get_dim_space_size(self, idx: Iterable[str]) -> float:
        prod = 1
        for i in idx:
            prod *= int(self.dim_sizes[i])
            if prod == 0 or prod > np.iinfo(np.int64).max:
                return float("inf")
        return float(prod)

    @staticmethod
    def mapjoin(op: Callable, *args: "TensorDef") -> "TensorDef":
        """
        Merge multiple TensorDef objects into a single tensor definition.

        This method takes any number of TensorDef objects and produces a new
        TensorDef whose index set is the union of all input indices. The dimension
        size for each axis is copied from the first input that contains that axis,
        and the fill value is computed by applying the operator `op` across all
        input fill values.

        Returns:
            TensorDef: A new TensorDef representing the merged tensor.
        """
        new_fill_value = op(*(s.fill_value for s in args))
        new_index_set = set().union(*(s.index_set for s in args))
        new_dim_sizes: dict = {}
        for index in new_index_set:
            for s in args:
                if index in s.index_set:
                    new_dim_sizes[index] = s.dim_sizes[index]
                    break
        assert set(new_dim_sizes.keys()) == new_index_set
        return TensorDef(new_index_set, new_dim_sizes, new_fill_value)

    @staticmethod
    def aggregate(
        op: Callable,
        init: Any | None,
        reduce_indices: Iterable[str],
        d: "TensorDef",
    ) -> "TensorDef":
        """
        Reduce a TensorDef along one or more axes to produce a new TensorDef.

        This constructs a new TensorDef by removing the axes in `reduce_indices`
        and computing a new fill value that reflects reducing the original
        fill over the size of the reduced subspace.

        Parameters:
        op : Callable
            The reduction operator.
        init : Any | None
            Explicit initial value for the reduction
        reduce_indices : Iterable[str]
            Axis names to reduce/eliminate from the definition.
        d : TensorDef
            The input tensor definition.

        Returns:
        A new TensorDef with `reduce_indices` removed and the combined
        fill value for the reduced tensor.
        """
        red_set = set(reduce_indices) & set(d.index_set)
        n = math.prod(int(d.dim_sizes[x]) for x in red_set)

        if init is None:
            if is_identity(op, d.fill_value) or is_idempotent(op):
                init = op(d.fill_value, d.fill_value)
            elif op is operator.add:
                init = d.fill_value * n
            elif op is operator.mul:
                init = d.fill_value**n
            else:
                # This is going to be VERY SLOW. Should raise a warning about reductions
                # over non-identity fill values. Depending on the
                # semantics of reductions, we might be able to do this faster.
                print(
                    "Warning: A reduction can take place over a tensor whose fill"
                    "value is not the reduction operator's identity. This can result in"
                    "a large slowdown as the new fill is calculated."
                )
                acc = d.fill_value
                for _ in range(max(n - 1, 0)):
                    acc = op(acc, d.fill_value)
                init = acc

        new_dim_sizes = OrderedDict(
            (ax, d.dim_sizes[ax]) for ax in d.dim_sizes if ax not in red_set
        )
        new_index_set = set(new_dim_sizes)
        return TensorDef(new_index_set, new_dim_sizes, init)
