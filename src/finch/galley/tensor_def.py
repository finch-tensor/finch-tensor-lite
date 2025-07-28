from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from finch.algebra import fill_value


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
        fv = fill_value(tensor)
        try:
            arr = np.asarray(tensor)
            if arr.size > 0:
                first = arr.flat[0]
                if np.all(arr == first):
                    fv = float(first)
        except (TypeError, ValueError):
            pass

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
