from collections import OrderedDict
from typing import Any, Iterable, Mapping, Set, Callable, Type
from abc import ABC, abstractmethod

from finch.algebra import fill_value

class TensorDef:
    def __init__(
        self,
        index_set: Iterable[str],
        dim_sizes: Mapping[str, float],
        fill_val: Any,
    ):
        self.index_set = set(index_set)
        self.dim_sizes = OrderedDict(dim_sizes)
        self.fill_val = fill_val

    def copy(self) -> "TensorDef":
        """
        Return:
            Deep copy of TensorDef fields
        """
        return TensorDef(
            index_set = self.index_set.copy(),
            dim_sizes = self.dim_sizes.copy(),
            fill_val = self.fill_val,
        )

    @classmethod
    def from_tensor(klass, tensor: Any, indices: Iterable[str]) -> "TensorDef":
        """
        Storing axis, sizes, and fill_val of the tensor

        """
        shape = tensor.shape
        dim_sizes = OrderedDict((axis, float(shape[i])) for i, axis in enumerate(indices))
        fv = fill_value(tensor)
        return klass(
            index_set = indices,
            dim_sizes = dim_sizes,
            fill_val = fv,
        )

    def reindex_def(self, new_axis: Iterable[str]) -> "TensorDef":
        """
        Return
            :TensorDef with a new reindexed index_set and dim sizes
        """
        new_axis = list(new_axis)
        new_dim_sizes = OrderedDict((axis, self.dim_sizes[axis]) for axis in new_axis)
        return TensorDef(
            index_set = new_axis,
            dim_sizes = new_dim_sizes,
            fill_val = self.fill_val,
    )


    def set_fill_value(self, fill_val: Any) -> "TensorDef":
        """
        Return
            :TensorDef with  new fill_val
        """
        return TensorDef(
            index_set = self.index_set,
            dim_sizes = self.dim_sizes,
            fill_val  = fill_val,
        )

    def relabel_index(self, i: str, j: str) -> "TensorDef":
        """
        If axis `i == j` or axis ` j ` not present, returns self unchanged.
        """
        if i == j or i not in self.index_set:
            return self

        new_index_set = (self.index_set - {i}) | {j}
        new_dim_sizes = dict(self.dim_sizes)
        new_dim_sizes[i] = new_dim_sizes.pop(j)

        return TensorDef(
            index_set = new_index_set,
            dim_sizes = new_dim_sizes,
            fill_val  = self.fill_val,
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

        return TensorDef(new_index_set, new_dim_sizes, self.fill_val)

    def get_dim_sizes(self) -> Mapping[str, float]: return self.dim_sizes
    def get_dim_size(self, idx: str) -> float: return self.dim_sizes[idx]
    def get_index_set(self) -> Set[str]: return self.index_set
    def get_fill_value(self) -> Any: return self.fill_val
