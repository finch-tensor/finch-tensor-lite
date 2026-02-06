import math
import operator
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from finchlite.finch_logic import (
    MapJoin,
    Table,
    Literal,
    Alias,
    Field,
)
from ...algebra import fill_value, is_idempotent, is_identity


class TensorDef:
    def __init__(
        self,
        index_order : tuple[Field, ...],
        dim_sizes: Mapping[Field, float],
        fill_value: Any,
    ):
        self._index_order = tuple(index_order)
        self._dim_sizes = OrderedDict(dim_sizes)
        self._fill_value = fill_value

    def copy(self) -> "TensorDef":
        """
        Return:
            Deep copy of TensorDef fields
        """
        return TensorDef(
            index_order=self._index_order,
            dim_sizes=self._dim_sizes.copy(),
            fill_value=self._fill_value,
        ) 

    @classmethod
    # indices ->()
    def from_tensor(cls, tensor: Any, indices: tuple[Field, ...]) -> "TensorDef":
        """
        Storing axis, sizes, and fill_value of the tensor

        """
        shape = tensor.shape
        dim_sizes = OrderedDict(
            (axis, float(shape[i])) for i, axis in enumerate(indices)
        )
        fv = fill_value(tensor)

        return cls(
            index_order=indices,
            dim_sizes=dim_sizes,
            fill_value=fv,
        )

    def reindex_def(self, new_axis:tuple[Field, ...]) -> "TensorDef":
        """
        Return
            :TensorDef with a new reindexed index_order and dim sizes
        """

        new_dim_sizes = OrderedDict((axis, self.dim_sizes[axis]) for axis in new_axis)
        return TensorDef(
            index_order=new_axis,
            dim_sizes=new_dim_sizes,
            fill_value=self.fill_value,
        )

    def set_fill_value(self, fill_value: Any) -> "TensorDef":
        """
        Return
            :TensorDef with  new fill_value
        """
        return TensorDef(
            index_order=self.index_order,
            dim_sizes=self.dim_sizes,
            fill_value=fill_value,
        )


    def add_dummy_idx(self, idx: Field) -> "TensorDef":
        """
        Add a new axis `idx` of size 1

        Return:
        TensorDef with new axis `idx` of size 1

        """
        if idx in self.index_order:
            return self
        

        new_index_order = self.index_order + (idx,)
        new_dim_sizes = dict(self.dim_sizes)
        new_dim_sizes[idx] = 1.0

        return TensorDef(new_index_order, new_dim_sizes, self.fill_value)

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
    def fill_value(self) -> Any:
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value: Any):
        self._fill_value = value

    def get_dim_space_size(self, idx:Iterable[Field]) -> float:
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
        new_index_order = MapJoin(Literal(op), tuple(
            Table(Alias(f"_{i}"), tuple((a.index_order))) for i, a in enumerate(args)
        )).fields()
        new_dim_sizes: dict = {}
        for index in new_index_order:
            for s in args:
                if index in s.index_order:
                    new_dim_sizes[index] = s.dim_sizes[index]
                    break
        assert set(new_dim_sizes.keys()) == set(new_index_order)
        return TensorDef(new_index_order, new_dim_sizes, new_fill_value)

    @staticmethod
    def aggregate(
        op: Callable,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
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
        reduce_indices : tuple[str,...]
            Axis names to reduce/eliminate from the definition.
        d : TensorDef
            The input tensor definition.

        Returns:
        A new TensorDef with `reduce_indices` removed and the combined
        fill value for the reduced tensor.
        """
        red_set = set(reduce_indices) & set(d.index_order)
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
        new_index_order = tuple(new_dim_sizes)
        return TensorDef(new_index_order, new_dim_sizes, init)
    
    @staticmethod
    def relabel(d: "TensorDef",
                 relabel_indices: tuple[Field, ...]) -> "TensorDef":
        """
        Relabel the axes in the given TensorDef to new labels

        This constructs a new TensorDef by associating the new labels for axes with the dimension size of the old names.

        Parameters:
        relabel_indices : tuple[str,...]
            Axis names to relabel the names of the old axes.
        d : TensorDef
            The input tensor definition.

        Returns:
        A new TensorDef with relabled indices and the same fill value as the tensor remains unaffected
        """
        if len(relabel_indices)!=len(d.index_order):
            raise ValueError(
                f"Tensor has {len(d.index_order)} dims, "
                f"but {len(relabel_indices)} names provided."
            )
        
        new_dim_sizes = OrderedDict(zip(relabel_indices,d.dim_sizes.values()))

        return TensorDef(relabel_indices,new_dim_sizes,d.fill_value)
    
    @staticmethod
    def reorder(stats: "TensorDef", reorder_indices: tuple[Field, ...]) -> "TensorDef":
        for old_idx in stats.index_order:
            if old_idx not in set(reorder_indices) and stats.get_dim_size(old_idx) != 1:
                raise ValueError(
                    f"Trying to drop dimension '{old_idx}' of size"
                    f" {stats.get_dim_size(old_idx)}."
                    " Only size 1 dimensions can be dropped."
                )

        new_dims = OrderedDict()
        for idx in reorder_indices:
            if idx in stats.index_order:
                new_dims[idx] = stats.get_dim_size(idx)
            else:
                new_dims[idx] = 1

        return TensorDef(reorder_indices, new_dims, stats.fill_value)
        

