from __future__ import annotations

from typing import Any

import numpy as np

import finchlite.interface.eager as eager
from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field

from .numeric_stats import NumericStats
from .tensor_def import TensorDef
from .tensor_stats import BaseTensorStatsFactory


class ExactStatsFactory(BaseTensorStatsFactory["ExactStats"]):
    def __init__(self):
        super().__init__(ExactStats)

    def copy_stats(self, stat: ExactStats) -> ExactStats:
        if not isinstance(stat, ExactStats):
            raise TypeError("copy_stats expected a ExactStats instance")
        return ExactStats.from_tensor(stat.tensordef.copy(), stat.tensor)

    def _merge(
        self, new_def: TensorDef, op: FinchOperator, all_stats: list[ExactStats]
    ) -> ExactStats:
        if len(all_stats) == 1:
            return ExactStats.from_tensor(new_def, all_stats[0].tensor)

        target_order = list(new_def.index_order)
        result = all_stats[0].tensor
        cur_order = list(all_stats[0].tensordef.index_order)

        for s in all_stats[1:]:
            cur_exp = result
            cur_cur = list(cur_order)
            for pos, idx in enumerate(target_order):
                if idx not in cur_cur:
                    cur_exp = eager.expand_dims(cur_exp, axis=pos)
                    cur_cur.insert(pos, idx)

            s_exp = s.tensor
            s_cur = list(s.tensordef.index_order)
            for pos, idx in enumerate(target_order):
                if idx not in s_cur:
                    s_exp = eager.expand_dims(s_exp, axis=pos)
                    s_cur.insert(pos, idx)

            a, b = eager.broadcast_arrays(cur_exp, s_exp)
            result = eager.elementwise(op, a, b)
            cur_order = target_order

        return ExactStats.from_tensor(new_def, result)

    def _mapjoin_join(
        self, new_def: TensorDef, op: FinchOperator, join_args: list[ExactStats]
    ) -> ExactStats:
        if len(join_args) == 0:
            new = eager.full(
                tuple(int(new_def.dim_sizes[f]) for f in new_def.index_order), 0.0
            )
            return ExactStats.from_tensor(new_def, new)
        return self._merge(new_def, op, join_args)

    def _mapjoin_union(
        self, new_def: TensorDef, op: FinchOperator, union_args: list[ExactStats]
    ) -> ExactStats:
        return self._merge(new_def, op, union_args)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: ExactStats,
    ) -> ExactStats:
        if not isinstance(stats, ExactStats):
            raise TypeError("ExactStats expected for aggregate")

        new_def = TensorDef.aggregate(op, init, reduce_indices, stats.tensordef)
        cur_order = list(stats.tensordef.index_order)
        reduce_axes = [
            cur_order.index(idx) for idx in reduce_indices if idx in cur_order
        ]
        result = eager.reduce(op, stats.tensor, axis=tuple(reduce_axes), init=init)
        return ExactStats.from_tensor(new_def, result)

    def relabel(
        self, stats: ExactStats, relabel_indices: tuple[Field, ...]
    ) -> ExactStats:
        if not isinstance(stats, ExactStats):
            raise TypeError("ExactStats expected for relabel")
        new_def = TensorDef.relabel(stats.tensordef, relabel_indices)
        return ExactStats.from_tensor(new_def, stats.tensor)

    def reorder(
        self, stats: ExactStats, reorder_indices: tuple[Field, ...]
    ) -> ExactStats:
        if not isinstance(stats, ExactStats):
            raise TypeError("ExactStats expected for reorder")
        new_def = TensorDef.reorder(stats.tensordef, reorder_indices)
        cur_order = list(stats.tensordef.index_order)
        perm = [cur_order.index(idx) for idx in reorder_indices]
        result = eager.permute_dims(stats.tensor, tuple(perm))
        return ExactStats.from_tensor(new_def, result)


class ExactStats(NumericStats):
    def __init__(self, tensor, fields):
        self.tensor = tensor
        self.tensordef = TensorDef.from_tensor(tensor, fields)
        arr = tensor.to_numpy()
        fill = tensor.fill_value
        self.nnz = float(np.count_nonzero(arr != fill))

    @classmethod
    def from_tensor(cls, tensordef: TensorDef, tensor) -> ExactStats:
        obj = object.__new__(cls)
        obj.tensordef = tensordef
        obj.tensor = tensor
        arr = tensor.to_numpy()
        fill = tensor.fill_value
        obj.nnz = float(np.count_nonzero(arr != fill))
        return obj

    def estimate_non_fill_values(self) -> float:
        return self.nnz

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        return np.log2(np.array(sizes + [self.nnz + 1]))
