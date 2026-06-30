from __future__ import annotations

import math
from typing import Any

import numpy as np

from finchlite.algebra.algebra import FinchOperator, is_annihilator, is_identity
from finchlite.finch_logic import Field

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory


class UniformStatsFactory(BaseTensorStatsFactory["UniformStats"]):
    def __init__(self):
        super().__init__(UniformStats)

    def _mapjoin_union(
        self,
        new_def: BaseTensorStats,
        op: FinchOperator,
        union_args: list[UniformStats],
    ) -> UniformStats:

        new_vol = new_def.get_dim_space_size(new_def.index_order)

        if new_vol == 0.0:
            return UniformStats.from_def(new_def, nnz=0.0)

        inv_p = 1.0

        for s in union_args:
            vol = s.get_dim_space_size(s.index_order)
            if isinstance(s, NumericStats):
                p = s.estimate_non_fill_values() / vol if vol > 0 else 0.0
                inv_p *= 1 - p
            else:
                raise TypeError("Stats Class must be inherit from NumericStats")

        res_p = 1 - inv_p

        return UniformStats.from_def(new_def, nnz=res_p * new_vol)

    def _mapjoin_join(
        self, new_def: BaseTensorStats, op: FinchOperator, join_args: list[UniformStats]
    ) -> UniformStats:

        new_vol = new_def.get_dim_space_size(new_def.index_order)

        if new_vol == 0.0:
            return UniformStats.from_def(new_def, nnz=0.0)

        res_p = 1.0

        for s in join_args:
            vol = s.get_dim_space_size(s.index_order)
            if isinstance(s, NumericStats):
                p = s.estimate_non_fill_values() / vol if vol > 0 else 0.0
                res_p *= p
            else:
                raise TypeError("Stats Class must be inherit from NumericStats")

        return UniformStats.from_def(new_def, nnz=res_p * new_vol)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: UniformStats,
    ) -> UniformStats:
        new_def = super().aggregate(op, init, reduce_indices, stats)
        res_vol = new_def.get_dim_space_size(new_def.index_order)
        red_set = set(reduce_indices) & set(stats.index_order)
        k = math.prod(int(stats.dim_sizes[x]) for x in red_set)
        old_vol = stats.get_dim_space_size(stats.index_order)
        if isinstance(stats, NumericStats):
            p_old = stats.estimate_non_fill_values() / old_vol if old_vol > 0 else 0.0
        else:
            raise TypeError("Stats Class must be inherit from NumericStats")
        if is_annihilator(op, stats.fill_value):
            res_p = math.pow(p_old, k)
        elif is_identity(op, stats.fill_value):
            res_p = 1 - math.pow((1 - p_old), k)
        else:
            res_p = 1.0

        return UniformStats.from_def(new_def, nnz=res_p * res_vol)

    def relabel(
        self, stats: UniformStats, relabel_indices: tuple[Field, ...]
    ) -> UniformStats:
        new_def = super().relabel(stats, relabel_indices)
        if isinstance(stats, NumericStats):
            return UniformStats.from_def(new_def, nnz=stats.estimate_non_fill_values())
        raise TypeError("Stats Class must be inherit from NumericStats")

    def reorder(
        self, stats: UniformStats, reorder_indices: tuple[Field, ...]
    ) -> UniformStats:
        new_def = super().reorder(stats, reorder_indices)
        if isinstance(stats, NumericStats):
            return UniformStats.from_def(new_def, nnz=stats.estimate_non_fill_values())
        raise TypeError("Stats Class must be inherit from NumericStats")


class UniformStats(NumericStats):
    nnz: float

    def __init__(self, tensor: Any, fields: tuple[Field, ...]):
        super().__init__(tensor, fields)
        val = tensor

        if hasattr(val, "tns"):
            val = val.tns.val
        if hasattr(val, "val") and not hasattr(val, "to_numpy"):
            val = val.val

        if hasattr(val, "to_numpy"):
            arr = val.to_numpy()
            self.nnz = float(np.sum(arr != self.fill_value))
        else:
            self.nnz = float(self.get_dim_space_size(self.index_order))

    def estimate_non_fill_values(self) -> float:
        return self.nnz

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        volume = self.get_dim_space_size(self.index_order)

        prob = self.nnz / volume if volume > 0 else 0.0

        uniform_embedding = sizes + [prob]

        return np.log2(uniform_embedding)
