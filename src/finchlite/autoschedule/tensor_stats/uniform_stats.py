from __future__ import annotations

import math
from typing import Any

import numpy as np

from finchlite.algebra.algebra import FinchOperator, is_annihilator, is_identity
from finchlite.finch_logic import Field

from .numeric_stats import NumericStats
from .tensor_def import TensorDef
from .tensor_stats import BaseTensorStatsFactory


class UniformStatsFactory(BaseTensorStatsFactory["UniformStats"]):
    def __init__(self):
        super().__init__(UniformStats)

    def copy_stats(self, stat: UniformStats) -> UniformStats:
        if not isinstance(stat, UniformStats):
            raise TypeError("copy_stats expected a UniformStats instance")
        return UniformStats.from_def(stat.tensordef.copy(), stat.nnz)

    def mapjoin(self, op: FinchOperator, *args: UniformStats) -> UniformStats:
        def_args = [stat.tensordef for stat in args]
        new_def = TensorDef.mapjoin(op, *def_args)
        new_vol = UniformStats._get_volume(new_def)

        if new_vol == 0.0:
            return UniformStats.from_def(new_def, 0.0)

        join_probs: list[float] = []
        union_probs: list[float] = []

        for s in args:
            vol = UniformStats._get_volume(s.tensordef)
            if isinstance(s, NumericStats):
                p = s.estimate_non_fill_values() / vol if vol > 0 else 0.0
            else:
                raise TypeError("Stats Class must be inherit from NumericStats")

            if is_annihilator(op, s.tensordef.fill_value):
                join_probs.append(p)
            else:
                union_probs.append(p)

        res_p = 1.0
        if join_probs:
            for p in join_probs:
                res_p *= p
        elif union_probs:
            inv_p = 1.0
            for p in union_probs:
                inv_p *= 1 - p
            res_p = 1 - inv_p
        else:
            res_p = 1.0

        return UniformStats.from_def(new_def, res_p * new_vol)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: UniformStats,
    ) -> UniformStats:
        new_def = TensorDef.aggregate(op, init, reduce_indices, stats.tensordef)
        res_vol = UniformStats._get_volume(new_def)
        red_set = set(reduce_indices) & set(stats.tensordef.index_order)
        k = math.prod(int(stats.tensordef.dim_sizes[x]) for x in red_set)
        old_vol = UniformStats._get_volume(stats.tensordef)
        if isinstance(stats, NumericStats):
            p_old = stats.estimate_non_fill_values() / old_vol if old_vol > 0 else 0.0
        else:
            raise TypeError("Stats Class must be inherit from NumericStats")
        if is_annihilator(op, stats.tensordef.fill_value):
            res_p = math.pow(p_old, k)
        elif is_identity(op, stats.tensordef.fill_value):
            res_p = 1 - math.pow((1 - p_old), k)
        else:
            res_p = 1.0

        return UniformStats.from_def(new_def, res_p * res_vol)

    def issimilar(self, a: UniformStats, b: UniformStats) -> bool:
        return (
            isinstance(a, UniformStats)
            and isinstance(b, UniformStats)
            and a.dim_sizes == b.dim_sizes
            and a.fill_value == b.fill_value
            and math.isclose(a.nnz, b.nnz, rel_tol=1e-9)
        )

    def relabel(
        self, stats: UniformStats, relabel_indices: tuple[Field, ...]
    ) -> UniformStats:
        d = stats.tensordef
        new_def = TensorDef.relabel(d, relabel_indices)
        if isinstance(stats, NumericStats):
            return UniformStats.from_def(new_def, stats.estimate_non_fill_values())
        raise TypeError("Stats Class must be inherit from NumericStats")

    def reorder(
        self, stats: UniformStats, reorder_indices: tuple[Field, ...]
    ) -> UniformStats:
        d = stats.tensordef
        new_def = TensorDef.reorder(d, reorder_indices)
        if isinstance(stats, NumericStats):
            return UniformStats.from_def(new_def, stats.estimate_non_fill_values())
        raise TypeError("Stats Class must be inherit from NumericStats")


class UniformStats(NumericStats):
    nnz: float

    def __init__(self, tensor: Any, fields: tuple[Field, ...]):
        self.tensordef = TensorDef.from_tensor(tensor, fields)
        val = tensor

        if hasattr(val, "tns"):
            val = val.tns.val
        if hasattr(val, "val") and not hasattr(val, "to_numpy"):
            val = val.val

        if hasattr(val, "to_numpy"):
            arr = val.to_numpy()
            self.nnz = float(np.sum(arr != self.tensordef.fill_value))
        else:
            self.nnz = float(self._get_volume(self.tensordef))

    @classmethod
    def from_def(cls, d: TensorDef, nnz: float | None = None) -> UniformStats:
        us = object.__new__(cls)
        us.tensordef = d.copy()
        if nnz is None:
            us.nnz = float(cls._get_volume(d))
        else:
            us.nnz = float(nnz)
        return us

    @staticmethod
    def _get_volume(d: TensorDef) -> float:
        vol = 1.0
        for size in d.dim_sizes.values():
            vol *= size
        return vol

    def estimate_non_fill_values(self) -> float:
        return self.nnz

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        volume = self._get_volume(self.tensordef)

        prob = self.nnz / volume if volume > 0 else 0.0

        uniform_embedding = sizes + [prob]

        return np.log2(uniform_embedding)
