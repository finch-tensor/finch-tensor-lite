from __future__ import annotations

import math
from typing import Any

import numpy as np

from finchlite.algebra.algebra import FinchOperator
from finchlite.finch_logic import Field

from .dc_stats import DCStats
from .numeric_stats import NumericStats
from .tensor_def import TensorDef
from .tensor_stats import BaseTensorStatsFactory


class DatabaseStatsFactory(BaseTensorStatsFactory["DatabaseStats"]):
    def __init__(self):
        super().__init__(DatabaseStats)

    def copy_stats(self, stat: DatabaseStats) -> DatabaseStats:
        if not isinstance(stat, DatabaseStats):
            raise TypeError("copy_stats expected a DatabaseStats instance")
        return DatabaseStats.from_def(stat.tensordef.copy(), stat.nnz, stat.V.copy())

    def _merge_join(
        self, new_def: TensorDef, all_stats: list[DatabaseStats]
    ) -> DatabaseStats:

        if len(all_stats) == 1:
            return DatabaseStats.from_def(
                new_def, all_stats[0].nnz, dict(all_stats[0].V)
            )

        cur_nnz = all_stats[0].nnz
        cur_V: dict[Field, float] = dict(all_stats[0].V)
        cur_indices: set[Field] = set(all_stats[0].index_order)

        # Join case: A_ij * B_jk
        for i in all_stats[1:]:
            shared = cur_indices & set(i.index_order)
            only_cur = cur_indices - set(i.index_order)
            only_i = set(i.index_order) - cur_indices

            cur_shared = max((cur_V.get(j, 1.0) for j in shared), default=1.0)
            i_shared = max((i.V.get(j, 1.0) for j in shared), default=1.0)
            # nnz(C) = nnz(A) * nnz(B) / (max(nnz(\sum_k B_jk), nnz(\sum_i A_ij))
            if max(cur_shared, i_shared) == 0.0:
                new_nnz = 0.0
            else:
                new_nnz = cur_nnz * i.nnz / max(cur_shared, i_shared)

            new_V: dict[Field, float] = {}
            for idx in set(i.index_order).union(cur_indices):
                n = new_def.dim_sizes[idx]
                if idx in only_cur:
                    # V(C, i) = min(n_i, V(A,i), nnz(C))
                    new_V[idx] = min(n, cur_V[idx], new_nnz)
                elif idx in shared:
                    # V(C, j) = min(n_j, V(A,j), V(B,j))
                    new_V[idx] = min(n, cur_V[idx], i.V[idx])
                elif idx in only_i:
                    # V(C, k) = min(n_k, V(B,k), nnz(C))
                    new_V[idx] = min(n, i.V[idx], new_nnz)

            cur_nnz = new_nnz
            cur_V = new_V
            cur_indices = set(i.index_order).union(cur_indices)

        return DatabaseStats.from_def(new_def, new_nnz, new_V)

    def _merge_union(
        self, new_def: TensorDef, all_stats: list[DatabaseStats]
    ) -> DatabaseStats:

        if len(all_stats) == 1:
            return DatabaseStats.from_def(
                new_def, all_stats[0].nnz, dict(all_stats[0].V)
            )

        cur_nnz = all_stats[0].nnz
        cur_V: dict[Field, float] = dict(all_stats[0].V)
        cur_indices: set[Field] = set(all_stats[0].index_order)

        for i in all_stats[1:]:
            shared = cur_indices & set(i.index_order)
            only_cur = cur_indices - set(i.index_order)
            only_i = set(i.index_order) - cur_indices
            new_V: dict[Field, float] = {}
            # Elementwise case: A_ij + B_ij
            if shared and not only_cur and not only_i:
                # nnz(C) = nnz(A) + nnz(B)
                new_nnz = cur_nnz + i.nnz
                for idx in set(i.index_order).union(cur_indices):
                    # V(C, i) = min(n_i, V(A, i) + V(B,i))
                    # V(C, j) = min(n_j, V(A, j) + V(B,j))
                    n = new_def.dim_sizes[idx]
                    new_V[idx] = min(n, cur_V[idx] + i.V[idx])

            # Broadcast case: A_ij + B_jk
            else:
                cur_dim = math.prod(new_def.dim_sizes[k] for k in only_cur)
                new_dim = math.prod(new_def.dim_sizes[k] for k in only_i)

                # nnz(C) = nnz(A) * n_k + n_i * nnz(B)
                new_nnz = cur_nnz * new_dim + i.nnz * cur_dim

                for idx in set(i.index_order).union(cur_indices):
                    n = new_def.dim_sizes[idx]
                    if idx in only_cur:
                        # V(C, i) = n
                        new_V[idx] = n
                    elif idx in shared:
                        # V(C, j) = min(n_j, V(A, j) + V(B,j))
                        new_V[idx] = min(n, cur_V[idx] + i.V[idx])
                    else:
                        # V(C, k) = n
                        new_V[idx] = n

            cur_nnz = new_nnz
            cur_V = new_V
            cur_indices = set(i.index_order).union(cur_indices)

        return DatabaseStats.from_def(new_def, cur_nnz, new_V)

    def _mapjoin_union(
        self, new_def: TensorDef, op: FinchOperator, union_args: list[DatabaseStats]
    ) -> DatabaseStats:

        return self._merge_union(new_def, union_args)

    def _mapjoin_join(
        self, new_def: TensorDef, op: FinchOperator, join_args: list[DatabaseStats]
    ) -> DatabaseStats:

        if not join_args:
            return DatabaseStats.from_def(new_def, 0.0, {})
        join_cover = set().union(*(s.tensordef.index_order for s in join_args))
        if join_cover == set(new_def.index_order):
            return self._merge_join(new_def, join_args)
        return self._merge_union(new_def, join_args)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: DatabaseStats,
    ) -> DatabaseStats:
        if not isinstance(stats, DatabaseStats):
            raise TypeError("DatabaseStats expected for aggregate")

        new_def = TensorDef.aggregate(op, init, reduce_indices, stats.tensordef)
        new_nnz = min(
            stats.nnz,
            math.prod(
                stats.V.get(j, 1.0)
                for j in stats.index_order
                if j not in reduce_indices
            ),
        )

        new_V: dict[Field, float] = {}
        for idx in new_def.index_order:
            # V(C,i) = V(A,i)
            new_V[idx] = stats.V[idx]

        return DatabaseStats.from_def(new_def, new_nnz, new_V)

    def relabel(
        self, stats: DatabaseStats, relabel_indices: tuple[Field, ...]
    ) -> DatabaseStats:
        if not isinstance(stats, DatabaseStats):
            raise TypeError("DatabaseStats expected for relabel")
        new_def = TensorDef.relabel(stats.tensordef, relabel_indices)
        V = {}
        for old, new in zip(stats.index_order, relabel_indices, strict=True):
            V[new] = stats.V[old]
        return DatabaseStats.from_def(new_def, stats.nnz, V)

    def reorder(
        self, stats: DatabaseStats, reorder_indices: tuple[Field, ...]
    ) -> DatabaseStats:
        if not isinstance(stats, DatabaseStats):
            raise TypeError("DatabaseStats expected for reorder")
        new_def = TensorDef.reorder(stats.tensordef, reorder_indices)
        return DatabaseStats.from_def(new_def, stats.nnz, stats.V.copy())


class DatabaseStats(NumericStats):
    def __init__(self, tensor: Any, fields: tuple[Field, ...]):
        self.tensordef = TensorDef.from_tensor(tensor, fields)
        val = tensor

        if hasattr(val, "tns"):
            val = val.tns.val
        if hasattr(val, "val") and not hasattr(val, "to_numpy"):
            val = val.val

        dc = DCStats(tensor, fields)

        self.nnz = 0.0
        for d in dc.dcs:
            if d.from_indices == frozenset() and d.to_indices == frozenset(fields):
                self.nnz = d.value
                break

        self.V: dict[Field, float] = dict.fromkeys(fields, 0.0)
        for idx in fields:
            for d in dc.dcs:
                if d.from_indices == frozenset() and d.to_indices == frozenset({idx}):
                    self.V[idx] = d.value

    @classmethod
    def from_def(
        cls, tensordef: TensorDef, nnz: float, V: dict[Field, float]
    ) -> DatabaseStats:
        obj = object.__new__(cls)
        obj.tensordef = tensordef
        obj.nnz = nnz
        obj.V = V
        return obj

    def estimate_non_fill_values(self) -> float:
        return self.nnz

    def get_embedding(self):
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        v_vals = [float(self.V.get(field, 0.0)) for field in self.index_order]

        size_part = np.log2(np.array(sizes))
        v_vals_part = np.log2(np.array(v_vals))
        nnz_part = np.log2(np.array([self.nnz]))

        return np.concatenate([size_part, v_vals_part, nnz_part])
