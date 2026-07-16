from __future__ import annotations

import math
from typing import Any

import numpy as np

from finchlite.algebra.algebra import FinchOperator
from finchlite.finch_logic import Field

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStatsFactory
from .util import degree_count_scan


class DatabaseStatsFactory(BaseTensorStatsFactory["DatabaseStats"]):
    def __init__(self):
        super().__init__(DatabaseStats)

    def _mapjoin_union(
        self,
        op: FinchOperator,
        *union_args: DatabaseStats,
    ) -> DatabaseStats:
        base_stats = super()._mapjoin_defs(op, *union_args)

        if len(union_args) == 1:
            return DatabaseStats.from_base_stats(
                base_stats, nnz=union_args[0].nnz, V=dict(union_args[0].V)
            )

        cur_nnz = union_args[0].nnz
        cur_V: dict[Field, float] = dict(union_args[0].V)
        cur_indices: set[Field] = set(union_args[0].index_order)

        for i in union_args[1:]:
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
                    n = base_stats.dim_sizes[idx]
                    new_V[idx] = min(n, cur_V[idx] + i.V[idx])

            # Broadcast case: A_ij + B_jk
            else:
                cur_dim = math.prod(base_stats.dim_sizes[k] for k in only_cur)
                new_dim = math.prod(base_stats.dim_sizes[k] for k in only_i)

                # nnz(C) = nnz(A) * n_k + n_i * nnz(B)
                new_nnz = cur_nnz * new_dim + i.nnz * cur_dim

                for idx in set(i.index_order).union(cur_indices):
                    n = base_stats.dim_sizes[idx]
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

        return DatabaseStats.from_base_stats(base_stats, nnz=cur_nnz, V=new_V)

    def _mapjoin_join(
        self,
        op: FinchOperator,
        *join_args: DatabaseStats,
    ) -> DatabaseStats:
        base_stats = super()._mapjoin_defs(op, *join_args)

        if len(join_args) == 1:
            return DatabaseStats.from_base_stats(
                base_stats, nnz=join_args[0].nnz, V=dict(join_args[0].V)
            )

        cur_nnz = join_args[0].nnz
        cur_V: dict[Field, float] = dict(join_args[0].V)
        cur_indices: set[Field] = set(join_args[0].index_order)

        # Join case: A_ij * B_jk
        for i in join_args[1:]:
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
                n = base_stats.dim_sizes[idx]
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

        return DatabaseStats.from_base_stats(base_stats, nnz=new_nnz, V=new_V)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: DatabaseStats,
    ) -> DatabaseStats:
        if not isinstance(stats, DatabaseStats):
            raise TypeError("DatabaseStats expected for aggregate")

        base_stats = self.aggregate_def(op, init, reduce_indices, stats)
        new_nnz = min(
            stats.nnz,
            math.prod(
                stats.V.get(j, 1.0)
                for j in stats.index_order
                if j not in reduce_indices
            ),
        )

        new_V: dict[Field, float] = {}
        for idx in base_stats.index_order:
            # V(C,i) = V(A,i)
            new_V[idx] = stats.V[idx]

        return DatabaseStats.from_base_stats(base_stats, nnz=new_nnz, V=new_V)

    def relabel(
        self, stats: DatabaseStats, relabel_indices: tuple[Field, ...]
    ) -> DatabaseStats:
        if not isinstance(stats, DatabaseStats):
            raise TypeError("DatabaseStats expected for relabel")
        base_stats = self.relabel_def(stats, relabel_indices)
        V = {}
        for old, new in zip(stats.index_order, relabel_indices, strict=True):
            V[new] = stats.V[old]
        return DatabaseStats.from_base_stats(base_stats, nnz=stats.nnz, V=V)

    def reorder(
        self, stats: DatabaseStats, reorder_indices: tuple[Field, ...]
    ) -> DatabaseStats:
        if not isinstance(stats, DatabaseStats):
            raise TypeError("DatabaseStats expected for reorder")
        base_stats = self.reorder_def(stats, reorder_indices)
        return DatabaseStats.from_base_stats(
            base_stats, nnz=stats.nnz, V=stats.V.copy()
        )


class DatabaseStats(NumericStats):
    def __init__(self, tensor: Any, fields: tuple[Field, ...]):
        super().__init__(tensor, fields)
        fields = tuple(fields)

        # Scalar tensor: a single non-fill value and no per-dimension domains.
        if tensor.ndim == 0:
            self.nnz = 1.0
            self.V: dict[Field, float] = {}
            return

        # ``counts[i][v]`` is the number of non-fill entries with index ``v`` in
        # dimension ``i``; ``nnz`` is the total. The number of distinct values
        # (domain size) of dimension ``i`` is the count of nonzero degrees.
        counts, nnz = degree_count_scan(tensor, fields, self.fill_value)
        self.nnz = float(nnz)
        self.V = {
            field: float(np.count_nonzero(counts[i])) for i, field in enumerate(fields)
        }

    def estimate_non_fill_values(self) -> float:
        return self.nnz

    def get_embedding(self):
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        v_vals = [float(self.V.get(field, 0.0)) for field in self.index_order]

        size_part = np.log2(np.array(sizes))
        v_vals_part = np.log2(np.array(v_vals))
        nnz_part = np.log2(np.array([self.nnz]))

        return np.concatenate([size_part, v_vals_part, nnz_part])
