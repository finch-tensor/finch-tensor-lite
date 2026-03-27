from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ...algebra import is_annihilator
from ...finch_logic import Field
from .dc_stats import DCStats
from .tensor_def import TensorDef
from .tensor_stats import TensorStats


class DatabaseStats(TensorStats):
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

    @staticmethod
    def mapjoin(op: Callable[..., Any], *all_stats: TensorStats) -> TensorStats:
        if not all(isinstance(s, DatabaseStats) for s in all_stats):
            raise TypeError("DatabaseStats expected for mapjoin")

        a, b = all_stats[0], all_stats[1]
        new_def = TensorDef.mapjoin(op, *(s.tensordef for s in all_stats))

        shared = set(a.index_order) & set(b.index_order)
        only_a = set(a.index_order) - set(b.index_order)
        only_b = set(b.index_order) - set(a.index_order)

        # Join case: A_ij * B_jk
        if shared and only_a and only_b and is_annihilator(op, a.tensordef.fill_value):
            a_shared = math.prod(a.V.get(j, 1.0) for j in shared)
            b_shared = math.prod(b.V.get(j, 1.0) for j in shared)
            # nnz(C) = nnz(A) * nnz(B) / (max(nnz(\sum_k B_jk), nnz(\sum_i A_ij))
            new_nnz = a.nnz * b.nnz / max(a_shared, b_shared)

            new_V: dict[Field, float] = {}
            for idx in new_def.index_order:
                n = new_def.dim_sizes[idx]
                if idx in only_a:
                    # V(C, i) = min(n_i, V(A,i), nnz(C))
                    new_V[idx] = min(n, a.V[idx], new_nnz)
                elif idx in shared:
                    # V(C, j) = min(n_j, V(A,j), V(B,j))
                    new_V[idx] = min(n, a.V[idx], b.V[idx])
                elif idx in only_b:
                    # V(C, k) = min(n_k, V(B,k), nnz(C))
                    new_V[idx] = min(n, b.V[idx], new_nnz)

        # Elementwise case: A_ij + B_ij
        elif shared and not only_a and not only_b:
            # nnz(C) = nnz(A) + nnz(B)
            new_nnz = a.nnz + b.nnz

            new_V: dict[Field, float] = {}
            for idx in new_def.index_order:
                # V(C, i) = min(n_i, V(A, i) + V(B,i))
                # V(C, j) = min(n_j, V(A, j) + V(B,j))
                n = new_def.dim_sizes[idx]
                new_V[idx] = min(n, a.V[idx] + b.V[idx])

        # Broadcast case: A_ij + B_jk
        else:
            k_dim = math.prod(new_def.dim_sizes[k] for k in only_b)
            i_dim = math.prod(new_def.dim_sizes[i] for i in only_a)
            # nnz(C) = nnz(A) * n_k + n_i * nnz(B)
            new_nnz = a.nnz * k_dim + b.nnz * i_dim

            new_V: dict[Field, float] = {}
            for idx in new_def.index_order:
                n = new_def.dim_sizes[idx]
                if idx in only_a:
                    # V(C, i) = min(n_i, V(A, i) + n_i)
                    new_V[idx] = min(n, a.V[idx] + n)
                elif idx in shared:
                    # V(C, j) = min(n_j, V(A, j) + V(B,j))
                    new_V[idx] = min(n, a.V[idx] + b.V[idx])
                else:
                    # V(C, k) = min(n_k, n_k + V(B,k))
                    new_V[idx] = min(n, b.V[idx] + n)

        return DatabaseStats.from_def(new_def, new_nnz, new_V)

    @staticmethod
    def aggregate(
        op: Callable[..., Any],
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: TensorStats,
    ) -> TensorStats:
        if not isinstance(stats, DatabaseStats):
            raise TypeError("DatabaseStats expected for aggregate")

        new_def = TensorDef.aggregate(op, init, reduce_indices, stats.tensordef)
        j_shared = math.prod(stats.V.get(j, 1.0) for j in reduce_indices)
        # nnz(C) = nnz(A) / V(A,j)
        new_nnz = stats.nnz / j_shared

        new_V: dict[Field, float] = {}
        for idx in new_def.index_order:
            # V(C,i) = V(A,i)
            new_V[idx] = stats.V[idx]

        return DatabaseStats.from_def(new_def, new_nnz, new_V)

    @staticmethod
    def issimilar(a: TensorStats, b: TensorStats) -> bool:
        if not (isinstance(a, DatabaseStats) and isinstance(b, DatabaseStats)):
            return False
        return (
            a.fill_value == b.fill_value
            and a.dim_sizes == b.dim_sizes
            and math.isclose(a.nnz, b.nnz, rel_tol=1e-9)
        )

    @staticmethod
    def relabel(
        stats: TensorStats, relabel_indices: tuple[Field, ...]
    ) -> DatabaseStats:
        new_def = TensorDef.relabel(stats.tensordef, relabel_indices)
        if isinstance(stats, DatabaseStats):
            V = stats.V.copy()
            nnz = stats.nnz
        else:
            V = {}
            nnz = stats.estimate_non_fill_values()
        return DatabaseStats.from_def(new_def, nnz, V)

    @staticmethod
    def reorder(
        stats: TensorStats, reorder_indices: tuple[Field, ...]
    ) -> DatabaseStats:
        new_def = TensorDef.reorder(stats.tensordef, reorder_indices)
        if isinstance(stats, DatabaseStats):
            V = stats.V.copy()
            nnz = stats.nnz
        else:
            V = {}
            nnz = stats.estimate_non_fill_values()
        return DatabaseStats.from_def(new_def, nnz, V)

    @staticmethod
    def copy_stats(stat: TensorStats) -> DatabaseStats:
        if not isinstance(stat, DatabaseStats):
            raise TypeError("copy_stats expected a DatabaseStats instance")
        return DatabaseStats.from_def(stat.tensordef.copy(), stat.nnz, stat.V.copy())

    def estimate_non_fill_values(self) -> float:
        return self.nnz
