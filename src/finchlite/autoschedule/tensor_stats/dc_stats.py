from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from finchlite.algebra import Tensor
from finchlite.algebra.algebra import FinchOperator
from finchlite.finch_logic import Field

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory
from .util import degree_count_scan


@dataclass(frozen=True)
class DC:
    """
    A degree constraint (DC) record representing structural cardinality

    Attributes:
        from_indices: Conditioning index names.
        to_indices: Index names whose distinct combinations are counted
            when `from_indices` are fixed.
        value: Estimated number of distinct combinations for `to_indices`
            given the fixed `from_indices`.
    """

    from_indices: frozenset[Field]
    to_indices: frozenset[Field]
    value: float


class DCStatsFactory(BaseTensorStatsFactory["DCStats"]):
    def __init__(self):
        super().__init__(DCStats)

    def _mapjoin_union(self, op: FinchOperator, *union_args: DCStats) -> DCStats:
        base_stats = super()._mapjoin_defs(op, *union_args)

        if len(union_args) == 1:
            return DCStats.from_base_stats(union_args[0], dcs=set(union_args[0].dcs))

        dc_keys: Counter[tuple[frozenset[Field], frozenset[Field]]] = Counter()
        stats_dcs: list[dict[tuple[frozenset[Field], frozenset[Field]], float]] = []
        for stats in union_args:
            dcs: dict[tuple[frozenset[Field], frozenset[Field]], float] = {}
            Z = tuple(x for x in base_stats.index_order if x not in stats.index_order)
            Z_dim_size = base_stats.get_dim_space_size(Z)
            for dc in stats.dcs:
                new_key = (dc.from_indices, dc.to_indices)
                dcs[new_key] = dc.value
                dc_keys[new_key] += 1

                ext_dc_key = (dc.from_indices, dc.to_indices | frozenset(Z))
                if ext_dc_key not in dcs:
                    dc_keys[ext_dc_key] += 1
                prev = dcs.get(ext_dc_key, math.inf)
                dcs[ext_dc_key] = min(prev, dc.value * Z_dim_size)
            stats_dcs.append(dcs)

        new_dcs: dict[tuple[frozenset[Field], frozenset[Field]], float] = {}
        for key, count in dc_keys.items():
            if count == len(union_args):
                total = sum(d.get(key, 0.0) for d in stats_dcs)
                X, Y = key
                if Y.issubset(base_stats.index_order):
                    total = min(total, base_stats.get_dim_space_size(Y))
                new_dcs[key] = min(float(2**64), total)

        new_stats = {DC(X, Y, d) for (X, Y), d in new_dcs.items()}
        return DCStats.from_base_stats(base_stats, dcs=new_stats)

    def _mapjoin_join(self, op: FinchOperator, *join_args: DCStats) -> DCStats:
        base_stats = super()._mapjoin_defs(op, *join_args)

        if len(join_args) == 1:
            return DCStats.from_base_stats(base_stats, dcs=set(join_args[0].dcs))

        new_dc: dict[tuple[frozenset[Field], frozenset[Field]], float] = {}
        for stats in join_args:
            for dc in stats.dcs:
                dc_key = (dc.from_indices, dc.to_indices)
                current_dc = new_dc.get(dc_key, math.inf)
                if dc.value < current_dc:
                    new_dc[dc_key] = dc.value

        new_stats = {DC(X, Y, d) for (X, Y), d in new_dc.items()}
        return DCStats.from_base_stats(base_stats, dcs=new_stats)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: DCStats,
    ) -> DCStats:
        fields = reduce_indices
        base_stats: BaseTensorStats
        if len(fields) == 0:
            base_stats = stats.copy()
        else:
            base_stats = self.aggregate_def(op, init, fields, stats)

        dcs = set(stats.dcs) if isinstance(stats, DCStats) else set()
        return DCStats.from_base_stats(base_stats, dcs=dcs)

    def relabel(self, stats: DCStats, relabel_indices: tuple[Field, ...]) -> DCStats:
        base_stats = self.relabel_def(stats, relabel_indices)
        if isinstance(stats, DCStats):
            mapping = dict(zip(stats.index_order, relabel_indices, strict=True))
            dcs = {
                DC(
                    frozenset(mapping.get(f, f) for f in dc.from_indices),
                    frozenset(mapping.get(f, f) for f in dc.to_indices),
                    dc.value,
                )
                for dc in stats.dcs
            }
        else:
            dcs = set()
        return DCStats.from_base_stats(base_stats, dcs=dcs)

    def reorder(self, stats: DCStats, reorder_indices: tuple[Field, ...]) -> DCStats:
        base_stats = self.reorder_def(stats, reorder_indices)
        dcs: set[DC] = set(stats.dcs) if isinstance(stats, DCStats) else set()
        return DCStats.from_base_stats(base_stats, dcs=dcs)


class DCStats(NumericStats):
    """
    Structural statistics derived from a tensor using degree constraint (DCs).

    DCStats scans a tensor and computes degree constraint (DC) records that
    summarize how index sets relate. These DCs can be used to estimate the
    number of non-fill values without materializing sparse coordinates.
    """

    def __init__(self, tensor: Any, fields: tuple[Field, ...]):
        """
        Initialize DCStats from a tensor and its axis names, build the BaseTensorStats,
        and compute degree constraint (DC) records from the tensor’s structure.
        """
        super().__init__(tensor, fields)
        self.dcs = self._structure_to_dcs(tensor, fields)

    def _structure_to_dcs(self, arr: Tensor, fields: Iterable[Field]) -> set[DC]:
        """
        Dispatch DC extraction based on tensor dimensionality.

        Returns:
            set[DC]: One of the following, depending on `self.tensor.ndim`:
                • Empty set, if the tensor is empty (`self.tensor.size == 0`)
                • 1D → _vector_structure_to_dcs()
                • 2D → _matrix_structure_to_dcs()
                • 3D → _3d_structure_to_dcs()
                • 4D → _4d_structure_to_dcs()

        Raises:
            NotImplementedError: If dimensionality is not in {1, 2, 3, 4}.
        """
        ndim = arr.ndim

        if ndim == 0:
            return {DC(frozenset(), frozenset(), 1.0)}

        return self._array_to_dcs(arr, fields)

    # Given an arbitrary n-dimensional tensor, we produce 2n+1 degree constraints.
    # For each field i, we compute DC({}, {i}) and DC({i}, {*fields}).
    # Additionally, we compute the nnz for the full tensor DC({}, {*fields}).
    def _array_to_dcs(self, arr: Any, fields: Iterable[Field]) -> set[DC]:
        fields = list(fields)
        counts, nnz = degree_count_scan(arr, fields, self.fill_value)

        dcs = set()
        for i in range(len(fields)):
            proj = float(np.count_nonzero(counts[i]))
            max_deg = float(counts[i].max()) if counts[i].size else 0.0
            dcs.add(DC(frozenset({}), frozenset({fields[i]}), proj))
            dcs.add(DC(frozenset({fields[i]}), frozenset({*fields}), max_deg))
        dcs.add(DC(frozenset({}), frozenset({*fields}), float(nnz)))
        return dcs

    def estimate_non_fill_values(self) -> float:
        """
        Estimate the number of non-fill values using DCs.

        This uses the stored degree constraint (DC) as multiplicative factors to
        grow coverage over the target indices and finds the smallest product that
        covers all target indices. The result is clamped by the tensor’s dense
        capacity (the product of the target dimensions).

        Returns:
            the estimated number of non-fill entries in the tensor.
        """
        idx: frozenset[Field] = frozenset(self.dim_sizes.keys())
        if len(idx) == 0:
            return 1.0

        best: dict[frozenset[Field], float] = {frozenset(): 1.0}
        frontier: set[frozenset[Field]] = {frozenset()}

        while True:
            current_bound = best.get(idx, math.inf)
            new_frontier: set[frozenset[Field]] = set()

            for node in frontier:
                for dc in self.dcs:
                    if node.issuperset(dc.from_indices):
                        y = node.union(dc.to_indices)
                        if best[node] > float(2 ** (64 - 2)) or float(dc.value) > float(
                            2 ** (64 - 2)
                        ):
                            y_weight = float(2**64)
                        else:
                            y_weight = best[node] * dc.value
                        if min(current_bound, best.get(y, math.inf)) > y_weight:
                            best[y] = y_weight
                            new_frontier.add(y)
            if len(new_frontier) == 0:
                break
            frontier = new_frontier

        min_weight = float(self.get_dim_space_size(idx))
        for node, weight in best.items():
            if node.issuperset(idx):
                min_weight = min(min_weight, weight)
        return min_weight

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        dcs = self.dcs
        dc_embedding = [
            dc.value
            for dc in sorted(
                dcs,
                key=lambda dc: (
                    tuple(sorted(str(f) for f in dc.from_indices)),
                    tuple(sorted(str(f) for f in dc.to_indices)),
                ),
            )
        ]
        return np.log2(np.array([*sizes, *dc_embedding]))
