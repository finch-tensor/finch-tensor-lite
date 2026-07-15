from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from finchlite.algebra import Tensor
from finchlite.algebra.algebra import FinchOperator
from finchlite.finch_logic import Field

from ._degree_scan import degree_count_scan
from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory

DEFAULT_PS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, math.inf)


@dataclass(frozen=True)
class LpDC:
    """
    An ``ell_p``-norm degree constraint used by the LP bound.

    Generalizes a degree constraint (:class:`DC`) from a single scalar count to
    the ``ell_p``-norm of a degree sequence. ``p = 1`` recovers the relation
    cardinality (``nnz``); ``p = inf`` recovers the maximum degree.

    Attributes:
        from_indices: Conditioning index names ``X``.
        to_indices: Index names ``Y`` whose degrees (given ``X``) are counted.
        p: The norm order (``math.inf`` is allowed).
        value: ``||deg(to_indices | from_indices)||_p`` in linear space.
    """

    from_indices: frozenset[Field]
    to_indices: frozenset[Field]
    p: float
    value: float


def _lp_norm(counts: np.ndarray, p: float) -> float:
    """``ell_p``-norm of the nonzero degrees in ``counts``."""
    nz = counts[counts != 0].astype(np.float64)
    if nz.size == 0:
        return 0.0
    if math.isinf(p):
        return float(nz.max())
    return float(np.power(np.power(nz, p).sum(), 1.0 / p))


def _flow_bound(lpdcs: list[LpDC], target_vars: list[Field]) -> float | None:
    """
    Compute ``log2`` of the LP (flow) upper bound on the joint size of
    ``target_vars`` given the constraints ``lpdcs``.

    This is a faithful port of the ``LPflow`` network-flow LP from the LpBound
    reference implementation (``fdbresearch/LpBound``, ``LpFlow/flow_bound.cpp``).
    A single set of per-constraint coefficients ``a_i >= 0`` must simultaneously
    route one unit of flow from the sink (the empty set) to every target
    variable; the bound is ``min sum_i a_i * log2(value_i)``.

    Returns ``None`` if the LP is infeasible or the solver fails.
    """
    if not target_vars or not lpdcs:
        return None

    empty: frozenset[Field] = frozenset()

    def is_simple(dc: LpDC) -> bool:
        return len(dc.from_indices) <= 1

    simple = [dc for dc in lpdcs if is_simple(dc)]

    # --- construct the flow network (construct_flow_network) ---
    vertices: set[frozenset[Field]] = {frozenset({v}) for v in target_vars}
    edges: set[tuple[frozenset[Field], frozenset[Field]]] = set()

    def add_edge(x: frozenset[Field], y: frozenset[Field]) -> None:
        if not (x <= y or y <= x):
            return
        if x != y and (y, x) not in edges:
            vertices.add(x)
            vertices.add(y)
            edges.add((x, y))

    for dc in simple:
        add_edge(dc.from_indices, dc.to_indices)
        if not math.isinf(dc.p):
            add_edge(empty, dc.from_indices)
    for dc in simple:
        add_edge(dc.to_indices, empty)
        for elem in dc.to_indices:
            add_edge(dc.to_indices, frozenset({elem}))

    edge_list = sorted(
        edges, key=lambda e: (sorted(map(str, e[0])), sorted(map(str, e[1])))
    )

    # --- variable layout: a_i for each dc, then f_{t, edge} per target ---
    n_dc = len(lpdcs)
    flow_col: dict[tuple[int, frozenset[Field], frozenset[Field]], int] = {}
    col = n_dc
    n_targets = len(target_vars)
    for t in range(n_targets):
        for x, y in edge_list:
            flow_col[(t, x, y)] = col
            col += 1
    n_vars = col

    # --- constraints, accumulated as (coeffs, lb) meaning sum(coeffs*x) >= lb ---
    cons: dict[tuple[int, frozenset[Field]], list] = {}  # (t, Z) -> [dict, lb]
    caps: dict[tuple[int, frozenset[Field], frozenset[Field]], list] = {}

    for t in range(n_targets):
        for z in vertices:
            if not z:
                continue
            lb = 1.0 if (len(z) == 1 and next(iter(z)) == target_vars[t]) else 0.0
            cons[(t, z)] = [{}, lb]

    for t in range(n_targets):
        for x, y in edge_list:
            f = flow_col[(t, x, y)]
            if x:
                cons[(t, x)][0][f] = cons[(t, x)][0].get(f, 0.0) - 1.0
            if y:
                cons[(t, y)][0][f] = cons[(t, y)][0].get(f, 0.0) + 1.0
            if len(x) <= len(y):
                caps[(t, x, y)] = [{f: -1.0}, 0.0]

    for i, dc in enumerate(lpdcs):
        for t in range(n_targets):
            if is_simple(dc):
                if len(dc.from_indices) != len(dc.to_indices):
                    cap_key = (t, dc.from_indices, dc.to_indices)
                    if cap_key in caps:
                        caps[cap_key][0][i] = caps[cap_key][0].get(i, 0.0) + 1.0
                if not math.isinf(dc.p) and dc.from_indices:
                    cap_key = (t, empty, dc.from_indices)
                    if cap_key in caps:
                        caps[cap_key][0][i] = caps[cap_key][0].get(i, 0.0) + 1.0 / dc.p
            else:
                for elem in dc.to_indices - dc.from_indices:
                    con_key = (t, frozenset({elem}))
                    if con_key in cons:
                        cons[con_key][0][i] = cons[con_key][0].get(i, 0.0) + 1.0
                if not math.isinf(dc.p):
                    for elem in dc.from_indices:
                        con_key = (t, frozenset({elem}))
                        if con_key in cons:
                            cons[con_key][0][i] = (
                                cons[con_key][0].get(i, 0.0) + 1.0 / dc.p
                            )

    # --- assemble A_ub x <= b_ub (each "expr >= lb" becomes "-expr <= -lb") ---
    rows_i: list[int] = []
    cols_i: list[int] = []
    data: list[float] = []
    b_ub: list[float] = []
    row = 0
    for coeffs, lb in list(cons.values()) + list(caps.values()):
        for c, v in coeffs.items():
            rows_i.append(row)
            cols_i.append(c)
            data.append(-v)
        b_ub.append(-lb)
        row += 1

    if row == 0:
        return None

    a_ub = coo_matrix((data, (rows_i, cols_i)), shape=(row, n_vars))

    c = np.zeros(n_vars)
    for i, dc in enumerate(lpdcs):
        c[i] = math.log2(dc.value)

    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * n_dc
    bounds += [(None, None)] * (n_vars - n_dc)
    for (_t, x, y), idx in flow_col.items():
        bounds[idx] = (None, None) if len(x) <= len(y) else (0.0, None)

    res = linprog(c, A_ub=a_ub, b_ub=np.array(b_ub), bounds=bounds, method="highs")
    if not res.success:
        return None
    return float(res.fun)


class LPStats(NumericStats):
    """
    Structural statistics implementing the LP (LpBound) upper-bound framework.

    Like :class:`DCStats`, ``LPStats`` scans a tensor and stores degree
    statistics, but records ``ell_p``-norms of degree sequences for a
    configurable set of ``p`` values rather than only the distinct-count and
    max-degree endpoints. The size upper bound is computed by solving the
    ``LPflow`` network-flow linear program over those constraints.
    """

    def __init__(
        self,
        tensor: Any,
        fields: tuple[Field, ...],
        ps: Iterable[float] = DEFAULT_PS,
    ):
        super().__init__(tensor, fields)
        self.ps: tuple[float, ...] = tuple(ps)
        self.lpdcs = self._structure_to_lpdcs(tensor, fields)

    def _structure_to_lpdcs(self, arr: Tensor, fields: Iterable[Field]) -> set[LpDC]:
        fields = list(fields)
        if arr.ndim == 0:
            return {LpDC(frozenset(), frozenset(), 1.0, 1.0)}

        counts, nnz = degree_count_scan(arr, fields, self.fill_value)
        lpdcs: set[LpDC] = set()
        all_fields = frozenset(fields)
        for i, field in enumerate(fields):
            proj = float(np.count_nonzero(counts[i]))
            lpdcs.add(LpDC(frozenset(), frozenset({field}), 1.0, proj))
            for p in self.ps:
                norm = _lp_norm(counts[i], p)
                lpdcs.add(LpDC(frozenset({field}), all_fields, p, norm))
        lpdcs.add(LpDC(frozenset(), all_fields, 1.0, float(nnz)))
        return lpdcs

    def estimate_non_fill_values(self) -> float:
        idx: frozenset[Field] = frozenset(self.dim_sizes.keys())
        if len(idx) == 0:
            return 1.0

        # An empty (all-fill) input has no non-fill values.
        for dc in self.lpdcs:
            if len(dc.from_indices) == 0 and dc.value == 0:
                return 0.0

        positive = [dc for dc in self.lpdcs if dc.value > 0]
        dense_cap = float(self.get_dim_space_size(idx))
        sol = _flow_bound(positive, list(self.dim_sizes.keys()))
        if sol is None:
            return dense_cap
        try:
            bound = math.pow(2.0, sol)
        except OverflowError:
            return dense_cap
        return min(bound, dense_cap)

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        lpdc_embedding = [
            dc.value
            for dc in sorted(
                self.lpdcs,
                key=lambda dc: (
                    tuple(sorted(str(f) for f in dc.from_indices)),
                    tuple(sorted(str(f) for f in dc.to_indices)),
                    dc.p,
                ),
            )
        ]
        return np.log2(np.array([*sizes, *lpdc_embedding]))


class LPStatsFactory(BaseTensorStatsFactory["LPStats"]):
    def __init__(self, ps: Iterable[float] = DEFAULT_PS):
        super().__init__(LPStats)
        self.ps: tuple[float, ...] = tuple(ps)

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> LPStats:
        return LPStats(tensor, fields, self.ps)

    def _mapjoin_union(self, op: FinchOperator, *union_args: LPStats) -> LPStats:
        base_stats = super()._mapjoin_defs(op, *union_args)

        if len(union_args) == 1:
            return LPStats.from_base_stats(
                union_args[0], lpdcs=set(union_args[0].lpdcs), ps=self.ps
            )

        Key = tuple[frozenset[Field], frozenset[Field], float]
        dc_keys: Counter[Key] = Counter()
        stats_dcs: list[dict[Key, float]] = []
        for stats in union_args:
            dcs: dict[Key, float] = {}
            Z = tuple(x for x in base_stats.index_order if x not in stats.index_order)
            Z_dim_size = base_stats.get_dim_space_size(Z)
            for dc in stats.lpdcs:
                new_key = (dc.from_indices, dc.to_indices, dc.p)
                dcs[new_key] = dc.value
                dc_keys[new_key] += 1

                ext_key = (dc.from_indices, dc.to_indices | frozenset(Z), dc.p)
                if ext_key not in dcs:
                    dc_keys[ext_key] += 1
                prev = dcs.get(ext_key, math.inf)
                dcs[ext_key] = min(prev, dc.value * Z_dim_size)
            stats_dcs.append(dcs)

        new_dcs: dict[Key, float] = {}
        for key, count in dc_keys.items():
            if count == len(union_args):
                total = sum(d.get(key, 0.0) for d in stats_dcs)
                _, Y, _ = key
                if Y.issubset(base_stats.index_order):
                    total = min(total, base_stats.get_dim_space_size(Y))
                new_dcs[key] = min(float(2**64), total)

        new_stats = {LpDC(X, Y, p, d) for (X, Y, p), d in new_dcs.items()}
        return LPStats.from_base_stats(base_stats, lpdcs=new_stats, ps=self.ps)

    def _mapjoin_join(self, op: FinchOperator, *join_args: LPStats) -> LPStats:
        base_stats = super()._mapjoin_defs(op, *join_args)

        if len(join_args) == 1:
            return LPStats.from_base_stats(
                base_stats, lpdcs=set(join_args[0].lpdcs), ps=self.ps
            )

        Key = tuple[frozenset[Field], frozenset[Field], float]
        new_dc: dict[Key, float] = {}
        for stats in join_args:
            for dc in stats.lpdcs:
                key = (dc.from_indices, dc.to_indices, dc.p)
                if dc.value < new_dc.get(key, math.inf):
                    new_dc[key] = dc.value

        new_stats = {LpDC(X, Y, p, d) for (X, Y, p), d in new_dc.items()}
        return LPStats.from_base_stats(base_stats, lpdcs=new_stats, ps=self.ps)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: LPStats,
    ) -> LPStats:
        if len(reduce_indices) == 0:
            base_stats: BaseTensorStats = stats.copy()
        else:
            base_stats = self.aggregate_def(op, init, reduce_indices, stats)
        lpdcs = set(stats.lpdcs) if isinstance(stats, LPStats) else set()
        return LPStats.from_base_stats(base_stats, lpdcs=lpdcs, ps=self.ps)

    def relabel(self, stats: LPStats, relabel_indices: tuple[Field, ...]) -> LPStats:
        base_stats = self.relabel_def(stats, relabel_indices)
        if isinstance(stats, LPStats):
            mapping = dict(zip(stats.index_order, relabel_indices, strict=True))
            lpdcs = {
                LpDC(
                    frozenset(mapping.get(f, f) for f in dc.from_indices),
                    frozenset(mapping.get(f, f) for f in dc.to_indices),
                    dc.p,
                    dc.value,
                )
                for dc in stats.lpdcs
            }
        else:
            lpdcs = set()
        return LPStats.from_base_stats(base_stats, lpdcs=lpdcs, ps=self.ps)

    def reorder(self, stats: LPStats, reorder_indices: tuple[Field, ...]) -> LPStats:
        base_stats = self.reorder_def(stats, reorder_indices)
        lpdcs = set(stats.lpdcs) if isinstance(stats, LPStats) else set()
        return LPStats.from_base_stats(base_stats, lpdcs=lpdcs, ps=self.ps)
