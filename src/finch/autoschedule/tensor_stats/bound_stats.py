from __future__ import annotations

import math
from abc import abstractmethod
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from finch.algebra import Tensor
from finch.algebra.algebra import FinchOperator
from finch.finch_logic import Field, StatsFactory

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory
from .util import degree_count_scan

__all__ = [
    "DC",
    "DEFAULT_PS",
    "BoundStats",
    "BoundStatsFactory",
    "DCStats",
    "DCStatsFactory",
    "LPStats",
    "LPStatsFactory",
]

# Default ell_p norm orders kept by LPStats.
DEFAULT_PS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, math.inf)


@dataclass(frozen=True)
class DC:
    """
    A degree constraint: ``value`` bounds ``||deg(to_indices | from_indices)||_p``.

    ``p = inf`` (the default) is the classic max-degree constraint; ``p = 1`` is
    the distinct count; intermediate ``p`` capture skew between the two.
    """

    from_indices: frozenset[Field]
    to_indices: frozenset[Field]
    value: float
    p: float = math.inf


def _lp_norm(counts: np.ndarray, p: float) -> float:
    """``ell_p``-norm of the nonzero degrees in ``counts``."""
    nz = counts[counts != 0].astype(np.float64)
    if nz.size == 0:
        return 0.0
    if math.isinf(p):
        return float(nz.max())
    return float(np.power(np.power(nz, p).sum(), 1.0 / p))


# ─────────────────────────────── base classes ────────────────────────────────


class BoundStats(NumericStats):
    """
    Base for degree-constraint upper-bound statistics: a guaranteed bound on the
    number of non-fill values, built from ``ell_p``-norms of degree sequences.

    Subclasses differ only in which ``p`` values they keep (``ps``) and how they
    turn the constraints into an estimate (:meth:`estimate_non_fill_values`); the
    scan and all propagation (in :class:`BoundStatsFactory`) are shared.
    """

    default_ps: ClassVar[tuple[float, ...]] = ()

    def __init__(
        self,
        base: BaseTensorStats,
        dcs: Iterable[DC],
        ps: Iterable[float] | None = None,
    ):
        super().__init__(base)
        self.ps: tuple[float, ...] = tuple(self.default_ps if ps is None else ps)
        self.dcs = set(dcs)

    @abstractmethod
    def estimate_non_fill_values(self) -> float: ...

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        dc_embedding = [
            dc.value
            for dc in sorted(
                self.dcs,
                key=lambda dc: (
                    tuple(sorted(str(f) for f in dc.from_indices)),
                    tuple(sorted(str(f) for f in dc.to_indices)),
                    dc.p,
                ),
            )
        ]
        return np.log2(np.array([*sizes, *dc_embedding]))


TS = TypeVar("TS", bound="BoundStats")

# A degree-constraint dictionary key: (from_indices, to_indices, p).
_Key = tuple[frozenset[Field], frozenset[Field], float]


class BoundStatsFactory(BaseTensorStatsFactory[TS], StatsFactory[TS], Generic[TS]):
    """
    Shared statistics propagation for :class:`BoundStats` subclasses, combining
    degree constraints keyed by ``(from_indices, to_indices, p)``. The only
    per-subclass knob is ``ps``.
    """

    def __init__(self, stats_cls: type[TS], ps: Iterable[float] | None = None):
        super().__init__(stats_cls)
        self.ps: tuple[float, ...] = tuple(stats_cls.default_ps if ps is None else ps)

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> TS:
        base = super().__call__(tensor, fields)
        dcs = self.structure_to_dcs(tensor, fields, base.fill_value, self.ps)
        return self.stats_cls(base, dcs, self.ps)

    # Per field: the distinct-value count DC(∅, {i}) and, for each p, the ell_p
    # norm DC({i}, {*fields}, p); plus the full-tensor nnz DC(∅, {*fields}). The
    # ∅-conditioned records are length-1 sequences (p-independent), so kept once.
    @staticmethod
    def structure_to_dcs(
        tensor: Tensor,
        fields: Iterable[Field],
        fill_value: Any,
        ps: Iterable[float],
    ) -> set[DC]:
        fields = list(fields)
        if tensor.ndim == 0:
            return {DC(frozenset(), frozenset(), 1.0)}

        ps = tuple(ps)
        counts, nnz = degree_count_scan(tensor, fields, fill_value)
        dcs: set[DC] = set()
        all_fields = frozenset(fields)
        for i, field in enumerate(fields):
            proj = float(np.count_nonzero(counts[i]))
            dcs.add(DC(frozenset(), frozenset({field}), proj))
            for p in ps:
                dcs.add(DC(frozenset({field}), all_fields, _lp_norm(counts[i], p), p))
        dcs.add(DC(frozenset(), all_fields, float(nnz)))
        return dcs

    def _rebuild(self, base: BaseTensorStats, dcs: Iterable[DC]) -> TS:
        return self.stats_cls(base, dcs, self.ps)

    def _mapjoin_union(self, op: FinchOperator, *union_args: TS) -> TS:
        base = super()._mapjoin_defs(op, *union_args)

        if len(union_args) == 1:
            return self._rebuild(base, union_args[0].dcs)

        dc_keys: Counter[_Key] = Counter()
        stats_dcs: list[dict[_Key, float]] = []
        for stats in union_args:
            dcs: dict[_Key, float] = {}
            extra_fields = tuple(
                x for x in base.index_order if x not in stats.index_order
            )
            extra_size = base.get_dim_space_size(extra_fields)
            for dc in stats.dcs:
                new_key = (dc.from_indices, dc.to_indices, dc.p)
                dcs[new_key] = dc.value
                dc_keys[new_key] += 1

                ext_key = (
                    dc.from_indices,
                    dc.to_indices | frozenset(extra_fields),
                    dc.p,
                )
                if ext_key not in dcs:
                    dc_keys[ext_key] += 1
                prev = dcs.get(ext_key, math.inf)
                dcs[ext_key] = min(prev, dc.value * extra_size)
            stats_dcs.append(dcs)

        new_dcs: dict[_Key, float] = {}
        for key, count in dc_keys.items():
            if count == len(union_args):
                total = sum(d.get(key, 0.0) for d in stats_dcs)
                _, to_indices, _ = key
                if to_indices.issubset(base.index_order):
                    total = min(total, base.get_dim_space_size(to_indices))
                new_dcs[key] = min(float(2**64), total)

        new_stats = {
            DC(from_indices, to_indices, value, p)
            for (from_indices, to_indices, p), value in new_dcs.items()
        }
        return self._rebuild(base, new_stats)

    def _mapjoin_join(self, op: FinchOperator, *join_args: TS) -> TS:
        base = super()._mapjoin_defs(op, *join_args)

        if len(join_args) == 1:
            return self._rebuild(base, join_args[0].dcs)

        new_dc: dict[_Key, float] = {}
        for stats in join_args:
            for dc in stats.dcs:
                key = (dc.from_indices, dc.to_indices, dc.p)
                if dc.value < new_dc.get(key, math.inf):
                    new_dc[key] = dc.value

        new_stats = {
            DC(from_indices, to_indices, value, p)
            for (from_indices, to_indices, p), value in new_dc.items()
        }
        return self._rebuild(base, new_stats)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: TS,
    ) -> TS:
        if len(reduce_indices) == 0:
            base: BaseTensorStats = stats.copy()
        else:
            base = self.aggregate_def(op, init, reduce_indices, stats)
        return self._rebuild(base, stats.dcs)

    def relabel(self, stats: TS, relabel_indices: tuple[Field, ...]) -> TS:
        base = self.relabel_def(stats, relabel_indices)
        mapping = dict(zip(stats.index_order, relabel_indices, strict=True))
        dcs = {
            DC(
                frozenset(mapping.get(f, f) for f in dc.from_indices),
                frozenset(mapping.get(f, f) for f in dc.to_indices),
                dc.value,
                dc.p,
            )
            for dc in stats.dcs
        }
        return self._rebuild(base, dcs)

    def reorder(self, stats: TS, reorder_indices: tuple[Field, ...]) -> TS:
        base = self.reorder_def(stats, reorder_indices)
        return self._rebuild(base, stats.dcs)


# ────────────────────────────────── DCStats ──────────────────────────────────


class DCStats(BoundStats):
    """
    Classic degree constraints -- max degrees (``p = inf``) and distinct counts.
    Estimates size with a best-first product search over the constraints,
    clamped by the tensor's dense capacity.
    """

    default_ps: ClassVar[tuple[float, ...]] = (math.inf,)

    def estimate_non_fill_values(self) -> float:
        """Smallest degree-constraint product covering all indices, clamped by
        dense capacity."""
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


class DCStatsFactory(BoundStatsFactory["DCStats"]):
    """Factory for :class:`DCStats` -- keeps only max-degree (``p = inf``)."""

    def __init__(self):
        super().__init__(DCStats)

    @staticmethod
    def structure_to_dcs(
        tensor: Tensor,
        fields: Iterable[Field],
        fill_value: Any,
        ps: Iterable[float] = DCStats.default_ps,
    ) -> set[DC]:
        return BoundStatsFactory.structure_to_dcs(tensor, fields, fill_value, ps)


# ────────────────────────────────── LPStats ──────────────────────────────────


def _flow_bound(dcs: list[DC], target_vars: list[Field]) -> float | None:
    """
    ``log2`` of the LPflow upper bound on the joint size of ``target_vars``.

    A faithful port of ``LpFlow/flow_bound.cpp`` from the LpBound reference
    (``fdbresearch/LpBound``): coefficients ``a_i >= 0`` route one unit of flow
    to every target simultaneously, minimizing ``sum_i a_i * log2(value_i)``.
    Returns ``None`` if the LP is infeasible or the solver fails.
    """
    if not target_vars or not dcs:
        return None

    empty: frozenset[Field] = frozenset()

    def is_simple(dc: DC) -> bool:
        return len(dc.from_indices) <= 1

    simple = [dc for dc in dcs if is_simple(dc)]

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
    n_dc = len(dcs)
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

    for i, dc in enumerate(dcs):
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
    for i, dc in enumerate(dcs):
        c[i] = math.log2(dc.value)

    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * n_dc
    bounds += [(None, None)] * (n_vars - n_dc)
    for (_t, x, y), idx in flow_col.items():
        bounds[idx] = (None, None) if len(x) <= len(y) else (0.0, None)

    res = linprog(c, A_ub=a_ub, b_ub=np.array(b_ub), bounds=bounds, method="highs")
    if not res.success:
        return None
    return float(res.fun)


class LPStats(BoundStats):
    """
    LpBound statistics: keeps ``ell_p``-norms of degree sequences for a
    configurable set of ``p`` (not just max degree), and bounds size via the
    ``LPflow`` network-flow LP (:func:`_flow_bound`).
    """

    default_ps: ClassVar[tuple[float, ...]] = DEFAULT_PS

    def estimate_non_fill_values(self) -> float:
        idx: frozenset[Field] = frozenset(self.dim_sizes.keys())
        if len(idx) == 0:
            return 1.0

        # An empty (all-fill) input has no non-fill values.
        for dc in self.dcs:
            if len(dc.from_indices) == 0 and dc.value == 0:
                return 0.0

        positive = [dc for dc in self.dcs if dc.value > 0]
        dense_cap = float(self.get_dim_space_size(idx))
        sol = _flow_bound(positive, list(self.dim_sizes.keys()))
        if sol is None:
            return dense_cap
        try:
            bound = math.pow(2.0, sol)
        except OverflowError:
            return dense_cap
        return min(bound, dense_cap)


class LPStatsFactory(BoundStatsFactory["LPStats"]):
    """Factory for :class:`LPStats` with a configurable set of ``p`` norms."""

    def __init__(self, ps: Iterable[float] | None = None):
        super().__init__(LPStats, ps)
