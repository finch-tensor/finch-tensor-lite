"""
Manual / demo: ``branch_and_bound_dfs(k=inf)`` with ``memo[vars_key] = (cost, order)``
using a **four-index** chain (``i,j,k,l``).

``branch_and_bound_dfs`` prints ``memo`` on each stack ``pop`` and ``memo`` assign;
use ``pytest -s`` or ``python tests/test1.py`` to see output.

    cd finch-tensor-lite
    python -m pytest tests/test1.py -s --no-cov
    python tests/test1.py
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

import finchlite as fl
from finchlite.algebra import ffunc
from finchlite.autoschedule.galley.logical_optimizer import AnnotatedQuery
from finchlite.autoschedule.galley.logical_optimizer.branch_and_bound import (
    _aq_with_stats,
    branch_and_bound_dfs,
)
from finchlite.autoschedule.tensor_stats import DenseStatsFactory
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Query,
    Table,
)

_DENSE_STATS_FACTORY = DenseStatsFactory()


def _make_aq_four_index_chain() -> AnnotatedQuery:
    """sum_{i,j,k,l} A[i,j]*B[j,k]*C[k,l] — one component, four indices."""
    A = fl.asarray(np.ones((3, 10)))
    B = fl.asarray(np.ones((10, 5)))
    C = fl.asarray(np.ones((5, 2)))
    q = Query(
        Alias("out"),
        Aggregate(
            Literal(ffunc.add),
            Literal(0),
            MapJoin(
                Literal(ffunc.mul),
                (
                    Table(Literal(A), (Field("i"), Field("j"))),
                    Table(Literal(B), (Field("j"), Field("k"))),
                    Table(Literal(C), (Field("k"), Field("l"))),
                ),
            ),
            (Field("i"), Field("j"), Field("k"), Field("l")),
        ),
    )
    return AnnotatedQuery(_DENSE_STATS_FACTORY, q, bindings=OrderedDict())


def _fmt_vars_key(vk: frozenset) -> str:
    if not vk:
        return "∅"
    return "{" + ", ".join(sorted(repr(x) for x in vk)) + "}"


def run_dfs_perm_memo_demo() -> None:
    aq = _aq_with_stats(_make_aq_four_index_chain())
    component = aq.connected_components[0]
    assert len(component) >= 4
    (order, _, _, total_cost), costs, perm_by_vars = branch_and_bound_dfs(
        aq, component, float("inf"), OrderedDict()
    )

    print("\n=== branch_and_bound_dfs result (exports split from memo) ===")
    print(f"best order: {[repr(x) for x in order]}")
    print(f"total cost: {total_cost}\n")

    print("optimal_subquery_costs (= memo[..][0]):")
    for vk, c in costs.items():
        print(f"  {_fmt_vars_key(vk)} -> {c}")

    print("\noptimal_perm_by_vars (= memo[..][1]):")
    for vk, perm in perm_by_vars.items():
        print(f"  {_fmt_vars_key(vk)} -> {[repr(x) for x in perm]}")


def test_dfs_perm_memo_smoke() -> None:
    aq = _aq_with_stats(_make_aq_four_index_chain())
    component = aq.connected_components[0]
    assert len(component) >= 4
    (_, _, _, cost), _, perm_by_vars = branch_and_bound_dfs(
        aq, component, float("inf"), OrderedDict()
    )
    assert cost > 0
    assert frozenset(component) in perm_by_vars


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.is_dir():
        sys.path.insert(0, str(src))
    run_dfs_perm_memo_demo()
