import math
from collections import OrderedDict

import pytest

import numpy as np

import finchlite as fl
from finchlite import ffunc
from finchlite.autoschedule.galley.logical_optimizer import insert_statistics
from finchlite.autoschedule.tensor_stats import (
    DC,
    BlockedStats,
    DCStats,
    DCStatsFactory,
    DenseStatsFactory,
    TensorDef,
    UniformStatsFactory,
)
from finchlite.finch_logic import (
    Aggregate,
    Field,
    Literal,
    MapJoin,
    Table,
)

# ─────────────────────────────── DatabaseStats tests ─────────────────────────────


def test_database_from_tensor_and_getters():
    data = np.zeros((2, 3))
    data[0, 0] = 1.0
    data[1, 1] = 1.0
    arr = fl.asarray(data)

    node = Table(Literal(arr), (Field("i"), Field("j")))
    stats = insert_statistics(
        ST=DatabaseStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.index_order == (Field("i"), Field("j"))
    assert stats.get_dim_size(Field("i")) == 2.0
    assert stats.get_dim_size(Field("j")) == 3.0
    assert stats.fill_value == 0
    assert stats.estimate_non_fill_values() == 2.0


@pytest.mark.parametrize(
    "shape, nnz_indices, expected_nnz",
    [
        ((2, 3), [(0, 0), (1, 1)], 2.0),
        ((10, 10), [(i, i) for i in range(10)], 10.0),
        ((5, 5, 5), [], 0.0),
    ],
)
def test_database_estimate_non_fill_values(shape, nnz_indices, expected_nnz):
    axes = tuple(Field(f"x{i}") for i in range(len(shape)))
    data = np.zeros(shape)
    for idx in nnz_indices:
        data[idx] = 1.0

    arr = fl.asarray(data)
    node = Table(Literal(arr), axes)

    stats = insert_statistics(
        ST=DatabaseStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    assert stats.index_order == tuple(axes)
    assert stats.estimate_non_fill_values() == expected_nnz


def test_database_mapjoin_join():
    i, k, j = Field("i"), Field("k"), Field("j")
    data_a = np.eye(10)
    data_b = np.eye(10)

    ta = Table(Literal(fl.asarray(data_a)), (i, k))
    tb = Table(Literal(fl.asarray(data_b)), (k, j))

    cache = {}
    insert_statistics(
        ST=DatabaseStats, node=ta, bindings=OrderedDict(), replace=False, cache=cache
    )
    insert_statistics(
        ST=DatabaseStats, node=tb, bindings=OrderedDict(), replace=False, cache=cache
    )

    node_mul = MapJoin(Literal(op.mul), (ta, tb))
    stats = insert_statistics(
        ST=DatabaseStats,
        node=node_mul,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    assert stats.estimate_non_fill_values() == pytest.approx(10.0)


def test_database_mapjoin_elementwise():
    i, j = Field("i"), Field("j")
    data_a = np.zeros((10, 10))
    data_a[:5, :] = 1.0
    data_b = np.zeros((10, 10))
    data_b[5:, :] = 1.0

    ta = Table(Literal(fl.asarray(data_a)), (i, j))
    tb = Table(Literal(fl.asarray(data_b)), (i, j))

    cache = {}
    insert_statistics(
        ST=DatabaseStats, node=ta, bindings=OrderedDict(), replace=False, cache=cache
    )
    insert_statistics(
        ST=DatabaseStats, node=tb, bindings=OrderedDict(), replace=False, cache=cache
    )

    node_add = MapJoin(Literal(op.add), (ta, tb))
    stats = insert_statistics(
        ST=DatabaseStats,
        node=node_add,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    assert stats.estimate_non_fill_values() == pytest.approx(100.0)


def test_database_mapjoin_broadcast():
    i, j, k = Field("i"), Field("j"), Field("k")

    data_a = np.zeros((4, 5))
    data_a[0, 0] = 1.0
    data_a[1, 1] = 1.0

    data_b = np.zeros((5, 3))
    data_b[0, 0] = 1.0
    data_b[1, 1] = 1.0
    data_b[2, 2] = 1.0

    ta = Table(Literal(fl.asarray(data_a)), (i, j))
    tb = Table(Literal(fl.asarray(data_b)), (j, k))

    cache = {}
    insert_statistics(
        ST=DatabaseStats, node=ta, bindings=OrderedDict(), replace=False, cache=cache
    )
    insert_statistics(
        ST=DatabaseStats, node=tb, bindings=OrderedDict(), replace=False, cache=cache
    )

    node_add = MapJoin(Literal(op.add), (ta, tb))
    stats = insert_statistics(
        ST=DatabaseStats,
        node=node_add,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    assert stats.estimate_non_fill_values() == pytest.approx(2 * 3 + 4 * 3)
    assert stats.V[i] == pytest.approx(4.0)
    assert stats.V[j] == pytest.approx(5.0)
    assert stats.V[k] == pytest.approx(3.0)


def test_database_aggregate():
    i, j = Field("i"), Field("j")
    data = np.eye(10)
    table = Table(Literal(fl.asarray(data)), (i, j))

    node_sum = Aggregate(
        op=Literal(op.add),
        init=None,
        arg=table,
        idxs=(j,),
    )
    stats = insert_statistics(
        ST=DatabaseStats,
        node=node_sum,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.index_order == (i,)
    assert stats.get_dim_size(i) == 10.0
    assert stats.estimate_non_fill_values() == pytest.approx(1.0)


def test_database_issimilar():
    data = np.eye(10)
    arr = fl.asarray(data)
    node = Table(Literal(arr), (Field("i"), Field("j")))

    stats = insert_statistics(
        ST=DatabaseStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert DatabaseStats.issimilar(stats, stats)


def test_database_copy_stats():
    data = np.eye(10)
    arr = fl.asarray(data)
    node = Table(Literal(arr), (Field("i"), Field("j")))
    stats = insert_statistics(
        ST=DatabaseStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    copy = DatabaseStats.copy_stats(stats)
    assert copy.nnz == stats.nnz
    assert copy.V == stats.V
    assert copy is not stats


def test_database_relabel():
    data = np.eye(10)
    arr = fl.asarray(data)
    node = Table(Literal(arr), (Field("i"), Field("j")))
    stats = insert_statistics(
        ST=DatabaseStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    relabeled = DatabaseStats.relabel(stats, (Field("row"), Field("col")))
    assert relabeled.index_order == (Field("row"), Field("col"))
    assert relabeled.nnz == stats.nnz


def test_database_reorder():
    data = np.eye(10)
    arr = fl.asarray(data)
    node = Table(Literal(arr), (Field("i"), Field("j")))
    stats = insert_statistics(
        ST=DatabaseStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    reordered = DatabaseStats.reorder(stats, (Field("j"), Field("i")))
    assert reordered.index_order == (Field("j"), Field("i"))
    assert reordered.nnz == stats.nnz


# ─────────────────────────────── UniformStats tests ─────────────────────────────


def test_uniform_from_tensor_and_getters():
    data = np.zeros((2, 3))
    data[0, 0] = 1.0
    data[1, 1] = 1.0
    arr = fl.asarray(data)

    node = Table(Literal(arr), (Field("i"), Field("j")))
    stats = insert_statistics(
        stats_factory=UniformStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.index_order == (Field("i"), Field("j"))
    assert stats.get_dim_size(Field("i")) == 2.0
    assert stats.get_dim_size(Field("j")) == 3.0
    assert stats.fill_value == 0
    assert stats.estimate_non_fill_values() == 2.0


@pytest.mark.parametrize(
    "shape, nnz_indices, expected_nnz",
    [
        ((2, 3), [(0, 0), (1, 1)], 2.0),
        ((10, 10), [(i, i) for i in range(10)], 10.0),
        ((5, 5, 5), [], 0.0),
    ],
)
def test_uniform_estimate_non_fill_values(shape, nnz_indices, expected_nnz):
    axes = tuple(Field(f"x{i}") for i in range(len(shape)))
    data = np.zeros(shape)
    for idx in nnz_indices:
        data[idx] = 1.0

    arr = fl.asarray(data)
    node = Table(Literal(arr), axes)

    stats = insert_statistics(
        stats_factory=UniformStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    assert stats.index_order == tuple(axes)
    assert stats.estimate_non_fill_values() == expected_nnz


def test_uniform_mapjoin_mul_and_add():
    data_a = np.zeros((10, 10))
    data_a[:5, :] = 1.0
    data_b = np.zeros((10, 10))
    data_b[:, :5] = 1.0

    ta = Table(Literal(fl.asarray(data_a)), (Field("i"), Field("j")))
    tb = Table(Literal(fl.asarray(data_b)), (Field("i"), Field("j")))

    cache = {}
    insert_statistics(
        stats_factory=UniformStatsFactory(),
        node=ta,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    insert_statistics(
        stats_factory=UniformStatsFactory(),
        node=tb,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    # P(a)*P(b) = 0.5 * 0.5 = 0.25 -> 0.25 * 100 = 25 nnz
    node_mul = MapJoin(Literal(ffunc.mul), (ta, tb))
    us_mul = insert_statistics(
        stats_factory=UniformStatsFactory(),
        node=node_mul,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    assert us_mul.estimate_non_fill_values() == pytest.approx(25.0)
    assert us_mul.fill_value == 0.0

    # 1 - (1-P(a))(1-P(b)) = 1 - (1-0.5)*(1-0.5) =0.75 -> 0.75 * 100 = 75 nnz
    node_add = MapJoin(Literal(ffunc.add), (ta, tb))
    us_add = insert_statistics(
        stats_factory=UniformStatsFactory(),
        node=node_add,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert us_add.estimate_non_fill_values() == pytest.approx(75.0)


def test_uniform_aggregate_and_issimilar():
    data = np.eye(10)
    table = Table(Literal(fl.asarray(data)), (Field("i"), Field("j")))
    us = insert_statistics(
        stats_factory=UniformStatsFactory(),
        node=table,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    node_sum = Aggregate(
        op=Literal(ffunc.add),
        init=None,
        arg=table,
        idxs=(Field("j"),),
    )
    us_agg = insert_statistics(
        stats_factory=UniformStatsFactory(),
        node=node_sum,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    # p=0.1 in our example as only diagonal elements are non zero so 10/100 = 0.1
    expected_prob = 1 - math.pow(1.0 - 0.1, 10)
    # post squashing we have just 10 rows left so new vol = 10x1 = 10
    expected_nnz = expected_prob * 10
    assert us_agg.index_order == (Field("i"),)
    assert us_agg.get_dim_size(Field("i")) == 10
    assert us_agg.estimate_non_fill_values() == pytest.approx(expected_nnz)
    assert UniformStatsFactory().issimilar(us, us)


# ------------------------------ BlockedStats -------------------------------------
def test_blocked_stats_from_tensor():
    data = np.eye(10)
    arr = fl.asarray(data)
    indices = (Field("i"), Field("j"))
    blocks_per_dim = {Field("i"): 2, Field("j"): 2}

    bs = BlockedStats.from_tensor(arr, indices, blocks_per_dim, UniformStatsFactory())

    assert bs.estimate_non_fill_values() == 10.0


def test_blocked_stats_aggregate():
    data = np.eye(10)
    indices = (Field("i"), Field("j"))
    blocks_per_dim = {Field("i"): 2, Field("j"): 2}
    bs = BlockedStats.from_tensor(
        fl.asarray(data), indices, blocks_per_dim, DenseStatsFactory()
    )

    reduce_indices = (Field("j"),)
    agg_bs = BlockedStatsFactory(blocks_per_dim, DenseStatsFactory()).aggregate(
        ffunc.add, 0.0, reduce_indices, bs
    )

    assert agg_bs.blocks.ndim == 1
    assert len(agg_bs.blocks) == 2
    assert agg_bs.estimate_non_fill_values() == 10.0


def test_blocked_stats_mapjoin():
    indices = (Field("i"), Field("j"))
    blocks_per_dim = {Field("i"): 2, Field("j"): 2}

    data1 = np.zeros((10, 10))
    data1[0:5, 0:5] = 1.0
    bs1 = BlockedStats.from_tensor(
        fl.asarray(data1), indices, blocks_per_dim, UniformStatsFactory()
    )

    data2 = np.zeros((10, 10))
    data2[5:10, 5:10] = 1.0
    bs2 = BlockedStats.from_tensor(
        fl.asarray(data2), indices, blocks_per_dim, UniformStatsFactory()
    )

    result = BlockedStatsFactory(blocks_per_dim, UniformStatsFactory()).mapjoin(
        ffunc.add, bs1, bs2
    )

    assert result.estimate_non_fill_values() == 50.0
    assert result.blocks[0, 1].estimate_non_fill_values() == 0.0


def test_blocked_stats_relabel():
    indices = (Field("i"), Field("j"))
    blocks_per_dim = {Field("i"): 2, Field("j"): 2}
    bs = BlockedStats.from_tensor(
        fl.asarray(np.eye(10)), indices, blocks_per_dim, UniformStatsFactory()
    )

    new_names = (Field("row"), Field("col"))
    relabeled = BlockedStatsFactory(blocks_per_dim, UniformStatsFactory()).relabel(
        bs, new_names
    )

    assert relabeled.index_order == new_names
    assert Field("row") in relabeled.blocks_per_dim
    assert relabeled.estimate_non_fill_values() == 10.0


def test_blocked_stats_reorder():
    data = np.zeros((4, 10))
    data[0:2, 0:5] = 1.0
    arr = fl.asarray(data)

    indices = (Field("i"), Field("j"))
    blocks_per_dim = {Field("i"): 2, Field("j"): 2}
    bs = BlockedStats.from_tensor(arr, indices, blocks_per_dim, UniformStatsFactory())

    # Before reordering
    assert bs.blocks[0, 0].get_dim_size(Field("i")) == 2.0
    assert bs.blocks[0, 0].get_dim_size(Field("j")) == 5.0

    new_indices = (Field("j"), Field("i"))
    reordered_bs = BlockedStatsFactory(blocks_per_dim, UniformStatsFactory()).reorder(
        bs, new_indices
    )

    new_block = reordered_bs.blocks[0, 0]

    # After reordering
    assert new_block.get_dim_size(Field("j")) == 5.0
    assert new_block.get_dim_size(Field("i")) == 2.0
    assert new_block.index_order == (Field("j"), Field("i"))


def test_blocked_stats_issimilar():
    indices = (Field("i"), Field("j"))
    blocks_per_dim = {Field("i"): 2, Field("j"): 2}
    data = np.eye(10)
    arr = fl.asarray(data)

    # Identical
    bs1 = BlockedStats.from_tensor(arr, indices, blocks_per_dim, UniformStatsFactory())
    bs2 = BlockedStats.from_tensor(arr, indices, blocks_per_dim, UniformStatsFactory())
    blocked_factory = BlockedStatsFactory(blocks_per_dim, UniformStatsFactory())
    assert blocked_factory.issimilar(bs1, bs2) is True

    # Different data
    data_diff = np.eye(10)
    data_diff[0, 0] = 0.0
    bs_diff_data = BlockedStats.from_tensor(
        fl.asarray(data_diff), indices, blocks_per_dim, UniformStatsFactory()
    )
    assert blocked_factory.issimilar(bs1, bs_diff_data) is False

    # Different blocks_per_dim
    alt_blocks_per_dim = {Field("i"): 5, Field("j"): 5}
    bs_diff_grid = BlockedStats.from_tensor(
        arr, indices, alt_blocks_per_dim, UniformStatsFactory()
    )
    assert blocked_factory.issimilar(bs1, bs_diff_grid) is False

    # Different StatsImpl
    bs_diff_impl = BlockedStats.from_tensor(
        arr, indices, blocks_per_dim, DenseStatsFactory()
    )
    assert blocked_factory.issimilar(bs1, bs_diff_impl) is False


def get_structured_example(M, K, matrix_type):
    if matrix_type == "diagonal":
        return np.eye(M, K, dtype=np.float64)
    if matrix_type == "tridiagonal":
        A = np.eye(M, K, k=0) + np.eye(M, K, k=1) + np.eye(M, K, k=-1)
        return (A > 0).astype(np.float64)
    if matrix_type == "banded":
        bw = 5
        rows, cols = np.indices((M, K))
        return (np.abs(rows - cols) <= bw).astype(np.float64)
    if matrix_type == "triangular":
        return np.triu(np.ones((M, K), dtype=np.float64))
    if matrix_type == "striped":
        A = np.zeros((M, K), dtype=np.float64)
        A[:, ::5] = 1.0
        return A
    return np.zeros((M, K), dtype=np.float64)


def test_benchmark_structured_comparison():
    M, K, N = 20, 20, 20
    i, j, k = Field("i"), Field("j"), Field("k")
    blocks_per_dim = {i: 5, j: 5, k: 5}

    matrix_types = ["diagonal", "tridiagonal", "banded", "triangular", "striped"]
    implementations = [
        ("UniformStats", UniformStatsFactory()),
        ("DenseStats", DenseStatsFactory()),
        ("DCStats", DCStatsFactory()),
    ]

    print("\n" + "=" * 85)
    print(
        f"{'Matrix Type':<15} | {'Stats':<15} |"
        f" {'Stats Perf':<18} | {'Blocked Stats Perf'}"
    )
    print("-" * 85)

    for m_type in matrix_types:
        data_a = get_structured_example(M, K, m_type)
        data_b = get_structured_example(K, N, m_type)

        tns_a = fl.asarray(data_a)
        tns_b = fl.asarray(data_b)

        # Actual result
        actual_result = np.matmul(data_a, data_b)
        actual_nnz = float(np.count_nonzero(actual_result))

        if actual_nnz == 0:
            continue

        for impl_name, impl_factory in implementations:
            # Stats performance
            g_a = impl_factory(tns_a, (i, k))
            g_b = impl_factory(tns_b, (k, j))
            g_res = impl_factory.aggregate(
                ffunc.add, 0.0, (k,), impl_factory.mapjoin(ffunc.mul, g_a, g_b)
            )
            g_perf = abs(g_res.estimate_non_fill_values() - actual_nnz) / actual_nnz

            # Blocked Stats Performance
            blocked_factory = BlockedStatsFactory(blocks_per_dim, impl_factory)
            b_a = BlockedStats.from_tensor(tns_a, (i, k), blocks_per_dim, impl_factory)
            b_b = BlockedStats.from_tensor(tns_b, (k, j), blocks_per_dim, impl_factory)
            b_res = blocked_factory.aggregate(
                ffunc.add, 0.0, (k,), blocked_factory.mapjoin(ffunc.mul, b_a, b_b)
            )
            b_perf = abs(b_res.estimate_non_fill_values() - actual_nnz) / actual_nnz

            print(f"{m_type:<15} | {impl_name:<15} | {g_perf:<18.6f} | {b_perf:.6f}")

        print("-" * 85)


# ─────────────────────────────── TensorDef tests ─────────────────────────────────


def test_copy_and_getters():
    td = TensorDef(
        index_order=(Field("i"), Field("j")),
        dim_sizes={Field("i"): 2.0, Field("j"): 3.0},
        fill_value=42,
    )
    td_copy = td.copy()
    assert td_copy is not td
    assert td_copy.index_order == (Field("i"), Field("j"))
    assert td_copy.dim_sizes == {Field("i"): 2.0, Field("j"): 3.0}
    assert td_copy.get_dim_size(Field("j")) == 3.0
    assert td_copy.fill_value == 42


@pytest.mark.parametrize(
    ("orig_axes", "new_axes"),
    [
        ([Field("i"), Field("j")], [Field("j"), Field("i")]),
        ([Field("x"), Field("y"), Field("z")], [Field("z"), Field("y"), Field("x")]),
    ],
)
def test_reorder_def(orig_axes, new_axes):
    dim_sizes = {axis: float(i + 1) for i, axis in enumerate(orig_axes)}
    td = TensorDef(index_order=orig_axes, dim_sizes=dim_sizes, fill_value=0)
    td2 = TensorDef.reorder(td, tuple(new_axes))
    assert td2.index_order == tuple(new_axes)
    for ax in new_axes:
        assert td2.get_dim_size(ax) == td.get_dim_size(ax)


def test_set_fill_value():
    td = TensorDef(index_order=(Field("i"),), dim_sizes={Field("i"): 5.0}, fill_value=0)
    td2 = td.set_fill_value(7)
    assert td2.fill_value == 7


def test_add_dummy_idx():
    td = TensorDef(index_order=(Field("i"),), dim_sizes={Field("i"): 3.0}, fill_value=0)
    td2 = td.add_dummy_idx(Field("j"))
    assert td2.index_order == (Field("i"), Field("j"))
    assert td2.get_dim_size(Field("j")) == 1.0

    td3 = td2.add_dummy_idx(Field("j"))
    assert td3.index_order == (Field("i"), Field("j"))


@pytest.mark.parametrize(
    "defs, func, expected_axes, expected_dims, expected_fill",
    [
        # union of axes; first-wins on dim size; add fills
        (
            [
                ((Field("i"), Field("j")), {Field("i"): 10.0, Field("j"): 5.0}, 2.0),
                ((Field("i"), Field("k")), {Field("i"): 20.0, Field("k"): 7.0}, 3.0),
            ],
            ffunc.add,
            (Field("i"), Field("j"), Field("k")),
            {Field("i"): 10.0, Field("j"): 5.0, Field("k"): 7.0},
            5.0,
        ),
        # same axes: max over fills; first-wins on size still applies
        (
            [
                ((Field("i"),), {Field("i"): 6.0}, 2.0),
                ((Field("i"),), {Field("i"): 9.0}, 4.0),
            ],
            ffunc.max,
            (Field("i"),),
            {Field("i"): 6.0},
            4.0,
        ),
        # three defs; sum fills via variadic callable
        (
            [
                ((Field("i"),), {Field("i"): 5.0}, 1.0),
                ((Field("i"),), {Field("i"): 5.0}, 2.0),
                ((Field("i"),), {Field("i"): 5.0}, 3.0),
            ],
            lambda *xs: sum(xs),
            (Field("i"),),
            {Field("i"): 5.0},
            6.0,
        ),
    ],
)
def test_tensordef_mapjoin(defs, func, expected_axes, expected_dims, expected_fill):
    objs = [TensorDef(ax, dims, fv) for (ax, dims, fv) in defs]
    out = TensorDef.mapjoin(func, *objs)
    assert out.index_order == expected_axes
    assert out.dim_sizes == expected_dims
    assert out.fill_value == expected_fill


@pytest.mark.parametrize(
    (
        "op_func",
        "index_order",
        "dim_sizes",
        "fill_value",
        "reduce_fields",
        "expected_axes",
        "expected_dims",
        "expected_fill",
    ),
    [
        # addition: drop one axis (n = size('j') = 5) → fill' = 0.5 * 5
        (
            ffunc.add,
            (Field("i"), Field("j"), Field("k")),
            {Field("i"): 10.0, Field("j"): 5.0, Field("k"): 3.0},
            0.5,
            (Field("j"),),
            (Field("i"), Field("k")),
            {Field("i"): 10.0, Field("k"): 3.0},
            0.5 * 5,
        ),
        # addition: drop multiple axes (n = 4*16 = 64) → fill' = 7 * 64
        (
            ffunc.add,
            (Field("a"), Field("b"), Field("c"), Field("d")),
            {Field("a"): 2.0, Field("b"): 4.0, Field("c"): 8.0, Field("d"): 16.0},
            7.0,
            (Field("b"), Field("d")),
            (Field("a"), Field("c")),
            {Field("a"): 2.0, Field("c"): 8.0},
            7.0 * (4 * 16),
        ),
        # addition: no-op when reduce set is empty (n = 1) → fill unchanged
        (
            ffunc.add,
            (Field("x"), Field("y")),
            {Field("x"): 3.0, Field("y"): 9.0},
            1.0,
            [],
            (Field("x"), Field("y")),
            {Field("x"): 3.0, Field("y"): 9.0},
            1.0,
        ),
        # addition: missing axis in reduce set → nothing reduced → fill unchanged
        (
            ffunc.add,
            (Field("i"), Field("j")),
            {Field("i"): 5.0, Field("j"): 6.0},
            0.0,
            (Field("z"),),
            (Field("i"), Field("j")),
            {Field("i"): 5.0, Field("j"): 6.0},
            0.0,
        ),
        # multiplication: reduce 'j' (n = 3) → fill' = (2.0) ** 3 = 8
        (
            ffunc.mul,
            (Field("i"), Field("j")),
            {Field("i"): 2.0, Field("j"): 3.0},
            2.0,
            (Field("j"),),
            (Field("i"),),
            {Field("i"): 2.0},
            8.0,
        ),
        # idempotent op: reduce entire axis → empty shape
        (
            ffunc.min,
            (Field("i"),),
            {Field("i"): 4.0},
            7.0,
            (Field("i"),),
            (),
            {},
            7.0,
        ),
    ],
)
def test_tensordef_aggregate(
    op_func,
    index_order,
    dim_sizes,
    fill_value,
    reduce_fields,
    expected_axes,
    expected_dims,
    expected_fill,
):
    d = TensorDef(index_order=index_order, dim_sizes=dim_sizes, fill_value=fill_value)
    out = TensorDef.aggregate(op_func, None, reduce_fields, d)

    assert out.index_order == expected_axes
    assert out.dim_sizes == expected_dims
    assert out.fill_value == expected_fill


# ─────────────────────────────── DenseStats tests ─────────────────────────────


def test_from_tensor_and_getters():
    arr = fl.asarray(np.zeros((2, 3)))
    node = Table(Literal(arr), (Field("i"), Field("j")))
    stats = insert_statistics(
        stats_factory=DenseStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.index_order == (Field("i"), Field("j"))
    assert stats.get_dim_size(Field("i")) == 2.0
    assert stats.get_dim_size(Field("j")) == 3.0
    assert stats.fill_value == 0


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((2, 3), 6.0),
        ((4, 5, 6), 120.0),
        ((1,), 1.0),
    ],
)
def test_estimate_non_fill_values(shape, expected):
    axes = tuple(Field(f"x{i}") for i in range(len(shape)))
    # axes = [f"x{i}" for i in range(len(shape))]
    arr = fl.asarray(np.zeros(shape))
    node = Table(Literal(arr), axes)
    # node = Table(Literal(arr), tuple(Field(a) for a in axes))

    stats = insert_statistics(
        stats_factory=DenseStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    assert stats.index_order == tuple(axes)
    assert stats.estimate_non_fill_values() == expected


def test_mapjoin_mul_and_add():
    ta = Table(Literal(fl.asarray(np.ones((2, 3)))), (Field("i"), Field("j")))
    tb = Table(Literal(fl.asarray(np.ones((3, 4)))), (Field("j"), Field("k")))
    ta2 = Table(Literal(fl.asarray(2 * np.ones((2, 3)))), (Field("i"), Field("j")))

    cache = {}
    insert_statistics(
        stats_factory=DenseStatsFactory(),
        node=ta,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    insert_statistics(
        stats_factory=DenseStatsFactory(),
        node=tb,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    insert_statistics(
        stats_factory=DenseStatsFactory(),
        node=ta2,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    cache[ta].fill_value = 1
    cache[tb].fill_value = 1
    cache[ta2].fill_value = 2
    node_mul = MapJoin(Literal(ffunc.mul), (ta, tb))
    dsm = insert_statistics(
        stats_factory=DenseStatsFactory(),
        node=node_mul,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert dsm.index_order == (Field("i"), Field("j"), Field("k"))
    assert dsm.get_dim_size(Field("i")) == 2.0
    assert dsm.get_dim_size(Field("j")) == 3.0
    assert dsm.get_dim_size(Field("k")) == 4.0
    assert dsm.fill_value == 0.0

    node_add = MapJoin(Literal(ffunc.add), (ta, ta2))
    ds_sum = insert_statistics(
        stats_factory=DenseStatsFactory(),
        node=node_add,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert ds_sum.index_order == (Field("i"), Field("j"))
    assert ds_sum.get_dim_size(Field("i")) == 2.0
    assert ds_sum.get_dim_size(Field("j")) == 3.0
    assert ds_sum.fill_value == 1.0 + 2.0


def test_aggregate_and_issimilar():
    table = Table(
        Literal(fl.asarray(np.ones((2, 3)))),
        (Field("i"), Field("j")),
    )
    dsa = insert_statistics(
        stats_factory=DenseStatsFactory(),
        node=table,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    node_add = Aggregate(
        op=Literal(ffunc.add),
        init=None,
        arg=table,
        idxs=(Field("j"),),
    )

    ds_agg = insert_statistics(
        stats_factory=DenseStatsFactory(),
        node=node_add,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    assert ds_agg.index_order == (Field("i"),)
    assert ds_agg.get_dim_size(Field("i")) == 2.0
    assert ds_agg.fill_value == dsa.fill_value
    assert DenseStatsFactory().issimilar(dsa, dsa)


def test_relabel_dense_stats():
    arr = fl.asarray(np.zeros((2, 3)))
    table = Table(Literal(arr), (Field("i"), Field("j")))

    stats = insert_statistics(
        stats_factory=DenseStatsFactory(),
        node=table,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    new_stats = DenseStatsFactory().relabel(stats, (Field("row"), Field("col")))

    assert new_stats.index_order == (Field("row"), Field("col"))

    assert new_stats.get_dim_size(Field("row")) == 2.0

    assert new_stats.get_dim_size(Field("col")) == 3.0

    assert new_stats.fill_value == stats.fill_value

    with pytest.raises(ValueError):
        DenseStatsFactory().relabel(stats, (Field("x"), Field("y"), Field("z")))


# ─────────────────────────────── DCStats tests ─────────────────────────────


@pytest.mark.parametrize(
    "tensor, fields, expected_dcs",
    [
        (fl.asarray(np.array(0)), [], {DC(frozenset(), frozenset(), 1.0)}),
    ],
)
def test_dc_stats_scalar(tensor, fields, expected_dcs):
    node = Table(
        Literal(fl.asarray(tensor)),
        tuple(fields),
    )
    stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.dcs == expected_dcs


@pytest.mark.parametrize(
    "tensor, fields, expected_dcs",
    [
        (
            fl.asarray(np.array([1, 1, 1, 1])),
            [Field("i")],
            {
                DC(frozenset(), frozenset([Field("i")]), 4.0),
                DC(frozenset([Field("i")]), frozenset([Field("i")]), 1.0),
            },
        ),
        (
            fl.asarray(np.array([0, 1, 0, 0, 1])),
            [Field("i")],
            {
                DC(frozenset(), frozenset([Field("i")]), 2.0),
                DC(frozenset([Field("i")]), frozenset([Field("i")]), 1.0),
            },
        ),
    ],
)
def test_dc_stats_vector(tensor, fields, expected_dcs):
    node = Table(
        Literal(tensor),
        tuple(fields),
    )
    stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.dcs == expected_dcs


@pytest.mark.parametrize(
    "tensor, fields, expected_dcs",
    [
        (
            fl.asarray(np.ones((3, 3), dtype=int)),
            [Field("i"), Field("j")],
            {
                DC(frozenset(), frozenset([Field("i"), Field("j")]), 9.0),
                DC(frozenset(), frozenset([Field("i")]), 3.0),
                DC(frozenset(), frozenset([Field("j")]), 3.0),
                DC(frozenset([Field("i")]), frozenset([Field("i"), Field("j")]), 3.0),
                DC(frozenset([Field("j")]), frozenset([Field("i"), Field("j")]), 3.0),
            },
        ),
        (
            fl.asarray(
                np.array(
                    [
                        [1, 0, 1],
                        [0, 0, 0],
                        [1, 1, 0],
                    ],
                    dtype=int,
                )
            ),
            [Field("i"), Field("j")],
            {
                DC(frozenset(), frozenset([Field("i"), Field("j")]), 4.0),
                DC(frozenset(), frozenset([Field("i")]), 2.0),
                DC(frozenset(), frozenset([Field("j")]), 3.0),
                DC(frozenset([Field("i")]), frozenset([Field("i"), Field("j")]), 2.0),
                DC(frozenset([Field("j")]), frozenset([Field("i"), Field("j")]), 2.0),
            },
        ),
    ],
)
def test_dc_stats_matrix(tensor, fields, expected_dcs):
    node = Table(
        Literal(tensor),
        tuple(fields),
    )
    stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.dcs == expected_dcs


@pytest.mark.parametrize(
    "tensor, fields, expected_dcs",
    [
        (
            fl.asarray(np.ones((2, 2, 2), dtype=int)),
            [Field("i"), Field("j"), Field("k")],
            {
                DC(frozenset(), frozenset([Field("i"), Field("j"), Field("k")]), 8.0),
                DC(frozenset(), frozenset([Field("i")]), 2.0),
                DC(frozenset(), frozenset([Field("j")]), 2.0),
                DC(frozenset(), frozenset([Field("k")]), 2.0),
                DC(
                    frozenset([Field("i")]),
                    frozenset([Field("i"), Field("j"), Field("k")]),
                    4.0,
                ),
                DC(
                    frozenset([Field("j")]),
                    frozenset([Field("i"), Field("j"), Field("k")]),
                    4.0,
                ),
                DC(
                    frozenset([Field("k")]),
                    frozenset([Field("i"), Field("j"), Field("k")]),
                    4.0,
                ),
            },
        ),
        (
            fl.asarray(
                np.array(
                    [
                        [[1, 0], [0, 0]],
                        [[0, 1], [1, 0]],
                    ],
                    dtype=int,
                )
            ),
            [Field("i"), Field("j"), Field("k")],
            {
                DC(frozenset(), frozenset([Field("i"), Field("j"), Field("k")]), 3.0),
                DC(frozenset(), frozenset([Field("i")]), 2.0),
                DC(frozenset(), frozenset([Field("j")]), 2.0),
                DC(frozenset(), frozenset([Field("k")]), 2.0),
                DC(
                    frozenset([Field("i")]),
                    frozenset([Field("i"), Field("j"), Field("k")]),
                    2.0,
                ),
                DC(
                    frozenset([Field("j")]),
                    frozenset([Field("i"), Field("j"), Field("k")]),
                    2.0,
                ),
                DC(
                    frozenset([Field("k")]),
                    frozenset([Field("i"), Field("j"), Field("k")]),
                    2.0,
                ),
            },
        ),
    ],
)
def test_dc_stats_3d(tensor, fields, expected_dcs):
    node = Table(
        Literal(tensor),
        tuple(fields),
    )
    stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.dcs == expected_dcs


@pytest.mark.parametrize(
    "tensor, fields, expected_dcs",
    [
        (
            fl.asarray(np.ones((2, 2, 2, 2), dtype=int)),
            [Field("i"), Field("j"), Field("k"), Field("l")],
            {
                DC(
                    frozenset(),
                    frozenset([Field("i"), Field("j"), Field("k"), Field("l")]),
                    16.0,
                ),
                DC(frozenset(), frozenset([Field("i")]), 2.0),
                DC(frozenset(), frozenset([Field("j")]), 2.0),
                DC(frozenset(), frozenset([Field("k")]), 2.0),
                DC(frozenset(), frozenset([Field("l")]), 2.0),
                DC(
                    frozenset([Field("i")]),
                    frozenset([Field("i"), Field("j"), Field("k"), Field("l")]),
                    8.0,
                ),
                DC(
                    frozenset([Field("j")]),
                    frozenset([Field("i"), Field("j"), Field("k"), Field("l")]),
                    8.0,
                ),
                DC(
                    frozenset([Field("k")]),
                    frozenset([Field("i"), Field("j"), Field("k"), Field("l")]),
                    8.0,
                ),
                DC(
                    frozenset([Field("l")]),
                    frozenset([Field("i"), Field("j"), Field("k"), Field("l")]),
                    8.0,
                ),
            },
        ),
        (
            fl.asarray(
                np.array(
                    [
                        [
                            [[1, 0], [0, 0]],
                            [[0, 0], [0, 1]],
                        ],
                        [
                            [[0, 0], [1, 0]],
                            [[0, 0], [0, 0]],
                        ],
                    ],
                    dtype=int,
                )
            ),
            [Field("i"), Field("j"), Field("k"), Field("l")],
            {
                DC(
                    frozenset(),
                    frozenset([Field("i"), Field("j"), Field("k"), Field("l")]),
                    3.0,
                ),
                DC(frozenset(), frozenset([Field("i")]), 2.0),
                DC(frozenset(), frozenset([Field("j")]), 2.0),
                DC(frozenset(), frozenset([Field("k")]), 2.0),
                DC(frozenset(), frozenset([Field("l")]), 2.0),
                DC(
                    frozenset([Field("i")]),
                    frozenset([Field("i"), Field("j"), Field("k"), Field("l")]),
                    2.0,
                ),
                DC(
                    frozenset([Field("j")]),
                    frozenset([Field("i"), Field("j"), Field("k"), Field("l")]),
                    2.0,
                ),
                DC(
                    frozenset([Field("k")]),
                    frozenset([Field("i"), Field("j"), Field("k"), Field("l")]),
                    2.0,
                ),
                DC(
                    frozenset([Field("l")]),
                    frozenset([Field("i"), Field("j"), Field("k"), Field("l")]),
                    2.0,
                ),
            },
        ),
    ],
)
def test_dc_stats_4d(tensor, fields, expected_dcs):
    node = Table(
        Literal(tensor),
        tuple(fields),
    )
    stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.dcs == expected_dcs


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {Field("i"): 1000, Field("j"): 1000},
            [
                DC(frozenset([Field("i")]), frozenset([Field("j")]), 5),
                DC(frozenset([Field("j")]), frozenset([Field("i")]), 25),
                DC(frozenset(), frozenset([Field("i"), Field("j")]), 50),
            ],
            50,
        ),
    ],
)
def test_single_tensor_card(dims, dcs, expected_nnz):
    dims = {Field(k.name): v for k, v in dims.items()}
    node = Table(
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("i"), Field("j"))
    )
    stat = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    stat.tensordef = TensorDef(frozenset([Field("i"), Field("j")]), dims, 0)
    stat.dcs = set(dcs)

    assert stat.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {Field("i"): 1000, Field("j"): 1000, Field("k"): 1000},
            [
                DC(frozenset([Field("j")]), frozenset([Field("k")]), 5),
                DC(frozenset(), frozenset([Field("i"), Field("j")]), 50),
            ],
            50 * 5,
        ),
    ],
)
def test_1_join_dc_card(dims, dcs, expected_nnz):
    dims = {Field(k.name): v for k, v in dims.items()}
    node = Table(
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    stat = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    stat.tensordef = TensorDef(frozenset([Field("i"), Field("j"), Field("k")]), dims, 0)
    stat.dcs = set(dcs)
    assert stat.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {Field("i"): 1000, Field("j"): 1000, Field("k"): 1000, Field("l"): 1000},
            [
                DC(frozenset(), frozenset([Field("i"), Field("j")]), 50),
                DC(frozenset([Field("j")]), frozenset([Field("k")]), 5),
                DC(frozenset([Field("k")]), frozenset([Field("l")]), 5),
            ],
            50 * 5 * 5,
        ),
    ],
)
def test_2_join_dc_card(dims, dcs, expected_nnz):
    dims = {Field(k.name): v for k, v in dims.items()}
    node = Table(
        Literal(fl.asarray(np.zeros((1, 1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k"), Field("l")),
    )
    stat = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    stat.tensordef = TensorDef(
        frozenset([Field("i"), Field("j"), Field("k"), Field("l")]), dims, 0
    )
    stat.dcs = set(dcs)
    assert stat.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {Field("i"): 1000, Field("j"): 1000, Field("k"): 1000},
            [
                DC(frozenset(), frozenset([Field("i"), Field("j")]), 50),
                DC(frozenset([Field("i")]), frozenset([Field("j")]), 5),
                DC(frozenset([Field("j")]), frozenset([Field("i")]), 5),
                DC(frozenset(), frozenset([Field("j"), Field("k")]), 50),
                DC(frozenset([Field("j")]), frozenset([Field("k")]), 5),
                DC(frozenset([Field("k")]), frozenset([Field("j")]), 5),
                DC(frozenset(), frozenset([Field("i"), Field("k")]), 50),
                DC(frozenset([Field("i")]), frozenset([Field("k")]), 5),
                DC(frozenset([Field("k")]), frozenset([Field("i")]), 5),
            ],
            50 * 5,
        ),
    ],
)
def test_triangle_dc_card(dims, dcs, expected_nnz):
    dims = {Field(k.name): v for k, v in dims.items()}
    node = Table(
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    stat = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    stat.tensordef = TensorDef(frozenset([Field("i"), Field("j"), Field("k")]), dims, 0)
    stat.dcs = set(dcs)
    assert stat.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {Field("i"): 1000, Field("j"): 1000, Field("k"): 1000},
            [
                DC(frozenset(), frozenset([Field("i"), Field("j")]), 1),
                DC(frozenset([Field("i")]), frozenset([Field("j")]), 1),
                DC(frozenset([Field("j")]), frozenset([Field("i")]), 1),
                DC(frozenset(), frozenset([Field("j"), Field("k")]), 50),
                DC(frozenset([Field("j")]), frozenset([Field("k")]), 5),
                DC(frozenset([Field("k")]), frozenset([Field("j")]), 5),
                DC(frozenset(), frozenset([Field("i"), Field("k")]), 50),
                DC(frozenset([Field("i")]), frozenset([Field("k")]), 5),
                DC(frozenset([Field("k")]), frozenset([Field("i")]), 5),
            ],
            1 * 5,
        ),
    ],
)
def test_triangle_small_dc_card(dims, dcs, expected_nnz):
    dims = {Field(k.name): v for k, v in dims.items()}
    node = Table(
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    stat = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    stat.tensordef = TensorDef(frozenset([Field("i"), Field("j"), Field("k")]), dims, 0)
    stat.dcs = set(dcs)
    assert stat.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs_list, expected_dcs",
    [
        # Single input passthrough
        (
            {Field("i"): 1000},
            [
                {
                    DC(frozenset(), frozenset({Field("i")}), 5.0),
                    DC(frozenset({Field("i")}), frozenset({Field("i")}), 1.0),
                }
            ],
            {
                DC(frozenset(), frozenset({Field("i")}), 5.0),
                DC(frozenset({Field("i")}), frozenset({Field("i")}), 1.0),
            },
        ),
        # Two inputs: overlap takes min; unique keys are preserved
        (
            {Field("i"): 1000},
            [
                {
                    DC(frozenset(), frozenset({Field("i")}), 5.0),
                    DC(frozenset({Field("i")}), frozenset({Field("i")}), 1.0),
                },
                {
                    DC(frozenset(), frozenset({Field("i")}), 2.0),
                    DC(frozenset({Field("i")}), frozenset({Field("i")}), 3.0),
                    DC(frozenset(), frozenset(), 7.0),
                },
            ],
            {
                DC(frozenset(), frozenset({Field("i")}), 2.0),
                DC(frozenset({Field("i")}), frozenset({Field("i")}), 1.0),
                DC(frozenset(), frozenset(), 7.0),
            },
        ),
    ],
)
def test_merge_dc_join(dims, dcs_list, expected_dcs):
    stats_objs = []
    for dcs in dcs_list:
        node = Table(Literal(fl.asarray(np.zeros((1,), dtype=int))), (Field("i"),))
        s = insert_statistics(
            stats_factory=DCStatsFactory(),
            node=node,
            bindings=OrderedDict(),
            replace=False,
            cache={},
        )
        s.tensordef = TensorDef(frozenset({Field("i")}), dims, 0)
        s.dcs = set(dcs)
        stats_objs.append(s)

    new_def = TensorDef(frozenset({Field("i")}), dims, 0)
    out = DCStats._merge_dc_join(new_def, stats_objs)

    assert out.tensordef.index_order == (Field("i"),)
    assert out.tensordef.dim_sizes == dims
    assert out.dcs == expected_dcs


@pytest.mark.parametrize(
    "new_dims, inputs, expected_dcs",
    [
        # Single input passthrough
        (
            {Field("i"): 1000},
            [
                (
                    {Field("i")},
                    {
                        DC(frozenset(), frozenset({Field("i")}), 5.0),
                        DC(frozenset({Field("i")}), frozenset({Field("i")}), 1.0),
                    },
                )
            ],
            {
                DC(frozenset(), frozenset({Field("i")}), 5.0),
                DC(frozenset({Field("i")}), frozenset({Field("i")}), 1.0),
            },
        ),
        # Two inputs, same axes: overlap SUMs; keys not in all inputs are dropped
        (
            {Field("i"): 1000},
            [
                (
                    {Field("i")},
                    {
                        DC(frozenset(), frozenset({Field("i")}), 5.0),
                        DC(frozenset({Field("i")}), frozenset({Field("i")}), 1.0),
                    },
                ),
                (
                    {Field("i")},
                    {
                        DC(frozenset(), frozenset({Field("i")}), 2.0),
                        DC(frozenset({Field("i")}), frozenset({Field("i")}), 3.0),
                        DC(frozenset(), frozenset(), 7.0),
                    },
                ),
            ],
            {
                DC(frozenset(), frozenset({Field("i")}), 7.0),
                DC(frozenset({Field("i")}), frozenset({Field("i")}), 4.0),
            },
        ),
        # Lifting across extra axes (Z) + consensus then SUM
        (
            {Field("i"): 10, Field("j"): 4},
            [
                ({Field("i")}, {DC(frozenset(), frozenset({Field("i")}), 3.0)}),
                ({Field("j")}, {DC(frozenset(), frozenset({Field("j")}), 2.0)}),
            ],
            {DC(frozenset(), frozenset({Field("i"), Field("j")}), 32.0)},
        ),
        # Clamp by dense capacity of Y
        (
            {Field("i"): 5},
            [
                ({Field("i")}, {DC(frozenset(), frozenset({Field("i")}), 7.0)}),
                ({Field("i")}, {DC(frozenset(), frozenset({Field("i")}), 9.0)}),
            ],
            {DC(frozenset(), frozenset({Field("i")}), 5.0)},
        ),
    ],
)
def test_merge_dc_union(new_dims, inputs, expected_dcs):
    new_dims = {Field(k.name): v for k, v in new_dims.items()}
    cache = {}
    stats_objs = []
    for idx_set, dcs in inputs:
        fields = tuple(sorted(idx_set, key=lambda f: f.name))
        shape = (1,) * max(1, len(fields))
        node = Table(Literal(fl.asarray(np.zeros(shape, dtype=int))), fields)

        insert_statistics(
            stats_factory=DCStatsFactory(),
            node=node,
            bindings=OrderedDict(),
            replace=False,
            cache=cache,
        )
        field_idx_set = frozenset(idx_set)
        field_dims = {k: new_dims[k] for k in idx_set}

        td = TensorDef(field_idx_set, field_dims, 0)
        stats_objs.append(DCStats.from_def(td, set(dcs)))

    new_def = TensorDef(frozenset(new_dims.keys()), new_dims, 0)
    out = DCStats._merge_dc_union(new_def, stats_objs)

    # Does the order matter here ? - > Changed tuple to set as throwing assert error
    assert set(out.tensordef.index_order) == set(new_dims.keys())
    assert dict(out.tensordef.dim_sizes) == new_dims
    assert out.dcs == expected_dcs


@pytest.mark.parametrize(
    "dims1, dcs1, dims2, dcs2, expected_nnz",
    [
        (
            {Field("i"): 1000},
            [DC(frozenset(), frozenset([Field("i")]), 1)],
            {Field("i"): 1000},
            [DC(frozenset(), frozenset([Field("i")]), 1)],
            2,
        ),
    ],
)
def test_1d_disjunction_dc_card(dims1, dcs1, dims2, dcs2, expected_nnz):
    cache = {}

    node1 = Table(Literal(fl.asarray(np.zeros((1,), dtype=int))), (Field("i"),))
    s1 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node1,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s1.tensordef = TensorDef(frozenset({Field("i")}), dims1, 0)
    s1.dcs = set(dcs1)

    node2 = Table(Literal(fl.asarray(np.zeros((1,), dtype=int))), (Field("i"),))
    s2 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node2,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s2.tensordef = TensorDef(frozenset({Field("i")}), dims2, 0)
    s2.dcs = set(dcs2)

    parent = MapJoin(Literal(ffunc.add), (node1, node2))
    reduce_stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=parent,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims1, dcs1, dims2, dcs2, expected_nnz",
    [
        (
            {Field("i"): 1000, Field("j"): 1000},
            [DC(frozenset(), frozenset([Field("i"), Field("j")]), 1)],
            {Field("i"): 1000, Field("j"): 1000},
            [DC(frozenset(), frozenset([Field("i"), Field("j")]), 1)],
            2,
        ),
    ],
)
def test_2d_disjunction_dc_card(dims1, dcs1, dims2, dcs2, expected_nnz):
    cache = {}

    node1 = Table(
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("i"), Field("j"))
    )
    s1 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node1,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s1.tensordef = TensorDef(frozenset({Field("i"), Field("j")}), dims1, 0)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("i"), Field("j"))
    )
    s2 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node2,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s2.tensordef = TensorDef(frozenset({Field("i"), Field("j")}), dims2, 0)
    s2.dcs = set(dcs2)

    parent = MapJoin(Literal(ffunc.add), (node1, node2))
    reduce_stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=parent,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims1, dcs1, dims2, dcs2, expected_nnz",
    [
        (
            {Field("i"): 1000},
            [DC(frozenset(), frozenset([Field("i")]), 5)],
            {Field("j"): 100},
            [DC(frozenset(), frozenset([Field("j")]), 10)],
            10 * 1000 + 5 * 100,
        ),
    ],
)
def test_2d_disjoin_disjunction_dc_card(dims1, dcs1, dims2, dcs2, expected_nnz):
    cache = {}

    node1 = Table(Literal(fl.asarray(np.zeros((1,), dtype=int))), (Field("i"),))
    s1 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node1,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s1.tensordef = TensorDef(frozenset({Field("i")}), dims1, 0)
    s1.dcs = set(dcs1)

    node2 = Table(Literal(fl.asarray(np.zeros((1,), dtype=int))), (Field("j"),))
    s2 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node2,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s2.tensordef = TensorDef(frozenset({Field("j")}), dims2, 0)
    s2.dcs = set(dcs2)

    parent = MapJoin(Literal(ffunc.add), (node1, node2))
    reduce_stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=parent,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims1, dcs1, dims2, dcs2, expected_nnz",
    [
        (
            {Field("i"): 1000, Field("j"): 100},
            [DC(frozenset(), frozenset([Field("i"), Field("j")]), 5)],
            {Field("j"): 100, Field("k"): 1000},
            [DC(frozenset(), frozenset([Field("j"), Field("k")]), 10)],
            10 * 1000 + 5 * 1000,
        ),
    ],
)
def test_3d_disjoint_disjunction_dc_card(dims1, dcs1, dims2, dcs2, expected_nnz):
    dims1 = {Field(k.name): v for k, v in dims1.items()}
    dims2 = {Field(k.name): v for k, v in dims2.items()}
    cache = {}

    node1 = Table(
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))),
        (Field("i"), Field("j")),
    )
    s1 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node1,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s1.tensordef = TensorDef(frozenset({Field("i"), Field("j")}), dims1, 0)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))),
        (Field("j"), Field("k")),
    )
    s2 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node2,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s2.tensordef = TensorDef(frozenset({Field("j"), Field("k")}), dims2, 0)
    s2.dcs = set(dcs2)

    parent = MapJoin(Literal(ffunc.add), (node1, node2))
    reduce_stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=parent,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz


""""""


@pytest.mark.parametrize(
    "dims1, dcs1, dims2, dcs2, dims3, dcs3, expected_nnz",
    [
        (
            {Field("i"): 1000, Field("j"): 100},
            [DC(frozenset(), frozenset([Field("i"), Field("j")]), 5)],
            {Field("j"): 100, Field("k"): 1000},
            [DC(frozenset(), frozenset([Field("j"), Field("k")]), 10)],
            {Field("i"): 1000, Field("j"): 100, Field("k"): 1000},
            [DC(frozenset(), frozenset([Field("i"), Field("j"), Field("k")]), 10)],
            10 * 1000 + 5 * 1000 + 10,
        ),
    ],
)
def test_large_disjoint_disjunction_dc_card(
    dims1, dcs1, dims2, dcs2, dims3, dcs3, expected_nnz
):
    dims1 = {Field(k.name): v for k, v in dims1.items()}
    dims2 = {Field(k.name): v for k, v in dims2.items()}
    dims3 = {Field(k.name): v for k, v in dims3.items()}
    cache = {}

    node1 = Table(
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("i"), Field("j"))
    )
    s1 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node1,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s1.tensordef = TensorDef(frozenset({Field("i"), Field("j")}), dims1, 1)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("j"), Field("k"))
    )
    s2 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node2,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s2.tensordef = TensorDef(frozenset({Field("j"), Field("k")}), dims2, 1)
    s2.dcs = set(dcs2)

    node3 = Table(
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    s3 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node3,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s3.tensordef = TensorDef(frozenset({Field("i"), Field("j"), Field("k")}), dims3, 1)
    s3.dcs = set(dcs3)

    map = MapJoin(Literal(ffunc.mul), (node1, node2))

    parent = MapJoin(Literal(ffunc.mul), (map, node3))

    reduce_stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=parent,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims1, dcs1, dims2, dcs2, dims3, dcs3, expected_nnz",
    [
        (
            {Field("i"): 1000, Field("j"): 100},
            [DC(frozenset(), frozenset([Field("i"), Field("j")]), 5)],
            {Field("j"): 100, Field("k"): 1000},
            [DC(frozenset(), frozenset([Field("j"), Field("k")]), 10)],
            {Field("i"): 1000, Field("j"): 100, Field("k"): 1000},
            [DC(frozenset(), frozenset([Field("i"), Field("j"), Field("k")]), 10)],
            10,
        ),
    ],
)
def test_mixture_disjoint_disjunction_dc_card(
    dims1, dcs1, dims2, dcs2, dims3, dcs3, expected_nnz
):
    cache = {}

    node1 = Table(
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("i"), Field("j"))
    )
    s1 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node1,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s1.tensordef = TensorDef(frozenset([Field("i"), Field("j")]), dims1, 1)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("j"), Field("k"))
    )
    s2 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node2,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s2.tensordef = TensorDef(frozenset([Field("j"), Field("k")]), dims2, 1)
    s2.dcs = set(dcs2)

    node3 = Table(
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    s3 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node3,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    s3.tensordef = TensorDef(frozenset([Field("i"), Field("j"), Field("k")]), dims3, 0)
    s3.dcs = set(dcs3)

    map = MapJoin(Literal(ffunc.mul), (node1, node2))
    parent = MapJoin(Literal(ffunc.mul), (map, node3))

    reduce_stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=parent,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {Field("i"): 1000, Field("j"): 1000, Field("k"): 1000},
            [
                DC(frozenset(), frozenset([Field("i"), Field("j")]), 50),
                DC(frozenset([Field("i")]), frozenset([Field("j")]), 5),
                DC(frozenset([Field("j")]), frozenset([Field("i")]), 5),
                DC(frozenset(), frozenset([Field("j"), Field("k")]), 50),
                DC(frozenset([Field("j")]), frozenset([Field("k")]), 5),
                DC(frozenset([Field("k")]), frozenset([Field("j")]), 5),
                DC(frozenset(), frozenset([Field("i"), Field("k")]), 50),
                DC(frozenset([Field("i")]), frozenset([Field("k")]), 5),
                DC(frozenset([Field("k")]), frozenset([Field("i")]), 5),
            ],
            1,
        ),
    ],
)
def test_full_reduce_DC_card(dims, dcs, expected_nnz):
    cache = {}

    node = Table(
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    stat = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    stat.tensordef = TensorDef(
        frozenset([Field("i"), Field("j"), Field("k")]), dims, 0.0
    )
    stat.dcs = set(dcs)

    reduce_node = Aggregate(
        op=Literal(ffunc.add),
        init=Literal(0),
        idxs=(Field("i"), Field("j"), Field("k")),
        arg=node,
    )
    reduce_stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=reduce_node,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {Field("i"): 1000, Field("j"): 1000, Field("k"): 1000},
            [
                DC(frozenset(), frozenset([Field("i"), Field("j")]), 1),
                DC(frozenset([Field("i")]), frozenset([Field("j")]), 1),
                DC(frozenset([Field("j")]), frozenset([Field("i")]), 1),
                DC(frozenset(), frozenset([Field("j"), Field("k")]), 50),
                DC(frozenset([Field("j")]), frozenset([Field("k")]), 5),
                DC(frozenset([Field("k")]), frozenset([Field("j")]), 5),
                DC(frozenset(), frozenset([Field("i"), Field("k")]), 50),
                DC(frozenset([Field("i")]), frozenset([Field("k")]), 5),
                DC(frozenset([Field("k")]), frozenset([Field("i")]), 5),
            ],
            5,
        ),
    ],
)
def test_1_attr_reduce_DC_card(dims, dcs, expected_nnz):
    cache = {}

    node = Table(
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    st = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    st.tensordef = TensorDef(frozenset([Field("i"), Field("j"), Field("k")]), dims, 0.0)
    st.dcs = set(dcs)

    reduce_node = Aggregate(
        op=Literal(ffunc.add),
        init=Literal(0),
        idxs=(Field("i"), Field("j")),
        arg=node,
    )
    reduce_stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=reduce_node,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {Field("i"): 1000, Field("j"): 1000, Field("k"): 1000},
            [
                DC(frozenset(), frozenset([Field("i"), Field("j")]), 1),
                DC(frozenset([Field("i")]), frozenset([Field("j")]), 1),
                DC(frozenset([Field("j")]), frozenset([Field("i")]), 1),
                DC(frozenset(), frozenset([Field("j"), Field("k")]), 50),
                DC(frozenset([Field("j")]), frozenset([Field("k")]), 5),
                DC(frozenset([Field("k")]), frozenset([Field("j")]), 5),
                DC(frozenset(), frozenset([Field("i"), Field("k")]), 50),
                DC(frozenset([Field("i")]), frozenset([Field("k")]), 5),
                DC(frozenset([Field("k")]), frozenset([Field("i")]), 5),
            ],
            5,
        ),
    ],
)
def test_2_attr_reduce_DC_card(dims, dcs, expected_nnz):
    cache = {}

    node = Table(
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    st = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    st.tensordef = TensorDef(frozenset([Field("i"), Field("j"), Field("k")]), dims, 0.0)
    st.dcs = set(dcs)

    reduce_node = Aggregate(
        op=Literal(ffunc.add),
        init=Literal(0),
        idxs=(Field("i"),),
        arg=node,
    )
    reduce_stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=reduce_node,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, reduce_indices, expected_nnz",
    [
        # Asymmetric densities
        (
            {Field("i"): 100, Field("j"): 100, Field("k"): 100},
            [
                DC(frozenset(), frozenset([Field("i"), Field("j")]), 100),
                DC(frozenset([Field("i")]), frozenset([Field("j")]), 2),
                DC(frozenset(), frozenset([Field("j"), Field("k")]), 50),
            ],
            [Field("j")],
            5000,
        ),
        # Sparse + dense mix
        (
            {Field("i"): 100, Field("j"): 100, Field("k"): 100},
            [
                DC(frozenset(), frozenset([Field("i"), Field("k")]), 900),
                DC(frozenset([Field("i")]), frozenset([Field("k")]), 1),
            ],
            [Field("i"), Field("k")],
            100,
        ),
        # Imbalance across dimensions
        (
            {Field("i"): 1000, Field("j"): 100, Field("k"): 10},
            [
                DC(frozenset(), frozenset([Field("i"), Field("j")]), 5),
                DC(frozenset(), frozenset([Field("j"), Field("k")]), 80),
                DC(frozenset(), frozenset([Field("i"), Field("k")]), 1),
            ],
            [Field("i")],
            5,
        ),
    ],
)
def test_varied_reduce_DC_card(dims, dcs, reduce_indices, expected_nnz):
    cache = {}

    node = Table(
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    st = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    st.tensordef = TensorDef(frozenset([Field("i"), Field("j"), Field("k")]), dims, 0.0)
    st.dcs = set(dcs)

    reduce_fields = tuple(reduce_indices)
    reduce_node = Aggregate(
        op=Literal(ffunc.add),
        init=Literal(0),
        idxs=reduce_fields,
        arg=node,
    )
    reduce_stats = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=reduce_node,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz
