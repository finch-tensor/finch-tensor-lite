import operator as op
from collections import OrderedDict

import pytest

import numpy as np

from finchlite.finch_logic import Aggregate, Field, Literal, Table, MapJoin
from finchlite.galley.LogicalOptimizer.logic_to_stats import _insert_statistics
from finchlite.galley.TensorStats.dc_stats import DC, DCStats
from finchlite.galley.TensorStats.dense_stat import DenseStats
from finchlite.galley.TensorStats.tensor_def import TensorDef
from finchlite.galley.TensorStats.tensor_stats import TensorStats

# ─────────────────────────────── TensorDef tests ─────────────────────────────────


def test_copy_and_getters():
    td = TensorDef(index_set=["i", "j"], dim_sizes={"i": 2.0, "j": 3.0}, fill_value=42)
    td_copy = td.copy()
    assert td_copy is not td
    assert td_copy.index_set == {"i", "j"}
    assert td_copy.dim_sizes == {"i": 2.0, "j": 3.0}
    assert td_copy.get_dim_size("j") == 3.0
    assert td_copy.fill_value == 42


@pytest.mark.parametrize(
    ("orig_axes", "new_axes"),
    [
        (["i", "j"], ["j", "i"]),
        (["x", "y", "z"], ["z", "y", "x"]),
    ],
)
def test_reindex_def(orig_axes, new_axes):
    dim_sizes = {axis: float(i + 1) for i, axis in enumerate(orig_axes)}
    td = TensorDef(index_set=orig_axes, dim_sizes=dim_sizes, fill_value=0)
    td2 = td.reindex_def(new_axes)
    assert td2.index_set == set(new_axes)
    for ax in new_axes:
        assert td2.get_dim_size(ax) == td.get_dim_size(ax)


def test_set_fill_value_and_relabel_index():
    td = TensorDef(index_set=["i"], dim_sizes={"i": 5.0}, fill_value=0)
    td2 = td.set_fill_value(7)
    assert td2.fill_value == 7

    td3 = td2.relabel_index("i", "k")
    assert "k" in td3.index_set and "i" not in td3.index_set
    assert td3.get_dim_size("k") == 5.0


def test_add_dummy_idx():
    td = TensorDef(index_set=["i"], dim_sizes={"i": 3.0}, fill_value=0)
    td2 = td.add_dummy_idx("j")
    assert td2.index_set == {"i", "j"}
    assert td2.get_dim_size("j") == 1.0

    td3 = td2.add_dummy_idx("j")
    assert td3.index_set == {"i", "j"}


@pytest.mark.parametrize(
    "defs, func, expected_axes, expected_dims, expected_fill",
    [
        # union of axes; first-wins on dim size; add fills
        (
            [
                ({"i", "j"}, {"i": 10.0, "j": 5.0}, 2.0),
                ({"i", "k"}, {"i": 20.0, "k": 7.0}, 3.0),
            ],
            op.add,
            {"i", "j", "k"},
            {"i": 10.0, "j": 5.0, "k": 7.0},
            5.0,
        ),
        # same axes: max over fills; first-wins on size still applies
        (
            [
                ({"i"}, {"i": 6.0}, 2.0),
                ({"i"}, {"i": 9.0}, 4.0),
            ],
            max,
            {"i"},
            {"i": 6.0},
            4.0,
        ),
        # three defs; sum fills via variadic callable
        (
            [
                ({"i"}, {"i": 5.0}, 1.0),
                ({"i"}, {"i": 5.0}, 2.0),
                ({"i"}, {"i": 5.0}, 3.0),
            ],
            lambda *xs: sum(xs),
            {"i"},
            {"i": 5.0},
            6.0,
        ),
    ],
)
def test_tensordef_mapjoin(defs, func, expected_axes, expected_dims, expected_fill):
    objs = [TensorDef(ax, dims, fv) for (ax, dims, fv) in defs]
    out = TensorDef.mapjoin(func, *objs)
    assert out.index_set == expected_axes
    assert out.dim_sizes == expected_dims
    assert out.fill_value == expected_fill


@pytest.mark.parametrize(
    (
        "op_func",
        "index_set",
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
            op.add,
            ["i", "j", "k"],
            {"i": 10.0, "j": 5.0, "k": 3.0},
            0.5,
            ["j"],
            {"i", "k"},
            {"i": 10.0, "k": 3.0},
            0.5 * 5,
        ),
        # addition: drop multiple axes (n = 4*16 = 64) → fill' = 7 * 64
        (
            op.add,
            ["a", "b", "c", "d"],
            {"a": 2.0, "b": 4.0, "c": 8.0, "d": 16.0},
            7.0,
            ["b", "d"],
            {"a", "c"},
            {"a": 2.0, "c": 8.0},
            7.0 * (4 * 16),
        ),
        # addition: no-op when reduce set is empty (n = 1) → fill unchanged
        (
            op.add,
            ["x", "y"],
            {"x": 3.0, "y": 9.0},
            1.0,
            [],
            {"x", "y"},
            {"x": 3.0, "y": 9.0},
            1.0,
        ),
        # addition: missing axis in reduce set → nothing reduced → fill unchanged
        (
            op.add,
            ["i", "j"],
            {"i": 5.0, "j": 6.0},
            0.0,
            ["z"],
            {"i", "j"},
            {"i": 5.0, "j": 6.0},
            0.0,
        ),
        # multiplication: reduce 'j' (n = 3) → fill' = (2.0) ** 3 = 8
        (
            op.mul,
            ["i", "j"],
            {"i": 2.0, "j": 3.0},
            2.0,
            ["j"],
            {"i"},
            {"i": 2.0},
            8.0,
        ),
        # idempotent op: reduce entire axis → empty shape
        (
            min,
            ["i"],
            {"i": 4.0},
            7.0,
            ["i"],
            set(),
            {},
            7.0,
        ),
    ],
)
def test_tensordef_aggregate(
    op_func,
    index_set,
    dim_sizes,
    fill_value,
    reduce_fields,
    expected_axes,
    expected_dims,
    expected_fill,
):
    d = TensorDef(index_set=index_set, dim_sizes=dim_sizes, fill_value=fill_value)
    out = TensorDef.aggregate(op_func, None, reduce_fields, d)

    assert out.index_set == expected_axes
    assert out.dim_sizes == expected_dims
    assert out.fill_value == expected_fill


# ─────────────────────────────── DenseStats tests ─────────────────────────────


def test_from_tensor_and_getters():
    arr = np.zeros((2, 3))
    node = Table(Literal(arr), (Field("i"), Field("j")))
    stats: TensorStats = _insert_statistics(
        ST=DenseStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.index_set == {"i", "j"}
    assert stats.get_dim_size("i") == 2.0
    assert stats.get_dim_size("j") == 3.0
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
    axes = [f"x{i}" for i in range(len(shape))]
    arr = np.zeros(shape)
    node = Table(Literal(arr), tuple(Field(a) for a in axes))

    stats: TensorStats = _insert_statistics(
        ST=DenseStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    assert stats.index_set == set(axes)
    assert stats.estimate_non_fill_values() == expected


def test_mapjoin_mul_and_add():
    A = np.ones((2, 3))
    B = np.ones((3, 4))
    dsa = DenseStats(A, ["i", "j"])
    dsb = DenseStats(B, ["j", "k"])

    dsm = DenseStats.mapjoin(op.mul, dsa, dsb)
    assert dsm.index_set == {"i", "j", "k"}
    assert dsm.get_dim_size("i") == 2.0
    assert dsm.get_dim_size("j") == 3.0
    assert dsm.get_dim_size("k") == 4.0
    assert dsm.fill_value == 0.0

    dsa2 = DenseStats(2 * A, ["i", "j"])
    ds_sum = DenseStats.mapjoin(op.add, dsa, dsa2)
    assert ds_sum.fill_value == 1 + 2

def test_aggregate_and_issimilar():
    table = Table(
        Literal(np.ones((2, 3))),
        (Field("i"), Field("j")),
    )
    dsa: TensorStats = _insert_statistics(
        ST=DenseStats, node=table, bindings=OrderedDict(), replace=False, cache={}
    )

    node_add = Aggregate(
        op=Literal(op.add),
        init=None,
        arg=table,
        idxs=(Field("j"),),
    )

    ds_agg: TensorStats = _insert_statistics(
        ST=DenseStats, node=node_add, bindings=OrderedDict(), replace=False, cache={}
    )

    assert ds_agg.index_set == {"i"}
    assert ds_agg.get_dim_size("i") == 2.0
    assert ds_agg.fill_value == dsa.fill_value
    assert DenseStats.issimilar(dsa, dsa)


# ─────────────────────────────── DCStats tests ─────────────────────────────


@pytest.mark.parametrize(
    "tensor, fields, expected_dcs",
    [
        (np.array([], dtype=int), ["i"], set()),
        (np.array([1, 1, 1, 1]), ["i"], {DC(frozenset(), frozenset(["i"]), 4.0)}),
        (np.array([0, 1, 0, 0, 1]), ["i"], {DC(frozenset(), frozenset(["i"]), 2.0)}),
    ],
)
def test_dc_stats_vector(tensor, fields, expected_dcs):
    node = Table(
        Literal(tensor),
        tuple(Field(f) for f in fields),
    )
    stats = _insert_statistics(
        ST=DCStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.dcs == expected_dcs


@pytest.mark.parametrize(
    "tensor, fields, expected_dcs",
    [
        (np.zeros((0, 0), dtype=int), ["i", "j"], set()),
        (
            np.ones((3, 3), dtype=int),
            ["i", "j"],
            {
                DC(frozenset(), frozenset(["i", "j"]), 9.0),
                DC(frozenset(), frozenset(["i"]), 3.0),
                DC(frozenset(), frozenset(["j"]), 3.0),
                DC(frozenset(["i"]), frozenset(["i", "j"]), 3.0),
                DC(frozenset(["j"]), frozenset(["i", "j"]), 3.0),
            },
        ),
        (
            np.array(
                [
                    [1, 0, 1],
                    [0, 0, 0],
                    [1, 1, 0],
                ],
                dtype=int,
            ),
            ["i", "j"],
            {
                DC(frozenset(), frozenset(["i", "j"]), 4.0),
                DC(frozenset(), frozenset(["i"]), 3.0),
                DC(frozenset(), frozenset(["j"]), 2.0),
                DC(frozenset(["i"]), frozenset(["i", "j"]), 2.0),
                DC(frozenset(["j"]), frozenset(["i", "j"]), 2.0),
            },
        ),
    ],
)
def test_dc_stats_matrix(tensor, fields, expected_dcs):
    node = Table(
        Literal(tensor),
        tuple(Field(f) for f in fields),
    )
    stats = _insert_statistics(
        ST=DCStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.dcs == expected_dcs



@pytest.mark.parametrize(
    "tensor, fields, expected_dcs",
    [
        (np.zeros((0, 0, 0), dtype=int), ["i", "j", "k"], set()),
        (
            np.ones((2, 2, 2), dtype=int),
            ["i", "j", "k"],
            {
                DC(frozenset(), frozenset(["i", "j", "k"]), 8.0),
                DC(frozenset(), frozenset(["i"]), 2.0),
                DC(frozenset(), frozenset(["j"]), 2.0),
                DC(frozenset(), frozenset(["k"]), 2.0),
                DC(frozenset(["i"]), frozenset(["j", "k"]), 4.0),
                DC(frozenset(["j"]), frozenset(["i", "k"]), 4.0),
                DC(frozenset(["k"]), frozenset(["i", "j"]), 4.0),
            },
        ),
        (
            np.array(
                [
                    [[1, 0], [0, 0]],
                    [[0, 1], [1, 0]],
                ],
                dtype=int,
            ),
            ["i", "j", "k"],
            {
                DC(frozenset(), frozenset(["i", "j", "k"]), 3.0),
                DC(frozenset(), frozenset(["i"]), 2.0),
                DC(frozenset(), frozenset(["j"]), 2.0),
                DC(frozenset(), frozenset(["k"]), 2.0),
                DC(frozenset(["i"]), frozenset(["j", "k"]), 2.0),
                DC(frozenset(["j"]), frozenset(["i", "k"]), 2.0),
                DC(frozenset(["k"]), frozenset(["i", "j"]), 2.0),
            },
        ),
    ],
)
def test_dc_stats_3d(tensor, fields, expected_dcs):
    node = Table(
        Literal(tensor),
        tuple(Field(f) for f in fields),
    )
    stats = _insert_statistics(
        ST=DCStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.dcs == expected_dcs



@pytest.mark.parametrize(
    "tensor, fields, expected_dcs",
    [
        (np.zeros((0, 0, 0, 0), dtype=int), ["i", "j", "k", "l"], set()),
        (
            np.ones((2, 2, 2, 2), dtype=int),
            ["i", "j", "k", "l"],
            {
                DC(frozenset(), frozenset(["i", "j", "k", "l"]), 16.0),
                DC(frozenset(), frozenset(["i"]), 2.0),
                DC(frozenset(), frozenset(["j"]), 2.0),
                DC(frozenset(), frozenset(["k"]), 2.0),
                DC(frozenset(), frozenset(["l"]), 2.0),
                DC(frozenset(["i"]), frozenset(["j", "k", "l"]), 8.0),
                DC(frozenset(["j"]), frozenset(["i", "k", "l"]), 8.0),
                DC(frozenset(["k"]), frozenset(["i", "j", "l"]), 8.0),
                DC(frozenset(["l"]), frozenset(["i", "j", "k"]), 8.0),
            },
        ),
        (
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
            ),
            ["i", "j", "k", "l"],
            {
                DC(frozenset(), frozenset(["i", "j", "k", "l"]), 3.0),
                DC(frozenset(), frozenset(["i"]), 2.0),
                DC(frozenset(), frozenset(["j"]), 2.0),
                DC(frozenset(), frozenset(["k"]), 2.0),
                DC(frozenset(), frozenset(["l"]), 2.0),
                DC(frozenset(["i"]), frozenset(["j", "k", "l"]), 2.0),
                DC(frozenset(["j"]), frozenset(["i", "k", "l"]), 2.0),
                DC(frozenset(["k"]), frozenset(["i", "j", "l"]), 2.0),
                DC(frozenset(["l"]), frozenset(["i", "j", "k"]), 2.0),
            },
        ),
    ],
)
def test_dc_stats_4d(tensor, fields, expected_dcs):
    node = Table(
        Literal(tensor),
        tuple(Field(f) for f in fields),
    )
    stats = _insert_statistics(
        ST=DCStats,
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
            {"i": 1000, "j": 1000},
            [
                DC(frozenset(["i"]), frozenset(["j"]), 5),
                DC(frozenset(["j"]), frozenset(["i"]), 25),
                DC(frozenset(), frozenset(["i", "j"]), 50),
            ],
            50,
        ),
    ],
)
def test_single_tensor_card(dims, dcs, expected_nnz):
    node = Table(
            Literal(np.zeros((1, 1), dtype=int)),
            (Field("i"), Field("j"))
    )
    stat = _insert_statistics(
        ST=DCStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    stat.tensordef = TensorDef(frozenset(["i", "j"]), dims, 0)
    stat.dcs = set(dcs)

    assert stat.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {"i": 1000, "j": 1000, "k": 1000},
            [
                DC(frozenset(["j"]), frozenset(["k"]), 5),
                DC(frozenset(), frozenset(["i", "j"]), 50),
            ],
            50 * 5,
        ),
    ],
)
def test_1_join_dc_card(dims, dcs, expected_nnz):
    node = Table(
            Literal(np.zeros((1, 1, 1), dtype=int)),
            (Field("i"), Field("j"), Field("k"))
    )
    stat = _insert_statistics(
        ST=DCStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    stat.tensordef = TensorDef(frozenset(["i", "j", "k"]), dims, 0)
    stat.dcs = set(dcs)
    assert stat.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {"i": 1000, "j": 1000, "k": 1000, "l": 1000},
            [
                DC(frozenset(), frozenset(["i", "j"]), 50),
                DC(frozenset(["j"]), frozenset(["k"]), 5),
                DC(frozenset(["k"]), frozenset(["l"]), 5),
            ],
            50 * 5 * 5,
        ),
    ],
)
def test_2_join_dc_card(dims, dcs, expected_nnz):
    node = Table(
            Literal(np.zeros((1, 1, 1, 1), dtype=int)),
            (Field("i"), Field("j"), Field("k"), Field("l"))
    )
    stat = _insert_statistics(
        ST=DCStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    stat.tensordef = TensorDef(frozenset(["i", "j", "k", "l"]), dims, 0)
    stat.dcs = set(dcs)
    assert stat.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {"i": 1000, "j": 1000, "k": 1000},
            [
                DC(frozenset(), frozenset(["i", "j"]), 50),
                DC(frozenset(["i"]), frozenset(["j"]), 5),
                DC(frozenset(["j"]), frozenset(["i"]), 5),
                DC(frozenset(), frozenset(["j", "k"]), 50),
                DC(frozenset(["j"]), frozenset(["k"]), 5),
                DC(frozenset(["k"]), frozenset(["j"]), 5),
                DC(frozenset(), frozenset(["i", "k"]), 50),
                DC(frozenset(["i"]), frozenset(["k"]), 5),
                DC(frozenset(["k"]), frozenset(["i"]), 5),
            ],
            50 * 5,
        ),
    ],
)
def test_triangle_dc_card(dims, dcs, expected_nnz):
    node = Table(
            Literal(np.zeros((1, 1, 1), dtype=int)),
            (Field("i"), Field("j"), Field("k"))
    )
    stat = _insert_statistics(
        ST=DCStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    stat.tensordef = TensorDef(frozenset(["i", "j", "k"]), dims, 0)
    stat.dcs = set(dcs)
    assert stat.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {"i": 1000, "j": 1000, "k": 1000},
            [
                DC(frozenset(), frozenset(["i", "j"]), 1),
                DC(frozenset(["i"]), frozenset(["j"]), 1),
                DC(frozenset(["j"]), frozenset(["i"]), 1),
                DC(frozenset(), frozenset(["j", "k"]), 50),
                DC(frozenset(["j"]), frozenset(["k"]), 5),
                DC(frozenset(["k"]), frozenset(["j"]), 5),
                DC(frozenset(), frozenset(["i", "k"]), 50),
                DC(frozenset(["i"]), frozenset(["k"]), 5),
                DC(frozenset(["k"]), frozenset(["i"]), 5),
            ],
            1 * 5,
        ),
    ],
)
def test_triangle_small_dc_card(dims, dcs, expected_nnz):
    node = Table(
            Literal(np.zeros((1, 1, 1), dtype=int)),
            (Field("i"), Field("j"), Field("k"))
    )
    stat = _insert_statistics(
        ST=DCStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    stat.tensordef = TensorDef(frozenset(["i", "j", "k"]), dims, 0)
    stat.dcs = set(dcs)
    assert stat.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs_list, expected_dcs",
    [
        # Single input passthrough
        (
            {"i": 1000},
            [
                {
                    DC(frozenset(), frozenset({"i"}), 5.0),
                    DC(frozenset({"i"}), frozenset({"i"}), 1.0),
                }
            ],
            {
                DC(frozenset(), frozenset({"i"}), 5.0),
                DC(frozenset({"i"}), frozenset({"i"}), 1.0),
            },
        ),
        # Two inputs: overlap takes min; unique keys are preserved
        (
            {"i": 1000},
            [
                {
                    DC(frozenset(), frozenset({"i"}), 5.0),
                    DC(frozenset({"i"}), frozenset({"i"}), 1.0),
                },
                {
                    DC(frozenset(), frozenset({"i"}), 2.0),
                    DC(frozenset({"i"}), frozenset({"i"}), 3.0),
                    DC(frozenset(), frozenset(), 7.0),
                },
            ],
            {
                DC(frozenset(), frozenset({"i"}), 2.0),
                DC(frozenset({"i"}), frozenset({"i"}), 1.0),
                DC(frozenset(), frozenset(), 7.0),
            },
        ),
    ],
)
def test_merge_dc_join(dims, dcs_list, expected_dcs):
    stats_objs = []
    for dcs in dcs_list:
        node = Table(Literal(np.zeros((1,), dtype=int)), (Field("i"),))
        s = _insert_statistics(
            ST=DCStats,
            node=node,
            bindings=OrderedDict(),
            replace=False,
            cache={},
        )
        s.tensordef = TensorDef(frozenset({"i"}), dims, 0)
        s.dcs = set(dcs)
        stats_objs.append(s)

    new_def = TensorDef(frozenset({"i"}), dims, 0)
    out = DCStats._merge_dc_join(new_def, stats_objs)

    assert out.tensordef.index_set == {"i"}
    assert out.tensordef.dim_sizes == dims
    assert out.dcs == expected_dcs

@pytest.mark.parametrize(
    "new_dims, inputs, expected_dcs",
    [
        # Single input passthrough
        (
            {"i": 1000},
            [
                (
                    {"i"},
                    {
                        DC(frozenset(), frozenset({"i"}), 5.0),
                        DC(frozenset({"i"}), frozenset({"i"}), 1.0),
                    },
                )
            ],
            {
                DC(frozenset(), frozenset({"i"}), 5.0),
                DC(frozenset({"i"}), frozenset({"i"}), 1.0),
            },
        ),
        # Two inputs, same axes: overlap SUMs; keys not in all inputs are dropped
        (
            {"i": 1000},
            [
                (
                    {"i"},
                    {
                        DC(frozenset(), frozenset({"i"}), 5.0),
                        DC(frozenset({"i"}), frozenset({"i"}), 1.0),
                    },
                ),
                (
                    {"i"},
                    {
                        DC(frozenset(), frozenset({"i"}), 2.0),
                        DC(frozenset({"i"}), frozenset({"i"}), 3.0),
                        DC(frozenset(), frozenset(), 7.0),
                    },
                ),
            ],
            {
                DC(frozenset(), frozenset({"i"}), 7.0),
                DC(frozenset({"i"}), frozenset({"i"}), 4.0),
            },
        ),
        # Lifting across extra axes (Z) + consensus then SUM
        (
            {"i": 10, "j": 4},
            [
                ({"i"}, {DC(frozenset(), frozenset({"i"}), 3.0)}),
                ({"j"}, {DC(frozenset(), frozenset({"j"}), 2.0)}),
            ],
            {DC(frozenset(), frozenset({"i", "j"}), 32.0)},
        ),
        # Clamp by dense capacity of Y
        (
            {"i": 5},
            [
                ({"i"}, {DC(frozenset(), frozenset({"i"}), 7.0)}),
                ({"i"}, {DC(frozenset(), frozenset({"i"}), 9.0)}),
            ],
            {DC(frozenset(), frozenset({"i"}), 5.0)},
        ),
    ],
)
def test_merge_dc_union(new_dims, inputs, expected_dcs):
    cache = {}

    stats_objs = []
    for idx_set, dcs in inputs:
        fields = tuple(Field(ax) for ax in sorted(idx_set))
        shape = (1,) * max(1, len(fields))
        node = Table(Literal(np.zeros(shape, dtype=int)), fields)

        _insert_statistics(
            ST=DCStats,
            node=node,
            bindings=OrderedDict(),
            replace=False,
            cache=cache,
        )

        td = TensorDef(frozenset(idx_set), {k: new_dims[k] for k in idx_set}, 0)
        stats_objs.append(DCStats.from_def(td, set(dcs)))

    new_def = TensorDef(frozenset(new_dims.keys()), new_dims, 0)
    out = DCStats._merge_dc_union(new_def, stats_objs)

    assert out.tensordef.index_set == set(new_dims.keys())
    assert dict(out.tensordef.dim_sizes) == new_dims
    assert out.dcs == expected_dcs

@pytest.mark.parametrize(
    "dims1, dcs1, dims2, dcs2, expected_nnz",
    [
        (
            {"i": 1000},
            [DC(frozenset(), frozenset(["i"]), 1)],
            {"i": 1000},
            [DC(frozenset(), frozenset(["i"]), 1)],
            2,
        ),
    ],
)
def test_1d_disjunction_dc_card(dims1, dcs1, dims2, dcs2, expected_nnz):
    cache = {}

    node1 = Table(
        Literal(np.zeros((1,), dtype=int)),
        (Field("i"),)
    )
    s1 = _insert_statistics(ST=DCStats, node=node1, bindings=OrderedDict(), replace=False, cache=cache)
    s1.tensordef = TensorDef(frozenset({"i"}), dims1, 0)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(np.zeros((1,), dtype=int)),
        (Field("i"),)
    )
    s2 = _insert_statistics(ST=DCStats, node=node2, bindings=OrderedDict(), replace=False, cache=cache)
    s2.tensordef = TensorDef(frozenset({"i"}), dims2, 0)
    s2.dcs = set(dcs2)

    parent = MapJoin(
        Literal(op.add),
        (node1, node2)
    )
    reduce_stats = _insert_statistics(ST=DCStats, node=parent, bindings=OrderedDict(), replace=False, cache=cache)

    assert reduce_stats.estimate_non_fill_values() == expected_nnz

@pytest.mark.parametrize(
    "dims1, dcs1, dims2, dcs2, expected_nnz",
    [
        (
            {"i": 1000, "j": 1000},
            [DC(frozenset(), frozenset(["i", "j"]), 1)],
            {"i": 1000, "j": 1000},
            [DC(frozenset(), frozenset(["i", "j"]), 1)],
            2,
        ),
    ],
)
def test_2d_disjunction_dc_card(dims1, dcs1, dims2, dcs2, expected_nnz):
    cache = {}

    node1 = Table(
        Literal(np.zeros((1, 1), dtype=int)),
        (Field("i"), Field("j"))
    )
    s1 = _insert_statistics(ST=DCStats, node=node1, bindings=OrderedDict(), replace=False, cache=cache)
    s1.tensordef = TensorDef(frozenset({"i", "j"}), dims1, 0)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(np.zeros((1, 1), dtype=int)),
        (Field("i"),  Field("j"))
    )
    s2 = _insert_statistics(ST=DCStats, node=node2, bindings=OrderedDict(), replace=False, cache=cache)
    s2.tensordef = TensorDef(frozenset({"i", "j"}), dims2, 0)
    s2.dcs = set(dcs2)

    parent = MapJoin(
        Literal(op.add),
        (node1, node2)
    )
    reduce_stats = _insert_statistics(ST=DCStats, node=parent, bindings=OrderedDict(), replace=False, cache=cache)

    assert reduce_stats.estimate_non_fill_values() == expected_nnz

@pytest.mark.parametrize(
    "dims1, dcs1, dims2, dcs2, expected_nnz",
    [
        (
            {"i": 1000},
            [DC(frozenset(), frozenset(["i"]), 5)],
            {"j": 100},
            [DC(frozenset(), frozenset(["j"]), 10)],
            10 * 1000 + 5 * 100,
        ),
    ],
)
def test_2d_disjoin_disjunction_dc_card(dims1, dcs1, dims2, dcs2, expected_nnz):
    cache = {}

    node1 = Table(
        Literal(np.zeros((1,), dtype=int)),
        (Field("i"),)
    )
    s1 = _insert_statistics(ST=DCStats, node=node1, bindings=OrderedDict(), replace=False, cache=cache)
    s1.tensordef = TensorDef(frozenset({"i"}), dims1, 0)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(np.zeros((1,), dtype=int)),
        (Field("j"),)
    )
    s2 = _insert_statistics(ST=DCStats, node=node2, bindings=OrderedDict(), replace=False, cache=cache)
    s2.tensordef = TensorDef(frozenset({"j"}), dims2, 0)
    s2.dcs = set(dcs2)

    parent = MapJoin(
        Literal(op.add),
        (node1, node2)
    )
    reduce_stats = _insert_statistics(ST=DCStats, node=parent, bindings=OrderedDict(), replace=False, cache=cache)

    assert reduce_stats.estimate_non_fill_values() == expected_nnz

@pytest.mark.parametrize(
    "dims1, dcs1, dims2, dcs2, expected_nnz",
    [
        (
            {"i": 1000, "j": 100},
            [DC(frozenset(), frozenset(["i", "j"]), 5)],
            {"j": 100, "k": 1000},
            [DC(frozenset(), frozenset(["j", "k"]), 10)],
            10 * 1000 + 5 * 1000,
        ),
    ],
)
def test_3d_disjoint_disjunction_dc_card(dims1, dcs1, dims2, dcs2, expected_nnz):
    cache = {}

    node1 = Table(
        Literal(np.zeros((1, 1), dtype=int)),
        (Field("i"), Field("j")),
    )
    s1 = _insert_statistics(
        ST=DCStats, node=node1, bindings=OrderedDict(), replace=False, cache=cache
    )
    s1.tensordef = TensorDef(frozenset({"i", "j"}), dims1, 0)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(np.zeros((1, 1), dtype=int)),
        (Field("j"), Field("k")),
    )
    s2 = _insert_statistics(
        ST=DCStats, node=node2, bindings=OrderedDict(), replace=False, cache=cache
    )
    s2.tensordef = TensorDef(frozenset({"j", "k"}), dims2, 0)
    s2.dcs = set(dcs2)

    parent = MapJoin(
        Literal(op.add),
        (node1, node2)
    )
    reduce_stats = _insert_statistics(
        ST=DCStats, node=parent, bindings=OrderedDict(), replace=False, cache=cache
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz

""""""
@pytest.mark.parametrize(
    "dims1, dcs1, dims2, dcs2, dims3, dcs3, expected_nnz",
    [
        (
            {"i": 1000, "j": 100},
            [DC(frozenset(), frozenset(["i", "j"]), 5)],
            {"j": 100, "k": 1000},
            [DC(frozenset(), frozenset(["j", "k"]), 10)],
            {"i": 1000, "j": 100, "k": 1000},
            [DC(frozenset(), frozenset(["i", "j", "k"]), 10)],
            10 * 1000 + 5 * 1000 + 10,
        ),
    ],
)
def test_large_disjoint_disjunction_dc_card(
    dims1, dcs1, dims2, dcs2, dims3, dcs3, expected_nnz
):
    cache = {}

    node1 = Table(
        Literal(np.zeros((1, 1), dtype=int)),
        (Field("i"), Field("j"))
    )
    s1 = _insert_statistics(ST=DCStats, node=node1, bindings=OrderedDict(), replace=False, cache=cache)
    s1.tensordef = TensorDef(frozenset({"i", "j"}), dims1, 1)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(np.zeros((1, 1), dtype=int)),
        (Field("j"), Field("k"))
    )
    s2 = _insert_statistics(ST=DCStats, node=node2, bindings=OrderedDict(), replace=False, cache=cache)
    s2.tensordef = TensorDef(frozenset({"j", "k"}), dims2, 1)
    s2.dcs = set(dcs2)

    node3 = Table(
        Literal(np.zeros((1, 1, 1), dtype=int)),
        (Field("i"), Field("j"), Field("k"))
    )
    s3 = _insert_statistics(ST=DCStats, node=node3, bindings=OrderedDict(), replace=False, cache=cache)
    s3.tensordef = TensorDef(frozenset({"i", "j", "k"}), dims3, 1)
    s3.dcs = set(dcs3)

    map = MapJoin(
        Literal(op.mul),
        (node1, node2)
    )

    parent = MapJoin(
        Literal(op.mul),
        (map, node3)
    )

    reduce_stats = _insert_statistics(ST=DCStats, node=parent, bindings=OrderedDict(), replace=False, cache=cache)

    assert reduce_stats.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims1, dcs1, dims2, dcs2, dims3, dcs3, expected_nnz",
    [
        (
            {"i": 1000, "j": 100},
            [DC(frozenset(), frozenset(["i", "j"]), 5)],
            {"j": 100, "k": 1000},
            [DC(frozenset(), frozenset(["j", "k"]), 10)],
            {"i": 1000, "j": 100, "k": 1000},
            [DC(frozenset(), frozenset(["i", "j", "k"]), 10)],
            10,
        ),
    ],
)
def test_mixture_disjoint_disjunction_dc_card(
    dims1, dcs1, dims2, dcs2, dims3, dcs3, expected_nnz
):
    cache = {}

    node1 = Table(
        Literal(np.zeros((1, 1), dtype=int)),
        (Field("i"), Field("j"))
    )
    s1 = _insert_statistics(ST=DCStats, node=node1, bindings=OrderedDict(), replace=False, cache=cache)
    s1.tensordef = TensorDef(frozenset(["i", "j"]), dims1, 1)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(np.zeros((1, 1), dtype=int)),
        (Field("j"), Field("k"))
    )
    s2 = _insert_statistics(ST=DCStats, node=node2, bindings=OrderedDict(), replace=False, cache=cache)
    s2.tensordef = TensorDef(frozenset(["j", "k"]), dims2, 1)
    s2.dcs = set(dcs2)

    node3 = Table(
        Literal(np.zeros((1, 1, 1), dtype=int)),
        (Field("i"), Field("j"), Field("k"))
    )
    s3 = _insert_statistics(ST=DCStats, node=node3, bindings=OrderedDict(), replace=False, cache=cache)
    s3.tensordef = TensorDef(frozenset(["i", "j", "k"]), dims3, 0)
    s3.dcs = set(dcs3)

    map = MapJoin(
        Literal(op.mul),
        (node1, node2)
    )
    parent = MapJoin(
        Literal(op.mul),
        (map, node3)
    )

    reduce_stats = _insert_statistics(
        ST=DCStats,
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
            {"i": 1000, "j": 1000, "k": 1000},
            [
                DC(frozenset(), frozenset(["i", "j"]), 50),
                DC(frozenset(["i"]), frozenset(["j"]), 5),
                DC(frozenset(["j"]), frozenset(["i"]), 5),
                DC(frozenset(), frozenset(["j", "k"]), 50),
                DC(frozenset(["j"]), frozenset(["k"]), 5),
                DC(frozenset(["k"]), frozenset(["j"]), 5),
                DC(frozenset(), frozenset(["i", "k"]), 50),
                DC(frozenset(["i"]), frozenset(["k"]), 5),
                DC(frozenset(["k"]), frozenset(["i"]), 5),
            ],
            1,
        ),
    ],
)
def test_full_reduce_DC_card(dims, dcs, expected_nnz):
    cache = {}

    node = Table(
        Literal(np.zeros((1, 1, 1), dtype=int)),
        (Field("i"), Field("j"), Field("k")),
    )
    stat = _insert_statistics(
        ST=DCStats, node=node, bindings=OrderedDict(), replace=False, cache=cache
    )
    stat.tensordef = TensorDef(frozenset(["i", "j", "k"]), dims, 0.0)
    stat.dcs = set(dcs)

    reduce_node = Aggregate(
        op=Literal(op.add),
        init=Literal(0),
        idxs=(Field("i"), Field("j"), Field("k")),
        arg=node,
    )
    reduce_stats = _insert_statistics(
        ST=DCStats, node=reduce_node, bindings=OrderedDict(), replace=False, cache=cache
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {"i": 1000, "j": 1000, "k": 1000},
            [
                DC(frozenset(), frozenset(["i", "j"]), 1),
                DC(frozenset(["i"]), frozenset(["j"]), 1),
                DC(frozenset(["j"]), frozenset(["i"]), 1),
                DC(frozenset(), frozenset(["j", "k"]), 50),
                DC(frozenset(["j"]), frozenset(["k"]), 5),
                DC(frozenset(["k"]), frozenset(["j"]), 5),
                DC(frozenset(), frozenset(["i", "k"]), 50),
                DC(frozenset(["i"]), frozenset(["k"]), 5),
                DC(frozenset(["k"]), frozenset(["i"]), 5),
            ],
            5,
        ),
    ],
)
def test_1_attr_reduce_DC_card(dims, dcs, expected_nnz):
    cache = {}

    node = Table(
        Literal(np.zeros((1, 1, 1), dtype=int)),
        (Field("i"), Field("j"), Field("k")),
    )
    st = _insert_statistics(
        ST=DCStats, node=node, bindings=OrderedDict(), replace=False, cache=cache
    )
    st.tensordef = TensorDef(frozenset(["i", "j", "k"]), dims, 0.0)
    st.dcs = set(dcs)

    reduce_node = Aggregate(
        op=Literal(op.add),
        init=Literal(0),
        idxs=(Field("i"), Field("j")),
        arg=node,
    )
    reduce_stats = _insert_statistics(
        ST=DCStats, node=reduce_node, bindings=OrderedDict(), replace=False, cache=cache
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz


@pytest.mark.parametrize(
    "dims, dcs, expected_nnz",
    [
        (
            {"i": 1000, "j": 1000, "k": 1000},
            [
                DC(frozenset(), frozenset(["i", "j"]), 1),
                DC(frozenset(["i"]), frozenset(["j"]), 1),
                DC(frozenset(["j"]), frozenset(["i"]), 1),
                DC(frozenset(), frozenset(["j", "k"]), 50),
                DC(frozenset(["j"]), frozenset(["k"]), 5),
                DC(frozenset(["k"]), frozenset(["j"]), 5),
                DC(frozenset(), frozenset(["i", "k"]), 50),
                DC(frozenset(["i"]), frozenset(["k"]), 5),
                DC(frozenset(["k"]), frozenset(["i"]), 5),
            ],
            5,
        ),
    ],
)
def test_2_attr_reduce_DC_card(dims, dcs, expected_nnz):
    cache = {}

    node = Table(
        Literal(np.zeros((1, 1, 1), dtype=int)),
        (Field("i"), Field("j"), Field("k")),
    )
    st = _insert_statistics(
        ST=DCStats, node=node, bindings=OrderedDict(), replace=False, cache=cache
    )
    st.tensordef = TensorDef(frozenset(["i", "j", "k"]), dims, 0.0)
    st.dcs = set(dcs)

    reduce_node = Aggregate(
        op=Literal(op.add),
        init=Literal(0),
        idxs=(Field("i"),),
        arg=node,
    )
    reduce_stats = _insert_statistics(
        ST=DCStats, node=reduce_node, bindings=OrderedDict(), replace=False, cache=cache
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz

@pytest.mark.parametrize(
    "dims, dcs, reduce_indices, expected_nnz",
    [
        # Asymmetric densities
        (
            {"i": 100, "j": 100, "k": 100},
            [
                DC(frozenset(), frozenset(["i", "j"]), 100),
                DC(frozenset(["i"]), frozenset(["j"]), 2),
                DC(frozenset(), frozenset(["j", "k"]), 50),
            ],
            ["j"],
            5000,
        ),
        # Sparse + dense mix
        (
            {"i": 100, "j": 100, "k": 100},
            [
                DC(frozenset(), frozenset(["i", "k"]), 900),
                DC(frozenset(["i"]), frozenset(["k"]), 1),
            ],
            ["i", "k"],
            100,
        ),
        # Imbalance across dimensions
        (
            {"i": 1000, "j": 100, "k": 10},
            [
                DC(frozenset(), frozenset(["i", "j"]), 5),
                DC(frozenset(), frozenset(["j", "k"]), 80),
                DC(frozenset(), frozenset(["i", "k"]), 1),
            ],
            ["i"],
            5,
        ),
    ],
)
def test_varied_reduce_DC_card(dims, dcs, reduce_indices, expected_nnz):
    cache = {}

    node = Table(
        Literal(np.zeros((1, 1, 1), dtype=int)),
        (Field("i"), Field("j"), Field("k")),
    )
    st = _insert_statistics(
        ST=DCStats, node=node, bindings=OrderedDict(), replace=False, cache=cache
    )
    st.tensordef = TensorDef(frozenset(["i", "j", "k"]), dims, 0.0)
    st.dcs = set(dcs)

    reduce_fields = tuple(Field(ax) for ax in reduce_indices)
    reduce_node = Aggregate(
        op=Literal(op.add),
        init=Literal(0),
        idxs=reduce_fields,
        arg=node,
    )
    reduce_stats = _insert_statistics(
        ST=DCStats, node=reduce_node, bindings=OrderedDict(), replace=False, cache=cache
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz