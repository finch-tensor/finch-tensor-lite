import operator as op
from collections import OrderedDict

import pytest

import numpy as np

import finchlite
import finchlite as fl
from finchlite.finch_logic import (
    Aggregate,
    Field,
    Literal,
    MapJoin,
    Table,
)
from finchlite.galley.LogicalOptimizer import (
    AnnotatedQuery,
    find_lowest_roots,
    get_idx_connected_components,
    get_lazy_tensor_stats,
    get_reducible_idxs,
    insert_statistics,
    replace_and_remove_nodes,
)
from finchlite.galley.TensorStats import DC, DCStats, DenseStats, TensorDef
from finchlite.interface import lazy

# ─────────────────────────────── TensorDef tests ─────────────────────────────────


def test_copy_and_getters():
    td = TensorDef(index_order=("i", "j"), dim_sizes={"i": 2.0, "j": 3.0}, fill_value=42)
    td_copy = td.copy()
    assert td_copy is not td
    assert td_copy.index_order == ("i", "j")
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
    td = TensorDef(index_order=orig_axes, dim_sizes=dim_sizes, fill_value=0)
    td2 = td.reindex_def(new_axes)
    assert td2.index_order == tuple(new_axes)
    for ax in new_axes:
        assert td2.get_dim_size(ax) == td.get_dim_size(ax)


def test_set_fill_value():
    td = TensorDef(index_order=("i",), dim_sizes={"i": 5.0}, fill_value=0)
    td2 = td.set_fill_value(7)
    assert td2.fill_value == 7


def test_add_dummy_idx():
    td = TensorDef(index_order=("i",), dim_sizes={"i": 3.0}, fill_value=0)
    td2 = td.add_dummy_idx("j")
    assert td2.index_order == ("i", "j")
    assert td2.get_dim_size("j") == 1.0

    td3 = td2.add_dummy_idx("j")
    assert td3.index_order == ("i", "j")


@pytest.mark.parametrize(
    "defs, func, expected_axes, expected_dims, expected_fill",
    [
        # union of axes; first-wins on dim size; add fills
        (
            [
                (("i", "j"), {"i": 10.0, "j": 5.0}, 2.0),
                (("i", "k"), {"i": 20.0, "k": 7.0}, 3.0),
            ],
            op.add,
            ("i", "j", "k"),
            {"i": 10.0, "j": 5.0, "k": 7.0},
            5.0,
        ),
        # same axes: max over fills; first-wins on size still applies
        (
            [
                (("i"), {"i": 6.0}, 2.0),
                (("i"), {"i": 9.0}, 4.0),
            ],
            max,
            ("i",),
            {"i": 6.0},
            4.0,
        ),
        # three defs; sum fills via variadic callable
        (
            [
                (("i"), {"i": 5.0}, 1.0),
                (("i"), {"i": 5.0}, 2.0),
                (("i"), {"i": 5.0}, 3.0),
            ],
            lambda *xs: sum(xs),
            ("i",),
            {"i": 5.0},
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
            op.add,
            
            ("i", "j", "k"),
            {"i": 10.0, "j": 5.0, "k": 3.0},
            0.5,
            ("j",),
            ("i", "k"),
            {"i": 10.0, "k": 3.0},
            0.5 * 5,
        ),
        # addition: drop multiple axes (n = 4*16 = 64) → fill' = 7 * 64
        (
            op.add,
            ("a", "b", "c", "d"),
            {"a": 2.0, "b": 4.0, "c": 8.0, "d": 16.0},
            7.0,
            ("b", "d"),
            ("a", "c"),
            {"a": 2.0, "c": 8.0},
            7.0 * (4 * 16),
        ),
        # addition: no-op when reduce set is empty (n = 1) → fill unchanged
        (
            op.add,
            ("x", "y"),
            {"x": 3.0, "y": 9.0},
            1.0,
            [],
            ("x", "y"),
            {"x": 3.0, "y": 9.0},
            1.0,
        ),
        # addition: missing axis in reduce set → nothing reduced → fill unchanged
        (
            op.add,
            ("i", "j"),
            {"i": 5.0, "j": 6.0},
            0.0,
            ("z",),
            ("i", "j"),
            {"i": 5.0, "j": 6.0},
            0.0,
        ),
        # multiplication: reduce 'j' (n = 3) → fill' = (2.0) ** 3 = 8
        (
            op.mul,
            ("i", "j"),
            {"i": 2.0, "j": 3.0},
            2.0,
            ("j",),
            ("i",),
            {"i": 2.0},
            8.0,
        ),
        # idempotent op: reduce entire axis → empty shape
        (
            min,
            ("i",),
            {"i": 4.0},
            7.0,
            ("i",),
            tuple(),
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
        ST=DenseStats,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.index_order == ("i", "j")
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
    arr = fl.asarray(np.zeros(shape))
    node = Table(Literal(arr), tuple(Field(a) for a in axes))

    stats = insert_statistics(
        ST=DenseStats,
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
        ST=DenseStats, node=ta, bindings=OrderedDict(), replace=False, cache=cache
    )
    insert_statistics(
        ST=DenseStats, node=tb, bindings=OrderedDict(), replace=False, cache=cache
    )
    insert_statistics(
        ST=DenseStats, node=ta2, bindings=OrderedDict(), replace=False, cache=cache
    )
    cache[ta].fill_value = 1
    cache[tb].fill_value = 1
    cache[ta2].fill_value = 2
    node_mul = MapJoin(Literal(op.mul), (ta, tb))
    dsm = insert_statistics(
        ST=DenseStats, node=node_mul, bindings=OrderedDict(), replace=False, cache=cache
    )

    assert dsm.index_order == ("i", "j", "k")
    assert dsm.get_dim_size("i") == 2.0
    assert dsm.get_dim_size("j") == 3.0
    assert dsm.get_dim_size("k") == 4.0
    assert dsm.fill_value == 0.0

    node_add = MapJoin(Literal(op.add), (ta, ta2))
    ds_sum = insert_statistics(
        ST=DenseStats, node=node_add, bindings=OrderedDict(), replace=False, cache=cache
    )

    assert ds_sum.index_order == ("i", "j")
    assert ds_sum.get_dim_size("i") == 2.0
    assert ds_sum.get_dim_size("j") == 3.0
    assert ds_sum.fill_value == 1.0 + 2.0


def test_aggregate_and_issimilar():
    table = Table(
        Literal(fl.asarray(np.ones((2, 3)))),
        (Field("i"), Field("j")),
    )
    dsa = insert_statistics(
        ST=DenseStats, node=table, bindings=OrderedDict(), replace=False, cache={}
    )

    node_add = Aggregate(
        op=Literal(op.add),
        init=None,
        arg=table,
        idxs=(Field("j"),),
    )

    ds_agg = insert_statistics(
        ST=DenseStats, node=node_add, bindings=OrderedDict(), replace=False, cache={}
    )

    assert ds_agg.index_order == ("i",)
    assert ds_agg.get_dim_size("i") == 2.0
    assert ds_agg.fill_value == dsa.fill_value
    assert DenseStats.issimilar(dsa, dsa)

def test_relabel_dense_stats():
    arr = fl.asarray(np.zeros((2, 3)))
    table = Table(Literal(arr), (Field("i"), Field("j")))
    
    stats = insert_statistics(
        ST=DenseStats,
        node=table,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    new_stats = DenseStats.relabel(stats, ("row", "col"))

    assert new_stats.index_order == ("row", "col")

    assert new_stats.get_dim_size("row") == 2.0
    
    assert new_stats.get_dim_size("col") == 3.0

    assert new_stats.fill_value == stats.fill_value

    with pytest.raises(ValueError):
        DenseStats.relabel(stats, ("x", "y", "z"))

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
        tuple(Field(f) for f in fields),
    )
    stats = insert_statistics(
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
        (
            fl.asarray(np.array([1, 1, 1, 1])),
            ["i"],
            {
                DC(frozenset(), frozenset(["i"]), 4.0),
                DC(frozenset(["i"]), frozenset(["i"]), 1.0),
            },
        ),
        (
            fl.asarray(np.array([0, 1, 0, 0, 1])),
            ["i"],
            {
                DC(frozenset(), frozenset(["i"]), 2.0),
                DC(frozenset("i"), frozenset(["i"]), 1.0),
            },
        ),
    ],
)
def test_dc_stats_vector(tensor, fields, expected_dcs):
    node = Table(
        Literal(tensor),
        tuple(Field(f) for f in fields),
    )
    stats = insert_statistics(
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
        (
            fl.asarray(np.ones((3, 3), dtype=int)),
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
            ["i", "j"],
            {
                DC(frozenset(), frozenset(["i", "j"]), 4.0),
                DC(frozenset(), frozenset(["i"]), 2.0),
                DC(frozenset(), frozenset(["j"]), 3.0),
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
    stats = insert_statistics(
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
        (
            fl.asarray(np.ones((2, 2, 2), dtype=int)),
            ["i", "j", "k"],
            {
                DC(frozenset(), frozenset(["i", "j", "k"]), 8.0),
                DC(frozenset(), frozenset(["i"]), 2.0),
                DC(frozenset(), frozenset(["j"]), 2.0),
                DC(frozenset(), frozenset(["k"]), 2.0),
                DC(frozenset(["i"]), frozenset(["i", "j", "k"]), 4.0),
                DC(frozenset(["j"]), frozenset(["i", "j", "k"]), 4.0),
                DC(frozenset(["k"]), frozenset(["i", "j", "k"]), 4.0),
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
            ["i", "j", "k"],
            {
                DC(frozenset(), frozenset(["i", "j", "k"]), 3.0),
                DC(frozenset(), frozenset(["i"]), 2.0),
                DC(frozenset(), frozenset(["j"]), 2.0),
                DC(frozenset(), frozenset(["k"]), 2.0),
                DC(frozenset(["i"]), frozenset(["i", "j", "k"]), 2.0),
                DC(frozenset(["j"]), frozenset(["i", "j", "k"]), 2.0),
                DC(frozenset(["k"]), frozenset(["i", "j", "k"]), 2.0),
            },
        ),
    ],
)
def test_dc_stats_3d(tensor, fields, expected_dcs):
    node = Table(
        Literal(tensor),
        tuple(Field(f) for f in fields),
    )
    stats = insert_statistics(
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
        (
            fl.asarray(np.ones((2, 2, 2, 2), dtype=int)),
            ["i", "j", "k", "l"],
            {
                DC(frozenset(), frozenset(["i", "j", "k", "l"]), 16.0),
                DC(frozenset(), frozenset(["i"]), 2.0),
                DC(frozenset(), frozenset(["j"]), 2.0),
                DC(frozenset(), frozenset(["k"]), 2.0),
                DC(frozenset(), frozenset(["l"]), 2.0),
                DC(frozenset(["i"]), frozenset(["i", "j", "k", "l"]), 8.0),
                DC(frozenset(["j"]), frozenset(["i", "j", "k", "l"]), 8.0),
                DC(frozenset(["k"]), frozenset(["i", "j", "k", "l"]), 8.0),
                DC(frozenset(["l"]), frozenset(["i", "j", "k", "l"]), 8.0),
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
            ["i", "j", "k", "l"],
            {
                DC(frozenset(), frozenset(["i", "j", "k", "l"]), 3.0),
                DC(frozenset(), frozenset(["i"]), 2.0),
                DC(frozenset(), frozenset(["j"]), 2.0),
                DC(frozenset(), frozenset(["k"]), 2.0),
                DC(frozenset(), frozenset(["l"]), 2.0),
                DC(frozenset(["i"]), frozenset(["i", "j", "k", "l"]), 2.0),
                DC(frozenset(["j"]), frozenset(["i", "j", "k", "l"]), 2.0),
                DC(frozenset(["k"]), frozenset(["i", "j", "k", "l"]), 2.0),
                DC(frozenset(["l"]), frozenset(["i", "j", "k", "l"]), 2.0),
            },
        ),
    ],
)
def test_dc_stats_4d(tensor, fields, expected_dcs):
    node = Table(
        Literal(tensor),
        tuple(Field(f) for f in fields),
    )
    stats = insert_statistics(
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
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("i"), Field("j"))
    )
    stat = insert_statistics(
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
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    stat = insert_statistics(
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
        Literal(fl.asarray(np.zeros((1, 1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k"), Field("l")),
    )
    stat = insert_statistics(
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
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    stat = insert_statistics(
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
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    stat = insert_statistics(
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
        node = Table(Literal(fl.asarray(np.zeros((1,), dtype=int))), (Field("i"),))
        s = insert_statistics(
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

    assert out.tensordef.index_order == ("i",)
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
        node = Table(Literal(fl.asarray(np.zeros(shape, dtype=int))), fields)

        insert_statistics(
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

    #Does the order matter here ? - > Changed tuple to set as throwing assert error 
    assert set(out.tensordef.index_order) == set(new_dims.keys())
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

    node1 = Table(Literal(fl.asarray(np.zeros((1,), dtype=int))), (Field("i"),))
    s1 = insert_statistics(
        ST=DCStats, node=node1, bindings=OrderedDict(), replace=False, cache=cache
    )
    s1.tensordef = TensorDef(frozenset({"i"}), dims1, 0)
    s1.dcs = set(dcs1)

    node2 = Table(Literal(fl.asarray(np.zeros((1,), dtype=int))), (Field("i"),))
    s2 = insert_statistics(
        ST=DCStats, node=node2, bindings=OrderedDict(), replace=False, cache=cache
    )
    s2.tensordef = TensorDef(frozenset({"i"}), dims2, 0)
    s2.dcs = set(dcs2)

    parent = MapJoin(Literal(op.add), (node1, node2))
    reduce_stats = insert_statistics(
        ST=DCStats, node=parent, bindings=OrderedDict(), replace=False, cache=cache
    )

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
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("i"), Field("j"))
    )
    s1 = insert_statistics(
        ST=DCStats, node=node1, bindings=OrderedDict(), replace=False, cache=cache
    )
    s1.tensordef = TensorDef(frozenset({"i", "j"}), dims1, 0)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("i"), Field("j"))
    )
    s2 = insert_statistics(
        ST=DCStats, node=node2, bindings=OrderedDict(), replace=False, cache=cache
    )
    s2.tensordef = TensorDef(frozenset({"i", "j"}), dims2, 0)
    s2.dcs = set(dcs2)

    parent = MapJoin(Literal(op.add), (node1, node2))
    reduce_stats = insert_statistics(
        ST=DCStats, node=parent, bindings=OrderedDict(), replace=False, cache=cache
    )

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

    node1 = Table(Literal(fl.asarray(np.zeros((1,), dtype=int))), (Field("i"),))
    s1 = insert_statistics(
        ST=DCStats, node=node1, bindings=OrderedDict(), replace=False, cache=cache
    )
    s1.tensordef = TensorDef(frozenset({"i"}), dims1, 0)
    s1.dcs = set(dcs1)

    node2 = Table(Literal(fl.asarray(np.zeros((1,), dtype=int))), (Field("j"),))
    s2 = insert_statistics(
        ST=DCStats, node=node2, bindings=OrderedDict(), replace=False, cache=cache
    )
    s2.tensordef = TensorDef(frozenset({"j"}), dims2, 0)
    s2.dcs = set(dcs2)

    parent = MapJoin(Literal(op.add), (node1, node2))
    reduce_stats = insert_statistics(
        ST=DCStats, node=parent, bindings=OrderedDict(), replace=False, cache=cache
    )

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
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))),
        (Field("i"), Field("j")),
    )
    s1 = insert_statistics(
        ST=DCStats, node=node1, bindings=OrderedDict(), replace=False, cache=cache
    )
    s1.tensordef = TensorDef(frozenset({"i", "j"}), dims1, 0)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))),
        (Field("j"), Field("k")),
    )
    s2 = insert_statistics(
        ST=DCStats, node=node2, bindings=OrderedDict(), replace=False, cache=cache
    )
    s2.tensordef = TensorDef(frozenset({"j", "k"}), dims2, 0)
    s2.dcs = set(dcs2)

    parent = MapJoin(Literal(op.add), (node1, node2))
    reduce_stats = insert_statistics(
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
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("i"), Field("j"))
    )
    s1 = insert_statistics(
        ST=DCStats, node=node1, bindings=OrderedDict(), replace=False, cache=cache
    )
    s1.tensordef = TensorDef(frozenset({"i", "j"}), dims1, 1)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("j"), Field("k"))
    )
    s2 = insert_statistics(
        ST=DCStats, node=node2, bindings=OrderedDict(), replace=False, cache=cache
    )
    s2.tensordef = TensorDef(frozenset({"j", "k"}), dims2, 1)
    s2.dcs = set(dcs2)

    node3 = Table(
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    s3 = insert_statistics(
        ST=DCStats, node=node3, bindings=OrderedDict(), replace=False, cache=cache
    )
    s3.tensordef = TensorDef(frozenset({"i", "j", "k"}), dims3, 1)
    s3.dcs = set(dcs3)

    map = MapJoin(Literal(op.mul), (node1, node2))

    parent = MapJoin(Literal(op.mul), (map, node3))

    reduce_stats = insert_statistics(
        ST=DCStats, node=parent, bindings=OrderedDict(), replace=False, cache=cache
    )

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
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("i"), Field("j"))
    )
    s1 = insert_statistics(
        ST=DCStats, node=node1, bindings=OrderedDict(), replace=False, cache=cache
    )
    s1.tensordef = TensorDef(frozenset(["i", "j"]), dims1, 1)
    s1.dcs = set(dcs1)

    node2 = Table(
        Literal(fl.asarray(np.zeros((1, 1), dtype=int))), (Field("j"), Field("k"))
    )
    s2 = insert_statistics(
        ST=DCStats, node=node2, bindings=OrderedDict(), replace=False, cache=cache
    )
    s2.tensordef = TensorDef(frozenset(["j", "k"]), dims2, 1)
    s2.dcs = set(dcs2)

    node3 = Table(
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    s3 = insert_statistics(
        ST=DCStats, node=node3, bindings=OrderedDict(), replace=False, cache=cache
    )
    s3.tensordef = TensorDef(frozenset(["i", "j", "k"]), dims3, 0)
    s3.dcs = set(dcs3)

    map = MapJoin(Literal(op.mul), (node1, node2))
    parent = MapJoin(Literal(op.mul), (map, node3))

    reduce_stats = insert_statistics(
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
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    stat = insert_statistics(
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
    reduce_stats = insert_statistics(
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
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    st = insert_statistics(
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
    reduce_stats = insert_statistics(
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
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    st = insert_statistics(
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
    reduce_stats = insert_statistics(
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
        Literal(fl.asarray(np.zeros((1, 1, 1), dtype=int))),
        (Field("i"), Field("j"), Field("k")),
    )
    st = insert_statistics(
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
    reduce_stats = insert_statistics(
        ST=DCStats, node=reduce_node, bindings=OrderedDict(), replace=False, cache=cache
    )

    assert reduce_stats.estimate_non_fill_values() == expected_nnz

# ─────────────────────────────── Annotated_Query tests ─────────────────────────────
@pytest.mark.parametrize(
    "reduce_idxs,parent_idxs,expected",
    [
        # Some indices have parents
        (["i", "j", "k"], {"i": [], "j": ["i"], "k": []}, ["i", "k"]),
        # Keys missing from parent map should be treated as zero parents.
        (["i", "j", "k"], {"j": ["i"]}, ["i", "k"]),
        # All have parents
        (["a", "b"], {"a": ["b"], "b": ["a"]}, []),
        # Empty input
        ([], {}, []),
        # Order preserved among reducible indices
        (["x", "y", "z"], {"y": ["x"]}, ["x", "z"]),
    ],
)
def test_get_reducible_idxs(reduce_idxs, parent_idxs, expected):
    names = set(reduce_idxs)
    names.update(parent_idxs.keys())
    for i in parent_idxs.values():
        names.update(i)

    fields: dict[str, Field] = {x: Field(x) for x in names}
    reduce_fields: list[Field] = [fields[name] for name in reduce_idxs]
    parent_fields: OrderedDict[Field, list[Field]] = OrderedDict(
        (fields[key], [fields[p] for p in parents])
        for key, parents in parent_idxs.items()
    )

    aq = object.__new__(AnnotatedQuery)
    aq.ST = object
    aq.output_name = None
    aq.reduce_idxs = reduce_fields
    aq.point_expr = None
    aq.idx_lowest_root = OrderedDict()
    aq.idx_op = OrderedDict()
    aq.idx_init = OrderedDict()
    aq.parent_idxs = parent_fields
    aq.original_idx = OrderedDict()
    aq.connected_components = []
    aq.connected_idxs = OrderedDict()
    aq.output_order = None
    aq.output_format = None

    result = [field.name for field in get_reducible_idxs(aq)]
    assert result == expected


@pytest.mark.parametrize(
    "parent_idxs, connected_idxs, expected",
    [
        # Single component; order within component follows connected_idxs key order
        (
            {},
            {"a": ["b"], "b": ["a"]},
            [["a", "b"]],
        ),
        # Two components: {a,b} and {c}
        (
            {},
            {"a": ["b"], "b": ["a"], "c": []},
            [["a", "b"], ["c"]],
        ),
        # Parent edge is ignored for connectivity
        (
            {"b": ["a"]},
            {"a": ["b"], "b": ["a"]},
            [["a"], ["b"]],
        ),
        # Ordering across components is enforced
        (
            {"b": ["a"]},
            {"b": [], "a": []},
            [["a"], ["b"]],
        ),
        # Chain of three separate components with parents
        (
            {"b": ["a"], "c": ["b"]},
            {"c": [], "b": [], "a": []},
            [["a"], ["b"], ["c"]],
        ),
        # Single big component
        (
            {"b": ["a"], "c": ["b"]},
            {"a": ["b"], "b": ["a", "c"], "c": ["b"]},
            [["a"], ["b"], ["c"]],
        ),
    ],
)
def test_get_idx_connected_components(parent_idxs, connected_idxs, expected):
    names: set[str] = set(parent_idxs.keys()) | set(connected_idxs.keys())
    for i in parent_idxs.values():
        names.update(i)
    for j in connected_idxs.values():
        names.update(j)

    name = {x: Field(x) for x in names}

    parent_field_idxs: dict[Field, list[Field]] = {
        name[k]: [name[p] for p in v] for k, v in parent_idxs.items()
    }
    connected_field_idxs: dict[Field, list[Field]] = {
        name[k]: [name[n] for n in v] for k, v in connected_idxs.items()
    }

    components = get_idx_connected_components(parent_field_idxs, connected_field_idxs)
    result = [[field.name for field in comp] for comp in components]

    assert result == expected


@pytest.mark.parametrize(
    "expr,node_to_replace,new_node,nodes_to_remove,expected_names",
    [
        (
            MapJoin(
                Literal("op"),
                (
                    Table(Literal("a"), (Field("a"),)),
                    Table(Literal("b"), (Field("b"),)),
                    Table(Literal("c"), (Field("c"),)),
                ),
            ),
            Table(Literal("b"), (Field("b"),)),
            Table(Literal("a"), (Field("a"),)),
            set(),
            ["a", "a", "c"],
        ),
        (
            MapJoin(
                Literal("op"),
                (
                    Table(Literal("a"), (Field("a"),)),
                    Table(Literal("b"), (Field("b"),)),
                    Table(Literal("c"), (Field("c"),)),
                ),
            ),
            Table(Literal("b"), (Field("b"),)),
            Table(Literal("a"), (Field("a"),)),
            {Table(Literal("c"), (Field("c"),))},
            ["a", "a"],
        ),
        (
            MapJoin(
                Literal("op"),
                (
                    Table(Literal("a"), (Field("a"),)),
                    Table(Literal("b"), (Field("b"),)),
                    Table(Literal("c"), (Field("c"),)),
                ),
            ),
            Table(Literal("c"), (Field("c"),)),
            Table(Literal("a"), (Field("a"),)),
            {Table(Literal("c"), (Field("c"),))},
            ["a", "b"],
        ),
        (
            MapJoin(
                Literal("op"),
                (
                    Table(Literal("a"), (Field("a"),)),
                    Table(Literal("b"), (Field("b"),)),
                    Table(Literal("c"), (Field("c"),)),
                ),
            ),
            Table(Literal("b"), (Field("b"),)),
            Table(Literal("a"), (Field("a"),)),
            {Table(Literal("b"), (Field("b"),))},
            ["a", "c"],
        ),
        (
            MapJoin(
                Literal("op"),
                (
                    Table(Literal("a"), (Field("a"),)),
                    Table(Literal("b"), (Field("b"),)),
                    Table(Literal("c"), (Field("c"),)),
                ),
            ),
            Table(Literal("c"), (Field("c"),)),
            Table(Literal("a"), (Field("a"),)),
            set(),
            ["a", "b", "a"],
        ),
    ],
)
def test_replace_and_remove_nodes(
    expr,
    node_to_replace,
    new_node,
    nodes_to_remove,
    expected_names,
):
    out = replace_and_remove_nodes(
        expr=expr,
        node_to_replace=node_to_replace,
        new_node=new_node,
        nodes_to_remove=nodes_to_remove,
    )

    result = [tbl.idxs[0].name for tbl in out.args]
    assert result == expected_names


@pytest.mark.parametrize(
    "root, idx_name, expected",
    [
        # Distributive case:
        # root = MapJoin(mul, [A(i), B(j)]), reduce over j → [B]
        (
            MapJoin(
                Literal(op.mul),
                (
                    Table(Literal("A"), (Field("i"),)),
                    Table(Literal("B"), (Field("j"),)),
                ),
            ),
            "j",
            ["B"],
        ),
        # Split-push case:
        # root = MapJoin(add, [A(i), B(i), C(j)]), reduce over i → [C, A, B]
        (
            MapJoin(
                Literal(op.add),
                (
                    Table(Literal("A"), (Field("i"),)),
                    Table(Literal("B"), (Field("i"),)),
                    Table(Literal("C"), (Field("j"),)),
                ),
            ),
            "i",
            ["C", "A", "B"],
        ),
        # Leaf case:
        # root = Table(A(i)), reduce over i → [A]
        (
            Table(Literal("A"), (Field("i"),)),
            "i",
            ["A"],
        ),
        # Nested case:
        # root = MapJoin(mul, [A(i,j), B(j)]), reduce over i → [A]
        (
            MapJoin(
                Literal(op.mul),
                (
                    Table(Literal("A"), (Field("i"), Field("j"))),
                    Table(Literal("B"), (Field("j"),)),
                ),
            ),
            "i",
            ["A"],
        ),
        # Special case: max(C(i), D(j)), reduce over i → [max(C,D)]
        (
            MapJoin(
                Literal(max),
                (
                    Table(Literal("C"), (Field("i"),)),
                    Table(Literal("D"), (Field("j"),)),
                ),
            ),
            "i",
            [
                MapJoin(
                    Literal(max),
                    (
                        Table(Literal("C"), (Field("i"),)),
                        Table(Literal("D"), (Field("j"),)),
                    ),
                )
            ],
        ),
        # root = MapJoin(mul, [A(j), MapJoin(max, [B(i), C(j)])]), reduce over i
        (
            MapJoin(
                Literal(op.mul),
                (
                    Table(Literal("A"), (Field("j"),)),
                    MapJoin(
                        Literal(max),
                        (
                            Table(Literal("B"), (Field("i"),)),
                            Table(Literal("C"), (Field("j"),)),
                        ),
                    ),
                ),
            ),
            "i",
            [
                MapJoin(
                    Literal(max),
                    (
                        Table(Literal("B"), (Field("i"),)),
                        Table(Literal("C"), (Field("j"),)),
                    ),
                )
            ],
        ),
    ],
)
def test_find_lowest_roots(root, idx_name, expected):
    roots = find_lowest_roots(Literal(op.add), Field(idx_name), root)

    # Special-case: the max(C(i), D(j)) example – we expect the MapJoin itself.
    if expected and not isinstance(expected[0], str):
        assert roots == expected
    else:
        # All other cases:
        result: list[str] = []
        for node in roots:
            assert isinstance(node, Table)
            assert isinstance(node.tns, Literal)
            result.append(node.tns.val)

        assert result == expected


@pytest.mark.parametrize(
    "expr_func, expected_dim_sizes, expected_index_order, expected_fill_value, "
    "expected_non_fill",
    [
        # Base MapJoin: C = A + B
        (
            lambda A, B: A + B,
            {"i": 2, "j": 3},
            ("i", "j"),
            0.0,
            6.0,
        ),
        # Aggregate: D = C.sum(axis=0)
        (
            lambda A, B: finchlite.sum(A + B, axis=0),
            {"j": 3},
            ("j",),
            0.0,
            3.0,
        ),
        # Combination : F = ((A + B) * 3).sum(axis=1)
        (
            lambda A, B: finchlite.sum((A + B) * 3, axis=1),
            {"i": 2},
            ("i",),
            0.0,
            2.0,
        ),
    ],
)
def test_lazy_tensor_stats_parametrized(
    expr_func,
    expected_dim_sizes,
    expected_index_order,
    expected_fill_value,
    expected_non_fill,
):
    arr1 = np.zeros((2, 3))
    arr2 = np.ones((2, 3))
    A = lazy(arr1)
    B = lazy(arr2)

    expr = expr_func(A, B)
    stats = get_lazy_tensor_stats(expr, DenseStats)

    assert isinstance(stats, DenseStats)
    #The names of the indices are changed hence setting them back to the expected ones - Check
    if expected_index_order:
            stats = DenseStats.relabel(stats, expected_index_order)
    assert stats.dim_sizes == expected_dim_sizes
    assert stats.index_order == expected_index_order
    assert stats.fill_value == expected_fill_value
    assert stats.estimate_non_fill_values() == expected_non_fill
