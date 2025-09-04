from operator import add, mul

import pytest

import numpy as np

from finchlite.galley.dc_stats import DC, DCStats
from finchlite.galley.dense_stat import DenseStats
from finchlite.galley.tensor_def import TensorDef

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


# ─────────────────────────────── DenseStats tests ─────────────────────────────


def test_from_tensor_and_getters():
    arr = np.zeros((2, 3))
    ds = DenseStats(arr, ["i", "j"])

    assert ds.index_set == {"i", "j"}
    assert ds.get_dim_size("i") == 2.0
    assert ds.get_dim_size("j") == 3.0
    assert ds.fill_value == 0


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((2, 3), 6.0),
        ((4, 5, 6), 120.0),
        ((1,), 1.0),
    ],
)
def test_estimate_non_fill_values(shape, expected):
    arr = np.zeros(shape)
    ds = DenseStats(arr, [f"x{i}" for i in range(len(shape))])
    assert ds.estimate_non_fill_values() == expected


def test_mapjoin_mul_and_add():
    A = np.ones((2, 3))
    B = np.ones((3, 4))
    dsa = DenseStats(A, ["i", "j"])
    dsb = DenseStats(B, ["j", "k"])

    dsm = DenseStats.mapjoin(mul, dsa, dsb)
    assert dsm.index_set == {"i", "j", "k"}
    assert dsm.get_dim_size("i") == 2.0
    assert dsm.get_dim_size("j") == 3.0
    assert dsm.get_dim_size("k") == 4.0
    assert dsm.fill_value == 0.0

    dsa2 = DenseStats(2 * A, ["i", "j"])
    ds_sum = DenseStats.mapjoin(add, dsa, dsa2)
    assert ds_sum.fill_value == 1 + 2


def test_aggregate_and_issimilar():
    A = np.ones((2, 3))
    dsa = DenseStats(A, ["i", "j"])

    ds_agg = DenseStats.aggregate(sum, ["j"], dsa)
    assert ds_agg.index_set == {"i"}
    assert ds_agg.get_dim_size("i") == 2.0
    assert ds_agg.fill_value == dsa.fill_value
    assert DenseStats.issimilar(dsa, dsa)
    B = np.ones((3, 4))
    dsb = DenseStats.from_tensor(B, ["j", "k"])
    assert not DenseStats.issimilar(dsa, dsb)


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
    stats = DCStats(tensor, fields)
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
    stats = DCStats(tensor, fields)
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
    stats = DCStats(tensor, fields)
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
    stats = DCStats(tensor, fields)
    assert stats.dcs == expected_dcs


@pytest.mark.parametrize(
    "tensor, fields",
    [
        (np.array([0, 1, 0, 1, 1], dtype=int), ["i"]),
        (np.zeros((5,), dtype=int), ["i"]),
        (np.ones((5,), dtype=int), ["i"]),
        (np.array([[1, 0, 1], [0, 0, 0], [1, 1, 0]], dtype=int), ["i", "j"]),
        (np.zeros((3, 3), dtype=int), ["i", "j"]),
        (np.ones((3, 3), dtype=int), ["i", "j"]),
        (
            np.array(
                [
                    [[1, 0], [0, 0]],
                    [[0, 1], [1, 0]],
                ],
                dtype=int,
            ),
            ["i", "j", "k"],
        ),
        (np.zeros((2, 2, 2), dtype=int), ["i", "j", "k"]),
        (np.ones((2, 2, 2), dtype=int), ["i", "j", "k"]),
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
        ),
        (np.zeros((2, 3, 1, 4), dtype=int), ["i", "j", "k", "l"]),
        (np.ones((2, 2, 2, 2), dtype=int), ["i", "j", "k", "l"]),
    ],
)
def test_estimate_nnz(tensor, fields):
    stats = DCStats(tensor, fields)
    estimated = stats.estimate_non_fill_values()
    expected = float(np.count_nonzero(tensor))
    assert estimated == expected
