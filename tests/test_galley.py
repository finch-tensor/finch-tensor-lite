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
    stat = DCStats(np.zeros((1, 1), dtype=int), ["i", "j"])
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
    stat = DCStats(np.zeros((1, 1, 1), dtype=int), ["i", "j", "k"])
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
    stat = DCStats(np.zeros((1, 1, 1, 1), dtype=int), ["i", "j", "k", "l"])
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
    stat = DCStats(np.zeros((1, 1, 1), dtype=int), ["i", "j", "k"])
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
    stat = DCStats(np.zeros((1, 1, 1), dtype=int), ["i", "j", "k"])
    stat.tensordef = TensorDef(frozenset(["i", "j", "k"]), dims, 0)
    stat.dcs = set(dcs)
    assert stat.estimate_non_fill_values() == expected_nnz

# @pytest.mark.parametrize(
#     "dims, dcs, expected_nnz",
#     [
#         (
#             {"i": 1000, "j": 1000, "k": 1000},
#             [
#                 DC(frozenset(), frozenset(["i", "j"]), 50),
#                 DC(frozenset(["i"]), frozenset(["j"]), 5),
#                 DC(frozenset(["j"]), frozenset(["i"]), 5),
#                 DC(frozenset(), frozenset(["j", "k"]), 50),
#                 DC(frozenset(["j"]), frozenset(["k"]), 5),
#                 DC(frozenset(["k"]), frozenset(["j"]), 5),
#                 DC(frozenset(), frozenset(["i", "k"]), 50),
#                 DC(frozenset(["i"]), frozenset(["k"]), 5),
#                 DC(frozenset(["k"]), frozenset(["i"]), 5),
#             ],
#             1,
#         ),
#     ],
# )
# def test_full_reduce_dc_card(dims, dcs, expected_nnz):
#     stat = DCStats(np.zeros((1, 1, 1), dtype=int), ["i", "j", "k"])
#     stat.tensordef = TensorDef(frozenset(["i", "j", "k"]), dims, 0)
#     stat.dcs = set(dcs)
#     reduce_stats = aggregate(operator.add, 0.0, {"i", "j", "k"}, stat)
#     assert reduce_stats.estimate_non_fill_values() == expected_nnz

# @pytest.mark.parametrize(
#     "dims, dcs, expected_nnz",
#     [
#         (
#             {"i": 1000, "j": 1000, "k": 1000},
#             [
#                 DC(frozenset(), frozenset(["i", "j"]), 1),
#                 DC(frozenset(["i"]), frozenset(["j"]), 1),
#                 DC(frozenset(["j"]), frozenset(["i"]), 1),
#                 DC(frozenset(), frozenset(["j", "k"]), 50),
#                 DC(frozenset(["j"]), frozenset(["k"]), 5),
#                 DC(frozenset(["k"]), frozenset(["j"]), 5),
#                 DC(frozenset(), frozenset(["i", "k"]), 50),
#                 DC(frozenset(["i"]), frozenset(["k"]), 5),
#                 DC(frozenset(["k"]), frozenset(["i"]), 5),
#             ],
#             5,
#         ),
#     ],
# )
# def test_2_attr_reduce_dc_card(dims, dcs, expected_nnz):
#     stat = DCStats(np.zeros((1, 1, 1), dtype=int), ["i", "j", "k"])
#     stat.tensordef = TensorDef(frozenset(["i", "j", "k"]), dims, 0)
#     stat.dcs = set(dcs)
#     reduce_stats = aggregate(operator.add, 0.0, {"i", "j", "k"}, stat)
#     assert reduce_stats.estimate_non_fill_values() == expected_nnz

# @pytest.mark.parametrize(
#     "dims, dcs1, dcs2, expected_nnz",
#     [
#         (
#             {"i": 1000},
#             [
#                 DC(frozenset(), frozenset(["i"]), 1),
#             ],
#             [
#                 DC(frozenset(), frozenset(["i"]), 1),
#             ],
#             2,
#         ),
#     ],
# )
# def test_1d_disjunction_dc_card(dims, dcs1, dcs2, expected_nnz):
#     stat_1 = DCStats(np.zeros((1), dtype=int), ["i"])
#     stat_1.tensordef = TensorDef(frozenset(["i"]), dims, 0)
#     stat_1.dcs = set(dcs1)

#     stat_2 = DCStats(np.zeros((1), dtype=int), ["i"])
#     stat_2.tensordef = TensorDef(frozenset(["i"]), dims, 0)
#     stat_2.dcs = set(dcs2)
#     reduce_stats = mapjoin(operator.add, stat_1, stat_2)
#     assert reduce_stats.estimate_non_fill_values() == expected_nnz

# @pytest.mark.parametrize(
#     "dims, dcs1, dcs2, expected_nnz",
#     [
#         (
#             {"i": 1000, "j": 100},
#             [
#                 DC(frozenset(), frozenset([1, 2]), 1),
#             ],
#             [
#                 DC(frozenset(), frozenset([1, 2]), 1),
#             ],
#             2,
#         ),
#     ],
# )
# def test_2d_disjunction_dc_card(dims, dcs1, dcs2, expected_nnz):
#     stat_1 = DCStats(np.zeros((1, 1), dtype=int), ["i", "j"])
#     stat_1.tensordef = TensorDef(frozenset(["i"]), dims, 0)
#     stat_1.dcs = set(dcs1)

#     stat_2 = DCStats(np.zeros((1, 1), dtype=int), ["i"])
#     stat_2.tensordef = TensorDef(frozenset(["i", "j"]), dims, 0)
#     stat_2.dcs = set(dcs2)
#     reduce_stats = mapjoin(operator.add, stat_1, stat_2)
#     assert reduce_stats.estimate_non_fill_values() == expected_nnz

# @pytest.mark.parametrize(
#     "dims, dcs1, dims2, dcs2, expected_nnz",
#     [
#         (
#             {"i": 1000},
#             [
#                 DC(frozenset(), frozenset([1]), 5),
#             ],
#             {"j": 100},
#             [
#                 DC(frozenset(), frozenset([2]), 10),
#             ],
#             10 * 1000 + 5 * 100,
#         ),
#     ],
# )
# def test_2d_disjoint_disjunction_dc_card(dims, dcs1, dcs2, expected_nnz):
#     stat_1 = DCStats(np.zeros((1, 1), dtype=int), ["i", "j"])
#     stat_1.tensordef = TensorDef(frozenset(["i"]), dims, 0)
#     stat_1.dcs = set(dcs1)

#     stat_2 = DCStats(np.zeros((1, 1), dtype=int), ["i"])
#     stat_2.tensordef = TensorDef(frozenset(["i", "j"]), dims, 0)
#     stat_2.dcs = set(dcs2)
#     reduce_stats = mapjoin(operator.add, stat_1, stat_2)
#     assert reduce_stats.estimate_non_fill_values() == expected_nnz

# @pytest.mark.parametrize(
#     "dims1, dcs1, dims2, dcs2, expected_nnz",
#     [
#         (
#             {"i": 1000, "j": 100},
#             {DC(frozenset(), frozenset(["i", "j"]), 5.0)},
#             {"j": 100, "k": 1000},
#             {DC(frozenset(), frozenset(["j", "k"]), 10.0)},
#             10 * 1000 + 5 * 1000,
#         ),
#     ],
# )
# def test_3d_disjoint_disjunction_dc_card(dims1, dims2, dcs1, dcs2, expected_nnz):
#     stat_1 = DCStats(np.zeros((1, 1), dtype=int), ["i", "j"])
#     stat_1.tensordef = TensorDef(frozenset(["i", "j"]), dims1, 0)
#     stat_1.dcs = set(dcs1)

#     stat_2 = DCStats(np.zeros((1, 1), dtype=int), ["j", "k"])
#     stat_2.tensordef = TensorDef(frozenset(["j", "k"]), dims2, 0)
#     stat_2.dcs = set(dcs2)
#     reduce_stats = mapjoin(operator.add, stat_1, stat_2)
#     assert reduce_stats.estimate_non_fill_values() == expected_nnz

        @testset "Mixture Disjunction Conjunction DC Card" begin
            dims1 = Dict(:i => 1000, :j => 100)
            def1 = TensorDef(StableSet([:i, :j]), dims1, 1, nothing, nothing, nothing)
            dcs1 = StableSet([DC(StableSet{Int}(), StableSet([1, 2]), 5)])
            stat1 = DCStats(def1, idx_2_int, int_2_idx, dcs1)

            idx_2_int = Dict(:i => 1, :j => 2)
            int_2_idx = Dict(1 => :i, 2 => :j)
            dims2 = Dict(:j => 100, :k => 1000)
            def2 = TensorDef(StableSet([:j, :k]), dims2, 1, nothing, nothing, nothing)
            idx_2_int = Dict(:j => 2, :k => 3)
            int_2_idx = Dict(2 => :j, 3 => :k)
            dcs2 = StableSet([DC(StableSet{Int}(), StableSet([2, 3]), 10)])
            stat2 = DCStats(def2, idx_2_int, int_2_idx, dcs2)

            dims3 = Dict(:i => 1000, :j => 100, :k => 1000)
            idx_2_int = Dict(:i => 1, :j => 2, :k => 3)
            int_2_idx = Dict(1 => :i, 2 => :j, 3 => :k)
            def3 = TensorDef(
                StableSet([:i, :j, :k]), dims3, 0.0, nothing, nothing, nothing
            )
            dcs3 = StableSet([DC(StableSet{Int}(), StableSet([1, 2, 3]), 10)])
            stat3 = DCStats(def3, idx_2_int, int_2_idx, dcs3)

            merge_stats = merge_tensor_stats(*, stat1, stat2, stat3)
            @test estimate_nnz(merge_stats) == 10
        end
@pytest.mark.parametrize(
    "dims1, dcs1, dims2, dcs2, dims3, dcs3 expected_nnz",
    [
        (
            {"i": 1000, "j": 100},
            {DC(frozenset(), frozenset(["i", "j"]), 5.0)},
            {"j": 100, "k": 1000},
            {DC(frozenset(), frozenset(["j", "k"]), 10.0)},
            10,
        ),
    ],
)
def test_mixture_disjunction_conjunction_dc_card(dims1, dims2, dcs1, dcs2, expected_nnz):
    stat_1 = DCStats(np.zeros((1, 1), dtype=int), ["i", "j"])
    stat_1.tensordef = TensorDef(frozenset(["i", "j"]), dims1, 0)
    stat_1.dcs = set(dcs1)

    stat_2 = DCStats(np.zeros((1, 1), dtype=int), ["j", "k"])
    stat_2.tensordef = TensorDef(frozenset(["j", "k"]), dims2, 0)
    stat_2.dcs = set(dcs2)
    reduce_stats = mapjoin(operator.mul, stat_1, stat_2)
    assert reduce_stats.estimate_non_fill_values() == expected_nnz
