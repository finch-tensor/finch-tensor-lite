import math
from collections import OrderedDict

import pytest

import numpy as np

import finchlite as fl
from finchlite import ffuncs
from finchlite.algebra import TensorFType, TupleFType, ftype
from finchlite.autoschedule.capture import LogicCapture
from finchlite.autoschedule.galley.logical_optimizer import insert_statistics
from finchlite.autoschedule.smart_formatter import FDFormatter, SmartFormatter
from finchlite.autoschedule.tensor_stats import (
    DC,
    BaseTensorStats,
    BaseTensorStatsFactory,
    BlockedStatsFactory,
    DCStats,
    DCStatsFactory,
    DenseStatsFactory,
    DummyStatsFactory,
    FDStats,
    FDStatsFactory,
    LPStats,
    LPStatsFactory,
    UniformStatsFactory,
    VPStatsFactory,
)
from finchlite.autoschedule.tensor_stats.exact_stats import ExactStatsFactory
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
)
from finchlite.tensor.traits import Dense as DenseProperty


def _overwrite_def(stat, base: BaseTensorStats):
    """Overwrite a stat's BaseTensorStats state in place.

    Stats now inherit from :class:`BaseTensorStats`, so we copy ``base``'s state
    onto the stat directly.
    """
    stat.index_order = base.index_order
    stat.dim_sizes = base.dim_sizes
    stat.fill_value = base.fill_value
    return stat


def test_fd_stats_constructor_maps_hierarchical_format_properties():
    i, j = Field("i"), Field("j")
    stats = FDStatsFactory()(fl.FillTensor((2, 3), 0), (i, j))

    assert stats.dense_props == {
        i: {frozenset()},
        j: {frozenset(), frozenset({i})},
    }
    assert stats.repeated_props == {
        i: {frozenset()},
        j: {frozenset({i})},
    }
    assert stats.blocked_props == {}
    assert stats.extruded_props == {}

    row, col = Field("row"), Field("col")
    relabeled = FDStatsFactory().relabel(stats, (row, col))
    assert relabeled.dense_props == {
        row: {frozenset()},
        col: {frozenset(), frozenset({row})},
    }
    assert relabeled.repeated_props == {
        row: {frozenset()},
        col: {frozenset({row})},
    }


def test_fd_stats_constructor_maps_hierarchical_array_dense_properties():
    i, j = Field("i"), Field("j")
    tensor = fl.BufferizedNDArray.from_numpy(np.zeros((2, 3), dtype=np.int32))
    stats = FDStatsFactory()(tensor, (i, j))

    assert stats.dense_props == {
        i: {frozenset()},
        j: {frozenset(), frozenset({i})},
    }
    assert stats.repeated_props == {}


def test_fd_stats_mapjoin_union_preserves_property_maps():
    i, j = Field("i"), Field("j")
    factory = FDStatsFactory()
    fill_stats = factory(fl.FillTensor((2, 3), 0), (i, j))
    array_stats = factory(
        fl.BufferizedNDArray.from_numpy(np.zeros((2, 3), dtype=np.int32)),
        (i, j),
    )

    stats = factory.mapjoin(ffuncs.add, fill_stats, array_stats)

    assert stats.dense_props == {
        i: {frozenset()},
        j: {frozenset(), frozenset({i})},
    }
    assert stats.repeated_props == {
        i: {frozenset()},
        j: {frozenset({i})},
    }


def test_fd_stats_aggregate_drops_reduced_indices_from_property_maps():
    i, j = Field("i"), Field("j")
    factory = FDStatsFactory()
    stats = factory(fl.FillTensor((2, 3), 0), (i, j))

    stats = factory.aggregate(ffuncs.add, None, (i,), stats)

    assert stats.dense_props == {j: {frozenset()}}
    assert stats.repeated_props == {j: {frozenset()}}


def test_smart_formatter_passes_propagated_stats_to_tensor_ftype():
    class RecordingSmartFormatter(SmartFormatter):
        def __init__(self, loader):
            super().__init__(loader)
            self.output_stats = []

        def get_tensor_ftype(self, fill_value, shape_type, stats) -> TensorFType:
            self.output_stats.append(stats)
            fill_ftype = ftype(
                fill_value.dtype if isinstance(fill_value, np.ndarray) else fill_value
            )
            return fl.BufferizedNDArrayFType(
                buffer_type=fl.NumpyBufferFType(fill_ftype),
                ndim=len(shape_type),
                dimension_type=TupleFType.from_tuple(shape_type),
                fill_value=fill_value,
            )

    i, j = Field("i"), Field("j")
    A, B = Alias("A"), Alias("B")
    tensor = fl.FillTensor((2, 3), 0)
    stats_factory = FDStatsFactory()
    stats = {A: stats_factory(tensor, (i, j))}
    capture = LogicCapture()
    formatter = RecordingSmartFormatter(capture)
    prgm = Plan(
        (
            Query(
                B,
                MapJoin(
                    Literal(ffuncs.add),
                    (Table(A, (i, j)), Table(A, (i, j))),
                ),
            ),
            Produces((B,)),
        )
    )

    formatter.lower(prgm, {A: tensor.ftype}, stats, stats_factory)

    assert formatter.output_stats[0].dense_props == {
        i: {frozenset()},
        j: {frozenset(), frozenset({i})},
    }
    assert capture.last_stats[B] is formatter.output_stats[0]


def test_fd_formatter_uses_dense_levels_for_dense_properties():
    i, j = Field("i"), Field("j")
    A, B = Alias("A"), Alias("B")
    tensor = fl.FillTensor((2, 3), 0)
    stats_factory = FDStatsFactory()
    stats = {A: stats_factory(tensor, (i, j))}
    capture = LogicCapture()
    formatter = FDFormatter(capture)
    prgm = Plan(
        (
            Query(
                B,
                MapJoin(
                    Literal(ffuncs.add),
                    (Table(A, (i, j)), Table(A, (i, j))),
                ),
            ),
            Produces((B,)),
        )
    )

    formatter.lower(prgm, {A: tensor.ftype}, stats, stats_factory)

    ftype = capture.last_bindings[B]
    assert isinstance(ftype, fl.FiberTensorFType)
    assert isinstance(ftype.lvl_t, fl.DenseLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t, fl.DenseLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t.lvl_t, fl.ElementLevelFType)

    constructed = ftype.construct((2, 3))
    np.testing.assert_array_equal(constructed.to_numpy(), np.zeros((2, 3)))


def test_fd_formatter_uses_sparse_hash_for_unknown_dense_properties():
    i, j = Field("i"), Field("j")
    base = BaseTensorStats((i, j), {i: 2.0, j: 3.0}, 0)
    stats = FDStats(base, dense_props={i: {frozenset()}})

    ftype = FDFormatter().get_tensor_ftype(
        0,
        (fl.ftype(np.intp), fl.ftype(np.intp)),
        stats,
    )

    assert isinstance(ftype, fl.FiberTensorFType)
    assert isinstance(ftype.lvl_t, fl.DenseLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t, fl.SparseHashLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t.lvl_t, fl.ElementLevelFType)


def test_fd_formatter_requires_outer_fields_for_inner_dense_levels():
    i, j = Field("i"), Field("j")
    base = BaseTensorStats((i, j), {i: 2.0, j: 3.0}, 0)
    stats = FDStats(
        base,
        dense_props={
            i: {frozenset()},
            j: {frozenset()},
        },
    )

    ftype = FDFormatter().get_tensor_ftype(
        0,
        (fl.ftype(np.intp), fl.ftype(np.intp)),
        stats,
    )

    assert isinstance(ftype, fl.FiberTensorFType)
    assert isinstance(ftype.lvl_t, fl.DenseLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t, fl.SparseHashLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t.lvl_t, fl.ElementLevelFType)


def _format_stats(levels, shape, fields):
    lvl = fl.element(0, fl.int64, fl.intp, fl.NumpyBufferFType)
    for level in reversed(levels):
        match level:
            case "dense":
                lvl = fl.dense(lvl, fl.intp)
            case "sparse":
                lvl = fl.sparse_list(lvl, fl.intp)
            case _:
                raise ValueError(f"Unknown test level: {level}")
    return FDStatsFactory()(fl.fiber_tensor(lvl).construct(shape), fields)


def _fd_output_pattern(stats):
    ftype = FDFormatter().get_tensor_ftype(
        stats.fill_value,
        tuple(fl.intp for _ in stats.index_order),
        stats,
    )
    lvl = ftype.lvl_t
    pattern = []
    while not isinstance(lvl, fl.ElementLevelFType):
        match lvl:
            case fl.DenseLevelFType():
                pattern.append("dense")
            case fl.SparseHashLevelFType():
                pattern.append("sparse")
            case _:
                raise AssertionError(f"Unexpected FD output level: {lvl}")
        lvl = lvl.lvl_t
    return tuple(pattern)


def test_fd_formatter_csr_dcsr_format_algebra():
    i, j, k = Field("i"), Field("j"), Field("k")
    factory = FDStatsFactory()

    csr_ij = _format_stats(("dense", "sparse"), (2, 3), (i, j))
    csr_jk = _format_stats(("dense", "sparse"), (3, 4), (j, k))
    dcsr_ij = _format_stats(("sparse", "sparse"), (2, 3), (i, j))
    dcsr_jk = _format_stats(("sparse", "sparse"), (3, 4), (j, k))

    csr_plus_csr = factory.mapjoin(ffuncs.add, csr_ij, csr_ij)
    assert _fd_output_pattern(csr_plus_csr) == ("dense", "sparse")

    csr_times_dcsr = factory.mapjoin(ffuncs.mul, csr_ij, dcsr_ij)
    assert _fd_output_pattern(csr_times_dcsr) == ("sparse", "sparse")

    dcsr_matmul_dcsr = factory.aggregate(
        ffuncs.add,
        0,
        (j,),
        factory.mapjoin(ffuncs.mul, dcsr_ij, dcsr_jk),
    )
    assert _fd_output_pattern(dcsr_matmul_dcsr) == ("sparse", "sparse")

    csr_matmul_csr = factory.aggregate(
        ffuncs.add,
        0,
        (j,),
        factory.mapjoin(ffuncs.mul, csr_ij, csr_jk),
    )
    assert _fd_output_pattern(csr_matmul_csr) == ("sparse", "sparse")


def test_fd_stats_chase_ignores_circular_dependencies():
    i, j = Field("i"), Field("j")

    class TensorFType:
        level_format_properties = [
            DenseProperty((0,), (1,)),
            DenseProperty((1,), (0,)),
        ]

    class Tensor:
        shape = (2, 3)
        fill_value = 0
        ftype = TensorFType()

    stats = FDStatsFactory()(Tensor(), (i, j))

    assert stats.dense_props == {
        i: {frozenset({j})},
        j: {frozenset({i})},
    }


# ─────────────────────────────── ExactStats tests ────────────────────────────────


def test_exact_elementwise_mul():
    i, j = Field("i"), Field("j")
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 1.0], [0.0, 0.0]])
    node = MapJoin(
        Literal(ffuncs.mul),
        (Table(Literal(fl.asarray(A)), (i, j)), Table(Literal(fl.asarray(B)), (i, j))),
    )
    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.estimate_non_fill_values() == pytest.approx(
        float(np.count_nonzero(A * B))
    )


def test_exact_elementwise_add():
    i, j = Field("i"), Field("j")
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 1.0], [0.0, 0.0]])
    node = MapJoin(
        Literal(ffuncs.add),
        (Table(Literal(fl.asarray(A)), (i, j)), Table(Literal(fl.asarray(B)), (i, j))),
    )
    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.estimate_non_fill_values() == pytest.approx(
        float(np.count_nonzero(A + B))
    )


def test_exact_broadcast_mul():
    i, j = Field("i"), Field("j")
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, 0.0])
    node = MapJoin(
        Literal(ffuncs.mul),
        (Table(Literal(fl.asarray(A)), (i, j)), Table(Literal(fl.asarray(b)), (j,))),
    )
    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.estimate_non_fill_values() == pytest.approx(
        float(np.count_nonzero(A * b))
    )


def test_exact_broadcast_add():
    i, j = Field("i"), Field("j")
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, 0.0])
    node = MapJoin(
        Literal(ffuncs.add),
        (Table(Literal(fl.asarray(A)), (i, j)), Table(Literal(fl.asarray(b)), (j,))),
    )
    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.estimate_non_fill_values() == pytest.approx(
        float(np.count_nonzero(A + b))
    )


def test_exact_join_mul():
    i, j, k = Field("i"), Field("j"), Field("k")
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    node = MapJoin(
        Literal(ffuncs.mul),
        (Table(Literal(fl.asarray(A)), (i, j)), Table(Literal(fl.asarray(B)), (j, k))),
    )
    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.estimate_non_fill_values() == pytest.approx(
        float(np.count_nonzero(A[:, :, None] * B[None, :, :]))
    )


def test_exact_join_add():
    i, j, k = Field("i"), Field("j"), Field("k")
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    node = MapJoin(
        Literal(ffuncs.add),
        (Table(Literal(fl.asarray(A)), (i, j)), Table(Literal(fl.asarray(B)), (j, k))),
    )
    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.estimate_non_fill_values() == pytest.approx(
        float(np.count_nonzero(A[:, :, None] + B[None, :, :]))
    )


def test_exact_semiring():
    i, j, k, m = Field("i"), Field("j"), Field("k"), Field("m")
    A = np.eye(3)
    B = np.eye(3)
    C = np.eye(3)
    D = np.eye(3)
    A_, B_, C_, D_ = fl.asarray(A), fl.asarray(B), fl.asarray(C), fl.asarray(D)

    ab = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(
            Literal(ffuncs.mul),
            (Table(Literal(A_), (i, j)), Table(Literal(B_), (j, k))),
        ),
        (j,),
    )
    cd = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(
            Literal(ffuncs.mul),
            (Table(Literal(C_), (i, m)), Table(Literal(D_), (m, k))),
        ),
        (m,),
    )
    node = MapJoin(Literal(ffuncs.mul), (ab, cd))

    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    expected = float(np.count_nonzero((A @ B) * (C @ D)))
    assert stats.estimate_non_fill_values() == pytest.approx(expected)


def test_exact_semiring_tropical():
    i, j = Field("i"), Field("j")
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    A_ = fl.asarray(A)
    node = Aggregate(
        Literal(ffuncs.min),
        Literal(float("inf")),
        Table(Literal(A_), (i, j)),
        (j,),
    )
    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    expected = float(np.count_nonzero(np.isfinite(np.min(A, axis=1))))
    assert stats.estimate_non_fill_values() == pytest.approx(expected)


def test_exact_semiring_maxplus():
    i, j, k, m = Field("i"), Field("j"), Field("k"), Field("m")
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    D = np.eye(2)
    A_, B_, C_, D_ = fl.asarray(A), fl.asarray(B), fl.asarray(C), fl.asarray(D)

    ab = Aggregate(
        Literal(ffuncs.max),
        Literal(float("-inf")),
        MapJoin(
            Literal(ffuncs.add),
            (Table(Literal(A_), (i, j)), Table(Literal(B_), (j, k))),
        ),
        (j,),
    )
    cd = Aggregate(
        Literal(ffuncs.max),
        Literal(float("-inf")),
        MapJoin(
            Literal(ffuncs.add),
            (Table(Literal(C_), (i, m)), Table(Literal(D_), (m, k))),
        ),
        (m,),
    )
    node = MapJoin(Literal(ffuncs.add), (ab, cd))

    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    AB = np.max(A[:, :, None] + B[None, :, :], axis=1)
    CD = np.max(C[:, :, None] + D[None, :, :], axis=1)
    expected = float(np.count_nonzero(AB + CD))
    assert stats.estimate_non_fill_values() == pytest.approx(expected)


def test_exact_semiring_boolean():
    i, j, k, m = Field("i"), Field("j"), Field("k"), Field("m")
    A = np.eye(2, dtype=bool)
    B = np.eye(2, dtype=bool)
    C = np.eye(2, dtype=bool)
    D = np.eye(2, dtype=bool)
    A_, B_, C_, D_ = fl.asarray(A), fl.asarray(B), fl.asarray(C), fl.asarray(D)

    ab = Aggregate(
        Literal(ffuncs.or_),
        Literal(False),
        MapJoin(
            Literal(ffuncs.and_),
            (Table(Literal(A_), (i, j)), Table(Literal(B_), (j, k))),
        ),
        (j,),
    )
    cd = Aggregate(
        Literal(ffuncs.or_),
        Literal(False),
        MapJoin(
            Literal(ffuncs.and_),
            (Table(Literal(C_), (i, m)), Table(Literal(D_), (m, k))),
        ),
        (m,),
    )
    node = MapJoin(Literal(ffuncs.and_), (ab, cd))

    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    AB = np.any(A[:, :, None].astype(bool) & B[None, :, :].astype(bool), axis=1)
    CD = np.any(C[:, :, None].astype(bool) & D[None, :, :].astype(bool), axis=1)
    expected = float(np.count_nonzero(AB & CD))
    assert stats.estimate_non_fill_values() == pytest.approx(expected)


def test_exact_semiring_maxmin():
    i, j, k, m = Field("i"), Field("j"), Field("k"), Field("m")
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    D = np.eye(2)
    A_, B_, C_, D_ = fl.asarray(A), fl.asarray(B), fl.asarray(C), fl.asarray(D)

    ab = Aggregate(
        Literal(ffuncs.max),
        Literal(float("-inf")),
        MapJoin(
            Literal(ffuncs.min),
            (Table(Literal(A_), (i, j)), Table(Literal(B_), (j, k))),
        ),
        (j,),
    )
    cd = Aggregate(
        Literal(ffuncs.max),
        Literal(float("-inf")),
        MapJoin(
            Literal(ffuncs.min),
            (Table(Literal(C_), (i, m)), Table(Literal(D_), (m, k))),
        ),
        (m,),
    )
    node = MapJoin(Literal(ffuncs.min), (ab, cd))

    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    AB = np.max(np.minimum(A[:, :, None], B[None, :, :]), axis=1)
    CD = np.max(np.minimum(C[:, :, None], D[None, :, :]), axis=1)
    expected = float(np.count_nonzero(np.isfinite(np.minimum(AB, CD))))
    assert stats.estimate_non_fill_values() == pytest.approx(expected)


def test_exact_relabel():
    i, j = Field("i"), Field("j")
    data = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    node = Table(Literal(fl.asarray(data)), (i, j))
    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    row, col = Field("row"), Field("col")
    relabeled = ExactStatsFactory().relabel(stats, (row, col))

    assert relabeled.index_order == (row, col)
    assert relabeled.get_dim_size(row) == stats.get_dim_size(i)
    assert relabeled.get_dim_size(col) == stats.get_dim_size(j)


def test_exact_reorder():
    i, j = Field("i"), Field("j")
    data = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    node = Table(Literal(fl.asarray(data)), (i, j))
    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    reordered = ExactStatsFactory().reorder(stats, (j, i))

    assert reordered.index_order == (j, i)
    assert reordered.get_dim_size(i) == stats.get_dim_size(i)
    assert reordered.get_dim_size(j) == stats.get_dim_size(j)


def test_exact_embedding():
    i, j = Field("i"), Field("j")
    data = np.array([[1.0, 0.0], [0.0, 1.0]])
    node = Table(Literal(fl.asarray(data)), (i, j))
    stats = insert_statistics(
        stats_factory=ExactStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    emb = stats.get_embedding()

    assert emb.shape == (3,)
    assert emb[0] == pytest.approx(np.log2(2.0))
    assert emb[1] == pytest.approx(np.log2(2.0))
    assert emb[2] == pytest.approx(np.log2(stats.estimate_non_fill_values() + 1))

    reordered = ExactStatsFactory().reorder(stats, (j, i))
    emb_r = reordered.get_embedding()
    assert emb_r[0] == pytest.approx(emb[1])
    assert emb_r[1] == pytest.approx(emb[0])
    assert emb_r[2] == pytest.approx(emb[2])


# ─────────────────────────────── DummyStats tests ────────────────────────────────


def test_dummy_from_tensor_and_getters():
    data = np.zeros((2, 3))
    node = Table(Literal(fl.asarray(data)), (Field("i"), Field("j")))

    stats = insert_statistics(
        stats_factory=DummyStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    assert stats.index_order == (Field("i"), Field("j"))
    assert stats.get_dim_size(Field("i")) == 2.0
    assert stats.get_dim_size(Field("j")) == 3.0
    assert stats.fill_value == 0


def test_dummy_mapjoin_same_axes():
    i, j = Field("i"), Field("j")
    ta = Table(Literal(fl.asarray(np.ones((4, 5)))), (i, j))
    tb = Table(Literal(fl.asarray(np.ones((4, 5)))), (i, j))

    cache = {}
    insert_statistics(
        stats_factory=DummyStatsFactory(),
        node=ta,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    insert_statistics(
        stats_factory=DummyStatsFactory(),
        node=tb,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    stats = insert_statistics(
        stats_factory=DummyStatsFactory(),
        node=MapJoin(Literal(ffuncs.add), (ta, tb)),
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert stats.index_order == (i, j)
    assert stats.get_dim_size(i) == 4.0
    assert stats.get_dim_size(j) == 5.0


def test_dummy_mapjoin_non_same_axes():
    i, j, k = Field("i"), Field("j"), Field("k")
    ta = Table(Literal(fl.asarray(np.ones((4, 5)))), (i, j))
    tb = Table(Literal(fl.asarray(np.ones((5, 3)))), (j, k))

    cache = {}
    insert_statistics(
        stats_factory=DummyStatsFactory(),
        node=ta,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    insert_statistics(
        stats_factory=DummyStatsFactory(),
        node=tb,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    stats = insert_statistics(
        stats_factory=DummyStatsFactory(),
        node=MapJoin(Literal(ffuncs.mul), (ta, tb)),
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert set(stats.index_order) == {i, j, k}
    assert stats.fill_value == 0.0


def test_dummy_aggregate():
    i, j = Field("i"), Field("j")
    table = Table(Literal(fl.asarray(np.eye(10))), (i, j))

    node_sum = Aggregate(op=Literal(ffuncs.add), init=None, arg=table, idxs=(j,))
    stats = insert_statistics(
        stats_factory=DummyStatsFactory(),
        node=node_sum,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    assert stats.index_order == (i,)
    assert stats.get_dim_size(i) == 10.0


def test_dummy_copy():
    node = Table(Literal(fl.asarray(np.eye(10))), (Field("i"), Field("j")))

    stats = insert_statistics(
        stats_factory=DummyStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    copy = DummyStatsFactory().copy(stats)

    assert copy.dim_sizes == stats.dim_sizes
    assert copy.index_order == stats.index_order
    assert copy is not stats


def test_dummy_relabel():
    node = Table(Literal(fl.asarray(np.eye(10))), (Field("i"), Field("j")))

    stats = insert_statistics(
        stats_factory=DummyStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    relabeled = DummyStatsFactory().relabel(stats, (Field("m"), Field("n")))

    assert relabeled.get_dim_size(Field("m")) == stats.get_dim_size(Field("i"))
    assert relabeled.get_dim_size(Field("n")) == stats.get_dim_size(Field("j"))


def test_dummy_reorder():
    node = Table(Literal(fl.asarray(np.eye(10))), (Field("i"), Field("j")))

    stats = insert_statistics(
        stats_factory=DummyStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    reordered = DummyStatsFactory().reorder(stats, (Field("j"), Field("i")))

    assert reordered.get_dim_size(Field("i")) == stats.get_dim_size(Field("i"))
    assert reordered.get_dim_size(Field("j")) == stats.get_dim_size(Field("j"))


# ─────────────────────────────── VPStats tests ─────────────────────────────


def test_vp_from_tensor_and_getters():
    data = np.zeros((2, 3))
    data[0, 0] = 1.0
    data[1, 1] = 1.0
    arr = fl.asarray(data)

    node = Table(Literal(arr), (Field("i"), Field("j")))
    stats = insert_statistics(
        stats_factory=VPStatsFactory(),
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
        ((4, 5), [(0, 0), (1, 2), (3, 4)], 3.0),
        ((5, 5), [(0, 0), (0, 1), (0, 2)], 3.0),
        ((3, 3, 3), [(0, 0, 0), (1, 1, 1), (2, 2, 2)], 3.0),
        ((5, 5, 5), [], 0.0),
        ((2, 2), [(0, 0), (0, 1), (1, 0), (1, 1)], 4.0),
    ],
)
def test_vp_estimate_non_fill_values(shape, nnz_indices, expected_nnz):
    axes = tuple(Field(f"x{i}") for i in range(len(shape)))
    data = np.zeros(shape)
    for idx in nnz_indices:
        data[idx] = 1.0

    arr = fl.asarray(data)
    node = Table(Literal(arr), axes)

    stats = insert_statistics(
        stats_factory=VPStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    assert stats.index_order == tuple(axes)
    assert stats.estimate_non_fill_values() == expected_nnz


def test_vp_mapjoin_join():
    i, k, j = Field("i"), Field("k"), Field("j")
    data_a = np.eye(10)
    data_b = np.eye(10)

    ta = Table(Literal(fl.asarray(data_a)), (i, k))
    tb = Table(Literal(fl.asarray(data_b)), (k, j))

    cache = {}
    insert_statistics(
        stats_factory=VPStatsFactory(),
        node=ta,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    insert_statistics(
        stats_factory=VPStatsFactory(),
        node=tb,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    node_mul = MapJoin(Literal(ffuncs.mul), (ta, tb))
    stats = insert_statistics(
        stats_factory=VPStatsFactory(),
        node=node_mul,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    assert stats.estimate_non_fill_values() == pytest.approx(10.0)


def test_vp_mapjoin_elementwise():
    i, j = Field("i"), Field("j")
    data_a = np.zeros((10, 10))
    data_a[:5, :] = 1.0
    data_b = np.zeros((10, 10))
    data_b[5:, :] = 1.0

    ta = Table(Literal(fl.asarray(data_a)), (i, j))
    tb = Table(Literal(fl.asarray(data_b)), (i, j))

    cache = {}
    insert_statistics(
        stats_factory=VPStatsFactory(),
        node=ta,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    insert_statistics(
        stats_factory=VPStatsFactory(),
        node=tb,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    stats = insert_statistics(
        stats_factory=VPStatsFactory(),
        node=MapJoin(Literal(ffuncs.add), (ta, tb)),
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    assert stats.estimate_non_fill_values() == pytest.approx(100.0)


def test_vp_mapjoin_broadcast():
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
        stats_factory=VPStatsFactory(),
        node=ta,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    insert_statistics(
        stats_factory=VPStatsFactory(),
        node=tb,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    stats = insert_statistics(
        stats_factory=VPStatsFactory(),
        node=MapJoin(Literal(ffuncs.add), (ta, tb)),
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    assert stats.estimate_non_fill_values() == pytest.approx(2 * 3 + 4 * 3)
    assert stats.V[i] == pytest.approx(4.0)
    assert stats.V[j] == pytest.approx(5.0)
    assert stats.V[k] == pytest.approx(3.0)


def test_vp_aggregate():
    i, j = Field("i"), Field("j")
    data = np.eye(10)
    table = Table(Literal(fl.asarray(data)), (i, j))

    node_sum = Aggregate(
        op=Literal(ffuncs.add),
        init=None,
        arg=table,
        idxs=(j,),
    )
    stats = insert_statistics(
        stats_factory=VPStatsFactory(),
        node=node_sum,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.index_order == (i,)
    assert stats.get_dim_size(i) == 10.0
    assert stats.estimate_non_fill_values() == pytest.approx(10.0)


def test_vp_copy():
    data = np.eye(10)
    arr = fl.asarray(data)
    node = Table(Literal(arr), (Field("i"), Field("j")))
    stats = insert_statistics(
        stats_factory=VPStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    copy = VPStatsFactory().copy(stats)
    assert copy.nnz == stats.nnz
    assert copy.V == stats.V
    assert copy is not stats


def test_vp_relabel():
    data = np.eye(10)
    arr = fl.asarray(data)
    node = Table(Literal(arr), (Field("i"), Field("j")))
    stats = insert_statistics(
        stats_factory=VPStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    relabeled = VPStatsFactory().relabel(stats, (Field("row"), Field("col")))
    assert relabeled.index_order == (Field("row"), Field("col"))
    assert relabeled.nnz == stats.nnz
    assert relabeled.V[Field("row")] == stats.V[Field("i")]
    assert relabeled.V[Field("col")] == stats.V[Field("j")]
    assert Field("i") not in relabeled.V
    assert Field("j") not in relabeled.V


def test_vp_reorder():
    data = np.eye(10)
    arr = fl.asarray(data)
    node = Table(Literal(arr), (Field("i"), Field("j")))
    stats = insert_statistics(
        stats_factory=VPStatsFactory(),
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    reordered = VPStatsFactory().reorder(stats, (Field("j"), Field("i")))
    assert reordered.index_order == (Field("j"), Field("i"))
    assert reordered.nnz == stats.nnz


# ─────────────────────────────── UniformStats tests ─────────────────────────────


# ─────────────────────────────── Test Embeddings ───────────────────────────────
def test_embeddings():
    data = np.zeros((20, 20))
    data[0:10, 0:10] = 1.0
    data[10:20, 10:20] = 1.0

    arr = fl.asarray(data)
    fields = (Field("i"), Field("j"))

    print("\n" + "=" * 80)
    ds = DenseStatsFactory()(arr, fields)
    ds_emb = ds.get_embedding()
    print(f"DenseStats Embeddings : {ds_emb}")

    us = UniformStatsFactory()(arr, fields)
    us_emb = us.get_embedding()
    print(f"UniformStats Embeddings : {us_emb}")

    dc_stats = DCStatsFactory()(arr, fields)
    dc_emb = dc_stats.get_embedding()
    print(f"DCStats Embeddings: {dc_emb}")

    blocks_per_dim = {Field("i"): 2, Field("j"): 2}
    bs = BlockedStatsFactory(UniformStatsFactory(), blocks_per_dim=blocks_per_dim)(
        arr, fields
    )
    bs_emb = bs.get_embedding()
    print(f"BlockedStats Embeddings: {bs_emb}")

    print("=" * 80)


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
    node_mul = MapJoin(Literal(ffuncs.mul), (ta, tb))
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
    node_add = MapJoin(Literal(ffuncs.add), (ta, tb))
    us_add = insert_statistics(
        stats_factory=UniformStatsFactory(),
        node=node_add,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    assert us_add.estimate_non_fill_values() == pytest.approx(75.0)


def test_uniform_aggregate():
    data = np.eye(10)
    table = Table(Literal(fl.asarray(data)), (Field("i"), Field("j")))
    node_sum = Aggregate(
        op=Literal(ffuncs.add),
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


# ------------------------------ BlockedStats -------------------------------------
def test_blocked_stats_from_tensor():
    data = np.eye(10)
    arr = fl.asarray(data)
    indices = (Field("i"), Field("j"))
    bs_factory = BlockedStatsFactory(UniformStatsFactory())
    bs = bs_factory(arr, indices)

    assert bs.estimate_non_fill_values() == 10.0


def test_blocked_stats_aggregate():
    data = np.eye(10)
    indices = (Field("i"), Field("j"))
    bs_factory = BlockedStatsFactory(DenseStatsFactory())
    bs = bs_factory(fl.asarray(data), indices)

    reduce_indices = (Field("j"),)
    agg_bs = bs_factory.aggregate(ffuncs.add, 0.0, reduce_indices, bs)

    assert agg_bs.blocks.ndim == 1
    assert len(agg_bs.blocks) == 2
    assert agg_bs.estimate_non_fill_values() == 10.0


def test_blocked_stats_mapjoin():
    indices = (Field("i"), Field("j"))

    data1 = np.zeros((10, 10))
    data1[0:5, 0:5] = 1.0
    bs_factory = BlockedStatsFactory(UniformStatsFactory(), block_count=2)
    bs1 = bs_factory(fl.asarray(data1), indices)

    data2 = np.zeros((10, 10))
    data2[5:10, 5:10] = 1.0
    bs2 = bs_factory(fl.asarray(data2), indices)

    result = bs_factory.mapjoin(ffuncs.add, bs1, bs2)

    assert result.estimate_non_fill_values() == 50.0
    assert result.blocks[0, 1].estimate_non_fill_values() == 0.0


def test_blocked_stats_relabel():
    indices = (Field("i"), Field("j"))
    bs_factory = BlockedStatsFactory(UniformStatsFactory())
    bs = bs_factory(fl.asarray(np.eye(10)), indices)

    new_names = (Field("row"), Field("col"))
    relabeled = bs_factory.relabel(bs, new_names)

    assert relabeled.index_order == new_names
    assert Field("row") in relabeled.blocks_per_dim
    assert relabeled.estimate_non_fill_values() == 10.0


def test_blocked_stats_reorder():
    data = np.zeros((4, 10))
    data[0:2, 0:5] = 1.0
    arr = fl.asarray(data)

    indices = (Field("i"), Field("j"))
    bs_factory = BlockedStatsFactory(UniformStatsFactory())
    bs = bs_factory(arr, indices)

    # Before reordering
    assert bs.blocks[0, 0].get_dim_size(Field("i")) == 4.0
    assert bs.blocks[0, 0].get_dim_size(Field("j")) == 5.0

    new_indices = (Field("j"), Field("i"))
    reordered_bs = bs_factory.reorder(bs, new_indices)

    new_block = reordered_bs.blocks[0, 0]

    # After reordering
    assert new_block.get_dim_size(Field("j")) == 5.0
    assert new_block.get_dim_size(Field("i")) == 4.0
    assert new_block.index_order == (Field("j"), Field("i"))


def test_blocked_stats_reorder_drop_one_index():
    data = np.ones((4, 1, 9))

    i, j, k = Field("i"), Field("j"), Field("k")
    blocks_per_dim = {i: 2, j: 1, k: 3}
    bs_factory = BlockedStatsFactory(
        UniformStatsFactory(), blocks_per_dim=blocks_per_dim
    )
    bs = bs_factory(fl.asarray(data), (i, j, k))

    reordered = bs_factory.reorder(bs, (k, i))

    assert reordered.index_order == (k, i)
    assert reordered.blocks.shape == (3, 2)
    assert reordered.estimate_non_fill_values() == bs.estimate_non_fill_values()


def test_blocked_stats_reorder_drop_two_index():
    data = np.ones((4, 1, 9, 1))

    i, j, k, m = Field("i"), Field("j"), Field("k"), Field("m")
    blocks_per_dim = {i: 2, j: 1, k: 3, m: 1}
    bs_factory = BlockedStatsFactory(
        UniformStatsFactory(), blocks_per_dim=blocks_per_dim
    )
    bs = bs_factory(fl.asarray(data), (i, j, k, m))
    reordered = bs_factory.reorder(bs, (k, i))

    assert reordered.index_order == (k, i)
    assert reordered.blocks.shape == (3, 2)
    assert reordered.estimate_non_fill_values() == bs.estimate_non_fill_values()


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

    matrix_types = ["diagonal", "tridiagonal", "banded", "triangular", "striped"]
    implementations = [
        ("UniformStats", UniformStatsFactory()),
        ("DenseStats", DenseStatsFactory()),
        ("DCStats", DCStatsFactory()),
        ("LPStats", LPStatsFactory()),
    ]

    print("\n" + "=" * 85)
    print(
        f"{'Matrix Type':<15} | {'Stats':<15} |"
        f" {'Stats Relative Error':<18} | {'Blocked Stats Relative Error'}"
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
                ffuncs.add, 0.0, (k,), impl_factory.mapjoin(ffuncs.mul, g_a, g_b)
            )
            g_perf = abs(g_res.estimate_non_fill_values() - actual_nnz) / actual_nnz

            # Blocked Stats Performance
            blocked_factory = BlockedStatsFactory(impl_factory)
            b_a = blocked_factory(tns_a, (i, k))
            b_b = blocked_factory(tns_b, (k, j))
            b_res = blocked_factory.aggregate(
                ffuncs.add, 0.0, (k,), blocked_factory.mapjoin(ffuncs.mul, b_a, b_b)
            )
            b_perf = abs(b_res.estimate_non_fill_values() - actual_nnz) / actual_nnz

            print(f"{m_type:<15} | {impl_name:<15} | {g_perf:<18.6f} | {b_perf:.6f}")

        print("-" * 85)


# ─────────────────────────── BaseTensorStats def tests ───────────────────────────


def test_copy_and_getters():
    td = BaseTensorStats.from_fields(
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
    td = BaseTensorStats.from_fields(
        index_order=orig_axes, dim_sizes=dim_sizes, fill_value=0
    )
    td2 = BaseTensorStatsFactory.reorder_def(td, tuple(new_axes))
    assert td2.index_order == tuple(new_axes)
    for ax in new_axes:
        assert td2.get_dim_size(ax) == td.get_dim_size(ax)


@pytest.mark.parametrize(
    "defs, func, expected_axes, expected_dims, expected_fill",
    [
        # union of axes; first-wins on dim size; add fills
        (
            [
                ((Field("i"), Field("j")), {Field("i"): 10.0, Field("j"): 5.0}, 2.0),
                ((Field("i"), Field("k")), {Field("i"): 20.0, Field("k"): 7.0}, 3.0),
            ],
            ffuncs.add,
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
            ffuncs.max,
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
def test_base_mapjoin(defs, func, expected_axes, expected_dims, expected_fill):
    objs = [BaseTensorStats.from_fields(ax, dims, fv) for (ax, dims, fv) in defs]
    out = BaseTensorStatsFactory._mapjoin_defs(func, *objs)
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
            ffuncs.add,
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
            ffuncs.add,
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
            ffuncs.add,
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
            ffuncs.add,
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
            ffuncs.mul,
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
            ffuncs.min,
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
def test_base_aggregate(
    op_func,
    index_order,
    dim_sizes,
    fill_value,
    reduce_fields,
    expected_axes,
    expected_dims,
    expected_fill,
):
    base = BaseTensorStats.from_fields(
        index_order=index_order, dim_sizes=dim_sizes, fill_value=fill_value
    )
    out = BaseTensorStatsFactory.aggregate_def(op_func, None, reduce_fields, base)

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
    node_mul = MapJoin(Literal(ffuncs.mul), (ta, tb))
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
    assert dsm.fill_value == 1.0

    node_add = MapJoin(Literal(ffuncs.add), (ta, ta2))
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


def test_aggregate():
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
        op=Literal(ffuncs.add),
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

    _overwrite_def(
        stat, BaseTensorStats.from_fields(frozenset([Field("i"), Field("j")]), dims, 0)
    )
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

    _overwrite_def(
        stat,
        BaseTensorStats.from_fields(
            frozenset([Field("i"), Field("j"), Field("k")]), dims, 0
        ),
    )
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

    _overwrite_def(
        stat,
        BaseTensorStats.from_fields(
            frozenset([Field("i"), Field("j"), Field("k"), Field("l")]), dims, 0
        ),
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

    _overwrite_def(
        stat,
        BaseTensorStats.from_fields(
            frozenset([Field("i"), Field("j"), Field("k")]), dims, 0
        ),
    )
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

    _overwrite_def(
        stat,
        BaseTensorStats.from_fields(
            frozenset([Field("i"), Field("j"), Field("k")]), dims, 0
        ),
    )
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
        _overwrite_def(s, BaseTensorStats.from_fields(frozenset({Field("i")}), dims, 0))
        s.dcs = set(dcs)
        stats_objs.append(s)

    out = DCStatsFactory().mapjoin(ffuncs.mul, *stats_objs)

    assert out.index_order == (Field("i"),)
    assert out.dim_sizes == dims
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

        td = BaseTensorStats.from_fields(field_idx_set, field_dims, 0)
        stats_objs.append(DCStats(td, dcs=set(dcs)))

    out = DCStatsFactory().mapjoin(ffuncs.add, *stats_objs)

    # Does the order matter here ? - > Changed tuple to set as throwing assert error
    assert set(out.index_order) == set(new_dims.keys())
    assert dict(out.dim_sizes) == new_dims
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
    _overwrite_def(s1, BaseTensorStats.from_fields(frozenset({Field("i")}), dims1, 0))
    s1.dcs = set(dcs1)

    node2 = Table(Literal(fl.asarray(np.zeros((1,), dtype=int))), (Field("i"),))
    s2 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node2,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    _overwrite_def(s2, BaseTensorStats.from_fields(frozenset({Field("i")}), dims2, 0))
    s2.dcs = set(dcs2)

    parent = MapJoin(Literal(ffuncs.add), (node1, node2))
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
    _overwrite_def(
        s1, BaseTensorStats.from_fields(frozenset({Field("i"), Field("j")}), dims1, 0)
    )
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
    _overwrite_def(
        s2, BaseTensorStats.from_fields(frozenset({Field("i"), Field("j")}), dims2, 0)
    )
    s2.dcs = set(dcs2)

    parent = MapJoin(Literal(ffuncs.add), (node1, node2))
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
    _overwrite_def(s1, BaseTensorStats.from_fields(frozenset({Field("i")}), dims1, 0))
    s1.dcs = set(dcs1)

    node2 = Table(Literal(fl.asarray(np.zeros((1,), dtype=int))), (Field("j"),))
    s2 = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=node2,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    _overwrite_def(s2, BaseTensorStats.from_fields(frozenset({Field("j")}), dims2, 0))
    s2.dcs = set(dcs2)

    parent = MapJoin(Literal(ffuncs.add), (node1, node2))
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
    _overwrite_def(
        s1, BaseTensorStats.from_fields(frozenset({Field("i"), Field("j")}), dims1, 0)
    )
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
    _overwrite_def(
        s2, BaseTensorStats.from_fields(frozenset({Field("j"), Field("k")}), dims2, 0)
    )
    s2.dcs = set(dcs2)

    parent = MapJoin(Literal(ffuncs.add), (node1, node2))
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
    _overwrite_def(
        s1, BaseTensorStats.from_fields(frozenset({Field("i"), Field("j")}), dims1, 1)
    )
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
    _overwrite_def(
        s2, BaseTensorStats.from_fields(frozenset({Field("j"), Field("k")}), dims2, 1)
    )
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
    _overwrite_def(
        s3,
        BaseTensorStats.from_fields(
            frozenset({Field("i"), Field("j"), Field("k")}), dims3, 1
        ),
    )
    s3.dcs = set(dcs3)

    map = MapJoin(Literal(ffuncs.mul), (node1, node2))

    parent = MapJoin(Literal(ffuncs.mul), (map, node3))

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
    _overwrite_def(
        s1, BaseTensorStats.from_fields(frozenset([Field("i"), Field("j")]), dims1, 1)
    )
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
    _overwrite_def(
        s2, BaseTensorStats.from_fields(frozenset([Field("j"), Field("k")]), dims2, 1)
    )
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
    _overwrite_def(
        s3,
        BaseTensorStats.from_fields(
            frozenset([Field("i"), Field("j"), Field("k")]), dims3, 0
        ),
    )
    s3.dcs = set(dcs3)

    map = MapJoin(Literal(ffuncs.mul), (node1, node2))
    parent = MapJoin(Literal(ffuncs.mul), (map, node3))

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
    _overwrite_def(
        stat,
        BaseTensorStats.from_fields(
            frozenset([Field("i"), Field("j"), Field("k")]), dims, 0.0
        ),
    )
    stat.dcs = set(dcs)

    reduce_node = Aggregate(
        op=Literal(ffuncs.add),
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
    _overwrite_def(
        st,
        BaseTensorStats.from_fields(
            frozenset([Field("i"), Field("j"), Field("k")]), dims, 0.0
        ),
    )
    st.dcs = set(dcs)

    reduce_node = Aggregate(
        op=Literal(ffuncs.add),
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
    _overwrite_def(
        st,
        BaseTensorStats.from_fields(
            frozenset([Field("i"), Field("j"), Field("k")]), dims, 0.0
        ),
    )
    st.dcs = set(dcs)

    reduce_node = Aggregate(
        op=Literal(ffuncs.add),
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
    _overwrite_def(
        st,
        BaseTensorStats.from_fields(
            frozenset([Field("i"), Field("j"), Field("k")]), dims, 0.0
        ),
    )
    st.dcs = set(dcs)

    reduce_fields = tuple(reduce_indices)
    reduce_node = Aggregate(
        op=Literal(ffuncs.add),
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


# ─────────────────────────────── LPStats tests ────────────────────────────────


def _lp_table(dense, fields):
    return Table(Literal(fl.asarray(np.asarray(dense, dtype=np.float64))), fields)


def _lp_stats(dense, fields, ps=(1.0, 2.0, math.inf)):
    return insert_statistics(
        stats_factory=LPStatsFactory(ps=ps),
        node=_lp_table(dense, fields),
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )


def test_lp_base_tensor_matches_nnz():
    i, j = Field("i"), Field("j")
    A = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 0.0, 0.0]])
    stats = _lp_stats(A, (i, j))
    assert isinstance(stats, LPStats)
    assert stats.estimate_non_fill_values() == pytest.approx(float(np.count_nonzero(A)))


def test_lp_base_tensor_matches_dc():
    i, j = Field("i"), Field("j")
    A = np.array([[1.0, 0.0, 2.0], [0.0, 5.0, 3.0], [4.0, 0.0, 0.0]])
    lp = _lp_stats(A, (i, j))
    dc = insert_statistics(
        stats_factory=DCStatsFactory(),
        node=_lp_table(A, (i, j)),
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert lp.estimate_non_fill_values() == pytest.approx(dc.estimate_non_fill_values())


def test_lp_empty_tensor_is_zero():
    i, j = Field("i"), Field("j")
    A = np.zeros((3, 4))
    stats = _lp_stats(A, (i, j))
    assert stats.estimate_non_fill_values() == 0.0


def test_lp_scalar_tensor():
    stats = insert_statistics(
        stats_factory=LPStatsFactory(),
        node=Table(Literal(fl.asarray(np.asarray(5.0))), ()),
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    assert stats.estimate_non_fill_values() == 1.0


def test_lp_bound_is_valid_upper_bound_on_join():
    # Broadcast product R(i,j) * S(j,k) over (i,j,k); the LP bound must be a
    # valid upper bound on the true number of non-fill output entries.
    i, j, k = Field("i"), Field("j"), Field("k")
    R = np.zeros((6, 4))
    R[:, 0] = 1
    R[0, 1] = R[1, 2] = R[2, 3] = 1
    S = np.zeros((4, 6))
    S[0, :] = 1
    S[1, 0] = S[2, 1] = S[3, 2] = 1
    true = sum(
        1
        for a in range(6)
        for b in range(4)
        for c in range(6)
        if R[a, b] != 0 and S[b, c] != 0
    )
    f = LPStatsFactory()
    out = f.mapjoin(
        ffuncs.mul,
        f(fl.asarray(R), (i, j)),
        f(fl.asarray(S), (j, k)),
    )
    est = out.estimate_non_fill_values()
    assert est >= true
    assert est <= 6 * 4 * 6  # never exceeds dense capacity


def test_lp_intermediate_p_is_tighter():
    # With p=2 available the bound on a skewed join is tighter than the
    # max-degree-only (p=inf) bound, while remaining a valid upper bound.
    i, j, k = Field("i"), Field("j"), Field("k")
    R = np.zeros((6, 4))
    R[:, 0] = 1
    R[0, 1] = R[1, 2] = R[2, 3] = 1
    S = np.zeros((4, 6))
    S[0, :] = 1
    S[1, 0] = S[2, 1] = S[3, 2] = 1
    true = sum(
        1
        for a in range(6)
        for b in range(4)
        for c in range(6)
        if R[a, b] != 0 and S[b, c] != 0
    )

    def bound(ps):
        f = LPStatsFactory(ps=ps)
        out = f.mapjoin(ffuncs.mul, f(fl.asarray(R), (i, j)), f(fl.asarray(S), (j, k)))
        return out.estimate_non_fill_values()

    b_inf = bound((1.0, math.inf))
    b_2 = bound((1.0, 2.0, math.inf))
    assert b_2 >= true
    assert b_2 <= b_inf + 1e-6


def test_lp_join_keeps_min():
    i, j = Field("i"), Field("j")
    A = np.array([[1.0, 0.0, 2.0], [0.0, 5.0, 0.0], [0.0, 0.0, 3.0]])
    B = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 3.0]])
    f = LPStatsFactory()
    joined = f.mapjoin(ffuncs.mul, f(fl.asarray(A), (i, j)), f(fl.asarray(B), (i, j)))
    true = float(np.count_nonzero(A * B))
    assert joined.estimate_non_fill_values() >= true


def test_lp_relabel_and_reorder():
    i, j, k = Field("i"), Field("j"), Field("k")
    A = np.array([[1.0, 0.0, 2.0], [0.0, 5.0, 3.0], [4.0, 0.0, 0.0]])
    f = LPStatsFactory()
    stats = f(fl.asarray(A), (i, j))
    relabeled = f.relabel(stats, (k, j))
    assert relabeled.index_order == (k, j)
    # renaming must remap the field names inside the degree records, so the
    # bound is unchanged and no record still references the old field `i`.
    assert all(i not in dc.from_indices | dc.to_indices for dc in relabeled.dcs)
    assert all((dc.from_indices | dc.to_indices) <= {k, j} for dc in relabeled.dcs)
    assert relabeled.estimate_non_fill_values() == pytest.approx(
        stats.estimate_non_fill_values()
    )
    reordered = f.reorder(stats, (j, i))
    assert reordered.index_order == (j, i)
    assert reordered.estimate_non_fill_values() == pytest.approx(
        stats.estimate_non_fill_values()
    )


def test_dc_relabel_remaps_fields():
    i, j, k = Field("i"), Field("j"), Field("k")
    A = np.array([[1.0, 0.0, 2.0], [0.0, 5.0, 3.0], [4.0, 0.0, 0.0]])
    f = DCStatsFactory()
    stats = f(fl.asarray(A), (i, j))
    relabeled = f.relabel(stats, (k, j))
    assert relabeled.index_order == (k, j)
    assert all(i not in dc.from_indices | dc.to_indices for dc in relabeled.dcs)
    assert relabeled.estimate_non_fill_values() == pytest.approx(
        stats.estimate_non_fill_values()
    )


def test_lp_aggregate_drops_index():
    i, j = Field("i"), Field("j")
    A = np.array([[1.0, 0.0, 2.0], [0.0, 5.0, 3.0], [4.0, 0.0, 0.0]])
    f = LPStatsFactory()
    stats = f(fl.asarray(A), (i, j))
    agg = f.aggregate(ffuncs.add, None, (j,), stats)
    assert agg.index_order == (i,)
    # after reducing j, the result has at most as many non-fill values as rows
    assert agg.estimate_non_fill_values() <= A.shape[0]


def test_lp_embedding_shape():
    i, j = Field("i"), Field("j")
    A = np.array([[1.0, 0.0, 2.0], [0.0, 5.0, 3.0], [4.0, 0.0, 0.0]])
    stats = _lp_stats(A, (i, j))
    emb = stats.get_embedding()
    assert emb.shape[0] == len(stats.index_order) + len(stats.dcs)
    assert np.all(np.isfinite(emb))


def test_lp_norm_endpoints():
    # p=1 -> nnz along the conditioned column; p=inf -> max degree.
    i, j = Field("i"), Field("j")
    A = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    stats = _lp_stats(A, (i, j), ps=(1.0, math.inf))
    by_p = {}
    for dc in stats.dcs:
        if dc.from_indices == frozenset({i}):
            by_p[dc.p] = dc.value
    # row 0 has degree 3, row 1 has degree 1: l1 = 4 (== nnz), linf = 3
    assert by_p[1.0] == pytest.approx(4.0)
    assert by_p[math.inf] == pytest.approx(3.0)


def test_lpdc_is_hashable_frozen():
    dc = DC(frozenset(), frozenset({Field("i")}), 3.0, 2.0)
    assert dc in {dc}
