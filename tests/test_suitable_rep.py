import numpy as np

from finchlite.finch_logic.nodes import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Reorder,
    Table,
)
from finchlite.galley.PhysicalOptimizer import (
    DenseData,
    ElementData,
    SparseData,
    SuitableRep,
    map_rep,
    toposort,
)


def test_literal():
    ctx = SuitableRep()
    expr = Literal(5)
    rep = ctx(expr)

    assert isinstance(rep, ElementData)
    assert rep.ndims() == 0


def test_table():
    ctx = SuitableRep()
    arr = np.array([1, 2, 3])
    expr = Table(Literal(arr), (Field("i"),))
    rep = ctx(expr)

    assert isinstance(rep, DenseData)
    assert rep.ndims() == 1


def test_alias_lookup():
    ctx = SuitableRep()
    arr = np.array([1, 2, 3])
    table_expr = Table(Literal(arr), (Field("i"),))

    alias = Alias("A")
    ctx.bindings[alias] = ctx(table_expr)

    assert ctx(alias) is ctx.bindings[alias]


def test_toposort_simple():
    result = toposort([["a", "b", "c"]])
    assert result == ["a", "b", "c"]


def test_toposort_multiple():
    result = toposort([["a", "b"], ["b", "c"]])
    assert result[0] == "a"
    assert result.index("b") < result.index("c")


def test_toposort_cycle():
    result = toposort([["a", "b"], ["b", "c"], ["c", "a"]])
    assert result is None


def test_dense_add():
    ctx = SuitableRep()
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])

    table1 = Table(Literal(arr1), (Field("i"),))
    table2 = Table(Literal(arr2), (Field("i"),))
    expr = Reorder(
        MapJoin(Literal(lambda x, y: x + y), (table1, table2)), (Field("i"),)
    )

    rep = ctx(expr)
    assert isinstance(rep, DenseData)
    assert rep.ndims() == 1


def test_dense_multiply():
    ctx = SuitableRep()
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6], [7, 8]])

    table1 = Table(Literal(arr1), (Field("i"), Field("j")))
    table2 = Table(Literal(arr2), (Field("i"), Field("j")))
    expr = Reorder(
        MapJoin(Literal(lambda x, y: x * y), (table1, table2)), (Field("i"), Field("j"))
    )

    rep = ctx(expr)
    assert isinstance(rep, DenseData)
    assert rep.ndims() == 2


def test_broadcast_operation():
    ctx = SuitableRep()
    arr2d = np.array([[1, 2], [3, 4]])
    arr1d = np.array([10, 20])

    table2d = Table(Literal(arr2d), (Field("i"), Field("j")))
    table1d = Table(Literal(arr1d), (Field("j"),))
    expr = Reorder(
        MapJoin(Literal(lambda x, y: x + y), (table2d, table1d)),
        (Field("i"), Field("j")),
    )

    rep = ctx(expr)
    assert rep.ndims() == 2


def test_sum_rows():
    ctx = SuitableRep()
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    table = Table(Literal(arr), (Field("i"), Field("j")))
    expr = Aggregate(Literal(lambda x, y: x + y), Literal(0), table, (Field("i"),))

    rep = ctx(expr)
    assert rep.ndims() == 1


def test_sum_all():
    ctx = SuitableRep()
    arr = np.array([[1, 2], [3, 4]])
    table = Table(Literal(arr), (Field("i"), Field("j")))
    expr = Aggregate(
        Literal(lambda x, y: x + y), Literal(0), table, (Field("i"), Field("j"))
    )

    rep = ctx(expr)
    assert rep.ndims() == 0
    assert isinstance(rep, ElementData)


def test_transpose():
    ctx = SuitableRep()
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    table = Table(Literal(arr), (Field("i"), Field("j")))
    expr = Reorder(table, (Field("j"), Field("i")))

    rep = ctx(expr)
    assert rep.ndims() == 2


def test_squeeze():
    ctx = SuitableRep()
    arr = np.array([[1, 2, 3]])
    table = Table(Literal(arr), (Field("i"), Field("j")))
    expr = Reorder(table, (Field("j"),))

    rep = ctx(expr)
    assert rep.ndims() == 1


def test_permute_3d():
    ctx = SuitableRep()
    arr = np.zeros((2, 3, 4))
    table = Table(Literal(arr), (Field("i"), Field("j"), Field("k")))
    expr = Reorder(table, (Field("k"), Field("i"), Field("j")))

    rep = ctx(expr)
    assert rep.ndims() == 3


def test_mapjoin_then_aggregate():
    ctx = SuitableRep()
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[10, 10], [10, 10]])

    table1 = Table(Literal(arr1), (Field("i"), Field("j")))
    table2 = Table(Literal(arr2), (Field("i"), Field("j")))
    mapjoin = Reorder(
        MapJoin(Literal(lambda x, y: x * y), (table1, table2)), (Field("i"), Field("j"))
    )
    aggregate = Aggregate(
        Literal(lambda x, y: x + y), Literal(0), mapjoin, (Field("i"),)
    )

    rep = ctx(aggregate)
    assert rep.ndims() == 1


def test_sparse_plus_dense():
    sparse = SparseData(ElementData(0, int))
    dense = DenseData(ElementData(0, int))
    result = map_rep(lambda x, y: x + y, sparse, dense)
    assert isinstance(result, DenseData)


def test_dense_plus_sparse():
    dense = DenseData(ElementData(0, int))
    sparse = SparseData(ElementData(0, int))
    result = map_rep(lambda x, y: x + y, dense, sparse)
    assert isinstance(result, DenseData)


def test_sparse_times_dense():
    sparse = SparseData(ElementData(0, int))
    dense = DenseData(ElementData(0, int))
    result = map_rep(lambda x, y: x * y, sparse, dense)
    assert isinstance(result, SparseData)


def test_dense_times_sparse():
    dense = DenseData(ElementData(0, int))
    sparse = SparseData(ElementData(0, int))
    result = map_rep(lambda x, y: x * y, dense, sparse)
    assert isinstance(result, SparseData)


def test_sparse_plus_sparse():
    sparse1 = SparseData(ElementData(0, int))
    sparse2 = SparseData(ElementData(0, int))
    result = map_rep(lambda x, y: x + y, sparse1, sparse2)
    assert isinstance(result, SparseData)


def test_permutedims_simple():
    from finchlite.galley.PhysicalOptimizer import permutedims_rep

    rep = DenseData(DenseData(ElementData(0, int)))
    result = permutedims_rep(rep, [1, 0])
    assert result.ndims() == 2


def test_permutedims_identity():
    from finchlite.galley.PhysicalOptimizer import permutedims_rep

    rep = DenseData(DenseData(ElementData(0, int)))
    result = permutedims_rep(rep, [0, 1])
    assert result.ndims() == 2
    assert isinstance(result, DenseData)


def test_permutedims_3d():
    from finchlite.galley.PhysicalOptimizer import permutedims_rep

    rep = DenseData(DenseData(DenseData(ElementData(0, int))))
    result = permutedims_rep(rep, [2, 0, 1])
    assert result.ndims() == 3


def test_permutedims_sparse():
    from finchlite.galley.PhysicalOptimizer import SparseData, permutedims_rep

    rep = SparseData(SparseData(ElementData(0, int)))
    result = permutedims_rep(rep, [1, 0])
    assert isinstance(result, SparseData)
    assert result.ndims() == 2


def test_permutedims_mixed():
    from finchlite.galley.PhysicalOptimizer import SparseData, permutedims_rep

    rep = DenseData(SparseData(ElementData(0, int)))
    result = permutedims_rep(rep, [1, 0])
    assert result.ndims() == 2
